/**
 * マイク / ライン入力からの音声取得.
 *
 * 重要な制約が 3 つある。
 *
 * 1. getUserMedia の echoCancellation / noiseSuppression / autoGainControl は
 *    既定で有効であり、いずれも CW のトーンを破壊する。**必ず全部 false にする**。
 * 2. リサンプルは状態を保持しなければならない。ステートレスな変換はブロック境界に
 *    歪みを生じ dot/dash を壊す (src/infer/audio.py の記録)。
 *    主系は AudioContext を 8 kHz で作りブラウザ内蔵のリサンプラに任せる。
 *    それが効かない環境では FirDecimator にフォールバックする。
 * 3. デコード経路にはバンドパスフィルタ (BPF) を入れる。デスクトップ版
 *    (`src/app/workers.py` の `_StreamingBPF`) は既定で有効な BPF を全ブロックに
 *    かけてから推論に渡しており、これが無いとブラウザ版だけ帯域外ノイズを
 *    そのまま推論に食わせることになる (実受信精度に直結する)。
 *    **BPF はデコード経路にだけ入れ、AnalyserNode は生の source に繋ぐ。**
 *    スペクトル表示は同調確認に使うので通過帯域の外も見える必要があるため。
 */
import { FirDecimator } from './fir-decimator'
import { captureWorkletUrl } from './capture-worklet'
// サンプルレートは手書きしない (MelConfig.sample_rate から生成)
import { SAMPLE_RATE as TARGET_SAMPLE_RATE } from '../generated/tokens'

// BPF のパラメータ。デスクトップ版 `_StreamingBPF` の既定値と揃えてある
// (中心 600 Hz / 帯域 400 Hz = 通過帯域 400〜800 Hz)。
const BPF_CENTER_HZ = 600
const BPF_BANDWIDTH_HZ = 400
/** BiquadFilterNode 'bandpass' の Q は 中心周波数 / -3 dB 帯域幅。 */
const BPF_Q = BPF_CENTER_HZ / BPF_BANDWIDTH_HZ
/**
 * 2 次のバイカッドを 2 段直列 (4 次相当) にして肩の急峻さを稼ぐ。デスクトップ版は
 * scipy `butter(4, ..., btype='bandpass')` = SOS 4 セクション (実効 8 次) なので、
 * ロールオフ (肩の落ち方) はブラウザ版のほうが約半分緩い。
 */
const BPF_STAGES = 2

export interface CaptureOptions {
  /** 8 kHz の音声ブロックが届くたびに呼ばれる。 */
  onAudio: (block: Float32Array) => void
  deviceId?: string
  /**
   * デコード経路のバンドパスフィルタを有効にするか。既定 true
   * (デスクトップ版 `src/app/workers.py` の `bpf_enabled=True` と揃える)。
   */
  bandpassEnabled?: boolean
}

export class AudioCapture {
  private constructor(
    private readonly context: AudioContext,
    private readonly stream: MediaStream,
    private readonly node: AudioWorkletNode,
    readonly analyser: AnalyserNode,
    readonly sampleRate: number,
    readonly usedFallbackResampler: boolean,
    /** source → (BPF) → worklet の配線を切り替えるための参照 */
    private readonly source: MediaStreamAudioSourceNode,
    private readonly bandpass: readonly BiquadFilterNode[],
    private bandpassEnabled: boolean,
  ) {}

  /**
   * 中心 600 Hz / 帯域 400 Hz のバンドパスを 2 段直列で作る。
   *
   * **デスクトップ版 (`src/app/workers.py` の `_StreamingBPF`) は scipy の
   * `butter(4, ..., btype='bandpass')` で、bandpass 指定は次数を 2 倍にするため
   * SOS は 4 セクション = 実効 8 次。こちらの BiquadFilterNode 'bandpass' は
   * RBJ Audio EQ Cookbook のバイカッドを 2 段 = 4 次相当であり、段数が半分
   * (ロールオフも約半分緩い) なので周波数特性は厳密には一致しない**
   * (通過帯域の平坦さ・肩の落ち方・群遅延が異なる)。BPF オン/オフの比較で
   * 差が小さくても、この段数差があるため「BPF が無意味」とは即断できない。
   * 同等の目的 (帯域外のノイズを落として推論に渡す) を果たすための近似で
   * あって、同一のフィルタではない点に注意すること。
   */
  private static createBandpass(context: BaseAudioContext): BiquadFilterNode[] {
    return Array.from({ length: BPF_STAGES }, () => {
      const filter = context.createBiquadFilter()
      filter.type = 'bandpass'
      filter.frequency.value = BPF_CENTER_HZ
      filter.Q.value = BPF_Q
      return filter
    })
  }

  static async start(options: CaptureOptions): Promise<AudioCapture> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: options.deviceId,
        // CW を壊すため必ず無効にする
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    })

    // getUserMedia 成功後は、どこで例外が起きてもここで stream (と生成済みなら
    // context) を解放してから元の例外を再送出する。解放しないとマイクの録音
    // インジケータが点灯したまま残り、再試行のたびにストリームが積み上がる。
    let context: AudioContext | undefined
    try {
      // 主系: AudioContext を 8 kHz で構築する
      try {
        context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
      } catch {
        context = new AudioContext()
      }
      const needsFallback = context.sampleRate !== TARGET_SAMPLE_RATE

      let decimator: FirDecimator | null = null
      if (needsFallback) {
        if (!FirDecimator.supports(context.sampleRate, TARGET_SAMPLE_RATE)) {
          throw new Error(
            `このデバイスのサンプルレート ${context.sampleRate} Hz には対応していません ` +
              `(8000 Hz の整数倍が必要です)`,
          )
        }
        decimator = new FirDecimator(context.sampleRate, TARGET_SAMPLE_RATE)
      }

      const url = captureWorkletUrl()
      try {
        await context.audioWorklet.addModule(url)
      } finally {
        URL.revokeObjectURL(url)
      }

      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser()
      analyser.fftSize = 2048
      analyser.smoothingTimeConstant = 0.6

      const node = new AudioWorkletNode(context, 'cw-capture')
      node.port.onmessage = (event: MessageEvent<Float32Array>) => {
        const block = event.data
        options.onAudio(decimator ? decimator.process(block) : block)
      }

      // スペクトル表示は**生の** source から取る。BPF の後ろに繋ぐと通過帯域の
      // 外が見えなくなり、同調 (トーンがどの周波数に来ているか) の確認に
      // 使えなくなるため。
      source.connect(analyser)

      // デコード経路だけ BPF を通す。段は常に作っておき、有効/無効は配線の
      // 付け替えで切り替える (フィルタ状態はグラフが保持するので、自前の
      // 状態管理は不要 — ステートレスなリサンプルで踏んだ轍を避ける)。
      const bandpass = AudioCapture.createBandpass(context)
      for (let i = 0; i < bandpass.length - 1; i++) bandpass[i].connect(bandpass[i + 1])

      const bandpassEnabled = options.bandpassEnabled ?? true
      if (bandpassEnabled) {
        source.connect(bandpass[0])
        bandpass[bandpass.length - 1].connect(node)
      } else {
        source.connect(node)
      }

      // Worklet は出力を持たないが、一部ブラウザは destination に繋がないと
      // グラフが動かないため 0 ゲインで接続しておく
      const mute = context.createGain()
      mute.gain.value = 0
      node.connect(mute).connect(context.destination)

      await context.resume()

      return new AudioCapture(
        context,
        stream,
        node,
        analyser,
        context.sampleRate,
        needsFallback,
        source,
        bandpass,
        bandpassEnabled,
      )
    } catch (err) {
      // 後始末自体が失敗しても元の例外を隠さないよう、それぞれ個別に握りつぶす。
      try {
        stream.getTracks().forEach((t) => t.stop())
      } catch {
        // 後始末の失敗は無視 (元のエラーを優先して伝える)
      }
      if (context) {
        try {
          await context.close()
        } catch {
          // 後始末の失敗は無視 (元のエラーを優先して伝える)
        }
      }
      throw err
    }
  }

  /** BPF が有効か。 */
  get isBandpassEnabled(): boolean {
    return this.bandpassEnabled
  }

  /**
   * デコード経路の BPF をオン/オフする (受信を止めずに切り替えられる)。
   *
   * 実信号で効果を比較できるようにするための機能。配線を付け替えるだけなので
   * AudioContext の再起動もリサンプラの作り直しも不要。AnalyserNode は
   * どちらの状態でも生の source に繋がったままなので、スペクトル表示は変わらない。
   */
  setBandpassEnabled(enabled: boolean): void {
    if (enabled === this.bandpassEnabled) return
    const last = this.bandpass[this.bandpass.length - 1]
    if (enabled) {
      this.source.disconnect(this.node)
      this.source.connect(this.bandpass[0])
      last.connect(this.node)
    } else {
      this.source.disconnect(this.bandpass[0])
      last.disconnect(this.node)
      this.source.connect(this.node)
    }
    this.bandpassEnabled = enabled
  }

  async stop(): Promise<void> {
    this.node.port.onmessage = null
    this.node.disconnect()
    this.stream.getTracks().forEach((track) => track.stop())
    await this.context.close()
  }
}
