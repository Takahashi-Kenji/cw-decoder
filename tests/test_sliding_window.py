"""SlidingWindowDecoder (prefix commit) のテスト."""
from __future__ import annotations

import numpy as np

from src.infer.engine import FrameToken, InferenceEngine
from src.infer.sliding_window import (
    CommittedToken,
    DecodeView,
    SlidingWindowDecoder,
)


def _decoder(**kw) -> SlidingWindowDecoder:
    # 既定は decode_left_context_s = window_s (= フルウィンドウ再デコード) とし、
    # 固定トークン monkeypatch の frame 位置が絶対位置に一致するようにする
    # (動的区間短縮は Task 4 の test_decode_window_shortens_after_commit で個別検証).
    eng = InferenceEngine.untrained("cpu")
    params = dict(
        window_s=30.0, hop_s=1.0, commit_lag_s=2.5,
        head_guard_s=1.0, decode_left_context_s=30.0,
        commit_jitter_margin_s=0.02, sample_rate=8000,
        # **助走は切る。** ここのテストは確定の水位が関心事で、frame 位置が
        # そのまま絶対位置になる前提で書かれている (助走を入れると原点が
        # ずれる)。助走そのものは TestLeadIn で見る
        lead_in_s=0.0,
    )
    params.update(kw)
    return SlidingWindowDecoder(eng, **params)


def test_empty_decode_returns_empty_view() -> None:
    d = _decoder()
    view = d.redecode()
    assert isinstance(view, DecodeView)
    assert view.committed == []
    assert view.provisional == []
    assert view.newly_committed == []


def _patch_tokens(d: SlidingWindowDecoder, frame_tokens: list[FrameToken]) -> None:
    """engine.decode_chunk を固定トークン列に差し替え (決定論テスト用).

    ヘルパは decode_left_context_s == window_s なので decode_start は常に
    ring 先頭 = 絶対 0 起点となり、frame 位置 == 絶対位置 (offset 安定).
    """
    d.engine.decode_chunk = lambda wave: list(frame_tokens)  # type: ignore[assignment]


def test_commits_only_inside_commit_zone() -> None:
    # hop=80 samples/frame @8kHz → 1 frame = 0.01s. 100 frame = 1s.
    d = _decoder(window_s=30.0, commit_lag_s=2.5, head_guard_s=1.0)
    d.push(np.zeros(80000, dtype=np.float32))   # 10s
    _patch_tokens(d, [
        FrameToken(token_id=5, confidence=0.9, frame_start=50, frame_end=55),
        FrameToken(token_id=6, confidence=0.9, frame_start=500, frame_end=505),
        FrameToken(token_id=7, confidence=0.9, frame_start=900, frame_end=905),
    ])
    view = d.redecode()
    # commit_limit = 10s-2.5s = 7.5s. 9s(frame900) は abs_end>commit_limit → 暫定
    assert [t.token_id for t in view.committed] == [5, 6]
    assert [t.token_id for t in view.provisional] == [7]


def test_boundary_straddling_token_is_provisional_not_committed() -> None:
    # レビュー #1: 開始は commit 境界前・終了は境界後のトークンは確定しない.
    d = _decoder(window_s=30.0, commit_lag_s=2.5, head_guard_s=0.0)
    d.push(np.zeros(80000, dtype=np.float32))   # 10s → commit_limit = 7.5s (abs 60000)
    # frame740..760 = abs 59200..60800 (開始 7.4s < 7.5s, 終了 7.6s > 7.5s)
    _patch_tokens(d, [FrameToken(token_id=8, confidence=0.9, frame_start=740, frame_end=760)])
    view = d.redecode()
    assert view.committed == []                       # 旧実装 (abs_start 判定) では誤って確定した
    assert [t.token_id for t in view.provisional] == [8]


def test_realistic_25wpm_five_chars_all_commit_in_one_pass() -> None:
    # レビュー #2 (最重大): 25 WPM (dot=48ms=384smp) の文字間ギャップ 3 dot≈144ms で
    # 連続 5 文字がすべて確定すること. 旧 tol=0.3s 実装ではここが脱落していた.
    d = _decoder(window_s=30.0, commit_lag_s=1.0, head_guard_s=0.0)
    d.push(np.zeros(8000 * 5, dtype=np.float32))      # 5s → commit_limit = 4.0s (abs 32000)
    # 各文字 ~5 frame 長、文字間 14 frame(=140ms) 間隔で 5 文字並べる
    toks = []
    fs = 10
    for tid in range(1, 6):
        toks.append(FrameToken(token_id=tid, confidence=0.9, frame_start=fs, frame_end=fs + 5))
        fs += 19                                       # 5(長) + 14(間隔) = 19 frame
    _patch_tokens(d, toks)
    view = d.redecode()
    assert [t.token_id for t in view.committed] == [1, 2, 3, 4, 5]


def test_short_token_not_double_committed_on_redecode() -> None:
    # レビュー #2: 短トークン (E=dot) が次パスで二重確定されないこと (中点ウォーターマーク).
    d = _decoder(window_s=30.0, commit_lag_s=1.0, head_guard_s=0.0)
    d.push(np.zeros(8000 * 3, dtype=np.float32))      # 3s
    e_tok = [FrameToken(token_id=2, confidence=0.9, frame_start=100, frame_end=102)]
    _patch_tokens(d, e_tok)
    v1 = d.redecode()
    assert [t.token_id for t in v1.newly_committed] == [2]
    d.push(np.zeros(8000, dtype=np.float32))          # +1s, 同じ E が再出現
    v2 = d.redecode()
    assert v2.newly_committed == []                    # 二重確定しない
    assert [t.token_id for t in v2.committed] == [2]


def test_head_guard_drops_decode_region_front() -> None:
    d = _decoder(window_s=30.0, commit_lag_s=2.5, head_guard_s=1.0)
    # 35s 投入 → ring 末尾 30s、ring_start_abs = 5s。left_context=window なので decode_start=ring_start
    d.push(np.zeros(8000 * 35, dtype=np.float32))
    # frame50(=窓内 0.5s, 絶対 5.5s) は head_guard(1s) 内 → 不採用 / frame500(絶対 10s) は採用
    _patch_tokens(d, [
        FrameToken(token_id=5, confidence=0.9, frame_start=50, frame_end=55),
        FrameToken(token_id=6, confidence=0.9, frame_start=500, frame_end=505),
    ])
    view = d.redecode()
    assert [t.token_id for t in view.committed] == [6]


def test_provisional_becomes_committed_after_lag() -> None:
    d = _decoder(window_s=30.0, commit_lag_s=2.5)
    d.push(np.zeros(80000, dtype=np.float32))         # 10s, トークン @9.5s (暫定)
    _patch_tokens(d, [FrameToken(token_id=9, confidence=0.9, frame_start=950, frame_end=955)])
    v1 = d.redecode()
    assert [t.token_id for t in v1.provisional] == [9]
    assert v1.committed == []
    d.push(np.zeros(8000 * 3, dtype=np.float32))      # +3s → total 13s, commit_limit=10.5s
    v2 = d.redecode()
    assert [t.token_id for t in v2.committed] == [9]
    assert v2.provisional == []


def test_decode_window_shortens_after_commit() -> None:
    # レビュー #5: 確定後は left_context 分のみ再デコード (毎回 window 全体ではない).
    d = _decoder(window_s=30.0, commit_lag_s=2.5, decode_left_context_s=5.0)
    d.push(np.zeros(8000 * 20, dtype=np.float32))     # 20s (ring=160000, ring_start_abs=0)
    d._last_commit_end = 8000 * 15                     # 15s まで確定済みと仮定
    d._committed = [CommittedToken(1, 0.9, 0, 8000 * 15)]
    received: list[int] = []
    d.engine.decode_chunk = lambda wave: received.append(wave.size) or []  # type: ignore
    d.redecode()
    # decode_start = max(0, 15s - 5s) = 10s → 渡されるのは末尾 10s = 80000 samples
    assert received == [80000]


def test_finalize_commits_pending_provisional() -> None:
    # 送信終了時: commit_lag 圏内の暫定を最終確定する.
    d = _decoder(window_s=30.0, commit_lag_s=2.5)
    d.push(np.zeros(80000, dtype=np.float32))         # 10s, @9.5s は通常 redecode では暫定
    _patch_tokens(d, [FrameToken(token_id=9, confidence=0.9, frame_start=950, frame_end=955)])
    assert d.redecode().committed == []
    final = d.finalize()
    assert [t.token_id for t in final.committed] == [9]
    assert final.provisional == []


def test_reset_clears_committed() -> None:
    d = _decoder()
    d.push(np.zeros(80000, dtype=np.float32))
    _patch_tokens(d, [FrameToken(token_id=5, confidence=0.9, frame_start=300, frame_end=305)])
    d.redecode()
    d.reset()
    assert d._last_commit_end is None
    assert d.redecode().committed == []


# ============================================================
# 2 段階確定 (ターン終了時の書き直し)
# ============================================================
#
# **「確定済みは後から変化しない」の例外である** (2026-08-12、運用者の承認済み)。
# ライブ確定は実効 2.25 秒の右文脈しか使えず、held-out 実録音 21 件の実測で
# オフライン全文脈より TER が 3.2pt 悪い。ターンが終わった瞬間に、そのターンの
# 音を丸ごとデコードし直して置き換えれば、行単位でオフライン精度になる。
# 書き換えは「ターン終了時に 1 回だけ」に限る。

_GAP = 3 * 8000            # 3 秒無音 = ターンの切れ目 (line_break と同じ)


def _committed_turn(d: SlidingWindowDecoder, tokens: list[tuple[int, int, int]]) -> None:
    """(id, abs_start, abs_end) を確定済みとして直接植える (経路は他のテストが担保)."""
    d._committed = [
        CommittedToken(token_id=i, confidence=0.9,
                       absolute_sample_start=s, absolute_sample_end=e)
        for i, s, e in tokens
    ]


class _RefineEngine:
    """refine の呼び出しを記録し、決めたトークンを返す偽エンジン."""

    frame_hop_samples = 40

    def __init__(self, result: list[FrameToken]):
        self.result = result
        self.calls: list[int] = []          # 渡された波形の長さ

    def decode_chunk(self, wave):
        self.calls.append(int(wave.size))
        return list(self.result)


def _refine_decoder(engine, *, window_s: float = 30.0) -> SlidingWindowDecoder:
    return SlidingWindowDecoder(
        engine, window_s=window_s, hop_s=1.0, commit_lag_s=2.5, sample_rate=8000,
    )


def test_閉じたターンが全文脈で置き換わる() -> None:
    h = _RefineEngine.frame_hop_samples
    # 置き換え結果: span 相対 frame 100 の位置にトークン 99
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.95,
                                    frame_start=100, frame_end=105)])
    d = _refine_decoder(eng)
    # 11 秒。ターン 2 の末尾からの無音は 4.5 秒 < gap 3s + lag 2.5s なので
    # ターン 2 は開いたまま (閉じるのはターン 1 だけ)
    d.push(np.zeros(88000, dtype=np.float32))
    # ターン 1 (8000..16000) と、3 秒以上あとのターン 2 (48000..)
    _committed_turn(d, [(5, 8000, 12000), (6, 12800, 16000), (7, 48000, 52000)])

    changed = d.refine_closed_turns(_GAP, lead_in_s=0.0)

    assert changed is True
    assert len(eng.calls) == 1
    ids = [t.token_id for t in d._committed]
    assert ids == [99, 7]                            # ターン 1 が置き換わり、2 は不変
    # **区間は前後の無音まるごと** — 先頭のターンなのでリング先頭 (絶対 0) から、
    # 次のターンの 0.25 秒手前まで。オフラインと同じ入力にするため
    assert eng.calls[0] == 48000 - 2000
    assert d._committed[0].absolute_sample_start == 0 + 100 * h


def test_開いているターンは触らない() -> None:
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng)
    d.push(np.zeros(40000, dtype=np.float32))       # 5 秒
    # 末尾のトークンは 2 秒前に終わったばかり (gap + lag = 5.5 秒に満たない)
    _committed_turn(d, [(5, 8000, 24000)])

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is False
    assert eng.calls == []
    assert [t.token_id for t in d._committed] == [5]


def test_無音が続けば最後のターンも閉じる() -> None:
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))       # 10 秒
    # 末尾 16000 から 64000 サンプル (8 秒) 無音 > gap 3s + lag 2.5s
    _committed_turn(d, [(5, 8000, 16000)])

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is True
    assert [t.token_id for t in d._committed] == [99]


def test_同じターンを二度直さない() -> None:
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is True
    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is False      # 2 回目は何もしない
    assert len(eng.calls) == 1


def test_結果が同じなら変更なしと報告する() -> None:
    """無駄な再描画を起こさない。ただし「直した」ことは覚える."""
    h = _RefineEngine.frame_hop_samples
    # ちょうど元と同じトークンを返す (span はリング先頭からなので絶対位置と同じ)
    eng = _RefineEngine([FrameToken(token_id=5, confidence=0.9,
                                    frame_start=8000 // h, frame_end=16000 // h)])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is False
    assert len(eng.calls) == 1
    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is False      # 覚えているので呼ばない
    assert len(eng.calls) == 1


def test_音が窓から落ちたターンは諦める() -> None:
    """リングに音が残っていなければ第 1 段の結果を残す (黙って消さない)."""
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng, window_s=5.0)
    d.push(np.zeros(80000, dtype=np.float32))       # 10 秒 → 先頭 5 秒はもう無い
    _committed_turn(d, [(5, 8000, 16000)])          # 音が消えた領域

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is False
    assert eng.calls == []
    assert [t.token_id for t in d._committed] == [5]


def test_置き換えが空なら元を残す() -> None:
    """全文脈で何も出なくても、見えていた文字を黙って消さない."""
    eng = _RefineEngine([])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is False
    assert [t.token_id for t in d._committed] == [5]


def test_all_turns_で最後のターンも直す() -> None:
    """停止時 (finalize 後) は待たずに全部直す."""
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng)
    d.push(np.zeros(40000, dtype=np.float32))       # 5 秒 (無音では閉じない長さ)
    _committed_turn(d, [(5, 8000, 24000)])

    assert d.refine_closed_turns(_GAP, all_turns=True, lead_in_s=0.0) is True
    assert [t.token_id for t in d._committed] == [99]


def test_reset_で直した記録も消える() -> None:
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])
    d.refine_closed_turns(_GAP, lead_in_s=0.0)

    d.reset()
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is True       # また直せる


def test_録音の先頭近くのターンも余白を詰めて直す() -> None:
    """**余白が取れないことは諦める理由にならない。**

    「余白ぶんの音まで無いと諦める」実装だと、録音の先頭近くのターン
    (余白の取りようがない) を全部諦める。held-out の余白掃引で
    「余白を増やすほど 1 段階の値に戻る」という逆向きの結果が出て発覚した。
    """
    h = _RefineEngine.frame_hop_samples
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    # ターンの先頭が 2000 (余白 0.5 秒 = 4000 より手前)
    _committed_turn(d, [(5, 2000, 16000)])

    assert d.refine_closed_turns(_GAP, lead_in_s=0.0) is True
    assert [t.token_id for t in d._committed] == [99]
    # span はリングの先頭 (絶対 0) に詰められる
    assert d._committed[0].absolute_sample_start == 0 + 0 * h


# ---- 助走の無音 (先頭の幻覚対策) ----
#
# **モデルは音の立ち上がりで幻覚を出す。** held-out 21 件で先頭に余計な
# トークンが出たのは 18 件、別実装のデコーダは同じ音で 0/10 件だった。
# 0.3 秒の無音を足すと TER 24.21% → 22.32%、先頭の誤りは 18 → 13 件に減る。


def test_助走の無音を足してデコードする() -> None:
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=0, frame_end=1)])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    d.refine_closed_turns(_GAP, lead_in_s=0.5)

    # 渡した波形が助走ぶん長い (区間 0..80000 + 0.5 秒 = 4000)
    assert eng.calls[0] == 80000 + 4000


def test_助走の中に出たトークンは捨てる() -> None:
    """**そこに音が無いのだから符号ではありえない。**"""
    h = _RefineEngine.frame_hop_samples
    eng = _RefineEngine([
        FrameToken(token_id=98, confidence=0.9, frame_start=0, frame_end=1),      # 無音の中
        FrameToken(token_id=99, confidence=0.9,
                   frame_start=4000 // h + 10, frame_end=4000 // h + 11),         # 音の中
    ])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    d.refine_closed_turns(_GAP, lead_in_s=0.5)

    assert [t.token_id for t in d._committed] == [99]


def test_助走を足しても位置がずれない() -> None:
    """位置は助走ぶん戻す。ずれるとターンの切り方が壊れる."""
    h = _RefineEngine.frame_hop_samples
    lead = 4000
    eng = _RefineEngine([FrameToken(token_id=99, confidence=0.9,
                                    frame_start=(lead + 8000) // h,
                                    frame_end=(lead + 12000) // h)])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    d.refine_closed_turns(_GAP, lead_in_s=0.5)

    # span はリング先頭 (絶対 0) からなので、助走を戻すと絶対 8000 に来る
    assert d._committed[0].absolute_sample_start == 8000


def test_既定で助走が入る() -> None:
    from src.infer.sliding_window import DEFAULT_LEAD_IN_S

    assert DEFAULT_LEAD_IN_S > 0.0
    eng = _RefineEngine([])
    d = _refine_decoder(eng)
    d.push(np.zeros(80000, dtype=np.float32))
    _committed_turn(d, [(5, 8000, 16000)])

    d.refine_closed_turns(_GAP)

    assert eng.calls[0] == 80000 + int(DEFAULT_LEAD_IN_S * 8000)


class TestLeadInOnFirstPass:
    """第 1 段 (受信中の速い確定) の助走.

    **交信の冒頭でモデルが幻覚を出すと、その 1 文字が確定して後から消せない。**
    第 2 段の書き直しまで残るので、第 1 段で防ぐ意味がある。
    """

    @staticmethod
    def _decoder(**kw):
        eng = InferenceEngine.untrained("cpu")
        params = dict(
            window_s=30.0, hop_s=1.0, commit_lag_s=2.5,
            head_guard_s=1.0, decode_left_context_s=30.0,
            sample_rate=8000, lead_in_s=0.5,
        )
        params.update(kw)
        return SlidingWindowDecoder(eng, **params)

    def test_区間の頭が音のときは助走を足す(self) -> None:
        d = self._decoder()
        sizes: list[int] = []
        d.engine.decode_chunk = lambda w: sizes.append(w.size) or []
        d.push(np.zeros(80000, dtype=np.float32))
        d.redecode()
        assert sizes[0] == 80000 + 4000          # 0.5 秒ぶん長い

    def test_助走の中に出たトークンは捨てる(self) -> None:
        """**そこに音が無いのだから符号ではありえない。**"""
        d = self._decoder()
        hop = d.engine.frame_hop_samples
        d.engine.decode_chunk = lambda w: [
            FrameToken(token_id=7, confidence=0.9, frame_start=0, frame_end=1),
            FrameToken(token_id=8, confidence=0.9,
                       frame_start=4000 // hop + 5, frame_end=4000 // hop + 6),
        ]
        d.push(np.zeros(80000, dtype=np.float32))
        view = d.redecode()
        assert [t.token_id for t in view.committed] == [8]

    def test_位置が助走ぶんずれない(self) -> None:
        d = self._decoder()
        hop = d.engine.frame_hop_samples
        at = (4000 + 8000) // hop
        d.engine.decode_chunk = lambda w: [
            FrameToken(token_id=8, confidence=0.9, frame_start=at, frame_end=at + 1)
        ]
        d.push(np.zeros(80000, dtype=np.float32))
        view = d.redecode()
        assert view.committed[0].absolute_sample_start == 8000

    def test_左文脈があるときは足さない(self) -> None:
        """既に前の音があるなら助走は要らない (窓を無駄に長くしない)."""
        d = self._decoder(decode_left_context_s=1.0)
        sizes: list[int] = []
        d.engine.decode_chunk = lambda w: sizes.append(w.size) or []
        d.push(np.zeros(80000, dtype=np.float32))
        d._last_commit_end = 40000               # 確定済みがある = 途中から復号
        d.redecode()
        # 復号開始は 40000 - 8000 (左文脈 1 秒) = 32000。助走は足さない
        assert sizes[-1] == 80000 - 32000
