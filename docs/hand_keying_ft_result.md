# 手打ちキーイング分布ファインチューニング 結果と判断根拠

作成日: 2026-08-06（対照群追加により改訂）
対象: `feature/hand-keying-synth`（Task 7: ファインチューニングと評価）
結論: **不採用**（`models/ft_hand_tail/` `models/ft_hand_notail/` `models/ft_control_old/`
いずれも配布モデル差し替えは行わない）。ただし判定の確度は評価軸ごとに異なる（§0参照）。

---

## 0. 結論を先に

- `data/real/train`（90ペア、実打鍵）を 70% 混合比で 1000 step ファインチューニングした。
  極端テールあり (`ft_hand_tail`) ／なし (`ft_hand_notail`) の2条件に加え、
  **合成分布を従来（手打ち分布なし）に戻した対照群 (`ft_control_old`)** を追加で回した。
  対照群は起点・データ・step数・real_ratio・その他ハイパーパラメータをすべて
  `ft_hand_tail`/`ft_hand_notail` と揃え、`--no-hand-keying` のみを追加している。

- **当初の懸念（交絡）が実際に確認された。** `models/ft_1k/best.pt` は既に同じ
  `data/real/train` で1000 step FT済みのモデルであり、そこへさらに1000 step
  追加FTしたため、観測された悪化が (a) 手打ち合成分布のせいか (b) 同じ90ファイルへの
  追加1000 stepによる過学習のせいか、対照群なしでは分離できなかった。

- **軸ごとに交絡の有無が異なる、という結果になった。**
  - **keyed_val（実打鍵20件）**: **tail は対照群に対して両指標とも同等以上だった。**
    TER は tail +2.68pt に対し対照群 +3.36pt（tail の方が悪化幅が小さい）、CER は
    tail ±0.00pt に対し対照群 +2.52pt（tail は baseline と同値、対照群の方が明確に悪い）。
    notail は TER で対照群より悪い（+5.37pt vs +3.36pt）が、CER では対照群より良い
    （+0.46pt vs +2.52pt）。すなわち4条件（TER×2アーム、CER×2アーム）のうち3つで
    手打ち分布アームが対照群と同等以上であり、「対照群が最も悪化する」という組み合わせが
    複数の指標で観測されている。**手打ち分布を混ぜたこと自体が keyed_val を対照群より
    悪化させたという証拠は見当たらない。** これは keyed_val の悪化が手打ち分布固有の
    問題ではなく、90ペアという少量データへの追加1000 step FT（過学習）自体に起因する
    可能性が高いことを示している。→ **keyed_val 軸は「改善なし」で不合格だが、悪化の
    原因が手打ち分布かどうかは判定不能**（§4で「基準充足」と「原因の帰属」を分けて記載）。
  - **synth_val（合成+実ノイズ400件）**: 対照群はほぼ横ばい（TER +0.03pt、
    CER −0.03pt、いずれも誤差の範囲）だったのに対し、手打ち分布アームは両方とも
    明確に悪化した（tail: TER +0.49pt / CER +0.48pt、notail: TER +1.19pt / CER +0.94pt）。
    対照群が悪化していない以上、この悪化は**追加FTという行為自体ではなく、手打ち分布
    そのものに起因する**と結論できる。→ **synth_val 軸は不採用が確定**。

- **総合判定**: 採用基準は「keyed_val が改善し、かつ synth_val が悪化しないこと」。
  synth_val 側は交絡なしで悪化が確認され、この時点で基準を満たさないことが確定する。
  keyed_val 側がなぜ改善しなかったかは判定不能（過学習との交絡）だが、たとえその交絡を
  解消して将来 keyed_val が改善したとしても、synth_val 側の不合格は独立に確定しているため、
  **今回の手打ち分布FTレシピは総合として不採用**とする。ただし「手打ち分布自体が
  keyed_val を悪化させた」とは言えない点は明記しておく（§5参照）。

---

## 1. 実行条件

`models/ft_1k/best.pt`（`best_infer.pt` と39テンソル全一致、フェーズAで確認済み）を起点に、
`data/real/train`（90ペア、角括弧表記なしのクリーンな実打鍵データ）を用いて
`--mix-synth --real-ratio 0.7` で 1000 step ファインチューニングした。`--num-workers 0` は
GPU の孤児プロセスを避けるため必須（過去に踏んだ実績あり）。

```
# アーム1: 極端テールあり（手打ち分布）
.venv/Scripts/python.exe scripts/finetune.py \
  --data-dir data/real/train --resume models/ft_1k/best.pt \
  --ckpt-dir models/ft_hand_tail --steps 1000 --num-workers 0 \
  --mix-synth --real-ratio 0.7

# アーム2: 極端テールなし（手打ち分布）
.venv/Scripts/python.exe scripts/finetune.py \
  --data-dir data/real/train --resume models/ft_1k/best.pt \
  --ckpt-dir models/ft_hand_notail --steps 1000 --num-workers 0 \
  --mix-synth --real-ratio 0.7 --no-extreme-tail

# アーム3（対照群）: 従来分布（手打ちなし）。起点・データ・step数・real_ratioは同一
.venv/Scripts/python.exe scripts/finetune.py \
  --data-dir data/real/train --resume models/ft_1k/best.pt \
  --ckpt-dir models/ft_control_old --steps 1000 --num-workers 0 \
  --mix-synth --real-ratio 0.7 --no-hand-keying
```

学習時ログの eval（`data/real/train` 内の 18 サンプル held-out）は3アームとも
step 400〜600 付近で TER 0.00% に到達しているが、これは学習データと同分布のミニ評価であり、
過学習の兆候であって精度の実力を示すものではない。判定は下記の keyed_val / synth_val /
held-out 追加サンプルの3系統でのみ行う。

---

## 2. keyed_val（実打鍵20件）・synth_val（合成+実ノイズ400件）評価

```
.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/ft_hand_tail/best.pt --keyed-dir data/keying_scripts \
  --noise-dir data/keying_scripts --out models/eval/ft_hand_tail.json \
  --baseline models/eval/baseline.json --device cuda

.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/ft_hand_notail/best.pt --keyed-dir data/keying_scripts \
  --noise-dir data/keying_scripts --out models/eval/ft_hand_notail.json \
  --baseline models/eval/baseline.json --device cuda

.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/ft_control_old/best.pt --keyed-dir data/keying_scripts \
  --noise-dir data/keying_scripts --out models/eval/ft_control_old.json \
  --baseline models/eval/baseline.json --device cuda
```

| 指標 | baseline | 対照群（従来分布） | tail あり | tail なし |
|---|---|---|---|---|
| keyed_val TER | 23.2662% | 26.6219%（+3.36pt） | 25.9508%（+2.68pt） | 28.6353%（+5.37pt） |
| keyed_val CER | 32.5688% | 35.0917%（**+2.52pt**） | 32.5688%（±0.00pt） | 33.0275%（+0.46pt） |
| synth_val TER | 35.3810% | 35.4149%（+0.03pt≈横ばい） | 35.8731%（+0.49pt） | 36.5688%（+1.19pt） |
| synth_val CER | 41.0506% | 41.0176%（−0.03pt≈横ばい） | 41.5281%（+0.48pt） | 41.9891%（+0.94pt） |

出典: `models/eval/baseline.json`、`models/eval/ft_control_old.json`、
`models/eval/ft_hand_tail.json`、`models/eval/ft_hand_notail.json`
（各 `keyed_val.overall` / `synth_val.overall`）。n はそれぞれ 20 / 400。

**keyed_val**: 4条件中もっとも悪いのは notail（+5.37pt）、次いで対照群（+3.36pt）、
tail が最も悪化幅が小さい（+2.68pt）という順序であり、**手打ち分布アームが対照群より
一貫して悪いという関係にはなっていない**。CER で見ると対照群がむしろ最悪
（tail は baseline と同値）であり、手打ち分布そのものが keyed_val を悪化させたとは
言えない。90ペアという少量データへの追加1000 step FTという行為自体が、分布の種類に
関わらず keyed_val を悪化させている可能性が高い。

**synth_val**: 対照群は baseline とほぼ同値（TER +0.03pt、CER −0.03pt）で、
統計的に意味のある悪化とは言えない。一方 tail・notail は synth_val TER/CER 双方で
明確に悪化しており、悪化幅は tail < notail の順（極端テールを外すとむしろ synth_val への
悪影響が増える）。対照群が横ばいである以上、この悪化は追加FTという行為そのものではなく
**手打ち分布の導入自体に起因する**と判断できる。

---

## 3. held-out 追加サンプル（`data/keyed_extra`）

```
.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/ft_hand_tail/best.pt --keyed-dir data/keyed_extra \
  --out models/eval/extra_tail.json --device cuda

.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/ft_hand_notail/best.pt --keyed-dir data/keyed_extra \
  --out models/eval/extra_notail.json --device cuda

.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/ft_control_old/best.pt --keyed-dir data/keyed_extra \
  --out models/eval/extra_control_old.json --device cuda

.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/full/best_infer.pt --keyed-dir data/keyed_extra \
  --out models/eval/extra_baseline.json --device cuda
```

| 指標 | baseline | 対照群 | tail あり | tail なし |
|---|---|---|---|---|
| CER（n=1, 参照28トークン） | 42.3077% | 38.4615%（改善） | 46.1538%（悪化） | 42.3077%（同値） |
| TER | 39.2857% | 35.7143%（改善） | 42.8571%（悪化） | 35.7143%（改善） |

出典: `models/eval/extra_baseline.json`、`models/eval/extra_control_old.json`、
`models/eval/extra_tail.json`、`models/eval/extra_notail.json`。

**baseline 値について**: 本サンプル単体の baseline は **CER 42.3077%** で確定する。
これは `eval_model.py` を用いて他の3アームと完全に同一の手順・条件で測定した値であり、
事前に共有されていた「40.0%」は空白除去後の文字列に対する `error_rate` 直接呼び出しという
異なる指標経路（`eval_model.py` の指標経路とは別物）による値だったため、本レポートでは
不使用とし、42.3077% に統一する。

n=1 のため単独では統計的な結論を出せないが、対照群と notail が baseline を上回り
（改善）、tail のみ悪化している。§2 の n=20 / n=400 の結果と合わせて見ても、この
n=1 サンプル単独から §0 の結論を覆す材料は得られない。

---

## 4. 採否判定

採用基準: **keyed_val が改善し、かつ synth_val が悪化しないこと。片方だけなら不採用。**

| 評価軸 | 基準充足（改善したか） | 原因の帰属（なぜ悪化したか） |
|---|---|---|
| keyed_val | **改善なし（不合格）**。TER は tail +2.68pt・notail +5.37pt・対照群 +3.36pt で、全アームとも baseline を上回れていない | **原因は交絡により特定不能**（追加1000 step FTの過学習か手打ち分布かを分離できていない） |
| synth_val | **改善なし（不合格）**。tail +0.48〜0.49pt・notail +0.94〜1.19pt 悪化 | **手打ち分布に起因すると特定できる**（対照群は横ばいで交絡なし） |

「基準充足」列で見るとおり、keyed_val は全アームで明確に不合格であり、判定不能なのは
基準を満たしたかどうかではなく「なぜ悪化したか」という因果の帰属である。採用基準は
AND条件であり、synth_val 側が交絡なしで不合格と確定した時点で総合判定は不採用となる。
keyed_val 側の原因帰属の不能（判定不能）は「手打ち分布は無罪の可能性がある」ことを
示すに留まり、「手打ち分布は無罪である」ことの証明にはならない点に注意。

**総合判定: 不採用。** `models/full/best_infer.pt`（配布モデル）はそのまま維持する。
`models/ft_hand_tail/` `models/ft_hand_notail/` `models/ft_control_old/` の書き出し
（`export_infer_checkpoint.py` / `export_onnx.py` / `export_golden.py`）は行わない。

過去の実験との対比:

| 実験 | keyed_val | synth_val | 不採用の理由 |
|---|---|---|---|
| B1（実効SNR拡張学習） | +19.8pt 悪化 | −25.1pt 改善 | synth 改善・keyed 悪化 |
| 実ノイズ混合FT | +1.1〜+2.8pt 悪化 | 改善 | synth 改善・keyed 悪化 |
| WORD_BREAK 閾値掃引 | +4.03pt 改善 | +2.75pt 悪化 | keyed 改善・synth 悪化 |
| **本タスク（対照群込み3アーム）** | **未改善（交絡あり・判定不能）** | **手打ちアームのみ悪化（交絡なし・確定）** | **synth側は手打ち分布起因で確定不合格。keyed側は追加FT自体の影響と分離できず判定不能** |

過去3回はいずれか一方の指標が明確に改善するトレードオフ構造だったが、今回はどちらの指標も
改善しなかった。ただし今回新たに分かったのは、**keyed_val の悪化は手打ち分布固有の問題では
なく、少量実データへの追加FTという手法自体に起因する可能性が高い**という点である
（対照群がCERで最も悪化していることがその根拠）。

---

## 5. なぜ効かなかったか（考察）

### 5.1 keyed_val（判定不能の理由）

`models/ft_1k/best.pt` は既に `data/real/train`（同じ90ペア）で1000 step FT済みである。
そこへ同一データで**さらに**1000 step 追加FTを行うと、分布の種類（従来分布 / 手打ち分布
tail あり / tail なし）によらず keyed_val TER が軒並み悪化した（+2.68〜+5.37pt）。
CER では対照群（従来分布）が最も悪化している。これは「同じ90ファイルに対する2回目の
1000 step FT」が、モデルを keyed_val 全体の汎化から遠ざけ、90ペア内の特徴（あるいは
train内held-out 18サンプルへの過学習、学習ログでTER 0.00%に到達している点と整合）に
過剰適合させた可能性が高い。手打ち分布の導入がこれを悪化させたのか改善させたのかは、
今回の実験からは判定できない。

### 5.2 synth_val（不採用確定の理由）

対照群（従来分布での追加FT）は synth_val をほぼ動かしていない（±0.03pt）。
これに対し手打ち分布アームは tail で +0.48〜0.49pt、notail で +0.94〜1.19pt 悪化しており、
悪化幅は「極端テールを外す」方が大きいという直感に反する結果になった。手打ちキーイングの
タイミング分布（長音ジッタ・符号間ギャップ等）を合成データに混ぜたことで、
`data/real/train` という小さな実データセット（90ペア）に対して、モデルが
synth_val が使う従来のタイミング分布から乖離する方向に適合してしまった可能性がある。
extreme_tail を外すとむしろ悪化が大きいことから、単純に「手打ち分布の分散が大きすぎる」
という説明だけでは足りず、tail の有無による分布形状の違いが学習のどこに効いているかは
本タスクの範囲では特定していない。

---

## 6. 次の一手への示唆

- **FTレシピ自体（分布に依存しない部分）の見直しが優先度が高い。** 候補:
  - step数を減らす（例: 200〜400 step、学習ログでは200 step時点でも既にkeyed_val的な
    過学習の兆候が出ている可能性がある）
  - learning rate を下げる
  - `models/ft_1k/best.pt` ではなく素の `models/full/best.pt` から再開する
    （同一データへの2回目のFTという交絡そのものを避けられる）
  - `--real-ratio` を下げて実データへの依存度を下げる
- **手打ち分布そのものの synth_val への悪影響**は対照群比較で交絡なく確認されたため、
  現状の手打ち分布実装をそのまま追加FTに使うのは推奨しない。ただし Task 1〜6 で
  合成器に実装した手打ち分布自体（`--mix-synth` 併用時の分布拡張機能）を否定するものではなく、
  今回検証したのは「`data/real/train` を使った追加ファインチューニングにこの分布を使う」
  という利用方法の効果である。
- `data/real/` 直下の4ファイル（`[ホレ]` `[ラタ]` 等の角括弧表記、`text_to_codes` が
  `KeyError` でクラッシュする）はラベル修正すれば `data/real/train` に追加できる可能性があるが、
  そのうち3つはラベルが `[ホレ]` のみ・1つは空であり、学習データとしての価値は低い。

---

## 7. 成果物

- `scripts/finetune.py`: `--no-hand-keying` / `--no-extreme-tail` / `--electronic-keyer-prob`
  CLI オプション追加（フェーズA、コミット済み）
- `tests/test_finetune_dataset.py`: CLI 配線テスト追加（フェーズA、コミット済み）
- `models/ft_hand_tail/`, `models/ft_hand_notail/`, `models/ft_control_old/`:
  ファインチューニング成果物（`.gitignore` 対象、git 管理下になし）
- `models/eval/ft_hand_tail.json`, `models/eval/ft_hand_notail.json`,
  `models/eval/ft_control_old.json`, `models/eval/extra_tail.json`,
  `models/eval/extra_notail.json`, `models/eval/extra_control_old.json`,
  `models/eval/extra_baseline.json`: 評価結果（`.gitignore` 対象、git 管理下になし）
- 配布モデル (`models/full/best_infer.pt`) への変更: **なし**（不採用のため）
