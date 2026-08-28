# L4 弱信号 held-out (凍結)

2026-08-27 に `data/l4/20260824_5w_18m` (5W・コントラスト中央 11.7 dB) から取り分けた 45 件。

**全モデルが本文もファイルも未学習であることを実測で確認した**
(`data/ft/{l4_ja_only,l4_bal90,real_train,v4/train,v5/train}` のどれにも
ファイル名で入っておらず、本文の重複も 33 本文中 2 件のみ)。

**以後の学習に使わないこと。** `scripts/split_real_data.py --exclude data/heldout` で除外する。

役割: 実運用に近い**弱信号の動作点**を測る。運用者の実受信で最良だった `wabun` を、
`data/keying_scripts` (held-out) は「baseline より悪い」と評価したが、この集合は
0.53% と正しく突出を示した (`docs/models_and_results.md` §3)。

## 偏り (承知のうえで使う)

**欧文 38 件 / 和文 7 件** と欧文に偏っている。`wabun` の学習データ
(`data/ft/combined_ja`) が和文中心 (709/739) で、8/24 の和文はほぼ学習に
使われたため、未学習で残ったのが欧文だった。
**和文の弱信号性能はこの集合では測れない。** 8/25 の val (`data/ft/v5/val`、
和文 48 件) と併せて見ること。
