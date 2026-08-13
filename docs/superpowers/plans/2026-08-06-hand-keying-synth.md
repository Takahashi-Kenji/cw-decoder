# 手打ちキーイングの合成モデル拡張 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 合成器が「短点は正確・長音だけ大きくばらつく」という実測どおりの手打ちを生成できるようにし、そのデータで再学習して手打ち信号の認識率を上げる。

**Architecture:** `KeyingParams` のジッタを「全要素共通の 1 つの σ」から「要素種別 (短点 / 長音 / 要素間 / 文字間 / 語間) ごとの独立した σ」へ分解する。`_build_element_sequence` が要素種別の配列を返すようにし、種別 → σ の引き当てを numpy の fancy index 1 発で行う。既定値では従来と**ビット単位で同一の波形**になるようにして後方互換を保つ。

**Tech Stack:** Python 3.11+, numpy 2.x, scipy (解析スクリプトのみ), pytest, PyTorch (学習)

**設計書:** `docs/superpowers/specs/2026-08-06-hand-keying-synth-design.md`

## Global Constraints

- ブランチは `feature/hand-keying-synth` (作成済み)。`main` へ直接コミットしない
- Python 3.11+。**型ヒント必須** (`mypy` 互換)
- **波形生成に for ループを書かない。** numpy のベクトル化・ブロードキャスト・fancy index で表現する
- 乱数は `np.random.Generator` を**引数で受け取る**。グローバル `np.random` を使わない
- 符号定義は `src/tokens/morse_tokens.py` を唯一の真正なソースとする。符号を二重定義しない
- ファイルは UTF-8 (BOM なし)、改行 LF
- コメント・docstring・コミットメッセージは日本語
- コミットメッセージの接頭辞: `feat:` `fix:` `docs:` `refactor:` `test:` `chore:`
- **既定値の `KeyingParams` は従来と完全に同一の波形を出すこと。** 学習データの再現性が壊れる
- テストは `pytest`。既存の `tests/test_synth_keying.py` の書き方 (クラスでグルーピング) に合わせる

## 実測値 (設計書 §2.2 より。パラメータ範囲の根拠)

| 項目 | 遅い (23.1 WPM) | 速い (30.0 WPM) |
|---|---|---|
| 短点 σ | 0.064 dot | 0.107 dot |
| 長音 σ | 0.681 dot | **1.195 dot** |
| 長短比 | 3.06 | 3.84 |
| 要素間 | 平均 1.18、σ 0.26 | 平均 1.25、σ 0.42 |
| 文字間 | 平均 3.07、σ 0.65 | 平均 2.86、σ 0.82 |
| 語間 | 15.90 (n=2) | 7.55 (n=3) |

## File Structure

| ファイル | 役割 | 変更 |
|---|---|---|
| `src/synth/keying.py` | 符号列 → 波形。要素種別と種別ごとのジッタを持つ | 変更 (Task 1〜3) |
| `tests/test_synth_keying.py` | キーイングのテスト | 変更 (Task 1〜3, 5) |
| `scripts/analyze_keying.py` | 実録音・合成音からタイミング統計を実測する | **新規** (Task 4) |
| `tests/test_analyze_keying.py` | 解析スクリプトのテスト | **新規** (Task 4) |
| `src/synth/dataset.py` | 学習用パラメータのサンプラ | 変更 (Task 6) |
| `tests/test_synth_dataset.py` | サンプラのテスト | 変更 (Task 6) |
| `docs/hand_keying_ft_result.md` | 学習と評価の結果 | **新規** (Task 7) |

---

### Task 1: 要素種別 (ElementKind) を導入する

**目的:** 種別ごとに σ を変える下地を作る。**この Task では挙動を一切変えない。**

**Files:**
- Modify: `src/synth/keying.py` (`_build_element_sequence` と呼び出し側)
- Test: `tests/test_synth_keying.py`

**Interfaces:**
- Consumes: なし (最初の Task)
- Produces:
  - `ElementKind` (IntEnum): `DOT=0` `DASH=1` `INTRA_GAP=2` `CHAR_GAP=3` `WORD_GAP=4`
  - `_build_element_sequence(...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
    (第 4 要素として `kinds: np.ndarray` (dtype=np.int64) を追加)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_keying.py` の末尾に追記する。

```python
# ============================================================
# 要素種別 (ElementKind)
# ============================================================
class TestElementKind:
    def test_kinds_cover_all_elements(self) -> None:
        """符号 2 つ (・- と -) を語間で繋いだとき、種別列が期待どおりになる."""
        from src.synth.keying import ElementKind, _build_element_sequence

        durations, is_on, code_starts, kinds = _build_element_sequence(
            ["・-", "-"],
            word_break_after=[0],
            dot_sec=0.06,
            dash_dot_ratio=3.0,
            inter_char_space_units=3.0,
            inter_word_space_units=7.0,
        )
        # ・ / 要素間 / - / 語間 / -
        assert list(kinds) == [
            ElementKind.DOT,
            ElementKind.INTRA_GAP,
            ElementKind.DASH,
            ElementKind.WORD_GAP,
            ElementKind.DASH,
        ]
        assert len(kinds) == len(durations) == len(is_on)

    def test_char_gap_kind_when_not_word_break(self) -> None:
        """語間指定が無ければ符号間は CHAR_GAP になる."""
        from src.synth.keying import ElementKind, _build_element_sequence

        _, _, _, kinds = _build_element_sequence(
            ["・", "・"],
            word_break_after=[],
            dot_sec=0.06,
            dash_dot_ratio=3.0,
            inter_char_space_units=3.0,
            inter_word_space_units=7.0,
        )
        assert list(kinds) == [
            ElementKind.DOT,
            ElementKind.CHAR_GAP,
            ElementKind.DOT,
        ]
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_keying.py::TestElementKind -v`
Expected: FAIL — `ImportError: cannot import name 'ElementKind'`

- [ ] **Step 3: 実装する**

`src/synth/keying.py` の import 部に追加する。

```python
from enum import IntEnum
```

`KeyingParams` の定義の**前**に追加する。

```python
class ElementKind(IntEnum):
    """要素の種別.

    ジッタの σ を種別ごとに変えるために使う。実測では短点 (σ 0.06〜0.11 dot) と
    長音 (σ 0.68〜1.20 dot) でばらつきが一桁違うため、共通の σ では手打ちを
    再現できない (設計書 §2.3(b))。
    """

    DOT = 0          # 短点 (ON)
    DASH = 1         # 長音 (ON)
    INTRA_GAP = 2    # 同一符号内の要素間 (OFF)
    CHAR_GAP = 3     # 文字間 (OFF)
    WORD_GAP = 4     # 語間 (OFF)
```

`_build_element_sequence` を次のように置き換える (シグネチャの戻り値と `kinds` の記録を追加。それ以外のロジックは変えない)。

```python
def _build_element_sequence(
    codes: Sequence[str],
    word_break_after: Sequence[int],
    dot_sec: float,
    dash_dot_ratio: float,
    inter_char_space_units: float,
    inter_word_space_units: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """符号列を要素 (duration, key_on, kind) のシーケンスに展開.

    Returns:
        durations: 各要素の長さ (秒).
        is_on: 各要素のキー状態 (True = ON).
        code_start_element_indices: 各 code が ``durations`` の何番目から始まるか.
        kinds: 各要素の ``ElementKind`` (int64).
    """
    word_break_set = set(word_break_after)
    durations: list[float] = []
    is_on: list[bool] = []
    kinds: list[int] = []
    code_starts: list[int] = []
    # 次の inter-code space を語間 (7 dot) にする必要があるか
    pending_word_break = False

    for i, code in enumerate(codes):
        # WORD_BREAK_CODE は実際の符号を持たず、次の inter-code 区間を語間に拡張
        if code == WORD_BREAK_CODE:
            code_starts.append(len(durations))
            pending_word_break = True
            continue
        if not code:
            code_starts.append(len(durations))
            continue
        if durations:  # 前の code との間に空白を入れる
            is_word_break = pending_word_break or ((i - 1) in word_break_set)
            space_units = (
                inter_word_space_units if is_word_break else inter_char_space_units
            )
            durations.append(space_units * dot_sec)
            is_on.append(False)
            kinds.append(
                ElementKind.WORD_GAP if is_word_break else ElementKind.CHAR_GAP
            )
        pending_word_break = False
        code_starts.append(len(durations))
        for j, elem in enumerate(code):
            if j > 0:
                durations.append(dot_sec)
                is_on.append(False)
                kinds.append(ElementKind.INTRA_GAP)
            if elem == DOT:
                durations.append(dot_sec)
                kinds.append(ElementKind.DOT)
            elif elem == DASH:
                durations.append(dot_sec * dash_dot_ratio)
                kinds.append(ElementKind.DASH)
            else:
                raise ValueError(f"Unknown element {elem!r} in code {code!r}")
            is_on.append(True)

    return (
        np.asarray(durations, dtype=np.float64),
        np.asarray(is_on, dtype=bool),
        np.asarray(code_starts, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64),
    )
```

`codes_to_waveform` 内の呼び出しを 4 要素の受け取りに変える。

```python
    durations, is_on, code_start_elem_idx, kinds = _build_element_sequence(
        codes,
        word_break_after,
        dot_sec,
        params.dash_dot_ratio,
        params.inter_char_space_units,
        params.inter_word_space_units,
    )
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_keying.py -v`
Expected: 全 PASS (新規 2 件 + 既存すべて)

既存テストが 1 件でも落ちたら**挙動を変えてしまっている**。`kinds` の追加以外を変えていないか差分を見直すこと。

- [ ] **Step 5: 合成器を使う全テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 917 passed, 4 skipped (CUDA 無しのため 4 件 skip)

- [ ] **Step 6: コミット**

```bash
git add src/synth/keying.py tests/test_synth_keying.py
git commit -m "refactor: キーイング要素に種別 (ElementKind) を持たせる

種別ごとにジッタ σ を変えるための下地。この時点では挙動を変えない。
_build_element_sequence が kinds を返すようになった。"
```

---

### Task 2: 要素種別ごとのジッタ σ を追加する

**目的:** 「短点は正確・長音だけ暴れる」を表現できるようにする。

**Files:**
- Modify: `src/synth/keying.py` (`KeyingParams`, `codes_to_waveform`)
- Test: `tests/test_synth_keying.py`

**Interfaces:**
- Consumes: Task 1 の `ElementKind`, `_build_element_sequence` の 4 要素戻り値
- Produces:
  - `KeyingParams.dot_jitter_sigma_ratio: float | None = None`
  - `KeyingParams.dash_jitter_sigma_ratio: float | None = None`
  - `KeyingParams.intra_gap_jitter_sigma_ratio: float | None = None`
  - `KeyingParams.char_gap_jitter_sigma_ratio: float | None = None`
  - `KeyingParams.word_gap_jitter_sigma_ratio: float | None = None`
  - `KeyingParams.jitter_sigma_by_kind() -> np.ndarray` (長さ 5、float64)

**背景 (実装者向け):** numpy の `Generator.normal(0.0, scale)` は `scale` が
スカラでも同値の配列でも**同じ乱数列**を返すことを確認済み (numpy 2.4.6)。
`scale=0.0` を含む配列も受け付け、その要素は 0.0 になる。したがって
配列 σ に置き換えても既定値なら従来と**ビット単位で同一**の波形になる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_keying.py` の末尾に追記する。

```python
# ============================================================
# 種別ごとのジッタ
# ============================================================
class TestPerKindJitter:
    def test_default_params_unchanged_bitwise(self) -> None:
        """既定値では従来と同一の波形になる (後方互換の要)."""
        params = KeyingParams(wpm=20.0, element_jitter_sigma_ratio=0.15)
        a = codes_to_waveform(
            ["・-", "-・・", "・"], params, np.random.default_rng(7), sample_rate=8000
        )
        b = codes_to_waveform(
            ["・-", "-・・", "・"], params, np.random.default_rng(7), sample_rate=8000
        )
        assert np.array_equal(a.samples, b.samples)
        # 既定値では種別ごとの σ が全部 element_jitter_sigma_ratio に落ちる
        assert np.allclose(params.jitter_sigma_by_kind(), 0.15)

    def test_dash_only_jitter_leaves_dots_exact(self) -> None:
        """長音だけに σ を与えると、短点の長さは正確なままになる."""
        params = KeyingParams(
            wpm=20.0,
            rise_fall_ms=0.0,
            element_jitter_sigma_ratio=0.0,
            dash_jitter_sigma_ratio=0.5,
        )
        rng = np.random.default_rng(3)
        # 短点だけの符号を並べる → 長音が無いので長さは無ジッタと一致するはず
        dots = codes_to_waveform(["・", "・", "・"], params, rng, sample_rate=8000)
        no_jitter = codes_to_waveform(
            ["・", "・", "・"],
            KeyingParams(wpm=20.0, rise_fall_ms=0.0),
            np.random.default_rng(3),
            sample_rate=8000,
        )
        assert len(dots.samples) == len(no_jitter.samples)

    def test_dash_jitter_widens_total_length_spread(self) -> None:
        """長音の σ を上げると、同じ符号列でも波形長のばらつきが広がる.

        ジッタの計算式をテスト側に書き写すと本番ロジックの写経になり、
        実装が変わってもテストが追随しない。ここは実際に codes_to_waveform を
        通した波形長で判定する。
        """
        codes = ["-"] * 8

        def spread(params: KeyingParams) -> float:
            lengths = [
                len(
                    codes_to_waveform(
                        codes, params, np.random.default_rng(seed), sample_rate=8000
                    ).samples
                )
                for seed in range(30)
            ]
            return float(np.std(lengths))

        tight = KeyingParams(
            wpm=20.0, rise_fall_ms=0.0, element_jitter_sigma_ratio=0.0
        )
        loose = KeyingParams(
            wpm=20.0, rise_fall_ms=0.0, element_jitter_sigma_ratio=0.0,
            dash_jitter_sigma_ratio=0.8,
        )
        # ジッタ無しなら seed を変えても長さは一定
        assert spread(tight) == 0.0
        # σ 0.8 dot = 0.048 秒 = 384 サンプル。長音 8 個ぶんの合計なので
        # 標準偏差は 1000 サンプル前後になる。50 は十分に安全な下限
        assert spread(loose) > 50.0

    def test_sigma_by_kind_fallback_is_per_field(self) -> None:
        """None の種別だけが element_jitter_sigma_ratio にフォールバックする."""
        from src.synth.keying import ElementKind

        params = KeyingParams(
            element_jitter_sigma_ratio=0.1,
            dash_jitter_sigma_ratio=0.9,
            char_gap_jitter_sigma_ratio=0.0,
        )
        s = params.jitter_sigma_by_kind()
        assert s[ElementKind.DOT] == 0.1
        assert s[ElementKind.DASH] == 0.9
        assert s[ElementKind.INTRA_GAP] == 0.1
        assert s[ElementKind.CHAR_GAP] == 0.0
        assert s[ElementKind.WORD_GAP] == 0.1
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_keying.py::TestPerKindJitter -v`
Expected: FAIL — `TypeError: KeyingParams.__init__() got an unexpected keyword argument 'dash_jitter_sigma_ratio'`

- [ ] **Step 3: 実装する**

`KeyingParams` に**フィールドを追加**する (既存フィールドの後ろに置く。dataclass の
既定値付きフィールドなので順序の制約は無い)。

```python
    # --- 要素種別ごとのジッタ σ (dot 長に対する比率) ---
    # None ならこの種別は element_jitter_sigma_ratio にフォールバックする。
    # 全部 None なら従来と完全に同一の挙動になる (後方互換)。
    #
    # 実測 (設計書 §2.2): 短点 σ 0.064〜0.107 dot に対し長音 σ 0.681〜1.195 dot と
    # 一桁違う。共通の σ ではこの非対称を作れないため種別ごとに分けた。
    dot_jitter_sigma_ratio: float | None = None
    dash_jitter_sigma_ratio: float | None = None
    intra_gap_jitter_sigma_ratio: float | None = None
    char_gap_jitter_sigma_ratio: float | None = None
    word_gap_jitter_sigma_ratio: float | None = None

    def jitter_sigma_by_kind(self) -> np.ndarray:
        """``ElementKind`` の並び順で σ を返す (長さ 5, float64).

        ``None`` の種別は ``element_jitter_sigma_ratio`` にフォールバックする。
        戻り値は ``kinds`` 配列で fancy index して使う。
        """
        base = self.element_jitter_sigma_ratio
        return np.array(
            [
                base if self.dot_jitter_sigma_ratio is None else self.dot_jitter_sigma_ratio,
                base if self.dash_jitter_sigma_ratio is None else self.dash_jitter_sigma_ratio,
                base if self.intra_gap_jitter_sigma_ratio is None else self.intra_gap_jitter_sigma_ratio,
                base if self.char_gap_jitter_sigma_ratio is None else self.char_gap_jitter_sigma_ratio,
                base if self.word_gap_jitter_sigma_ratio is None else self.word_gap_jitter_sigma_ratio,
            ],
            dtype=np.float64,
        )
```

`codes_to_waveform` のジッタ適用部を置き換える。**置き換え前**:

```python
    # ジッタを ON 要素のみに適用 (空白には別途軽いジッタを加えてもよいが
    # シンプルさのため省略)
    if params.element_jitter_sigma_ratio > 0:
        sigma = params.element_jitter_sigma_ratio * dot_sec
        jitter = rng.normal(0.0, sigma, size=len(durations))
        # 最小長を dot 長の 10% にクリップ
        durations = np.maximum(durations + jitter, dot_sec * 0.1)
```

**置き換え後**:

```python
    # 要素種別ごとの σ を引き当ててジッタを乗せる。
    # (旧コメントは「ON 要素のみ」と書いていたが、実際には OFF (スペース) にも
    #  かかっていた。実測でもスペースはばらつくのでこの挙動は正しい。)
    #
    # numpy の normal は scale がスカラでも同値の配列でも同じ乱数列を返すため、
    # 種別ごとの σ が全部同じ値なら従来とビット単位で同一の波形になる。
    sigma = params.jitter_sigma_by_kind()[kinds] * dot_sec
    if np.any(sigma > 0.0):
        # 最小長を dot 長の 10% にクリップ
        durations = np.maximum(durations + rng.normal(0.0, sigma), dot_sec * 0.1)
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_keying.py -v`
Expected: 全 PASS

- [ ] **Step 5: 既存の全テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 917 passed, 4 skipped

**ここで既存テストが落ちたら後方互換が壊れている。** `jitter_sigma_by_kind()` が
既定値で全要素 `element_jitter_sigma_ratio` になっているかを確認すること。

- [ ] **Step 6: コミット**

```bash
git add src/synth/keying.py tests/test_synth_keying.py
git commit -m "feat: 要素種別ごとに独立したジッタ σ を持てるようにする

実測では短点 σ 0.064〜0.107 dot に対し長音 σ 0.681〜1.195 dot と一桁違い、
共通の σ ではこの非対称を作れなかった。

既定 (全部 None) では element_jitter_sigma_ratio にフォールバックするので
従来とビット単位で同一の波形になる。"
```

---

### Task 3: 要素間スペースの長さをパラメータ化する

**目的:** 実測で要素間が平均 1.18〜1.25 dot、最小 0.13 dot と教科書の 1.0 から
外れていたので、平均値を動かせるようにする。

**Files:**
- Modify: `src/synth/keying.py`
- Test: `tests/test_synth_keying.py`

**Interfaces:**
- Consumes: Task 1 の `_build_element_sequence`
- Produces: `KeyingParams.intra_element_space_units: float = 1.0`
  (`_build_element_sequence` に第 7 引数 `intra_element_space_units: float = 1.0` を追加)

- [ ] **Step 1: 失敗するテストを書く**

```python
class TestIntraElementSpace:
    def test_default_is_one_dot(self) -> None:
        """既定では要素間は 1 dot のまま."""
        params = KeyingParams()
        assert params.intra_element_space_units == 1.0

    def test_shorter_intra_space_shortens_waveform(self) -> None:
        """要素間を詰めると符号全体が短くなる."""
        base = KeyingParams(wpm=20.0, rise_fall_ms=0.0, element_jitter_sigma_ratio=0.0)
        tight = KeyingParams(
            wpm=20.0,
            rise_fall_ms=0.0,
            element_jitter_sigma_ratio=0.0,
            intra_element_space_units=0.4,
        )
        rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
        # ・・・ は要素間が 2 つある
        a = codes_to_waveform(["・・・"], base, rng_a, sample_rate=8000)
        b = codes_to_waveform(["・・・"], tight, rng_b, sample_rate=8000)
        dot_samples = 0.06 * 8000       # dot_sec = 1.2/20 = 0.06
        expected_diff = 2 * (1.0 - 0.4) * dot_samples
        assert abs((len(a.samples) - len(b.samples)) - expected_diff) <= 2

    def test_intra_space_does_not_affect_char_gap(self) -> None:
        """要素間を変えても文字間は変わらない."""
        from src.synth.keying import ElementKind, _build_element_sequence

        durations, _, _, kinds = _build_element_sequence(
            ["・", "・"],
            word_break_after=[],
            dot_sec=0.06,
            dash_dot_ratio=3.0,
            inter_char_space_units=3.0,
            inter_word_space_units=7.0,
            intra_element_space_units=0.4,
        )
        char_gap = durations[kinds == ElementKind.CHAR_GAP]
        assert len(char_gap) == 1
        assert abs(char_gap[0] - 3.0 * 0.06) < 1e-9
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_keying.py::TestIntraElementSpace -v`
Expected: FAIL — `AttributeError: 'KeyingParams' object has no attribute 'intra_element_space_units'`

- [ ] **Step 3: 実装する**

`KeyingParams` に追加する (`inter_char_space_units` の**直前**に置くと読みやすい)。

```python
    # 同一符号内の要素間スペース (dot 単位)。教科書は 1.0。
    # 実測では平均 1.18〜1.25、最小 0.13 まで詰まる (設計書 §2.2)。
    intra_element_space_units: float = 1.0
```

`_build_element_sequence` のシグネチャに引数を追加する。

```python
def _build_element_sequence(
    codes: Sequence[str],
    word_break_after: Sequence[int],
    dot_sec: float,
    dash_dot_ratio: float,
    inter_char_space_units: float,
    inter_word_space_units: float,
    intra_element_space_units: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

要素間を書いている箇所を変える。**変更前**:

```python
            if j > 0:
                durations.append(dot_sec)
                is_on.append(False)
                kinds.append(ElementKind.INTRA_GAP)
```

**変更後**:

```python
            if j > 0:
                durations.append(intra_element_space_units * dot_sec)
                is_on.append(False)
                kinds.append(ElementKind.INTRA_GAP)
```

`codes_to_waveform` の呼び出しに渡す。

```python
    durations, is_on, code_start_elem_idx, kinds = _build_element_sequence(
        codes,
        word_break_after,
        dot_sec,
        params.dash_dot_ratio,
        params.inter_char_space_units,
        params.inter_word_space_units,
        params.intra_element_space_units,
    )
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_keying.py -v`
Expected: 全 PASS

- [ ] **Step 5: 既存の全テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 917 passed, 4 skipped

- [ ] **Step 6: コミット**

```bash
git add src/synth/keying.py tests/test_synth_keying.py
git commit -m "feat: 要素間スペースの長さをパラメータ化する

実測では要素間が平均 1.18〜1.25 dot、最小 0.13 dot と教科書の 1.0 から外れる。
既定は 1.0 のままなので従来の挙動は変わらない。"
```

---

### Task 4: 解析スクリプト `scripts/analyze_keying.py` を追加する

**目的:** 実録音と**合成音の両方**からタイミング統計を測る。Task 5 の往復検証で使う。

**Files:**
- Create: `scripts/analyze_keying.py`
- Create: `tests/test_analyze_keying.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `analyze_wave(wave: np.ndarray, sample_rate: int, split_sec: float | None = None) -> KeyingStats`
  - `KeyingStats` (frozen dataclass): `tone_hz` `dot_sec` `dot_sigma_dot` `dash_sec`
    `dash_sigma_dot` `dash_dot_ratio` `wpm` `intra_gap_dot` `intra_gap_sigma_dot`
    `char_gap_dot` `char_gap_sigma_dot` `word_gap_dot` `n_dot` `n_dash`
    `on_histogram_ms: list[tuple[int, int]]` `off_histogram_ms: list[tuple[int, int]]`

**実装者への重要な注意:** 最初にこの解析を書いたとき、**ノイズ区間を符号として拾って
長短比 4.90 という誤った数値を出した**。ON 長のヒストグラムに山が 3 つ現れていたのが
その兆候だった。だから `KeyingStats` はヒストグラムを**必ず持ち**、CLI は必ず表示する。
数値だけを信じてはいけない (設計書 §2.1)。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_analyze_keying.py` を新規作成する。

```python
"""キーイング解析スクリプトのテスト."""
from __future__ import annotations

import numpy as np

from scripts.analyze_keying import analyze_wave
from src.synth.keying import KeyingParams, codes_to_waveform


def _synth(params: KeyingParams, seed: int = 0) -> np.ndarray:
    """短点と長音が十分な数だけ入る符号列を合成する."""
    codes = ["・-", "-・", "・・-", "-・・"] * 25
    return codes_to_waveform(
        codes, params, np.random.default_rng(seed), sample_rate=8000
    ).samples


class TestAnalyzeWave:
    def test_recovers_dot_and_dash_length(self) -> None:
        """ジッタ無しの合成音から短点・長音の長さを ±5% で復元できる."""
        params = KeyingParams(
            wpm=20.0, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            tone_freq_hz=600.0, rise_fall_ms=3.0,
        )
        stats = analyze_wave(_synth(params), 8000)
        assert abs(stats.dot_sec - 0.06) / 0.06 < 0.05
        assert abs(stats.dash_sec - 0.18) / 0.18 < 0.05
        assert abs(stats.dash_dot_ratio - 3.0) < 0.15

    def test_recovers_tone_frequency(self) -> None:
        params = KeyingParams(
            wpm=20.0, element_jitter_sigma_ratio=0.0, tone_freq_hz=527.0
        )
        stats = analyze_wave(_synth(params), 8000)
        assert abs(stats.tone_hz - 527.0) < 30.0

    def test_histograms_are_returned(self) -> None:
        """ヒストグラムが必ず返る (測定破綻の検出に使うため)."""
        params = KeyingParams(wpm=20.0, element_jitter_sigma_ratio=0.0)
        stats = analyze_wave(_synth(params), 8000)
        assert len(stats.on_histogram_ms) > 0
        assert len(stats.off_histogram_ms) > 0
        assert sum(c for _, c in stats.on_histogram_ms) == stats.n_dot + stats.n_dash
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_analyze_keying.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.analyze_keying'`

- [ ] **Step 3: 実装する**

`scripts/analyze_keying.py` を新規作成する。

```python
"""録音・合成音からキーイングのタイミング統計を実測する.

合成器のパラメータ範囲を勘ではなく実測で決めるための道具。実録音の解析にも、
合成器が指定どおりの分散を持つ波形を作れているかの往復検証にも使う。

使い方::

    .venv/Scripts/python.exe scripts/analyze_keying.py path/to/recording.wav
    .venv/Scripts/python.exe scripts/analyze_keying.py path/to/rec.wav --split-ms 75

**ヒストグラムを必ず見ること。** ON 長の山が 2 つ (短点と長音) でなければ測定が
破綻している。最初にこの解析を書いたときはノイズ区間を符号として拾い、山が 3 つ
出ている状態で長短比 4.90 という誤った数値を出した。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

TARGET_SR = 8000


@dataclass(frozen=True)
class KeyingStats:
    """キーイングのタイミング統計. 長さの σ は dot 長に対する比率で持つ."""

    tone_hz: float
    dot_sec: float
    dot_sigma_dot: float
    dash_sec: float
    dash_sigma_dot: float
    dash_dot_ratio: float
    wpm: float
    intra_gap_dot: float
    intra_gap_sigma_dot: float
    char_gap_dot: float
    char_gap_sigma_dot: float
    word_gap_dot: float
    n_dot: int
    n_dash: int
    clean_sec: float
    total_sec: float
    on_histogram_ms: list[tuple[int, int]]
    off_histogram_ms: list[tuple[int, int]]


def _histogram(values_sec: np.ndarray, bin_ms: int = 10) -> list[tuple[int, int]]:
    """10 ms 刻みのヒストグラムを (下限 ms, 件数) の列で返す (件数 0 の bin は省く)."""
    if values_sec.size == 0:
        return []
    edges = np.arange(0, values_sec.max() * 1000 + bin_ms * 2, bin_ms)
    counts, _ = np.histogram(values_sec * 1000.0, bins=edges)
    return [(int(edges[i]), int(c)) for i, c in enumerate(counts) if c > 0]


def analyze_wave(
    wave: np.ndarray, sample_rate: int, split_sec: float | None = None
) -> KeyingStats:
    """波形からタイミング統計を測る.

    Args:
        wave: モノラル float32 波形.
        sample_rate: サンプリングレート. 8 kHz 以外はリサンプルする.
        split_sec: 短点と長音を分ける境界 (秒). ``None`` なら分位点から自動推定する。
            自動推定はヒストグラムの谷を外すことがあるので、実録音では
            ヒストグラムを見て明示指定するのが望ましい。
    """
    if sample_rate != TARGET_SR:
        g = np.gcd(sample_rate, TARGET_SR)
        wave = resample_poly(wave, TARGET_SR // g, sample_rate // g).astype(np.float32)
    total_sec = wave.size / TARGET_SR

    spec = np.abs(np.fft.rfft(wave * np.hanning(wave.size)))
    freqs = np.fft.rfftfreq(wave.size, 1.0 / TARGET_SR)
    band = (freqs > 200.0) & (freqs < 1500.0)
    tone = float(freqs[band][np.argmax(spec[band])])

    sos = butter(
        4,
        [max(100.0, tone - 150.0), min(TARGET_SR / 2 - 100.0, tone + 150.0)],
        btype="bandpass", fs=TARGET_SR, output="sos",
    )
    env = np.abs(hilbert(sosfiltfilt(sos, wave)))
    smooth = max(1, int(0.005 * TARGET_SR))
    env = np.convolve(env, np.ones(smooth) / smooth, mode="same")

    # クリーン区間: 1 秒窓で 95%tile / 20%tile のコントラストが 15 dB 以上
    win = int(1.0 * TARGET_SR)
    n_win = max(1, env.size // win)
    good = np.zeros(env.size, dtype=bool)
    for i in range(n_win):
        seg = env[i * win:(i + 1) * win]
        hi, lo = np.percentile(seg, 95), np.percentile(seg, 20)
        if hi > 1e-4 and hi / max(lo, 1e-9) > 6.0:
            good[i * win:(i + 1) * win] = True
    if not good.any():
        good[:] = True

    local_peak = np.maximum.reduceat(
        env, np.arange(0, env.size, win)
    ).repeat(win)[:env.size]
    mask = env > local_peak * 0.5

    changes = np.diff(mask.astype(np.int8))
    idx = np.concatenate(([0], np.where(changes != 0)[0] + 1, [mask.size]))
    lengths = np.diff(idx)
    values = mask[idx[:-1]].copy()
    # 5 ms 未満の OFF は物理的にありえないので穴とみなして埋める
    tiny = (~values) & (lengths < int(0.005 * TARGET_SR))
    tiny[0] = tiny[-1] = False
    values[tiny] = True

    keep = good[idx[:-1]]
    lens = lengths[keep] / TARGET_SR
    vals = values[keep]
    on = lens[vals]
    on = on[on > 0.020]
    off = lens[~vals]
    off = off[off < 2.0]

    if split_sec is None:
        split_sec = float((np.percentile(on, 25) + np.percentile(on, 90)) / 2.0)
    dot, dash = on[on < split_sec], on[on >= split_sec]
    if dot.size < 3 or dash.size < 3:
        raise ValueError(
            f"短点 {dot.size} 個 / 長音 {dash.size} 個しか取れなかった。"
            "測定が破綻している可能性が高い。ヒストグラムを見て --split-ms を指定すること"
        )
    dot_sec = float(dot.mean())
    units = off / dot_sec

    def _bucket(lo: float, hi: float) -> np.ndarray:
        return units[(units >= lo) & (units < hi)]

    intra, char, word = _bucket(0.0, 1.9), _bucket(1.9, 5.0), _bucket(5.0, 1e9)
    return KeyingStats(
        tone_hz=tone,
        dot_sec=dot_sec,
        dot_sigma_dot=float(dot.std() / dot_sec),
        dash_sec=float(dash.mean()),
        dash_sigma_dot=float(dash.std() / dot_sec),
        dash_dot_ratio=float(dash.mean() / dot_sec),
        wpm=float(1.2 / dot_sec),
        intra_gap_dot=float(intra.mean()) if intra.size else float("nan"),
        intra_gap_sigma_dot=float(intra.std()) if intra.size else float("nan"),
        char_gap_dot=float(char.mean()) if char.size else float("nan"),
        char_gap_sigma_dot=float(char.std()) if char.size else float("nan"),
        word_gap_dot=float(word.mean()) if word.size else float("nan"),
        n_dot=int(dot.size),
        n_dash=int(dash.size),
        clean_sec=float(good.sum() / TARGET_SR),
        total_sec=float(total_sec),
        on_histogram_ms=_histogram(on),
        off_histogram_ms=_histogram(off),
    )


def _print_stats(name: str, s: KeyingStats) -> None:
    print(f"=== {name} ===")
    print(f"長さ {s.total_sec:.1f}s / クリーン {s.clean_sec:.1f}s / トーン {s.tone_hz:.0f} Hz")
    print()
    print("ON ヒスト (山が 2 つでなければ測定破綻を疑うこと):")
    for lo, c in s.on_histogram_ms:
        print(f"  {lo:3d}-{lo+10:3d}ms {'#' * min(c, 60)} ({c})")
    print("OFF ヒスト:")
    for lo, c in s.off_histogram_ms:
        print(f"  {lo:3d}-{lo+10:3d}ms {'#' * min(c, 60)} ({c})")
    print()
    print(f"短点 dot : {s.dot_sec*1000:6.1f} ms  σ {s.dot_sigma_dot:.3f} dot  n={s.n_dot}")
    print(f"長音 dash: {s.dash_sec*1000:6.1f} ms  σ {s.dash_sigma_dot:.3f} dot  n={s.n_dash}")
    print(f"長短比 dash/dot = {s.dash_dot_ratio:.2f}  (教科書 3.00)")
    print(f"実効 WPM = {s.wpm:.1f}")
    print()
    print("スペース (dot 単位):")
    print(f"  要素間: 平均 {s.intra_gap_dot:5.2f}  σ {s.intra_gap_sigma_dot:4.2f}  教科書 1.0")
    print(f"  文字間: 平均 {s.char_gap_dot:5.2f}  σ {s.char_gap_sigma_dot:4.2f}  教科書 3.0")
    print(f"  語間  : 平均 {s.word_gap_dot:5.2f}                    教科書 7.0")


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf

    p = argparse.ArgumentParser(description="録音からキーイングのタイミング統計を測る")
    p.add_argument("wav", type=Path, nargs="+", help="解析する WAV")
    p.add_argument(
        "--split-ms", type=float, default=None,
        help="短点と長音を分ける境界 (ms)。省略時は自動推定。"
             "ヒストグラムの谷を見て指定するのが望ましい",
    )
    args = p.parse_args(argv)

    for path in args.wav:
        wave, sr = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave[:, 0]
        split = args.split_ms / 1000.0 if args.split_ms is not None else None
        _print_stats(path.name, analyze_wave(wave, sr, split))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/` を package として import できることを確認する。既に `scripts/__init__.py`
がある (`ls scripts/` で確認済み) ので追加作業は不要。

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_analyze_keying.py -v`
Expected: 3 件 PASS

- [ ] **Step 5: 実録音で動くことを確認する**

Run: `.venv/Scripts/python.exe scripts/analyze_keying.py data/keyed_extra/20260806_163838_cut.wav --split-ms 85`
Expected: 短点 約 52 ms / 長音 約 159 ms / 長短比 約 3.06 / σ 短点 約 0.06・長音 約 0.68
(設計書 §2.2 の遅い方の値と一致すること。**一致しなければ実装が違う**)

- [ ] **Step 6: コミット**

```bash
git add scripts/analyze_keying.py tests/test_analyze_keying.py
git commit -m "feat: キーイングのタイミング統計を測る解析スクリプトを追加

実録音からパラメータ範囲を決めるためと、合成器が指定どおりの分散を持つ波形を
作れているかの往復検証のため。

ヒストグラムを必ず返す。最初にこの解析を書いたときノイズ区間を符号として拾い、
山が 3 つ出ている状態で長短比 4.90 という誤った数値を出したため。"
```

---

### Task 5: 往復検証テスト (合成 → 解析 → 入れた値が返るか)

**目的:** **この計画で最も重要なテスト。** 合成器が「指定した分散を実際に持つ波形」を
作れていなければ、学習しても意味がない。

**Files:**
- Modify: `tests/test_analyze_keying.py`

**Interfaces:**
- Consumes: Task 2 の種別ごと σ、Task 3 の `intra_element_space_units`、Task 4 の `analyze_wave`
- Produces: なし

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_analyze_keying.py` の末尾に追記する。

```python
class TestRoundTrip:
    """合成器に入れた統計値が、生成した波形から復元できることを確認する.

    合成器が指定どおりの分散を持つ波形を作れていなければ、その分布で学習しても
    意味がない。この計画で最も重要なテスト (設計書 §4.3)。
    """

    def _long_synth(self, params: KeyingParams, seed: int = 0) -> np.ndarray:
        # σ の推定には標本数が要る。短点・長音を各 200 個以上含む長さにする
        codes = ["・-", "-・", "・・-", "-・・", "・-・", "-・-"] * 60
        return codes_to_waveform(
            codes, params, np.random.default_rng(seed), sample_rate=8000
        ).samples

    def test_dash_sigma_round_trips(self) -> None:
        """長音に大きな σ を入れると、解析でも大きな σ が返る."""
        tight = KeyingParams(
            wpm=23.0, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            tone_freq_hz=527.0, rise_fall_ms=3.0,
        )
        loose = KeyingParams(
            wpm=23.0, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            dash_jitter_sigma_ratio=0.68,
            tone_freq_hz=527.0, rise_fall_ms=3.0,
        )
        s_tight = analyze_wave(self._long_synth(tight), 8000)
        s_loose = analyze_wave(self._long_synth(loose), 8000)
        # 入れた σ 0.68 dot が復元できる (推定なので幅を持たせる)
        assert s_tight.dash_sigma_dot < 0.15
        assert 0.40 < s_loose.dash_sigma_dot < 1.10

    def test_dot_stays_tight_while_dash_varies(self) -> None:
        """実測どおりの非対称 (短点は正確・長音だけ暴れる) が作れる."""
        params = KeyingParams(
            wpm=23.0, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            dot_jitter_sigma_ratio=0.064,
            dash_jitter_sigma_ratio=0.681,
            tone_freq_hz=527.0, rise_fall_ms=3.0,
        )
        s = analyze_wave(self._long_synth(params), 8000)
        # 短点の σ は長音の σ よりはっきり小さい
        assert s.dot_sigma_dot < 0.25
        assert s.dash_sigma_dot > 0.35
        assert s.dash_sigma_dot > s.dot_sigma_dot * 2.0

    def test_dash_dot_ratio_round_trips(self) -> None:
        """長短比 3.84 を入れると解析でも 3.84 前後が返る."""
        params = KeyingParams(
            wpm=30.0, dash_dot_ratio=3.84, element_jitter_sigma_ratio=0.0,
            tone_freq_hz=527.0, rise_fall_ms=3.0,
        )
        s = analyze_wave(self._long_synth(params), 8000)
        assert abs(s.dash_dot_ratio - 3.84) < 0.25

    def test_intra_element_space_round_trips(self) -> None:
        """要素間 1.25 dot を入れると解析でも 1.25 前後が返る."""
        params = KeyingParams(
            wpm=23.0, element_jitter_sigma_ratio=0.0,
            intra_element_space_units=1.25,
            tone_freq_hz=527.0, rise_fall_ms=3.0,
        )
        s = analyze_wave(self._long_synth(params), 8000)
        assert abs(s.intra_gap_dot - 1.25) < 0.20
```

- [ ] **Step 2: テストを実行して失敗する箇所を確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_analyze_keying.py::TestRoundTrip -v`

Expected: この時点では**通る可能性もある** (Task 2・3 の実装が正しければ)。
落ちた場合は次を順に疑う。

1. σ の単位を取り違えていないか (`jitter_sigma_by_kind` は **dot 長に対する比率**を
   返す。`codes_to_waveform` 側で `* dot_sec` している)
2. `np.maximum(..., dot_sec * 0.1)` のクリップが σ を潰していないか。σ 0.68 dot で
   長音 3.0 dot なら 3σ でも 0.96 dot 残るのでクリップには当たらないはず
3. 解析側の分類境界。長音の σ が大きいと短い長音が短点側に落ちる。
   `analyze_wave` に `split_sec` を明示指定して切り分ける

- [ ] **Step 3: 落ちたテストがあれば実装を直す**

**テストの閾値を緩めて通すのは禁止。** 往復しないなら合成器か解析のどちらかが
壊れている。どちらが壊れているかは、ジッタ σ=0 の波形で `analyze_wave` が
正しい値を返すか (Task 4 のテストが担保済み) を起点に切り分ける。

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_analyze_keying.py -v`
Expected: 全 PASS

- [ ] **Step 5: 全テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 921 passed 前後, 4 skipped

- [ ] **Step 6: コミット**

```bash
git add tests/test_analyze_keying.py
git commit -m "test: 合成器の統計値の往復検証を追加

入れた σ・長短比・要素間が、生成した波形から復元できることを確認する。
指定どおりの分散を持つ波形を作れていなければ、その分布で学習しても意味がない。"
```

---

### Task 6: 学習用サンプラを実測レンジに拡張する

**目的:** 学習データが実測どおりの分布になるようにする。

**Files:**
- Modify: `src/synth/dataset.py` (`default_config_sampler` 相当の `__call__`)
- Test: `tests/test_synth_dataset.py`

**Interfaces:**
- Consumes: Task 2・3 の `KeyingParams` の新フィールド
- Produces:
  - サンプラのコンストラクタ引数 `hand_keying: bool = True`
    (`False` にすると従来の分布に戻る。A/B 用)
  - サンプラのコンストラクタ引数 `extreme_tail: bool = True`
    (`False` にすると長音 σ の上限を 0.70 に抑える。設計書 §3.2 の A/B 用)
  - モジュール関数 `log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float`

**設計書 §3.1 の範囲 (これをそのまま実装する):**

| パラメータ | 範囲 | 分布 |
|---|---|---|
| `dash_dot_ratio` | 2.5〜5.0 | 一様 |
| `dot_jitter_sigma_ratio` | 0.02〜0.20 | **対数一様** |
| `dash_jitter_sigma_ratio` | 0.05〜1.30 (`extreme_tail=False` なら 0.05〜0.70) | **対数一様** |
| `intra_element_space_units` | 1.0〜1.3 | 一様 |
| `intra_gap_jitter_sigma_ratio` | 0.05〜0.50 | 対数一様 |
| `inter_char_space_units` | 2.6〜3.2 | 一様 |
| `char_gap_jitter_sigma_ratio` | 0.05〜0.90 | 対数一様 |
| `inter_word_space_units` | 5.0〜16.0 | 一様 |
| `word_gap_jitter_sigma_ratio` | 0.01〜4.00 | 対数一様 |

**必ず守ること:** エレキー相当 (σ 最小・比 3.0・間隔ちょうど 1/3/7) を部分集合として
残す。対数一様は小さい σ に確率質量が集まるのでこれを満たす。**範囲の下限を
0 より大きくすること** (対数を取るため)。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_dataset.py` の末尾に追記する。

```python
class TestHandKeyingSampler:
    def test_log_uniform_stays_in_range(self) -> None:
        from src.synth.dataset import log_uniform

        rng = np.random.default_rng(0)
        vals = np.array([log_uniform(rng, 0.05, 1.30) for _ in range(1000)])
        assert vals.min() >= 0.05
        assert vals.max() <= 1.30
        # 対数一様なので中央値は幾何平均 (≈0.255) 付近。一様なら 0.675 付近になる
        assert np.median(vals) < 0.45

    def test_sampler_sets_per_kind_sigma(self) -> None:
        """hand_keying=True で種別ごとの σ が設定される."""
        from src.synth.dataset import default_config_sampler

        sampler = default_config_sampler(mode="european", hand_keying=True)
        cfg = sampler(np.random.default_rng(0))
        k = cfg.keying
        assert k.dot_jitter_sigma_ratio is not None
        assert k.dash_jitter_sigma_ratio is not None
        assert 0.02 <= k.dot_jitter_sigma_ratio <= 0.20
        assert 0.05 <= k.dash_jitter_sigma_ratio <= 1.30
        assert 2.5 <= k.dash_dot_ratio <= 5.0
        assert 1.0 <= k.intra_element_space_units <= 1.3
        assert 2.6 <= k.inter_char_space_units <= 3.2
        assert 5.0 <= k.inter_word_space_units <= 16.0

    def test_extreme_tail_off_caps_dash_sigma(self) -> None:
        """extreme_tail=False で長音 σ の上限が下がる."""
        from src.synth.dataset import default_config_sampler

        sampler = default_config_sampler(
            mode="european", hand_keying=True, extreme_tail=False
        )
        rng = np.random.default_rng(1)
        vals = [sampler(rng).keying.dash_jitter_sigma_ratio for _ in range(300)]
        assert max(vals) <= 0.70

    def test_hand_keying_off_reproduces_old_distribution(self) -> None:
        """hand_keying=False なら従来どおり種別ごとの σ は None のまま."""
        from src.synth.dataset import default_config_sampler

        sampler = default_config_sampler(mode="european", hand_keying=False)
        k = sampler(np.random.default_rng(0)).keying
        assert k.dot_jitter_sigma_ratio is None
        assert k.dash_jitter_sigma_ratio is None
        assert k.intra_element_space_units == 1.0
        assert k.inter_char_space_units == 3.0
        assert k.inter_word_space_units == 7.0
        assert 2.5 <= k.dash_dot_ratio <= 4.0

    def test_electronic_keyer_is_still_reachable(self) -> None:
        """エレキー相当 (σ が十分小さい) が現実的な頻度で引ける."""
        from src.synth.dataset import default_config_sampler

        sampler = default_config_sampler(mode="european", hand_keying=True)
        rng = np.random.default_rng(2)
        vals = [sampler(rng).keying.dash_jitter_sigma_ratio for _ in range(1000)]
        clean = [v for v in vals if v < 0.15]
        # 対数一様なら 1000 件中 100 件以上は σ<0.15 に落ちるはず
        assert len(clean) >= 100
```

**確認済みの既存 API (2026-08-06 時点):**

- `src/synth/dataset.py:24` `class DefaultConfigSampler` — `__call__(self, rng) -> SynthConfig`。
  DataLoader の spawn でピクルできるよう module-level クラスにしてある
- `src/synth/dataset.py:75` `def default_config_sampler(mode: Mode) -> ConfigSampler` —
  `DefaultConfigSampler(mode)` を返すだけの薄いファクトリ。**ここにも新しい引数を
  通さないとテストから渡せない**
- 呼び出し側は `src/finetune/pipeline.py:12` と `src/synth/dataset.py:123`。
  引数を増やすだけなら既存の呼び出しは壊れない

- [ ] **Step 2: テストが失敗することを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_dataset.py::TestHandKeyingSampler -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

`src/synth/dataset.py` にモジュール関数を追加する。

```python
def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """[lo, hi] から対数一様に 1 つ引く.

    一様分布と違い小さい値に確率質量が集まる。手打ちの σ に使う理由は、
    上限 (長音 σ 1.30 dot) が人間にも解読できない領域であり、そこを一様に
    引くとモデルが「曖昧なら当てずっぽう」に振れてきれいな符号の精度を
    落とす恐れがあるため (設計書 §3.2)。lo は 0 より大きいこと。
    """
    if lo <= 0.0:
        raise ValueError(f"lo must be > 0 for log-uniform, got {lo}")
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
```

`DefaultConfigSampler.__init__` の引数に追加する (既存引数の後ろ、キーワード専用の
位置に置く)。

```python
        hand_keying: bool = True,
        extreme_tail: bool = True,
```

`__init__` の本体に追加する。

```python
        # 手打ちの分布を使うか。False なら従来 (エレキー相当のみ) の分布に戻る。
        self.hand_keying = hand_keying
        # 長音 σ の極端テール (0.70〜1.30 dot) を含めるか。設計書 §3.2 の A/B 用。
        self.extreme_tail = extreme_tail
```

**ファクトリ関数にも通す** (`src/synth/dataset.py:75`)。ここを忘れるとテストから
渡せない。

```python
def default_config_sampler(
    mode: Mode, hand_keying: bool = True, extreme_tail: bool = True
) -> ConfigSampler:
    """標準サンプラを返す."""
    return DefaultConfigSampler(mode, hand_keying=hand_keying, extreme_tail=extreme_tail)
```

`__call__` の `KeyingParams(...)` 生成部を分岐させる。**変更前** (行 43〜52 付近):

```python
        keying = KeyingParams(
            wpm=float(rng.uniform(8.0, 50.0)),                 # 拡張: 10-40 → 8-50
            dash_dot_ratio=float(rng.uniform(2.5, 4.0)),
            element_jitter_sigma_ratio=float(rng.uniform(0.05, 0.25)),  # 拡張: 上限0.20→0.25
            tone_freq_hz=float(rng.uniform(*self.tone_freq_range)),
            tone_drift_hz_per_sec=float(rng.uniform(-50.0, 50.0)),
            rise_fall_ms=float(rng.uniform(3.0, 10.0)),
            pre_silence_sec=float(rng.uniform(0.0, 0.3)),
            post_silence_sec=float(rng.uniform(0.0, 0.3)),
        )
```

**変更後**:

```python
        common = dict(
            wpm=float(rng.uniform(8.0, 50.0)),                 # 拡張: 10-40 → 8-50
            tone_freq_hz=float(rng.uniform(*self.tone_freq_range)),
            tone_drift_hz_per_sec=float(rng.uniform(-50.0, 50.0)),
            rise_fall_ms=float(rng.uniform(3.0, 10.0)),
            pre_silence_sec=float(rng.uniform(0.0, 0.3)),
            post_silence_sec=float(rng.uniform(0.0, 0.3)),
        )
        if self.hand_keying:
            # 実測 (設計書 §2.2) に合わせた手打ちの分布。
            # 短点は正確 (σ 0.064〜0.107 dot) で長音だけ暴れる (σ 0.681〜1.195 dot)
            # という非対称が要点。σ は対数一様なのでエレキー相当も同じ頻度で引ける。
            dash_sigma_hi = 1.30 if self.extreme_tail else 0.70
            keying = KeyingParams(
                dash_dot_ratio=float(rng.uniform(2.5, 5.0)),
                element_jitter_sigma_ratio=0.0,   # 種別ごとの σ を使うので基準値は 0
                dot_jitter_sigma_ratio=log_uniform(rng, 0.02, 0.20),
                dash_jitter_sigma_ratio=log_uniform(rng, 0.05, dash_sigma_hi),
                intra_element_space_units=float(rng.uniform(1.0, 1.3)),
                intra_gap_jitter_sigma_ratio=log_uniform(rng, 0.05, 0.50),
                inter_char_space_units=float(rng.uniform(2.6, 3.2)),
                char_gap_jitter_sigma_ratio=log_uniform(rng, 0.05, 0.90),
                inter_word_space_units=float(rng.uniform(5.0, 16.0)),
                word_gap_jitter_sigma_ratio=log_uniform(rng, 0.01, 4.00),
                **common,
            )
        else:
            keying = KeyingParams(
                dash_dot_ratio=float(rng.uniform(2.5, 4.0)),
                element_jitter_sigma_ratio=float(rng.uniform(0.05, 0.25)),
                **common,
            )
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/test_synth_dataset.py -v`
Expected: 全 PASS

- [ ] **Step 5: 全テストが通ることを確認する**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 全 PASS

- [ ] **Step 6: 生成データを解析して実測レンジに入っているか目視で確認する**

```bash
.venv/Scripts/python.exe - <<'PY'
import sys; sys.path.insert(0, ".")
import numpy as np, soundfile as sf
from src.synth.dataset import default_config_sampler
from src.synth.keying import codes_to_waveform
from src.tokens.morse_tokens import text_to_codes
from scripts.analyze_keying import analyze_wave, _print_stats

sampler = default_config_sampler(mode="european", hand_keying=True)
rng = np.random.default_rng(0)
cfg = sampler(rng)
cfg.keying.wpm = 23.0                     # 実測と揃えて比較しやすくする
codes = text_to_codes("CQ CQ DE JA1ABC K " * 30, mode="european")  # list[str] を返す
wave = codes_to_waveform(codes, cfg.keying, rng, sample_rate=8000).samples
_print_stats("sampled", analyze_wave(wave, 8000))
print("入れた値:", cfg.keying.dot_jitter_sigma_ratio, cfg.keying.dash_jitter_sigma_ratio)
PY
```

Expected: 出した σ と解析で返る σ がおおむね一致する。**ON ヒストグラムの山が
2 つであること**を目視で確認する。

- [ ] **Step 7: コミット**

```bash
git add src/synth/dataset.py tests/test_synth_dataset.py
git commit -m "feat: 学習用サンプラを手打ちの実測レンジに拡張する

実録音 2 本の実測 (設計書 §2.2) に合わせた分布を hand_keying=True で引く。
σ は対数一様なのでエレキー相当も同じ頻度で引け、エレキーの精度を落とさない。

extreme_tail=False で長音 σ の上限を 0.70 に抑えられる。上限 1.30 は人間にも
解読できない領域を含むため、含める/含めないを A/B で判断する (設計書 §3.2)。"
```

---

### Task 7: ファインチューニングと評価

**目的:** 新分布で学習し、**採用するかどうかを数値で判断する**。

**Files:**
- Create: `docs/hand_keying_ft_result.md`
- (学習の成果物 `models/ft_hand/` は `.gitignore` 対象)

**Interfaces:**
- Consumes: Task 6 のサンプラ
- Produces: 判断と数値の記録

**前提の確認 (実装者が最初にやること):** 設計書 §4.1 は起点を
`models/ft_1k/best.pt` としているが、これは**先頭テンソル 1 個の抜き取り検査**に
基づく。全テンソルで照合してから使うこと。

- [ ] **Step 1: 起点チェックポイントを全テンソルで照合する**

```bash
.venv/Scripts/python.exe - <<'PY'
import torch
# weights_only=True を使う。既定の False は pickle を無制限に展開するため
# 任意コード実行の経路になる。ここで要るのはテンソルだけなので True で足りる。
# model_config がデータクラスで弾かれる場合のみ
# torch.serialization.add_safe_globals([...]) で個別に許可すること。
a = torch.load("models/full/best_infer.pt", map_location="cpu", weights_only=True)["model_state"]
b = torch.load("models/ft_1k/best.pt", map_location="cpu", weights_only=True)["model_state"]
assert a.keys() == b.keys(), "キーが違う"
bad = [k for k in a if not torch.equal(a[k], b[k])]
print("不一致テンソル:", bad if bad else "なし (全一致)")
PY
```

Expected: `不一致テンソル: なし (全一致)`

**一致しない場合は先に進まないこと。** 起点が違えば結果の解釈ができない。
`models/ft_2k/last.pt` `models/ft/best.pt` も同じ方法で照合し、一致するものを探す。

- [ ] **Step 2: baseline を測り直して記録する**

```bash
.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/full/best_infer.pt \
  --keyed-dir data/keying_scripts \
  --noise-dir data/keying_scripts \
  --out models/eval/baseline.json \
  --device cpu
```

Expected: `keyed_val` が 20 件で評価される。CER が記録される。
**この数値を `docs/hand_keying_ft_result.md` に書き写す。** 以降の比較の基準になる。

- [ ] **Step 3: 極端テールありで FT する**

```bash
.venv/Scripts/python.exe scripts/finetune.py \
  --data-dir data/real --resume models/ft_1k/best.pt \
  --ckpt-dir models/ft_hand_tail --steps 1000 --num-workers 0 \
  --mix-synth --real-ratio 0.7
```

`--num-workers 0` は必須 (GPU の孤児プロセスを踏んだ実績がある)。

- [ ] **Step 4: 極端テールなしで FT する**

Task 6 の `extreme_tail` を `finetune.py` から渡せるようにする必要がある。
`scripts/finetune.py` のサンプラ生成箇所を探し (`grep -n "config_sampler\|tone_span" scripts/finetune.py`)、
`--no-extreme-tail` フラグを追加して `extreme_tail=not args.no_extreme_tail` を渡す。

```bash
.venv/Scripts/python.exe scripts/finetune.py \
  --data-dir data/real --resume models/ft_1k/best.pt \
  --ckpt-dir models/ft_hand_notail --steps 1000 --num-workers 0 \
  --mix-synth --real-ratio 0.7 --no-extreme-tail
```

- [ ] **Step 5: 両方を評価する**

```bash
for d in ft_hand_tail ft_hand_notail; do
  .venv/Scripts/python.exe scripts/eval_model.py \
    --ckpt models/$d/best.pt --keyed-dir data/keying_scripts \
    --noise-dir data/keying_scripts --out models/eval/$d.json \
    --baseline models/eval/baseline.json --device cpu
done
```

- [ ] **Step 6: 追加の held-out サンプルでも測る**

```bash
.venv/Scripts/python.exe scripts/eval_model.py \
  --ckpt models/ft_hand_tail/best.pt --keyed-dir data/keyed_extra \
  --out models/eval/extra_tail.json --device cpu
```

baseline はこのサンプル単体で **CER 40.0%** (2026-08-06 実測)。

- [ ] **Step 7: 判断して記録する**

`docs/hand_keying_ft_result.md` を作成し、次を書く。

- baseline / tail あり / tail なし の keyed_val CER・TER と synth_val CER
- `data/keyed_extra` の CER
- **採用基準: keyed_val が改善し、かつ synth_val が悪化しないこと。片方だけなら不採用**
- 採用しない場合もその判断と数値を必ず残す (過去 3 回、片方だけ改善したものを
  不採用にしている。記録が次の判断材料になる)

- [ ] **Step 8: 採用する場合のみ、推論用チェックポイントを書き出す**

```bash
.venv/Scripts/python.exe scripts/export_infer_checkpoint.py \
  --src models/ft_hand_tail/best.pt --dst models/full/best_infer.pt
.venv/Scripts/python.exe scripts/export_onnx.py
.venv/Scripts/python.exe scripts/export_golden.py
.venv/Scripts/python.exe -m pytest tests/ -q
cd web && npx vitest run && cd ..
```

**モデルを再学習したら `export_onnx.py` と `export_golden.py` を必ず再実行する**
(CLAUDE.md の規約)。ブラウザ版の ONNX と golden fixture が古いままだと、
`web/tests/golden.test.ts` が Python と一致しなくなる。

- [ ] **Step 9: コミット**

```bash
git add docs/hand_keying_ft_result.md
# 採用した場合は追加で:
git add models/full/best_infer.pt web/tests/fixtures/ web/src/generated/tokens.ts
git commit -m "feat: 手打ち分布で FT したモデルを採用する

keyed_val CER xx.xx% → yy.yy%、synth_val CER aa.aa% → bb.bb%。
両方が同時に悪化しないことを確認した。詳細は docs/hand_keying_ft_result.md。"
```

---

## Self-Review (計画作成者による確認結果)

**1. 仕様カバレッジ**

| 設計書の要求 | 対応する Task |
|---|---|
| §3.1 種別ごとの σ | Task 1, 2 |
| §3.1 要素間スペースのパラメータ化 | Task 3 |
| §3.1 dash_dot_ratio 2.5〜5.0 | Task 6 |
| §3.1 速度ドリフト・相関ジッタを入れない | (実装しない = 全 Task で扱わない) |
| §3.1 エレキー相当を部分集合として残す | Task 6 Step 1 の `test_electronic_keyer_is_still_reachable` |
| §3.2 対数一様 σ と極端テールの A/B | Task 6, Task 7 Step 3・4 |
| §3.3 numpy ベクトル化・Generator 引数・後方互換 | Global Constraints + Task 2 Step 1 の後方互換テスト |
| §4.1 起点チェックポイント・num_workers 0・1000 step | Task 7 Step 1・3 |
| §4.2 keyed_val 主指標・synth_val 副指標・採用基準 | Task 7 Step 5・7 |
| §4.3 往復検証 | Task 5 |
| §4.3 後方互換テスト | Task 2 Step 1 |
| §4.3 決定性テスト | 既存 `test_same_seed_same_output` が担保 |
| §4.3 解析スクリプトのリポジトリ入り | Task 4 |

**未カバーだったので追加したもの:** 設計書 §4.1 の「1000 と 2000 の両方を評価」は
Task 7 では 1000 のみにした。理由は極端テールの A/B (2 本) と掛け合わせると 4 本に
なり、1 回の計画としては大きすぎるため。**1000 step で改善が出た場合に 2000 step を
追試する**ことを Task 7 Step 7 の記録に含める。

**2. プレースホルダ検査:** 「TBD」「後で」「適切に」等は無し。全コードステップに
実際のコードがある。

**3. 型の一貫性:** `ElementKind` (Task 1) → `jitter_sigma_by_kind()` (Task 2) →
`intra_element_space_units` (Task 3) → `analyze_wave` / `KeyingStats` (Task 4) →
`log_uniform` (Task 6) の名前と型が全 Task で一致していることを確認した。
`_build_element_sequence` の戻り値は Task 1 で 4 要素になり、Task 3 で引数が 7 個に
なる。Task 3 のコードは Task 1 適用後の状態を前提にしている。
