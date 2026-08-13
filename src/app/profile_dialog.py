"""経歴の編集画面.

**運用者が経歴を書く手段がこれまで無かった。** ``load_profile()`` は送信ダイアログが
読んでいたが ``save_profile()`` は本番のどこからも呼ばれておらず、``operator.json`` を
手で書くしかなかった。経歴が空だと型の ``{自局コール}`` ``{名前}`` が埋まらないので、
**型が 1 つも実用にならない**。

欧文用と和文用
--------------
``名前`` ``QTH`` ``リグ`` ``アンテナ`` ``出力`` は**独立した 2 つの値**を持つ
(``src/tx/profile.py``)。``「FT991」`` は ``FT991`` の読みではない。
``自局コール`` だけは 1 つ — 和文の交信でも欧文で送るためである。

保存の前に見せる
----------------
**書いた値が本当に送れるかを、保存の前に確かめる** (設計書 §4.2)。
判定は ``encoder.find_unsendable`` を使い、符号表を書き写さない (原則 2)。

**実際に通る道と同じ道を通すこと。** 生の値だけを見ると誤判定する:

    ``ニジュウド`` を生で判定       -> NG (小書きの ュ が符号表に無い)
    実際の経路 (to_sendable_kana)  -> ニジユウド に倒れて送れる

モードを固定するために ``{HORE}`` / ``{RATA}`` を頭に付けて調べる。
値そのものからモードを推測させると、**欧文用の欄にカタカナを書いた場合に
「和文だから送れる」と判定してしまう** — それはまさに見つけたい誤りである。

**警告しても保存は妨げない。** 書きかけを保存できないと育てるのが苦痛になる。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.tx.encoder import HORE, RATA, find_unsendable
from src.tx.profile import (
    BILINGUAL_FIELDS,
    BilingualField,
    OperatorProfile,
    load_profile,
    save_profile,
)
from src.tx.reading import to_sendable_kana

# 画面に出す欄の名前。**型の ``{…}`` と同じ言葉にする** (どれがどこに入るか
# 分かるように)。並びもこの順。
FIELD_LABELS: dict[str, str] = {
    "name": "名前",
    "qth": "QTH",
    "rig": "リグ",
    "antenna": "アンテナ",
    "power": "出力",
}

_SAVED_MESSAGE = "保存しました"


def unsendable_in_value(value: str, mode: str) -> str:
    """その値をそのモードで送れるか調べ、送れない文字を並べて返す.

    **実際に通る道と同じ道を通す** — ``to_sendable_kana`` を経てから
    ``find_unsendable`` にかける。生の値を見ると小書きカナ (``ュ``) を
    「送れない」と誤判定する。

    モードは ``{HORE}`` / ``{RATA}`` を頭に付けて固定する。値の中身から
    推測させると、**欧文用の欄に書いたカタカナが「和文だから送れる」と
    判定されてしまう**。
    """
    if not value:
        return ""
    converted = to_sendable_kana(value).text
    marker = HORE if mode == "japanese" else RATA
    return "".join(bad.char for bad in find_unsendable(marker + converted))


class ProfileDialog(QDialog):
    """経歴を編集する画面."""

    def __init__(
        self,
        profile: OperatorProfile | None = None,
        path: Path | str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("経歴")
        # **保存先は経歴と同じ場所** (``~/.cw-decorder/operator.json``)。
        # テストは必ず一時パスを渡すこと (利用者の実ファイルを壊さない)。
        self._path = Path(path) if path is not None else None
        self._profile = profile if profile is not None else self._load()
        self._build_ui()
        self._fill_from(self._profile)

    def _load(self) -> OperatorProfile:
        return load_profile(self._path) if self._path else load_profile()

    # ---- 画面 ----
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        # **コールサインは 1 つだけ。** 和文の交信でも欧文で送る
        grid.addWidget(QLabel("自局コール"), 0, 0)
        self.callsign_edit = QLineEdit()
        self.callsign_edit.setPlaceholderText("JH0ILL")
        grid.addWidget(self.callsign_edit, 0, 1, 1, 2)

        grid.addWidget(QLabel("欧文用"), 1, 1)
        grid.addWidget(QLabel("和文用"), 1, 2)

        # 欄ごとの (欧文用, 和文用)。**テストと保存の両方がここを見る**
        self.field_edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        placeholders = {
            "name": ("TARO", "タロウ"),
            "qth": ("YOKOHAMA", "ヨコハマシ"),
            "rig": ("FT991", "「FT991」"),
            "antenna": ("DP", "「DP」"),
            "power": ("50W", "「50W」"),
        }
        for row, attr in enumerate(BILINGUAL_FIELDS, start=2):
            grid.addWidget(QLabel(FIELD_LABELS[attr]), row, 0)
            european = QLineEdit()
            japanese = QLineEdit()
            european.setPlaceholderText(placeholders[attr][0])
            japanese.setPlaceholderText(placeholders[attr][1])
            grid.addWidget(european, row, 1)
            grid.addWidget(japanese, row, 2)
            self.field_edits[attr] = (european, japanese)
        layout.addLayout(grid)

        layout.addWidget(QLabel("自由記述 (AI の返信案に渡します)"))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setFixedHeight(70)
        layout.addWidget(self.notes_edit)

        layout.addWidget(QLabel("読み辞書 (機械変換が外す固有名詞を入れておきます)"))
        dict_row = QHBoxLayout()
        self.dict_table = QTableWidget(0, 2)
        self.dict_table.setHorizontalHeaderLabels(["語", "読み (カタカナ)"])
        self.dict_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.dict_table.setFixedHeight(140)
        dict_row.addWidget(self.dict_table, 1)
        dict_buttons = QVBoxLayout()
        self.add_row_btn = QPushButton("追加")
        self.add_row_btn.clicked.connect(lambda: self.add_dictionary_row())
        self.remove_row_btn = QPushButton("削除")
        self.remove_row_btn.clicked.connect(lambda: self.remove_dictionary_row())
        dict_buttons.addWidget(self.add_row_btn)
        dict_buttons.addWidget(self.remove_row_btn)
        dict_buttons.addStretch(1)
        dict_row.addLayout(dict_buttons)
        layout.addLayout(dict_row)

        # **警告はここに出す。保存は妨げない** (設計書 §4.2)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(lambda: self.save())
        self.close_btn = QPushButton("閉じる")
        self.close_btn.clicked.connect(self.accept)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    def _fill_from(self, profile: OperatorProfile) -> None:
        self.callsign_edit.setText(profile.callsign)
        for attr, (european, japanese) in self.field_edits.items():
            value: BilingualField = getattr(profile, attr)
            european.setText(value.european)
            japanese.setText(value.japanese)
        self.notes_edit.setPlainText(profile.notes)
        self.dict_table.setRowCount(0)
        for word, reading in profile.reading_dictionary.items():
            self.add_dictionary_row(word, reading)

    # ---- 読み辞書 ----
    def add_dictionary_row(self, word: str = "", reading: str = "") -> None:
        row = self.dict_table.rowCount()
        self.dict_table.insertRow(row)
        self.dict_table.setItem(row, 0, QTableWidgetItem(word))
        self.dict_table.setItem(row, 1, QTableWidgetItem(reading))

    def remove_dictionary_row(self) -> None:
        """選んでいる行を消す. 選んでいなければ最後の行を消す."""
        row = self.dict_table.currentRow()
        if row < 0:
            row = self.dict_table.rowCount() - 1
        if row >= 0:
            self.dict_table.removeRow(row)

    def dictionary(self) -> dict[str, str]:
        """表から読み辞書を作る. **語が空の行は捨てる** (設計書 §6)."""
        out: dict[str, str] = {}
        for row in range(self.dict_table.rowCount()):
            word_item = self.dict_table.item(row, 0)
            reading_item = self.dict_table.item(row, 1)
            word = word_item.text().strip() if word_item else ""
            reading = reading_item.text().strip() if reading_item else ""
            if word:
                out[word] = reading
        return out

    # ---- 経歴 ----
    def profile(self) -> OperatorProfile:
        """画面の内容から経歴を作る."""
        built = OperatorProfile(
            callsign=self.callsign_edit.text().strip(),
            notes=self.notes_edit.toPlainText(),
            reading_dictionary=self.dictionary(),
        )
        for attr, (european, japanese) in self.field_edits.items():
            setattr(
                built,
                attr,
                BilingualField(
                    european=european.text().strip(),
                    japanese=japanese.text().strip(),
                ),
            )
        return built

    def warnings(self) -> list[str]:
        """送れない値と、読み辞書の欧文の語を並べて返す. 無ければ空.

        **止めるためではなく、見せるためのもの。**
        """
        found: list[str] = []
        profile = self.profile()

        # 自局コールは欧文で送る
        bad = unsendable_in_value(profile.callsign, "european")
        if bad:
            found.append(f"欧文で送れません: 自局コール の {bad}")

        for attr in BILINGUAL_FIELDS:
            value: BilingualField = getattr(profile, attr)
            label = FIELD_LABELS[attr]
            bad = unsendable_in_value(value.european, "european")
            if bad:
                found.append(f"欧文で送れません: {label} の {bad}")
            bad = unsendable_in_value(value.japanese, "japanese")
            if bad:
                found.append(f"和文で送れません: {label} の {bad}")

        # **読み辞書は本文全体に当たる** (経歴の欄と違って避けようがない)。
        # 欧文の語を入れると、欧文の交信でその語が読みに置き換わって壊れる。
        # 入力は妨げない — 運用者の自由を奪わない (設計書 §4.3)。
        european_words = [
            word for word in profile.reading_dictionary
            if not unsendable_in_value(word, "european")
        ]
        if european_words:
            found.append(
                "読み辞書に欧文の語があります (" + "、".join(european_words) + ")。"
                "欧文の交信でこの語を打つと送れなくなります"
            )
        return found

    def save(self) -> None:
        """保存する. **警告があっても保存する** (書きかけを守る)."""
        profile = self.profile()
        try:
            if self._path is not None:
                save_profile(profile, self._path)
            else:
                save_profile(profile)
        except OSError as exc:
            # **保存できなかったことを黙らせない。** 書いたつもりで消える
            self.status_label.setText(f"保存できませんでした: {exc}")
            return
        self._profile = profile
        messages = self.warnings()
        self.status_label.setText(
            "\n".join([_SAVED_MESSAGE, *messages]) if messages else _SAVED_MESSAGE
        )


__all__ = ["FIELD_LABELS", "ProfileDialog", "unsendable_in_value"]
