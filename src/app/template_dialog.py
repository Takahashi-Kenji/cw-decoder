"""返信の型の編集画面.

**これまで型は JSON を手で書くしかなかった。** 経歴と同じ状態である
(``save_templates()`` は本番のどこからも呼ばれていなかった)。

並びに意味がある
----------------
**一覧の並びがそのまま送信ダイアログの並びになる。** よく使う型を上に置ける
ようにするため、``[上へ]`` ``[下へ]`` を用意する。並べ替えは保存の対象である。

保存の前に見せる
----------------
経歴と同じ流儀 (``src/app/profile_dialog.py``)。**警告しても保存は妨げない** —
書きかけを保存できないと育てるのが苦痛になる。

見るのは 2 つ:

* **送れない文字** — 一番よく起きるのは ``{HORE}`` の中に欧文を書くこと
  (``{HORE}コンニチハ RST 599{RATA}``)。RST は欧文で送るものなので和文モードの
  中では送れない。**交信中に気づくのでは遅い。**
* **知らない欄** — ``{名前まえ}`` のような書き間違いは、差し込みで置き換わらず
  そのまま ``{`` ``}`` が残り「送信できない文字」として出る。名指しするほうが早い。

**検証には経歴を渡す。** 渡さないと、経歴の値でだけ起きる失敗を原理的に
検出できない (``unsendable_in_template`` の docstring)。

下書きを見せる
--------------
選んでいる型に**いまの経歴を差し込んだ結果**を出す。型の本文だけを見ても
実際に何が電波に出るかは分からない (``{リグ}`` に何が入るのかは経歴次第)。
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from src.tx.encoder import needs_japanese_wrap, wrap_japanese
from src.tx.profile import OperatorProfile, load_profile
from src.tx.reading import to_sendable_kana
from src.tx.templates import (
    PLACEHOLDERS,
    ReplyTemplate,
    fill,
    load_templates,
    profile_values,
    save_templates,
    unsendable_in_template,
)

# モードの選択肢。**値は保存される文字列そのもの** (画面の文言と分ける)。
MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("european", "欧文"),
    ("japanese", "和文"),
    ("any", "どちらでも"),
)

# 型の本文にある ``{…}`` を拾う。``templates._FIELD_RE`` と同じ形。
# **ここで作り直しているのは「知らない欄」を名指しするためだけ**であり、
# 差し込みの判定は ``fill`` が行う (二重定義ではなく、用途が違う)。
_FIELD_RE = re.compile(r"\{([^{}]*)\}")

# 差し込みでも符号でもない、そのまま送るマーカー。
_MARKERS: frozenset[str] = frozenset({"HORE", "RATA", "SK", "SN", "AR", "BT", "KN"})

_SAVED_MESSAGE = "保存しました"
_NEW_NAME = "新しい型"


def unknown_placeholders(text: str) -> tuple[str, ...]:
    """``PLACEHOLDERS`` にもマーカーにも無い ``{…}`` を出てきた順に返す.

    書き間違い (``{名前まえ}``) は差し込みで置き換わらず、``{`` ``}`` が
    「送信できない文字」として出るだけで**どこが悪いのか分からない**。
    """
    found: list[str] = []
    for match in _FIELD_RE.finditer(text):
        name = match.group(1)
        if name in PLACEHOLDERS or name in _MARKERS or name in found:
            continue
        found.append(name)
    return tuple(found)


class TemplateDialog(QDialog):
    """返信の型を編集する画面."""

    def __init__(
        self,
        templates: list[ReplyTemplate] | None = None,
        path: Path | str | None = None,
        profile: OperatorProfile | None = None,
        profile_path: Path | str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("返信の型")
        # **テストは必ず一時パスを渡すこと** (利用者の実ファイルを壊さない)。
        self._path = Path(path) if path is not None else None
        self._profile_path = Path(profile_path) if profile_path is not None else None
        self._profile = profile if profile is not None else self._load_profile()
        self._templates: list[ReplyTemplate] = (
            list(templates) if templates is not None else self._load()
        )
        # 選び直しのたびに編集中の内容を書き戻すので、いまどれを見ているかを持つ
        self._current = -1
        self._build_ui()
        self._refresh_list()

    def _load(self) -> list[ReplyTemplate]:
        return load_templates(self._path) if self._path else load_templates()

    def _load_profile(self) -> OperatorProfile:
        return load_profile(self._profile_path) if self._profile_path else load_profile()

    # ---- 画面 ----
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        body = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel("型 (上から順に一覧へ出ます)"))
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_row_changed)
        left.addWidget(self.list_widget, 1)
        row_buttons = QHBoxLayout()
        self.add_btn = QPushButton("追加")
        self.add_btn.clicked.connect(lambda: self.add_template())
        self.copy_btn = QPushButton("複製")
        self.copy_btn.clicked.connect(lambda: self.copy_template())
        self.delete_btn = QPushButton("削除")
        self.delete_btn.clicked.connect(lambda: self.delete_template())
        for button in (self.add_btn, self.copy_btn, self.delete_btn):
            row_buttons.addWidget(button)
        left.addLayout(row_buttons)
        move_buttons = QHBoxLayout()
        self.up_btn = QPushButton("上へ")
        self.up_btn.clicked.connect(lambda: self.move_up())
        self.down_btn = QPushButton("下へ")
        self.down_btn.clicked.connect(lambda: self.move_down())
        move_buttons.addWidget(self.up_btn)
        move_buttons.addWidget(self.down_btn)
        left.addLayout(move_buttons)
        body.addLayout(left, 1)

        right = QVBoxLayout()
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("名前:"))
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_edited)
        name_row.addWidget(self.name_edit, 1)
        name_row.addWidget(QLabel("モード:"))
        self.mode_combo = QComboBox()
        for _value, label in MODE_CHOICES:
            self.mode_combo.addItem(label)
        self.mode_combo.currentIndexChanged.connect(self.on_edited)
        name_row.addWidget(self.mode_combo)
        right.addLayout(name_row)

        right.addWidget(QLabel("本文:"))
        self.text_edit = QPlainTextEdit()
        self.text_edit.textChanged.connect(self.on_edited)
        right.addWidget(self.text_edit, 1)

        placeholders = QLabel(
            "使える欄: " + "  ".join(f"{{{name}}}" for name in sorted(PLACEHOLDERS))
        )
        placeholders.setWordWrap(True)
        placeholders.setStyleSheet("color: #666;")
        right.addWidget(placeholders)

        right.addWidget(QLabel("いまの経歴を差し込んだ下書き:"))
        self.preview_view = QPlainTextEdit()
        self.preview_view.setReadOnly(True)
        self.preview_view.setFixedHeight(70)
        right.addWidget(self.preview_view)
        body.addLayout(right, 2)

        layout.addLayout(body)

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

    # ---- 一覧 ----
    def _refresh_list(self, select: int = 0) -> None:
        """一覧を作り直して ``select`` 行を選ぶ.

        作り直しのあいだは ``currentRowChanged`` を止める。止めないと、
        ``clear()`` が出す「-1 行が選ばれた」で編集中の内容が
        **消えた欄に書き戻される**。
        """
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for template in self._templates:
            label = dict(MODE_CHOICES).get(template.mode, template.mode)
            self.list_widget.addItem(f"{template.name}  ({label})")
        self.list_widget.blockSignals(False)
        if self._templates:
            row = max(0, min(select, len(self._templates) - 1))
            self._current = -1              # 書き戻しを起こさずに読み込ませる
            self.list_widget.setCurrentRow(row)
        else:
            self._current = -1
            self._show(None)
        self._update_buttons()

    def on_row_changed(self, row: int) -> None:
        """選び直し。**編集中の内容を先に書き戻してから**次を読み込む."""
        self._store_current()
        self._current = row
        self._show(self._templates[row] if 0 <= row < len(self._templates) else None)
        self._update_buttons()

    def _show(self, template: ReplyTemplate | None) -> None:
        editing = template is not None
        for widget in (self.name_edit, self.mode_combo, self.text_edit):
            widget.blockSignals(True)
        self.name_edit.setText(template.name if template else "")
        self.text_edit.setPlainText(template.text if template else "")
        values = [value for value, _label in MODE_CHOICES]
        self.mode_combo.setCurrentIndex(
            values.index(template.mode) if template and template.mode in values else 0
        )
        for widget in (self.name_edit, self.mode_combo, self.text_edit):
            widget.blockSignals(False)
        for widget in (self.name_edit, self.mode_combo, self.text_edit):
            widget.setEnabled(editing)
        self._refresh_preview()

    def _store_current(self) -> None:
        """編集中の欄を、いま選んでいる型へ書き戻す."""
        if not (0 <= self._current < len(self._templates)):
            return
        self._templates[self._current] = self.edited_template()

    def edited_template(self) -> ReplyTemplate:
        """編集中の欄から型を作る."""
        return ReplyTemplate(
            name=self.name_edit.text().strip() or _NEW_NAME,
            mode=MODE_CHOICES[self.mode_combo.currentIndex()][0],
            text=self.text_edit.toPlainText(),
        )

    def on_edited(self) -> None:
        """欄が変わったら、一覧の見出しと下書きを追いつかせる."""
        if not (0 <= self._current < len(self._templates)):
            return
        self._templates[self._current] = self.edited_template()
        template = self._templates[self._current]
        label = dict(MODE_CHOICES).get(template.mode, template.mode)
        item = self.list_widget.item(self._current)
        if item is not None:
            item.setText(f"{template.name}  ({label})")
        self._refresh_preview()

    def _update_buttons(self) -> None:
        has = bool(self._templates)
        row = self._current
        self.copy_btn.setEnabled(has)
        self.delete_btn.setEnabled(has)
        self.up_btn.setEnabled(has and row > 0)
        self.down_btn.setEnabled(has and 0 <= row < len(self._templates) - 1)

    # ---- 増やす・減らす・並べ替え ----
    def add_template(self) -> None:
        self._store_current()
        self._templates.append(ReplyTemplate(name=_NEW_NAME, mode="any", text=""))
        self._refresh_list(len(self._templates) - 1)

    def copy_template(self) -> None:
        """選んでいる型を複製する. **似た型を作るのが一番よくある編集である。**"""
        self._store_current()
        if not (0 <= self._current < len(self._templates)):
            return
        source = self._templates[self._current]
        self._templates.insert(
            self._current + 1,
            ReplyTemplate(name=f"{source.name} の複製", mode=source.mode, text=source.text),
        )
        self._refresh_list(self._current + 1)

    def delete_template(self) -> None:
        if not (0 <= self._current < len(self._templates)):
            return
        row = self._current
        self._current = -1                  # 消す前に書き戻しを止める
        del self._templates[row]
        self._refresh_list(row)

    def move_up(self) -> None:
        self._store_current()
        row = self._current
        if row <= 0:
            return
        self._templates[row - 1], self._templates[row] = (
            self._templates[row], self._templates[row - 1],
        )
        self._refresh_list(row - 1)

    def move_down(self) -> None:
        self._store_current()
        row = self._current
        if not (0 <= row < len(self._templates) - 1):
            return
        self._templates[row + 1], self._templates[row] = (
            self._templates[row], self._templates[row + 1],
        )
        self._refresh_list(row + 1)

    # ---- 下書きと警告 ----
    def _refresh_preview(self) -> None:
        """いまの経歴を差し込んだ結果を出す. **実際に通る道を通す.**"""
        if not (0 <= self._current < len(self._templates)):
            self.preview_view.setPlainText("")
            return
        template = self.edited_template()
        values = profile_values(self._profile, template.mode)
        converted = to_sendable_kana(fill(template.text, values), self._profile).text
        wire = wrap_japanese(converted) if needs_japanese_wrap(converted) else converted
        self.preview_view.setPlainText(wire)

    def warnings(self) -> list[str]:
        """送れない文字と、知らない欄を並べて返す. 無ければ空."""
        self._store_current()
        found: list[str] = []
        for template in self._templates:
            unknown = unknown_placeholders(template.text)
            if unknown:
                found.append(
                    f"知らない欄があります: {template.name} の "
                    + "、".join(f"{{{name}}}" for name in unknown)
                )
            # **経歴を渡す。** 渡さないと経歴の値でだけ起きる失敗を検出できない
            bad = unsendable_in_template(template, self._profile)
            if bad:
                found.append(f"送信できない文字があります: {template.name} の {bad}")
        return found

    def save(self) -> None:
        """保存する. **警告があっても保存する** (書きかけを守る)."""
        self._store_current()
        try:
            if self._path is not None:
                save_templates(self._templates, self._path)
            else:
                save_templates(self._templates)
        except OSError as exc:
            # **保存できなかったことを黙らせない。** 書いたつもりで消える
            self.status_label.setText(f"保存できませんでした: {exc}")
            return
        messages = self.warnings()
        self.status_label.setText(
            "\n".join([_SAVED_MESSAGE, *messages]) if messages else _SAVED_MESSAGE
        )

    def templates(self) -> list[ReplyTemplate]:
        """編集中の内容を含む型の一覧."""
        self._store_current()
        return list(self._templates)


__all__ = ["MODE_CHOICES", "TemplateDialog", "unknown_placeholders"]
