from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QComboBox, QMessageBox, QInputDialog, QHeaderView,
    QAbstractItemView, QColorDialog
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

_LINESTYLES = ["-", "--", "-.", ":"]


class CaseManagerDialog(QDialog):
    """
    Non-modal dialog for editing per-case style (color, linestyle) and
    renaming/removing cases.

    ``cases`` is a list of dicts, each with at minimum the keys:
        name, color, linestyle, source
    The dialog mutates the list in-place and emits ``cases_changed``
    after every mutation.
    """

    cases_changed = pyqtSignal()

    def __init__(self, parent=None, cases=None):
        super().__init__(parent)
        self._cases = cases if cases is not None else []
        self.setWindowTitle("Case Manager")
        self.setMinimumWidth(680)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Case Name", "Source File", "Color", "Linestyle"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(2, 80)
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, 100)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_rename = QPushButton("Rename")
        self._btn_rename.clicked.connect(self._on_rename)
        self._btn_remove = QPushButton("Remove")
        self._btn_remove.clicked.connect(self._on_remove)
        btn_row.addWidget(self._btn_rename)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch()
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

    def refresh(self):
        self._table.setRowCount(0)
        for row, case in enumerate(self._cases):
            self._table.insertRow(row)

            name_item = QTableWidgetItem(case["name"])
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            src_item = QTableWidgetItem(case.get("source", ""))
            src_item.setFlags(src_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, src_item)

            color_btn = QPushButton()
            color_btn.setFixedHeight(22)
            color_btn.setStyleSheet(
                f"background-color: {case['color']}; border: 1px solid #555;")
            color_btn.setProperty("row", row)
            color_btn.clicked.connect(self._on_color_clicked)
            self._table.setCellWidget(row, 2, color_btn)

            ls_combo = QComboBox()
            ls_combo.addItems(_LINESTYLES)
            ls_combo.setCurrentText(case["linestyle"])
            ls_combo.setProperty("row", row)
            ls_combo.currentTextChanged.connect(self._on_linestyle_changed)
            self._table.setCellWidget(row, 3, ls_combo)

    def _selected_row(self):
        return self._table.currentRow()

    def _on_color_clicked(self):
        row = self.sender().property("row")
        if not (0 <= row < len(self._cases)):
            return
        case = self._cases[row]
        new_color = QColorDialog.getColor(
            QColor(case["color"]), self, f"Color for '{case['name']}'")
        if not new_color.isValid():
            return
        case["color"] = new_color.name()
        self.refresh()
        self.cases_changed.emit()

    def _on_linestyle_changed(self, linestyle):
        row = self.sender().property("row")
        if not (0 <= row < len(self._cases)):
            return
        self._cases[row]["linestyle"] = linestyle
        self.cases_changed.emit()

    def _on_rename(self):
        row = self._selected_row()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a case to rename.")
            return
        case = self._cases[row]
        name = case["name"]
        new_name, ok = QInputDialog.getText(
            self, "Rename Case", f"New name for '{name}':", text=name)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == name:
            return
        if any(c["name"] == new_name for c in self._cases):
            QMessageBox.warning(self, "Name Taken",
                f"A case named '{new_name}' already exists.")
            return
        case["name"] = new_name
        self.refresh()
        self.cases_changed.emit()

    def _on_remove(self):
        row = self._selected_row()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a case to remove.")
            return
        name = self._cases[row]["name"]
        answer = QMessageBox.question(
            self, "Remove Case",
            f"Remove case '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._cases.pop(row)
        self.refresh()
        self.cases_changed.emit()
