from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QAbstractItemView, QRadioButton, QButtonGroup,
    QDialogButtonBox, QGroupBox
)
from PyQt6.QtCore import Qt


class QuantitySelectorDialog(QDialog):

    def __init__(self, quantity_names, module_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Compare — {module_name}")
        self.setMinimumWidth(300)
        self._selected_quantities = None
        self._layout_mode = None
        self._build_ui(quantity_names)

    def _build_ui(self, quantity_names):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(QLabel("Select quantities to compare:"))

        self._list = QListWidget()
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        for name in quantity_names:
            self._list.addItem(name)
        self._list.selectAll()
        layout.addWidget(self._list)

        layout_grp = QGroupBox("Plot layout")
        grp_lay = QVBoxLayout(layout_grp)
        self._rb_overlay   = QRadioButton(
            "One axes per quantity (cases overlaid)")
        self._rb_sidebyside = QRadioButton(
            "Side-by-side subplots per case")
        self._rb_overlay.setChecked(True)
        self._layout_bg = QButtonGroup(self)
        self._layout_bg.addButton(self._rb_overlay,    0)
        self._layout_bg.addButton(self._rb_sidebyside, 1)
        grp_lay.addWidget(self._rb_overlay)
        grp_lay.addWidget(self._rb_sidebyside)
        layout.addWidget(layout_grp)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        self._selected_quantities = [
            item.text() for item in self._list.selectedItems()
        ]
        self._layout_mode = (
            "overlay" if self._rb_overlay.isChecked() else "sidebyside"
        )
        self.accept()

    @property
    def selected_quantities(self):
        return self._selected_quantities

    @property
    def layout_mode(self):
        return self._layout_mode

    def get_selection(self):
        return self._selected_quantities, self._layout_mode
