"""
gui/mat_var_dialog.py
Variable-mapping confirmation dialog shown on every .mat load.
"""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QScrollArea, QWidget, QFrame,
    QGroupBox, QSpinBox, QCheckBox,
)
from PyQt6.QtCore import Qt

_5GB = 5 * 1024 ** 3   # byte threshold above which float16 option is shown

# (role, display label, is_optional)
_ROLES = [
    ("x",       "X coordinates",       False),
    ("y",       "Y coordinates",       False),
    ("u",       "U velocity",          False),
    ("v",       "V velocity",          False),
    ("w",       "W velocity (stereo)", True),
    ("isValid", "isValid mask",        True),
]


class MatVarMappingDialog(QDialog):
    """
    Confirms the role→variable mapping and snapshot subset before loading a .mat file.
    Always shown; auto-detected names are pre-filled for quick confirmation.
    """

    def __init__(self, filepath, var_info, initial_mapping, u_shape=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Variable Mapping")
        self.setMinimumWidth(480)
        self._filepath = filepath
        self._var_info = var_info
        self._mapping  = dict(initial_mapping)
        self._u_shape  = u_shape   # (Ny, Nx, N_total) in MATLAB logical order, or None
        self._combos   = {}
        self._build_ui()
        self._update_estimate()

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setSpacing(8)

        path_lbl = QLabel(f"<b>File:</b> {os.path.basename(self._filepath)}")
        path_lbl.setToolTip(self._filepath)
        path_lbl.setWordWrap(True)
        vl.addWidget(path_lbl)

        vl.addWidget(QLabel(
            "Confirm or adjust the variable assignments below, then click Load."
        ))

        # Variable summary table (scrollable)
        inner = QWidget()
        gl = QGridLayout(inner)
        gl.setContentsMargins(4, 4, 4, 4)
        gl.addWidget(QLabel("<b>Variable</b>"), 0, 0)
        gl.addWidget(QLabel("<b>Shape</b>"), 0, 1)
        for row, (name, shape) in enumerate(sorted(self._var_info.items()), 1):
            gl.addWidget(QLabel(name), row, 0)
            gl.addWidget(QLabel(str(shape)), row, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(160)
        scroll.setWidget(inner)
        vl.addWidget(scroll)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        vl.addWidget(sep)
        vl.addWidget(QLabel("<b>Assign variables to roles:</b>"))

        all_names       = sorted(self._var_info.keys())
        names_with_none = ["(none)"] + all_names

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        for row, (role, label, optional) in enumerate(_ROLES):
            suffix = "  (optional)" if optional else ""
            grid.addWidget(QLabel(f"{label}{suffix}"), row, 0)
            cb = QComboBox()
            cb.addItems(names_with_none if optional else all_names)
            presel = self._mapping.get(role)
            if presel and presel in all_names:
                offset = 1 if optional else 0
                cb.setCurrentIndex(all_names.index(presel) + offset)
            elif optional:
                cb.setCurrentIndex(0)   # "(none)"
            # Signals connected after all widgets are built (below)
            self._combos[role] = cb
            grid.addWidget(cb, row, 1)

        # Mask convention — always shown; only meaningful when isValid is assigned
        _conv_row = len(_ROLES)
        grid.addWidget(QLabel("  Mask convention"), _conv_row, 0)
        self._conv_combo = QComboBox()
        self._conv_combo.addItems([
            "1 = valid (uPrime default)",
            "1 = invalid (mask out)",
        ])
        grid.addWidget(self._conv_combo, _conv_row, 1)
        vl.addLayout(grid)

        # ---- Snapshot Range ----
        N_total = self._u_shape[2] if (self._u_shape and len(self._u_shape) >= 3) else None
        if N_total is not None:
            grp_title = f"Snapshot Range  (file contains N_total = {N_total:,} snapshots)"
        else:
            grp_title = "Snapshot Range"
        snap_grp = QGroupBox(grp_title)
        sg = QGridLayout(snap_grp)
        sg.setHorizontalSpacing(8)

        n = N_total if N_total is not None else 99_999

        self._spin_start = QSpinBox()
        self._spin_start.setRange(0, max(0, n - 1))
        self._spin_start.setValue(0)

        self._spin_end = QSpinBox()
        self._spin_end.setRange(0, max(0, n - 1))
        self._spin_end.setValue(max(0, n - 1))

        self._spin_step = QSpinBox()
        self._spin_step.setRange(1, max(1, n))
        self._spin_step.setValue(1)

        sg.addWidget(QLabel("Start snapshot:"), 0, 0)
        sg.addWidget(self._spin_start, 0, 1)
        sg.addWidget(QLabel("End snapshot:"), 1, 0)
        sg.addWidget(self._spin_end, 1, 1)
        sg.addWidget(QLabel("Step:"), 2, 0)
        sg.addWidget(self._spin_step, 2, 1)

        self._lbl_estimate = QLabel("")
        self._lbl_estimate.setStyleSheet("font-size: 10px;")
        sg.addWidget(self._lbl_estimate, 3, 0, 1, 2)
        vl.addWidget(snap_grp)

        # ---- Storage precision (shown only for large files) ----
        self._prec_grp = QGroupBox("Storage precision")
        pg = QVBoxLayout(self._prec_grp)
        pg.setSpacing(4)
        self._chk_f16 = QCheckBox(
            "Use float16 (half precision) — halves memory usage")
        self._chk_f16.setChecked(False)
        pg.addWidget(self._chk_f16)
        self._lbl_f16_warn = QLabel(
            "Float16 has ~3 decimal digits of precision. All computations "
            "(Reynolds stresses, TKE budget, POD, DMD, FFT spectra, "
            "high-order moments) will run at this reduced precision and may "
            "be inaccurate. Catastrophic cancellation in fluctuation-based "
            "statistics is likely. Use only for preliminary inspection or "
            "mean-field viewing. For final results, reload with default float32."
        )
        self._lbl_f16_warn.setStyleSheet("color: red; font-size: 10px;")
        self._lbl_f16_warn.setWordWrap(True)
        self._lbl_f16_warn.setVisible(False)
        pg.addWidget(self._lbl_f16_warn)
        self._prec_grp.setVisible(False)
        vl.addWidget(self._prec_grp)

        # ---- Buttons ----
        hl = QHBoxLayout()
        self._btn_load = QPushButton("Load")
        btn_cancel     = QPushButton("Cancel")
        self._btn_load.setDefault(True)
        self._btn_load.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        hl.addStretch()
        hl.addWidget(self._btn_load)
        hl.addWidget(btn_cancel)
        vl.addLayout(hl)

        # Connect signals now that every widget exists
        for role in self._combos:
            self._combos[role].currentTextChanged.connect(self._update_estimate)
        self._spin_start.valueChanged.connect(self._update_estimate)
        self._spin_end.valueChanged.connect(self._update_estimate)
        self._spin_step.valueChanged.connect(self._update_estimate)
        self._chk_f16.toggled.connect(self._update_estimate)

    def _update_estimate(self):
        start = self._spin_start.value()
        end   = self._spin_end.value()
        step  = self._spin_step.value()

        if end < start:
            self._lbl_estimate.setText("End must be ≥ Start")
            self._lbl_estimate.setStyleSheet("color: red; font-size: 10px;")
            self._btn_load.setEnabled(False)
            return

        u_text  = self._combos["u"].currentText()
        u_shape = self._var_info.get(u_text) if u_text not in ("", "(none)") else None
        if u_shape is None:
            u_shape = self._u_shape

        n_loaded = len(range(start, end + 1, step))

        if u_shape and len(u_shape) >= 3:
            Ny, Nx = u_shape[0], u_shape[1]
            w_text = self._combos["w"].currentText()
            n_vel  = 2 + (1 if w_text not in ("", "(none)") else 0)

            est_bytes_f32 = n_loaded * Ny * Nx * n_vel * 4
            show_prec     = est_bytes_f32 >= _5GB
            self._prec_grp.setVisible(show_prec)

            use_f16  = show_prec and self._chk_f16.isChecked()
            self._lbl_f16_warn.setVisible(use_f16)
            bpe      = 2 if use_f16 else 4
            dtype_s  = "float16" if use_f16 else "float32"
            gb       = n_loaded * Ny * Nx * n_vel * bpe / 1e9
            size_s   = f"~{gb:.1f} GB {dtype_s}" if gb >= 1.0 else f"~{gb * 1024:.0f} MB {dtype_s}"
            self._lbl_estimate.setText(f"Will load {n_loaded:,} snapshots ({size_s})")
        else:
            self._prec_grp.setVisible(False)
            self._lbl_f16_warn.setVisible(False)
            self._lbl_estimate.setText(f"Will load {n_loaded:,} snapshots")

        self._lbl_estimate.setStyleSheet("color: #aaa; font-size: 10px;")
        self._check_load_enabled()

    def _check_load_enabled(self):
        required_filled = all(
            self._combos[role].currentText() not in ("", "(none)")
            for role, _, optional in _ROLES
            if not optional
        )
        valid_range = self._spin_end.value() >= self._spin_start.value()
        self._btn_load.setEnabled(required_filled and valid_range)

    def get_mapping(self):
        result = {}
        for role, _, optional in _ROLES:
            val = self._combos[role].currentText()
            result[role] = None if val == "(none)" else val
        return result

    def get_subset(self):
        """Return (snap_start, snap_end, snap_step) from the snapshot range spinboxes."""
        return (self._spin_start.value(),
                self._spin_end.value(),
                self._spin_step.value())

    def get_dtype(self):
        """Return np.float16 if the user ticked the half-precision box, else np.float32."""
        try:
            if self._prec_grp.isVisible() and self._chk_f16.isChecked():
                return np.float16
        except AttributeError:
            pass
        return np.float32

    def get_mask_convention(self):
        """Return 'valid' or 'invalid' per the Mask convention dropdown."""
        return "invalid" if self._conv_combo.currentIndex() == 1 else "valid"
