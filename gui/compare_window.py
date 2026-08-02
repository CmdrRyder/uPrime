"""
gui/compare_window.py
---------------------
Standalone "Compare Cases" viewer (Stage A, 1D only).

Opens with NO dataset loaded and reads uPrime's own 1D exports (.csv / .xlsx /
.dat) via core.case_io. The design is *variable-driven*: a "Variable:" dropdown
lists the canonical quantities present across all loaded files (U, V, W, TKE,
R_uu, ...). Selecting a variable filters the basket to the files that contain
it and plots only those series — different quantities are never overlaid.

Reuses existing infrastructure rather than duplicating it:
  * core.case_io            -> file reader + CaseSeries data model
  * gui.case_manager_dialog -> per-series color / linestyle editing
  * gui.comparison_mixin    -> TAB10 auto-colors + merged-CSV writer
  * gui.arrow_toolbar       -> DrawAwareToolbar (300 DPI PNG + PDF/SVG export)
"""

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox, QGroupBox,
    QSizePolicy, QStatusBar, QApplication, QTabWidget, QRadioButton,
    QButtonGroup, QListWidget, QListWidgetItem, QDoubleSpinBox, QCheckBox,
    QStackedWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import Normalize

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from gui.case_manager_dialog import CaseManagerDialog
from gui.comparison_mixin import _TAB10, _SimpleDF, _concat_to_csv
from gui.line_selector import LineSelectorWidget, compute_snapped_line
from core.reynolds_stress import extract_line_profile
from core.line_sample import sample_along_line
from core.case_io import (
    read_case_file, CaseSeries, QUANTITY_INFO, is_signed_component,
)

_FONT_AX   = 9
_FONT_TICK = 8
_FONT_LEG  = 8

_SUPPORTED_FILTER = (
    "uPrime exports (*.csv *.xlsx *.dat);;"
    "CSV (*.csv);;Excel (*.xlsx);;Tecplot DAT (*.dat);;All files (*)"
)

# Preferred ordering for the Variable dropdown; anything else is appended
# alphabetically after these.
_VAR_ORDER = ["U", "V", "W", "TKE",
              "R_uu", "R_vv", "R_ww", "R_uv", "R_uw", "R_vw"]


class CompareWindow(QWidget):
    """Self-contained, variable-driven multi-case 1D comparison viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Cases")
        self.resize(1200, 760)
        self.setAcceptDrops(True)

        self._records = []          # list[FileRecord]
        self._extracted = []        # CaseSeries extracted from 2D fields
        self._building = False      # guard tree.itemChanged during rebuild
        self._style_mgr_dlg = None
        self._style_dicts = []
        self._style_snapshot = []

        self._build_ui()
        self._refresh_variable_combo()
        self._set_status("Ready — add uPrime exports (.csv / .xlsx / .dat). "
                         "2D fields open in the Field Compare (2D) tab.")

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Shared "Add Files" strip (feeds BOTH tabs by content) ----
        top = QHBoxLayout()
        top.setContentsMargins(8, 6, 8, 2)
        self.btn_add = QPushButton("Add Files…")
        self.btn_add.clicked.connect(self._on_add_files)
        top.addWidget(self.btn_add)
        hint = QLabel("or drag & drop uPrime .csv / .xlsx / .dat files here")
        hint.setTextFormat(Qt.TextFormat.PlainText)
        hint.setStyleSheet("color:gray; font-size:11px;")
        top.addWidget(hint)
        top.addStretch()
        outer.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_1d_tab(), "1D Compare")
        self.field2d = Field2DTab(self)
        self.tabs.addTab(self.field2d, "Field Compare (2D)")
        outer.addWidget(self.tabs, stretch=1)

        # ---- Status bar (deep navy / coral), shared across tabs ----
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "QStatusBar { background:#0e0e1a; }"
            "QStatusBar QLabel { color:#e06c75; font-size:11px; }")
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#e06c75; font-size:11px;")
        self.status_bar.addWidget(self._status_lbl, 1)
        outer.addWidget(self.status_bar)

    def _build_1d_tab(self):
        tab = QWidget()
        content = QHBoxLayout(tab)
        content.setContentsMargins(6, 6, 6, 6)

        # ---- Left: case basket ----
        left = QWidget()
        left.setMinimumWidth(370)
        left.setMaximumWidth(460)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(2, 2, 2, 2)
        ll.setSpacing(6)

        var_row = QHBoxLayout()
        var_row.addWidget(QLabel("Variable:"))
        self.var_combo = QComboBox()
        self.var_combo.currentIndexChanged.connect(self._on_variable_changed)
        var_row.addWidget(self.var_combo, stretch=1)
        ll.addLayout(var_row)

        basket_grp = QGroupBox("Cases")
        bg = QVBoxLayout(basket_grp)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Case", "Info"])
        self.tree.setColumnWidth(0, 260)
        self.tree.itemChanged.connect(self._on_item_changed)
        bg.addWidget(self.tree)
        ll.addWidget(basket_grp, stretch=1)

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self._on_plot)
        ll.addWidget(self.btn_plot)

        self.btn_styles = QPushButton("Edit Styles")
        self.btn_styles.clicked.connect(self._open_edit_styles)
        ll.addWidget(self.btn_styles)

        exp_row = QHBoxLayout()
        self.btn_export_fig = QPushButton("Export Figure")
        self.btn_export_fig.clicked.connect(self._on_export_figure)
        self.btn_export_csv = QPushButton("Export Merged CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        exp_row.addWidget(self.btn_export_fig)
        exp_row.addWidget(self.btn_export_csv)
        ll.addLayout(exp_row)

        content.addWidget(left)

        # ---- Right: plot ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.toolbar = DrawAwareToolbar(self.canvas, self)
        rl.addWidget(self.toolbar)
        rl.addWidget(self.canvas)
        content.addWidget(right, stretch=1)
        return tab

    def _set_status(self, msg):
        self._status_lbl.setText(msg)

    # ------------------------------------------------------------------ #
    # Drag and drop
    # ------------------------------------------------------------------ #

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.isLocalFile()]
        paths = [p for p in paths if os.path.isfile(p)]
        if paths:
            self._load_paths(paths)

    # ------------------------------------------------------------------ #
    # File loading
    # ------------------------------------------------------------------ #

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add uPrime Export Files", "", _SUPPORTED_FILTER)
        if paths:
            self._load_paths(paths)

    def _load_paths(self, paths):
        n_ok, n_2d, errors = 0, 0, []
        for path in paths:
            rec = read_case_file(path)
            self._records.append(rec)
            if rec.error:
                errors.append((rec.source_file, rec.error))
            elif rec.is_2d:
                n_2d += 1
            else:
                n_ok += 1
        self._assign_colors()
        self._refresh_variable_combo()
        self._rebuild_tree()
        self._on_plot()
        self.field2d.refresh_sources()

        msg = f"Loaded {n_ok} 1D file(s)"
        if n_2d:
            msg += (f", {n_2d} 2D field(s) → Field Compare (2D) tab")
        if errors:
            msg += f", {len(errors)} failed"
        self._set_status(msg + ".")
        if errors:
            QMessageBox.warning(
                self, "Some files could not be read",
                "\n".join(f"• {f}: {e}" for f, e in errors))

    def two_d_records(self):
        """All successfully-parsed 2D field records."""
        return [r for r in self._records if r.is_2d and not r.error]

    def add_extracted_series(self, series_list):
        """Add CaseSeries extracted from 2D fields into the 1D basket."""
        self._extracted.extend(series_list)
        self._assign_colors()
        self._refresh_variable_combo()
        self._rebuild_tree()
        self._on_plot()

    def _all_series(self):
        for rec in self._records:
            for s in rec.series:
                yield s
        for s in self._extracted:
            yield s

    def _assign_colors(self):
        used = {s.style["color"] for s in self._all_series()
                if s.style.get("color")}
        for s in self._all_series():
            if s.style.get("color"):
                continue
            for color in _TAB10:
                if color not in used:
                    s.style["color"] = color
                    used.add(color)
                    break
            else:
                s.style["color"] = _TAB10[len(used) % len(_TAB10)]

    # ------------------------------------------------------------------ #
    # Variable dropdown
    # ------------------------------------------------------------------ #

    def _present_variables(self):
        keys = {s.quantity_key for s in self._all_series()}
        ordered = [k for k in _VAR_ORDER if k in keys]
        rest = sorted(k for k in keys if k not in _VAR_ORDER)
        return ordered + rest

    def _refresh_variable_combo(self):
        prev = self.var_combo.currentData()
        self.var_combo.blockSignals(True)
        self.var_combo.clear()
        for key in self._present_variables():
            self.var_combo.addItem(key, key)
        if prev is not None:
            idx = self.var_combo.findData(prev)
            if idx >= 0:
                self.var_combo.setCurrentIndex(idx)
        self.var_combo.blockSignals(False)

    def _current_variable(self):
        return self.var_combo.currentData()

    def _current_series(self):
        var = self._current_variable()
        if var is None:
            return []
        return [s for s in self._all_series() if s.quantity_key == var]

    def _on_variable_changed(self):
        self._rebuild_tree()
        self._on_plot()

    # ------------------------------------------------------------------ #
    # Tree (basket)
    # ------------------------------------------------------------------ #

    def _rebuild_tree(self):
        self._building = True
        self.tree.clear()

        var = self._current_variable()
        if var is not None:
            grp = QTreeWidgetItem([f"{var}  —  cases", ""])
            grp.setFlags(grp.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self.tree.addTopLevelItem(grp)
            for s in self._current_series():
                if s.source_kind == "extracted":
                    info = "extracted (2D line)"
                else:
                    info = "normalized" if s.um is not None else "raw"
                child = QTreeWidgetItem([s.label, info])
                child.setFlags(child.flags()
                               | Qt.ItemFlag.ItemIsUserCheckable
                               | Qt.ItemFlag.ItemIsEditable)
                child.setCheckState(
                    0, Qt.CheckState.Checked if s.enabled
                    else Qt.CheckState.Unchecked)
                child.setData(0, Qt.ItemDataRole.UserRole, s)
                tag = "  (extracted from 2D)" if s.source_kind == "extracted" else ""
                child.setToolTip(0, f"{s.source_file}  ·  {s.source_module}{tag}")
                grp.addChild(child)
            grp.setExpanded(True)

        self._building = False

    def _on_item_changed(self, item, column):
        if self._building:
            return
        s = item.data(0, Qt.ItemDataRole.UserRole)
        if s is None:
            return
        s.enabled = (item.checkState(0) == Qt.CheckState.Checked)
        new_label = item.text(0).strip()
        if new_label and new_label != s.label:
            s.label = new_label

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #

    def _on_plot(self):
        var = self._current_variable()
        self._set_status("Plotting…")
        QApplication.processEvents()
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            series = [s for s in self._current_series() if s.enabled]
            x_label = x_units = y_label = y_units = ""
            for s in series:
                st = s.style
                ax.plot(s.x_data, s.y_data,
                        color=st.get("color"),
                        linestyle=st.get("linestyle", "-"),
                        marker=(st.get("marker") or None),
                        linewidth=1.5, label=s.label)
                x_label, x_units = s.x_label, s.x_units
                y_label, y_units = s.y_label, s.y_units

            if series:
                ax.set_xlabel(f"{x_label} [{x_units}]" if x_units else x_label,
                              fontsize=_FONT_AX)
                ax.set_ylabel(f"{y_label} [{y_units}]" if y_units else y_label,
                              fontsize=_FONT_AX)
                ax.set_aspect("auto")
                ax.tick_params(labelsize=_FONT_TICK)
                ax.legend(fontsize=_FONT_LEG)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5,
                        "No enabled cases for this variable.\n"
                        "Add files and pick a Variable.",
                        transform=ax.transAxes, ha="center", va="center",
                        color="gray")

            self.fig.tight_layout(pad=0.5)
            self.canvas.draw()
            self.toolbar.set_home_limits()
            self._update_status(var, len(series))
        finally:
            QApplication.processEvents()

    def _update_status(self, var, n_plotted):
        if var is None:
            self._set_status("No plottable series loaded yet.")
            return
        msg = f"Plotting {n_plotted} case(s) for {var}."
        # Um-normalization mismatch across the enabled series of this variable.
        states = {(s.um is not None) for s in self._current_series()
                  if s.enabled}
        if len(states) > 1:
            msg += ("   ⚠ Um mismatch: some cases are Um-normalized and some "
                    "are raw — units may not match.")
        self._set_status(msg)

    # ------------------------------------------------------------------ #
    # Edit styles (reuse CaseManagerDialog)
    # ------------------------------------------------------------------ #

    def _open_edit_styles(self):
        series = self._current_series()
        if not series:
            QMessageBox.information(self, "No Cases",
                "There are no cases for the current variable to style.")
            return

        self._assign_colors()
        self._style_snapshot = list(series)
        self._style_dicts = []
        for s in series:
            self._style_dicts.append({
                "name":      s.label,
                "color":     s.style.get("color") or _TAB10[0],
                "linestyle": s.style.get("linestyle", "-"),
                "source":    s.source_file,
                "_series":   s,
            })

        if self._style_mgr_dlg is not None:
            try:
                self._style_mgr_dlg.close()
            except Exception:
                pass
        self._style_mgr_dlg = CaseManagerDialog(parent=self,
                                                cases=self._style_dicts)
        self._style_mgr_dlg.cases_changed.connect(self._apply_styles)
        self._style_mgr_dlg.show()
        self._style_mgr_dlg.raise_()
        self._style_mgr_dlg.activateWindow()

    def _apply_styles(self):
        kept = set()
        for d in self._style_dicts:
            s = d["_series"]
            kept.add(id(s))
            s.label = d["name"]
            s.style["color"] = d["color"]
            s.style["linestyle"] = d["linestyle"]
        for s in self._style_snapshot:
            if id(s) not in kept:
                s.enabled = False
        self._rebuild_tree()
        self._on_plot()

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _on_export_figure(self):
        # Reuse the toolbar's 300 DPI PNG + PDF/SVG save path.
        self.toolbar.save_figure()

    def set_variable(self, key):
        """Select a variable in the 1D dropdown (re-renders the basket/plot)."""
        idx = self.var_combo.findData(key)
        if idx >= 0:
            self.var_combo.setCurrentIndex(idx)

    def write_merged_csv(self, path, series):
        """Write CaseSeries to a merged long-format CSV (case/label column).

        The single canonical 1D merged-export writer, reused by the 1D tab's
        Export Merged CSV and the 2D line-extract Save button.
        """
        cols = ["case", "quantity", "x_label", "x_value", "y_label", "y_value"]
        frames = []
        for s in series:
            xlab = f"{s.x_label} [{s.x_units}]" if s.x_units else s.x_label
            ylab = f"{s.y_label} [{s.y_units}]" if s.y_units else s.y_label
            n = min(len(s.x_data), len(s.y_data))
            rows = [
                [s.label, s.quantity_key, xlab, s.x_data[i], ylab, s.y_data[i]]
                for i in range(n)
            ]
            frames.append(_SimpleDF(cols, rows))
        _concat_to_csv(frames, path)

    def _on_export_csv(self):
        var = self._current_variable()
        series = [s for s in self._current_series() if s.enabled]
        if not series:
            QMessageBox.information(self, "Nothing to Export",
                "No enabled cases for the current variable.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Merged CSV", f"compare_{var}.csv",
            "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.write_merged_csv(path, series)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        self._set_status(f"Exported {len(series)} case(s) to "
                         f"{os.path.basename(path)}.")
        QMessageBox.information(self, "Export Complete",
            f"Exported {len(series)} case(s) to:\n{path}")


_QWIDGETSIZE_MAX = 16777215


def _cmap_for(key, arr=None):
    """RdBu_r for signed components, viridis for positive; infer for unknown."""
    if is_signed_component(key):
        return "RdBu_r"
    if key in ("U", "MAG", "TKE", "R_uu", "R_vv", "R_ww"):
        return "viridis"
    if arr is not None and np.isfinite(arr).any() and \
            np.nanmin(arr) < 0 < np.nanmax(arr):
        return "RdBu_r"
    return "viridis"


class Field2DTab(PickerMixin, QWidget):
    """Field Compare (2D): tiled view + line extraction, fed by 2D .dat files.

    Kept fully separate from the Stage A 1D tab. Extracted line profiles are
    pushed back into the owner's 1D basket via ``owner.add_extracted_series``.
    """

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self._press_xy = None
        self._line_artist = None
        self._selection = None
        self._manual_active = False
        self._draw_rec = None
        self._x = self._y = None          # 2D meshgrids of the drawn field
        self._last_field_values = None
        self._tile_overrides = {}         # case label -> (vmin, vmax)

        self._build_ui()
        self.ax = self.fig2d.add_subplot(111)
        self._connect_mouse()
        self._setup_picker(self.canvas2d, self.ax,
                           status_label=None)
        self.refresh_sources()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        left = QWidget()
        left.setMinimumWidth(340)
        left.setMaximumWidth(430)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(2, 2, 2, 2)
        ll.setSpacing(6)

        var_row = QHBoxLayout()
        var_row.addWidget(QLabel("Variable:"))
        self.var2d = QComboBox()
        self.var2d.currentIndexChanged.connect(self._on_var_changed)
        var_row.addWidget(self.var2d, stretch=1)
        ll.addLayout(var_row)

        mode_grp = QGroupBox("Mode")
        mrow = QHBoxLayout(mode_grp)
        self.rb_tiled = QRadioButton("Tiled view")
        self.rb_line = QRadioButton("Line extract")
        self.rb_tiled.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self.rb_tiled)
        self._mode_grp.addButton(self.rb_line)
        self.rb_tiled.toggled.connect(self._on_mode_changed)
        mrow.addWidget(self.rb_tiled)
        mrow.addWidget(self.rb_line)
        ll.addWidget(mode_grp)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_tiled_panel())
        self.stack.addWidget(self._build_line_panel())
        ll.addWidget(self.stack, stretch=1)

        root.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        self.fig2d = Figure()
        self.canvas2d = FigureCanvas(self.fig2d)
        self.canvas2d.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)
        self.toolbar2d = DrawAwareToolbar(self.canvas2d, self)
        rl.addWidget(self.toolbar2d)
        rl.addWidget(self.canvas2d)
        root.addWidget(right, stretch=1)

    def _build_tiled_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(QLabel("Cases to tile (up to 6):"))
        self.tile_list = QListWidget()
        lay.addWidget(self.tile_list, stretch=1)

        scale_grp = QGroupBox("Color scale")
        sg = QVBoxLayout(scale_grp)
        sg.addWidget(QLabel("Shared across tiles by default. Override one tile:"))
        ov = QHBoxLayout()
        self.ov_tile = QComboBox()
        ov.addWidget(self.ov_tile, stretch=1)
        sg.addLayout(ov)
        sp = QHBoxLayout()
        sp.addWidget(QLabel("vmin"))
        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1e12, 1e12); self.spin_vmin.setDecimals(4)
        sp.addWidget(self.spin_vmin)
        sp.addWidget(QLabel("vmax"))
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1e12, 1e12); self.spin_vmax.setDecimals(4)
        sp.addWidget(self.spin_vmax)
        sg.addLayout(sp)
        brow = QHBoxLayout()
        btn_apply = QPushButton("Apply to tile")
        btn_apply.clicked.connect(self._on_apply_tile_scale)
        btn_reset = QPushButton("Reset to shared scale")
        btn_reset.clicked.connect(self._on_reset_scales)
        brow.addWidget(btn_apply); brow.addWidget(btn_reset)
        sg.addLayout(brow)
        lay.addWidget(scale_grp)

        self.btn_tiles = QPushButton("Plot Tiles")
        self.btn_tiles.clicked.connect(self._on_plot_tiles)
        lay.addWidget(self.btn_tiles)
        return w

    def _build_line_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        case_row = QHBoxLayout()
        case_row.addWidget(QLabel("Draw on case:"))
        self.line_case = QComboBox()
        self.line_case.currentIndexChanged.connect(self._draw_single_field)
        case_row.addWidget(self.line_case, stretch=1)
        lay.addLayout(case_row)

        # Line Entry (Draw / Manual)
        entry_grp = QGroupBox("Line Entry")
        eg = QVBoxLayout(entry_grp)
        er = QHBoxLayout()
        self.rb_draw = QRadioButton("Draw")
        self.rb_manual = QRadioButton("Manual")
        self.rb_draw.setChecked(True)
        self._entry_grp = QButtonGroup(self)
        self._entry_grp.addButton(self.rb_draw)
        self._entry_grp.addButton(self.rb_manual)
        self.rb_draw.toggled.connect(self._on_line_entry_changed)
        er.addWidget(self.rb_draw); er.addWidget(self.rb_manual)
        eg.addLayout(er)
        self.manual_widget = QWidget()
        mw = QHBoxLayout(self.manual_widget)
        mw.setContentsMargins(0, 0, 0, 0)
        self.sp_x0 = QDoubleSpinBox(); self.sp_y0 = QDoubleSpinBox()
        self.sp_x1 = QDoubleSpinBox(); self.sp_y1 = QDoubleSpinBox()
        for sp in (self.sp_x0, self.sp_y0, self.sp_x1, self.sp_y1):
            sp.setDecimals(3); sp.setRange(-1e6, 1e6)
        mw.addWidget(QLabel("x0")); mw.addWidget(self.sp_x0)
        mw.addWidget(QLabel("y0")); mw.addWidget(self.sp_y0)
        mw.addWidget(QLabel("x1")); mw.addWidget(self.sp_x1)
        mw.addWidget(QLabel("y1")); mw.addWidget(self.sp_y1)
        eg.addWidget(self.manual_widget)
        self.btn_manual = QPushButton("Plot line")
        self.btn_manual.clicked.connect(self._on_manual_plot)
        eg.addWidget(self.btn_manual)
        lay.addWidget(entry_grp)

        # Line Mode (Free/H/V) + spatial averaging — reuse analysis-module widget
        self.line_sel = LineSelectorWidget(show_avg=True)
        lay.addWidget(self.line_sel)

        self.chk_apply_all = QCheckBox("Apply to all loaded cases (same origin)")
        lay.addWidget(self.chk_apply_all)
        note = QLabel("Assumes all cases share a common origin / coordinate "
                      "system; the same p0→p1 and averaging are sampled from each.")
        note.setStyleSheet("color:gray; font-size:10px;")
        note.setWordWrap(True)
        lay.addWidget(note)

        btn_row = QHBoxLayout()
        self.btn_extract_plot = QPushButton("Plot")
        self.btn_extract_plot.setToolTip(
            "Extract all components along the line, add them to the 1D basket "
            "and switch to the 1D Compare tab.")
        self.btn_extract_plot.clicked.connect(self._on_extract_plot)
        self.btn_extract_save = QPushButton("Save")
        self.btn_extract_save.setToolTip(
            "Extract all components along the line and write them straight to a "
            "merged CSV (not added to the 1D basket).")
        self.btn_extract_save.clicked.connect(self._on_extract_save)
        btn_row.addWidget(self.btn_extract_plot)
        btn_row.addWidget(self.btn_extract_save)
        lay.addLayout(btn_row)
        lay.addStretch(1)

        self._on_line_entry_changed()
        return w

    # ------------------------------------------------------------------ #
    # Source refresh / variable + case lists
    # ------------------------------------------------------------------ #

    def _current_var(self):
        return self.var2d.currentData()

    def refresh_sources(self):
        recs = self.owner.two_d_records()
        keys = []
        for r in recs:
            for k in r.twod_components:
                if k not in keys:
                    keys.append(k)
        ordered = [k for k in _VAR_ORDER if k in keys]
        ordered += sorted(k for k in keys if k not in _VAR_ORDER)

        prev = self._current_var()
        self.var2d.blockSignals(True)
        self.var2d.clear()
        for k in ordered:
            self.var2d.addItem(k, k)
        if prev is not None:
            idx = self.var2d.findData(prev)
            if idx >= 0:
                self.var2d.setCurrentIndex(idx)
        self.var2d.blockSignals(False)
        self._refresh_cases()

    def _cases_for_var(self):
        var = self._current_var()
        if var is None:
            return []
        return [r for r in self.owner.two_d_records()
                if var in r.twod_components]

    def _refresh_cases(self):
        recs = self._cases_for_var()
        # tiled list (checkable), default-check first 6
        self.tile_list.clear()
        self.ov_tile.clear()
        for i, r in enumerate(recs):
            it = QListWidgetItem(os.path.splitext(r.source_file)[0])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if i < 6
                             else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, r)
            self.tile_list.addItem(it)
            self.ov_tile.addItem(os.path.splitext(r.source_file)[0])
        # line-extract case combo
        self.line_case.blockSignals(True)
        self.line_case.clear()
        for r in recs:
            self.line_case.addItem(os.path.splitext(r.source_file)[0], r)
        self.line_case.blockSignals(False)
        self._tile_overrides.clear()

    def _on_var_changed(self):
        self._refresh_cases()
        if self.rb_line.isChecked():
            self._draw_single_field()
        else:
            self._on_plot_tiles()

    def _on_mode_changed(self):
        if self.rb_tiled.isChecked():
            self.stack.setCurrentIndex(0)
            self._on_plot_tiles()
        else:
            self.stack.setCurrentIndex(1)
            self._draw_single_field()

    def _mode(self):
        return "line" if self.rb_line.isChecked() else "tiled"

    # ------------------------------------------------------------------ #
    # Tiled view
    # ------------------------------------------------------------------ #

    def _checked_tile_recs(self):
        recs = []
        for i in range(self.tile_list.count()):
            it = self.tile_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                recs.append(it.data(Qt.ItemDataRole.UserRole))
        return recs[:6]

    def _on_apply_tile_scale(self):
        label = self.ov_tile.currentText()
        if not label:
            return
        self._tile_overrides[label] = (self.spin_vmin.value(),
                                       self.spin_vmax.value())
        self._on_plot_tiles()

    def _on_reset_scales(self):
        self._tile_overrides.clear()
        self._on_plot_tiles()

    def _on_plot_tiles(self):
        var = self._current_var()
        recs = self._checked_tile_recs()
        # tiled view is not a drawing axes -> let the canvas expand
        self.canvas2d.setMinimumHeight(0)
        self.canvas2d.setMaximumHeight(_QWIDGETSIZE_MAX)
        self.ax = None
        self.owner._set_status("Rendering tiles…")
        QApplication.processEvents()
        try:
            self.fig2d.clear()
            if var is None or not recs:
                a = self.fig2d.add_subplot(111)
                a.text(0.5, 0.5, "Select a variable and check cases to tile.",
                       transform=a.transAxes, ha="center", va="center",
                       color="gray")
                self.canvas2d.draw()
                self.owner._set_status("No 2D cases selected for tiling.")
                return
            for r in recs:
                r.load_field()

            # shared range from the combined value ranges
            mins = [r.value_ranges[var][0] for r in recs
                    if var in r.value_ranges and np.isfinite(r.value_ranges[var][0])]
            maxs = [r.value_ranges[var][1] for r in recs
                    if var in r.value_ranges and np.isfinite(r.value_ranges[var][1])]
            svmin = min(mins) if mins else 0.0
            svmax = max(maxs) if maxs else 1.0
            cmap = _cmap_for(var, recs[0].components.get(var))

            n = len(recs)
            ncols = 1 if n <= 3 else 2          # few columns for wide fields
            nrows = int(np.ceil(n / ncols))

            # GridSpec: a thin top row for the shared horizontal colorbar, then
            # for each tile row a thin per-tile override-colorbar sub-row above
            # the tile sub-row. Colorbars live in their own axes, never over a
            # field.
            height_ratios = [0.22]
            for _ in range(nrows):
                height_ratios += [0.14, 1.0]
            gs = self.fig2d.add_gridspec(1 + 2 * nrows, ncols,
                                         height_ratios=height_ratios,
                                         hspace=0.55, wspace=0.28)

            any_shared = False
            for idx, r in enumerate(recs):
                row, col = idx // ncols, idx % ncols
                ax = self.fig2d.add_subplot(gs[1 + 2 * row + 1, col])
                label = os.path.splitext(r.source_file)[0]
                over = self._tile_overrides.get(label)
                vmin, vmax = over if over else (svmin, svmax)
                cf = self._contour_tile(ax, r, var, vmin, vmax, cmap)
                if over:
                    # Own small horizontal colorbar directly above this tile.
                    cax = self.fig2d.add_subplot(gs[1 + 2 * row, col])
                    cb = self.fig2d.colorbar(cf, cax=cax,
                                             orientation="horizontal",
                                             extend="neither")
                    cax.xaxis.set_ticks_position("top")
                    cax.tick_params(labelsize=_FONT_TICK)
                    cb.set_label(f"{label}", fontsize=_FONT_TICK)
                else:
                    any_shared = True

            if any_shared:
                lo, hi = (svmin, svmax) if svmax > svmin else (svmin, svmin + 1e-9)
                sm = cm.ScalarMappable(norm=Normalize(lo, hi), cmap=cmap)
                top_cax = self.fig2d.add_subplot(gs[0, :])
                cb = self.fig2d.colorbar(sm, cax=top_cax,
                                         orientation="horizontal",
                                         extend="neither")
                top_cax.xaxis.set_ticks_position("top")
                top_cax.xaxis.set_label_position("top")
                top_cax.tick_params(labelsize=_FONT_TICK)
                cb.set_label(var, fontsize=_FONT_AX)

            self.canvas2d.draw()
            self.toolbar2d.set_home_limits()
            self.owner._set_status(
                f"Tiled {n} case(s) of {var} "
                f"({'independent' if self._tile_overrides else 'shared'} scaling).")
        finally:
            QApplication.processEvents()

    def _contour_tile(self, ax, rec, key, vmin, vmax, cmap):
        Z = np.ma.masked_invalid(rec.components[key])
        if not vmax > vmin:
            vmax = vmin + 1e-9
        levels = np.linspace(vmin, vmax, 41)
        cf = ax.contourf(rec.x, rec.y, Z, levels=levels, cmap=cmap,
                         extend="neither")
        ax.set_title(os.path.splitext(rec.source_file)[0], fontsize=_FONT_AX)
        ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=_FONT_TICK)
        return cf

    # ------------------------------------------------------------------ #
    # Line extract — single field drawing
    # ------------------------------------------------------------------ #

    def _fix_canvas_height(self, x1, y1):
        x_ext = float(np.nanmax(x1) - np.nanmin(x1))
        y_ext = float(np.nanmax(y1) - np.nanmin(y1))
        ratio = (y_ext / x_ext) if x_ext > 0 else 0.5
        target_w = 640
        h = max(160, min(460, int(target_w * ratio) + 10))
        self.canvas2d.setFixedHeight(h)

    def _draw_single_field(self):
        rec = self.line_case.currentData()
        self._draw_rec = rec
        self._selection = None
        self._line_artist = None
        var = self._current_var()
        if rec is None or var is None:
            return
        self.owner._set_status("Loading field…")
        QApplication.processEvents()
        try:
            rec.load_field()
        finally:
            QApplication.processEvents()

        field = rec.components.get(var)
        if field is None:
            field = next(iter(rec.components.values()))
        self._x, self._y = np.meshgrid(rec.x, rec.y)
        self._last_field_values = field
        self._set_manual_ranges(rec.x, rec.y)
        self._fix_canvas_height(rec.x, rec.y)

        self.fig2d.clear()
        self.ax = self.fig2d.add_subplot(111)
        self._pick_field_ax = self.ax
        cmap = _cmap_for(var, field)
        self.ax.contourf(rec.x, rec.y, np.ma.masked_invalid(field),
                         levels=40, cmap=cmap, extend="neither")
        self.ax.set_title(f"{os.path.splitext(rec.source_file)[0]} — {var}"
                          "   (drag to draw a line)", fontsize=_FONT_AX - 1)
        self.ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        self.ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("white")
        self.ax.tick_params(labelsize=_FONT_TICK)
        self.fig2d.tight_layout(pad=0.5)
        self.canvas2d.draw()
        self.toolbar2d.set_home_limits()
        self.owner._set_status(f"Showing {var} for "
                               f"{os.path.splitext(rec.source_file)[0]}.")

    def _set_manual_ranges(self, x1, y1):
        xmin, xmax = float(np.nanmin(x1)), float(np.nanmax(x1))
        ymin, ymax = float(np.nanmin(y1)), float(np.nanmax(y1))
        for sp in (self.sp_x0, self.sp_x1):
            sp.setRange(xmin, xmax)
        for sp in (self.sp_y0, self.sp_y1):
            sp.setRange(ymin, ymax)
        self.sp_x0.setValue(xmin); self.sp_x1.setValue(xmax)
        self.sp_y0.setValue((ymin + ymax) / 2.0)
        self.sp_y1.setValue((ymin + ymax) / 2.0)

    def _on_line_entry_changed(self, *args):
        self._manual_active = self.rb_manual.isChecked()
        self.manual_widget.setVisible(self._manual_active)
        self.btn_manual.setVisible(self._manual_active)
        self.line_sel.setVisible(not self._manual_active)

    def _drawing_active(self):
        return self.rb_line.isChecked() and not self._manual_active

    def _clear_line_artist(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None

    def _connect_mouse(self):
        self.canvas2d.mpl_connect("button_press_event", self._on_press)
        self.canvas2d.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas2d.mpl_connect("button_release_event", self._on_release)

    def _line_axes_ready(self):
        return (self.rb_line.isChecked() and self.ax is not None
                and self._x is not None and not self._manual_active)

    def _on_press(self, event):
        if not self._line_axes_ready() or event.inaxes != self.ax:
            return
        if self._toolbar_active(self.toolbar2d):
            return
        if event.button == 1:
            self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._toolbar_active(self.toolbar2d):
            self._press_xy = None
            return
        x0, y0 = self._press_xy
        lmode = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, event.xdata, event.ydata, lmode)
        self._clear_line_artist()
        ln, = self.ax.plot([lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10)
        self._line_artist = ln
        self.canvas2d.draw()

    def _on_release(self, event):
        if self._press_xy is None:
            return
        if self._toolbar_active(self.toolbar2d):
            self._press_xy = None
            return
        if event.xdata is None or event.ydata is None:
            self._press_xy = None
            return
        x0, y0 = self._press_xy
        self._press_xy = None
        lmode = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, event.xdata, event.ydata, lmode)
        if abs(lx1 - lx0) < 0.1 and abs(ly1 - ly0) < 0.1:
            self.owner._set_status("Line too short — try again.")
            return
        self._set_line((lx0, ly0), (lx1, ly1))

    def _on_manual_plot(self):
        if self._draw_rec is None:
            QMessageBox.information(self, "No Case",
                "Pick a case to draw on first.")
            return
        p0 = (self.sp_x0.value(), self.sp_y0.value())
        p1 = (self.sp_x1.value(), self.sp_y1.value())
        if abs(p1[0] - p0[0]) < 0.1 and abs(p1[1] - p0[1]) < 0.1:
            self.owner._set_status("Line too short — adjust coordinates.")
            return
        self._set_line(p0, p1)

    def _set_line(self, p0, p1):
        """Single entry point shared by drawn and manual lines."""
        if self.ax is None:
            return
        self._clear_line_artist()
        ln, = self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                           "r-", linewidth=2, zorder=10)
        self._line_artist = ln
        self.canvas2d.draw()
        self._selection = {"x0": p0[0], "y0": p0[1], "x1": p1[0], "y1": p1[1]}
        self.owner._set_status(
            f"Line ({p0[0]:.1f},{p0[1]:.1f}) → ({p1[0]:.1f},{p1[1]:.1f}) mm. "
            "Use 'Plot' to add to the 1D tab or 'Save' to write a CSV.")

    # ------------------------------------------------------------------ #
    # Extraction -> 1D basket
    # ------------------------------------------------------------------ #

    def _profile(self, field, x1, y1, Xg, Yg, p0, p1, mode, avg):
        if self._manual_active:
            s, vals = sample_along_line(x1, y1, field, p0, p1)
            return vals, s
        vals, dist, _xp, _yp = extract_line_profile(
            field, Xg, Yg, p0[0], p0[1], p1[0], p1[1],
            mode=mode, avg_band=avg)
        return vals, dist

    def _extract_series(self):
        """Sample ALL components of the target file(s) along the current line.

        Returns (series_list, n_targets); series_list may be empty if nothing
        finite was found, or (None, 0) when there is no line / no target.
        Fresh CaseSeries are built on every call so Plot and Save are
        independent.
        """
        if self._selection is None:
            QMessageBox.information(self, "No Line",
                "Draw a line (or use Manual + Plot line) first.")
            return None, 0
        var = self._current_var()
        sel = self._selection
        p0 = (sel["x0"], sel["y0"])
        p1 = (sel["x1"], sel["y1"])
        if self.chk_apply_all.isChecked():
            targets = [r for r in self.owner.two_d_records()
                       if var in r.twod_components]
        else:
            targets = [self._draw_rec] if self._draw_rec else []
        if not targets:
            return None, 0

        mode = self.line_sel.get_mode()
        avg = self.line_sel.get_avg_band()
        self.owner._set_status("Extracting components along line…")
        QApplication.processEvents()
        new_series = []
        try:
            for rec in targets:
                rec.load_field()
                Xg, Yg = np.meshgrid(rec.x, rec.y)
                casebase = os.path.splitext(rec.source_file)[0]
                for key, field in rec.components.items():
                    vals, dist = self._profile(field, rec.x, rec.y, Xg, Yg,
                                               p0, p1, mode, avg)
                    if not np.isfinite(vals).any():
                        continue
                    info = QUANTITY_INFO.get(key)
                    y_label = info["y_label"] if info else key
                    y_units = info["y_units"] if info else ""
                    new_series.append(CaseSeries(
                        source_file=rec.source_file,
                        source_module=rec.source_module or "",
                        quantity_key=key,
                        x_type="arc_length", x_label="s", x_units="mm",
                        x_data=np.asarray(dist), y_label=y_label,
                        y_units=y_units, y_data=np.asarray(vals), um=None,
                        label=f"{casebase}_{key}",
                        style={"color": None, "linestyle": "-", "marker": ""},
                        enabled=True, source_kind="extracted"))
        finally:
            QApplication.processEvents()
        return new_series, len(targets)

    def _extract_summary(self, series, ntargets):
        ncomp = len({s.quantity_key for s in series})
        return (f"Extracted {len(series)} profile(s) "
                f"({ncomp} component(s) × {ntargets} case(s))")

    def _on_extract_plot(self):
        series, ntargets = self._extract_series()
        if series is None:
            return
        if not series:
            self.owner._set_status("No finite data extracted along the line.")
            return
        comps = [s.quantity_key for s in series]
        self.owner.add_extracted_series(series)
        # Make sure a just-extracted component is on screen, then switch tab.
        cur = self.owner._current_variable()
        target_var = cur if cur in comps else comps[0]
        self.owner.set_variable(target_var)
        self.owner.tabs.setCurrentIndex(0)      # 1D Compare
        self.owner._set_status(
            self._extract_summary(series, ntargets)
            + f" — plotted in the 1D tab under {target_var}.")

    def _on_extract_save(self):
        series, ntargets = self._extract_series()
        if series is None:
            return
        if not series:
            self.owner._set_status("No finite data extracted along the line.")
            return
        var = self._current_var()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Extracted Profiles", f"extracted_{var}.csv",
            "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.owner.write_merged_csv(path, series)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return
        self.owner._set_status(
            self._extract_summary(series, ntargets)
            + f" — saved to {os.path.basename(path)}.")
        QMessageBox.information(self, "Saved",
            f"Saved {len(series)} profile(s) to:\n{path}")
