"""
gui/tke_budget_window.py
------------------------
TKE budget analysis window.

Plot modes
----------
2D Contour  : filled contour of any budget term
Line Profile: left-click + drag on field to draw free / horizontal / vertical line

No ROI rectangle in this window -- contour or line only.

Layout
------
~45 / 55 split: left = field + controls, right = result canvas.
Field canvas uses setFixedHeight computed from data aspect ratio so that
set_aspect("equal") fills the widget without white margins, and every pixel
of the canvas is inside the axes -- left-click always registers.
"""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QRadioButton, QCheckBox, QSizePolicy,
    QMessageBox, QSplitter, QSpinBox, QComboBox,
    QDoubleSpinBox, QButtonGroup, QFileDialog, QProgressBar,
    QApplication,
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.reynolds_stress import extract_line_profile
from core.export import export_2d_tecplot, export_line_csv
from gui.line_selector import LineSelectorWidget, compute_snapped_line
from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from gui.comparison_mixin import ComparisonMixin

TERMS = {
    "k"    : {"label": "TKE  k",                "color": "tab:blue"},
    "P"    : {"label": "Production  P",          "color": "tab:red"},
    "C"    : {"label": "Convection  C",          "color": "tab:orange"},
    "D"    : {"label": "Turb. Diffusion  D",     "color": "tab:green"},
    "R"    : {"label": "Residual  R",            "color": "tab:purple"},
    "dkdt" : {"label": "\u2202k/\u2202t  (TR)", "color": "tab:brown"},
}

_FONT_AX   = 9
_FONT_TICK = 8
_FONT_LEG  = 8


class TKEBudgetWindow(PickerMixin, ComparisonMixin, QWidget):

    _module_name      = "TKE Budget"
    _expected_columns = ["mean_P", "mean_C", "mean_D", "mean_R"]
    _axis_columns     = ["dist_mm", "x_mm", "y_mm"]

    def __init__(self, dataset, is_time_resolved=False,
                 Nt_warn=2000, duration_warn=9999, parent=None):
        super().__init__(parent)
        self.dataset   = dataset
        self._is_tr    = is_time_resolved
        self.setWindowTitle("TKE Budget Analysis")
        self.resize(1700, 900)

        self._mode        = "contour"
        self._press_xy    = None
        self._line_artist = None
        self._selection   = None
        self._budget      = None
        self._last_line   = None

        self._build_ui()
        self._draw_field()
        self._connect_mouse()
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    # ----------------------------------------------------------------------- #

    def _drawing_active(self):
        return self._mode == "line"

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        from core.constants import CONVERGENCE_WARNING_N
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        Nt = self.dataset["Nt"]
        if Nt < CONVERGENCE_WARNING_N:
            lbl_warn = QLabel(
                f"⚠ Only N={Nt} snapshots loaded. "
                "Statistics may not be converged. "
                "Use the Mean & Convergence module to verify."
            )
            lbl_warn.setStyleSheet(
                "background:#FFF3CD; color:#856404; "
                "font-weight:bold; padding:4px 8px;"
            )
            lbl_warn.setWordWrap(True)
            outer.addWidget(lbl_warn)
        content = QWidget()
        root = QHBoxLayout(content)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- LEFT ----
        left = QWidget()
        left.setMinimumWidth(500)
        left.setMaximumWidth(720)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(4)

        # Field canvas -- height fixed by _draw_field() based on data aspect
        self.field_fig    = Figure()
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Fixed)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)

        # Plot Mode
        pm_grp = QGroupBox("Plot Mode")
        pm_lay = QHBoxLayout(pm_grp)
        self.rb_contour = QRadioButton("2D Contour")
        self.rb_line    = QRadioButton("Line Profile")
        self.rb_contour.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_contour)
        bg.addButton(self.rb_line)
        self.rb_contour.toggled.connect(self._on_mode_changed)
        pm_lay.addWidget(self.rb_contour)
        pm_lay.addWidget(self.rb_line)
        ll.addWidget(pm_grp)

        # Term selector
        self.contour_grp = QGroupBox("Term to Display")
        cl = QHBoxLayout(self.contour_grp)
        cl.addWidget(QLabel("Term:"))
        self.combo_term = QComboBox()
        for key, meta in TERMS.items():
            if key == "dkdt" and not self._is_tr:
                continue
            self.combo_term.addItem(meta["label"], key)
        cl.addWidget(self.combo_term)
        ll.addWidget(self.contour_grp)

        # Line selector
        self.line_sel = LineSelectorWidget(show_avg=True)
        self.line_sel.setVisible(False)
        ll.addWidget(self.line_sel)

        self.lbl_hint = QLabel("Select term and click 'Plot'.")
        self.lbl_hint.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        # Parameters
        comp_grp = QGroupBox("Parameters")
        cp = QVBoxLayout(comp_grp)

        n_row = QHBoxLayout()
        self.chk_norm = QCheckBox("Normalize by Um\u00b3/L")
        self.chk_norm.setChecked(False)
        n_row.addWidget(self.chk_norm)
        cp.addLayout(n_row)

        um_row = QHBoxLayout()
        um_row.addWidget(QLabel("Um [m/s]:"))
        self.spin_um = QDoubleSpinBox()
        self.spin_um.setRange(0.001, 1000)
        self.spin_um.setValue(1.0)
        self.spin_um.setDecimals(3)
        self.spin_um.setSingleStep(0.1)
        um_row.addWidget(self.spin_um)
        cp.addLayout(um_row)

        L_row = QHBoxLayout()
        L_row.addWidget(QLabel("L [mm]:"))
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.001, 10000)
        self.spin_L.setValue(7.5)
        self.spin_L.setDecimals(3)
        self.spin_L.setSingleStep(0.5)
        L_row.addWidget(self.spin_L)
        cp.addLayout(L_row)

        sm_row = QHBoxLayout()
        self.chk_smooth = QCheckBox("Smooth triple corr. (kernel):")
        self.chk_smooth.setChecked(True)
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 15)
        self.spin_kernel.setValue(3)
        self.spin_kernel.setSingleStep(2)
        sm_row.addWidget(self.chk_smooth)
        sm_row.addWidget(self.spin_kernel)
        cp.addLayout(sm_row)

        if self._is_tr:
            self.chk_dkdt = QCheckBox("Compute \u2202k/\u2202t (TR)")
            self.chk_dkdt.setChecked(True)
            cp.addWidget(self.chk_dkdt)
        else:
            self.chk_dkdt = None

        ll.addWidget(comp_grp)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["RdBu_r", "hot_r", "viridis", "plasma", "seismic"])
        cmap_row.addWidget(self.combo_cmap)
        ll.addLayout(cmap_row)

        self.btn_compute = QPushButton("Compute Budget")
        self.btn_compute.clicked.connect(self._on_compute)
        ll.addWidget(self.btn_compute)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        ll.addWidget(self.progress_bar)

        self.term_grp = QGroupBox("Terms to plot")
        tg_lay = QHBoxLayout(self.term_grp)
        tg_lay.setContentsMargins(4, 2, 4, 2)
        self.term_chks = {}
        _filter_keys = ["P", "C", "D", "R"] + (["dkdt"] if self._is_tr else [])
        for _key in _filter_keys:
            _chk = QCheckBox(TERMS[_key]["label"])
            _chk.setChecked(True)
            _chk.toggled.connect(self._on_term_filter_changed)
            self.term_chks[_key] = _chk
            tg_lay.addWidget(_chk)
        self.term_grp.setVisible(False)
        ll.addWidget(self.term_grp)

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.setEnabled(False)
        self.btn_plot.clicked.connect(self._on_plot)
        ll.addWidget(self.btn_plot)

        self.btn_export = QPushButton("Export Data...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        ll.addWidget(self.btn_export)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        self._init_comparison_toolbar(ll)

        ll.addStretch(1)

        # ---- RIGHT ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.result_fig    = Figure()
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)

        chk_row = QHBoxLayout()
        chk_row.addStretch()
        self.chk_hide_axes = QCheckBox("Hide axes")
        self.chk_hide_axes.stateChanged.connect(self._on_plot)
        chk_row.addWidget(self.chk_hide_axes)
        self.chk_hide_colorbar = QCheckBox("Hide colorbar")
        self.chk_hide_colorbar.stateChanged.connect(self._on_plot)
        chk_row.addWidget(self.chk_hide_colorbar)
        rl.addLayout(chk_row)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([750, 950])   # ~45 / 55
        outer.addWidget(content, stretch=1)

    # ----------------------------------------------------------------------- #
    # Field plot
    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds   = self.dataset
        x, y = ds["x"], ds["y"]
        from core.dataset_utils import get_masked
        speed = np.sqrt(np.nanmean(get_masked(ds, "U"), axis=2)**2 +
                        np.nanmean(get_masked(ds, "V"), axis=2)**2)
        speed[~ds["MASK"]] = np.nan

        # Fix canvas height to match data aspect ratio.
        # This makes set_aspect("equal") fill the widget with no white margins,
        # so every click inside the widget is inside the axes.
        x_ext = float(np.nanmax(x) - np.nanmin(x))
        y_ext = float(np.nanmax(y) - np.nanmin(y))
        ratio = (y_ext / x_ext) if x_ext > 0 else 0.5
        # Target width ~600px (left panel ~720px minus margins/toolbar ~120px)
        target_w = 600
        target_h = max(150, min(420, int(target_w * ratio) + 10))
        self.field_canvas.setFixedHeight(target_h)

        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        self.field_ax.contourf(x, y, np.ma.masked_invalid(speed), levels=40, cmap="RdBu_r")
        self.field_ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        self.field_ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        self.field_ax.set_title(
            "Line mode: left-click+drag to draw.  Free / Horizontal / Vertical.",
            fontsize=_FONT_AX - 1)
        self.field_ax.set_aspect("equal")   # safe -- canvas height is constrained
        self.field_ax.set_facecolor("white")
        self.field_ax.tick_params(labelsize=_FONT_TICK)
        self.field_fig.tight_layout(pad=0.2)
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()
        self._x = x
        self._y = y
        self._last_field_values = speed

    # ----------------------------------------------------------------------- #
    # Mode
    # ----------------------------------------------------------------------- #

    def _on_mode_changed(self):
        if self.rb_contour.isChecked():
            self._mode = "contour"
            self.contour_grp.setVisible(True)
            self.line_sel.setVisible(False)
            self.term_grp.setVisible(False)
            self.lbl_hint.setText("Select term and click 'Plot'.")
        else:
            self._mode = "line"
            self.contour_grp.setVisible(False)
            self.line_sel.setVisible(True)
            self.term_grp.setVisible(True)
            if self._budget is not None:
                self.lbl_hint.setText(
                    "Left-click+drag to draw a line, then click Plot.")
            else:
                self.lbl_hint.setText(
                    "Compute budget first.  Then left-click+drag to draw a line.")
        self._clear_line()
        self._selection = None

    # ----------------------------------------------------------------------- #
    # Mouse -- left-click drag, no ROI rectangle
    # ----------------------------------------------------------------------- #

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.field_ax:
            return
        if self._toolbar_active(self.field_toolbar):
            return
        if event.button == 1 and self._mode == "line":
            self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None:
            return
        if event.inaxes != self.field_ax or event.xdata is None:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        lmode  = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, x1, y1, lmode)

        self._clear_line()
        ln, = self.field_ax.plot(
            [lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10)
        self._line_artist = ln
        self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        self._press_xy = None

        if event.inaxes != self.field_ax or event.xdata is None:
            return

        x1, y1 = event.xdata, event.ydata
        lmode  = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, x1, y1, lmode)

        if abs(lx1 - lx0) < 0.1 and abs(ly1 - ly0) < 0.1:
            self.lbl_hint.setText("Line too short -- try again.")
            return

        # Commit line on canvas
        self._clear_line()
        ln, = self.field_ax.plot(
            [lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10)
        self._line_artist = ln
        self.field_canvas.draw()

        self._selection = {"x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1}
        self.lbl_hint.setText(
            f"Line ({lmode}): ({lx0:.1f},{ly0:.1f}) \u2192 ({lx1:.1f},{ly1:.1f}) mm")

        if self._budget is not None:
            self.btn_plot.setEnabled(True)
        else:
            self.lbl_status.setText("Line drawn. Compute budget, then click Plot.")

    def _clear_line(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None

    # ----------------------------------------------------------------------- #
    # Compute
    # ----------------------------------------------------------------------- #

    def _on_compute(self):
        from core.dataset_utils import get_masked
        from core.workers import TKEBudgetWorker

        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        ds     = self.dataset
        kernel = self.spin_kernel.value() if self.chk_smooth.isChecked() else 1
        dkdt   = self.chk_dkdt.isChecked() if self.chk_dkdt else False

        U = get_masked(ds, "U")
        V = get_masked(ds, "V")
        W = get_masked(ds, "W")

        self.lbl_status.setText("Busy: computing TKE budget...")
        self.btn_compute.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self._dkdt_requested = dkdt
        self._worker = TKEBudgetWorker(
            U, V, W, self._x, self._y,
            mask=ds.get("MASK"),
            smooth_kernel=kernel,
            compute_dkdt=dkdt,
        )
        self._worker.finished.connect(self._on_budget_result)
        self._worker.error.connect(self._on_budget_error)
        self._worker.finished.connect(lambda _: self._reset_compute_ui())
        self._worker.error.connect(lambda _: self._reset_compute_ui())
        self._worker.start()

    def _reset_compute_ui(self):
        self.btn_compute.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)

    def _on_budget_result(self, budget):
        self._budget = budget
        self.btn_plot.setEnabled(True)
        if self._mode == "line":
            if self._selection is not None:
                self.lbl_status.setText("Budget computed. Click Plot.")
            else:
                self.lbl_status.setText(
                    "Budget computed. Draw a line, then click Plot.")
            self.lbl_hint.setText(
                "Left-click+drag to draw a line, then click Plot.")
        else:
            self.lbl_status.setText("Budget computed. Click Plot.")

        if self._dkdt_requested and budget.get("dkdt") is not None:
            if self.combo_term.findData("dkdt") == -1:
                self.combo_term.addItem(TERMS["dkdt"]["label"], "dkdt")

    def _on_budget_error(self, tb_str):
        QMessageBox.critical(self, "TKE Budget Error", tb_str)
        self.lbl_status.setText("Error — see dialog.")

    # ----------------------------------------------------------------------- #
    # Normalization
    # ----------------------------------------------------------------------- #

    def _scale_for(self, key):
        """Return (scale, unit_label) for the given term key."""
        if self.chk_norm.isChecked():
            um  = self.spin_um.value()
            L_m = self.spin_L.value() / 1000.0
            if key == "k":
                s   = 1.0 / (um ** 2)
                lbl = f"k/Um\u00b2  (Um={um:.2f} m/s)"
            else:
                s   = L_m / (um ** 3)
                lbl = f"Um\u00b3/L  (Um={um:.2f} m/s, L={self.spin_L.value():.1f} mm)"
        else:
            s   = 1.0
            lbl = "[m\u00b2/s\u00b2]" if key == "k" else "[m\u00b2/s\u00b3]"
        return s, lbl

    # ----------------------------------------------------------------------- #
    # Plot
    # ----------------------------------------------------------------------- #

    def _on_plot(self):
        if self._budget is None:
            QMessageBox.information(self, "No Data",
                "Please compute the budget first.")
            return
        if self._mode == "contour":
            self._plot_contour()
        else:
            if self._selection is None:
                QMessageBox.information(self, "No Line",
                    "Please draw a line on the field (left-click+drag).")
                return
            self._plot_line()

    def _on_term_filter_changed(self):
        if (self._mode == "line"
                and self._budget is not None
                and self._selection is not None):
            self._plot_line()

    def _plot_contour(self):
        key   = self.combo_term.currentData()
        field = self._budget.get(key)
        if field is None:
            QMessageBox.warning(self, "Not Available",
                f"Term '{key}' is not available.")
            return

        scale, unit_str = self._scale_for(key)
        cmap = self.combo_cmap.currentText()
        data = field * scale

        if key == "k":
            vmin = 0
            vmax = np.nanmax(data)
        else:
            vmax = np.nanmax(np.abs(data))
            vmin = -vmax

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, np.ma.masked_invalid(data), levels=50, cmap=cmap,
                         vmin=vmin, vmax=vmax)
        cb = self.result_fig.colorbar(cf, ax=ax,
                                      label=f"{TERMS[key]['label']} {unit_str}",
                                      shrink=0.8)
        if self.chk_hide_colorbar.isChecked():
            cb.remove()
            self.result_fig.tight_layout(pad=0.5)
        ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        ax.set_title(TERMS[key]["label"], fontsize=_FONT_AX)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=_FONT_TICK)
        if self.chk_hide_axes.isChecked():
            ax.axis('off')
            ax.set_title('')
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.btn_export.setEnabled(True)
        self.lbl_status.setText(f"Contour: {TERMS[key]['label']}")

    def _plot_line(self):
        sel          = self._selection
        _, unit_str  = self._scale_for("P")
        _, k_unit    = self._scale_for("k")
        lmode        = self.line_sel.get_mode()
        avg_band     = self.line_sel.get_avg_band()
        xlabel       = {"horizontal": "x [mm]",
                        "vertical":   "y [mm]"}.get(lmode, "Distance from origin [mm]")

        self.result_fig.clear()
        ax_k, ax_bud = self.result_fig.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2]})

        self._last_line = {"dist": None, "xpts": None, "ypts": None,
                           "means": {}, "unit_str": unit_str}

        # ---- top panel: TKE k ----
        plotted_k = False
        if self._budget.get("k") is not None:
            scale_k, _ = self._scale_for("k")
            vals_k, dist_k, xpts_k, ypts_k = extract_line_profile(
                self._budget["k"] * scale_k, self._x, self._y,
                sel["x0"], sel["y0"], sel["x1"], sel["y1"],
                mode=lmode, avg_band=avg_band)
            valid_k = np.isfinite(vals_k)
            if np.any(valid_k):
                ax_k.plot(dist_k[valid_k], vals_k[valid_k],
                          color=TERMS["k"]["color"],
                          label=TERMS["k"]["label"],
                          linewidth=1.5)
                plotted_k = True
                self._last_line["dist"]      = dist_k
                self._last_line["xpts"]      = xpts_k
                self._last_line["ypts"]      = ypts_k
                self._last_line["means"]["k"] = vals_k

        if plotted_k:
            ax_k.set_ylabel(f"TKE k {k_unit}", fontsize=_FONT_AX)
            ax_k.legend(fontsize=_FONT_LEG)
            ax_k.grid(True, alpha=0.3)
            ax_k.tick_params(labelsize=_FONT_TICK)
        else:
            ax_k.text(0.5, 0.5, "k not available",
                      transform=ax_k.transAxes, ha="center", va="center",
                      fontsize=10, color="gray")
            ax_k.set_ylabel(f"TKE k {k_unit}", fontsize=_FONT_AX)
            ax_k.tick_params(labelsize=_FONT_TICK)
        ax_k.set_title("TKE Budget Profile", fontsize=_FONT_AX)

        # ---- bottom panel: P, C, D, R (dkdt) ----
        budget_keys = [key for key in ["P", "C", "D", "R", "dkdt"]
                       if self._budget.get(key) is not None]
        plotted_bud = False
        for key in budget_keys:
            if key in self.term_chks and not self.term_chks[key].isChecked():
                continue
            scale, _ = self._scale_for(key)
            vals, dist, xpts, ypts = extract_line_profile(
                self._budget[key] * scale, self._x, self._y,
                sel["x0"], sel["y0"], sel["x1"], sel["y1"],
                mode=lmode, avg_band=avg_band)
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            ax_bud.plot(dist[valid], vals[valid],
                        color=TERMS[key]["color"],
                        label=TERMS[key]["label"],
                        linewidth=1.2)
            plotted_bud = True
            if self._last_line["dist"] is None:
                self._last_line["dist"] = dist
                self._last_line["xpts"] = xpts
                self._last_line["ypts"] = ypts
            self._last_line["means"][key] = vals

        if plotted_bud:
            ax_bud.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_bud.set_ylabel(f"TKE Budget {unit_str}", fontsize=_FONT_AX)
            ax_bud.legend(fontsize=_FONT_LEG)
            ax_bud.grid(True, alpha=0.3)
            ax_bud.tick_params(labelsize=_FONT_TICK)
        else:
            ax_bud.text(0.5, 0.5, "No valid data along line",
                        transform=ax_bud.transAxes, ha="center", va="center",
                        fontsize=11, color="gray")
            ax_bud.set_ylabel(f"TKE Budget {unit_str}", fontsize=_FONT_AX)
            ax_bud.tick_params(labelsize=_FONT_TICK)
        ax_bud.set_xlabel(xlabel, fontsize=_FONT_AX)

        if self.chk_hide_axes.isChecked():
            ax_k.axis('off')
            ax_bud.axis('off')
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.btn_export.setEnabled(True)
        self.lbl_status.setText("Line profile plotted.")

    # ----------------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------------- #

    def _on_export(self):
        _, unit_str = self._scale_for("P")
        settings = {
            "Analysis"     : "TKE Budget",
            "Snapshots"    : self.dataset["Nt"],
            "2D assumption": "dz terms neglected",
            "Smoothing"    : (f"kernel={self.spin_kernel.value()}"
                              if self.chk_smooth.isChecked() else "None"),
            "Normalized"   : unit_str,
        }

        if self._mode == "contour" and self._budget:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export 2D Field", "tke_budget_all.dat",
                "Tecplot DAT (*.dat);;CSV (*.csv)")
            if not path:
                return
            fields, labels = [], []
            for key in TERMS:
                if self._budget.get(key) is not None:
                    s, _ = self._scale_for(key)
                    fields.append(self._budget[key] * s)
                    labels.append(TERMS[key]["label"])
            settings["Analysis"] = "TKE Budget - All Terms"
            export_2d_tecplot(path, self._x, self._y, fields, labels, settings)
            self.lbl_status.setText(
                f"Exported {len(fields)} budget terms to {os.path.basename(path)}")
            return

        elif self._mode == "line" and self._last_line and \
             self._last_line["dist"] is not None:
            try:
                _cn = QApplication.instance()._session_case_name or "Data_1"
            except AttributeError:
                _cn = "Data_1"
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Line Profile", f"{_cn}_tke_budget_line.csv",
                "CSV (*.csv)")
            if not path:
                return
            n = len(self._last_line["means"])
            export_line_csv(path,
                            self._last_line["dist"],
                            self._last_line["xpts"],
                            self._last_line["ypts"],
                            self._last_line["means"],
                            {}, settings)
            self.lbl_status.setText(
                f"Exported {n} budget terms to {os.path.basename(path)}")

    # ----------------------------------------------------------------------- #
    # ComparisonMixin interface
    # ----------------------------------------------------------------------- #

    _TERM_COLS = {"mean_P", "mean_C", "mean_D", "mean_R", "mean_k"}

    def _validate_csv(self, df):
        cols = set(df.columns)
        return "dist_mm" in cols and bool(cols & self._TERM_COLS)

    def _plot_comparison(self, selected_quantities, layout_mode):
        cases = [c for c in self._cases if c["data"] is not None]
        if not cases:
            QMessageBox.information(self, "No Cases", "No cases to compare.")
            return

        quantities = [q for q in selected_quantities
                      if any(q in c["data"].columns for c in cases)]
        if not quantities:
            QMessageBox.warning(self, "No Data",
                "None of the selected terms were found in any case.")
            return

        k_qty       = "mean_k" if "mean_k" in quantities else None
        budget_qtys = [q for q in quantities if q != "mean_k" and q in {
            "mean_P", "mean_C", "mean_D", "mean_R", "mean_dkdt"}]

        self.result_fig.clear()

        def _safe_xy(df, x_col, y_col):
            try:
                x = np.asarray(df[x_col].values, dtype=float)
                y = np.asarray(df[y_col].values, dtype=float)
            except (ValueError, TypeError):
                return None, None
            mask = np.isfinite(x) & np.isfinite(y)
            return (x[mask], y[mask]) if np.any(mask) else (None, None)

        if layout_mode == "overlay":
            ax_k, ax_bud = self.result_fig.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2]})

            # Top: k from all cases, distinguished by case color/linestyle
            for case in cases:
                df = case["data"]
                if k_qty is None or k_qty not in df.columns or "dist_mm" not in df.columns:
                    continue
                xv, yv = _safe_xy(df, "dist_mm", k_qty)
                if xv is None:
                    continue
                ax_k.plot(xv, yv,
                          color=case["color"], linestyle=case["linestyle"],
                          linewidth=1.5, label=case["name"])
            ax_k.set_ylabel("TKE k", fontsize=_FONT_AX)
            ax_k.set_title("TKE Budget Comparison", fontsize=_FONT_AX)
            ax_k.legend(fontsize=_FONT_LEG)
            ax_k.grid(True, alpha=0.3)
            ax_k.tick_params(labelsize=_FONT_TICK)

            # Bottom: budget terms — term color, case linestyle
            multi_case = len(cases) > 1
            for qty in budget_qtys:
                term_key   = qty.replace("mean_", "")
                term_label = TERMS.get(term_key, {}).get("label", qty)
                term_color = TERMS.get(term_key, {}).get("color")
                for case in cases:
                    df = case["data"]
                    if qty not in df.columns or "dist_mm" not in df.columns:
                        continue
                    xv, yv = _safe_xy(df, "dist_mm", qty)
                    if xv is None:
                        continue
                    lbl = f"{term_label} ({case['name']})" if multi_case else term_label
                    ax_bud.plot(xv, yv,
                                color=term_color, linestyle=case["linestyle"],
                                linewidth=1.5, label=lbl)
            ax_bud.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_bud.set_xlabel("dist [mm]", fontsize=_FONT_AX)
            ax_bud.set_ylabel("TKE Budget", fontsize=_FONT_AX)
            ax_bud.legend(fontsize=_FONT_LEG)
            ax_bud.grid(True, alpha=0.3)
            ax_bud.tick_params(labelsize=_FONT_TICK)

        else:  # sidebyside — 2 rows × N cases
            n      = len(cases)
            ax_arr = self.result_fig.subplots(
                2, n, sharex="col", sharey="row",
                gridspec_kw={"height_ratios": [1, 2]},
                squeeze=False)

            for i, case in enumerate(cases):
                ax_k   = ax_arr[0][i]
                ax_bud = ax_arr[1][i]
                df     = case["data"]

                if "dist_mm" not in df.columns:
                    continue

                # Top: k
                if k_qty and k_qty in df.columns:
                    xv, yv = _safe_xy(df, "dist_mm", k_qty)
                    if xv is not None:
                        ax_k.plot(xv, yv,
                                  color=TERMS["k"]["color"],
                                  linewidth=1.5,
                                  label=TERMS["k"]["label"])
                ax_k.set_title(case["name"], fontsize=_FONT_AX)
                if i == 0:
                    ax_k.set_ylabel("TKE k", fontsize=_FONT_AX)
                ax_k.legend(fontsize=_FONT_LEG)
                ax_k.grid(True, alpha=0.3)
                ax_k.tick_params(labelsize=_FONT_TICK)

                # Bottom: budget terms
                for qty in budget_qtys:
                    if qty not in df.columns:
                        continue
                    term_key = qty.replace("mean_", "")
                    xv, yv   = _safe_xy(df, "dist_mm", qty)
                    if xv is None:
                        continue
                    ax_bud.plot(xv, yv,
                                color=TERMS.get(term_key, {}).get("color"),
                                linewidth=1.5,
                                label=TERMS.get(term_key, {}).get("label", qty))
                ax_bud.axhline(0, color="gray", linewidth=0.8,
                               linestyle="--", alpha=0.5)
                ax_bud.set_xlabel("dist [mm]", fontsize=_FONT_AX)
                if i == 0:
                    ax_bud.set_ylabel("TKE Budget", fontsize=_FONT_AX)
                ax_bud.legend(fontsize=_FONT_LEG)
                ax_bud.grid(True, alpha=0.3)
                ax_bud.tick_params(labelsize=_FONT_TICK)

        self.result_fig.tight_layout(pad=1.0)
        self.result_canvas.draw()
