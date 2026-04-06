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
    QDoubleSpinBox, QButtonGroup, QFileDialog, QApplication
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.tke_budget import compute_tke_budget
from core.reynolds_stress import extract_line_profile
from core.export import export_2d_tecplot, export_line_csv
from gui.line_selector import LineSelectorWidget, compute_snapped_line
from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin

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


class TKEBudgetWindow(PickerMixin, QWidget):

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

        self._show_warnings(is_time_resolved, Nt_warn, duration_warn)
        self._build_ui()
        self._draw_field()
        self._connect_mouse()
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    # ----------------------------------------------------------------------- #

    def _show_warnings(self, is_tr, Nt, duration):
        if is_tr:
            if duration < 2.0:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Dataset is {duration:.2f} s (< 2 s).\n"
                    "TKE budget terms may not be converged.")
        else:
            if Nt < 2000:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Only {Nt} snapshots (< 2000 recommended).\n"
                    "Triple correlations (diffusion) are especially sensitive.")

    def _drawing_active(self):
        return self._mode == "line"

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        root = QHBoxLayout(self)
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

    # ----------------------------------------------------------------------- #
    # Field plot
    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds   = self.dataset
        x, y = ds["x"], ds["y"]
        speed = np.sqrt(np.nanmean(ds["U"], axis=2)**2 +
                        np.nanmean(ds["V"], axis=2)**2)
        vf = np.mean(ds["valid"], axis=2)
        speed[vf < 0.5] = np.nan

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
        self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
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
            self.lbl_hint.setText("Select term and click 'Plot'.")
        else:
            self._mode = "line"
            self.contour_grp.setVisible(False)
            self.line_sel.setVisible(True)
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
        ds     = self.dataset
        kernel = self.spin_kernel.value() if self.chk_smooth.isChecked() else 1
        dkdt   = self.chk_dkdt.isChecked() if self.chk_dkdt else False

        self.lbl_status.setText("Busy: computing TKE budget...")
        self.btn_compute.setEnabled(False)
        QApplication.processEvents()

        try:
            self._budget = compute_tke_budget(
                ds["U"], ds["V"], ds["W"], self._x, self._y,
                smooth_kernel=kernel, compute_dkdt=dkdt)

            vf   = np.mean(ds["valid"], axis=2)
            mask = vf < 0.5
            for key in self._budget:
                if self._budget[key] is not None:
                    self._budget[key][mask] = np.nan

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

            if dkdt and self._budget.get("dkdt") is not None:
                if self.combo_term.findData("dkdt") == -1:
                    self.combo_term.addItem(TERMS["dkdt"]["label"], "dkdt")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText(f"Error: {e}")
        finally:
            self.btn_compute.setEnabled(True)

    # ----------------------------------------------------------------------- #
    # Normalization
    # ----------------------------------------------------------------------- #

    def _scale(self):
        if self.chk_norm.isChecked():
            um  = self.spin_um.value()
            L   = self.spin_L.value() / 1000.0
            s   = 1.0 / (um ** 3 / L)
            lbl = f"Um\u00b3/L  (Um={um:.2f} m/s, L={self.spin_L.value():.1f} mm)"
        else:
            s   = 1.0
            lbl = "[m\u00b2/s\u00b3]"
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

    def _plot_contour(self):
        key   = self.combo_term.currentData()
        field = self._budget.get(key)
        if field is None:
            QMessageBox.warning(self, "Not Available",
                f"Term '{key}' is not available.")
            return

        scale, unit_str = self._scale()
        cmap = self.combo_cmap.currentText()
        data = field * scale

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, data, levels=50, cmap=cmap)
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
        sel      = self._selection
        scale, unit_str = self._scale()
        lmode    = self.line_sel.get_mode()
        avg_band = self.line_sel.get_avg_band()

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)

        self._last_line = {"dist": None, "xpts": None, "ypts": None,
                           "means": {}, "unit_str": unit_str}
        plotted = False

        # Plot budget terms (all except k) first, then k on top
        other_keys = [k for k in TERMS if k != "k" and self._budget.get(k) is not None]
        ordered_keys = other_keys + (["k"] if self._budget.get("k") is not None else [])

        for key in ordered_keys:
            field = self._budget[key] * scale
            vals, dist, xpts, ypts = extract_line_profile(
                field, self._x, self._y,
                sel["x0"], sel["y0"], sel["x1"], sel["y1"],
                mode=lmode, avg_band=avg_band)

            valid = np.isfinite(vals)
            if not np.any(valid):
                continue

            if key == "k":
                ax.plot(dist[valid], vals[valid],
                        color=TERMS[key]["color"],
                        label=TERMS[key]["label"],
                        linewidth=2.0,
                        linestyle="--",
                        zorder=10)
            else:
                ax.plot(dist[valid], vals[valid],
                        color=TERMS[key]["color"],
                        label=TERMS[key]["label"],
                        linewidth=1.2)
            plotted = True
            self._last_line["dist"] = dist
            self._last_line["xpts"] = xpts
            self._last_line["ypts"] = ypts
            self._last_line["means"][key] = vals

        if not plotted:
            ax.text(0.5, 0.5, "No valid data along line",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray")
        else:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            xlabel = {"horizontal": "x [mm]",
                      "vertical":   "y [mm]"}.get(lmode, "Distance from origin [mm]")
            ax.set_xlabel(xlabel, fontsize=_FONT_AX)
            ax.set_ylabel(f"TKE Budget {unit_str}", fontsize=_FONT_AX)
            ax.set_title("TKE Budget Profile", fontsize=_FONT_AX)
            ax.legend(fontsize=_FONT_LEG)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=_FONT_TICK)

        if self.chk_hide_axes.isChecked():
            ax.axis('off')
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.btn_export.setEnabled(True)
        self.lbl_status.setText("Line profile plotted.")

    # ----------------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------------- #

    def _on_export(self):
        scale, unit_str = self._scale()
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
                    fields.append(self._budget[key] * scale)
                    labels.append(TERMS[key]["label"])
            settings["Analysis"] = "TKE Budget - All Terms"
            export_2d_tecplot(path, self._x, self._y, fields, labels, settings)
            self.lbl_status.setText(
                f"Exported {len(fields)} budget terms to {os.path.basename(path)}")
            return

        elif self._mode == "line" and self._last_line and \
             self._last_line["dist"] is not None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Line Profile", "tke_budget_line.csv",
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
