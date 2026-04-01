"""
gui/tke_budget_window.py
------------------------
TKE budget analysis window.

Terms: k, P (production), C (convection), D (turbulent diffusion), R (residual)
Optional: dkdt (TR data only)

Plot modes:
  - 2D contour: dropdown to select term
  - Line profile: all terms overlaid, free/horizontal/vertical with spatial avg
  - ROI: line profiles averaged over ROI

Export: all line terms in one CSV, 2D contour as Tecplot DAT
"""

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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRect

from core.tke_budget import compute_tke_budget
from core.reynolds_stress import extract_line_profile
from core.export import export_2d_tecplot, export_line_csv
from gui.line_selector import LineSelectorWidget, compute_snapped_line
from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin


# Term metadata
TERMS = {
    "k"    : {"label": "TKE  k",               "color": "tab:blue"},
    "P"    : {"label": "Production  P",         "color": "tab:red"},
    "C"    : {"label": "Convection  C",         "color": "tab:orange"},
    "D"    : {"label": "Turb. Diffusion  D",    "color": "tab:green"},
    "R"    : {"label": "Residual  R",           "color": "tab:purple"},
    "dkdt" : {"label": "∂k/∂t  (TR only)",     "color": "tab:brown"},
}
CMAPS = {"k": "hot_r", "P": "RdBu_r", "C": "RdBu_r",
         "D": "RdBu_r", "R": "RdBu_r", "dkdt": "RdBu_r"}


class TKEBudgetWindow(PickerMixin, QWidget):

    def __init__(self, dataset, is_time_resolved=False,
                 Nt_warn=2000, duration_warn=9999, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self._is_tr  = is_time_resolved
        self.setWindowTitle("TKE Budget Analysis")
        self.resize(1700, 900)

        self._mode       = "contour"
        self._press_xy   = None
        self._artist     = None
        self._selection  = None
        self._budget     = None   # computed budget dict
        self._last_line  = None   # for export

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

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- LEFT: field + controls ----
        left = QWidget()
        left.setMinimumWidth(480); left.setMaximumWidth(580)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)

        self.field_fig    = Figure()
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas, stretch=5)

        # Plot mode
        pm_grp = QGroupBox("Plot Mode")
        pm_lay = QHBoxLayout(pm_grp)
        self.rb_contour = QRadioButton("2D Contour")
        self.rb_line    = QRadioButton("Line Profile")
        self.rb_contour.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_contour); bg.addButton(self.rb_line)
        self.rb_contour.toggled.connect(self._on_mode_changed)
        pm_lay.addWidget(self.rb_contour); pm_lay.addWidget(self.rb_line)
        ll.addWidget(pm_grp, stretch=0)

        # Contour term selector
        self.contour_grp = QGroupBox("Term to Display")
        cl = QHBoxLayout(self.contour_grp)
        cl.addWidget(QLabel("Term:"))
        self.combo_term = QComboBox()
        for key, meta in TERMS.items():
            if key == "dkdt" and not self._is_tr:
                continue
            self.combo_term.addItem(meta["label"], key)
        cl.addWidget(self.combo_term)
        ll.addWidget(self.contour_grp, stretch=0)

        # Line selector widget (hidden in contour mode)
        self.line_sel = LineSelectorWidget(show_avg=True)
        self.line_sel.setVisible(False)
        ll.addWidget(self.line_sel, stretch=0)

        self.lbl_hint = QLabel("Click 'Compute' to calculate all budget terms first.")
        self.lbl_hint.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint, stretch=0)

        # Computation parameters
        comp_grp = QGroupBox("Parameters")
        cp = QVBoxLayout(comp_grp)

        # Normalization
        n_row = QHBoxLayout()
        self.chk_norm = QCheckBox("Normalize by Um³/L")
        self.chk_norm.setChecked(False)
        n_row.addWidget(self.chk_norm)
        cp.addLayout(n_row)

        um_row = QHBoxLayout()
        um_row.addWidget(QLabel("Um [m/s]:"))
        self.spin_um = QDoubleSpinBox()
        self.spin_um.setRange(0.001, 1000); self.spin_um.setValue(1.0)
        self.spin_um.setDecimals(3); self.spin_um.setSingleStep(0.1)
        um_row.addWidget(self.spin_um)
        cp.addLayout(um_row)

        L_row = QHBoxLayout()
        L_row.addWidget(QLabel("L [mm]:"))
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.001, 10000); self.spin_L.setValue(7.5)
        self.spin_L.setDecimals(3); self.spin_L.setSingleStep(0.5)
        L_row.addWidget(self.spin_L)
        cp.addLayout(L_row)

        # Smoothing for triple correlations
        sm_row = QHBoxLayout()
        self.chk_smooth = QCheckBox("Smooth triple corr. (kernel):")
        self.chk_smooth.setChecked(True)
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 15); self.spin_kernel.setValue(3)
        self.spin_kernel.setSingleStep(2)
        sm_row.addWidget(self.chk_smooth); sm_row.addWidget(self.spin_kernel)
        cp.addLayout(sm_row)

        # TR: temporal derivative option
        if self._is_tr:
            self.chk_dkdt = QCheckBox("Compute ∂k/∂t (time-resolved)")
            self.chk_dkdt.setChecked(True)
            cp.addWidget(self.chk_dkdt)
        else:
            self.chk_dkdt = None

        ll.addWidget(comp_grp, stretch=0)

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["RdBu_r", "hot_r", "viridis", "plasma", "seismic"])
        cmap_row.addWidget(self.combo_cmap)
        ll.addLayout(cmap_row)

        self.btn_compute = QPushButton("Compute Budget")
        self.btn_compute.clicked.connect(self._on_compute)
        ll.addWidget(self.btn_compute, stretch=0)

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.setEnabled(False)
        self.btn_plot.clicked.connect(self._on_plot)
        ll.addWidget(self.btn_plot, stretch=0)

        self.btn_export = QPushButton("Export Data...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        ll.addWidget(self.btn_export, stretch=0)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status, stretch=0)

        # ---- RIGHT: result canvas ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.result_fig    = Figure()
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = NavToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([920, 780])

    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds   = self.dataset
        x, y = ds["x"], ds["y"]
        speed = np.sqrt(np.nanmean(ds["U"], axis=2)**2 +
                        np.nanmean(ds["V"], axis=2)**2)
        vf = np.mean(ds["valid"], axis=2)
        speed[vf < 0.5] = np.nan

        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
        self.field_ax.set_xlabel("x [mm]", fontsize=9)
        self.field_ax.set_ylabel("y [mm]", fontsize=9)
        self.field_ax.set_title("Compute budget first, then draw selection", fontsize=8)
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_ax.tick_params(labelsize=8)
        self.field_fig.tight_layout(pad=0.3)
        self.field_canvas.draw()
        self._x = x; self._y = y
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
            self.lbl_hint.setText(self.line_sel.hint_text() if self._budget else
                                  "Compute budget first, then draw a line.")
        self._clear_artist()
        self._selection = None

    # ----------------------------------------------------------------------- #
    # Mouse (line drawing -- only active after budget is computed)
    # ----------------------------------------------------------------------- #

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.field_ax: return
        if self._toolbar_active(self.field_toolbar): return
        if self._mode != "line" or self._budget is None: return
        self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.field_ax: return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None; return
        x0, y0 = self._press_xy
        lmode = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, event.xdata, event.ydata, lmode)
        self._clear_artist()
        ln, = self.field_ax.plot([lx0,lx1],[ly0,ly1],"r-",linewidth=2,zorder=10)
        self._artist = ln
        self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None: return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None; return
        if event.inaxes != self.field_ax:
            self._press_xy = None; return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._press_xy = None
        lmode = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, x1, y1, lmode)
        if abs(lx1-lx0)<0.1 and abs(ly1-ly0)<0.1:
            self.lbl_hint.setText("Line too short -- try again."); return
        self._selection = {"x0":lx0,"y0":ly0,"x1":lx1,"y1":ly1}
        self.lbl_hint.setText(f"Line ({lmode}): ({lx0:.1f},{ly0:.1f})->({lx1:.1f},{ly1:.1f}) mm")
        # Redraw committed
        self._clear_artist()
        ln, = self.field_ax.plot([lx0,lx1],[ly0,ly1],"r-",linewidth=2,zorder=10)
        self._artist = ln
        self.field_canvas.draw()
        self.btn_plot.setEnabled(True)

    def _clear_artist(self):
        if self._artist is not None:
            try: self._artist.remove()
            except: pass
            self._artist = None
        self.field_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Compute budget
    # ----------------------------------------------------------------------- #

    def _on_compute(self):
        ds = self.dataset
        kernel = self.spin_kernel.value() if self.chk_smooth.isChecked() else 1
        dkdt   = self.chk_dkdt.isChecked() if self.chk_dkdt else False

        self.lbl_status.setText("Busy: computing TKE budget...")
        self.btn_compute.setEnabled(False)
        QApplication.processEvents()

        try:
            self._budget = compute_tke_budget(
                ds["U"], ds["V"], ds["W"], self._x, self._y,
                smooth_kernel=kernel,
                compute_dkdt=dkdt
            )
            # Mask invalid regions
            valid_frac = np.mean(ds["valid"], axis=2)
            mask = valid_frac < 0.5
            for key in self._budget:
                if self._budget[key] is not None:
                    self._budget[key][mask] = np.nan

            self.lbl_status.setText("Budget computed. Now select plot mode and draw selection.")
            self.btn_plot.setEnabled(True)
            self.lbl_hint.setText("Draw a line on the field (line mode) or click 'Plot' (contour).")

            # Update dkdt in term dropdown if computed
            if dkdt and self._budget.get("dkdt") is not None:
                if self.combo_term.findData("dkdt") == -1:
                    self.combo_term.addItem(TERMS["dkdt"]["label"], "dkdt")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText(f"Error: {e}")
        finally:
            self.btn_compute.setEnabled(True)

    # ----------------------------------------------------------------------- #
    # Normalization helper
    # ----------------------------------------------------------------------- #

    def _scale(self):
        if self.chk_norm.isChecked():
            um = self.spin_um.value()
            L  = self.spin_L.value() / 1000.0   # mm -> m
            s  = 1.0 / (um**3 / L)
            lbl = f"Um³/L  (Um={um:.2f} m/s, L={self.spin_L.value():.1f} mm)"
        else:
            s   = 1.0
            lbl = "[m²/s³]"
        return s, lbl

    # ----------------------------------------------------------------------- #
    # Plot
    # ----------------------------------------------------------------------- #

    def _on_plot(self):
        if self._budget is None:
            QMessageBox.information(self, "No Data",
                "Please compute the budget first."); return

        if self._mode == "contour":
            self._plot_contour()
        else:
            if self._selection is None:
                QMessageBox.information(self, "No Selection",
                    "Please draw a line on the field."); return
            self._plot_line()

    def _plot_contour(self):
        key   = self.combo_term.currentData()
        field = self._budget.get(key)
        if field is None:
            QMessageBox.warning(self, "Not Available",
                f"Term '{key}' is not available."); return

        scale, unit_str = self._scale()
        cmap = self.combo_cmap.currentText()
        data = field * scale

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, data, levels=50, cmap=cmap)
        self.result_fig.colorbar(cf, ax=ax,
                                 label=f"{TERMS[key]['label']} {unit_str}",
                                 shrink=0.8)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(TERMS[key]["label"]); ax.set_aspect("equal")
        ax.set_facecolor("white")
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
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

        keys_to_plot = [k for k in TERMS if self._budget.get(k) is not None]

        for key in keys_to_plot:
            field = self._budget[key] * scale
            vals, dist, xpts, ypts = extract_line_profile(
                field, self._x, self._y,
                sel["x0"], sel["y0"], sel["x1"], sel["y1"],
                mode=lmode, avg_band=avg_band
            )
            valid = np.isfinite(vals)
            if not np.any(valid): continue

            ax.plot(dist[valid], vals[valid],
                    color=TERMS[key]["color"],
                    label=TERMS[key]["label"],
                    linewidth=1.5)

            self._last_line["dist"] = dist
            self._last_line["xpts"] = xpts
            self._last_line["ypts"] = ypts
            self._last_line["means"][key] = vals

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Distance along line [mm]")
        ax.set_ylabel(f"TKE Budget {unit_str}")
        ax.set_title("TKE Budget Profile -- check residual for balance")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
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
            "Smoothing"    : f"kernel={self.spin_kernel.value()}" if self.chk_smooth.isChecked() else "None",
            "Normalized"   : unit_str,
        }

        if self._mode == "contour" and self._budget:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export 2D Field", "tke_budget_2d.dat",
                "Tecplot DAT (*.dat);;CSV (*.csv)")
            if not path: return
            key   = self.combo_term.currentData()
            field = self._budget[key] * scale
            export_2d_tecplot(path, self._x, self._y,
                              [field], [TERMS[key]["label"]], settings)

        elif self._mode == "line" and self._last_line and self._last_line["dist"] is not None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Line Profile", "tke_budget_line.csv",
                "CSV (*.csv)")
            if not path: return
            export_line_csv(path,
                            self._last_line["dist"],
                            self._last_line["xpts"],
                            self._last_line["ypts"],
                            self._last_line["means"],
                            {},   # no std dev for budget terms
                            settings)
        self.lbl_status.setText(f"Exported.")
