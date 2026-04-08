"""
gui/tke_window.py
-----------------
TKE analysis window.

TKE modes:
  - Full 3C: k = 0.5*(uu+vv+ww)  -- only if W available
  - 2D:      k = 0.5*(uu+vv)      -- always available

Plot modes:
  - 2D contour map
  - Line profile (with normalization option)
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGroupBox, QPushButton, QRadioButton, QCheckBox,
    QSizePolicy, QMessageBox, QSplitter, QComboBox,
    QDoubleSpinBox, QButtonGroup, QFileDialog
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from matplotlib.figure import Figure

from core.reynolds_stress import compute_reynolds_stresses, extract_line_profile, compute_tke_std
from gui.line_selector import LineSelectorWidget, compute_snapped_line
from core.export import export_2d_tecplot, export_line_csv


class TKEWindow(PickerMixin, QWidget):

    def __init__(self, dataset, is_time_resolved=False, Nt_warn=2000,
                 duration_warn=9999, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Turbulent Kinetic Energy Analysis")
        self.resize(1400, 780)

        self._mode        = "contour"
        self._press_xy    = None
        self._line_artist = None
        self._selection   = None

        self._show_convergence_warning(is_time_resolved, Nt_warn, duration_warn)

        # Pre-compute stresses
        self._stresses, _ = compute_reynolds_stresses(
            dataset["U"], dataset["V"], dataset["W"]
        )

        # Compute TKE variants
        uu = self._stresses["uu"]
        vv = self._stresses["vv"]
        ww = self._stresses["ww"]

        self._k2d = 0.5 * (uu + vv)
        self._k3d = 0.5 * (uu + vv + ww) if ww is not None else None
        self._k2d_std = compute_tke_std(dataset["U"], dataset["V"], dataset["W"], mode="2d")
        self._k3d_std = compute_tke_std(dataset["U"], dataset["V"], dataset["W"], mode="3d") if ww is not None else None

        self._build_ui()
        self._draw_field()
        self._connect_mouse()
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    # ----------------------------------------------------------------------- #

    def _show_convergence_warning(self, is_tr, Nt, duration):
        if is_tr:
            if duration < 2.0:
                QMessageBox.warning(
                    self, "Convergence Warning",
                    f"Dataset is {duration:.2f} s (< 2 s).\n"
                    "TKE statistics may not be converged."
                )
        else:
            if Nt < 2000:
                QMessageBox.warning(
                    self, "Convergence Warning",
                    f"Only {Nt} snapshots (< 2000 recommended).\n"
                    "TKE statistics may not be converged."
                )

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left: field + controls ----
        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        self.field_fig    = Figure(figsize=(6, 4), tight_layout=True)
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)

        # Mode
        mode_grp = QGroupBox("Plot Mode")
        mode_lay = QHBoxLayout(mode_grp)
        self.rb_contour = QRadioButton("2D Contour Map")
        self.rb_line    = QRadioButton("Line Profile")
        self.rb_contour.setChecked(True)
        self.rb_contour.toggled.connect(self._on_mode_changed)
        mode_lay.addWidget(self.rb_contour)
        mode_lay.addWidget(self.rb_line)
        ll.addWidget(mode_grp)

        # Line selection mode + spatial averaging
        self.line_sel = LineSelectorWidget(show_avg=True)
        self.line_sel.setVisible(False)
        ll.addWidget(self.line_sel)

        self.lbl_hint = QLabel("Select TKE type and click 'Plot'.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        # Options
        opt_grp = QGroupBox("Options")
        opt_lay = QVBoxLayout(opt_grp)

        # TKE type
        tke_row = QHBoxLayout()
        tke_row.addWidget(QLabel("TKE type:"))
        self.combo_tke = QComboBox()
        self.combo_tke.addItem("2D  k = 0.5(uu+vv)", "2d")
        if self._k3d is not None:
            self.combo_tke.addItem("3C  k = 0.5(uu+vv+ww)", "3d")
        tke_row.addWidget(self.combo_tke)
        opt_lay.addLayout(tke_row)

        # Normalization
        norm_row = QHBoxLayout()
        self.chk_norm = QCheckBox("Normalize by Um²")
        self.chk_norm.setChecked(False)
        self.spin_um = QDoubleSpinBox()
        self.spin_um.setRange(0.001, 1000.0)
        self.spin_um.setValue(1.0)
        self.spin_um.setDecimals(3)
        self.spin_um.setSingleStep(0.1)
        self.spin_um.setToolTip("Mean inlet velocity Um [m/s]")
        norm_row.addWidget(self.chk_norm)
        norm_row.addWidget(QLabel("Um [m/s]:"))
        norm_row.addWidget(self.spin_um)
        opt_lay.addLayout(norm_row)

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["hot_r", "RdBu_r", "viridis", "plasma", "jet"])
        cmap_row.addWidget(self.combo_cmap)
        opt_lay.addLayout(cmap_row)

        ll.addWidget(opt_grp)

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self._on_plot)
        ll.addWidget(self.btn_plot)

        self.chk_std_band = QCheckBox("Show ±1σ band on line plot")
        self.chk_std_band.setChecked(True)
        ll.addWidget(self.chk_std_band)

        self.btn_export = QPushButton("Export Data...")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        ll.addWidget(self.btn_export)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_status)

        # ---- Right: result ----
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self.result_fig    = Figure(figsize=(6, 5), tight_layout=True)
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds    = self.dataset
        x, y  = ds["x"], ds["y"]
        speed = np.sqrt(np.nanmean(ds["U"], axis=2)**2 +
                        np.nanmean(ds["V"], axis=2)**2)
        valid_frac = np.mean(ds["valid"], axis=2)
        speed[valid_frac < 0.5] = np.nan

        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        cf = self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
        self.field_fig.colorbar(cf, ax=self.field_ax, label="Mean |V| [m/s]", shrink=0.8)
        self.field_ax.set_xlabel("x [mm]")
        self.field_ax.set_ylabel("y [mm]")
        self.field_ax.set_title("Draw a line for profile mode")
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()
        self._x = x
        self._y = y
        self._last_field_values = speed

    # ----------------------------------------------------------------------- #
    # Mouse
    # ----------------------------------------------------------------------- #

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_mode_changed(self):
        if self.rb_contour.isChecked():
            self._mode = "contour"
            self.btn_plot.setText("Plot")
            self.lbl_hint.setText("Select TKE type and click 'Plot'.")
            self.line_sel.setVisible(False)
        else:
            self._mode = "line"
            self.btn_plot.setText("Plot")
            self.line_sel.setVisible(True)
            self.lbl_hint.setText(self.line_sel.hint_text())
        self._clear_graphics()
        self._selection = None

    def _on_press(self, event):
        if event.inaxes != self.field_ax or self._mode != "line":
            return
        if self._toolbar_active(self.field_toolbar):
            return
        self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.field_ax:
            return
        x0, y0 = self._press_xy
        lmode = self.line_sel.get_mode() if self._mode == "line" else "free"
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, event.xdata, event.ydata, lmode
        )
        self._clear_graphics()
        self._line_artist, = self.field_ax.plot(
            [lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10
        )
        self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None or self._mode != "line":
            return
        if event.inaxes != self.field_ax:
            self._press_xy = None
            return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._press_xy = None
        lmode = self.line_sel.get_mode() if self._mode == "line" else "free"
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, x1, y1, lmode
        )
        if abs(lx1-lx0) < 0.1 and abs(ly1-ly0) < 0.1:
            self.lbl_hint.setText("Line too short -- try again.")
            return
        self._selection = {"x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1}
        self.lbl_hint.setText(
            f"Line ({lmode}): ({lx0:.1f},{ly0:.1f}) -> ({lx1:.1f},{ly1:.1f}) mm"
        )

    def _clear_graphics(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None
        self.field_canvas.draw()

    # ----------------------------------------------------------------------- #

    def _get_tke_field(self):
        tke_type = self.combo_tke.currentData()
        if tke_type == "3d" and self._k3d is not None:
            k = self._k3d.copy()
            label = r"$k = \frac{1}{2}(\langle u'u'\rangle+\langle v'v'\rangle+\langle w'w'\rangle)$"
        else:
            k = self._k2d.copy()
            label = r"$k = \frac{1}{2}(\langle u'u'\rangle+\langle v'v'\rangle)$"

        if self.chk_norm.isChecked():
            um = self.spin_um.value()
            k  = k / (um**2)
            label += f" / Um²  (Um={um:.2f} m/s)"
        else:
            label += "  [m²/s²]"

        return k, label

    # ----------------------------------------------------------------------- #

    def _on_plot(self):
        if self._mode == "contour":
            self._plot_contour()
        else:
            if self._selection is None:
                QMessageBox.information(self, "No Line",
                    "Please draw a line on the field first.")
                return
            self._plot_line()

    def _plot_contour(self):
        k, label = self._get_tke_field()

        valid_frac = np.mean(self.dataset["valid"], axis=2)
        k[valid_frac < 0.5] = np.nan

        cmap = self.combo_cmap.currentText()

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, k, levels=50, cmap=cmap)
        self.result_fig.colorbar(cf, ax=ax, label=label, shrink=0.8)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Turbulent Kinetic Energy")
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        self._last_contour_k = k
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.btn_export.setEnabled(True)
        self.lbl_status.setText("TKE contour plotted.")

    def _plot_line(self):
        k, label = self._get_tke_field()
        tke_type  = self.combo_tke.currentData()
        std_field = self._k3d_std if tke_type == "3d" and self._k3d_std is not None else self._k2d_std
        scale     = 1.0 / (self.spin_um.value()**2) if self.chk_norm.isChecked() else 1.0
        sel       = self._selection
        show_band = self.chk_std_band.isChecked()

        lmode    = self.line_sel.get_mode()
        avg_band = self.line_sel.get_avg_band()
        vals, dist, xpts, ypts = extract_line_profile(
            k, self._x, self._y,
            sel["x0"], sel["y0"], sel["x1"], sel["y1"],
            mode=lmode, avg_band=avg_band
        )
        std_vals, _, _, _ = extract_line_profile(
            std_field * scale, self._x, self._y,
            sel["x0"], sel["y0"], sel["x1"], sel["y1"],
            mode=lmode, avg_band=avg_band
        )

        valid = np.isfinite(vals)
        self._last_line_data = {"dist": dist, "xpts": xpts, "ypts": ypts,
                                 "vals": vals, "std_vals": std_vals, "label": label}

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)

        if np.any(valid):
            ax.plot(dist[valid], vals[valid], color="tab:blue",
                    linewidth=1.8, label="TKE")
            if show_band:
                ax.fill_between(dist[valid],
                                vals[valid] - std_vals[valid],
                                vals[valid] + std_vals[valid],
                                alpha=0.2, color="tab:blue")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlabel("Distance along line [mm]")
            ax.set_ylabel(label)
            ax.set_title("TKE Profile Along Line")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            self.btn_export.setEnabled(True)
        else:
            ax.text(0.5, 0.5, "No valid data along line",
                    transform=ax.transAxes, ha="center", va="center")

        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.lbl_status.setText("TKE line profile plotted.")

    def _on_export(self):
        k, label = self._get_tke_field()
        tke_type = self.combo_tke.currentData()
        scale    = 1.0 / (self.spin_um.value()**2) if self.chk_norm.isChecked() else 1.0
        is_line  = self._mode == "line"

        settings = {
            "Analysis"    : f"TKE - {'3C' if tke_type == '3d' else '2D'}",
            "Snapshots"   : self.dataset["Nt"],
            "Grid"        : f"{self.dataset['nx']} x {self.dataset['ny']}",
            "Normalized"  : f"Yes, Um={self.spin_um.value():.3f} m/s" if self.chk_norm.isChecked() else "No",
        }

        if is_line:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export TKE Line Profile", "tke_line.csv", "CSV Files (*.csv)"
            )
            if not path:
                return
            ld = self._last_line_data
            export_line_csv(path, ld["dist"], ld["xpts"], ld["ypts"],
                            {"TKE": ld["vals"]}, {"TKE": ld["std_vals"]}, settings)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export TKE 2D Field", "tke_2d.dat",
                "Tecplot DAT (*.dat);;CSV Files (*.csv)"
            )
            if not path:
                return
            field = self._last_contour_k.copy()
            valid_frac = np.mean(self.dataset["valid"], axis=2)
            field[valid_frac < 0.5] = np.nan
            export_2d_tecplot(path, self._x, self._y, [field], [label], settings)

        self.lbl_status.setText(f"Exported to {path}")
