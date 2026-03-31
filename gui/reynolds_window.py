"""
gui/reynolds_window.py
----------------------
Reynolds stress analysis window.

Two modes:
  1. Contour map  -- 2D filled contour of any Rij component
  2. Line profile -- user draws a line, plots selected Rij along it

Options:
  - Scale by Um^2 (user-entered)
  - Overlay multiple components on line plot
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGroupBox, QPushButton, QRadioButton, QCheckBox,
    QSizePolicy, QMessageBox, QSplitter, QComboBox,
    QDoubleSpinBox, QButtonGroup
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from core.reynolds_stress import compute_reynolds_stresses, extract_line_profile


# Component display names and labels
COMP_LABELS = {
    "uu": r"$\langle u'u' \rangle$",
    "vv": r"$\langle v'v' \rangle$",
    "ww": r"$\langle w'w' \rangle$",
    "uv": r"$\langle u'v' \rangle$",
    "uw": r"$\langle u'w' \rangle$",
    "vw": r"$\langle v'w' \rangle$",
}
COMP_COLORS = {
    "uu": "tab:blue", "vv": "tab:orange", "ww": "tab:green",
    "uv": "tab:red",  "uw": "tab:purple", "vw": "tab:brown",
}


class ReynoldsWindow(QWidget):

    def __init__(self, dataset, is_time_resolved=False, Nt_warn=2000,
                 duration_warn=2.0, parent=None):
        super().__init__(parent)
        self.dataset   = dataset
        self.setWindowTitle("Reynolds Stress Analysis")
        self.resize(1400, 780)

        self._mode       = "contour"
        self._press_xy   = None
        self._line_artist = None
        self._selection  = None

        # Convergence warning
        self._show_convergence_warning(is_time_resolved, Nt_warn, duration_warn)

        # Pre-compute stresses once
        self._stresses, self._k = compute_reynolds_stresses(
            dataset["U"], dataset["V"], dataset["W"]
        )
        self._available = [k for k, v in self._stresses.items() if v is not None]

        self._build_ui()
        self._draw_field()
        self._connect_mouse()

    # ----------------------------------------------------------------------- #

    def _show_convergence_warning(self, is_tr, Nt, duration):
        if is_tr:
            if duration < 2.0:
                QMessageBox.warning(
                    self, "Convergence Warning",
                    f"Dataset is {duration:.2f} s (< 2 s).\n"
                    "Reynolds stress statistics may not be converged."
                )
        else:
            if Nt < 2000:
                QMessageBox.warning(
                    self, "Convergence Warning",
                    f"Only {Nt} snapshots loaded (< 2000 recommended).\n"
                    "Reynolds stress statistics may not be converged."
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
        self.field_toolbar = NavToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)

        # Mode selector
        mode_grp = QGroupBox("Plot Mode")
        mode_lay = QHBoxLayout(mode_grp)
        self.rb_contour = QRadioButton("2D Contour Map")
        self.rb_line    = QRadioButton("Line Profile")
        self.rb_contour.setChecked(True)
        self.rb_contour.toggled.connect(self._on_mode_changed)
        mode_lay.addWidget(self.rb_contour)
        mode_lay.addWidget(self.rb_line)
        ll.addWidget(mode_grp)

        self.lbl_hint = QLabel("Select component and click 'Plot Contour'.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        # Options group
        opt_grp = QGroupBox("Options")
        opt_lay = QVBoxLayout(opt_grp)

        # Component selector for contour mode
        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("Component:"))
        self.combo_comp = QComboBox()
        for c in self._available:
            self.combo_comp.addItem(COMP_LABELS.get(c, c), c)
        comp_row.addWidget(self.combo_comp)
        opt_lay.addLayout(comp_row)

        # Scaling
        scale_row = QHBoxLayout()
        self.chk_scale = QCheckBox("Scale by Um²")
        self.chk_scale.setChecked(False)
        self.spin_um = QDoubleSpinBox()
        self.spin_um.setRange(0.001, 1000.0)
        self.spin_um.setValue(1.0)
        self.spin_um.setDecimals(3)
        self.spin_um.setSingleStep(0.1)
        self.spin_um.setToolTip("Mean inlet velocity Um [m/s]")
        scale_row.addWidget(self.chk_scale)
        scale_row.addWidget(QLabel("Um [m/s]:"))
        scale_row.addWidget(self.spin_um)
        opt_lay.addLayout(scale_row)

        # Colormap for contour
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["RdBu_r", "hot_r", "viridis", "plasma", "seismic"])
        cmap_row.addWidget(self.combo_cmap)
        opt_lay.addLayout(cmap_row)

        ll.addWidget(opt_grp)

        # Line plot component checkboxes (shown when in line mode)
        self.line_comp_grp = QGroupBox("Components to overlay (line mode)")
        self.line_comp_grp.setVisible(False)
        lc_lay = QHBoxLayout(self.line_comp_grp)
        self.comp_chks = {}
        for c in self._available:
            chk = QCheckBox(COMP_LABELS.get(c, c))
            chk.setChecked(True)
            self.comp_chks[c] = chk
            lc_lay.addWidget(chk)
        ll.addWidget(self.line_comp_grp)

        self.btn_plot = QPushButton("Plot Contour")
        self.btn_plot.clicked.connect(self._on_plot)
        ll.addWidget(self.btn_plot)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_status)

        # ---- Right: result canvas ----
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self.result_fig    = Figure(figsize=(6, 5), tight_layout=True)
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = NavToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds = self.dataset
        x  = ds["x"]
        y  = ds["y"]
        speed = np.sqrt(np.nanmean(ds["U"], axis=2)**2 +
                        np.nanmean(ds["V"], axis=2)**2)
        if ds["W"] is not None:
            speed = np.sqrt(speed**2 + np.nanmean(ds["W"], axis=2)**2)
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
        self._x = x
        self._y = y

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
            self.btn_plot.setText("Plot Contour")
            self.lbl_hint.setText("Select component and click 'Plot Contour'.")
            self.line_comp_grp.setVisible(False)
            self.combo_comp.setVisible(True)
        else:
            self._mode = "line"
            self.btn_plot.setText("Plot Line Profile")
            self.lbl_hint.setText("Click and drag to draw a line on the field.")
            self.line_comp_grp.setVisible(True)
            self.combo_comp.setVisible(False)
        self._clear_graphics()
        self._selection = None

    def _on_press(self, event):
        if event.inaxes != self.field_ax or self._mode != "line":
            return
        self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.field_ax:
            return
        x0, y0 = self._press_xy
        self._clear_graphics()
        self._line_artist, = self.field_ax.plot(
            [x0, event.xdata], [y0, event.ydata],
            "r-", linewidth=2, zorder=10
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
        if abs(x1-x0) < 0.1 and abs(y1-y0) < 0.1:
            self.lbl_hint.setText("Line too short -- try again.")
            return
        self._selection = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        self.lbl_hint.setText(
            f"Line: ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f}) mm  "
            "-- click 'Plot Line Profile'"
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
    # Scaling helper
    # ----------------------------------------------------------------------- #

    def _scale_factor(self):
        if self.chk_scale.isChecked():
            um = self.spin_um.value()
            return 1.0 / (um**2), f"/ Um² (Um={um:.2f} m/s)"
        return 1.0, "[m²/s²]"

    # ----------------------------------------------------------------------- #
    # Plot
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
        comp  = self.combo_comp.currentData()
        field = self._stresses[comp].copy()

        scale, unit_str = self._scale_factor()
        field = field * scale

        valid_frac = np.mean(self.dataset["valid"], axis=2)
        field[valid_frac < 0.5] = np.nan

        cmap  = self.combo_cmap.currentText()
        label = COMP_LABELS.get(comp, comp) + " " + unit_str

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, field, levels=50, cmap=cmap)
        self.result_fig.colorbar(cf, ax=ax, label=label, shrink=0.8)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"Reynolds Stress: {COMP_LABELS.get(comp, comp)}")
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        self.result_canvas.draw()
        self.lbl_status.setText(f"Plotted contour: {comp}")

    def _plot_line(self):
        sel   = self._selection
        scale, unit_str = self._scale_factor()

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)

        plotted = False
        for comp, chk in self.comp_chks.items():
            if not chk.isChecked():
                continue
            field = self._stresses[comp]
            if field is None:
                continue

            vals, dist, _, _ = extract_line_profile(
                field * scale,
                self._x, self._y,
                sel["x0"], sel["y0"], sel["x1"], sel["y1"]
            )

            valid = np.isfinite(vals)
            if not np.any(valid):
                continue

            ax.plot(dist[valid], vals[valid],
                    color=COMP_COLORS[comp],
                    label=COMP_LABELS.get(comp, comp),
                    linewidth=1.5)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No valid data along line",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlabel("Distance along line [mm]")
            ax.set_ylabel(f"Reynolds Stress {unit_str}")
            ax.set_title("Reynolds Stress Profile Along Line")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        self.result_canvas.draw()
        self.lbl_status.setText("Plotted line profile.")
