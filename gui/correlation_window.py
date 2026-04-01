"""
gui/correlation_window.py
-------------------------
Correlation Analysis window for uPrime.

Three sub-tabs:
  1. Spatial   -- point: 2D map + 1D slices  |  ROI: two 1D curves (x and y)
  2. Temporal  -- R(tau) curve  [greyed out if non-TR]
  3. Scales    -- L_x, L_y (mm), T (ms), lambda_t (ms) as text readout

Reference modes
  Point : click on field.  Optional 3x3 kernel toggle.
  ROI   : right-click drag on field (rectangle).

Integral scales computed automatically after each correlation and
displayed in a compact panel below each plot.

Colorbar: flat ends (extend='neither'), consistent with all other windows.
Aspect ratio: equal on all 2D plots.
Font sizes / colormap: matches reynolds_window, tke_budget_window, etc.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QComboBox, QSpinBox, QCheckBox,
    QSizePolicy, QMessageBox, QSplitter, QTabWidget,
    QGridLayout, QApplication, QRadioButton, QButtonGroup,
    QFileDialog, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from core.export import export_2d_tecplot, export_line_csv

from core.two_point_corr import (
    nearest_grid_point,
    compute_spatial_correlation_point,
    compute_spatial_scales_point,
    compute_spatial_correlation_roi,
    compute_temporal_correlation,
    compute_time_scales,
    _integral_to_zero,
)

_CMAP_DIV  = "RdBu_r"
_FONT_AX   = 9
_FONT_TICK = 8
_FONT_LEG  = 8
_SCALE_FONT = QFont("Arial", 10)


# ---------------------------------------------------------------------------
# Small helper: horizontal separator line
# ---------------------------------------------------------------------------

def _hline():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color:#555;")
    return f


# ---------------------------------------------------------------------------
# Scale readout widget -- shows Lx, Ly, T, lambda_t
# ---------------------------------------------------------------------------

class ScaleReadout(QWidget):
    """Compact key=value grid shown below plots."""

    def __init__(self, labels, parent=None):
        """labels : list of str, e.g. ['Lx', 'Ly']"""
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(16)
        self._labels = labels
        self._vals   = {}
        for lbl in labels:
            vl = QVBoxLayout()
            name = QLabel(lbl)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name.setStyleSheet("color:#aaa; font-size:9px;")
            val = QLabel("--")
            val.setFont(_SCALE_FONT)
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet("font-weight:bold;")
            vl.addWidget(name)
            vl.addWidget(val)
            lay.addLayout(vl)
            self._vals[lbl] = val

    def set(self, lbl, text):
        if lbl in self._vals:
            self._vals[lbl].setText(text)

    def reset(self):
        for v in self._vals.values():
            v.setText("--")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CorrelationWindow(PickerMixin, QWidget):

    def __init__(self, dataset, fs=1000.0, is_time_resolved=False,
                 Nt_warn=2000, duration_warn=9999.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Correlation Analysis")
        self.resize(1700, 900)

        self.dataset          = dataset
        self.fs               = fs
        self.dt               = 1.0 / fs
        self.is_time_resolved = is_time_resolved

        self._x        = dataset["x"]
        self._y        = dataset["y"]
        self.U         = dataset["U"]
        self.V         = dataset["V"]
        self.W         = dataset["W"]
        self.is_stereo = dataset["is_stereo"]

        # Reference state
        self._pick_mode   = "point"   # "point" or "roi"
        self._ref_row     = None
        self._ref_col     = None
        self._use_kernel  = False
        self._roi_coords  = None      # (x0, x1, y0, y1) in mm
        self._roi_patch   = None
        self._roi_start   = None
        self._roi_active  = False

        # PickerMixin
        self._last_field_values = None

        # Cached results for export
        self._last_R_norm  = None
        self._last_R_x     = None
        self._last_R_y     = None
        self._last_dx      = None
        self._last_dy      = None
        self._last_R_tau   = None
        self._last_tau_sec = None

        self._show_convergence_warning(is_time_resolved, Nt_warn, duration_warn)
        self._build_ui()
        self._draw_mean_field()
        self._connect_mouse()
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    # -----------------------------------------------------------------------
    # Convergence warning
    # -----------------------------------------------------------------------

    def _show_convergence_warning(self, is_tr, Nt, duration):
        if is_tr:
            if duration < 2.0:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Dataset is {duration:.2f} s (< 2 s).\n"
                    "Correlations may not be statistically converged.")
        else:
            if Nt < 2000:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Only {Nt} snapshots (< 2000 recommended).\n"
                    "Spatial correlations may not be converged.\n"
                    "Temporal tab is not available for non-time-resolved data.")

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left: field + controls ----
        left = QWidget()
        left.setMinimumWidth(460)
        left.setMaximumWidth(560)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(4)

        self.field_fig    = Figure()
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas, stretch=5)

        self.lbl_ref = QLabel("Reference: not selected")
        self.lbl_ref.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_ref.setWordWrap(True)
        ll.addWidget(self.lbl_ref)

        # Reference mode group
        ref_grp = QGroupBox("Reference Mode")
        ref_lay = QVBoxLayout(ref_grp)

        mode_row = QHBoxLayout()
        self.rb_point = QRadioButton("Point")
        self.rb_roi   = QRadioButton("ROI")
        self.rb_point.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_point)
        bg.addButton(self.rb_roi)
        self.rb_point.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.rb_point)
        mode_row.addWidget(self.rb_roi)
        ref_lay.addLayout(mode_row)

        self.chk_kernel = QCheckBox("3\u00d73 kernel average (point mode)")
        self.chk_kernel.setChecked(False)
        ref_lay.addWidget(self.chk_kernel)

        self.lbl_hint = QLabel(
            "Left-click: pick point.   Right-click drag: draw ROI.")
        self.lbl_hint.setStyleSheet("color: #888; font-size: 10px;")
        self.lbl_hint.setWordWrap(True)
        ref_lay.addWidget(self.lbl_hint)
        ll.addWidget(ref_grp)

        # Options
        opt_grp = QGroupBox("Options")
        opt_lay = QVBoxLayout(opt_grp)

        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("Component:"))
        self.combo_comp = QComboBox()
        self.combo_comp.addItems(["R_uu", "R_vv", "R_ww"])
        if not self.is_stereo:
            self.combo_comp.model().item(2).setEnabled(False)
        comp_row.addWidget(self.combo_comp)
        opt_lay.addLayout(comp_row)

        lag_row = QHBoxLayout()
        lag_row.addWidget(QLabel("Max lag (% of Nt):"))
        self.spin_max_lag = QSpinBox()
        self.spin_max_lag.setRange(5, 50)
        self.spin_max_lag.setValue(50)
        self.spin_max_lag.setSuffix(" %")
        lag_row.addWidget(self.spin_max_lag)
        opt_lay.addLayout(lag_row)

        ll.addWidget(opt_grp)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        splitter.addWidget(left)

        # ---- Right: tabs ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_spatial_tab(),  "Spatial")
        self.tabs.addTab(self._build_temporal_tab(), "Temporal")

        if not self.is_time_resolved:
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabToolTip(
                1, "Temporal correlation requires time-resolved data.")

        rl.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setSizes([500, 1200])

    # -- Spatial tab --

    def _build_spatial_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        btn_row = QHBoxLayout()
        self.btn_spatial = QPushButton("Compute Spatial Correlation")
        self.btn_spatial.setEnabled(False)
        self.btn_spatial.clicked.connect(self._run_spatial)
        btn_row.addWidget(self.btn_spatial)
        btn_row.addStretch()
        self.btn_export_s2d = QPushButton("Export 2D Map...")
        self.btn_export_s2d.setEnabled(False)
        self.btn_export_s2d.clicked.connect(self._export_spatial_2d)
        btn_row.addWidget(self.btn_export_s2d)
        self.btn_export_s1d = QPushButton("Export 1D Slices...")
        self.btn_export_s1d.setEnabled(False)
        self.btn_export_s1d.clicked.connect(self._export_spatial_1d)
        btn_row.addWidget(self.btn_export_s1d)
        vl.addLayout(btn_row)

        # Plot area: 2D map | 1D x-slice | 1D y-slice
        # In ROI mode the 2D map widget is hidden
        self.spatial_plot_split = QSplitter(Qt.Orientation.Horizontal)

        # 2D map
        self.sp2d_wrap  = QWidget()
        sp2d_l = QVBoxLayout(self.sp2d_wrap)
        sp2d_l.setContentsMargins(0, 0, 0, 0)
        self.spatial_2d_fig    = Figure()
        self.spatial_2d_canvas = FigureCanvas(self.spatial_2d_fig)
        self.spatial_2d_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                              QSizePolicy.Policy.Expanding)
        self.spatial_2d_toolbar = DrawAwareToolbar(self.spatial_2d_canvas, w)
        sp2d_l.addWidget(self.spatial_2d_toolbar)
        sp2d_l.addWidget(self.spatial_2d_canvas)
        self.spatial_plot_split.addWidget(self.sp2d_wrap)

        # 1D x-slice
        self.sp1dx_wrap = QWidget()
        sp1dx_l = QVBoxLayout(self.sp1dx_wrap)
        sp1dx_l.setContentsMargins(0, 0, 0, 0)
        self.spatial_1dx_fig    = Figure()
        self.spatial_1dx_canvas = FigureCanvas(self.spatial_1dx_fig)
        self.spatial_1dx_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                               QSizePolicy.Policy.Expanding)
        self.spatial_1dx_toolbar = DrawAwareToolbar(self.spatial_1dx_canvas, w)
        sp1dx_l.addWidget(self.spatial_1dx_toolbar)
        sp1dx_l.addWidget(self.spatial_1dx_canvas)
        self.spatial_plot_split.addWidget(self.sp1dx_wrap)

        # 1D y-slice
        self.sp1dy_wrap = QWidget()
        sp1dy_l = QVBoxLayout(self.sp1dy_wrap)
        sp1dy_l.setContentsMargins(0, 0, 0, 0)
        self.spatial_1dy_fig    = Figure()
        self.spatial_1dy_canvas = FigureCanvas(self.spatial_1dy_fig)
        self.spatial_1dy_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                               QSizePolicy.Policy.Expanding)
        self.spatial_1dy_toolbar = DrawAwareToolbar(self.spatial_1dy_canvas, w)
        sp1dy_l.addWidget(self.spatial_1dy_toolbar)
        sp1dy_l.addWidget(self.spatial_1dy_canvas)
        self.spatial_plot_split.addWidget(self.sp1dy_wrap)

        vl.addWidget(self.spatial_plot_split, stretch=1)

        # Scale readout below plots
        vl.addWidget(_hline())
        self.spatial_scales = ScaleReadout(
            ["Lx (mm)", "Ly (mm)"])
        vl.addWidget(self.spatial_scales)

        return w

    # -- Temporal tab --

    def _build_temporal_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        btn_row = QHBoxLayout()
        self.btn_temporal = QPushButton("Compute Temporal Correlation")
        self.btn_temporal.setEnabled(False)
        self.btn_temporal.clicked.connect(self._run_temporal)
        btn_row.addWidget(self.btn_temporal)
        btn_row.addStretch()
        self.btn_export_tau = QPushButton("Export R(\u03c4)...")
        self.btn_export_tau.setEnabled(False)
        self.btn_export_tau.clicked.connect(self._export_temporal)
        btn_row.addWidget(self.btn_export_tau)
        vl.addLayout(btn_row)

        self.temporal_fig    = Figure()
        self.temporal_canvas = FigureCanvas(self.temporal_fig)
        self.temporal_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Expanding)
        self.temporal_toolbar = DrawAwareToolbar(self.temporal_canvas, w)
        vl.addWidget(self.temporal_toolbar)
        vl.addWidget(self.temporal_canvas, stretch=1)

        vl.addWidget(_hline())
        self.temporal_scales = ScaleReadout(["T (ms)", "\u03bb_t (ms)"])
        vl.addWidget(self.temporal_scales)

        return w

    # -----------------------------------------------------------------------
    # Reference mode toggle
    # -----------------------------------------------------------------------

    def _on_mode_changed(self):
        if self.rb_point.isChecked():
            self._pick_mode = "point"
            self.chk_kernel.setEnabled(True)
            self.lbl_hint.setText(
                "Left-click on field to pick reference point.")
            # In point mode show 2D map
            self.sp2d_wrap.setVisible(True)
        else:
            self._pick_mode = "roi"
            self.chk_kernel.setEnabled(False)
            self.lbl_hint.setText(
                "Right-click drag on field to draw ROI rectangle.")
            # In ROI mode hide 2D map, show only 1D panels
            self.sp2d_wrap.setVisible(False)

    # -----------------------------------------------------------------------
    # Mean field
    # -----------------------------------------------------------------------

    def _draw_mean_field(self):
        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)

        U_mean     = np.nanmean(self.U, axis=2)
        valid_frac = np.mean(self.dataset["valid"], axis=2)
        U_mean[valid_frac < 0.5] = np.nan

        cf = self.field_ax.contourf(
            self._x, self._y, U_mean, levels=50,
            cmap=_CMAP_DIV, extend="neither")
        self.field_fig.colorbar(cf, ax=self.field_ax,
                                label="Mean U [m/s]", shrink=0.8)
        self.field_ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        self.field_ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        self.field_ax.set_title("Mean U  —  click to select reference",
                                fontsize=_FONT_AX)
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_ax.tick_params(labelsize=_FONT_TICK)
        self.field_fig.tight_layout(pad=0.5)
        self.field_canvas.draw()
        self._last_field_values = U_mean

    def _redraw_marker(self):
        self._draw_mean_field()
        ax = self.field_ax

        if self._pick_mode == "point" and self._ref_row is not None:
            xr = self._x[self._ref_row, self._ref_col]
            yr = self._y[self._ref_row, self._ref_col]
            ax.plot(xr, yr, "k+", markersize=14, markeredgewidth=2, zorder=20)
            if self._use_kernel:
                # draw 3x3 box
                dx = float(np.abs(self._x[0, 1] - self._x[0, 0]))
                dy = float(np.abs(self._y[1, 0] - self._y[0, 0]))
                rect = Rectangle(
                    (xr - 1.5*dx, yr - 1.5*dy), 3*dx, 3*dy,
                    linewidth=1.5, edgecolor="yellow", facecolor="none",
                    linestyle="--", zorder=19)
                ax.add_patch(rect)

        elif self._pick_mode == "roi" and self._roi_coords is not None:
            x0, x1, y0, y1 = self._roi_coords
            rect = Rectangle(
                (min(x0,x1), min(y0,y1)), abs(x1-x0), abs(y1-y0),
                linewidth=1.5, edgecolor="yellow", facecolor="yellow",
                alpha=0.15, linestyle="--", zorder=19)
            ax.add_patch(rect)

        self.field_canvas.draw()

    # -----------------------------------------------------------------------
    # Mouse
    # -----------------------------------------------------------------------

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        if event.inaxes != self.field_ax:
            return
        if self._toolbar_active(self.field_toolbar):
            return

        if event.button == 1 and self._pick_mode == "point":
            row, col = nearest_grid_point(
                self._x, self._y, event.xdata, event.ydata)
            self._set_point_reference(row, col)

        elif event.button == 3:
            self._roi_active = True
            self._roi_start  = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self._roi_active:
            return
        if event.inaxes != self.field_ax or event.xdata is None:
            return
        x0, y0 = self._roi_start
        x1, y1 = event.xdata, event.ydata
        if self._roi_patch is not None:
            try:
                self._roi_patch.remove()
            except Exception:
                pass
        self._roi_patch = Rectangle(
            (min(x0, x1), min(y0, y1)), abs(x1-x0), abs(y1-y0),
            linewidth=1.5, edgecolor="yellow", facecolor="yellow",
            alpha=0.15, linestyle="--", zorder=20)
        self.field_ax.add_patch(self._roi_patch)
        self.field_canvas.draw_idle()

    def _on_release(self, event):
        if not self._roi_active:
            return
        self._roi_active = False
        if self._roi_patch is not None:
            try:
                self._roi_patch.remove()
            except Exception:
                pass
            self._roi_patch = None

        if event.inaxes != self.field_ax or self._roi_start is None:
            return
        if event.xdata is None:
            return

        x0, y0 = self._roi_start
        x1, y1 = event.xdata, event.ydata

        if abs(x1 - x0) < 0.5 and abs(y1 - y0) < 0.5:
            self.lbl_hint.setText("ROI too small -- try again.")
            return

        self._roi_coords = (x0, x1, y0, y1)
        self._pick_mode  = "roi"
        self.rb_roi.setChecked(True)

        self._redraw_marker()
        cx = 0.5*(x0+x1); cy = 0.5*(y0+y1)
        self.lbl_ref.setText(
            f"ROI: x=[{min(x0,x1):.1f}, {max(x0,x1):.1f}] mm  "
            f"y=[{min(y0,y1):.1f}, {max(y0,y1):.1f}] mm")
        self.btn_spatial.setEnabled(True)
        if self.is_time_resolved:
            self.btn_temporal.setEnabled(True)
        self.lbl_status.setText("ROI selected. Click 'Compute'.")

    def _set_point_reference(self, row, col):
        self._ref_row = row
        self._ref_col = col
        xr = self._x[row, col]
        yr = self._y[row, col]
        self._use_kernel = self.chk_kernel.isChecked()
        kernel_str = " (3\u00d73 kernel)" if self._use_kernel else ""
        self.lbl_ref.setText(
            f"Reference: x = {xr:.1f} mm,  y = {yr:.1f} mm{kernel_str}")
        self._redraw_marker()
        self.btn_spatial.setEnabled(True)
        if self.is_time_resolved:
            self.btn_temporal.setEnabled(True)
        self.lbl_status.setText(
            f"Reference set at ({xr:.1f}, {yr:.1f}) mm. Click 'Compute'.")

    # -----------------------------------------------------------------------
    # Spatial
    # -----------------------------------------------------------------------

    def _run_spatial(self):
        comp = self._component_key()
        self.lbl_status.setText("Busy: computing spatial correlation...")
        self.btn_spatial.setEnabled(False)
        self.spatial_scales.reset()
        QApplication.processEvents()

        try:
            if self._pick_mode == "point":
                self._use_kernel = self.chk_kernel.isChecked()
                R_norm, R_x, R_y = compute_spatial_correlation_point(
                    self.U, self.V, self.W,
                    self._ref_row, self._ref_col,
                    component=comp,
                    use_kernel=self._use_kernel)

                Lx, Ly = compute_spatial_scales_point(
                    R_norm, self._ref_row, self._ref_col,
                    self._x, self._y)

                self._last_R_norm = R_norm
                dx_vals = self._x[self._ref_row, self._ref_col:]
                dy_vals = self._y[self._ref_row:, self._ref_col]
                # Full x/y arrays for 1D slice plots
                self._last_R_x  = R_x
                self._last_R_y  = R_y
                self._last_dx   = self._x[self._ref_row, :]
                self._last_dy   = self._y[:, self._ref_col]

                self._plot_spatial_2d(R_norm)
                self._plot_spatial_1dx(
                    self._x[self._ref_row, :], R_x,
                    xlabel="x [mm]", ref_val=self._x[self._ref_row, self._ref_col])
                self._plot_spatial_1dy(
                    self._y[:, self._ref_col], R_y,
                    ylabel="y [mm]", ref_val=self._y[self._ref_row, self._ref_col])

                self.sp2d_wrap.setVisible(True)
                self.btn_export_s2d.setEnabled(True)

            else:
                if self._roi_coords is None:
                    QMessageBox.warning(self, "No ROI",
                        "Please draw an ROI first (right-click drag).")
                    return
                x0, x1, y0, y1 = self._roi_coords
                dx_arr, R_x, dy_arr, R_y, Lx, Ly = \
                    compute_spatial_correlation_roi(
                        self.U, self.V, self.W,
                        self._x, self._y,
                        x0, x1, y0, y1,
                        component=comp)

                self._last_R_norm = None
                self._last_R_x    = R_x
                self._last_R_y    = R_y
                self._last_dx     = dx_arr
                self._last_dy     = dy_arr

                self.sp2d_wrap.setVisible(False)
                self._plot_spatial_1dx(
                    dx_arr, R_x, xlabel="\u0394x [mm]", ref_val=0)
                self._plot_spatial_1dy(
                    dy_arr, R_y, ylabel="\u0394y [mm]", ref_val=0)
                self.btn_export_s2d.setEnabled(False)

            self.btn_export_s1d.setEnabled(True)

            # Update scale readout
            lx_str = f"{Lx:.2f}" if not np.isnan(Lx) else "--"
            ly_str = f"{Ly:.2f}" if not np.isnan(Ly) else "--"
            self.spatial_scales.set("Lx (mm)", lx_str)
            self.spatial_scales.set("Ly (mm)", ly_str)

            self.lbl_status.setText(
                f"Done. Spatial ({self.combo_comp.currentText()})  "
                f"Lx = {lx_str} mm  Ly = {ly_str} mm")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText(f"Error: {e}")
        finally:
            self.btn_spatial.setEnabled(True)

    def _plot_spatial_2d(self, R_norm):
        self.spatial_2d_fig.clear()
        ax = self.spatial_2d_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, R_norm,
                         levels=np.linspace(-1, 1, 41),
                         cmap=_CMAP_DIV, extend="neither")
        self.spatial_2d_fig.colorbar(cf, ax=ax, label="R [ ]", shrink=0.8)
        xr = self._x[self._ref_row, self._ref_col]
        yr = self._y[self._ref_row, self._ref_col]
        ax.plot(xr, yr, "k+", markersize=10, markeredgewidth=2, zorder=10)
        ax.contour(self._x, self._y, R_norm, levels=[0],
                   colors="k", linewidths=0.8, linestyles="--")
        ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        ax.set_title(f"Spatial correlation  {self.combo_comp.currentText()}",
                     fontsize=_FONT_AX)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=_FONT_TICK)
        self.spatial_2d_fig.tight_layout(pad=0.5)
        self.spatial_2d_canvas.draw()

    def _plot_spatial_1dx(self, x_vals, R_x, xlabel="x [mm]", ref_val=0):
        self.spatial_1dx_fig.clear()
        ax = self.spatial_1dx_fig.add_subplot(111)
        ax.plot(x_vals, R_x, "b-", linewidth=1.5,
                label=self.combo_comp.currentText())
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
        ax.axvline(ref_val, color="gray", linewidth=0.8,
                   linestyle="--", label="ref")
        # Mark first zero crossing
        zc = np.where(R_x <= 0)[0]
        if len(zc) > 0:
            ax.axvline(x_vals[zc[0]], color="r", linewidth=0.8,
                       linestyle=":", alpha=0.7, label="zero")
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel(xlabel, fontsize=_FONT_AX)
        ax.set_ylabel("R [ ]", fontsize=_FONT_AX)
        ax.set_title(f"{self.combo_comp.currentText()}  —  x direction",
                     fontsize=_FONT_AX)
        ax.legend(fontsize=_FONT_LEG)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=_FONT_TICK)
        self.spatial_1dx_fig.tight_layout(pad=0.5)
        self.spatial_1dx_canvas.draw()

    def _plot_spatial_1dy(self, y_vals, R_y, ylabel="y [mm]", ref_val=0):
        self.spatial_1dy_fig.clear()
        ax = self.spatial_1dy_fig.add_subplot(111)
        ax.plot(y_vals, R_y, "r-", linewidth=1.5,
                label=self.combo_comp.currentText())
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
        ax.axvline(ref_val, color="gray", linewidth=0.8,
                   linestyle="--", label="ref")
        zc = np.where(R_y <= 0)[0]
        if len(zc) > 0:
            ax.axvline(y_vals[zc[0]], color="r", linewidth=0.8,
                       linestyle=":", alpha=0.7, label="zero")
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel(ylabel, fontsize=_FONT_AX)
        ax.set_ylabel("R [ ]", fontsize=_FONT_AX)
        ax.set_title(f"{self.combo_comp.currentText()}  —  y direction",
                     fontsize=_FONT_AX)
        ax.legend(fontsize=_FONT_LEG)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=_FONT_TICK)
        self.spatial_1dy_fig.tight_layout(pad=0.5)
        self.spatial_1dy_canvas.draw()

    # -----------------------------------------------------------------------
    # Temporal
    # -----------------------------------------------------------------------

    def _run_temporal(self):
        if not self.is_time_resolved:
            return
        comp         = self._component_key()
        max_lag_frac = self.spin_max_lag.value() / 100.0

        self.lbl_status.setText("Busy: computing temporal autocorrelation...")
        self.btn_temporal.setEnabled(False)
        self.temporal_scales.reset()
        QApplication.processEvents()

        try:
            if self._pick_mode == "roi" and self._roi_coords is not None:
                roi_kw = dict(roi_coords=self._roi_coords,
                              x=self._x, y=self._y)
                ref_row = ref_col = 0
                use_k = False
            else:
                roi_kw  = {}
                ref_row = self._ref_row
                ref_col = self._ref_col
                use_k   = self.chk_kernel.isChecked()

            R_tau, lags = compute_temporal_correlation(
                self.U, self.V, self.W,
                ref_row, ref_col,
                component=comp,
                use_kernel=use_k,
                max_lag_fraction=max_lag_frac,
                **roi_kw)

            tau_sec = lags * self.dt
            T, lambda_t = compute_time_scales(R_tau, lags, self.dt)

            self._last_R_tau   = R_tau
            self._last_tau_sec = tau_sec

            T_str  = f"{T*1e3:.3f}"      if not np.isnan(T)        else "--"
            lt_str = f"{lambda_t*1e3:.3f}" if not np.isnan(lambda_t) else "--"
            self.temporal_scales.set("T (ms)",     T_str)
            self.temporal_scales.set("\u03bb_t (ms)", lt_str)

            self._plot_temporal(R_tau, tau_sec, T, lambda_t)
            self.btn_export_tau.setEnabled(True)
            self.lbl_status.setText(
                f"Done. Temporal ({self.combo_comp.currentText()})  "
                f"T = {T_str} ms   \u03bb_t = {lt_str} ms")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText(f"Error: {e}")
        finally:
            self.btn_temporal.setEnabled(True)

    def _plot_temporal(self, R_tau, tau_sec, T, lambda_t):
        self.temporal_fig.clear()
        ax = self.temporal_fig.add_subplot(111)
        comp_label = self.combo_comp.currentText()
        ax.plot(tau_sec * 1e3, R_tau, "b-", linewidth=1.5, label=comp_label)
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")

        if not np.isnan(T):
            ax.axvline(T*1e3, color="r", linewidth=1.2, linestyle="--",
                       label=f"T = {T*1e3:.2f} ms")
        if not np.isnan(lambda_t):
            ax.axvline(lambda_t*1e3, color="g", linewidth=1.2, linestyle="-.",
                       label=f"\u03bb_t = {lambda_t*1e3:.2f} ms")

        zc = np.where(R_tau <= 0)[0]
        if len(zc) > 0:
            ax.axvline(tau_sec[zc[0]]*1e3, color="gray",
                       linewidth=0.8, linestyle=":", label="zero crossing")

        ax.set_xlabel("\u03c4 [ms]", fontsize=_FONT_AX)
        ax.set_ylabel("R(\u03c4) [ ]", fontsize=_FONT_AX)
        ax.set_title(f"Temporal autocorrelation  {comp_label}", fontsize=_FONT_AX)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=_FONT_LEG)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=_FONT_TICK)
        self.temporal_fig.tight_layout(pad=0.5)
        self.temporal_canvas.draw()

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def _export_spatial_2d(self):
        if self._last_R_norm is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Spatial Correlation 2D",
            "spatial_corr_2d.dat", "Tecplot DAT (*.dat);;All Files (*)")
        if not path:
            return
        comp = self.combo_comp.currentText()
        settings = {"Analysis": f"Spatial Correlation 2D  {comp}",
                    "Snapshots": self.dataset["Nt"],
                    "Component": comp}
        export_2d_tecplot(path, self._x, self._y,
                          [self._last_R_norm], [f"R_{comp} [ ]"], settings)
        self.lbl_status.setText(f"Exported 2D map to {path}")

    def _export_spatial_1d(self):
        if self._last_R_x is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export 1D Slices", "spatial_corr_1d.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        comp = self.combo_comp.currentText()
        settings = {"Analysis": f"Spatial Correlation 1D  {comp}",
                    "Snapshots": self.dataset["Nt"]}
        means = {"R_x": self._last_R_x, "R_y": self._last_R_y}
        export_line_csv(path, self._last_dx, self._last_dx,
                        self._last_dx, means, {}, settings)
        self.lbl_status.setText(f"Exported 1D slices to {path}")

    def _export_temporal(self):
        if self._last_R_tau is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Temporal Correlation", "temporal_corr.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        comp = self.combo_comp.currentText()
        settings = {"Analysis": f"Temporal Autocorrelation  {comp}",
                    "Snapshots": self.dataset["Nt"], "fs [Hz]": self.fs}
        means = {"R_tau": self._last_R_tau}
        tau_ms = self._last_tau_sec * 1e3
        export_line_csv(path, tau_ms, tau_ms, tau_ms, means, {}, settings)
        self.lbl_status.setText(f"Exported R(tau) to {path}")

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    def _component_key(self):
        return self.combo_comp.currentText().replace("R_", "")
