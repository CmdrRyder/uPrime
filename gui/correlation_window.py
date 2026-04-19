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
  ROI   : left-click drag on field (rectangle).

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
    QGridLayout, QRadioButton, QButtonGroup,
    QFileDialog, QFrame, QDialog, QProgressBar
)
from PyQt6.QtCore import Qt
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
    compute_length_scale,
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
        from core.dataset_utils import get_masked
        self.U = get_masked(dataset, "U")
        self.V = get_masked(dataset, "V")
        self.W = get_masked(dataset, "W")
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
            "Left-click on field to pick reference point.")
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

        chk_row = QHBoxLayout()
        chk_row.addStretch()
        self.chk_hide_axes = QCheckBox("Hide axes")
        self.chk_hide_axes.stateChanged.connect(self._replot_current)
        chk_row.addWidget(self.chk_hide_axes)
        self.chk_hide_colorbar = QCheckBox("Hide colorbar")
        self.chk_hide_colorbar.stateChanged.connect(self._replot_current)
        chk_row.addWidget(self.chk_hide_colorbar)
        rl.addLayout(chk_row)

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
        self.btn_show_diagnostic = QPushButton("Show Diagnostic Plot")
        self.btn_show_diagnostic.setEnabled(False)
        self.btn_show_diagnostic.clicked.connect(self._show_diagnostic)
        btn_row.addWidget(self.btn_show_diagnostic)
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

        self.progress_bar_s = QProgressBar()
        self.progress_bar_s.setFixedHeight(6)
        self.progress_bar_s.setTextVisible(False)
        self.progress_bar_s.setVisible(False)
        vl.addWidget(self.progress_bar_s)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale method:"))
        self.combo_scale_method = QComboBox()
        self.combo_scale_method.addItem("Zero crossing",  "zero_crossing")
        self.combo_scale_method.addItem("Exponential fit", "exp_fit")
        self.combo_scale_method.addItem("1/e point",       "one_over_e")
        self.combo_scale_method.addItem("Domain integral", "domain")
        scale_row.addWidget(self.combo_scale_method)
        scale_row.addStretch()
        vl.addLayout(scale_row)

        # Plot area: 2D map on top, single 1D panel below (x and y overlaid)
        # In ROI mode the 2D map widget is hidden
        self.spatial_plot_split = QSplitter(Qt.Orientation.Vertical)

        # 2D map (top)
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

        # 1D panels (bottom) — x-direction and y-direction side by side
        sp1d_split = QSplitter(Qt.Orientation.Horizontal)

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
        sp1d_split.addWidget(self.sp1dx_wrap)

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
        sp1d_split.addWidget(self.sp1dy_wrap)

        self.spatial_plot_split.addWidget(sp1d_split)

        vl.addWidget(self.spatial_plot_split, stretch=1)

        # Scale readout below plots
        vl.addWidget(_hline())
        self.spatial_scales = ScaleReadout(["Lx (mm)", "Ly (mm)"])
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

        self.progress_bar_t = QProgressBar()
        self.progress_bar_t.setFixedHeight(6)
        self.progress_bar_t.setTextVisible(False)
        self.progress_bar_t.setVisible(False)
        vl.addWidget(self.progress_bar_t)

        # Length scale method for temporal
        tm_row = QHBoxLayout()
        tm_row.addWidget(QLabel("Scale method:"))
        self.combo_temp_scale_method = QComboBox()
        self.combo_temp_scale_method.addItem("Zero crossing",   "zero_crossing")
        self.combo_temp_scale_method.addItem("Exponential fit",  "exp_fit")
        self.combo_temp_scale_method.addItem("1/e point",        "one_over_e")
        self.combo_temp_scale_method.addItem("Domain integral",  "domain")
        tm_row.addWidget(self.combo_temp_scale_method)
        tm_row.addStretch()
        vl.addLayout(tm_row)

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

    def _replot_current(self):
        if self.tabs.currentIndex() == 0:
            self._run_spatial()
        else:
            self._run_temporal()

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
                "Left-click and drag on field to draw ROI rectangle.")
            # In ROI mode hide 2D map, show only 1D panels
            self.sp2d_wrap.setVisible(False)

    # -----------------------------------------------------------------------
    # Mean field
    # -----------------------------------------------------------------------

    def _draw_mean_field(self):
        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)

        U_mean = np.nanmean(self.U, axis=2)
        U_mean[~self.dataset["MASK"]] = np.nan

        self.field_ax.contourf(
            self._x, self._y, U_mean, levels=50,
            cmap=_CMAP_DIV, extend="neither")
        self.field_ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        self.field_ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        self.field_ax.set_title("Mean U  —  click to select reference",
                                fontsize=_FONT_AX)
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_ax.tick_params(labelsize=_FONT_TICK)
        self.field_fig.tight_layout(pad=0.5)
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()
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

        elif event.button == 1 and self._pick_mode == "roi":
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
        if self._pick_mode == "point" and self._ref_row is None:
            return
        if self._pick_mode == "roi" and self._roi_coords is None:
            QMessageBox.warning(self, "No ROI",
                "Please draw an ROI first (left-click drag in ROI mode).")
            return

        from core.workers import CorrelationWorker

        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        comp = self._component_key()
        self.lbl_status.setText("Busy: computing spatial correlation...")
        self.btn_spatial.setEnabled(False)
        self.spatial_scales.reset()
        self.progress_bar_s.setRange(0, 0)
        self.progress_bar_s.setVisible(True)

        if self._pick_mode == "point":
            self._use_kernel = self.chk_kernel.isChecked()
            mode = "spatial_point"
            roi_coords = None
        else:
            mode = "spatial_roi"
            roi_coords = self._roi_coords

        self._worker = CorrelationWorker(
            self.U, self.V, self.W,
            self._x, self._y,
            mode=mode,
            ref_row=self._ref_row if self._ref_row is not None else 0,
            ref_col=self._ref_col if self._ref_col is not None else 0,
            component=comp,
            use_kernel=self._use_kernel,
            max_lag_frac=self.spin_max_lag.value() / 100.0,
            dt=self.dt,
            roi_coords=roi_coords,
        )
        self._worker.finished.connect(self._on_corr_result)
        self._worker.error.connect(self._on_corr_error)
        self._worker.finished.connect(lambda _: self._reset_spatial_ui())
        self._worker.error.connect(lambda _: self._reset_spatial_ui())
        self._worker.start()

    def _reset_spatial_ui(self):
        self.btn_spatial.setEnabled(True)
        self.progress_bar_s.setRange(0, 100)
        self.progress_bar_s.setVisible(False)

    def _plot_spatial_2d(self, R_norm):
        self.spatial_2d_fig.clear()
        ax = self.spatial_2d_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, R_norm,
                         levels=np.linspace(-1, 1, 41),
                         cmap=_CMAP_DIV, extend="neither")
        cb = self.spatial_2d_fig.colorbar(cf, ax=ax, label="R [ ]", shrink=0.8)
        if self.chk_hide_colorbar.isChecked():
            cb.remove()
            self.spatial_2d_fig.tight_layout(pad=0.5)
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
        if self.chk_hide_axes.isChecked():
            ax.axis('off')
            ax.set_title('')
        self.spatial_2d_fig.tight_layout(pad=0.5)
        self.spatial_2d_canvas.draw()
        self.spatial_2d_toolbar.set_home_limits()

    def _plot_spatial_1d(self, r_vals, R, xlabel, ref_val, extras, L,
                         direction='x', color="tab:blue"):
        """Plot a correlation slice into its own dedicated canvas.

        direction='x' plots into the left panel (spatial_1dx_fig/canvas).
        direction='y' plots into the right panel (spatial_1dy_fig/canvas).
        """
        if direction == 'x':
            fig    = self.spatial_1dx_fig
            canvas = self.spatial_1dx_canvas
        else:
            fig    = self.spatial_1dy_fig
            canvas = self.spatial_1dy_canvas

        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_ylim(-0.3, 1.1)
        ax.set_xlabel(xlabel, fontsize=_FONT_AX)
        ax.set_ylabel("R [ ]", fontsize=_FONT_AX)
        comp = self.combo_comp.currentText()
        ax.set_title(f"{comp} ({direction}-direction)", fontsize=_FONT_AX)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=_FONT_TICK)

        # Convert to separation distance starting at 0
        if ref_val != 0:
            ref_idx = np.argmin(np.abs(r_vals - ref_val))
            r_plot      = np.abs(r_vals[ref_idx:] - ref_val)
            R_plot_data = R[ref_idx:]
        else:
            r_plot      = r_vals
            R_plot_data = R

        method = self.combo_scale_method.currentData()
        L_str  = f"{L:.2f}" if not np.isnan(L) else "--"

        # R curve — label carries L only for methods where no separate vline
        # explains L; for all other methods the vline label is sufficient.
        ax.plot(r_plot, R_plot_data, color=color, linewidth=1.5,
                label=f"{comp} ({direction})")

        # Method-specific reference lines
        if method == "zero_crossing":
            ax.axhline(0, color="k", linewidth=0.8, linestyle=":",
                       label="R = 0")
            zc = extras.get("zero_crossing")
            if zc is not None:
                ax.axvline(zc, color="k", linewidth=0.8, linestyle=":",
                           label=f"zero crossing = {zc:.2f} mm")

        elif method == "one_over_e":
            ax.axhline(1.0 / np.e, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.7, label="1/e")

        elif method == "exp_fit":
            if extras.get("fit_r") is not None:
                ax.plot(extras["fit_r"], extras["fit_R"],
                        "--", color="k", linewidth=1.0, alpha=0.7,
                        label="exp fit")

        elif method == "domain":
            ax.axhline(0, color="k", linewidth=0.8, linestyle=":",
                       label="R = 0")

        # Integral scale vline — red dashed when L is valid
        if method == "one_over_e":
            marker_x = extras.get("marker_x", np.nan)
            if marker_x is not None and not np.isnan(marker_x):
                ax.axvline(marker_x, color="red", linewidth=1.2,
                           linestyle="--", alpha=0.9,
                           label=f"L{direction} = {marker_x:.2f} mm")
        elif method in ("zero_crossing", "domain"):
            # Line at the actual crossing lag; L (the integral) shown in label
            cross_lag = extras.get("crossing_lag")
            if cross_lag is None:
                # domain method uses cutoff_idx when there is no crossing
                ci = extras.get("cutoff_idx")
                if ci is not None:
                    cross_lag = ci * (extras["r_axis"][1] - extras["r_axis"][0]) if len(extras["r_axis"]) > 1 else np.nan
            if cross_lag is not None and not np.isnan(cross_lag) and not np.isnan(L):
                suffix = " (domain)" if method == "domain" else ""
                ax.axvline(cross_lag, color="red", linewidth=1.2,
                           linestyle="--", alpha=0.9,
                           label=f"L{direction} = {L_str} mm{suffix}")
        elif not np.isnan(L):
            ax.axvline(L, color="red", linewidth=1.2,
                       linestyle="--", alpha=0.9,
                       label=f"L{direction} = {L_str} mm")

        ax.legend(fontsize=_FONT_LEG, loc="upper right")
        if self.chk_hide_axes.isChecked():
            ax.axis('off')
        fig.tight_layout(pad=0.5)
        canvas.draw()
        toolbar = self.spatial_1dx_toolbar if direction == 'x' else self.spatial_1dy_toolbar
        toolbar.set_home_limits()

    def _show_diagnostic(self):
        """Open a small dialog showing the cumulative integral for x and y."""
        ex = getattr(self, "_last_extras_x", None)
        ey = getattr(self, "_last_extras_y", None)
        Lx_str = self.spatial_scales._vals.get("Lx (mm)")
        Ly_str = self.spatial_scales._vals.get("Ly (mm)")
        Lx = float(Lx_str.text()) if Lx_str and Lx_str.text() != "--" else np.nan
        Ly = float(Ly_str.text()) if Ly_str and Ly_str.text() != "--" else np.nan

        dlg = QDialog(self)
        dlg.setWindowTitle("Cumulative Integral Diagnostic")
        dlg.resize(800, 400)
        lay = QVBoxLayout(dlg)

        fig = Figure(tight_layout=True)
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        for ax, extras, L, label, color in [
            (ax1, ex, Lx, "x", "tab:blue"),
            (ax2, ey, Ly, "y", "tab:red"),
        ]:
            cumul = extras.get("cumulative") if extras else None
            r_ax  = extras.get("r_axis")    if extras else None
            if cumul is None or r_ax is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
            else:
                ax.plot(r_ax, cumul, color=color, linewidth=1.5,
                        label="Cumulative integral")
                if not np.isnan(L):
                    ax.axhline(L, color=color, linewidth=1.2,
                               linestyle="--", alpha=0.8,
                               label=f"L{label} = {L:.2f} mm")
                ax.set_xlabel("r [mm]", fontsize=_FONT_AX)
                ax.set_ylabel("∫₀ʳ R dr  [mm]", fontsize=_FONT_AX)
                ax.set_title(f"Cumulative integral ({label}-dir)", fontsize=_FONT_AX)
                ax.legend(fontsize=_FONT_LEG)
                ax.grid(True, alpha=0.3)
                ax.set_box_aspect(1)
                ax.tick_params(labelsize=_FONT_TICK)

        canvas.draw()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()



    def _on_corr_result(self, result):
        comp = self.combo_comp.currentText()
        mode = result['mode']

        if mode == 'spatial_point':
            R_norm = result['R_norm']
            R_x    = result['R_x']
            R_y    = result['R_y']

            method = self.combo_scale_method.currentData()
            Lx, Ly, ex, ey = compute_spatial_scales_point(
                R_norm, self._ref_row, self._ref_col,
                self._x, self._y, method=method)
            self._last_extras_x = ex
            self._last_extras_y = ey
            self._last_R_norm   = R_norm
            self._last_R_x      = R_x
            self._last_R_y      = R_y
            self._last_dx       = self._x[self._ref_row, :]
            self._last_dy       = self._y[:, self._ref_col]

            self._plot_spatial_2d(R_norm)
            self._plot_spatial_1d(
                self._x[self._ref_row, :], R_x,
                xlabel="x [mm]",
                ref_val=self._x[self._ref_row, self._ref_col],
                extras=ex, L=Lx, direction='x', color="tab:blue")
            self._plot_spatial_1d(
                self._y[:, self._ref_col], R_y,
                xlabel="y [mm]",
                ref_val=self._y[self._ref_row, self._ref_col],
                extras=ey, L=Ly, direction='y', color="tab:red")

            self.sp2d_wrap.setVisible(True)
            self.btn_export_s2d.setEnabled(True)
            self.btn_export_s1d.setEnabled(True)
            self.btn_show_diagnostic.setEnabled(True)

            lx_str = f"{Lx:.2f}" if not np.isnan(Lx) else "--"
            ly_str = f"{Ly:.2f}" if not np.isnan(Ly) else "--"
            self.spatial_scales.set("Lx (mm)", lx_str)
            self.spatial_scales.set("Ly (mm)", ly_str)
            self.lbl_status.setText(
                f"Done. Spatial ({comp})  Lx = {lx_str} mm  Ly = {ly_str} mm")

        elif mode == 'spatial_roi':
            dx_arr = result['dx_arr']
            R_x    = result['R_x']
            dy_arr = result['dy_arr']
            R_y    = result['R_y']

            method = self.combo_scale_method.currentData()
            dx_sp = dx_arr[1] - dx_arr[0] if len(dx_arr) > 1 else 1.0
            dy_sp = dy_arr[1] - dy_arr[0] if len(dy_arr) > 1 else 1.0
            Lx, ex = compute_length_scale(R_x, dx_sp, method)
            Ly, ey = compute_length_scale(R_y, dy_sp, method)
            self._last_extras_x = ex
            self._last_extras_y = ey
            self._last_R_norm    = None
            self._last_R_x       = R_x
            self._last_R_y       = R_y
            self._last_dx        = dx_arr
            self._last_dy        = dy_arr

            self.sp2d_wrap.setVisible(False)
            self._plot_spatial_1d(
                dx_arr, R_x, xlabel="\u0394x [mm]", ref_val=0,
                extras=ex, L=Lx, direction='x', color="tab:blue")
            self._plot_spatial_1d(
                dy_arr, R_y, xlabel="\u0394y [mm]", ref_val=0,
                extras=ey, L=Ly, direction='y', color="tab:red")
            self.btn_export_s2d.setEnabled(False)
            self.btn_export_s1d.setEnabled(True)
            self.btn_show_diagnostic.setEnabled(True)

            lx_str = f"{Lx:.2f}" if not np.isnan(Lx) else "--"
            ly_str = f"{Ly:.2f}" if not np.isnan(Ly) else "--"
            self.spatial_scales.set("Lx (mm)", lx_str)
            self.spatial_scales.set("Ly (mm)", ly_str)
            self.lbl_status.setText(
                f"Done. Spatial ({comp})  Lx = {lx_str} mm  Ly = {ly_str} mm")

        else:  # temporal
            R_tau = result['R_tau']
            lags  = result['lags']
            tau_sec = lags * self.dt
            T, lambda_t, t_extras = compute_time_scales(R_tau, lags, self.dt)

            self._last_R_tau   = R_tau
            self._last_tau_sec = tau_sec

            T_str  = f"{T*1e3:.3f}"        if not np.isnan(T)        else "--"
            lt_str = f"{lambda_t*1e3:.3f}" if not np.isnan(lambda_t) else "--"
            self.temporal_scales.set("T (ms)",          T_str)
            self.temporal_scales.set("\u03bb_t (ms)",  lt_str)

            self._plot_temporal(R_tau, tau_sec, T, lambda_t, t_extras)
            self.btn_export_tau.setEnabled(True)
            self.lbl_status.setText(
                f"Done. Temporal ({comp})  T = {T_str} ms   \u03bb_t = {lt_str} ms")

    def _on_corr_error(self, tb_str):
        QMessageBox.critical(self, "Correlation Error", tb_str)
        self.lbl_status.setText("Error — see dialog.")

    # -----------------------------------------------------------------------
    # Temporal
    # -----------------------------------------------------------------------

    def _run_temporal(self):
        if not self.is_time_resolved:
            return
        if self._pick_mode == "point" and self._ref_row is None:
            return

        from core.workers import CorrelationWorker

        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        comp         = self._component_key()
        max_lag_frac = self.spin_max_lag.value() / 100.0

        if self._pick_mode == "roi" and self._roi_coords is not None:
            roi_coords = self._roi_coords
            ref_row    = ref_col = 0
            use_k      = False
        else:
            roi_coords = None
            ref_row    = self._ref_row if self._ref_row is not None else 0
            ref_col    = self._ref_col if self._ref_col is not None else 0
            use_k      = self.chk_kernel.isChecked()

        self.lbl_status.setText("Busy: computing temporal autocorrelation...")
        self.btn_temporal.setEnabled(False)
        self.temporal_scales.reset()
        self.progress_bar_t.setRange(0, 0)
        self.progress_bar_t.setVisible(True)

        self._worker = CorrelationWorker(
            self.U, self.V, self.W,
            self._x, self._y,
            mode='temporal',
            ref_row=ref_row,
            ref_col=ref_col,
            component=comp,
            use_kernel=use_k,
            max_lag_frac=max_lag_frac,
            dt=self.dt,
            roi_coords=roi_coords,
        )
        self._worker.finished.connect(self._on_corr_result)
        self._worker.error.connect(self._on_corr_error)
        self._worker.finished.connect(lambda _: self._reset_temporal_ui())
        self._worker.error.connect(lambda _: self._reset_temporal_ui())
        self._worker.start()

    def _reset_temporal_ui(self):
        self.btn_temporal.setEnabled(True)
        self.progress_bar_t.setRange(0, 100)
        self.progress_bar_t.setVisible(False)

    def _plot_temporal(self, R_tau, tau_sec, T, lambda_t, extras):
        """Plot temporal autocorrelation with fit and cumulative inset."""
        self.temporal_fig.clear()

        # Two subplots side by side: R(tau) left, cumulative right
        ax1 = self.temporal_fig.add_subplot(1, 2, 1)
        ax2 = self.temporal_fig.add_subplot(1, 2, 2)

        comp_label = self.combo_comp.currentText()
        tau_ms = tau_sec * 1e3

        # --- Left: R(tau) ---
        ax1.plot(tau_ms, R_tau, color="tab:blue", linewidth=1.5, label=comp_label)
        ax1.axhline(1.0/np.e, color="gray", linewidth=0.8, linestyle="--", label="1/e")
        ax1.axhline(0, color="k", linewidth=0.5, linestyle=":")

        # Exponential fit overlay
        method = self.combo_temp_scale_method.currentData()
        if method == "exp_fit" and extras.get("fit_r") is not None:
            fit_tau_ms = extras["fit_r"] * 1e3
            ax1.plot(fit_tau_ms, extras["fit_R"], "k--",
                     linewidth=1.2, alpha=0.7, label="exp fit")

        # Mark T
        T_ms = T * 1e3 if not np.isnan(T) else np.nan
        T_str_ms = f"{T_ms:.2f} ms" if not np.isnan(T_ms) else "--"
        if method == "one_over_e":
            marker_x = extras.get("marker_x", np.nan)
            if marker_x is not None and not np.isnan(marker_x):
                marker_ms = marker_x * 1e3
                ax1.axvline(marker_ms, color="red", linewidth=1.2,
                            linestyle="--", alpha=0.9,
                            label=f"T = {marker_ms:.2f} ms")
        elif method in ("zero_crossing", "domain"):
            # Line at the crossing lag; T (the integral) shown in label
            cross_lag = extras.get("crossing_lag")
            if cross_lag is None:
                ci = extras.get("cutoff_idx")
                if ci is not None and len(extras.get("r_axis", [])) > 1:
                    cross_lag = ci * (extras["r_axis"][1] - extras["r_axis"][0])
            if cross_lag is not None and not np.isnan(T_ms):
                cross_ms = cross_lag * 1e3
                suffix = " (domain)" if method == "domain" else ""
                ax1.axvline(cross_ms, color="tab:orange", linewidth=1.2,
                            linestyle="-", label=f"T = {T_str_ms}{suffix}")
        elif not np.isnan(T_ms):
            ax1.axvline(T_ms, color="tab:orange", linewidth=1.2,
                        linestyle="-", label=f"T = {T_str_ms}")

        # Mark lambda_t
        if not np.isnan(lambda_t):
            lt_ms = lambda_t * 1e3
            ax1.axvline(lt_ms, color="tab:green", linewidth=1.0,
                        linestyle="--", alpha=0.7, label=f"λ_t = {lt_ms:.2f} ms")

        ax1.set_ylim(-0.3, 1.1)
        ax1.set_xlabel("τ [ms]", fontsize=_FONT_AX)
        ax1.set_ylabel("R(τ) [ ]", fontsize=_FONT_AX)
        no_cross = extras.get("no_crossing", False)
        T_str = f"{T_ms:.2f} ms" if not np.isnan(T_ms) else "--"
        flag = " (>)" if no_cross else ""
        ax1.set_title(f"Autocorrelation    T = {T_str}{flag}", fontsize=_FONT_AX)
        ax1.legend(fontsize=_FONT_LEG, loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_box_aspect(1)

        # --- Right: cumulative integral ---
        cumul = extras.get("cumulative")
        r_ax  = extras.get("r_axis")
        if cumul is not None and r_ax is not None:
            ax2.plot(r_ax * 1e3, cumul * 1e3, color="tab:purple",
                     linewidth=1.5, label="Cumulative")
            if not np.isnan(T_ms):
                ax2.axhline(T_ms, color="tab:orange", linewidth=1.2,
                            linestyle="--", label=f"T = {T_str}")
            ax2.set_xlabel("τ [ms]", fontsize=_FONT_AX)
            ax2.set_ylabel("∫₀^τ R dτ  [ms]", fontsize=_FONT_AX)
            ax2.set_title("Cumulative integral", fontsize=_FONT_AX)
            ax2.legend(fontsize=_FONT_LEG)
            ax2.grid(True, alpha=0.3)
            ax2.set_box_aspect(1)

        if self.chk_hide_axes.isChecked():
            for a in self.temporal_fig.axes:
                a.axis('off')
        self.temporal_fig.tight_layout(pad=0.5)
        self.temporal_canvas.draw()
        self.temporal_toolbar.set_home_limits()

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
