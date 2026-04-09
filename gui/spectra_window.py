"""
gui/spectra_window.py
---------------------
Unified spectral analysis window with four tabs:
  1. Spatial  E(k)           -- always available (1D line/ROI)
  2. Spatial  E(k) using FFT -- always available (FFT-based)
  3. Temporal    E(f)     -- TR data only
  4. Spatiotemporal E(k,f)-- TR data only

Replaces separate spectral_window.py and spatial_spectra_window.py in the menu.
"""
import traceback

import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QRadioButton, QCheckBox, QSizePolicy,
    QMessageBox, QSplitter, QSpinBox, QButtonGroup,
    QFileDialog, QApplication, QTabWidget, QDoubleSpinBox,
    QComboBox
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRect

from core.spatial_spectra import spatial_psd_line, spatial_psd_roi
try:
    from core.spatial_spectra_fft import compute_spectra_from_fluctuations, subtract_temporal_mean
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False
from core.spatiotemporal_spectra import compute_st_spectra
from core.spectral import psd_at_point, psd_in_region, nearest_grid_point
from core.export import export_spectra_csv
from gui.line_selector import compute_snapped_line
from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin


class SpectraWindow(PickerMixin, QWidget):

    def __init__(self, dataset, is_time_resolved=False, fs=1000.0, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self._is_tr  = is_time_resolved
        self._fs     = fs
        self.setWindowTitle("Space-Time Spectral Analysis")
        self.resize(1700, 900)

        Nt = dataset["Nt"]
        if Nt < 2:
            QMessageBox.critical(self, "Insufficient Data",
                "Spectral analysis requires multiple snapshots.")
            return

        # Convergence warnings
        if is_time_resolved:
            dur = Nt / fs
            if dur < 2.0:
                QMessageBox.warning(self, "Convergence Warning",
                    f"{Nt} snapshots @ {fs:.0f} Hz = {dur:.2f} s.\n"
                    "Less than 2 s -- spectra may not be converged.")
        else:
            if Nt < 1000:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Only {Nt} snapshots -- spatial spectra may not be converged.")

        # State
        self._mode        = "horizontal"
        self._press_xy    = None
        self._artist      = None
        self._selection   = None
        self._last_result = None

        self._build_ui()
        self._draw_field()
        self._connect_mouse()
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

    def _drawing_active(self):
        """Suppress PickerMixin red-cross; this window manages all clicks."""
        return True

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- LEFT: field + controls ----
        left = QWidget()
        left.setMinimumWidth(500); left.setMaximumWidth(600)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)

        self.field_fig    = Figure()
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas, stretch=6)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_spatial_tab(),     "Spatial E(k)")
        self.tabs.addTab(self._build_3d_spatial_tab(),   "Spatial E(k) using FFT")
        if not PYFFTW_AVAILABLE:
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabToolTip(1, "Requires pyFFTW -- install with: pip install pyfftw")
        self.tabs.addTab(self._build_temporal_tab(),    "Temporal E(f)")
        self.tabs.addTab(self._build_st_tab(),          "Spatiotemporal E(k,f)")

        if not self._is_tr:
            self.tabs.setTabEnabled(2, False)
            self.tabs.setTabToolTip(2, "Requires time-resolved data")
            self.tabs.setTabEnabled(3, False)
            self.tabs.setTabToolTip(3, "Requires time-resolved data")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        ll.addWidget(self.tabs, stretch=0)

        self.lbl_hint = QLabel("Select a mode and draw on the field.")
        self.lbl_hint.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint, stretch=0)

        self.btn_compute = QPushButton("Compute")
        self.btn_compute.setEnabled(False)
        self.btn_compute.clicked.connect(self._on_compute)
        ll.addWidget(self.btn_compute, stretch=0)

        self.btn_export = QPushButton("Export Data...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        ll.addWidget(self.btn_export, stretch=0)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status, stretch=0)

        # ---- RIGHT: result canvas -- centered, max 65% of panel width ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        # Toolbar sits at top full-width
        self.result_fig    = Figure()
        self.result_canvas = FigureCanvas(self.result_fig)
        # Constrain canvas so it never goes portrait:
        # fixed max height = 520px, expands horizontally
        self.result_canvas.setMinimumHeight(380)
        self.result_canvas.setMaximumHeight(560)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Preferred)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        chk_row = QHBoxLayout()
        chk_row.addStretch()
        self.chk_hide_axes = QCheckBox("Hide axes")
        self.chk_hide_axes.stateChanged.connect(self._on_compute)
        chk_row.addWidget(self.chk_hide_axes)
        self.chk_hide_colorbar = QCheckBox("Hide colorbar")
        self.chk_hide_colorbar.stateChanged.connect(self._on_compute)
        chk_row.addWidget(self.chk_hide_colorbar)
        rl.addLayout(chk_row)

        # Center the canvas with side margins so 2D plots don't fill wall-to-wall
        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addWidget(self.result_canvas, stretch=8)
        center_row.addStretch(1)
        rl.addLayout(center_row)

        self.lbl_mask_warning = QLabel("")
        self.lbl_mask_warning.setStyleSheet("color: gray; font-size: 10px;")
        self.lbl_mask_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_mask_warning.setVisible(False)
        rl.addWidget(self.lbl_mask_warning)

        rl.addStretch(1)   # push canvas to top-center, empty space below

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([920, 780])

    # ----------------------------------------------------------------------- #
    # Tab builders
    # ----------------------------------------------------------------------- #

    def _build_spatial_tab(self):
        w = QWidget(); ll = QVBoxLayout(w)
        ll.setContentsMargins(4,4,4,4); ll.setSpacing(4)

        lbl_welch = QLabel(
            "Welch: segments the spatial line into overlapping windows and averages "
            "their FFTs. Smoother result, lower spatial frequency resolution."
        )
        lbl_welch.setWordWrap(True)
        lbl_welch.setStyleSheet("color: gray;")
        ll.addWidget(lbl_welch)

        sg = QGroupBox("Selection Mode")
        sl = QVBoxLayout(sg)
        mr = QHBoxLayout()
        self.rb_horiz = QRadioButton("Horizontal (kx)")
        self.rb_vert  = QRadioButton("Vertical (ky)")
        self.rb_horiz.setChecked(True)
        bg = QButtonGroup(self)
        for rb in [self.rb_horiz, self.rb_vert]:
            bg.addButton(rb); mr.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        sl.addLayout(mr)
        ar = QHBoxLayout(); ar.addWidget(QLabel("Spatial avg ± pts:"))
        self.spin_avg = QSpinBox(); self.spin_avg.setRange(0,20); self.spin_avg.setValue(0)
        ar.addWidget(self.spin_avg); sl.addLayout(ar)
        ll.addWidget(sg)

        pg = QGroupBox("Welch Parameters"); pl = QVBoxLayout(pg)
        Nt = self.dataset["Nt"]
        r1 = QHBoxLayout(); r1.addWidget(QLabel("Segment (pts):"))
        self.spin_nperseg = QSpinBox(); self.spin_nperseg.setRange(4,10000)
        _default_seg = max(16, min(256, Nt//2))
        self.spin_nperseg.setValue(_default_seg)
        r1.addWidget(self.spin_nperseg); pl.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Overlap (pts):"))
        self.spin_overlap = QSpinBox(); self.spin_overlap.setRange(0,10000)
        self.spin_overlap.setValue(_default_seg // 2)
        r2.addWidget(self.spin_overlap); pl.addLayout(r2)
        self.chk_subtract = QCheckBox("Subtract spatial mean"); self.chk_subtract.setChecked(True)
        pl.addWidget(self.chk_subtract); ll.addWidget(pg)

        dg = QGroupBox("Display"); dl = QVBoxLayout(dg)
        self.chk_kolmogorov = QCheckBox("Show -5/3 slope"); self.chk_kolmogorov.setChecked(True)
        dl.addWidget(self.chk_kolmogorov)
        cr = QHBoxLayout()
        self.chk_compensate = QCheckBox("Compensate k^α, α ="); self.chk_compensate.setChecked(False)
        self.spin_alpha = QDoubleSpinBox(); self.spin_alpha.setRange(0.0,5.0)
        self.spin_alpha.setValue(5/3); self.spin_alpha.setDecimals(2)
        self.spin_alpha.setSingleStep(0.1); self.spin_alpha.setFixedWidth(60)
        cr.addWidget(self.chk_compensate); cr.addWidget(self.spin_alpha); dl.addLayout(cr)
        ll.addWidget(dg)

        cg = QGroupBox("Components"); cl = QHBoxLayout(cg)
        self.chk_u = QCheckBox("u"); self.chk_u.setChecked(True)
        self.chk_v = QCheckBox("v"); self.chk_v.setChecked(True)
        self.chk_w = QCheckBox("w")
        self.chk_w.setChecked(self.dataset["is_stereo"])
        self.chk_w.setEnabled(self.dataset["is_stereo"])
        cl.addWidget(self.chk_u); cl.addWidget(self.chk_v); cl.addWidget(self.chk_w)
        ll.addWidget(cg)
        return w

    def _build_temporal_tab(self):
        w = QWidget(); ll = QVBoxLayout(w)
        ll.setContentsMargins(4,4,4,4); ll.setSpacing(4)

        ll.addWidget(QLabel("Left-click: point.  Right-click drag: rectangle ROI."))

        sg = QGroupBox("Selection Mode"); sl = QHBoxLayout(sg)
        self.rb_temp_point = QRadioButton("Point"); self.rb_temp_point.setChecked(True)
        self.rb_temp_rect  = QRadioButton("Rectangle")
        bg2 = QButtonGroup(self)
        for rb in [self.rb_temp_point, self.rb_temp_rect]:
            bg2.addButton(rb); sl.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)   # was missing -- mode never updated
        ll.addWidget(sg)

        pg = QGroupBox("Parameters"); pl = QVBoxLayout(pg)
        Nt = self.dataset["Nt"]
        r1 = QHBoxLayout(); r1.addWidget(QLabel("Segment (pts):"))
        self.spin_temp_nperseg = QSpinBox(); self.spin_temp_nperseg.setRange(4,100000)
        self.spin_temp_nperseg.setValue(max(16, min(Nt//4, 4096)))
        r1.addWidget(self.spin_temp_nperseg); pl.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Overlap (pts):"))
        self.spin_temp_overlap = QSpinBox(); self.spin_temp_overlap.setRange(0,100000)
        self.spin_temp_overlap.setValue(max(8, min(Nt//8, 2048)))
        r2.addWidget(self.spin_temp_overlap); pl.addLayout(r2)
        fsr = QHBoxLayout(); fsr.addWidget(QLabel("fs [Hz]:"))
        self.spin_temp_fs = QDoubleSpinBox(); self.spin_temp_fs.setRange(1,1e6)
        self.spin_temp_fs.setValue(self._fs); self.spin_temp_fs.setDecimals(1)
        fsr.addWidget(self.spin_temp_fs); pl.addLayout(fsr)
        self.chk_temp_kolmogorov = QCheckBox("Show -5/3 slope"); self.chk_temp_kolmogorov.setChecked(True)
        pl.addWidget(self.chk_temp_kolmogorov)
        ll.addWidget(pg)

        cg = QGroupBox("Components"); cl = QHBoxLayout(cg)
        self.chk_temp_u = QCheckBox("u"); self.chk_temp_u.setChecked(True)
        self.chk_temp_v = QCheckBox("v"); self.chk_temp_v.setChecked(True)
        self.chk_temp_w = QCheckBox("w")
        self.chk_temp_w.setChecked(self.dataset["is_stereo"])
        self.chk_temp_w.setEnabled(self.dataset["is_stereo"])
        cl.addWidget(self.chk_temp_u); cl.addWidget(self.chk_temp_v); cl.addWidget(self.chk_temp_w)
        ll.addWidget(cg)
        return w

    def _build_3d_spatial_tab(self):
        w = QWidget(); ll = QVBoxLayout(w)
        ll.setContentsMargins(4,4,4,4); ll.setSpacing(4)

        lbl_fft = QLabel(
            "FFT: computes a single 2D FFT over the full ROI for each snapshot, "
            "then averages over time. Noisier but retains full spatial resolution."
        )
        lbl_fft.setWordWrap(True)
        lbl_fft.setStyleSheet("color: gray;")
        ll.addWidget(lbl_fft)

        info = QLabel(
            "Computes 2D spatial spectra (kx, ky) using pyFFTW.\n"
            "Draw a rectangular ROI on the field to begin."
        )
        info.setWordWrap(True)
        ll.addWidget(info)

        # Components
        cg = QGroupBox("Components"); cl = QHBoxLayout(cg)
        self.chk_3d_u = QCheckBox("u"); self.chk_3d_u.setChecked(True)
        self.chk_3d_v = QCheckBox("v"); self.chk_3d_v.setChecked(True)
        self.chk_3d_w = QCheckBox("w")
        self.chk_3d_w.setChecked(self.dataset["is_stereo"])
        self.chk_3d_w.setEnabled(self.dataset["is_stereo"])
        cl.addWidget(self.chk_3d_u); cl.addWidget(self.chk_3d_v); cl.addWidget(self.chk_3d_w)
        ll.addWidget(cg)

        # Display options
        dg2 = QGroupBox("Display"); dl2 = QVBoxLayout(dg2)
        self.chk_3d_kolmogorov = QCheckBox("Show -5/3 slope"); self.chk_3d_kolmogorov.setChecked(True)
        dl2.addWidget(self.chk_3d_kolmogorov)
        
        cr = QHBoxLayout()
        self.chk_3d_compensate = QCheckBox("Compensate k^α, α ="); self.chk_3d_compensate.setChecked(False)
        self.spin_3d_alpha = QDoubleSpinBox(); self.spin_3d_alpha.setRange(0.0,5.0)
        self.spin_3d_alpha.setValue(5/3); self.spin_3d_alpha.setDecimals(2)
        self.spin_3d_alpha.setSingleStep(0.1); self.spin_3d_alpha.setFixedWidth(60)
        cr.addWidget(self.chk_3d_compensate); cr.addWidget(self.spin_3d_alpha); dl2.addLayout(cr)
        ll.addWidget(dg2)

        return w

    def _build_st_tab(self):
        w = QWidget(); ll = QVBoxLayout(w)
        ll.setContentsMargins(4,4,4,4); ll.setSpacing(4)

        dg = QGroupBox("Line Direction"); dl = QHBoxLayout(dg)
        self.rb_st_horiz = QRadioButton("Horizontal (kx)"); self.rb_st_horiz.setChecked(True)
        self.rb_st_vert  = QRadioButton("Vertical (ky)")
        bg3 = QButtonGroup(self)
        for rb in [self.rb_st_horiz, self.rb_st_vert]:
            bg3.addButton(rb); dl.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        ll.addWidget(dg)

        ar = QHBoxLayout(); ar.addWidget(QLabel("Spatial avg ± pts:"))
        self.spin_st_avg = QSpinBox(); self.spin_st_avg.setRange(0,20); self.spin_st_avg.setValue(0)
        ar.addWidget(self.spin_st_avg); ll.addLayout(ar)

        fsr = QHBoxLayout(); fsr.addWidget(QLabel("fs [Hz]:"))
        self.spin_st_fs = QDoubleSpinBox(); self.spin_st_fs.setRange(1,1e6)
        self.spin_st_fs.setValue(self._fs); self.spin_st_fs.setDecimals(1)
        fsr.addWidget(self.spin_st_fs); ll.addLayout(fsr)

        cg = QGroupBox("Convection Velocity Overlay"); cl = QVBoxLayout(cg)
        self.chk_uc = QCheckBox("Show Uc line"); self.chk_uc.setChecked(False)
        cl.addWidget(self.chk_uc)
        ucr = QHBoxLayout(); ucr.addWidget(QLabel("Uc [m/s]:"))
        self.spin_uc = QDoubleSpinBox(); self.spin_uc.setRange(0.001,1000)
        self.spin_uc.setValue(1.0); self.spin_uc.setDecimals(3)
        ucr.addWidget(self.spin_uc); cl.addLayout(ucr); ll.addWidget(cg)

        cg2 = QGroupBox("Component"); cl2 = QHBoxLayout(cg2)
        self.rb_st_u = QRadioButton("u"); self.rb_st_u.setChecked(True)
        self.rb_st_v = QRadioButton("v")
        self.rb_st_w = QRadioButton("w"); self.rb_st_w.setEnabled(self.dataset["is_stereo"])
        bg4 = QButtonGroup(self)
        for rb in [self.rb_st_u, self.rb_st_v, self.rb_st_w]:
            bg4.addButton(rb); cl2.addWidget(rb)
        ll.addWidget(cg2)
        return w

    # ----------------------------------------------------------------------- #
    # Field plot
    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds = self.dataset; x, y = ds["x"], ds["y"]
        speed = np.sqrt(np.nanmean(ds["U"],axis=2)**2 + np.nanmean(ds["V"],axis=2)**2)
        vf = np.mean(ds["valid"],axis=2); speed[vf<0.5] = np.nan
        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
        self.field_ax.set_xlabel("x [mm]", fontsize=9)
        self.field_ax.set_ylabel("y [mm]", fontsize=9)
        # No set_aspect("equal") on field canvas -- see tke_budget_window comment
        self.field_ax.set_facecolor("white")
        self.field_ax.tick_params(labelsize=8)
        self.field_fig.tight_layout(pad=0.3)
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()
        self._x = x; self._y = y; self._last_field_values = speed

    # ----------------------------------------------------------------------- #
    # Mode / Tab
    # ----------------------------------------------------------------------- #

    def _on_tab_changed(self, idx):
        self._clear_artist(); self._selection = None
        # For 3D tab, compute button enabled only with selection
        if idx == 1:
            self.btn_compute.setEnabled(False)  # Require ROI selection
        else:
            self.btn_compute.setEnabled(False)
        self._update_hint()

    def _on_mode_changed(self):
        self._clear_artist(); self._selection = None
        # For 3D tab, compute button enabled only with selection
        if self._current_tab() == 1:
            self.btn_compute.setEnabled(False)  # Require ROI selection
        else:
            self.btn_compute.setEnabled(False)
        self._update_hint()

    def _current_tab(self):
        return self.tabs.currentIndex()  # 0=spatial, 1=temporal, 2=ST

    def _update_hint(self):
        tab = self._current_tab()
        if tab == 0:
            self._mode = "horizontal" if self.rb_horiz.isChecked() else "vertical"
            hints = {"horizontal": "Drag horizontally to select a line; use ± pts to average over a band.",
                     "vertical":   "Drag vertically to select a line; use ± pts to average over a band."}
            self.lbl_hint.setText(hints[self._mode])
        elif tab == 1:
            self._mode = "3d_roi"
            self.lbl_hint.setText("Right-click drag to draw ROI for 3D spectral analysis.")
        elif tab == 2:
            if self.rb_temp_point.isChecked():
                self._mode = "temp_point"
                self.lbl_hint.setText("Left-click to pick a point.")
            else:
                self._mode = "temp_rect"
                self.lbl_hint.setText("Right-click drag to draw a rectangle.")
        else:
            self._mode = "st_horiz" if self.rb_st_horiz.isChecked() else "st_vert"
            self.lbl_hint.setText("Drag to select a line for E(k,f).")

    # ----------------------------------------------------------------------- #
    # Mouse
    # ----------------------------------------------------------------------- #

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.field_ax: return
        if self._toolbar_active(self.field_toolbar): return

        # Left click -- point or line selection
        if event.button == 1:
            if self._mode == "temp_point":
                self._clear_artist()
                dot, = self.field_ax.plot(event.xdata, event.ydata,
                                           "r+", markersize=14,
                                           markeredgewidth=2, zorder=20)
                self._artist = dot; self.field_canvas.draw()
                row, col = np.unravel_index(
                    np.argmin((self._x-event.xdata)**2 + (self._y-event.ydata)**2),
                    self._x.shape)
                self._selection = {"type":"temp_point","row":row,"col":col,
                                   "xd":event.xdata,"yd":event.ydata}
                self.lbl_hint.setText(f"Point: ({event.xdata:.1f}, {event.ydata:.1f}) mm")
                self.btn_compute.setEnabled(True)
            elif self._mode not in ("temp_rect", "3d_roi"):
                self._press_xy = (event.xdata, event.ydata)
            elif self._mode == "3d_roi":
                self._press_xy = (event.xdata, event.ydata)

        # Right click -- rectangle (temporal rect or 3D ROI)
        elif event.button == 3:
            if self._mode in ("temp_rect", "3d_roi"):
                self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.field_ax: return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None; return
        x0, y0 = self._press_xy; x1, y1 = event.xdata, event.ydata
        self._clear_artist()
        if self._mode in ("temp_rect", "3d_roi"):  # rect modes -- yellow
            p = MplRect((min(x0,x1),min(y0,y1)),abs(x1-x0),abs(y1-y0),
                        linewidth=1.5,edgecolor="#e8a000",facecolor="#ffe066",
                        alpha=0.25,linestyle="--",zorder=10)
            self.field_ax.add_patch(p); self._artist = p
        else:
            lm = ("horizontal" if self._mode in ("horizontal","st_horiz") else "vertical")
            lx0,ly0,lx1,ly1 = compute_snapped_line(self._x,self._y,x0,y0,x1,y1,lm)
            ln, = self.field_ax.plot([lx0,lx1],[ly0,ly1],"r-",linewidth=2,zorder=10)
            self._artist = ln
        self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None: return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None; return
        if event.inaxes != self.field_ax:
            self._press_xy = None; return
        x0, y0 = self._press_xy; x1, y1 = event.xdata, event.ydata
        self._press_xy = None

        if self._mode in ("temp_rect", "3d_roi"):
            if abs(x1-x0)<1 or abs(y1-y0)<1:
                self.lbl_hint.setText("Too small -- try again."); return
            self._selection = {"type":"rect","x0":x0,"x1":x1,"y0":y0,"y1":y1}
            self.lbl_hint.setText(
                f"Rect: x=[{min(x0,x1):.1f},{max(x0,x1):.1f}] "
                f"y=[{min(y0,y1):.1f},{max(y0,y1):.1f}] mm")
            # Redraw committed
            self._clear_artist()
            x0r,x1r=min(x0,x1),max(x0,x1); y0r,y1r=min(y0,y1),max(y0,y1)
            p=MplRect((x0r,y0r),x1r-x0r,y1r-y0r,linewidth=2,
                      edgecolor="#e8a000",facecolor="#ffe066",alpha=0.25,
                      linestyle="--",zorder=10)
            self.field_ax.add_patch(p); self._artist=p; self.field_canvas.draw()
        else:
            lm = ("horizontal" if self._mode in ("horizontal","st_horiz") else "vertical")
            lx0,ly0,lx1,ly1 = compute_snapped_line(self._x,self._y,x0,y0,x1,y1,lm)
            if abs(lx1-lx0)<1 and abs(ly1-ly0)<1:
                self.lbl_hint.setText("Too short -- try again."); return
            self._selection = {"type":lm,"x0":lx0,"y0":ly0,"x1":lx1,"y1":ly1}
            self.lbl_hint.setText(f"Line ({lm}): ({lx0:.1f},{ly0:.1f})->({lx1:.1f},{ly1:.1f}) mm")
            self._clear_artist()
            ln,=self.field_ax.plot([lx0,lx1],[ly0,ly1],"r-",linewidth=2,zorder=10)
            self._artist=ln; self.field_canvas.draw()

        self.btn_compute.setEnabled(True)
        if self._current_tab() == 0:
            self._auto_set_welch(self._selection)

    def _auto_set_welch(self, sel):
        """Auto-set nperseg / noverlap based on selection geometry (spatial tab only)."""
        x = self._x; y = self._y
        if sel["type"] == "horizontal":
            x0, x1 = min(sel["x0"], sel["x1"]), max(sel["x0"], sel["x1"])
            N = int(np.sum((x[0, :] >= x0) & (x[0, :] <= x1)))
        else:  # vertical
            y0, y1 = min(sel["y0"], sel["y1"]), max(sel["y0"], sel["y1"])
            N = int(np.sum((y[:, 0] >= y0) & (y[:, 0] <= y1)))

        if N < 8:
            return  # too few points to suggest anything useful

        # nperseg = N//2 (2 non-overlapping windows minimum); floor at 8
        nperseg = max(8, N // 2)
        noverlap = nperseg // 2

        self.spin_nperseg.setValue(nperseg)
        self.spin_overlap.setValue(noverlap)
        self.lbl_status.setText(f"Auto: N={N}  seg={nperseg}  ovlp={noverlap}")

    def _clear_artist(self):
        if self._artist is not None:
            try: self._artist.remove()
            except: pass
            self._artist = None
        self.field_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Compute
    # ----------------------------------------------------------------------- #

    def _on_compute(self):
        if self._selection is None and self._current_tab() != 1:  # Only 3D tab requires selection
            return
        self.result_fig.clear(); self.result_canvas.draw()
        self.lbl_status.setText("⏳ Busy: computing...")
        self.btn_compute.setEnabled(False)
        QApplication.processEvents()
        try:
            tab = self._current_tab()
            if tab == 0:   self._compute_spatial()
            elif tab == 1: self._compute_3d_spatial()
            elif tab == 2: self._compute_temporal()
            else:          self._compute_st()
            self.btn_export.setEnabled(True)
            self.lbl_status.setText("✓ Done. Draw new selection to recompute.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e) + "\nTraceback:\n" + traceback.format_exc())
            self.lbl_status.setText(f"Error: {e}")
        finally:
            self.btn_compute.setEnabled(True)

    # ---- Spatial ----
    def _compute_spatial(self):
        ds=self.dataset; sel=self._selection
        nperseg=self.spin_nperseg.value(); noverlap=self.spin_overlap.value()
        subtract=self.chk_subtract.isChecked(); avg_band=self.spin_avg.value()
        if noverlap>=nperseg: raise ValueError("Overlap must be less than segment length.")
        direction="x" if sel["type"]=="horizontal" else "y"
        k,psds=spatial_psd_line(
            ds["U"],ds["V"],ds["W"],self._x,self._y,
            sel["x0"],sel["y0"],sel["x1"],sel["y1"],
            direction,avg_band,nperseg,noverlap,subtract)
        self._last_result={"tab":"spatial","type":"line","direction":direction,"k":k,"psds":psds}
        self._plot_spatial_line(k,psds,direction)

    # ---- 3D Spatial ----
    def _compute_3d_spatial(self):
        if not PYFFTW_AVAILABLE:
            raise ValueError("pyFFTW is not installed. Run: pip install pyfftw")
        ds=self.dataset
        sel=self._selection
        Lz=1.0  # fixed; kz is not plotted or exported for 2D PIV

        if sel is None or sel["type"] != "rect":
            raise ValueError("Please draw a rectangle ROI for 3D spectral analysis.")
        
        # Get ROI coordinates and compute domain sizes
        x0, x1 = min(sel["x0"], sel["x1"]), max(sel["x0"], sel["x1"])
        y0, y1 = min(sel["y0"], sel["y1"]), max(sel["y0"], sel["y1"])
        
        # Get the actual grid points within the ROI
        cols = np.where((self._x[0, :] >= x0) & (self._x[0, :] <= x1))[0]
        rows = np.where((self._y[:, 0] >= y0) & (self._y[:, 0] <= y1))[0]
        
        if len(cols) == 0 or len(rows) == 0:
            raise ValueError("ROI is too small or outside the field.")
        
        # Compute domain sizes from ROI
        Lx = abs(x1 - x0) / 1000.0  # Convert mm to m
        Ly = abs(y1 - y0) / 1000.0  # Convert mm to m
        
        # Extract velocity data within ROI
        U_roi = ds["U"][rows[0]:rows[-1]+1, cols[0]:cols[-1]+1, :].copy()
        V_roi = ds["V"][rows[0]:rows[-1]+1, cols[0]:cols[-1]+1, :].copy()

        # Handle W component if available
        if ds["W"] is not None:
            W_roi = ds["W"][rows[0]:rows[-1]+1, cols[0]:cols[-1]+1, :].copy()
        else:
            W_roi = np.zeros_like(U_roi)

        # Diagnostic: count NaN / masked points
        ny_roi, nx_roi, nt = U_roi.shape
        nan_mask = np.isnan(U_roi) | np.isnan(V_roi)
        if ds["W"] is not None:
            nan_mask |= np.isnan(W_roi)
        total_pts = ny_roi * nx_roi * nt
        n_nan    = int(np.sum(nan_mask))
        n_valid  = total_pts - n_nan
        mask_pct = 100.0 * n_nan / total_pts
        print(f"[FFT Spectra] ROI: x=[{x0:.1f}, {x1:.1f}] mm  y=[{y0:.1f}, {y1:.1f}] mm")
        print(f"[FFT Spectra] Grid: {ny_roi}×{nx_roi} × {nt} snapshots = {total_pts} pts")
        print(f"[FFT Spectra] Valid: {n_valid}/{total_pts}  ({100-mask_pct:.1f}% valid, {mask_pct:.1f}% masked)")

        if n_valid < total_pts * 0.5:
            raise ValueError(
                f"{mask_pct:.1f}% of ROI points are masked/NaN — too few valid vectors. "
                "Select a region with more valid data."
            )

        # Fill NaNs with zero so FFT can proceed; warn in the plot
        if n_nan > 0:
            U_roi[nan_mask] = 0.0
            V_roi[nan_mask] = 0.0
            W_roi[nan_mask] = 0.0
        U_4d = U_roi.reshape(1, ny_roi, nx_roi, nt)
        V_4d = V_roi.reshape(1, ny_roi, nx_roi, nt)
        W_4d = W_roi.reshape(1, ny_roi, nx_roi, nt)
        
        # Subtract temporal mean to get fluctuations
        U_fluct, V_fluct, W_fluct = subtract_temporal_mean(U_4d, V_4d, W_4d)
        
        # Compute spectra using FFT
        result = compute_spectra_from_fluctuations(U_fluct, V_fluct, W_fluct, Lx, Ly, Lz)
        
        self._last_result={"tab":"3d_spatial","result":result,
                           "roi": {"x0":x0,"x1":x1,"y0":y0,"y1":y1},
                           "mask_pct": mask_pct}
        self._plot_spatial_fft(result, mask_pct=mask_pct)

    # ---- Temporal ----
    def _compute_temporal(self):
        ds=self.dataset; sel=self._selection
        fs=self.spin_temp_fs.value()
        nperseg=self.spin_temp_nperseg.value(); noverlap=self.spin_temp_overlap.value()
        if noverlap>=nperseg: raise ValueError("Overlap must be less than segment length.")
        W=ds["W"]
        if sel["type"]=="temp_point":
            freq,psds=psd_at_point(ds["U"],ds["V"],W,
                                   sel["row"],sel["col"],fs,nperseg,noverlap)
            self._last_result={"tab":"temporal","type":"point","freq":freq,"psds":psds}
            self._plot_temporal(freq,psds,f"Point ({sel['xd']:.1f}, {sel['yd']:.1f}) mm")
        else:  # rect
            freq,psds,npts=psd_in_region(ds["U"],ds["V"],W,
                                          self._x,self._y,
                                          sel["x0"],sel["x1"],sel["y0"],sel["y1"],
                                          fs,nperseg,noverlap)
            self._last_result={"tab":"temporal","type":"rect","freq":freq,"psds":psds}
            self._plot_temporal(freq,psds,f"Rectangle avg ({npts} pts)")

    # ---- Spatiotemporal ----
    def _compute_st(self):
        ds=self.dataset; sel=self._selection
        fs=self.spin_st_fs.value(); avg=self.spin_st_avg.value()
        direction="x" if self._mode=="st_horiz" else "y"
        k,f,psds=compute_st_spectra(
            ds["U"],ds["V"],ds["W"],self._x,self._y,
            sel["x0"],sel["y0"],sel["x1"],sel["y1"],
            direction,avg,fs)
        self._last_result={"tab":"st","direction":direction,"k":k,"f":f,"psds":psds}
        self._plot_st(k,f,psds,direction)

    # ----------------------------------------------------------------------- #
    # Plot helpers
    # ----------------------------------------------------------------------- #

    def _active_comps_spatial(self):
        c=[]
        if self.chk_u.isChecked(): c.append("u")
        if self.chk_v.isChecked(): c.append("v")
        if self.chk_w.isChecked() and self.dataset["is_stereo"]: c.append("w")
        return c

    def _active_comps_temporal(self):
        c=[]
        if self.chk_temp_u.isChecked(): c.append("u")
        if self.chk_temp_v.isChecked(): c.append("v")
        if self.chk_temp_w.isChecked() and self.dataset["is_stereo"]: c.append("w")
        return c

    def _plot_psd_ax(self, ax, k, psd, label, color, x_label="k [rad/m]",
                     compensate=False, alpha_exp=5/3, show_kolmogorov=None):
        """
        show_kolmogorov: override the checkbox; None means use self.chk_kolmogorov.
        """
        if k is None or psd is None:
            ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center",va="center")
            ax.set_title(label,fontsize=9); return
        mask=k>0; kp=k[mask]; p=psd[mask]
        if compensate and np.any(kp>0):
            p_plot=p*kp**alpha_exp
            y_label=f"k^{alpha_exp:.2f}·PSD"
        else:
            p_plot=p; y_label="PSD [(m/s)\u00b2/(rad/m)]" if "rad/m" in x_label else "PSD [(m/s)\u00b2/Hz]"
        valid=np.isfinite(p_plot)&(p_plot>0)
        if not np.any(valid):
            ax.text(0.5,0.5,"No valid data",transform=ax.transAxes,ha="center",va="center"); return
        ax.loglog(kp[valid],p_plot[valid],color=color,linewidth=1.2,label=label)
        do_kolm = self.chk_kolmogorov.isChecked() if show_kolmogorov is None else show_kolmogorov
        if do_kolm:
            nv=np.sum(valid); ilo=max(0,int(nv*0.10)); ihi=min(nv-1,int(nv*0.60))
            if ihi>ilo+2:
                klo=kp[valid][ilo]; khi=kp[valid][ihi]
                kl=np.logspace(np.log10(klo),np.log10(khi),50)
                pa=p_plot[valid][ilo]
                slope=-5/3+alpha_exp if compensate else -5/3
                ax.loglog(kl,pa*(kl/klo)**slope,"k--",linewidth=1.5,alpha=0.7,
                          label=f"$k^{{{slope:.2f}}}$")
        ax.set_xlabel(x_label,fontsize=8); ax.set_ylabel(y_label,fontsize=7)
        ax.set_title(label,fontsize=9); ax.tick_params(labelsize=7)
        ax.set_aspect("auto")   # never let loglog impose equal aspect
        ax.grid(True,which="both",alpha=0.3); ax.legend(fontsize=7)

    def _plot_spatial_line(self,k,psds,direction):
        comps=[c for c in self._active_comps_spatial() if psds.get(c) is not None]
        if not comps: self.lbl_status.setText("No valid spectra."); return
        colors={"u":"tab:blue","v":"tab:orange","w":"tab:green"}
        dl="x" if direction=="x" else "y"
        n=len(comps)
        # Always landscape: all subplots in one row
        comp=self.chk_compensate.isChecked(); alpha=self.spin_alpha.value()
        self.result_fig.clear()
        for i,c in enumerate(comps):
            ax=self.result_fig.add_subplot(1,n,i+1)
            self._plot_psd_ax(ax,k,psds[c],f"E_{c}(k_{dl})",colors[c],
                              compensate=comp,alpha_exp=alpha)
        if self.chk_hide_axes.isChecked():
            for a in self.result_fig.axes:
                a.axis('off')
        self.result_fig.tight_layout(pad=1.2); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

    def _plot_spatial_fft(self, result, mask_pct=0.0):
        # Get active components
        comps = []
        if self.chk_3d_u.isChecked(): comps.append("u")
        if self.chk_3d_v.isChecked(): comps.append("v")
        if self.chk_3d_w.isChecked() and self.dataset["is_stereo"]: comps.append("w")
        
        if not comps:
            self.lbl_status.setText("No components selected.")
            return
            
        colors={"u":"tab:blue","v":"tab:orange","w":"tab:green"}
        comp=self.chk_3d_compensate.isChecked(); alpha=self.spin_3d_alpha.value()
        
        # Plot 3D spectrum
        k_3d = result.get('k_3d')
        spectrum_3d = result.get('spectrum_3d')
        
        if k_3d is not None and spectrum_3d is not None:
            self.result_fig.clear()
            ax1 = self.result_fig.add_subplot(121)
            
            # Determine spectrum label based on stereo mode
            is_stereo = self.dataset["is_stereo"]
            if is_stereo:
                spectrum_label = "2D3C Spectrum E(k)"
            else:
                spectrum_label = "2D Spectrum E(k)"

            # Get ROI info for suptitle if available
            suptitle = spectrum_label
            if hasattr(self, '_last_result') and self._last_result.get("tab") == "3d_spatial":
                roi = self._last_result.get("roi", {})
                if roi:
                    suptitle = (
                        f"{spectrum_label} - ROI: x=[{roi['x0']:.1f}, {roi['x1']:.1f}] mm, "
                        f"y=[{roi['y0']:.1f}, {roi['y1']:.1f}] mm"
                    )

            self._plot_psd_ax(ax1, k_3d, spectrum_3d, spectrum_label, "black",
                             compensate=comp, alpha_exp=alpha,
                             show_kolmogorov=self.chk_3d_kolmogorov.isChecked())
            self.result_fig.suptitle(suptitle)
            
            # Plot 1D spectra
            ax2 = self.result_fig.add_subplot(122)
            
            # Plot each component's 1D spectra (kx and ky only)
            for i, c in enumerate(comps):
                kx = result.get('kx')
                spectrum_kx = result.get(f'{c}_kx')
                if kx is not None and spectrum_kx is not None:
                    ax2.loglog(kx, spectrum_kx, color=colors[c], linewidth=1.2,
                              label=f'E_{c}(k_x)', alpha=0.8)

                ky = result.get('ky')
                spectrum_ky = result.get(f'{c}_ky')
                if ky is not None and spectrum_ky is not None:
                    ax2.loglog(ky, spectrum_ky, '--', color=colors[c], linewidth=1.2,
                              label=f'E_{c}(k_y)', alpha=0.8)

            # -5/3 reference line based on kx range
            if self.chk_3d_kolmogorov.isChecked():
                kx = result.get('kx')
                if kx is not None:
                    kx_valid = kx[kx > 0]
                    if len(kx_valid) > 4:
                        nv = len(kx_valid)
                        ilo = max(0, int(nv * 0.10)); ihi = min(nv - 1, int(nv * 0.60))
                        if ihi > ilo + 2:
                            klo = kx_valid[ilo]; khi = kx_valid[ihi]
                            kl = np.logspace(np.log10(klo), np.log10(khi), 50)
                            # anchor at median line visible in the plot
                            ax2_lines = ax2.get_lines()
                            if ax2_lines:
                                y_anchor = ax2_lines[0].get_ydata()
                                x_anchor = ax2_lines[0].get_xdata()
                                x_anchor = x_anchor[np.isfinite(x_anchor) & (x_anchor > 0)]
                                y_anchor = y_anchor[np.isfinite(y_anchor) & (y_anchor > 0)]
                                if len(x_anchor) > 0 and len(y_anchor) > 0:
                                    idx0 = np.searchsorted(x_anchor, klo)
                                    idx0 = min(idx0, len(y_anchor) - 1)
                                    pa = y_anchor[idx0]
                                    ax2.loglog(kl, pa * (kl / klo) ** (-5/3), "k--",
                                               linewidth=1.5, alpha=0.7, label="$k^{-5/3}$")

            ax2.set_xlabel("k [rad/m]", fontsize=8)
            ax2.set_ylabel("PSD [(m/s)^2/(rad/m)]", fontsize=7)
            ax2.set_title("1D Spectra E(kx), E(ky)", fontsize=9)
            ax2.tick_params(labelsize=7)
            ax2.grid(True, which="both", alpha=0.3)
            ax2.legend(fontsize=6)
            ax2.set_aspect("auto")

        if mask_pct > 5.0:
            self.lbl_mask_warning.setText(f"{mask_pct:.1f}% of ROI was masked and filled with 0 before FFT.")
            self.lbl_mask_warning.setVisible(True)
        else:
            self.lbl_mask_warning.setVisible(False)

        if self.chk_hide_axes.isChecked():
            for a in self.result_fig.axes:
                a.axis('off')
        
        self.result_fig.tight_layout(pad=1.5); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

    def _plot_spatial_roi(self,results,n_lines):
        comps=self._active_comps_spatial()
        colors={"u":"tab:blue","v":"tab:orange","w":"tab:green"}
        pairs=[]
        for d in ["x","y"]:
            k=results[d]["k"]
            for c in comps:
                p=results[d]["psds"].get(c)
                if p is not None and k is not None:
                    pairs.append((k,p,f"E_{c}(k_{d}) [{n_lines[d]}L]",colors[c]))
        if not pairs: self.lbl_status.setText("No valid spectra."); return
        comp=self.chk_compensate.isChecked(); alpha=self.spin_alpha.value()
        n=len(pairs)
        # Landscape: at most 3 per row, wrap to 2 rows if needed
        ncols=min(n,3); nrows=(n+ncols-1)//ncols
        self.result_fig.clear()
        for i,(k_,p_,lbl,col) in enumerate(pairs):
            ax=self.result_fig.add_subplot(nrows,ncols,i+1)
            self._plot_psd_ax(ax,k_,p_,lbl,col,compensate=comp,alpha_exp=alpha)
        if self.chk_hide_axes.isChecked():
            for a in self.result_fig.axes:
                a.axis('off')
        self.result_fig.tight_layout(pad=1.2); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

    def _plot_temporal(self,freq,psds,title_suffix):
        comps=self._active_comps_temporal()
        colors={"u":"tab:blue","v":"tab:orange","w":"tab:green"}
        valid_comps=[c for c in comps if psds.get(c) is not None]
        if not valid_comps: return
        n=len(valid_comps)
        show_k=self.chk_temp_kolmogorov.isChecked()
        self.result_fig.clear()
        for i,c in enumerate(valid_comps):
            ax=self.result_fig.add_subplot(1,n,i+1)
            self._plot_psd_ax(ax,freq,psds[c],f"E_{c}(f)  {title_suffix}",
                              colors[c],x_label="f [Hz]",
                              compensate=False,alpha_exp=5/3,
                              show_kolmogorov=show_k)
        if self.chk_hide_axes.isChecked():
            for a in self.result_fig.axes:
                a.axis('off')
        self.result_fig.tight_layout(pad=1.2); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

    def _plot_st(self,k,f,psds,direction):
        comp="u" if self.rb_st_u.isChecked() else ("v" if self.rb_st_v.isChecked() else "w")
        E=psds.get(comp)
        if k is None or f is None or E is None:
            self.lbl_status.setText("No valid spatiotemporal spectrum."); return
        dl="x" if direction=="x" else "y"
        k_plot=k[1:]; f_plot=f[1:]; E_plot=E[1:,1:]
        E_plot=np.where(E_plot>0,E_plot,np.nan)
        logE=np.log10(E_plot)
        self.result_fig.clear()
        ax=self.result_fig.add_subplot(111)
        pcm=ax.pcolormesh(k_plot,f_plot,logE.T,cmap="inferno",shading="nearest")
        cb = self.result_fig.colorbar(pcm,ax=ax,
                                      label=r"$\log_{10}$ E(k,f) [(m/s)²/(rad/m·Hz)]")
        if self.chk_hide_colorbar.isChecked():
            cb.remove()
            self.result_fig.tight_layout(pad=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"k_{dl} [rad/m]",fontsize=10)
        ax.set_ylabel("f [Hz]",fontsize=10)
        ax.set_title(f"Spatiotemporal spectrum E_{comp}(k_{dl},f)",fontsize=10)
        ax.set_aspect("auto")   # log-log can otherwise impose square
        ax.grid(True,which="both",alpha=0.2)
        if self.chk_uc.isChecked():
            Uc=self.spin_uc.value()
            k_line=k_plot[k_plot>0]
            f_line=Uc*k_line/(2*np.pi)
            valid=(f_line>=f_plot[0])&(f_line<=f_plot[-1])
            if np.any(valid):
                ax.plot(k_line[valid],f_line[valid],"w--",linewidth=1.5,
                        alpha=0.8,label=f"Uc={Uc:.2f} m/s")
                ax.legend(fontsize=9)
        if self.chk_hide_axes.isChecked():
            ax.axis('off')
            ax.set_title('')
        self.result_fig.tight_layout(pad=0.5); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

    # ----------------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------------- #

    def _on_export(self):
        if self._last_result is None: return
        res=self._last_result
        if res["tab"]=="spatial":
            default="spatial_spectra_all.csv"
        elif res["tab"]=="3d_spatial":
            default="3d_spatial_spectra.csv"
        elif res["tab"]=="temporal":
            default="temporal_spectra_all.csv"
        else:
            default="spatiotemporal_spectra_all.csv"
        path,_=QFileDialog.getSaveFileName(self,"Export",default,"CSV (*.csv)")
        if not path: return
        settings={"Analysis":"Spectral","Snapshots":self.dataset["Nt"]}
        if res["tab"]=="spatial":
            export_spectra_csv(path,res["k"],res["psds"],settings)
            n=len([p for p in res["psds"].values() if p is not None])
            self.lbl_status.setText(f"✓ Exported {n} component spectra to {os.path.basename(path)}")
        elif res["tab"]=="3d_spatial":
            # Export 3D spatial spectra
            result = res["result"]
            roi = res.get("roi", {})
            rows = ["# 3D Spatial Spectra (FFT-based)", "# Generated by uPrime"]
            rows.append(f"# Snapshots: {self.dataset['Nt']}")
            if roi:
                rows.append(f"# ROI: x=[{roi['x0']:.1f}, {roi['x1']:.1f}] mm, y=[{roi['y0']:.1f}, {roi['y1']:.1f}] mm")
                Lx = abs(roi['x1'] - roi['x0']) / 1000.0
                Ly = abs(roi['y1'] - roi['y0']) / 1000.0
                rows.append(f"# Domain: Lx={Lx:.4f}m, Ly={Ly:.4f}m")
            rows.append("")
            
            # Export 3D spectrum
            k_3d = result.get('k_3d')
            spectrum_3d = result.get('spectrum_3d')
            if k_3d is not None and spectrum_3d is not None:
                rows.append("# 3D Spectrum (spherical shells)")
                rows.append("# k_rad_m, E_k_m3_s2")
                for i in range(len(k_3d)):
                    rows.append(f"{k_3d[i]:.6e}, {spectrum_3d[i]:.6e}")
                rows.append("")
            
            # Export 1D spectra for each component (kx, ky only)
            for direction, label in [('kx', 'x'), ('ky', 'y')]:
                k = result.get(direction)
                if k is not None:
                    for comp in ['u', 'v', 'w']:
                        spectrum = result.get(f'{comp}_{direction}')
                        if spectrum is not None:
                            rows.append(f"# 1D Spectrum {comp}-component {label}-direction")
                            rows.append(f"# k_{label}_rad_m, E_{comp}_k_{label}_m3_s2")
                            for i in range(len(k)):
                                rows.append(f"{k[i]:.6e}, {spectrum[i]:.6e}")
                            rows.append("")
            
            open(path, "w").write("\n".join(rows))
            self.lbl_status.setText(f"✓ Exported 3D spatial spectra to {os.path.basename(path)}")
        elif res["tab"]=="temporal":
            export_spectra_csv(path,res["freq"],res["psds"],settings)
            n=len([p for p in res["psds"].values() if p is not None])
            self.lbl_status.setText(f"✓ Exported {n} component spectra to {os.path.basename(path)}")
        else:
            k=res["k"][1:]; f=res["f"][1:]
            combined_st={}
            for comp,E in res["psds"].items():
                if E is not None:
                    combined_st[comp]=E[1:,1:]
            if combined_st:
                rows=["# Spatiotemporal spectra - all components","# k_rad_m,f_Hz," + ",".join(combined_st.keys())]
                for ik,kv in enumerate(k):
                    for jf,fv in enumerate(f):
                        vals=",".join(f"{combined_st[c][ik,jf]:.6e}" for c in combined_st)
                        rows.append(f"{kv:.6e},{fv:.6e},{vals}")
                open(path,"w").write("\n".join(rows))
                self.lbl_status.setText(f"✓ Exported {len(combined_st)} component spatiotemporal spectra to {os.path.basename(path)}")
