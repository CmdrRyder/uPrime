"""
gui/vortex_window.py
--------------------
Vortex Identification window for uPrime.

Computes omega, Q, lambda_ci, lambda2, or Gamma1/Gamma2 from the mean or
a single instantaneous velocity field, thresholds the result, and
optionally overlays detected vortex boundaries.

Layout
------
260px fixed left panel | expanding right canvas

Left panel groups:
  Input           -- field selector, mean/inst toggle, frame spin, S spin
  Display         -- colormap range
  Vortex Detection -- threshold slider, sign filter, min area, detect button
  Statistics      -- qty selector, bins, show stats button
  Export          -- scalar field, CSV table, probability map

Right panel:
  Layout A (default): single field axes
  Layout B (stats):   probability map + histogram, Back button above canvas
"""

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QSlider,
    QSizePolicy, QMessageBox, QSplitter, QFileDialog,
    QScrollArea, QProgressBar,
)
from PyQt6.QtCore import Qt

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from core.vortex_id import (
    compute_gradients, compute_vortex_fields,
    compute_gamma, detect_vortices,
    compute_spatial_probability, export_vortex_csv,
)
from core.dataset_utils import get_masked
from core.export import export_2d_tecplot

_FONT_AX   = 9
_FONT_TICK = 8

GAMMA2_THRESHOLD = 2.0 / np.pi   # ~0.6366

_FIELD_ITEMS = [
    ("Vorticity (ω)",           "omega"),
    ("Q-criterion",             "Q"),
    ("Swirling strength (λci)", "lambda_ci"),
    ("Lambda-2 (λ₂)",          "lambda2"),
    ("Γ₁ (Graftieaux)",        "gamma1"),
]

_UNITS = {
    "omega"     : "1/s",
    "Q"         : "1/s²",
    "lambda_ci" : "1/s",
    "lambda2"   : "1/s²",
    "gamma1"    : "—",
}

_CMAPS = {
    "omega"     : "RdBu_r",
    "Q"         : "RdBu_r",
    "lambda_ci" : "RdBu_r",
    "lambda2"   : "RdBu_r",
    "gamma1"    : "RdBu_r",
}


class VortexWindow(PickerMixin, QWidget):

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Vortex Identification")
        self.resize(1400, 900)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._current_field  = None
        self._omega_field    = None
        self._gamma2_field   = None
        self._vortices       = None
        self._filtered_mask  = None
        self._layout_mode    = "field"   # "field" or "stats"
        self._cbar           = None

        self._build_ui()
        self._setup_picker(self.canvas, self._ax_field,
                           status_label=self.lbl_status)

    # ----------------------------------------------------------------------- #
    # UI construction
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # ---- Top ribbon ----
        self.chk_clean_export  = QCheckBox("Clean export (hide axes)")
        self.chk_hide_colorbar = QCheckBox("Hide colorbar")
        self.chk_clean_export.toggled.connect(self._plot_field)
        self.chk_hide_colorbar.toggled.connect(self._plot_field)

        top_ribbon = QHBoxLayout()
        top_ribbon.addStretch()
        top_ribbon.addWidget(self.chk_clean_export)
        top_ribbon.addWidget(self.chk_hide_colorbar)
        root.addLayout(top_ribbon)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- LEFT PANEL ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(270)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.setSpacing(6)
        self._build_controls(ll)
        ll.addStretch(1)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color:gray; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ---- RIGHT PANEL ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)

        # Back button (hidden in Layout A)
        self.btn_back = QPushButton("◀  Back to Field")
        self.btn_back.setVisible(False)
        self.btn_back.clicked.connect(self._on_back)
        rl.addWidget(self.btn_back)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.toolbar = DrawAwareToolbar(self.canvas, self)
        rl.addWidget(self.toolbar)
        rl.addWidget(self.canvas)

        splitter.addWidget(right)
        splitter.setSizes([270, 1130])

        # Initialise axes in Layout A
        self._ax_field = self.fig.add_subplot(111)
        self._ax_field.set_facecolor("#2b2b2b")
        self._ax_field.axis("off")
        self.canvas.draw()

    def _build_controls(self, ll):
        # ---- Group: Input ----
        grp_input = QGroupBox("Input")
        gl = QVBoxLayout(grp_input)

        gl.addWidget(QLabel("Scalar field:"))
        self.combo_field = QComboBox()
        for label, key in _FIELD_ITEMS:
            self.combo_field.addItem(label, key)
        self.combo_field.currentIndexChanged.connect(self._on_field_changed)
        gl.addWidget(self.combo_field)

        mode_row = QHBoxLayout()
        self.radio_mean = QRadioButton("Mean field")
        self.radio_inst = QRadioButton("Instantaneous")
        self.radio_mean.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.radio_mean)
        bg.addButton(self.radio_inst)
        self.radio_mean.toggled.connect(self._on_mode_toggled)
        mode_row.addWidget(self.radio_mean)
        mode_row.addWidget(self.radio_inst)
        gl.addLayout(mode_row)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, max(0, self.dataset["Nt"] - 1))
        self.spin_frame.setValue(0)
        self.spin_frame.setEnabled(False)
        frame_row.addWidget(self.spin_frame)
        gl.addLayout(frame_row)

        self._S_row = QHBoxLayout()
        self._S_row.addWidget(QLabel("Neighborhood S:"))
        self.spin_S = QSpinBox()
        self.spin_S.setRange(1, 7)
        self.spin_S.setValue(2)
        self.spin_S.setSingleStep(1)
        self._S_row.addWidget(self.spin_S)
        self._S_widget = QWidget()
        self._S_widget.setLayout(self._S_row)
        self._S_widget.setVisible(False)
        gl.addWidget(self._S_widget)

        self.btn_compute = QPushButton("Compute Field")
        self.btn_compute.clicked.connect(self._on_compute)
        gl.addWidget(self.btn_compute)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        gl.addWidget(self.progress_bar)

        ll.addWidget(grp_input)

        # ---- Group: Display ----
        grp_disp = QGroupBox("Display")
        dl = QVBoxLayout(grp_disp)

        self.chk_auto_range = QCheckBox("Auto colormap range")
        self.chk_auto_range.setChecked(True)
        self.chk_auto_range.stateChanged.connect(self._on_auto_range_changed)
        dl.addWidget(self.chk_auto_range)

        vmin_row = QHBoxLayout()
        vmin_row.addWidget(QLabel("Min:"))
        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1e9, 1e9)
        self.spin_vmin.setDecimals(4)
        self.spin_vmin.setEnabled(False)
        vmin_row.addWidget(self.spin_vmin)
        dl.addLayout(vmin_row)

        vmax_row = QHBoxLayout()
        vmax_row.addWidget(QLabel("Max:"))
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1e9, 1e9)
        self.spin_vmax.setDecimals(4)
        self.spin_vmax.setEnabled(False)
        vmax_row.addWidget(self.spin_vmax)
        dl.addLayout(vmax_row)

        ll.addWidget(grp_disp)

        # ---- Group: Vortex Detection ----
        grp_det = QGroupBox("Vortex Detection")
        vl = QVBoxLayout(grp_det)

        vl.addWidget(QLabel("Threshold (fraction of |max|):"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(30)
        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        vl.addWidget(self.thresh_slider)

        self.lbl_thresh_readout = QLabel("0.30 × |max| = --")
        self.lbl_thresh_readout.setStyleSheet("color:gray; font-size:10px;")
        vl.addWidget(self.lbl_thresh_readout)

        vl.addWidget(QLabel("Sign filter:"))
        self.combo_sign = QComboBox()
        self.combo_sign.addItem("All vortices",    "all")
        self.combo_sign.addItem("Positive (CCW)",  "positive")
        self.combo_sign.addItem("Negative (CW)",   "negative")
        vl.addWidget(self.combo_sign)

        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Min area (mm²):"))
        self.spin_min_area = QDoubleSpinBox()
        self.spin_min_area.setRange(0.1, 9999.0)
        self.spin_min_area.setValue(1.0)
        self.spin_min_area.setDecimals(1)
        area_row.addWidget(self.spin_min_area)
        vl.addLayout(area_row)

        self.chk_overlay = QCheckBox("Show vortex boundaries")
        self.chk_overlay.setChecked(True)
        vl.addWidget(self.chk_overlay)

        self.btn_detect = QPushButton("Detect Vortices")
        self.btn_detect.clicked.connect(self._on_detect)
        vl.addWidget(self.btn_detect)

        self.lbl_count = QLabel("Detected: --")
        self.lbl_count.setStyleSheet("color:gray; font-size:10px;")
        vl.addWidget(self.lbl_count)

        ll.addWidget(grp_det)

        # ---- Group: Statistics ----
        grp_stats = QGroupBox("Statistics")
        sl = QVBoxLayout(grp_stats)

        sl.addWidget(QLabel("Quantity:"))
        self.combo_qty = QComboBox()
        self.combo_qty.addItem("Area (mm²)",         "area")
        self.combo_qty.addItem("Circulation (m²/s)", "circulation")
        self.combo_qty.addItem("Aspect ratio",       "aspect_ratio")
        sl.addWidget(self.combo_qty)

        bins_row = QHBoxLayout()
        bins_row.addWidget(QLabel("Bins:"))
        self.spin_bins = QSpinBox()
        self.spin_bins.setRange(5, 100)
        self.spin_bins.setValue(20)
        bins_row.addWidget(self.spin_bins)
        sl.addLayout(bins_row)

        self.btn_stats = QPushButton("Show Statistics")
        self.btn_stats.clicked.connect(self._on_stats)
        sl.addWidget(self.btn_stats)

        ll.addWidget(grp_stats)

        # ---- Group: Export ----
        grp_exp = QGroupBox("Export")
        el = QVBoxLayout(grp_exp)

        self.btn_export_field = QPushButton("Export scalar field...")
        self.btn_export_field.clicked.connect(self._on_export_field)
        el.addWidget(self.btn_export_field)

        self.btn_export_csv = QPushButton("Export vortex table (CSV)...")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        el.addWidget(self.btn_export_csv)

        self.btn_export_prob = QPushButton("Export probability map...")
        self.btn_export_prob.clicked.connect(self._on_export_prob)
        el.addWidget(self.btn_export_prob)

        ll.addWidget(grp_exp)

    # ----------------------------------------------------------------------- #
    # Control callbacks
    # ----------------------------------------------------------------------- #

    def _on_field_changed(self):
        key = self.combo_field.currentData()
        self._S_widget.setVisible(key == "gamma1")
        self.lbl_count.setText("Detected: --")

    def _on_mode_toggled(self):
        self.spin_frame.setEnabled(self.radio_inst.isChecked())

    def _on_auto_range_changed(self):
        manual = not self.chk_auto_range.isChecked()
        self.spin_vmin.setEnabled(manual)
        self.spin_vmax.setEnabled(manual)

    def _on_thresh_changed(self):
        frac = self.thresh_slider.value() / 100.0
        if self._current_field is not None:
            field_max = float(np.nanmax(np.abs(self._current_field)))
            key = self.combo_field.currentData()
            units = _UNITS.get(key, "")
            abs_val = frac * field_max
            self.lbl_thresh_readout.setText(
                f"{frac:.2f} × |max| = {abs_val:.3g} [{units}]"
            )
        else:
            self.lbl_thresh_readout.setText(f"{frac:.2f} × |max| = --")

    # ----------------------------------------------------------------------- #
    # Compute
    # ----------------------------------------------------------------------- #

    def _on_compute(self):
        from core.workers import VortexWorker

        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        ds   = self.dataset
        key  = self.combo_field.currentData()
        mask = ds["MASK"]

        U_full = get_masked(ds, "U")
        V_full = get_masked(ds, "V")

        if self.radio_mean.isChecked():
            U = np.nanmean(U_full, axis=2)
            V = np.nanmean(V_full, axis=2)
        else:
            frame_idx = self.spin_frame.value()
            U = U_full[:, :, frame_idx]
            V = V_full[:, :, frame_idx]

        if np.all(~np.isfinite(U)):
            QMessageBox.warning(self, "No Data",
                "Velocity field is all-NaN after masking. Cannot compute.")
            return

        if key == "gamma1":
            S = self.spin_S.value()
            ny, nx = U.shape
            if S >= min(ny, nx) // 4:
                QMessageBox.warning(self, "Neighborhood Warning",
                    f"S={S} is large relative to the domain size "
                    f"({ny}×{nx}). Results near boundaries may be unreliable.")
        else:
            S = self.spin_S.value()

        self.lbl_status.setText("Computing... please wait")
        self.btn_compute.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self._worker = VortexWorker(
            U, V, ds["x"], ds["y"], mask, key, S)
        self._worker.finished.connect(self._on_vortex_result)
        self._worker.error.connect(self._on_vortex_error)
        self._worker.finished.connect(lambda _: self._reset_compute_ui())
        self._worker.error.connect(lambda _: self._reset_compute_ui())
        self._worker.start()

    def _reset_compute_ui(self):
        self.btn_compute.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)

    def _on_vortex_result(self, result):
        field  = result["field"]
        omega  = result["omega"]
        gamma2 = result.get("gamma2")

        if np.all(~np.isfinite(field)):
            QMessageBox.warning(self, "Computation Warning",
                "Resulting field is all-NaN. Check mask and input data.")
            self.lbl_status.setText("Warning: result is all-NaN.")
            return

        self._current_field = field
        self._omega_field   = omega
        self._gamma2_field  = gamma2
        self._vortices      = None
        self._filtered_mask = None
        self._x = self.dataset["x"]
        self._y = self.dataset["y"]
        self.lbl_count.setText("Detected: --")
        self._layout_mode = "field"
        self.btn_back.setVisible(False)
        self._plot_field()
        self._on_thresh_changed()
        self.lbl_status.setText("Done.")

    def _on_vortex_error(self, tb_str):
        QMessageBox.critical(self, "Computation Error", tb_str)
        self.lbl_status.setText("Error — see dialog.")

    # ----------------------------------------------------------------------- #
    # Detection
    # ----------------------------------------------------------------------- #

    def _on_detect(self):
        if self._current_field is None:
            QMessageBox.warning(self, "No Field",
                "Compute a scalar field first.")
            return

        frac      = self.thresh_slider.value() / 100.0
        field_max = float(np.nanmax(np.abs(self._current_field)))
        threshold = frac * field_max

        sign_filter = self.combo_sign.currentData()
        min_area    = self.spin_min_area.value()
        x = self.dataset["x"]
        y = self.dataset["y"]

        self._vortices = detect_vortices(
            self._current_field, self._omega_field,
            x, y, threshold, sign_filter, min_area,
        )
        self.lbl_count.setText(f"Detected: {len(self._vortices)} vortices")

        # Build filtered mask: only blobs that survived the area filter
        from scipy import ndimage as _ndi
        field = self._current_field
        if sign_filter == "positive":
            raw_mask = field > threshold
        elif sign_filter == "negative":
            raw_mask = field < -threshold
        else:
            raw_mask = np.abs(field) > threshold
        raw_mask = raw_mask & np.isfinite(field)
        labeled, _ = _ndi.label(raw_mask)
        filtered_mask = np.zeros_like(raw_mask)
        for v in self._vortices:
            filtered_mask |= (labeled == v["id"])
        self._filtered_mask = filtered_mask

        self._plot_field()

    # ----------------------------------------------------------------------- #
    # Field plot (Layout A)
    # ----------------------------------------------------------------------- #

    def _plot_field(self):
        if self._current_field is None:
            return

        key   = self.combo_field.currentData()
        field = self._current_field
        x     = self.dataset["x"]
        y     = self.dataset["y"]
        cmap  = _CMAPS.get(key, "RdBu_r")
        units = _UNITS.get(key, "")

        # --- colormap limits ---
        if self.chk_auto_range.isChecked():
            absmax = float(np.nanmax(np.abs(field)))
            if key == "lambda_ci":
                vmin, vmax = -absmax, absmax
            else:
                vmin, vmax = -absmax, absmax
        else:
            vmin = self.spin_vmin.value()
            vmax = self.spin_vmax.value()

        # Extents
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        extent = [x_min, x_max, y_min, y_max]

        # Remove old colorbar before clearing
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self.fig.clear()
        self._ax_field = self.fig.add_subplot(111)
        ax = self._ax_field

        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if not self.chk_hide_colorbar.isChecked():
            self._cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._cbar.set_label(f"{self.combo_field.currentText()} [{units}]",
                                 fontsize=_FONT_AX)
            self._cbar.ax.tick_params(labelsize=_FONT_TICK)

        # --- Overlay contours ---
        if self.chk_overlay.isChecked() and self._filtered_mask is not None:
            fm = self._filtered_mask.astype(float)
            if fm.max() > 0:
                ax.contour(x, y, fm, levels=[0.5],
                           colors="black", linewidths=0.8)

            # Gamma2 boundary contour
            if key == "gamma1" and self._gamma2_field is not None:
                g2_abs = np.abs(self._gamma2_field)
                g2_abs[~np.isfinite(g2_abs)] = 0.0
                if g2_abs.max() > GAMMA2_THRESHOLD:
                    ax.contour(x, y, g2_abs, levels=[GAMMA2_THRESHOLD],
                               colors="black", linewidths=1.0,
                               linestyles="--")

        if self.chk_clean_export.isChecked():
            ax.set_axis_off()
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            ax.set_axis_on()
            ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
            ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
            ax.tick_params(labelsize=_FONT_TICK)
            self.fig.tight_layout(pad=0.5)

        self.canvas.draw()
        self.toolbar.update()

        if hasattr(self.toolbar, "set_home_limits"):
            self.toolbar.set_home_limits()

    # ----------------------------------------------------------------------- #
    # Statistics (Layout B)
    # ----------------------------------------------------------------------- #

    def _on_stats(self):
        try:
            self._on_stats_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Statistics Error",
                f"Error: {e}\n\nSee console for full traceback.")

    def _on_stats_impl(self):
        if not self._vortices:
            QMessageBox.warning(self, "No vortices",
                "Please detect vortices first before running statistics.")
            return

        qty_raw = self.combo_qty.currentData()
        qty_key_map = {
            "area"         : "area_mm2",
            "circulation"  : "circulation",
            "aspect_ratio" : "aspect_ratio",
        }
        qty_key = qty_key_map.get(qty_raw, qty_raw)
        qty_label_map = {
            "area"         : "Area (mm\u00b2)",
            "circulation"  : "Circulation (m\u00b2/s)",
            "aspect_ratio" : "Aspect ratio",
        }
        bins    = self.spin_bins.value()
        x       = self.dataset["x"]
        y       = self.dataset["y"]

        # Probability map
        if self._current_field is not None:
            frac      = self.thresh_slider.value() / 100.0
            threshold = frac * float(np.nanmax(np.abs(self._current_field)))
            sign_filter = self.combo_sign.currentData()
            prob_map = compute_spatial_probability(
                self._current_field, threshold, sign_filter)
        else:
            prob_map = None

        # Split vortices by sign
        pos_vals = [v[qty_key] for v in self._vortices if v["sign"] >= 0]
        neg_vals = [v[qty_key] for v in self._vortices if v["sign"] <  0]

        self._layout_mode = "stats"
        self.btn_back.setVisible(True)
        self.fig.clear()
        self._ax_prob = self.fig.add_subplot(2, 1, 1)
        self._ax_hist = self.fig.add_subplot(2, 1, 2)

        # Probability map
        if prob_map is not None:
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            im = self._ax_prob.imshow(
                prob_map, origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                cmap="viridis", vmin=0, vmax=1, aspect="auto",
                interpolation="nearest",
            )
            cb = self.fig.colorbar(im, ax=self._ax_prob,
                                   fraction=0.046, pad=0.04)
            cb.set_label("Detection probability", fontsize=_FONT_AX)
            cb.ax.tick_params(labelsize=_FONT_TICK)
            self._ax_prob.set_xlabel("x [mm]", fontsize=_FONT_AX)
            self._ax_prob.set_ylabel("y [mm]", fontsize=_FONT_AX)
            self._ax_prob.set_title("Spatial probability map",
                                    fontsize=_FONT_AX)
            self._ax_prob.tick_params(labelsize=_FONT_TICK)
        else:
            self._ax_prob.axis("off")

        # Histogram
        all_vals = pos_vals + neg_vals
        if all_vals:
            lo = min(all_vals)
            hi = max(all_vals)
            bin_edges = np.linspace(lo, hi, bins + 1)
        else:
            bin_edges = bins

        if pos_vals:
            self._ax_hist.hist(pos_vals, bins=bin_edges,
                               color="#d62728", alpha=0.6,
                               label=f"Positive (CCW)  n={len(pos_vals)}")
        if neg_vals:
            self._ax_hist.hist(neg_vals, bins=bin_edges,
                               color="#1f77b4", alpha=0.6,
                               label=f"Negative (CW)  n={len(neg_vals)}")
        if not pos_vals and not neg_vals:
            self._ax_hist.text(0.5, 0.5, "No vortices to plot",
                               transform=self._ax_hist.transAxes,
                               ha="center", va="center")

        self._ax_hist.set_xlabel(qty_label_map.get(qty_raw, qty_raw), fontsize=_FONT_AX)
        self._ax_hist.set_ylabel("Count", fontsize=_FONT_AX)
        self._ax_hist.set_title("Vortex property distribution",
                                fontsize=_FONT_AX)
        self._ax_hist.tick_params(labelsize=_FONT_TICK)
        if pos_vals or neg_vals:
            self._ax_hist.legend(fontsize=_FONT_TICK)

        self.fig.tight_layout(pad=0.5)
        self.canvas.draw()
        self.toolbar.update()

    def _on_back(self):
        self._layout_mode = "field"
        self.btn_back.setVisible(False)
        self.fig.clear()
        self._ax_field = self.fig.add_subplot(1, 1, 1)
        self.canvas.draw()
        self.toolbar.update()
        self._plot_field()

    # ----------------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------------- #

    def _on_export_field(self):
        if self._current_field is None:
            QMessageBox.warning(self, "No Data", "Compute a field first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Scalar Field", "",
            "Tecplot DAT (*.dat);;All Files (*)")
        if not path:
            return
        key = self.combo_field.currentData()
        try:
            export_2d_tecplot(
                path,
                self.dataset["x"], self.dataset["y"],
                [self._current_field],
                [self.combo_field.currentText()],
                {"field": key, "units": _UNITS.get(key, "")},
            )
            self.lbl_status.setText(f"Field exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_csv(self):
        if not self._vortices:
            QMessageBox.warning(self, "No Detections",
                "Run 'Detect Vortices' first (and ensure at least one "
                "vortex was found).")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Vortex Table", "",
            "CSV (*.csv);;All Files (*)")
        if not path:
            return
        try:
            export_vortex_csv(self._vortices, path)
            self.lbl_status.setText(f"CSV exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_prob(self):
        if self._current_field is None:
            QMessageBox.warning(self, "No Data", "Compute a field first.")
            return
        frac      = self.thresh_slider.value() / 100.0
        threshold = frac * float(np.nanmax(np.abs(self._current_field)))
        sign_filter = self.combo_sign.currentData()
        prob_map = compute_spatial_probability(
            self._current_field, threshold, sign_filter)

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Probability Map", "",
            "Tecplot DAT (*.dat);;All Files (*)")
        if not path:
            return
        try:
            export_2d_tecplot(
                path,
                self.dataset["x"], self.dataset["y"],
                [prob_map],
                ["Detection probability"],
                {"field": self.combo_field.currentData(),
                 "threshold_fraction": frac,
                 "sign_filter": sign_filter},
            )
            self.lbl_status.setText(f"Probability map exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
