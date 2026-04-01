"""
gui/spectral_window.py
----------------------
Temporal spectral analysis window.

Layout:
  Left  : mean field plot -- user clicks a point or draws a rectangle
  Right : PSD plot (semilogy, u/v/w as separate lines)
  Bottom: controls (fs, Welch params, selection mode, component toggles)
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QGroupBox, QPushButton,
    QRadioButton, QButtonGroup, QCheckBox, QSizePolicy,
    QMessageBox, QSplitter, QDoubleSpinBox, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from core.spectral import psd_at_point, psd_in_region, nearest_grid_point
from core.export import export_spectra_csv


class SpectralWindow(QWidget):
    """
    Standalone window for temporal spectral analysis.
    Receives the loaded dataset dict from the main window.
    """

    def __init__(self, dataset, default_fs=1000.0, parent=None):
        super().__init__(parent)
        self.dataset    = dataset
        self._default_fs = default_fs
        self.setWindowTitle("Temporal Spectral Analysis")
        self.resize(1400, 750)

        # State for interactive selection
        self._mode         = "point"    # "point" or "rect"
        self._press_xy     = None       # mouse press position
        self._rect_patch   = None       # rectangle patch on field plot
        self._cid_press    = None
        self._cid_release  = None
        self._cid_motion   = None
        self._selection    = None       # dict describing current selection

        self._build_ui()
        self._draw_field()
        self._connect_mouse()

        # Warn if snapshot count is low
        Nt = dataset["Nt"]
        if Nt < 2000:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Convergence Warning",
                f"Only {Nt} snapshots loaded.\n\n"
                "Reliable spectral estimates (Welch PSD) typically require "
                "at least ~2000 snapshots for good frequency resolution and "
                "statistical convergence.\n\n"
                "Results should be interpreted with caution."
            )

    # ----------------------------------------------------------------------- #
    # UI construction
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left: field canvas + controls ----
        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        # Field plot
        self.field_fig    = Figure(figsize=(6, 4), tight_layout=True)
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.field_toolbar = NavToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)

        # Selection mode
        mode_group = QGroupBox("Selection Mode")
        mode_layout = QHBoxLayout(mode_group)
        self.rb_point = QRadioButton("Point")
        self.rb_rect  = QRadioButton("Rectangle")
        self.rb_point.setChecked(True)
        self.rb_point.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.rb_point)
        mode_layout.addWidget(self.rb_rect)
        ll.addWidget(mode_group)

        self.lbl_hint = QLabel("Click on the field to select a point.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_hint)

        # ---- Right: PSD canvas + spectral controls ----
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(6)

        # PSD plot
        self.psd_fig    = Figure(figsize=(6, 4), tight_layout=True)
        self.psd_canvas = FigureCanvas(self.psd_fig)
        self.psd_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.psd_toolbar = NavToolbar(self.psd_canvas, self)
        rl.addWidget(self.psd_toolbar)
        rl.addWidget(self.psd_canvas)

        # Spectral controls
        ctrl_group  = QGroupBox("Spectral Parameters")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # fs
        row_fs = QHBoxLayout()
        row_fs.addWidget(QLabel("Sampling freq fs [Hz]:"))
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1.0, 1e6)
        self.spin_fs.setValue(self._default_fs)
        self.spin_fs.setDecimals(1)
        self.spin_fs.setSingleStep(100.0)
        row_fs.addWidget(self.spin_fs)
        ctrl_layout.addLayout(row_fs)

        # Welch segment length
        row_seg = QHBoxLayout()
        row_seg.addWidget(QLabel("Welch segment (samples):"))
        self.spin_nperseg = QSpinBox()
        self.spin_nperseg.setRange(8, 100000)
        Nt = self.dataset["Nt"]
        self.spin_nperseg.setValue(max(16, Nt // 4))
        row_seg.addWidget(self.spin_nperseg)
        ctrl_layout.addLayout(row_seg)

        # Overlap
        row_ov = QHBoxLayout()
        row_ov.addWidget(QLabel("Overlap (samples):"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 100000)
        self.spin_overlap.setValue(max(8, Nt // 8))
        row_ov.addWidget(self.spin_overlap)
        ctrl_layout.addLayout(row_ov)

        # Component toggles
        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("Components:"))
        self.chk_u = QCheckBox("u")
        self.chk_v = QCheckBox("v")
        self.chk_w = QCheckBox("w")
        self.chk_u.setChecked(True)
        self.chk_v.setChecked(True)
        self.chk_w.setChecked(self.dataset["is_stereo"])
        self.chk_w.setEnabled(self.dataset["is_stereo"])
        comp_row.addWidget(self.chk_u)
        comp_row.addWidget(self.chk_v)
        comp_row.addWidget(self.chk_w)
        ctrl_layout.addLayout(comp_row)

        # -5/3 slope option
        slope_row = QHBoxLayout()
        self.chk_kolmogorov = QCheckBox("Show -5/3 slope")
        self.chk_kolmogorov.setChecked(True)
        self.chk_kolmogorov.setToolTip("Overlay Kolmogorov -5/3 inertial subrange slope")
        slope_row.addWidget(self.chk_kolmogorov)
        ctrl_layout.addLayout(slope_row)

        # Compute button
        self.btn_compute = QPushButton("Compute Spectrum")
        self.btn_compute.setEnabled(False)
        self.btn_compute.clicked.connect(self._on_compute)
        ctrl_layout.addWidget(self.btn_compute)

        self.btn_export = QPushButton("Export Spectrum...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_spectra)
        ctrl_layout.addWidget(self.btn_export)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        ctrl_layout.addWidget(self.lbl_status)

        rl.addWidget(ctrl_group)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    # ----------------------------------------------------------------------- #
    # Draw mean field on left canvas
    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds = self.dataset
        x  = ds["x"]
        y  = ds["y"]
        U  = ds["U"]
        V  = ds["V"]
        W  = ds["W"]

        mean_u = np.nanmean(U, axis=2)
        mean_v = np.nanmean(V, axis=2)
        if W is not None:
            mean_w = np.nanmean(W, axis=2)
            speed  = np.sqrt(mean_u**2 + mean_v**2 + mean_w**2)
        else:
            speed  = np.sqrt(mean_u**2 + mean_v**2)

        valid_frac = np.mean(ds["valid"], axis=2)
        speed[valid_frac < 0.5] = np.nan

        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        cf = self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
        self.field_fig.colorbar(cf, ax=self.field_ax, label="Mean |V| [m/s]", shrink=0.8)
        self.field_ax.set_xlabel("x [mm]")
        self.field_ax.set_ylabel("y [mm]")
        self.field_ax.set_title("Click to select point or draw rectangle")
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_canvas.draw()

        # Store for hit testing
        self._x = x
        self._y = y

    # ----------------------------------------------------------------------- #
    # Mouse interaction
    # ----------------------------------------------------------------------- #

    def _connect_mouse(self):
        self._cid_press   = self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self._cid_release = self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion  = self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_mode_changed(self):
        if self.rb_point.isChecked():
            self._mode = "point"
            self.lbl_hint.setText("Click on the field to select a point.")
        else:
            self._mode = "rect"
            self.lbl_hint.setText("Click and drag to draw a rectangle.")
        self._clear_selection_graphics()
        self._selection = None
        self.btn_compute.setEnabled(False)

    def _on_press(self, event):
        if event.inaxes != self.field_ax:
            return
        self._press_xy = (event.xdata, event.ydata)

        if self._mode == "point":
            self._clear_selection_graphics()
            # Draw a marker at clicked point
            self.field_ax.plot(event.xdata, event.ydata, "r+", markersize=12,
                               markeredgewidth=2, zorder=10)
            self.field_canvas.draw()

            row, col = nearest_grid_point(self._x, self._y, event.xdata, event.ydata)
            self._selection = {"type": "point", "row": row, "col": col,
                               "xc": event.xdata, "yc": event.ydata}
            self.lbl_hint.setText(
                f"Point selected: x={self._x[row,col]:.1f} mm, y={self._y[row,col]:.1f} mm"
            )
            self.btn_compute.setEnabled(True)

    def _on_motion(self, event):
        if self._mode != "rect" or self._press_xy is None:
            return
        if event.inaxes != self.field_ax:
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata

        self._clear_selection_graphics()
        self._rect_patch = Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0), abs(y1 - y0),
            linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.15, zorder=10
        )
        self.field_ax.add_patch(self._rect_patch)
        self.field_canvas.draw()

    def _on_release(self, event):
        if self._mode != "rect" or self._press_xy is None:
            return
        if event.inaxes != self.field_ax:
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._press_xy = None

        if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
            self.lbl_hint.setText("Rectangle too small -- try again.")
            return

        self._selection = {"type": "rect",
                           "x0": x0, "x1": x1, "y0": y0, "y1": y1}
        self.lbl_hint.setText(
            f"Rectangle: x=[{min(x0,x1):.1f}, {max(x0,x1):.1f}] mm  "
            f"y=[{min(y0,y1):.1f}, {max(y0,y1):.1f}] mm"
        )
        self.btn_compute.setEnabled(True)

    def _clear_selection_graphics(self):
        # Remove previous markers and rectangle patches
        for artist in self.field_ax.lines + self.field_ax.patches:
            artist.remove()
        if self._rect_patch is not None:
            self._rect_patch = None
        self.field_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Compute and plot PSD
    # ----------------------------------------------------------------------- #

    def _on_compute(self):
        if self._selection is None:
            return

        ds       = self.dataset
        fs       = self.spin_fs.value()
        nperseg  = self.spin_nperseg.value()
        noverlap = self.spin_overlap.value()

        if noverlap >= nperseg:
            QMessageBox.warning(self, "Parameter Error",
                                "Overlap must be less than segment length.")
            return

        self.lbl_status.setText("Computing...")
        self.psd_canvas.draw()

        try:
            sel = self._selection

            if sel["type"] == "point":
                freq, psd = psd_at_point(
                    ds["U"], ds["V"], ds["W"],
                    sel["row"], sel["col"],
                    fs, nperseg, noverlap
                )
                title = (f"PSD at x={self._x[sel['row'],sel['col']]:.1f} mm, "
                         f"y={self._y[sel['row'],sel['col']]:.1f} mm")
                n_pts = 1

            else:
                freq, psd, n_pts = psd_in_region(
                    ds["U"], ds["V"], ds["W"],
                    self._x, self._y,
                    sel["x0"], sel["x1"], sel["y0"], sel["y1"],
                    fs, nperseg, noverlap
                )
                title = f"Mean PSD over rectangle ({n_pts} points averaged)"

            self._last_freq  = freq
            self._last_psd   = psd
            self._last_title = title
            self._last_nperseg  = nperseg
            self._last_noverlap = noverlap
            self._last_n_pts    = n_pts
            self._plot_psd(freq, psd, title)
            self.btn_export.setEnabled(True)
            self.lbl_status.setText(
                f"Done. fs={fs:.0f} Hz, nperseg={nperseg}, "
                f"noverlap={noverlap}, points={n_pts}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Spectral Error", str(e))
            self.lbl_status.setText(f"Error: {e}")

    def _plot_psd(self, freq, psd, title):
        self.psd_fig.clear()
        ax = self.psd_fig.add_subplot(111)

        colors = {"u": "tab:blue", "v": "tab:orange", "w": "tab:green"}
        plotted = False
        psd_max = 0.0
        psd_min = np.inf
        freq_mask = freq > 0

        for comp, chk in [("u", self.chk_u), ("v", self.chk_v), ("w", self.chk_w)]:
            if not chk.isChecked():
                continue
            if psd.get(comp) is None:
                continue
            p = psd[comp][freq_mask]
            f = freq[freq_mask]
            valid = p > 0
            if not np.any(valid):
                continue
            ax.loglog(f[valid], p[valid],
                      color=colors[comp], label=f"PSD({comp})", linewidth=1.2)
            psd_max = max(psd_max, np.max(p[valid]))
            psd_min = min(psd_min, np.min(p[valid]))
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No data to plot", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            # -5/3 Kolmogorov slope line
            if self.chk_kolmogorov.isChecked() and psd_max > 0:
                f_pos = freq[freq_mask]
                # Place the line in the upper-middle frequency decade
                f_mid  = np.exp(0.5 * (np.log(f_pos[1]) + np.log(f_pos[-1])))
                f_line = np.array([f_mid * 0.3, f_mid * 3.0])
                f_line = f_line[(f_line >= f_pos[0]) & (f_line <= f_pos[-1])]
                if len(f_line) >= 2:
                    # Anchor at geometric mean of plotted PSD range
                    p_anchor = np.sqrt(psd_max * max(psd_min, psd_max * 1e-4))
                    f_anchor = np.sqrt(f_line[0] * f_line[-1])
                    p_line   = p_anchor * (f_line / f_anchor) ** (-5/3)
                    ax.loglog(f_line, p_line, "k--", linewidth=1.5,
                              label=r"$f^{-5/3}$ (Kolmogorov)", alpha=0.7)

            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD [(m/s)²/Hz]")
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, which="both", alpha=0.3)

        self.psd_canvas.draw()

    def _on_export_spectra(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Spectrum", "spectrum.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        sel = self._selection
        settings = {
            "Analysis"      : "Temporal Spectral Analysis (Welch)",
            "Snapshots"     : self.dataset["Nt"],
            "fs [Hz]"       : self.spin_fs.value(),
            "nperseg"       : self._last_nperseg,
            "noverlap"      : self._last_noverlap,
            "Points averaged": self._last_n_pts,
            "Selection type": sel["type"] if sel else "unknown",
        }
        if sel and sel["type"] == "point":
            settings["Point x [mm]"] = f"{self._x[sel['row'], sel['col']]:.2f}"
            settings["Point y [mm]"] = f"{self._y[sel['row'], sel['col']]:.2f}"
        elif sel and sel["type"] == "rect":
            settings["Rect x"] = f"[{min(sel['x0'],sel['x1']):.1f}, {max(sel['x0'],sel['x1']):.1f}] mm"
            settings["Rect y"] = f"[{min(sel['y0'],sel['y1']):.1f}, {max(sel['y0'],sel['y1']):.1f}] mm"

        export_spectra_csv(path, self._last_freq, self._last_psd, settings)
        self.lbl_status.setText(f"Exported to {path}")
