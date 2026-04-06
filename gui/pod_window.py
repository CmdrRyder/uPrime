"""
gui/pod_window.py
-----------------
Proper Orthogonal Decomposition (POD) Analysis window for uPrime.

Four tabs:
  1. Energy Spectrum  -- bar chart of modal energy + cumulative curve
  2. Spatial Modes    -- contourf of selected mode / component
  3. Temporal Coeffs  -- time series + PSD of a_n(t)  [TR only]
  4. Reconstruction   -- original vs reconstructed snapshot

Colorbar: flat ends (extend='neither'), consistent with all other windows.
Aspect ratio: equal on all 2D plots.
Font sizes / colormap: matches correlation_window, reynolds_window, etc.
"""

import csv
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QComboBox, QSpinBox, QCheckBox,
    QSizePolicy, QMessageBox, QSplitter, QTabWidget,
    QApplication, QFileDialog, QFrame, QSlider,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.arrow_toolbar import DrawAwareToolbar
from core.export import export_2d_tecplot
from core.pod import compute_pod, reconstruct_snapshot

_CMAP_DIV   = "RdBu_r"
_FONT_AX    = 9
_FONT_TICK  = 8
_FONT_LEG   = 8
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
# ScaleReadout widget (same pattern as correlation_window.py)
# ---------------------------------------------------------------------------

class ScaleReadout(QWidget):
    """Compact key=value grid shown below plots."""

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(16)
        self._vals = {}
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

class PODWindow(QWidget):

    def __init__(self, dataset, is_time_resolved=False, fs=1000.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("POD Analysis")
        self.resize(1700, 900)

        self.dataset          = dataset
        self.is_time_resolved = is_time_resolved
        self.fs               = fs
        self.dt               = 1.0 / fs

        self._x        = dataset["x"]
        self._y        = dataset["y"]
        self.U         = dataset["U"]
        self.V         = dataset["V"]
        self.W         = dataset["W"]
        self.is_stereo = dataset["is_stereo"]

        self._pod_result = None   # filled after compute

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left panel ----
        left = QWidget()
        left.setMinimumWidth(260)
        left.setMaximumWidth(260)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        # -- Compute group --
        cmp_grp = QGroupBox("Compute")
        cmp_lay = QVBoxLayout(cmp_grp)

        modes_row = QHBoxLayout()
        modes_row.addWidget(QLabel("Modes to compute:"))
        self.spin_n_modes = QSpinBox()
        self.spin_n_modes.setRange(1, 500)
        self.spin_n_modes.setValue(25)
        modes_row.addWidget(self.spin_n_modes)
        cmp_lay.addLayout(modes_row)

        self.btn_compute = QPushButton("Compute POD")
        self.btn_compute.clicked.connect(self._run_pod)
        cmp_lay.addWidget(self.btn_compute)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        cmp_lay.addWidget(self.lbl_status)

        ll.addWidget(cmp_grp)

        # -- Display group --
        disp_grp = QGroupBox("Display")
        disp_lay = QVBoxLayout(disp_grp)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.spin_mode_idx = QSpinBox()
        self.spin_mode_idx.setRange(1, 25)
        self.spin_mode_idx.setValue(1)
        self.spin_mode_idx.setEnabled(False)
        self.spin_mode_idx.valueChanged.connect(self._on_display_changed)
        mode_row.addWidget(self.spin_mode_idx)
        disp_lay.addLayout(mode_row)

        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("Component:"))
        self.combo_component = QComboBox()
        self.combo_component.addItems(["U", "V", "W"])
        if not self.is_stereo:
            self.combo_component.model().item(2).setEnabled(False)
        self.combo_component.setEnabled(False)
        self.combo_component.currentIndexChanged.connect(self._on_display_changed)
        comp_row.addWidget(self.combo_component)
        disp_lay.addLayout(comp_row)

        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev_mode)
        self.btn_next = QPushButton("Next >")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next_mode)
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        disp_lay.addLayout(nav_row)

        ll.addWidget(disp_grp)

        ll.addStretch()

        self.btn_export = QPushButton("Export Modes...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_pod)
        ll.addWidget(self.btn_export)

        splitter.addWidget(left)

        # ---- Right panel: tabs ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        chk_row = QHBoxLayout()
        chk_row.addStretch()
        self.chk_hide_axes = QCheckBox("Hide axes")
        self.chk_hide_axes.stateChanged.connect(self._on_display_changed)
        chk_row.addWidget(self.chk_hide_axes)
        self.chk_hide_colorbar = QCheckBox("Hide colorbar")
        self.chk_hide_colorbar.stateChanged.connect(self._on_display_changed)
        chk_row.addWidget(self.chk_hide_colorbar)
        rl.addLayout(chk_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_energy_tab(),    "Energy Spectrum")
        self.tabs.addTab(self._build_modes_tab(),     "Spatial Modes")
        self.tabs.addTab(self._build_temporal_tab(),  "Temporal Coefficients")
        self.tabs.addTab(self._build_recon_tab(),     "Reconstruction")

        if not self.is_time_resolved:
            self.tabs.setTabEnabled(2, False)
            self.tabs.setTabToolTip(2, "Temporal coefficients require time-resolved data.")

        rl.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setSizes([260, 1440])

    # -- Energy Spectrum tab --

    def _build_energy_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        self.energy_fig    = Figure()
        self.energy_canvas = FigureCanvas(self.energy_fig)
        self.energy_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        vl.addWidget(self.energy_canvas, stretch=1)

        vl.addWidget(_hline())
        self.energy_readout = ScaleReadout(
            ["50% at mode", "80% at mode", "90% at mode"])
        vl.addWidget(self.energy_readout)

        return w

    # -- Spatial Modes tab --

    def _build_modes_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        self.mode_fig    = Figure()
        self.mode_canvas = FigureCanvas(self.mode_fig)
        self.mode_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        self.mode_toolbar = DrawAwareToolbar(self.mode_canvas, w)
        vl.addWidget(self.mode_toolbar)
        vl.addWidget(self.mode_canvas, stretch=1)

        return w

    # -- Temporal Coefficients tab --

    def _build_temporal_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        self.temporal_fig    = Figure()
        self.temporal_canvas = FigureCanvas(self.temporal_fig)
        self.temporal_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Expanding)
        self.temporal_toolbar = DrawAwareToolbar(self.temporal_canvas, w)
        vl.addWidget(self.temporal_toolbar)
        vl.addWidget(self.temporal_canvas, stretch=1)

        return w

    # -- Reconstruction tab --

    def _build_recon_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Snapshot:"))
        self.slider_snapshot = QSlider(Qt.Orientation.Horizontal)
        Nt = self.U.shape[2]
        self.slider_snapshot.setRange(0, Nt - 1)
        self.slider_snapshot.setValue(0)
        self.lbl_snapshot_idx = QLabel("0")
        self.slider_snapshot.valueChanged.connect(
            lambda v: self.lbl_snapshot_idx.setText(str(v)))
        ctrl_row.addWidget(self.slider_snapshot, stretch=1)
        ctrl_row.addWidget(self.lbl_snapshot_idx)

        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QLabel("Modes for reconstruction:"))
        self.spin_n_recon = QSpinBox()
        self.spin_n_recon.setRange(1, 25)
        self.spin_n_recon.setValue(10)
        self.spin_n_recon.setEnabled(False)
        ctrl_row.addWidget(self.spin_n_recon)

        self.btn_reconstruct = QPushButton("Reconstruct")
        self.btn_reconstruct.setEnabled(False)
        self.btn_reconstruct.clicked.connect(self._run_reconstruction)
        ctrl_row.addWidget(self.btn_reconstruct)
        vl.addLayout(ctrl_row)

        # Three side-by-side canvases: original | reconstructed | residual
        canvas_row = QHBoxLayout()

        self.recon_orig_fig    = Figure()
        self.recon_orig_canvas = FigureCanvas(self.recon_orig_fig)
        self.recon_orig_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                              QSizePolicy.Policy.Expanding)
        self.recon_orig_toolbar = DrawAwareToolbar(self.recon_orig_canvas, w)
        orig_wrap = QWidget()
        orig_l = QVBoxLayout(orig_wrap)
        orig_l.setContentsMargins(0, 0, 0, 0)
        orig_l.addWidget(self.recon_orig_toolbar)
        orig_l.addWidget(self.recon_orig_canvas)
        canvas_row.addWidget(orig_wrap)

        self.recon_rec_fig    = Figure()
        self.recon_rec_canvas = FigureCanvas(self.recon_rec_fig)
        self.recon_rec_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                             QSizePolicy.Policy.Expanding)
        self.recon_rec_toolbar = DrawAwareToolbar(self.recon_rec_canvas, w)
        rec_wrap = QWidget()
        rec_l = QVBoxLayout(rec_wrap)
        rec_l.setContentsMargins(0, 0, 0, 0)
        rec_l.addWidget(self.recon_rec_toolbar)
        rec_l.addWidget(self.recon_rec_canvas)
        canvas_row.addWidget(rec_wrap)

        self.recon_diff_fig    = Figure()
        self.recon_diff_canvas = FigureCanvas(self.recon_diff_fig)
        self.recon_diff_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                              QSizePolicy.Policy.Expanding)
        self.recon_diff_toolbar = DrawAwareToolbar(self.recon_diff_canvas, w)
        diff_wrap = QWidget()
        diff_l = QVBoxLayout(diff_wrap)
        diff_l.setContentsMargins(0, 0, 0, 0)
        diff_l.addWidget(self.recon_diff_toolbar)
        diff_l.addWidget(self.recon_diff_canvas)
        canvas_row.addWidget(diff_wrap)

        vl.addLayout(canvas_row, stretch=1)

        vl.addWidget(_hline())
        self.lbl_recon_error = QLabel("RMS error: --")
        self.lbl_recon_error.setStyleSheet("font-size: 11px;")
        vl.addWidget(self.lbl_recon_error)

        return w

    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------

    def _prev_mode(self):
        v = self.spin_mode_idx.value()
        if v > 1:
            self.spin_mode_idx.setValue(v - 1)

    def _next_mode(self):
        v = self.spin_mode_idx.value()
        if v < self.spin_mode_idx.maximum():
            self.spin_mode_idx.setValue(v + 1)

    def _on_display_changed(self):
        if self._pod_result is not None:
            self._plot_mode()
            if self.is_time_resolved:
                self._plot_temporal_coeffs()

    # -----------------------------------------------------------------------
    # Compute POD
    # -----------------------------------------------------------------------

    def _run_pod(self):
        n_modes = self.spin_n_modes.value()
        self.lbl_status.setText("Busy: computing POD…")
        self.btn_compute.setEnabled(False)
        QApplication.processEvents()

        try:
            W = self.W if self.is_stereo else None
            self._pod_result = compute_pod(self.U, self.V, W, n_modes=n_modes)

            n = self._pod_result["n_modes"]

            # Update display controls
            self.spin_mode_idx.setRange(1, n)
            self.spin_mode_idx.setValue(1)
            self.spin_mode_idx.setEnabled(True)
            self.combo_component.setEnabled(True)
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)
            self.spin_n_recon.setRange(1, n)
            self.spin_n_recon.setValue(min(10, n))
            self.spin_n_recon.setEnabled(True)
            self.btn_reconstruct.setEnabled(True)
            self.btn_export.setEnabled(True)

            # Enable tabs
            if self.is_time_resolved:
                self.tabs.setTabEnabled(2, True)
            self.tabs.setTabEnabled(3, True)

            self._plot_energy()
            self._plot_mode()
            if self.is_time_resolved:
                self._plot_temporal_coeffs()

            self.lbl_status.setText(
                f"Done. {n} modes computed  "
                f"({self._pod_result['cumul_energy'][min(9, n-1)]*100:.1f}% energy in first {min(10, n)})")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText(f"Error: {e}")
        finally:
            self.btn_compute.setEnabled(True)

    # -----------------------------------------------------------------------
    # Plot: energy spectrum
    # -----------------------------------------------------------------------

    def _plot_energy(self):
        r = self._pod_result
        n = r["n_modes"]
        energy_pct = r["energy_frac"] * 100.0
        cumul_pct  = r["cumul_energy"] * 100.0
        modes_x    = np.arange(1, n + 1)

        self.energy_fig.clear()
        ax = self.energy_fig.add_subplot(111)

        ax.bar(modes_x, energy_pct, color="steelblue", alpha=0.8,
               label="Mode energy %")
        ax.set_xlabel("Mode", fontsize=_FONT_AX)
        ax.set_ylabel("Energy  [%]", fontsize=_FONT_AX)
        ax.set_title("POD Energy Spectrum", fontsize=_FONT_AX)
        ax.tick_params(labelsize=_FONT_TICK)
        ax.set_xlim(0.5, n + 0.5)

        # Cumulative on twin axis
        ax2 = ax.twinx()
        ax2.plot(modes_x, cumul_pct, color="darkorange", linewidth=1.5,
                 marker=".", markersize=4, label="Cumulative %")
        ax2.set_ylabel("Cumulative energy  [%]", fontsize=_FONT_AX, color="darkorange")
        ax2.tick_params(labelsize=_FONT_TICK, colors="darkorange")
        ax2.set_ylim(0, 105)

        for thresh, ls in [(50, "--"), (80, "-."), (90, ":")]:
            ax2.axhline(thresh, color="gray", linewidth=0.8,
                        linestyle=ls, alpha=0.7, label=f"{thresh}%")

        # Combine legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=_FONT_LEG, loc="upper right")

        self.energy_fig.tight_layout(pad=0.5)
        self.energy_canvas.draw()

        # Energy threshold readout
        def _mode_at(thresh):
            idx = np.searchsorted(cumul_pct, thresh)
            return int(idx + 1) if idx < n else n

        self.energy_readout.set("50% at mode", str(_mode_at(50)))
        self.energy_readout.set("80% at mode", str(_mode_at(80)))
        self.energy_readout.set("90% at mode", str(_mode_at(90)))

    # -----------------------------------------------------------------------
    # Plot: spatial mode
    # -----------------------------------------------------------------------

    def _plot_mode(self):
        if self._pod_result is None:
            return

        r       = self._pod_result
        mode_n  = self.spin_mode_idx.value() - 1          # 0-based index
        comp_s  = self.combo_component.currentText()       # "U", "V", or "W"
        comp_map = {"U": 0, "V": 1, "W": 2}
        c_idx   = comp_map[comp_s]

        if c_idx >= r["Nc"]:
            return

        phi = r["modes"][mode_n, :, :, c_idx]             # (ny, nx)
        energy_pct = r["energy_frac"][mode_n] * 100.0

        self.mode_fig.clear()
        ax = self.mode_fig.add_subplot(111)

        vmax = np.nanmax(np.abs(phi))
        vmax = vmax if vmax > 0 else 1.0
        levels = np.linspace(-vmax, vmax, 41)

        cf = ax.contourf(self._x, self._y, phi,
                         levels=levels, cmap=_CMAP_DIV, extend="neither")
        cb = self.mode_fig.colorbar(cf, ax=ax, label="φ [ ]", shrink=0.8)
        if self.chk_hide_colorbar.isChecked():
            cb.remove()
            self.mode_fig.tight_layout(pad=0.5)
        ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        ax.set_title(
            f"Mode {mode_n + 1}  —  {comp_s}  ({energy_pct:.2f}% energy)",
            fontsize=_FONT_AX)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=_FONT_TICK)
        if self.chk_hide_axes.isChecked():
            ax.axis('off')
            ax.set_title('')

        self.mode_fig.tight_layout(pad=0.5)
        self.mode_canvas.draw()
        self.mode_toolbar.set_home_limits()

    # -----------------------------------------------------------------------
    # Plot: temporal coefficients
    # -----------------------------------------------------------------------

    def _plot_temporal_coeffs(self):
        if self._pod_result is None or not self.is_time_resolved:
            return

        r      = self._pod_result
        mode_n = self.spin_mode_idx.value() - 1
        a_n    = r["coeffs"][:, mode_n]
        Nt     = len(a_n)
        t      = np.arange(Nt) * self.dt

        self.temporal_fig.clear()
        ax1 = self.temporal_fig.add_subplot(1, 2, 1)
        ax2 = self.temporal_fig.add_subplot(1, 2, 2)

        # Time series
        ax1.plot(t, a_n, color="steelblue", linewidth=0.8)
        ax1.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax1.set_xlabel("t [s]", fontsize=_FONT_AX)
        ax1.set_ylabel("a(t)  [ ]", fontsize=_FONT_AX)
        ax1.set_title(f"Mode {mode_n + 1}  —  temporal coefficient",
                      fontsize=_FONT_AX)
        ax1.tick_params(labelsize=_FONT_TICK)
        ax1.grid(True, alpha=0.3)

        # PSD via Welch
        try:
            from scipy.signal import welch
            nperseg = min(256, Nt // 4) if Nt >= 32 else Nt
            f, Pxx = welch(a_n, fs=self.fs, nperseg=nperseg)
            ax2.semilogy(f, Pxx, color="steelblue", linewidth=1.0)
            ax2.set_xlabel("f [Hz]", fontsize=_FONT_AX)
            ax2.set_ylabel("PSD  [a.u./Hz]", fontsize=_FONT_AX)
            ax2.set_title(f"Mode {mode_n + 1}  —  PSD", fontsize=_FONT_AX)
            ax2.tick_params(labelsize=_FONT_TICK)
            ax2.grid(True, which="both", alpha=0.3)
        except ImportError:
            ax2.text(0.5, 0.5, "scipy not available\n(required for Welch PSD)",
                     transform=ax2.transAxes, ha="center", va="center",
                     color="gray", fontsize=_FONT_AX)

        if self.chk_hide_axes.isChecked():
            for a in self.temporal_fig.axes:
                a.axis('off')
                a.set_title('')
        self.temporal_fig.tight_layout(pad=0.5)
        self.temporal_canvas.draw()
        self.temporal_toolbar.set_home_limits()

    # -----------------------------------------------------------------------
    # Reconstruction
    # -----------------------------------------------------------------------

    def _run_reconstruction(self):
        try:
            self._run_reconstruction_inner()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Reconstruction Error", str(e))

    def _run_reconstruction_inner(self):
        if self._pod_result is None:
            return

        snap_idx  = self.slider_snapshot.value()
        n_recon   = self.spin_n_recon.value()
        comp      = self.combo_component.currentText()   # "U", "V", or "W"

        # Select original snapshot for chosen component
        comp_src = {"U": self.U, "V": self.V, "W": self.W}
        orig_field = comp_src[comp][:, :, snap_idx]

        # Reconstruct all components, then pick the right one
        U_rec, V_rec, W_rec = reconstruct_snapshot(
            self._pod_result, snap_idx, n_recon)
        rec_map = {"U": U_rec, "V": V_rec, "W": W_rec}
        rec_field = rec_map[comp]

        # Residual and error metrics
        diff     = orig_field - rec_field
        rms      = float(np.sqrt(np.nanmean(diff ** 2)))
        rms_orig = float(np.sqrt(np.nanmean(orig_field ** 2)))
        rms_pct  = (rms / rms_orig * 100.0) if rms_orig > 0 else float("nan")

        cb_label  = f"{comp} [m/s]"
        cb_dlabel = f"\u0394{comp} [m/s]"

        def _draw(fig, canvas, field, title, label, symmetric=False):
            fig.clear()
            ax = fig.add_subplot(111)
            if symmetric:
                vmax = float(np.nanmax(np.abs(field)))
                vmax = vmax if vmax > 0 else 1.0
                levels = np.linspace(-vmax, vmax, 41)
            else:
                fmin = float(np.nanmin(field))
                fmax = float(np.nanmax(field))
                levels = np.linspace(fmin, fmax, 41)
            cf = ax.contourf(self._x, self._y, field,
                             levels=levels, cmap=_CMAP_DIV, extend="neither")
            cb = fig.colorbar(cf, ax=ax, label=label, shrink=0.8)
            if self.chk_hide_colorbar.isChecked():
                cb.remove()
                fig.tight_layout(pad=0.5)
            ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
            ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
            ax.set_title(title, fontsize=_FONT_AX)
            ax.set_aspect("equal")
            ax.set_facecolor("white")
            ax.tick_params(labelsize=_FONT_TICK)
            if self.chk_hide_axes.isChecked():
                ax.axis('off')
                ax.set_title('')
            fig.tight_layout(pad=0.5)
            canvas.draw()

        _draw(self.recon_orig_fig, self.recon_orig_canvas,
              orig_field, f"Original {comp}  —  snapshot {snap_idx}",
              label=cb_label)
        self.recon_orig_toolbar.set_home_limits()
        _draw(self.recon_rec_fig, self.recon_rec_canvas,
              rec_field,  f"Reconstructed {comp}  —  {n_recon} modes",
              label=cb_label)
        self.recon_rec_toolbar.set_home_limits()
        _draw(self.recon_diff_fig, self.recon_diff_canvas,
              diff, f"Residual {comp} (original \u2212 reconstructed)",
              label=cb_dlabel, symmetric=True)
        self.recon_diff_toolbar.set_home_limits()

        pct_str = f"{rms_pct:.2f}%" if not np.isnan(rms_pct) else "--"
        self.lbl_recon_error.setText(
            f"RMS error ({comp}): {rms:.4f} m/s  ({pct_str} of original RMS)  |  "
            f"snapshot {snap_idx}  |  {n_recon} modes")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def _export_pod(self):
        if self._pod_result is None:
            return

        # -- Export modes as Tecplot --
        path_dat, _ = QFileDialog.getSaveFileName(
            self, "Export Spatial Modes (Tecplot)",
            "pod_modes.dat", "Tecplot DAT (*.dat);;All Files (*)")
        if not path_dat:
            return

        r       = self._pod_result
        n       = r["n_modes"]
        Nc      = r["Nc"]
        comp_names = ["U", "V", "W"][:Nc]

        fields      = []
        field_names = []
        for mode_n in range(n):
            for c_idx, cname in enumerate(comp_names):
                phi = r["modes"][mode_n, :, :, c_idx]
                fields.append(phi)
                field_names.append(f"phi_mode{mode_n + 1}_{cname}")

        settings = {
            "Analysis": "POD Spatial Modes",
            "Snapshots": self.U.shape[2],
            "Modes computed": n,
            "Nc": Nc,
        }
        try:
            export_2d_tecplot(path_dat, self._x, self._y,
                              fields, field_names, settings)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return

        # -- Export coefficients as CSV --
        base  = path_dat.replace(".dat", "")
        path_csv = base + "_coeffs.csv"

        try:
            coeffs = r["coeffs"]    # (Nt, n_modes)
            Nt     = coeffs.shape[0]
            with open(path_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = (["snapshot", "time_s"] +
                          [f"a_{i+1}" for i in range(n)] +
                          [f"lambda_{i+1}" for i in range(n)] +
                          [f"energy_frac_{i+1}" for i in range(n)])
                writer.writerow(header)
                for t_idx in range(Nt):
                    row = ([t_idx, t_idx * self.dt] +
                           [coeffs[t_idx, i] for i in range(n)] +
                           [r["eigenvalues"][i] for i in range(n)] +
                           [r["energy_frac"][i] for i in range(n)])
                    writer.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return

        self.lbl_status.setText(
            f"Exported modes → {path_dat}\nCoefficients → {path_csv}")
