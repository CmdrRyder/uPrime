"""
gui/dmd_window.py
-----------------
Dynamic Mode Decomposition analysis window for uPrime. TR-only.

Tab 1: DMD Spectrum -- scatter plot of frequency vs growth rate,
       bubble size/color proportional to mode amplitude.
Tab 2: Mode Viewer  -- spatial structure of selected mode.
"""

import csv
import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QSizePolicy, QMessageBox, QSplitter, QTabWidget,
    QFileDialog, QProgressBar,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from core.dmd import (
    build_snapshot_matrix, compute_dmd, scale_to_physical, get_mode_components
)
from core.export import export_2d_tecplot

_FONT_AX   = 9
_FONT_TICK = 8


class DmdWindow(PickerMixin, QWidget):

    def __init__(self, dataset, fs, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("DMD Analysis")
        self.resize(1400, 900)

        self.dataset = dataset
        self.fs      = fs
        self._x      = dataset["x"]
        self._y      = dataset["y"]

        self._valid              = False
        self._dmd_result         = None
        self._selected_mode      = None
        self._n_per              = None
        self._n_components       = None
        self._x_vals             = None
        self._gr_phys            = None
        self._filter_mask        = None
        self._modes_by_amplitude = []
        self._mode_rank          = 0
        self._spectrum_cbar      = None

        Nt = dataset["Nt"]
        if Nt < 50:
            QMessageBox.critical(self, "Insufficient Snapshots",
                f"DMD requires at least 50 snapshots. This dataset has {Nt}.")
            return

        if Nt < 200:
            freq_res = fs / Nt
            QMessageBox.warning(self, "Low Snapshot Count",
                f"Only {Nt} snapshots available. DMD results may be unreliable.\n"
                f"Recommended: 200+ snapshots.\n"
                f"At fs={fs:.0f} Hz, this covers only {Nt/fs*1000:.1f} ms\n"
                f"({freq_res:.1f} Hz frequency resolution).\n"
                f"Recommended duration: {200/fs*1000:.1f} ms (200 snapshots).")

        self._build_ui()

        # Set spin maxima / defaults now that Nt and fs are known
        self.spin_n_modes.setMaximum(min(Nt - 1, 500))
        self.spin_sigma_filter.setValue(10.0 * fs)

        self._valid = True

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

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

        # -- Input group --
        inp_grp = QGroupBox("Input")
        inp_lay = QVBoxLayout(inp_grp)

        inp_lay.addWidget(QLabel("Component:"))
        self.combo_comp = QComboBox()
        self.combo_comp.addItem("Stacked (all components)", "stacked")
        self.combo_comp.addItem("U only", "U")
        self.combo_comp.addItem("V only", "V")
        self.combo_comp.addItem("W only", "W")
        if not self.dataset.get("is_stereo", False):
            self.combo_comp.model().item(3).setEnabled(False)
        inp_lay.addWidget(self.combo_comp)

        self.chk_subtract_mean = QCheckBox("Subtract mean")
        self.chk_subtract_mean.setChecked(True)
        inp_lay.addWidget(self.chk_subtract_mean)

        modes_row = QHBoxLayout()
        modes_row.addWidget(QLabel("Modes:"))
        self.spin_n_modes = QSpinBox()
        self.spin_n_modes.setRange(10, 500)
        self.spin_n_modes.setValue(50)
        modes_row.addWidget(self.spin_n_modes)
        inp_lay.addLayout(modes_row)

        self.btn_compute = QPushButton("Compute DMD")
        self.btn_compute.clicked.connect(self._on_compute)
        inp_lay.addWidget(self.btn_compute)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        inp_lay.addWidget(self.progress_bar)

        self.lbl_status_compute = QLabel("")
        self.lbl_status_compute.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status_compute.setWordWrap(True)
        inp_lay.addWidget(self.lbl_status_compute)

        ll.addWidget(inp_grp)

        # -- Spectrum group --
        spec_grp = QGroupBox("Spectrum")
        spec_lay = QVBoxLayout(spec_grp)

        self.chk_strouhal = QCheckBox("Use Strouhal number")
        self.chk_strouhal.setChecked(False)
        self.chk_strouhal.toggled.connect(self._on_strouhal_toggled)
        spec_lay.addWidget(self.chk_strouhal)

        um_row = QHBoxLayout()
        um_row.addWidget(QLabel("Um [m/s]:"))
        self.spin_Um = QDoubleSpinBox()
        self.spin_Um.setRange(0.001, 1000)
        self.spin_Um.setValue(1.0)
        self.spin_Um.setDecimals(3)
        self.spin_Um.setEnabled(False)
        um_row.addWidget(self.spin_Um)
        spec_lay.addLayout(um_row)

        L_row = QHBoxLayout()
        L_row.addWidget(QLabel("L [mm]:"))
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.001, 9999)
        self.spin_L.setValue(1.0)
        self.spin_L.setDecimals(3)
        self.spin_L.setEnabled(False)
        L_row.addWidget(self.spin_L)
        spec_lay.addLayout(L_row)

        spec_lay.addWidget(QLabel("Growth rate filter |σ| <:"))
        self.spin_sigma_filter = QDoubleSpinBox()
        self.spin_sigma_filter.setRange(0, 1e9)
        self.spin_sigma_filter.setValue(1e6)
        self.spin_sigma_filter.setDecimals(1)
        self.spin_sigma_filter.setSuffix("  σ_max [1/s]")
        spec_lay.addWidget(self.spin_sigma_filter)

        freq_min_row = QHBoxLayout()
        freq_min_row.addWidget(QLabel("Min frequency [Hz]:"))
        self.spin_freq_min = QDoubleSpinBox()
        self.spin_freq_min.setRange(0.0, 1e6)
        self.spin_freq_min.setValue(0.5)
        self.spin_freq_min.setDecimals(2)
        freq_min_row.addWidget(self.spin_freq_min)
        spec_lay.addLayout(freq_min_row)

        self.lbl_selected = QLabel("Selected mode: --")
        self.lbl_selected.setStyleSheet("font-size: 10px;")
        self.lbl_selected.setWordWrap(True)
        spec_lay.addWidget(self.lbl_selected)

        self.chk_show_labels = QCheckBox("Show mode labels")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.toggled.connect(self._plot_spectrum)
        spec_lay.addWidget(self.chk_show_labels)

        self.chk_show_colorbar = QCheckBox("Show colorbar")
        self.chk_show_colorbar.setChecked(True)
        self.chk_show_colorbar.toggled.connect(self._plot_spectrum)
        spec_lay.addWidget(self.chk_show_colorbar)

        self.btn_replot_spectrum = QPushButton("Update Spectrum")
        self.btn_replot_spectrum.setEnabled(False)
        self.btn_replot_spectrum.clicked.connect(self._plot_spectrum)
        spec_lay.addWidget(self.btn_replot_spectrum)

        lbl_hint = QLabel("💡 Click a bubble in the spectrum\nto view its spatial structure.")
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color: #888888; font-style: italic;")
        spec_lay.addWidget(lbl_hint)

        ll.addWidget(spec_grp)

        # -- Export group --
        exp_grp = QGroupBox("Export")
        exp_lay = QVBoxLayout(exp_grp)

        self.btn_export_spectrum = QPushButton("Export spectrum CSV...")
        self.btn_export_spectrum.setEnabled(False)
        self.btn_export_spectrum.clicked.connect(self._on_export_spectrum)
        exp_lay.addWidget(self.btn_export_spectrum)

        self.btn_export_mode = QPushButton("Export mode field...")
        self.btn_export_mode.setEnabled(False)
        self.btn_export_mode.clicked.connect(self._on_export_mode)
        exp_lay.addWidget(self.btn_export_mode)

        ll.addWidget(exp_grp)
        ll.addStretch()

        splitter.addWidget(left)

        # ---- Right panel: tabs ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_spectrum_tab(), "DMD Spectrum")
        self.tabs.addTab(self._build_mode_tab(),     "Mode Viewer")
        rl.addWidget(self.tabs)

        splitter.addWidget(right)
        splitter.setSizes([260, 1140])

    def _build_spectrum_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        self.spec_fig    = Figure()
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        self.spec_toolbar = DrawAwareToolbar(self.spec_canvas, w)
        vl.addWidget(self.spec_toolbar)
        vl.addWidget(self.spec_canvas, stretch=1)

        return w

    def _build_mode_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        # Toolbar row: Display / Colormap / Prev / Counter / Next
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(8)
        toolbar_row.setContentsMargins(4, 4, 4, 4)

        toolbar_row.addWidget(QLabel("Display:"))
        self.combo_part = QComboBox()
        self.combo_part.addItem("Real part",      "real")
        self.combo_part.addItem("Imaginary part", "imag")
        self.combo_part.addItem("Magnitude",      "abs")
        toolbar_row.addWidget(self.combo_part)

        toolbar_row.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["RdBu_r", "viridis", "plasma", "seismic"])
        toolbar_row.addWidget(self.combo_cmap)

        toolbar_row.addStretch()

        self.btn_prev_mode = QPushButton("← Prev")
        self.btn_prev_mode.setMinimumWidth(70)
        self.btn_prev_mode.setEnabled(False)
        self.btn_prev_mode.clicked.connect(self._on_prev_mode)
        toolbar_row.addWidget(self.btn_prev_mode)

        self.lbl_mode_counter = QLabel("Mode -- of --")
        self.lbl_mode_counter.setMinimumWidth(100)
        self.lbl_mode_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar_row.addWidget(self.lbl_mode_counter)

        self.btn_next_mode = QPushButton("Next →")
        self.btn_next_mode.setMinimumWidth(70)
        self.btn_next_mode.setEnabled(False)
        self.btn_next_mode.clicked.connect(self._on_next_mode)
        toolbar_row.addWidget(self.btn_next_mode)

        vl.addLayout(toolbar_row)

        self.mode_fig    = Figure()
        self.mode_canvas = FigureCanvas(self.mode_fig)
        self.mode_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        self.mode_toolbar = DrawAwareToolbar(self.mode_canvas, w)
        vl.addWidget(self.mode_toolbar)
        vl.addWidget(self.mode_canvas, stretch=1)
        return w

    # ----------------------------------------------------------------------- #
    # Slots
    # ----------------------------------------------------------------------- #

    def _on_strouhal_toggled(self, checked):
        self.spin_Um.setEnabled(checked)
        self.spin_L.setEnabled(checked)
        if self._dmd_result is not None:
            self._plot_spectrum()

    def _on_prev_mode(self):
        if not self._modes_by_amplitude:
            return
        self._mode_rank = max(0, self._mode_rank - 1)
        self._update_mode_counter()
        self._plot_mode(self._modes_by_amplitude[self._mode_rank])

    def _on_next_mode(self):
        if not self._modes_by_amplitude:
            return
        self._mode_rank = min(len(self._modes_by_amplitude) - 1, self._mode_rank + 1)
        self._update_mode_counter()
        self._plot_mode(self._modes_by_amplitude[self._mode_rank])

    def _update_mode_counter(self):
        if not self._modes_by_amplitude:
            self.lbl_mode_counter.setText("Mode -- of --")
        else:
            self.lbl_mode_counter.setText(
                f"Mode {self._mode_rank + 1} of {len(self._modes_by_amplitude)}")

    # ----------------------------------------------------------------------- #
    # Compute
    # ----------------------------------------------------------------------- #

    def _on_compute(self):
        from core.dataset_utils import get_masked
        from core.workers import DMDWorker

        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        U = get_masked(self.dataset, "U")
        V = get_masked(self.dataset, "V")
        W = get_masked(self.dataset, "W")

        self.lbl_status_compute.setText("Computing DMD...")
        self.btn_compute.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self._worker = DMDWorker(
            U, V, W,
            mask=self.dataset["MASK"],
            component=self.combo_comp.currentData(),
            n_modes=self.spin_n_modes.value(),
            subtract_mean=self.chk_subtract_mean.isChecked(),
            fs=self.fs,
        )
        self._worker.finished.connect(self._on_dmd_result)
        self._worker.error.connect(self._on_dmd_error)
        self._worker.finished.connect(lambda _: self._reset_compute_ui())
        self._worker.error.connect(lambda _: self._reset_compute_ui())
        self._worker.start()

    def _reset_compute_ui(self):
        self.btn_compute.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)

    def _on_dmd_result(self, result):
        self._dmd_result   = result
        self._n_per        = result['n_per']
        self._n_components = result['n_components']

        self._modes_by_amplitude = np.argsort(
            result['amplitudes'])[::-1].tolist()
        self._mode_rank = 0
        self._update_mode_counter()
        self.btn_prev_mode.setEnabled(True)
        self.btn_next_mode.setEnabled(True)
        self.btn_replot_spectrum.setEnabled(True)
        self.btn_export_spectrum.setEnabled(True)

        self.lbl_status_compute.setText(
            f"Done. {result['rank']} modes computed.")

        self._plot_spectrum()

        default_rank = 0
        for rank_idx, mode_idx in enumerate(self._modes_by_amplitude):
            if abs(result['frequencies_hz'][mode_idx]) >= self.spin_freq_min.value():
                default_rank = rank_idx
                break
        self._mode_rank = default_rank
        first_mode_idx = self._modes_by_amplitude[self._mode_rank]
        self._plot_mode(first_mode_idx)
        n = len(self._modes_by_amplitude)
        self.lbl_mode_counter.setText(f"Mode {self._mode_rank + 1} of {n}")

    def _on_dmd_error(self, tb_str):
        QMessageBox.critical(self, "DMD Error", tb_str)
        self.lbl_status_compute.setText("Error — see dialog.")

    # ----------------------------------------------------------------------- #
    # Spectrum plot
    # ----------------------------------------------------------------------- #

    def _plot_spectrum(self):
        if self._dmd_result is None:
            return

        r   = self._dmd_result
        freq_hz = r['frequencies_hz']
        gr_phys = r['growth_rates_phys']
        amps    = r['amplitudes']

        # Filter: positive frequencies + growth rate filter + min frequency
        sigma_max  = self.spin_sigma_filter.value()
        freq_min   = self.spin_freq_min.value()
        pos_mask   = freq_hz >= 0
        sigma_mask = np.abs(gr_phys) < sigma_max
        freq_mask  = np.abs(freq_hz) >= freq_min
        filt       = pos_mask & sigma_mask & freq_mask

        freq_f  = freq_hz[filt]
        gr_f    = gr_phys[filt]
        amps_f  = amps[filt]

        # Store for pick indexing (maps filtered index → original index)
        self._filter_mask    = filt
        self._filter_indices = np.where(filt)[0]
        original_indices     = self._filter_indices

        if self.chk_strouhal.isChecked():
            Um = self.spin_Um.value()
            L  = self.spin_L.value() / 1000.0
            x_vals  = freq_f * L / Um
            x_label = "Strouhal number  St = fL/U"
        else:
            x_vals  = freq_f
            x_label = "Frequency [Hz]"

        self._x_vals  = x_vals
        self._gr_phys = gr_f

        sizes  = ((amps_f / amps_f.max()) * 200
                  if amps_f.max() > 0 else np.full(len(amps_f), 20))
        colors = np.log10(amps_f + 1e-12)

        self.spec_fig.clear()
        ax = self.spec_fig.add_subplot(111)

        sc = ax.scatter(x_vals, gr_f, s=sizes, c=colors,
                        cmap='viridis', alpha=0.7,
                        picker=True, pickradius=5)

        if hasattr(self, '_spectrum_cbar') and self._spectrum_cbar is not None:
            try:
                self._spectrum_cbar.remove()
            except Exception:
                pass
            self._spectrum_cbar = None
        if self.chk_show_colorbar.isChecked():
            self._spectrum_cbar = self.spec_fig.colorbar(
                sc, ax=ax, label="log₁₀(amplitude)", shrink=0.7)

        if self.chk_show_labels.isChecked():
            for i, (xv, yv) in enumerate(zip(x_vals, gr_f)):
                ax.annotate(
                    str(original_indices[i]),
                    xy=(xv, yv),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=8,
                    color='black',
                    ha='left', va='bottom',
                )

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel(x_label, fontsize=_FONT_AX)
        ax.set_ylabel("Growth rate σ [1/s]", fontsize=_FONT_AX)
        ax.set_title("DMD Spectrum", fontsize=_FONT_AX)
        ax.tick_params(labelsize=_FONT_TICK)

        # Highlight previously selected mode if still visible
        if self._selected_mode is not None:
            orig_idx = self._selected_mode
            hit = np.where(self._filter_indices == orig_idx)[0]
            if len(hit):
                i = hit[0]
                ax.scatter([x_vals[i]], [gr_f[i]],
                           s=300, facecolors='none',
                           edgecolors='red', linewidths=2, zorder=5)

        self.spec_fig.tight_layout(pad=0.5)
        self.spec_canvas.draw()
        self.spec_toolbar.set_home_limits()

        # Connect pick event
        self.spec_fig.canvas.mpl_connect('pick_event', self._on_pick)

    def _on_pick(self, event):
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
        i = event.ind[0]
        orig_idx = int(self._filter_indices[i])
        self._selected_mode = orig_idx

        r       = self._dmd_result
        freq    = r['frequencies_hz'][orig_idx]
        sigma   = r['growth_rates_phys'][orig_idx]
        st_str  = ""
        if self.chk_strouhal.isChecked():
            St     = freq * self.spin_L.value() / 1000.0 / self.spin_Um.value()
            st_str = f"  |  St = {St:.3f}"
        self.lbl_selected.setText(
            f"Mode {orig_idx}  |  f = {freq:.2f} Hz{st_str}  |  σ = {sigma:.2f} 1/s")

        # Sync mode_rank to clicked mode's position in amplitude list
        if orig_idx in self._modes_by_amplitude:
            self._mode_rank = self._modes_by_amplitude.index(orig_idx)
            self._update_mode_counter()

        # Redraw spectrum with highlight
        self._plot_spectrum()

        # Switch to mode viewer and plot
        self.tabs.setCurrentIndex(1)
        self._plot_mode(orig_idx)
        self.btn_export_mode.setEnabled(True)

    # ----------------------------------------------------------------------- #
    # Mode plot
    # ----------------------------------------------------------------------- #

    def _plot_mode(self, mode_idx):
        r          = self._dmd_result
        ny, nx     = self._y.shape[0], self._x.shape[1]
        mask       = self.dataset["MASK"]
        mode_flat  = r['modes'][:, mode_idx]
        part       = self.combo_part.currentData()
        cmap       = self.combo_cmap.currentText()

        components = get_mode_components(
            mode_flat, self._n_per, self._n_components, ny, nx, mask)

        comp_key = self.combo_comp.currentData()
        if comp_key == 'stacked':
            comp_names = ["U-mode", "V-mode", "W-mode"][:self._n_components]
        else:
            comp_names = [comp_key + "-mode"]

        self.mode_fig.clear()
        n_comp = len(components)
        axes   = [self.mode_fig.add_subplot(1, n_comp, i + 1)
                  for i in range(n_comp)]

        x, y = self._x, self._y

        for ax, field, name in zip(axes, components, comp_names):
            if part == 'real':
                data = field.real
            elif part == 'imag':
                data = field.imag
            else:
                data = np.abs(field)

            if part == 'abs':
                vmin = 0
                vmax = float(np.nanmax(data)) if np.any(np.isfinite(data)) else 1.0
            else:
                vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
                vmin = -vmax

            im = ax.imshow(data, origin='lower', aspect='equal',
                           extent=[x.min(), x.max(), y.min(), y.max()],
                           cmap=cmap, vmin=vmin, vmax=vmax)
            self.mode_fig.colorbar(im, ax=ax, shrink=0.6)
            ax.set_title(name, fontsize=_FONT_AX)
            ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
            ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
            ax.tick_params(labelsize=_FONT_TICK)

        freq  = r['frequencies_hz'][mode_idx]
        sigma = r['growth_rates_phys'][mode_idx]
        st_str = ""
        if self.chk_strouhal.isChecked():
            St     = freq * self.spin_L.value() / 1000.0 / self.spin_Um.value()
            st_str = f"  |  St = {St:.3f}"
        self.mode_fig.suptitle(
            f"Mode {mode_idx}  |  f = {freq:.2f} Hz{st_str}  |  σ = {sigma:.2f} 1/s",
            fontsize=_FONT_AX)

        self.mode_fig.tight_layout(pad=0.5)
        self.mode_canvas.draw()
        self.mode_toolbar.set_home_limits()

        self._selected_mode = mode_idx
        self.btn_export_mode.setEnabled(True)

    # ----------------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------------- #

    def _on_export_spectrum(self):
        if self._dmd_result is None:
            QMessageBox.warning(self, "No Data", "Compute DMD first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export DMD Spectrum", "dmd_spectrum.csv",
            "CSV (*.csv);;All Files (*)")
        if not path:
            return

        r = self._dmd_result
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["mode_index", "frequency_hz", "growth_rate_1s",
                     "amplitude", "strouhal"])
                for i in range(r['rank']):
                    freq  = r['frequencies_hz'][i]
                    sigma = r['growth_rates_phys'][i]
                    amp   = r['amplitudes'][i]
                    if self.chk_strouhal.isChecked():
                        st = freq * self.spin_L.value() / 1000.0 / self.spin_Um.value()
                    else:
                        st = float('nan')
                    writer.writerow([i, freq, sigma, amp, st])
            self.lbl_status_compute.setText(
                f"Spectrum exported: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_mode(self):
        if self._selected_mode is None:
            QMessageBox.warning(self, "No Mode Selected",
                "Click a mode in the DMD Spectrum first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mode Field", "dmd_mode.dat",
            "Tecplot DAT (*.dat);;All Files (*)")
        if not path:
            return

        r         = self._dmd_result
        mode_idx  = self._selected_mode
        ny, nx    = self._y.shape[0], self._x.shape[1]
        mask      = self.dataset["MASK"]
        mode_flat = r['modes'][:, mode_idx]

        components = get_mode_components(
            mode_flat, self._n_per, self._n_components, ny, nx, mask)

        comp_key = self.combo_comp.currentData()
        if comp_key == 'stacked':
            comp_names = ["U_mode", "V_mode", "W_mode"][:self._n_components]
        else:
            comp_names = [comp_key + "_mode"]

        fields, labels = [], []
        for field, name in zip(components, comp_names):
            fields.append(field.real);  labels.append(name + "_real")
            fields.append(field.imag);  labels.append(name + "_imag")
            fields.append(np.abs(field)); labels.append(name + "_abs")

        freq  = r['frequencies_hz'][mode_idx]
        sigma = r['growth_rates_phys'][mode_idx]
        settings = {
            "Analysis"  : "DMD Mode",
            "Mode index": mode_idx,
            "Frequency" : f"{freq:.4f} Hz",
            "Growth rate": f"{sigma:.4f} 1/s",
        }

        try:
            export_2d_tecplot(path, self._x, self._y, fields, labels, settings)
            self.lbl_status_compute.setText(
                f"Mode exported: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
