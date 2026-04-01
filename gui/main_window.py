"""
gui/main_window.py
uPrime - Main Application Window v0.2
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QProgressBar, QSplitter, QGroupBox, QSizePolicy,
    QMessageBox, QSpinBox, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QScrollArea, QStatusBar, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from core.loader import load_dataset
from gui.spectral_window import SpectralWindow
from gui.anisotropy_window import AnisotropyWindow
from gui.reynolds_window import ReynoldsWindow
from gui.tke_window import TKEWindow
from gui.tke_budget_window import TKEBudgetWindow
from gui.spatial_spectra_window import SpatialSpectraWindow
from gui.correlation_window import CorrelationWindow
from gui.arrow_toolbar import PickerMixin, DrawAwareToolbar


# ----------------------------------------------------------------------- #
# Background loader thread
# ----------------------------------------------------------------------- #

class LoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, file_list):
        super().__init__()
        self.file_list = file_list

    def run(self):
        try:
            dataset = load_dataset(self.file_list,
                                   progress_callback=self.progress.emit)
            self.finished.emit(dataset)
        except Exception as e:
            self.error.emit(str(e))


# ----------------------------------------------------------------------- #
# Plot canvas
# ----------------------------------------------------------------------- #

class PlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure  = Figure(tight_layout=True)
        self.canvas  = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.toolbar = DrawAwareToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def get_axes(self):
        self.figure.clear()
        return self.figure.add_subplot(111)


# ----------------------------------------------------------------------- #
# Main window
# ----------------------------------------------------------------------- #

class MainWindow(PickerMixin, QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("uPrime")
        self.setMinimumSize(1100, 650)
        self.resize(1400, 820)
        self.dataset       = None
        self.loader_thread = None
        self._windows      = []
        self._build_ui()

    # ----------------------------------------------------------------------- #
    # UI construction
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ================================================================
        # LEFT SIDEBAR (scrollable)
        # ================================================================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(240)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(8, 8, 8, 8)
        sl.setSpacing(6)

        # -- Logo --
        logo = QLabel("uPrime")
        logo.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(logo)

        sub = QLabel("Because u\u2019 matters")
        sub.setFont(QFont("Arial", 8))
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color: #888; font-style: italic;")
        sl.addWidget(sub)

        sl.addWidget(self._separator())

        # -- 1. Load Data --
        load_grp = QGroupBox("1. Load Data")
        load_lay = QVBoxLayout(load_grp)
        self.btn_load = QPushButton("📂  Select .dat Files...")
        self.btn_load.clicked.connect(self._on_load_files)
        load_lay.addWidget(self.btn_load)
        self.lbl_files = QLabel("No files loaded")
        self.lbl_files.setWordWrap(True)
        self.lbl_files.setStyleSheet("color:#888;font-size:10px;")
        load_lay.addWidget(self.lbl_files)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        load_lay.addWidget(self.progress_bar)
        sl.addWidget(load_grp)

        sl.addWidget(self._separator())

        # -- 2. Acquisition Type --
        self.acq_group = QGroupBox("2. Acquisition Type")
        self.acq_group.setVisible(False)
        acq_lay = QVBoxLayout(self.acq_group)
        acq_row = QHBoxLayout()
        self.rb_tr    = QRadioButton("Time-Resolved")
        self.rb_nontr = QRadioButton("Non-TR")
        self.rb_nontr.setChecked(True)
        self._acq_bg  = QButtonGroup()
        self._acq_bg.addButton(self.rb_tr)
        self._acq_bg.addButton(self.rb_nontr)
        self.rb_tr.toggled.connect(self._on_acq_changed)
        acq_row.addWidget(self.rb_tr)
        acq_row.addWidget(self.rb_nontr)
        acq_lay.addLayout(acq_row)
        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("fs [Hz]:"))
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1, 1e6)
        self.spin_fs.setValue(1000.0)
        self.spin_fs.setDecimals(1)
        self.spin_fs.setEnabled(False)
        fs_row.addWidget(self.spin_fs)
        acq_lay.addLayout(fs_row)
        sl.addWidget(self.acq_group)

        sl.addWidget(self._separator())

        # -- 3. Dataset Info --
        self.info_group = QGroupBox("3. Dataset Info")
        self.info_group.setVisible(False)
        info_lay = QVBoxLayout(self.info_group)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size:10px;")
        info_lay.addWidget(self.lbl_info)
        sl.addWidget(self.info_group)

        sl.addWidget(self._separator())

        # -- 4. Analysis Buttons --
        self.analysis_group = QGroupBox("4. Analysis")
        self.analysis_group.setVisible(False)
        an_lay = QVBoxLayout(self.analysis_group)

        analyses = [
            ("📊  Reynolds Stresses",      self._run_reynolds),
            ("⚡  TKE Budget",             self._run_tke_budget),
            ("〜  Spectral (Temporal)",    self._run_spectral_temporal),
            ("∿  Spectral (Spatial)",     self._run_spectral_spatial),
            ("△  Anisotropy Invariants",  self._run_anisotropy),
            ("⟺  Correlation Analysis",   self._run_correlation),
        ]
        self._analysis_btns = []
        for label, slot in analyses:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn.setStyleSheet("text-align: left; padding: 4px 8px;")
            an_lay.addWidget(btn)
            self._analysis_btns.append(btn)

        sl.addWidget(self.analysis_group)

        sl.addStretch()

        # About button
        self.btn_about = QPushButton("ℹ  About uPrime")
        self.btn_about.setStyleSheet("color:#888;font-size:10px;")
        self.btn_about.clicked.connect(self._on_about)
        sl.addWidget(self.btn_about)

        scroll.setWidget(sidebar)
        root.addWidget(scroll)

        # ================================================================
        # RIGHT: plot area
        # ================================================================
        right_widget = QWidget()
        right_lay    = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(4, 4, 4, 0)
        right_lay.setSpacing(2)

        # -- Options strip (horizontal, above plot) --
        self.options_strip = QWidget()
        self.options_strip.setVisible(False)
        opts = QHBoxLayout(self.options_strip)
        opts.setContentsMargins(4, 2, 4, 2)
        opts.setSpacing(8)

        opts.addWidget(QLabel("Field:"))
        self.combo_field = QComboBox()
        self.combo_field.setMinimumWidth(140)
        self.combo_field.currentIndexChanged.connect(self._on_field_changed)
        opts.addWidget(self.combo_field)

        opts.addWidget(QLabel("Plot:"))
        self.combo_plot = QComboBox()
        self.combo_plot.addItems(["Contourf + Vectors",
                                   "Contourf only",
                                   "Vectors only"])
        self.combo_plot.currentIndexChanged.connect(self._on_field_changed)
        opts.addWidget(self.combo_plot)

        opts.addWidget(self._vsep())
        opts.addWidget(QLabel("Skip x:"))
        self.spin_skip_x = QSpinBox()
        self.spin_skip_x.setRange(1, 50); self.spin_skip_x.setValue(10)
        self.spin_skip_x.setFixedWidth(55)
        opts.addWidget(self.spin_skip_x)

        opts.addWidget(QLabel("Skip y:"))
        self.spin_skip_y = QSpinBox()
        self.spin_skip_y.setRange(1, 50); self.spin_skip_y.setValue(10)
        self.spin_skip_y.setFixedWidth(55)
        opts.addWidget(self.spin_skip_y)

        opts.addWidget(QLabel("Vec length:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.01, 10.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setDecimals(2)
        self.spin_scale.setSingleStep(0.05)
        self.spin_scale.setFixedWidth(65)
        opts.addWidget(self.spin_scale)

        btn_apply = QPushButton("Apply")
        btn_apply.setFixedWidth(60)
        btn_apply.clicked.connect(self._on_field_changed)
        opts.addWidget(btn_apply)
        opts.addStretch()

        right_lay.addWidget(self.options_strip)

        # -- Plot canvas --
        self.plot_canvas = PlotCanvas()
        right_lay.addWidget(self.plot_canvas)

        root.addWidget(right_widget)

        # ================================================================
        # STATUS BAR (bottom, full width)
        # ================================================================
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.lbl_status = QLabel("Ready")
        self.status_bar.addWidget(self.lbl_status, 1)

        credit = QLabel("Jibu Tom Jose  ·  Transient Fluid Mechanics Lab, Technion")
        credit.setStyleSheet("color:#666;font-size:9px;font-style:italic;")
        self.status_bar.addPermanentWidget(credit)

        # Show welcome screen
        self._show_welcome()

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#444;")
        return line

    def _vsep(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("color:#444;")
        return line

    # ----------------------------------------------------------------------- #
    # Welcome screen
    # ----------------------------------------------------------------------- #

    def _show_welcome(self):
        ax = self.plot_canvas.get_axes()
        self.plot_canvas.figure.set_facecolor("#2b2b2b")
        ax.set_facecolor("#2b2b2b")
        ax.axis("off")
        ax.text(0.5, 0.60,
                "Turbulence is complex. Analysis shouldn\u2019t be.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=15, fontstyle="italic", color="#e0e0e0",
                fontweight="light")
        ax.text(0.5, 0.48, "\u2015" * 40,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#555")
        ax.text(0.5, 0.38,
                "These are derived quantities, not conclusions.\nInterpret with care.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, fontstyle="italic", color="#aaaaaa")
        ax.text(0.5, 0.18, "Load .dat files to begin  \u2192",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#777")
        self.plot_canvas.canvas.draw()

    # ----------------------------------------------------------------------- #
    # Load files
    # ----------------------------------------------------------------------- #

    def _on_load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select DaVis .dat files", "",
            "DaVis Data Files (*.dat);;All Files (*)"
        )
        if not paths:
            return
        self.lbl_files.setText(
            f"{len(paths)} file(s)\n{os.path.basename(paths[0])} ...")
        self.lbl_status.setText("Loading files...")
        self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.loader_thread = LoaderThread(paths)
        self.loader_thread.progress.connect(self.progress_bar.setValue)
        self.loader_thread.finished.connect(self._on_load_finished)
        self.loader_thread.error.connect(self._on_load_error)
        self.loader_thread.start()

    def _on_load_finished(self, dataset):
        self.dataset = dataset
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)

        Nt     = dataset["Nt"]
        nx     = dataset["nx"]
        ny     = dataset["ny"]
        stereo = dataset["is_stereo"]
        x      = dataset["x"]
        y      = dataset["y"]

        dx = abs(x[0, 1] - x[0, 0])
        dy = abs(y[1, 0] - y[0, 0])
        mem_mb = (3 if stereo else 2) * ny * nx * Nt * 4 / 1e6

        self.lbl_info.setText(
            f"Grid : {nx} \u00d7 {ny}\n"
            f"dx/dy : {dx:.3f} / {dy:.3f} mm\n"
            f"Snapshots : {Nt}\n"
            f"Type : {'Stereo' if stereo else '2D'}\n"
            f"Memory : ~{mem_mb:.0f} MB"
        )
        self.info_group.setVisible(True)
        self.acq_group.setVisible(True)
        self.analysis_group.setVisible(True)
        self.options_strip.setVisible(True)

        # Populate field selector
        self.combo_field.blockSignals(True)
        self.combo_field.clear()
        self.combo_field.addItems(["Mean |V| (speed)", "Mean U", "Mean V"])
        if stereo:
            self.combo_field.addItem("Mean W")
        self.combo_field.addItems(["Std U", "Std V"])
        if dataset.get("has_vort", False):
            self.combo_field.addItem("Mean Vorticity")
        self.combo_field.blockSignals(False)

        # Auto skip based on grid
        default_skip = max(1, min(nx, ny) // 20)
        self.spin_skip_x.setValue(default_skip)
        self.spin_skip_y.setValue(default_skip)

        self._check_convergence(Nt)
        self.lbl_status.setText(f"Loaded {Nt} snapshots.")
        self._plot_field()

        # Setup picker on main plot
        if self.plot_canvas.figure.axes:
            self._setup_picker(
                self.plot_canvas.canvas,
                self.plot_canvas.figure.axes[0],
                status_label=self.lbl_status
            )
        self._x = x
        self._y = y

    def _on_load_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.lbl_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Load Error", f"Failed to load:\n{msg}")

    def _check_convergence(self, Nt):
        if self.rb_tr.isChecked():
            fs  = self.spin_fs.value()
            dur = Nt / fs
            if dur < 2.0:
                QMessageBox.warning(self, "Convergence Warning",
                    f"{Nt} snapshots @ {fs:.0f} Hz = {dur:.2f} s.\n"
                    "Less than 2 s -- statistics may not be converged.")
        else:
            if Nt < 2000:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Only {Nt} snapshots (< 2000 recommended).\n"
                    "Results may not be statistically converged.")

    def _on_acq_changed(self):
        self.spin_fs.setEnabled(self.rb_tr.isChecked())

    def is_time_resolved(self):
        return self.rb_tr.isChecked()

    def get_fs(self):
        return self.spin_fs.value()

    # ----------------------------------------------------------------------- #
    # Plot field
    # ----------------------------------------------------------------------- #

    def _on_field_changed(self):
        if self.dataset is not None:
            self._plot_field()

    def _plot_field(self):
        if self.dataset is None:
            return

        ds         = self.dataset
        x, y       = ds["x"], ds["y"]
        U, V, W    = ds["U"], ds["V"], ds["W"]
        field_name = self.combo_field.currentText()
        plot_type  = self.combo_plot.currentText()
        skip_x     = self.spin_skip_x.value()
        skip_y     = self.spin_skip_y.value()
        vec_scale  = self.spin_scale.value()

        mean_u = np.nanmean(U, axis=2)
        mean_v = np.nanmean(V, axis=2)
        valid_frac   = np.mean(ds["valid"], axis=2)
        invalid_mask = valid_frac < 0.5

        if field_name == "Mean U":
            field = mean_u.copy(); field[invalid_mask] = np.nan
            cbar  = "Mean U [m/s]"
        elif field_name == "Mean V":
            field = mean_v.copy(); field[invalid_mask] = np.nan
            cbar  = "Mean V [m/s]"
        elif field_name == "Mean W" and W is not None:
            field = np.nanmean(W, axis=2); field[invalid_mask] = np.nan
            cbar  = "Mean W [m/s]"
        elif field_name == "Std U":
            field = np.nanstd(U, axis=2); field[invalid_mask] = np.nan
            cbar  = "Std(U) [m/s]"
        elif field_name == "Std V":
            field = np.nanstd(V, axis=2); field[invalid_mask] = np.nan
            cbar  = "Std(V) [m/s]"
        elif field_name == "Mean Vorticity" and ds.get("vort") is not None:
            field = np.nanmean(ds["vort"], axis=2); field[invalid_mask] = np.nan
            cbar  = "Vorticity [1/s]"
        else:
            mu = mean_u; mv = mean_v
            if W is not None:
                mw    = np.nanmean(W, axis=2)
                field = np.sqrt(mu**2 + mv**2 + mw**2)
            else:
                field = np.sqrt(mu**2 + mv**2)
            field[invalid_mask] = np.nan
            cbar = "Mean |V| [m/s]"

        # Reset figure background to default
        self.plot_canvas.figure.set_facecolor("white")
        ax = self.plot_canvas.get_axes()

        if "Contourf" in plot_type:
            cf = ax.contourf(x, y, field, levels=50, cmap="RdBu_r")
            self.plot_canvas.figure.colorbar(cf, ax=ax,
                                             label=cbar, shrink=0.8)

        if "Vectors" in plot_type:
            xs  = x[::skip_y, ::skip_x]
            ys  = y[::skip_y, ::skip_x]
            us  = mean_u[::skip_y, ::skip_x].copy()
            vs  = mean_v[::skip_y, ::skip_x].copy()
            inv = invalid_mask[::skip_y, ::skip_x]
            us[inv] = np.nan; vs[inv] = np.nan
            mag = np.sqrt(us**2 + vs**2)
            mag[mag == 0] = np.nan
            un  = us / mag; vn = vs / mag

            # Auto-normalize: scale quiver so length=1 gives ~1/20 of domain
            # User scale is a multiplier (1.0 = default, <1 shorter, >1 longer)
            x_range  = float(np.nanmax(xs) - np.nanmin(xs))
            y_range  = float(np.nanmax(ys) - np.nanmin(ys))
            domain   = max(x_range, y_range)
            n_arrows = max(xs.shape[0], xs.shape[1])
            # base_scale: makes arrows fit neatly between grid points
            base_scale = n_arrows / (domain * vec_scale + 1e-9)

            ax.quiver(xs, ys, un, vn, color="k",
                      scale=base_scale,
                      scale_units="xy",
                      angles="xy",
                      alpha=0.7, width=0.002,
                      headwidth=3, headlength=4)

        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(field_name, fontsize=10)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        self.plot_canvas.canvas.draw()

        # Update picker
        if self.plot_canvas.figure.axes:
            self._pick_field_ax = self.plot_canvas.figure.axes[0]
        self._last_field_values = field
        self.lbl_status.setText(f"Displaying: {field_name}")

    # ----------------------------------------------------------------------- #
    # Analysis launchers
    # ----------------------------------------------------------------------- #

    def _check_data(self):
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return False
        return True

    def _check_tr(self, name):
        if not self.is_time_resolved():
            QMessageBox.warning(self, "Time-Resolved Required",
                f"'{name}' requires time-resolved data.\n"
                "Select 'Time-Resolved' in Acquisition Type and set fs.")
            return False
        return True

    def _open_window(self, win):
        self._windows.append(win)
        win.show(); win.raise_(); win.activateWindow()

    def _run_reynolds(self):
        if not self._check_data(): return
        Nt = self.dataset["Nt"]; fs = self.get_fs()
        dur = Nt / fs if self.is_time_resolved() else 9999
        self._open_window(ReynoldsWindow(
            self.dataset, is_time_resolved=self.is_time_resolved(),
            Nt_warn=Nt, duration_warn=dur))

    def _run_tke_budget(self):
        if not self._check_data(): return
        Nt = self.dataset["Nt"]; fs = self.get_fs()
        dur = Nt / fs if self.is_time_resolved() else 9999
        self._open_window(TKEBudgetWindow(
            self.dataset, is_time_resolved=self.is_time_resolved(),
            Nt_warn=Nt, duration_warn=dur))

    def _run_spectral_temporal(self):
        if not self._check_data(): return
        if not self._check_tr("Spectral (Temporal)"): return
        self._open_window(SpectralWindow(
            self.dataset, default_fs=self.get_fs()))

    def _run_spectral_spatial(self):
        if not self._check_data(): return
        self._open_window(SpatialSpectraWindow(self.dataset))

    def _run_anisotropy(self):
        if not self._check_data(): return
        if not self.dataset["is_stereo"]:
            QMessageBox.warning(self, "Stereo Required",
                "Anisotropy analysis requires stereo PIV (u, v, w).")
            return
        self._open_window(AnisotropyWindow(self.dataset))

    def _run_correlation(self):
        if not self._check_data(): return
        Nt  = self.dataset["Nt"]
        fs  = self.get_fs()
        is_tr = self.is_time_resolved()
        dur = Nt / fs if is_tr else 9999.0
        self._open_window(CorrelationWindow(
            self.dataset,
            fs=fs,
            is_time_resolved=is_tr,
            Nt_warn=Nt,
            duration_warn=dur,
        ))

    # ----------------------------------------------------------------------- #
    # About dialog
    # ----------------------------------------------------------------------- #

    def _on_about(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("About uPrime")
        dlg.setFixedWidth(420)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8); lay.setContentsMargins(24, 24, 24, 24)

        for text, size, bold, italic, color in [
            ("uPrime",                          22, True,  False, None),
            ("Because u\u2019 matters",         10, False, True,  "gray"),
            ("v0.2  \u00b7  Alpha Release",     9,  False, False, "gray"),
        ]:
            lbl = QLabel(text)
            f   = QFont("Arial", size, QFont.Weight.Bold if bold else QFont.Weight.Normal)
            f.setItalic(italic)
            lbl.setFont(f)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if color: lbl.setStyleSheet(f"color:{color};")
            lay.addWidget(lbl)

        lay.addWidget(self._separator_dlg())

        desc = QLabel(
            "Open-source fluid velocity field analysis.\n"
            "Supports Tecplot .dat format from DaVis and CFD solvers."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size:10px;")
        lay.addWidget(desc)

        lay.addWidget(self._separator_dlg())

        credit = QLabel(
            "Developed by <b>Jibu Tom Jose</b><br>"
            "Transient Fluid Mechanics Lab, Technion<br>"
            "Built with assistance from Claude (Anthropic)"
        )
        credit.setWordWrap(True)
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit.setStyleSheet("font-size:10px;")
        lay.addWidget(credit)

        gh = QLabel(
            "GitHub: <i>github.com/CmdrRyder/uPrime</i><br>"
            "Lab: <i>[lab website placeholder]</i><br>"
            "Licensed under CC BY-NC-ND 4.0<br>"
            "Cite: Jibu Tom Jose, uPrime, Technion (2026)"
        )
        gh.setWordWrap(True)
        gh.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gh.setStyleSheet("color:gray;font-size:9px;")
        lay.addWidget(gh)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()

    def _separator_dlg(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#444;")
        return line