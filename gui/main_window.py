"""
gui/main_window.py
Main application window with file loader and plot canvas.
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QProgressBar, QSplitter, QGroupBox, QSizePolicy,
    QMessageBox, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
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


class LoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, file_list):
        super().__init__()
        self.file_list = file_list

    def run(self):
        try:
            dataset = load_dataset(self.file_list, progress_callback=self.progress.emit)
            self.finished.emit(dataset)
        except Exception as e:
            self.error.emit(str(e))


class PlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure  = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas  = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def get_axes(self):
        self.figure.clear()
        return self.figure.add_subplot(111)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("uPrime")
        self.resize(1300, 800)
        self.dataset       = None
        self.loader_thread = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ---- Left panel ----
        left = QWidget()
        left.setMaximumWidth(300)
        left.setMinimumWidth(240)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(8)

        title = QLabel("uPrime")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ll.addWidget(title)

        subtitle = QLabel("Because u’ matters")
        subtitle.setFont(QFont("Arial", 9))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: gray; font-style: italic;")
        ll.addWidget(subtitle)

        # File group
        fg = QGroupBox("1. Load Data")
        fl = QVBoxLayout(fg)
        self.btn_load = QPushButton("Select .dat Files...")
        self.btn_load.clicked.connect(self._on_load_files)
        fl.addWidget(self.btn_load)
        self.lbl_files = QLabel("No files loaded")
        self.lbl_files.setWordWrap(True)
        self.lbl_files.setStyleSheet("color: gray; font-size: 11px;")
        fl.addWidget(self.lbl_files)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        fl.addWidget(self.progress_bar)
        ll.addWidget(fg)

        # Info group
        self.info_group = QGroupBox("Dataset Info")
        self.info_group.setVisible(False)
        il = QVBoxLayout(self.info_group)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 11px;")
        il.addWidget(self.lbl_info)
        ll.addWidget(self.info_group)

        # Acquisition type group
        self.acq_group = QGroupBox("2. Acquisition Type")
        self.acq_group.setVisible(False)
        acq_lay = QVBoxLayout(self.acq_group)

        acq_mode_row = QHBoxLayout()
        self.rb_time_resolved  = QRadioButton("Time-Resolved")
        self.rb_non_tr         = QRadioButton("Non-Time-Resolved")
        self.rb_non_tr.setChecked(True)
        self._acq_btn_group = QButtonGroup()
        self._acq_btn_group.addButton(self.rb_time_resolved)
        self._acq_btn_group.addButton(self.rb_non_tr)
        self.rb_time_resolved.toggled.connect(self._on_acq_changed)
        acq_mode_row.addWidget(self.rb_time_resolved)
        acq_mode_row.addWidget(self.rb_non_tr)
        acq_lay.addLayout(acq_mode_row)

        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Sampling freq fs [Hz]:"))
        self.spin_fs_main = QDoubleSpinBox()
        self.spin_fs_main.setRange(1.0, 1e6)
        self.spin_fs_main.setValue(1000.0)
        self.spin_fs_main.setDecimals(1)
        self.spin_fs_main.setSingleStep(100.0)
        self.spin_fs_main.setEnabled(False)
        fs_row.addWidget(self.spin_fs_main)
        acq_lay.addLayout(fs_row)

        ll.addWidget(self.acq_group)

        # Display group
        self.display_group = QGroupBox("3. Display")
        self.display_group.setVisible(False)
        dl = QVBoxLayout(self.display_group)

        dl.addWidget(QLabel("Show field:"))
        self.combo_field = QComboBox()
        self.combo_field.currentIndexChanged.connect(self._on_field_changed)
        dl.addWidget(self.combo_field)

        dl.addWidget(QLabel("Plot type:"))
        self.combo_plot = QComboBox()
        self.combo_plot.addItems(["Contourf + Vectors", "Contourf only", "Vectors only"])
        self.combo_plot.currentIndexChanged.connect(self._on_field_changed)
        dl.addWidget(self.combo_plot)

        # Vector controls (shown inside display group)
        self.vec_group = QGroupBox("Vector Options")
        vl = QVBoxLayout(self.vec_group)

        # Skip x
        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("Skip x:"))
        self.spin_skip_x = QSpinBox()
        self.spin_skip_x.setRange(1, 50)
        self.spin_skip_x.setValue(10)
        self.spin_skip_x.setToolTip("Plot every Nth vector in x direction")
        row_x.addWidget(self.spin_skip_x)
        vl.addLayout(row_x)

        # Skip y
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Skip y:"))
        self.spin_skip_y = QSpinBox()
        self.spin_skip_y.setRange(1, 50)
        self.spin_skip_y.setValue(10)
        self.spin_skip_y.setToolTip("Plot every Nth vector in y direction")
        row_y.addWidget(self.spin_skip_y)
        vl.addLayout(row_y)

        # Vector scale
        row_s = QHBoxLayout()
        row_s.addWidget(QLabel("Length:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.01, 10.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setSingleStep(0.01)
        self.spin_scale.setDecimals(2)
        self.spin_scale.setToolTip("Vector length: higher = longer arrows")
        row_s.addWidget(self.spin_scale)
        vl.addLayout(row_s)

        # Apply button
        self.btn_plot = QPushButton("Apply")
        self.btn_plot.clicked.connect(self._on_field_changed)
        vl.addWidget(self.btn_plot)

        dl.addWidget(self.vec_group)
        ll.addWidget(self.display_group)

        # Analysis group
        self.analysis_group = QGroupBox("4. Analysis")
        self.analysis_group.setVisible(False)
        al = QVBoxLayout(self.analysis_group)
        al.addWidget(QLabel("Select analysis:"))
        self.combo_analysis = QComboBox()
        self.combo_analysis.addItems([
            "-- Select --",
            "Reynolds Stresses",
            "TKE Budget",
            "Spectral (Temporal)",
            "Spectral (Spatial)",
            "POD Modes",
            "Anisotropy Invariants",
            "FTLE",
        ])
        al.addWidget(self.combo_analysis)
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self._on_run_analysis)
        al.addWidget(self.btn_run)
        ll.addWidget(self.analysis_group)

        ll.addStretch()

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: gray; font-size: 10px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        # About button
        self.btn_about = QPushButton("About uPrime")
        self.btn_about.setStyleSheet("font-size: 10px; color: gray;")
        self.btn_about.clicked.connect(self._on_about)
        ll.addWidget(self.btn_about)

        # Developer credit at bottom
        credit = QLabel("Developed by Jibu Tom Jose")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        ll.addWidget(credit)

        # ---- Right canvas ----
        self.plot_canvas = PlotCanvas()

        splitter.addWidget(left)
        splitter.addWidget(self.plot_canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _on_about(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        from PyQt6.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("About uPrime")
        dlg.setFixedWidth(420)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)
        lay.setContentsMargins(24, 24, 24, 24)

        name_lbl = QLabel("uPrime")
        name_lbl.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(name_lbl)

        tagline = QLabel("Because u’ matters")
        tagline.setFont(QFont("Arial", 10))
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet("color: gray; font-style: italic;")
        lay.addWidget(tagline)

        version = QLabel("v0.1  ·  Alpha Release")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: gray; font-size: 10px;")
        lay.addWidget(version)

        sep = QLabel("―" * 40)
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("color: #444;")
        lay.addWidget(sep)

        desc = QLabel(
            "Open-source fluid velocity field analysis.\n"
            "Supports Tecplot .dat format from DaVis and CFD solvers.\n"
            "Works with PIV, stereo PIV, and CFD velocity data."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 10px;")
        lay.addWidget(desc)

        sep2 = QLabel("―" * 40)
        sep2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep2.setStyleSheet("color: #444;")
        lay.addWidget(sep2)

        credit = QLabel(
            "Developed by <b>Jibu Tom Jose</b><br>"
            "Transient Fluid Mechanics Lab, Technion<br>"
            "Built with assistance from Claude (Anthropic)"
        )
        credit.setWordWrap(True)
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit.setStyleSheet("font-size: 10px;")
        lay.addWidget(credit)

        gh = QLabel(
            "GitHub: <i>github.com/CmdrRyder/uPrime</i><br><br>"
            "Licensed under CC BY-NC-ND 4.0<br>"
            "Free for personal and academic use with attribution.<br>"
            "If used in published research, please cite:<br>"
            "<i>Jibu Tom Jose, uPrime, Technion (2026)</i>"
        )
        gh.setWordWrap(True)
        gh.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gh.setStyleSheet("color: gray; font-size: 9px;")
        lay.addWidget(gh)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        dlg.exec()

    def _on_load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select DaVis .dat files", "",
            "DaVis Data Files (*.dat);;All Files (*)"
        )
        if not paths:
            return

        self.lbl_files.setText(f"{len(paths)} file(s) selected\n{os.path.basename(paths[0])} ...")
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
        mem_mb = (3 if stereo else 2) * ny * nx * Nt * 4 / 1e6

        self.lbl_info.setText(
            f"Grid : {nx} x {ny} points\n"
            f"Snapshots : {Nt}\n"
            f"Type : {'Stereo (U,V,W)' if stereo else '2D (U,V)'}\n"
            f"Memory : ~{mem_mb:.0f} MB"
        )
        self.info_group.setVisible(True)

        self.combo_field.clear()
        self.combo_field.addItems(["Mean |V| (speed)", "Mean U", "Mean V"])
        if stereo:
            self.combo_field.addItem("Mean W")
        self.combo_field.addItems(["Std U", "Std V"])
        if dataset.get("has_vort", False):
            self.combo_field.addItem("Mean Vorticity")

        # Set sensible default skip based on grid size
        default_skip = max(1, min(nx, ny) // 20)
        self.spin_skip_x.setValue(default_skip)
        self.spin_skip_y.setValue(default_skip)

        self.acq_group.setVisible(True)
        self.display_group.setVisible(True)
        self.analysis_group.setVisible(True)
        self.lbl_status.setText(f"Loaded {Nt} snapshots successfully.")
        self._plot_field()
        self._check_convergence_warning(Nt)

    def _on_load_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.lbl_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Load Error", f"Failed to load files:\n{msg}")

    def _on_field_changed(self):
        if self.dataset is not None:
            self._plot_field()

    def _plot_field(self):
        if self.dataset is None:
            return

        ds         = self.dataset
        x          = ds["x"]
        y          = ds["y"]
        U          = ds["U"]
        V          = ds["V"]
        W          = ds["W"]
        field_name = self.combo_field.currentText()
        plot_type  = self.combo_plot.currentText()
        skip_x     = self.spin_skip_x.value()
        skip_y     = self.spin_skip_y.value()
        vec_scale  = self.spin_scale.value()

        mean_u = np.nanmean(U, axis=2)
        mean_v = np.nanmean(V, axis=2)

        # Valid mask: points with data in at least 50% of frames
        valid_frac   = np.mean(ds["valid"], axis=2)
        invalid_mask = valid_frac < 0.5

        if field_name == "Mean U":
            field = mean_u.copy()
            field[invalid_mask] = np.nan
            cbar_label = "Mean U [m/s]"
        elif field_name == "Mean V":
            field = mean_v.copy()
            field[invalid_mask] = np.nan
            cbar_label = "Mean V [m/s]"
        elif field_name == "Mean W" and W is not None:
            field = np.nanmean(W, axis=2)
            field[invalid_mask] = np.nan
            cbar_label = "Mean W [m/s]"
        elif field_name == "Std U":
            field = np.nanstd(U, axis=2)
            field[invalid_mask] = np.nan
            cbar_label = "Std(U) [m/s]"
        elif field_name == "Std V":
            field = np.nanstd(V, axis=2)
            field[invalid_mask] = np.nan
            cbar_label = "Std(V) [m/s]"
        elif field_name == "Mean Vorticity" and ds.get("vort") is not None:
            field = np.nanmean(ds["vort"], axis=2)
            field[invalid_mask] = np.nan
            cbar_label = "Vorticity [1/s]"
        else:
            if W is not None:
                mw = np.nanmean(W, axis=2)
                field = np.sqrt(mean_u**2 + mean_v**2 + mw**2)
            else:
                field = np.sqrt(mean_u**2 + mean_v**2)
            field[invalid_mask] = np.nan
            cbar_label = "Mean |V| [m/s]"

        ax = self.plot_canvas.get_axes()

        if "Contourf" in plot_type:
            cf = ax.contourf(x, y, field, levels=50, cmap="RdBu_r")
            self.plot_canvas.figure.colorbar(cf, ax=ax, label=cbar_label, shrink=0.8)

        if "Vectors" in plot_type:
            # Subsample with independent x and y skip
            xs  = x[::skip_y, ::skip_x]
            ys  = y[::skip_y, ::skip_x]
            us  = mean_u[::skip_y, ::skip_x].copy()
            vs  = mean_v[::skip_y, ::skip_x].copy()
            inv = invalid_mask[::skip_y, ::skip_x]

            # Mask invalid points
            us[inv] = np.nan
            vs[inv] = np.nan

            # Normalize to unit vectors (direction only)
            # This ensures arrows show correct angle regardless of U/V magnitude ratio
            mag = np.sqrt(us**2 + vs**2)
            mag[mag == 0] = np.nan
            un = us / mag
            vn = vs / mag

            ax.quiver(
                xs, ys, un, vn,
                color="k",
                scale=50.0 / max(vec_scale, 0.001),
                alpha=0.7,
                width=0.002,
                headwidth=3,
                headlength=4
            )

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(field_name)
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        self.plot_canvas.canvas.draw()
        self.lbl_status.setText(f"Displaying: {field_name}")

    def _on_acq_changed(self):
        self.spin_fs_main.setEnabled(self.rb_time_resolved.isChecked())

    def is_time_resolved(self):
        return self.rb_time_resolved.isChecked()

    def get_fs(self):
        return self.spin_fs_main.value()

    def _check_convergence_warning(self, Nt):
        """Warn user if dataset may not be statistically converged."""
        if self.rb_time_resolved.isChecked():
            fs = self.spin_fs_main.value()
            duration = Nt / fs
            if duration < 2.0:
                QMessageBox.warning(
                    self, "Convergence Warning",
                    f"Loaded {Nt} snapshots at fs={fs:.0f} Hz "
                    f"= {duration:.2f} seconds of data.\n\n"
                    "This is less than 2 seconds. Statistical results "
                    "(means, spectra, Reynolds stresses) may not be converged "
                    "and should be interpreted with caution."
                )
        else:
            if Nt < 2000:
                QMessageBox.warning(
                    self, "Convergence Warning",
                    f"Loaded {Nt} snapshots (non-time-resolved).\n\n"
                    "For reliable statistics, at least 2000 uncorrelated "
                    "snapshots are recommended. Results may not be converged."
                )

    def _on_run_analysis(self):
        choice = self.combo_analysis.currentText()
        if choice == "-- Select --":
            QMessageBox.information(self, "Select Analysis", "Please select an analysis from the list.")
            return

        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        # Analyses that require time-resolved data
        time_resolved_only = ["Spectral (Temporal)", "FTLE"]
        if choice in time_resolved_only and not self.is_time_resolved():
            QMessageBox.warning(
                self, "Time-Resolved Data Required",
                f"'{choice}' requires time-resolved PIV data.\n\n"
                "Please select 'Time-Resolved' in the Acquisition Type panel "
                "and enter the correct sampling frequency."
            )
            return

        if choice == "TKE Budget":
            self._windows = getattr(self, "_windows", [])
            Nt  = self.dataset["Nt"]
            fs  = self.get_fs()
            dur = Nt / fs if self.is_time_resolved() else 9999
            win = TKEWindow(
                self.dataset,
                is_time_resolved=self.is_time_resolved(),
                Nt_warn=Nt,
                duration_warn=dur,
            )
            win.show()
            self._windows.append(win)

        elif choice == "Reynolds Stresses":
            self._windows = getattr(self, "_windows", [])
            Nt  = self.dataset["Nt"]
            fs  = self.get_fs()
            dur = Nt / fs if self.is_time_resolved() else None
            win = ReynoldsWindow(
                self.dataset,
                is_time_resolved=self.is_time_resolved(),
                Nt_warn=Nt,
                duration_warn=dur if dur is not None else 9999,
            )
            win.show()
            self._windows.append(win)

        elif choice == "Anisotropy Invariants":
            if not self.dataset["is_stereo"]:
                QMessageBox.warning(self, "Stereo Required",
                    "Anisotropy analysis requires stereo PIV data (u, v, w).\n"
                    "The loaded dataset is 2D only.")
                return
            self._windows = getattr(self, "_windows", [])
            win = AnisotropyWindow(self.dataset)
            win.show()
            self._windows.append(win)

        elif choice == "Spectral (Temporal)":
            # Pass fs from main window into spectral window
            self._windows = getattr(self, "_windows", [])
            win = SpectralWindow(self.dataset, default_fs=self.get_fs())
            win.show()
            self._windows.append(win)

        else:
            QMessageBox.information(
                self, "Coming Soon",
                f"'{choice}' analysis panel is being built.\n\n"
                "Use the display options to explore your data for now."
            )