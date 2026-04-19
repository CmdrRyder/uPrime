"""
gui/main_window.py
uPrime - Main Application Window v0.2
"""

import os
import sys
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QProgressBar, QSplitter, QGroupBox, QSizePolicy,
    QMessageBox, QSpinBox, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QScrollArea, QStatusBar, QFrame,
    QDialog, QDialogButtonBox, QCheckBox, QColorDialog,
    QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon, QShortcut, QKeySequence


def _asset_path(filename):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "assets", filename)


def _is_dark_mode():
    palette = QApplication.instance().palette()
    return palette.window().color().lightness() < 128


def _load_logo_pixmap(size=None):
    name = "logo_dark.png" if _is_dark_mode() else "logo.png"
    path = _asset_path(name)
    pm = QPixmap(path)
    if size is not None:
        pm = pm.scaled(size[0], size[1],
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
    return pm

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from core.loader import load_dataset, cleanup_memmap, SIZE_THRESHOLD
from core.transform import transform_status_string
from gui.anisotropy_window import AnisotropyWindow
from gui.reynolds_window import ReynoldsWindow
from gui.tke_window import TKEWindow
from gui.tke_budget_window import TKEBudgetWindow
from gui.spectra_window import SpectraWindow
from gui.correlation_window import CorrelationWindow
from gui.pod_window import PODWindow
from gui.transform_window import TransformWindow
from gui.mask_window import MaskWindow
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
        self.setWindowTitle("uPrime v0.4.1")
        self.setWindowIcon(QIcon(_load_logo_pixmap(size=(256, 256))))
        self.setMinimumSize(1100, 650)
        self.resize(1400, 900)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - 1400) // 2, (screen.height() - 900) // 2)
        self.dataset           = None
        self.loader_thread     = None
        self._windows          = []
        self._dmd_win          = None
        self._full_file_list   = []   # complete original file list from dialog
        self._last_file_list   = []   # actually loaded file list (after subsampling)
        self._subsample_desc   = ""   # description of current subsampling
        self._rakes            = []     # list of rake dicts {p0,p1,n_seeds,lw,color}
        self._rake_artist      = None  # temporary line artist during drag
        self._rake_press_xy    = None  # mouse-down position during rake drag
        self._sl_color         = "#ffffff"  # streamline color
        self._full_dataset     = None # copy before subset is applied
        self._original_fs      = None # fs before any stride-based subset
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
        logo_lbl = QLabel()
        logo_lbl.setPixmap(_load_logo_pixmap(size=(120, 120)))
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(logo_lbl)

        sl.addWidget(self._separator())

        # -- 1. Load Data --
        load_grp = QGroupBox("1. Load Data")
        load_lay = QVBoxLayout(load_grp)
        self.btn_load = QPushButton("📂  Select .dat Files...")
        self.btn_load.clicked.connect(self._on_load_files)
        load_lay.addWidget(self.btn_load)
        self.btn_reload = QPushButton("\u21ba  Reload Last Dataset")
        self.btn_reload.clicked.connect(self._on_reload_files)
        self.btn_reload.setStyleSheet(
            "QPushButton { background: #1e2e3e; color: #90b8d8; }"
            "QPushButton:hover { background: #253545; }"
            "QPushButton:pressed { background: #162030; }")
        self.btn_reload.setVisible(False)
        load_lay.addWidget(self.btn_reload)
        self.btn_subset = QPushButton("\u2702  Select Subset...")
        self.btn_subset.clicked.connect(self._on_select_subset)
        self.btn_subset.setStyleSheet(
            "QPushButton { background: #1e3e2e; color: #90d8b8; }"
            "QPushButton:hover { background: #253545; }"
            "QPushButton:pressed { background: #162030; }")
        self.btn_subset.setVisible(False)
        load_lay.addWidget(self.btn_subset)
        self.btn_restore_full = QPushButton("\u21a9  Restore Full Dataset")
        self.btn_restore_full.clicked.connect(self._on_restore_full_dataset)
        self.btn_restore_full.setStyleSheet(
            "QPushButton { background: #3e2e1e; color: #d8b890; }"
            "QPushButton:hover { background: #453525; }"
            "QPushButton:pressed { background: #302010; }")
        self.btn_restore_full.setVisible(False)
        load_lay.addWidget(self.btn_restore_full)
        self.lbl_files = QLabel("No files loaded")
        self.lbl_files.setWordWrap(True)
        self.lbl_files.setStyleSheet("color:#888;font-size:10px;")
        load_lay.addWidget(self.lbl_files)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        load_lay.addWidget(self.progress_bar)
        sl.addWidget(load_grp)

        sl.addWidget(self._separator())

        # -- 2. Preprocess --
        self.transform_group = QGroupBox("2. Preprocess")
        self.transform_group.setVisible(False)
        tr_lay = QVBoxLayout(self.transform_group)
        self.btn_transform = QPushButton("\u21ba  Transform / Align...")
        self.btn_transform.clicked.connect(self._run_transform)
        self.btn_transform.setStyleSheet("text-align: left; padding: 4px 8px;")
        tr_lay.addWidget(self.btn_transform)
        self.btn_masking = QPushButton("\u2b21  Mask Editor...")
        self.btn_masking.clicked.connect(self._run_masking)
        self.btn_masking.setStyleSheet("text-align: left; padding: 4px 8px;")
        tr_lay.addWidget(self.btn_masking)
        sl.addWidget(self.transform_group)

        sl.addWidget(self._separator())

        # -- 3. Analysis Buttons --
        self.analysis_group = QGroupBox("3. Analysis")
        self.analysis_group.setVisible(False)
        an_lay = QVBoxLayout(self.analysis_group)

        analyses = [
            ("📊  Reynolds Stresses",      self._run_reynolds),
            ("⚡  TKE Budget",             self._run_tke_budget),
            ("〜  Space-Time Spectra",     self._run_spectra),
            ("△  Anisotropy Invariants",  self._run_anisotropy),
            ("⟺  Correlation Analysis",   self._run_correlation),
            ("◈  POD Analysis",            self._run_pod),
            ("◈  DMD Analysis",            self._run_dmd),
            ("⦾  Vortex Identification",  self._run_vortex),
        ]
        self._analysis_btns = []
        self.btn_dmd = None
        for label, slot in analyses:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn.setStyleSheet("text-align: left; padding: 4px 8px;")
            an_lay.addWidget(btn)
            self._analysis_btns.append(btn)
            if slot is self._run_dmd:
                self.btn_dmd = btn


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

        # -- Overlay radio buttons --
        opts.addWidget(QLabel("Overlay:"))
        self.rb_overlay_none = QRadioButton("None")
        self.rb_overlay_vec  = QRadioButton("Vectors")
        self.rb_overlay_sl   = QRadioButton("Streamlines")
        self.rb_overlay_vec.setChecked(True)
        self._overlay_bg = QButtonGroup()
        for rb in [self.rb_overlay_none, self.rb_overlay_vec, self.rb_overlay_sl]:
            self._overlay_bg.addButton(rb)
            opts.addWidget(rb)
            rb.toggled.connect(self._on_overlay_mode_changed)

        opts.addWidget(self._vsep())

        self.chk_draw_on_contour = QCheckBox("Draw on contour")
        self.chk_draw_on_contour.setChecked(True)
        self.chk_draw_on_contour.stateChanged.connect(self._on_field_changed)
        opts.addWidget(self.chk_draw_on_contour)

        self.chk_clean_export = QCheckBox("Clean export (hide axes)")
        self.chk_clean_export.setChecked(False)
        self.chk_clean_export.stateChanged.connect(self._on_field_changed)
        opts.addWidget(self.chk_clean_export)
        self.chk_hide_colorbar = QCheckBox("Hide colorbar")
        self.chk_hide_colorbar.setChecked(False)
        self.chk_hide_colorbar.stateChanged.connect(self._on_field_changed)
        opts.addWidget(self.chk_hide_colorbar)
        opts.addStretch()
        btn_help = QPushButton("? Manual")
        btn_help.setFixedHeight(24)
        btn_help.setFlat(True)
        btn_help.setToolTip("Open User Manual (F1)")
        btn_help.setStyleSheet("font-weight: bold; font-size: 12px; padding: 0px 6px;")
        btn_help.clicked.connect(self._open_manual)
        opts.addWidget(btn_help)

        right_lay.addWidget(self.options_strip)

        # -- Second ribbon: vector / streamline controls (hidden when None) --
        self.overlay_ribbon = QWidget()
        self.overlay_ribbon.setVisible(False)
        orb = QHBoxLayout(self.overlay_ribbon)
        orb.setContentsMargins(4, 2, 4, 2)
        orb.setSpacing(8)

        # Vector controls
        self._vec_controls = QWidget()
        vc = QHBoxLayout(self._vec_controls)
        vc.setContentsMargins(0, 0, 0, 0); vc.setSpacing(6)
        vc.addWidget(QLabel("Skip x:"))
        self.spin_skip_x = QSpinBox()
        self.spin_skip_x.setRange(1, 50); self.spin_skip_x.setValue(5)
        self.spin_skip_x.setFixedWidth(55)
        vc.addWidget(self.spin_skip_x)
        vc.addWidget(QLabel("Skip y:"))
        self.spin_skip_y = QSpinBox()
        self.spin_skip_y.setRange(1, 50); self.spin_skip_y.setValue(5)
        self.spin_skip_y.setFixedWidth(55)
        vc.addWidget(self.spin_skip_y)
        vc.addWidget(QLabel("Length:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.01, 10.0); self.spin_scale.setValue(1.0)
        self.spin_scale.setDecimals(1); self.spin_scale.setSingleStep(0.1)
        self.spin_scale.setFixedWidth(60)
        vc.addWidget(self.spin_scale)
        vc.addWidget(QLabel("Arrow size:"))
        self.spin_arrow_size = QDoubleSpinBox()
        self.spin_arrow_size.setRange(0.1, 5.0); self.spin_arrow_size.setValue(1.0)
        self.spin_arrow_size.setDecimals(1); self.spin_arrow_size.setSingleStep(0.1)
        self.spin_arrow_size.setFixedWidth(60)
        vc.addWidget(self.spin_arrow_size)
        btn_apply_vec = QPushButton("Apply")
        btn_apply_vec.setFixedWidth(60)
        btn_apply_vec.clicked.connect(self._on_field_changed)
        vc.addWidget(btn_apply_vec)
        orb.addWidget(self._vec_controls)

        # Streamline controls
        self._sl_controls = QWidget()
        sc = QHBoxLayout(self._sl_controls)
        sc.setContentsMargins(0, 0, 0, 0); sc.setSpacing(6)
        self.btn_draw_rake = QPushButton("Draw Rake")
        self.btn_draw_rake.setFixedWidth(85)
        self.btn_draw_rake.setCheckable(True)
        self.btn_draw_rake.clicked.connect(self._on_draw_rake_toggle)
        sc.addWidget(self.btn_draw_rake)
        sc.addWidget(QLabel("Seeds:"))
        self.spin_sl_seeds = QSpinBox()
        self.spin_sl_seeds.setRange(2, 100); self.spin_sl_seeds.setValue(20)
        self.spin_sl_seeds.setFixedWidth(50)
        sc.addWidget(self.spin_sl_seeds)
        sc.addWidget(QLabel("Line width:"))
        self.spin_sl_lw = QDoubleSpinBox()
        self.spin_sl_lw.setRange(0.1, 5.0); self.spin_sl_lw.setValue(1.0)
        self.spin_sl_lw.setDecimals(1); self.spin_sl_lw.setSingleStep(0.1)
        self.spin_sl_lw.setFixedWidth(55)
        sc.addWidget(self.spin_sl_lw)
        sc.addWidget(QLabel("Color:"))
        self.btn_sl_color = QPushButton()
        self.btn_sl_color.setFixedSize(24, 24)
        self.btn_sl_color.setStyleSheet(
            f"background: {self._sl_color}; border: 1px solid #888;")
        self.btn_sl_color.clicked.connect(self._on_sl_color_pick)
        sc.addWidget(self.btn_sl_color)
        self.btn_sl_reset = QPushButton("Reset")
        self.btn_sl_reset.setFixedWidth(55)
        self.btn_sl_reset.setToolTip("Clear all drawn rakes")
        self.btn_sl_reset.clicked.connect(self._on_sl_reset)
        sc.addWidget(self.btn_sl_reset)
        btn_apply_sl = QPushButton("Apply")
        btn_apply_sl.setFixedWidth(60)
        btn_apply_sl.clicked.connect(self._on_field_changed)
        sc.addWidget(btn_apply_sl)
        self._sl_controls.setVisible(False)
        orb.addWidget(self._sl_controls)

        orb.addStretch()
        right_lay.addWidget(self.overlay_ribbon)

        # -- Info ribbon (compact dataset summary, shown after load) --
        self.info_ribbon = QWidget()
        self.info_ribbon.setVisible(False)
        ribbon_lay = QHBoxLayout(self.info_ribbon)
        ribbon_lay.setContentsMargins(4, 2, 4, 2)
        ribbon_lay.setSpacing(6)

        self.lbl_info_ribbon = QLabel("")
        self.lbl_info_ribbon.setStyleSheet("font-size:13px; color:#aaa;")
        ribbon_lay.addWidget(self.lbl_info_ribbon)

        vsep = QFrame()
        vsep.setFrameShape(QFrame.Shape.VLine)
        vsep.setStyleSheet("color:#555;")
        ribbon_lay.addWidget(vsep)

        self.radio_tr    = QRadioButton("TR")
        self.radio_nontr = QRadioButton("Non-TR")
        self.radio_nontr.setChecked(True)
        self._acq_bg = QButtonGroup()
        self._acq_bg.addButton(self.radio_tr)
        self._acq_bg.addButton(self.radio_nontr)
        self.radio_tr.toggled.connect(self._on_tr_changed)
        self.radio_nontr.toggled.connect(self._on_tr_changed)
        self.radio_nontr.toggled.connect(self._update_dmd_btn_state)
        ribbon_lay.addWidget(self.radio_tr)
        ribbon_lay.addWidget(self.radio_nontr)

        ribbon_lay.addWidget(QLabel("fs [Hz]:"))
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(0.1, 1_000_000)
        self.spin_fs.setValue(1000.0)
        self.spin_fs.setDecimals(1)
        self.spin_fs.setFixedWidth(80)
        self.spin_fs.valueChanged.connect(self._on_fs_changed)
        self.radio_tr.toggled.connect(lambda checked: self.spin_fs.setEnabled(checked))
        self.spin_fs.setEnabled(self.radio_tr.isChecked())
        ribbon_lay.addWidget(self.spin_fs)

        ribbon_lay.addStretch()
        right_lay.addWidget(self.info_ribbon)

        # -- Transform status strip (hidden until a transform is applied) --
        self.transform_strip = QWidget()
        self.transform_strip.setVisible(False)
        self.transform_strip.setStyleSheet(
            "background:#0e0e1a; border-left:3px solid #e06c75;"
            "border-radius:2px; padding:2px;")
        ts_lay = QHBoxLayout(self.transform_strip)
        ts_lay.setContentsMargins(8, 3, 8, 3)
        ts_icon = QLabel("\u25cf")   # filled circle -- subtle alert dot
        ts_icon.setStyleSheet("color:#e06c75; font-size:10px;")
        ts_lay.addWidget(ts_icon)
        self.lbl_transform_status = QLabel("Data transformed:")
        self.lbl_transform_status.setStyleSheet(
            "color:#e06c75; font-size:13px; font-weight:bold;"
            "letter-spacing:0.3px;")
        ts_lay.addWidget(self.lbl_transform_status)
        ts_lay.addStretch()
        right_lay.addWidget(self.transform_strip)

        # -- Plot canvas --
        # Welcome manual bar (only visible on the welcome screen)
        self._welcome_manual_bar = QWidget()
        self._welcome_manual_bar.setVisible(False)
        wm_lay = QHBoxLayout(self._welcome_manual_bar)
        wm_lay.setContentsMargins(0, 4, 0, 0)
        btn_manual_welcome = QPushButton("? Manual")
        btn_manual_welcome.setFlat(True)
        btn_manual_welcome.setStyleSheet("font-size: 12px; color: #888888;")
        btn_manual_welcome.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_manual_welcome.clicked.connect(self._open_manual)
        wm_lay.addStretch()
        wm_lay.addWidget(btn_manual_welcome)
        wm_lay.addStretch()
        right_lay.addWidget(self._welcome_manual_bar)

        self.plot_canvas = PlotCanvas()
        right_lay.addWidget(self.plot_canvas)
        self._override_home_button()

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

        QShortcut(QKeySequence("F1"), self).activated.connect(self._open_manual)

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
        self._welcome_manual_bar.setVisible(True)

    # ----------------------------------------------------------------------- #
    # Load files
    # ----------------------------------------------------------------------- #

    def _show_subsample_dialog(self, n_total):
        """
        Show a dialog asking the user how many snapshots to load.

        Returns (indices, description) if OK, or (None, None) if cancelled.
        """
        # Use dataset dimensions if available, otherwise sensible defaults
        if self.dataset is not None:
            ny = self.dataset["ny"]
            nx = self.dataset["nx"]
            is_stereo = self.dataset.get("is_stereo", False)
        else:
            ny, nx, is_stereo = 253, 500, False

        n_vel_components = 3 if is_stereo else 2

        dlg = QDialog(self)
        dlg.setWindowTitle("Subsample Dataset")
        dlg.setFixedWidth(420)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        lay.addWidget(QLabel(
            f"{n_total} files available. Choose how many snapshots to load:"))

        # --- Radio buttons ---
        bg = QButtonGroup(dlg)

        rb_all    = QRadioButton(f"Load all {n_total} snapshots")
        rb_stride = QRadioButton("Stride: load every K-th snapshot")
        rb_limit  = QRadioButton("Limit: load first N snapshots")
        rb_all.setChecked(True)

        bg.addButton(rb_all,    0)
        bg.addButton(rb_stride, 1)
        bg.addButton(rb_limit,  2)

        lay.addWidget(rb_all)

        stride_row = QHBoxLayout()
        stride_row.addWidget(rb_stride)
        spin_stride = QSpinBox()
        spin_stride.setRange(2, 50)
        spin_stride.setValue(max(2, round(n_total / 2000)))
        spin_stride.setFixedWidth(70)
        spin_stride.setEnabled(False)
        stride_row.addWidget(spin_stride)
        stride_row.addStretch()
        lay.addLayout(stride_row)

        limit_row = QHBoxLayout()
        limit_row.addWidget(rb_limit)
        spin_limit = QSpinBox()
        spin_limit.setRange(10, 9999)
        spin_limit.setValue(min(n_total, 2000))
        spin_limit.setFixedWidth(80)
        spin_limit.setEnabled(False)
        limit_row.addWidget(spin_limit)
        limit_row.addStretch()
        lay.addLayout(limit_row)

        # --- Live-updating info labels ---
        lbl_count  = QLabel()
        lbl_memory = QLabel()
        lbl_count.setStyleSheet("color: #aaa; font-size: 11px;")
        lbl_memory.setStyleSheet("color: #aaa; font-size: 11px;")
        lay.addWidget(lbl_count)
        lay.addWidget(lbl_memory)

        lbl_memmap_warn = QLabel()
        lbl_memmap_warn.setWordWrap(True)
        lbl_memmap_warn.setStyleSheet("color: #E8A020; font-size: 10px;")
        lbl_memmap_warn.setVisible(False)
        lay.addWidget(lbl_memmap_warn)

        def _update():
            import tempfile
            if rb_all.isChecked():
                n = n_total
            elif rb_stride.isChecked():
                n = len(range(0, n_total, spin_stride.value()))
            else:
                n = min(spin_limit.value(), n_total)
            est_bytes = ny * nx * n_vel_components * n * 4
            mem_gb    = est_bytes / 1024 ** 3
            lbl_count.setText(f"Snapshots to load:  {n}")
            lbl_memory.setText(f"Estimated memory:  ~{mem_gb * 1024:.0f} MB")
            spin_stride.setEnabled(rb_stride.isChecked())
            spin_limit.setEnabled(rb_limit.isChecked())
            if est_bytes > SIZE_THRESHOLD:
                tmp = tempfile.gettempdir()
                lbl_memmap_warn.setText(
                    f"\u26a0 Dataset exceeds 4 GB. Velocity data will be cached "
                    f"to a temporary file in:\n{tmp}\n"
                    f"Approx. {mem_gb:.1f} GB of free disk space required. "
                    f"The file is deleted automatically when uPrime closes. "
                    f"If uPrime crashes, delete uprime_memmap_*.bin from "
                    f"that folder manually.")
                lbl_memmap_warn.setVisible(True)
            else:
                lbl_memmap_warn.setVisible(False)

        rb_all.toggled.connect(_update)
        rb_stride.toggled.connect(_update)
        rb_limit.toggled.connect(_update)
        spin_stride.valueChanged.connect(_update)
        spin_limit.valueChanged.connect(_update)
        _update()

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None, None

        if rb_all.isChecked():
            indices = list(range(n_total))
            desc    = "All snapshots"
        elif rb_stride.isChecked():
            K       = spin_stride.value()
            indices = list(range(0, n_total, K))
            desc    = f"Every {K}th snapshot ({len(indices)} of {n_total})"
        else:
            N       = min(spin_limit.value(), n_total)
            indices = list(range(N))
            desc    = f"First {N} of {n_total} snapshots"

        return indices, desc

    def _on_load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select DaVis .dat files", "",
            "DaVis Data Files (*.dat);;All Files (*)"
        )
        if not paths:
            return

        self._full_file_list = list(paths)

        total_size = sum(os.path.getsize(p) for p in paths)
        if total_size > 500 * 1024 * 1024:
            indices, desc = self._show_subsample_dialog(len(paths))
            if indices is None:
                return
            paths_to_load = [paths[i] for i in indices]
            self._subsample_desc = desc
        else:
            paths_to_load = paths
            self._subsample_desc = "All snapshots"

        self._last_file_list = list(paths_to_load)
        self._start_load(paths_to_load)

    def _on_reload_files(self):
        if not self._last_file_list:
            return
        n = len(self._last_file_list)
        self.lbl_status.setText(f"Reloading {n} files from last session...")
        self._start_load(self._last_file_list)

    def _start_load(self, file_list):
        if self.dataset:
            cleanup_memmap(self.dataset)
        self.lbl_files.setText(
            f"{len(file_list)} file(s)\n{os.path.basename(file_list[0])} ...")
        self.lbl_status.setText("Loading files...")
        self.btn_load.setEnabled(False)
        self.btn_reload.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.loader_thread = LoaderThread(file_list)
        self.loader_thread.progress.connect(self.progress_bar.setValue)
        self.loader_thread.finished.connect(self._on_load_finished)
        self.loader_thread.error.connect(self._on_load_error)
        self.loader_thread.start()

    def _on_load_finished(self, dataset):
        self.dataset = dataset
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self._last_file_list = list(self.loader_thread.file_list)
        self._full_dataset = None
        self._original_fs  = None
        self.btn_reload.setVisible(True)
        self.btn_reload.setEnabled(True)
        self.btn_subset.setVisible(True)
        self.btn_restore_full.setVisible(False)

        Nt     = dataset["Nt"]
        nx     = dataset["nx"]
        ny     = dataset["ny"]
        stereo = dataset["is_stereo"]
        x      = dataset["x"]
        y      = dataset["y"]

        self._update_ribbon()
        self.info_ribbon.setVisible(True)
        self.analysis_group.setVisible(True)
        self.transform_group.setVisible(True)
        self.options_strip.setVisible(True)
        self._welcome_manual_bar.setVisible(False)
        self._on_overlay_mode_changed()

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

        self.spin_skip_x.setValue(5)
        self.spin_skip_y.setValue(5)

        # --- Acquisition type popup ---
        acq_dlg = QDialog(self)
        acq_dlg.setWindowTitle("Acquisition Type")
        acq_dlg.setFixedWidth(280)
        vl = QVBoxLayout(acq_dlg)
        vl.addWidget(QLabel("Select acquisition type for this dataset:"))
        bg = QButtonGroup(acq_dlg)
        rb_tr  = QRadioButton("Time-Resolved (TR)")
        rb_ntr = QRadioButton("Non-TR")
        rb_ntr.setChecked(True)
        bg.addButton(rb_tr)
        bg.addButton(rb_ntr)
        vl.addWidget(rb_tr)
        vl.addWidget(rb_ntr)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("fs [Hz]:"))
        sp = QDoubleSpinBox()
        sp.setRange(0.1, 1_000_000)
        sp.setValue(self.spin_fs.value())
        sp.setDecimals(1)
        sp.setEnabled(rb_tr.isChecked())
        rb_tr.toggled.connect(lambda checked: sp.setEnabled(checked))
        hl.addWidget(sp)
        vl.addLayout(hl)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(acq_dlg.accept)
        vl.addWidget(btn_ok)
        if acq_dlg.exec() == QDialog.DialogCode.Accepted:
            self.radio_tr.setChecked(rb_tr.isChecked())
            self.radio_nontr.setChecked(rb_ntr.isChecked())
            self.spin_fs.setValue(sp.value())
            self._on_tr_changed()

        self._check_convergence(Nt)
        self.lbl_status.setText(f"Loaded {Nt} snapshots.")
        self._plot_field()

        # Setup picker on main plot
        if self.plot_canvas.figure.axes:
            self._setup_picker(
                self.plot_canvas.canvas,
                self.plot_canvas.figure.axes[0],
                status_label=None
            )
        if not hasattr(self, '_mouse_move_cid'):
            self._mouse_move_cid = self.plot_canvas.canvas.mpl_connect(
                "motion_notify_event", self._on_mouse_move)
        self._x = x
        self._y = y

    def _on_load_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.btn_reload.setEnabled(True)
        self.lbl_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Load Error", f"Failed to load:\n{msg}")

    def _on_select_subset(self):
        if self.dataset is None:
            return

        Nt = self.dataset["Nt"]
        is_stereo = self.dataset.get("is_stereo", False)

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Snapshot Subset")
        dlg.setFixedWidth(380)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)

        lay.addWidget(QLabel(
            f"Dataset has {Nt} snapshots loaded.\n"
            "Select a subset for analysis:"))

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From snapshot:"))
        spin_from = QSpinBox()
        spin_from.setRange(1, Nt)
        spin_from.setValue(1)
        spin_from.setFixedWidth(75)
        range_row.addWidget(spin_from)
        range_row.addSpacing(12)
        range_row.addWidget(QLabel("To snapshot:"))
        spin_to = QSpinBox()
        spin_to.setRange(1, Nt)
        spin_to.setValue(Nt)
        spin_to.setFixedWidth(75)
        range_row.addWidget(spin_to)
        range_row.addStretch()
        lay.addLayout(range_row)

        stride_row = QHBoxLayout()
        stride_row.addWidget(QLabel("Use every K-th:"))
        spin_stride = QSpinBox()
        spin_stride.setRange(1, 50)
        spin_stride.setValue(1)
        spin_stride.setFixedWidth(65)
        stride_row.addWidget(spin_stride)
        stride_row.addStretch()
        lay.addLayout(stride_row)

        lbl_count = QLabel()
        lbl_count.setStyleSheet("color: #aaa; font-size: 11px;")
        lay.addWidget(lbl_count)

        def _update():
            n = len(range(spin_from.value() - 1, spin_to.value(), spin_stride.value()))
            lbl_count.setText(f"Snapshots selected: {n}")

        spin_from.valueChanged.connect(_update)
        spin_to.valueChanged.connect(_update)
        spin_stride.valueChanged.connect(_update)
        _update()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        indices = list(range(spin_from.value() - 1, spin_to.value(), spin_stride.value()))
        if not indices:
            return

        original_Nt = Nt
        if self._full_dataset is None:
            import copy
            self._full_dataset = copy.deepcopy(self.dataset)

        ds = self.dataset
        ds["U"] = ds["U"][:, :, indices]
        ds["V"] = ds["V"][:, :, indices]
        if is_stereo:
            ds["W"] = ds["W"][:, :, indices]
        # MASK is frame-independent — no subsetting needed
        ds["Nt"] = len(indices)

        stride = spin_stride.value()
        status_msg = f"Subset applied: {len(indices)} of {original_Nt} snapshots"
        if self.radio_tr.isChecked() and stride > 1:
            if self._original_fs is None:
                self._original_fs = self.spin_fs.value()
            effective_fs = self._original_fs / stride
            self.spin_fs.setValue(effective_fs)
            status_msg += f". Effective fs updated to {effective_fs:.1f} Hz"

        self._update_ribbon()
        self.lbl_status.setText(status_msg)
        self.btn_restore_full.setVisible(True)
        self._plot_field()

    def _on_restore_full_dataset(self):
        if self._full_dataset is None:
            return
        self.dataset = self._full_dataset
        self._full_dataset = None
        self.btn_restore_full.setVisible(False)
        self._update_ribbon()
        status_msg = f"Full dataset restored: {self.dataset['Nt']} snapshots"
        if self._original_fs is not None:
            self.spin_fs.setValue(self._original_fs)
            status_msg += f". fs restored to {self._original_fs:.1f} Hz"
            self._original_fs = None
        self.lbl_status.setText(status_msg)
        self._plot_field()

    def _update_ribbon(self, *_):
        ds = self.dataset
        if ds is None:
            return
        Nt     = ds["Nt"]
        nx     = ds["nx"]
        ny     = ds["ny"]
        stereo = ds["is_stereo"]
        x      = ds["x"]
        dx     = abs(x[0, 1] - x[0, 0])
        mem_mb = (3 if stereo else 2) * ny * nx * Nt * 4 / 1e6
        acq_type = "2D3C" if stereo else "2D2C"
        parts = [
            f"Grid: {nx} \u00d7 {ny}",
            f"dx/dy: {dx:.3f} mm",
            f"Snapshots: {Nt}",
        ]
        if self.radio_tr.isChecked():
            dt_ms = 1000.0 / self.spin_fs.value()
            parts.append(f"dt: {dt_ms:.2f} ms")
        parts.append(f"Type: {acq_type}")
        parts.append(f"Memory: ~{mem_mb:.0f} MB")
        self.lbl_info_ribbon.setText(" \u00b7 ".join(parts))

    def _check_convergence(self, Nt):
        if self.radio_tr.isChecked():
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

    def _on_tr_changed(self):
        self._update_ribbon()

    def _on_fs_changed(self):
        self._update_ribbon()

    def is_time_resolved(self):
        return self.radio_tr.isChecked()

    def get_fs(self):
        return self.spin_fs.value()

    # ----------------------------------------------------------------------- #
    # Plot field
    # ----------------------------------------------------------------------- #

    def _on_field_changed(self):
        if self.dataset is not None:
            self._plot_field()

    def _override_home_button(self):
        toolbar = self.plot_canvas.toolbar
        for action in toolbar.actions():
            if 'home' in action.text().lower() or 'reset' in action.text().lower():
                try:
                    action.triggered.disconnect()
                except Exception:
                    pass
                action.triggered.connect(self._go_home)
                break
        actions = toolbar.actions()
        if actions:
            try:
                actions[0].triggered.disconnect()
            except Exception:
                pass
            actions[0].triggered.connect(self._go_home)

    def _go_home(self):
        if hasattr(self, '_home_xlim') and self.plot_canvas.figure.axes:
            ax = self.plot_canvas.figure.axes[0]
            ax.set_xlim(self._home_xlim)
            ax.set_ylim(self._home_ylim)
            self.plot_canvas.canvas.draw_idle()

    def _plot_field(self):
        if self.dataset is None:
            return

        ds         = self.dataset
        x, y       = ds["x"], ds["y"]
        from core.dataset_utils import get_masked
        U = get_masked(ds, "U"); V = get_masked(ds, "V"); W = get_masked(ds, "W")
        field_name = self.combo_field.currentText()
        skip_x     = self.spin_skip_x.value()
        skip_y     = self.spin_skip_y.value()
        vec_scale  = self.spin_scale.value()
        arrow_size = self.spin_arrow_size.value()

        mean_u       = np.nanmean(U, axis=2)
        mean_v       = np.nanmean(V, axis=2)
        invalid_mask = ~ds["MASK"]

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

        clean  = self.chk_clean_export.isChecked()
        hide_cb = self.chk_hide_colorbar.isChecked()

        draw_contour = self.chk_draw_on_contour.isChecked()

        if draw_contour:
            cf = ax.contourf(x, y, field, levels=50, cmap="RdBu_r")
            if hide_cb:
                cb = self.plot_canvas.figure.colorbar(cf, ax=ax,
                                                      label=cbar, shrink=0.8)
                cb.remove()
                self.plot_canvas.figure.tight_layout(pad=0.5)
            else:
                cb = self.plot_canvas.figure.colorbar(cf, ax=ax,
                                                      label=cbar, shrink=0.8)
                if clean:
                    cb.ax.set_ylabel("")
                    cb.ax.tick_params(size=0, labelsize=0)
                    cb.outline.set_visible(False)

        if self.rb_overlay_vec.isChecked():
            xs  = x[::skip_y, ::skip_x]
            ys  = y[::skip_y, ::skip_x]
            us  = mean_u[::skip_y, ::skip_x].copy()
            vs  = mean_v[::skip_y, ::skip_x].copy()
            inv = invalid_mask[::skip_y, ::skip_x]
            us[inv] = np.nan; vs[inv] = np.nan
            mag = np.sqrt(us**2 + vs**2)
            mag[inv] = np.nan
            max_mag = np.nanmax(mag)
            if not max_mag or not np.isfinite(max_mag):
                return
            u_scaled = us / max_mag
            v_scaled = vs / max_mag

            x_range   = float(np.nanmax(xs) - np.nanmin(xs))
            arrow_len = (x_range / xs.shape[1]) * vec_scale * 2.5

            ax.quiver(xs, ys, u_scaled, v_scaled,
                      color='k',
                      scale=1.0 / arrow_len,
                      scale_units='xy',
                      angles='xy',
                      width=0.0012 * arrow_size,
                      headwidth=4 * arrow_size,
                      headlength=4 * arrow_size,
                      headaxislength=3.5 * arrow_size,
                      alpha=0.75)

        elif self.rb_overlay_sl.isChecked() and self._rakes:
            # Use linspace to guarantee perfectly uniform spacing (PIV grids
            # are uniform but floating-point noise causes streamplot to reject them)
            x_raw = ds["x"][0, :]
            y_raw = ds["y"][:, 0]
            x_1d = np.linspace(float(x_raw[0]), float(x_raw[-1]), len(x_raw))
            y_1d = np.linspace(float(y_raw[0]), float(y_raw[-1]), len(y_raw))
            # Replace NaN with 0 so streamplot doesn't crash on masked areas
            u_sl = np.where(invalid_mask, 0.0, mean_u)
            v_sl = np.where(invalid_mask, 0.0, mean_v)

            for rake in self._rakes:
                x0r, y0r = rake["p0"]
                x1r, y1r = rake["p1"]
                n_seeds  = rake["n_seeds"]
                lw       = rake["lw"]
                color    = rake["color"]
                seed_x = np.linspace(x0r, x1r, n_seeds)
                seed_y = np.linspace(y0r, y1r, n_seeds)
                start_points = np.column_stack([seed_x, seed_y])
                try:
                    ax.streamplot(x_1d, y_1d, u_sl, v_sl,
                                  start_points=start_points,
                                  color=color,
                                  linewidth=lw,
                                  density=5,
                                  integration_direction='both',
                                  broken_streamlines=True)
                except Exception as e:
                    import traceback
                    print("[Streamlines] EXCEPTION:", e)
                    print(traceback.format_exc())
                    self.lbl_status.setText(f"Streamplot error: {e}")

        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(field_name, fontsize=10)
        ax.set_aspect("equal")
        ax.set_facecolor("white")

        if clean:
            ax.set_title("")
            ax.axis('off')

        self.plot_canvas.canvas.draw()
        self._home_xlim = ax.get_xlim()
        self._home_ylim = ax.get_ylim()

        # Update picker
        if self.plot_canvas.figure.axes:
            self._pick_field_ax = self.plot_canvas.figure.axes[0]
        self._x = x
        self._y = y
        self._last_field_values = field
        self.lbl_status.setText(f"Displaying: {field_name}")

    def _on_mouse_move(self, event):
        if not hasattr(self, '_pick_field_ax') or event.inaxes != self._pick_field_ax:
            return
        if not hasattr(self, '_last_field_values') or self._last_field_values is None:
            return
        x_mouse = event.xdata
        y_mouse = event.ydata
        if x_mouse is None or y_mouse is None:
            return

        col = int(np.argmin(np.abs(self._x[0, :] - x_mouse)))
        row = int(np.argmin(np.abs(self._y[:, 0] - y_mouse)))
        col = int(np.clip(col, 0, self._last_field_values.shape[1] - 1))
        row = int(np.clip(row, 0, self._last_field_values.shape[0] - 1))

        value = self._last_field_values[row, col]
        self.lbl_status.setText(
            f"x = {self._x[row, col]:.2f} mm  "
            f"y = {self._y[row, col]:.2f} mm  "
            f"value = {value:.4f}")

    # ----------------------------------------------------------------------- #
    # Overlay mode helpers
    # ----------------------------------------------------------------------- #

    def _on_overlay_mode_changed(self):
        vec  = self.rb_overlay_vec.isChecked()
        sl   = self.rb_overlay_sl.isChecked()
        show = (vec or sl) and self.dataset is not None
        self.overlay_ribbon.setVisible(show)
        self._vec_controls.setVisible(vec)
        self._sl_controls.setVisible(sl)
        self._on_field_changed()

    def _on_draw_rake_toggle(self, checked):
        if checked:
            self.btn_draw_rake.setText("Cancel")
            self._rake_press_xy = None
            self._cid_press   = self.plot_canvas.canvas.mpl_connect(
                "button_press_event",   self._on_rake_press)
            self._cid_motion  = self.plot_canvas.canvas.mpl_connect(
                "motion_notify_event",  self._on_rake_motion)
            self._cid_release = self.plot_canvas.canvas.mpl_connect(
                "button_release_event", self._on_rake_release)
            self.lbl_status.setText("Click and drag on the field to draw a rake line.")
        else:
            self.btn_draw_rake.setText("Draw Rake")
            for cid in ("_cid_press", "_cid_motion", "_cid_release"):
                if hasattr(self, cid):
                    self.plot_canvas.canvas.mpl_disconnect(getattr(self, cid))
            self._clear_rake_artist()

    def _on_rake_press(self, event):
        if event.inaxes is None: return
        self._rake_press_xy = (event.xdata, event.ydata)

    def _on_rake_motion(self, event):
        if self._rake_press_xy is None or event.inaxes is None: return
        self._clear_rake_artist()
        x0, y0 = self._rake_press_xy
        ln, = event.inaxes.plot([x0, event.xdata], [y0, event.ydata],
                                 "w--", linewidth=1.5, alpha=0.8, zorder=20)
        self._rake_artist = ln
        self.plot_canvas.canvas.draw_idle()

    def _on_rake_release(self, event):
        if self._rake_press_xy is None or event.inaxes is None: return
        x0, y0 = self._rake_press_xy
        x1, y1 = event.xdata, event.ydata
        self._rake_press_xy = None
        if abs(x1 - x0) < 0.5 and abs(y1 - y0) < 0.5:
            self.lbl_status.setText("Rake too short — try again.")
            return
        self._rakes.append({
            "p0": (x0, y0), "p1": (x1, y1),
            "n_seeds": self.spin_sl_seeds.value(),
            "lw": self.spin_sl_lw.value(),
            "color": self._sl_color,
        })
        # Disconnect events and reset button (rake artist stays visible during replot)
        self.btn_draw_rake.setChecked(False)
        self.btn_draw_rake.setText("Draw Rake")
        for cid in ("_cid_press", "_cid_motion", "_cid_release"):
            if hasattr(self, cid):
                self.plot_canvas.canvas.mpl_disconnect(getattr(self, cid))
        # Replot — this clears and redraws the figure (rake artist removed naturally)
        self._rake_artist = None   # prevent _clear_rake_artist from crashing on stale ref
        self._on_field_changed()

    def _clear_rake_artist(self):
        if self._rake_artist is not None:
            try: self._rake_artist.remove()
            except Exception: pass
            self._rake_artist = None
            self.plot_canvas.canvas.draw_idle()

    def _on_sl_color_pick(self):
        from PyQt6.QtGui import QColor
        presets = [
            QColor("white"), QColor("black"), QColor("red"),
            QColor("blue"),  QColor("green"), QColor("yellow"),
            QColor("cyan"),  QColor("magenta"),
        ]
        dlg = QColorDialog(QColor(self._sl_color), self)
        for i, c in enumerate(presets):
            QColorDialog.setCustomColor(i, c)
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            self._sl_color = dlg.selectedColor().name()
            self.btn_sl_color.setStyleSheet(
                f"background: {self._sl_color}; border: 1px solid #888;")

    def _on_sl_reset(self):
        self._rakes.clear()
        self._on_field_changed()

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

    def _run_spectra(self):
        if not self._check_data(): return
        self._open_window(SpectraWindow(
            self.dataset,
            is_time_resolved=self.is_time_resolved(),
            fs=self.get_fs()))

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

    def _run_pod(self):
        if not self._check_data(): return
        self._open_window(PODWindow(
            self.dataset,
            is_time_resolved=self.is_time_resolved(),
            fs=self.get_fs(),
        ))

    def _run_dmd(self):
        if not self._check_data(): return
        if not self.is_time_resolved():
            QMessageBox.warning(self, "TR Required",
                "DMD requires Time-Resolved data.\n"
                "Please select TR acquisition type.")
            return
        from gui.dmd_window import DmdWindow
        win = DmdWindow(self.dataset, fs=self.get_fs())
        if not win._valid:
            return
        self._dmd_win = win
        self._open_window(self._dmd_win)

    def _update_dmd_btn_state(self, nontr_checked):
        if self.btn_dmd is not None and self.dataset is not None:
            self.btn_dmd.setEnabled(not nontr_checked)

    def _run_vortex(self):
        if not self._check_data(): return
        from gui.vortex_window import VortexWindow
        self._open_window(VortexWindow(self.dataset))

    def _run_masking(self):
        if not self._check_data(): return
        win = MaskWindow(self.dataset, main_window=self)
        self._open_window(win)

    def _run_transform(self):
        if not self._check_data(): return
        # Ensure transform_log exists
        self.dataset.setdefault("transform_log", [])
        win = TransformWindow(
            self.dataset,
            on_transform_done=self._on_transform_done)
        self._open_window(win)

    def _on_transform_done(self):
        """Called by TransformWindow after each successful transform."""
        status = transform_status_string(self.dataset)
        if status:
            self.lbl_transform_status.setText(
                f"Data transformed:  {status}")
            self.transform_strip.setVisible(True)
        # Refresh main plot with transformed data
        self._plot_field()

    # ----------------------------------------------------------------------- #
    # Close event
    # ----------------------------------------------------------------------- #

    def closeEvent(self, event):
        if self.dataset is None or len(self.dataset.get("files", [])) == 0:
            event.accept()
            return
        reply = QMessageBox.question(
            self,
            "Exit uPrime",
            "Are you sure you want to exit uPrime?\nAny unsaved analysis results will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.dataset:
                cleanup_memmap(self.dataset)
            event.accept()
        else:
            event.ignore()

    # ----------------------------------------------------------------------- #
    # About dialog
    # ----------------------------------------------------------------------- #

    def _open_manual(self):
        import subprocess
        path = _asset_path("uPrime_Manual.pdf")
        if not os.path.exists(path):
            QMessageBox.warning(self, "Manual not found",
                "User manual not found in assets folder.\n"
                "Expected: " + path)
            return
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _on_about(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QLineEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("About uPrime")
        dlg.setFixedWidth(480)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(6); lay.setContentsMargins(24, 24, 24, 24)

        logo_lbl = QLabel()
        logo_lbl.setPixmap(_load_logo_pixmap(size=(120, 120)))
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(logo_lbl)

        ver_lbl = QLabel("v0.4.1  \u00b7  Alpha Release")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver_lbl.setStyleSheet("color:gray;")
        lay.addWidget(ver_lbl)

        lay.addWidget(self._separator_dlg())

        desc = QLabel(
            "Open-source fluid velocity field analysis.\n"
            "Supports Tecplot .dat format from DaVis and CFD solvers."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(desc)

        lay.addWidget(self._separator_dlg())

        credit = QLabel(
            "Developed by <b>Jibu Tom Jose</b><br>"
            "Transient Fluid Mechanics Lab, Technion<br>"
            "Built with assistance from Claude (Anthropic)"
        )
        credit.setWordWrap(True)
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(credit)

        github_lbl = QLabel('<a href="https://github.com/CmdrRyder/uPrime" style="color: #888888; text-decoration: none;">github.com/CmdrRyder/uPrime</a>')
        github_lbl.setOpenExternalLinks(True)
        github_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        github_lbl.setStyleSheet("color:gray;")
        lay.addWidget(github_lbl)

        license_lbl = QLabel("Licensed under GNU GPL v3")
        license_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_lbl.setStyleSheet("color:gray;")
        lay.addWidget(license_lbl)

        lay.addSpacing(8)

        citation_lbl = QLabel(
            "Jibu Tom Jose, & Ram, O. (2026). uPrime: Open-source software for velocity field "
            "and turbulence analysis from PIV and CFD data. TFML, Technion. Zenodo."
        )
        citation_lbl.setFixedWidth(440)
        citation_lbl.setWordWrap(True)
        citation_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(citation_lbl)

        doi_inner = QWidget()
        doi_inner.setMaximumWidth(440)
        doi_row = QHBoxLayout(doi_inner)
        doi_row.setContentsMargins(0, 0, 0, 0)
        doi_row.setSpacing(4)
        doi_box = QLineEdit("https://doi.org/10.5281/zenodo.19376184")
        doi_box.setReadOnly(True)
        doi_box.setMinimumWidth(340)
        doi_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        doi_box.setStyleSheet("border: none; background: transparent; color: palette(text);")
        doi_box.home(False)
        doi_row.addWidget(doi_box)
        btn_copy = QPushButton("\u2398")
        btn_copy.setFixedSize(24, 24)
        btn_copy.setFlat(True)
        btn_copy.setToolTip("Copy DOI")
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(doi_box.text()))
        doi_row.addWidget(btn_copy)

        doi_outer = QHBoxLayout()
        doi_outer.addStretch()
        doi_outer.addWidget(doi_inner)
        doi_outer.addStretch()
        lay.addLayout(doi_outer)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()

    def _separator_dlg(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#444;")
        return line