"""
gui/transform_window.py
-----------------------
Standalone coordinate transformation window for uPrime.

Allows the user to:
  1. Rotate the dataset to correct calibration tilt (≤ ±10°)
     - Draw a reference line on the field preview
     - Choose linear or cubic interpolation
  2. Shift the coordinate origin
     - Click on the preview to set origin, or type x/y values

All transforms are applied in-place via core.transform.
The parent MainWindow is notified via a callback so it can update its
status strip and re-plot the main field.

Layout
------
Left (~420px): controls (rotation section + shift section)
Right (~560px): field preview (updates after each applied transform)
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QRadioButton, QButtonGroup, QComboBox,
    QDoubleSpinBox, QSizePolicy, QMessageBox, QProgressBar,
    QApplication, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from core.transform import (apply_rotation, apply_shift,
                            apply_mirror_x, apply_mirror_y,
                            transform_status_string)


# ---------------------------------------------------------------------------
# Background worker for rotation (can be slow on large datasets)
# ---------------------------------------------------------------------------

class RotationWorker(QThread):
    progress  = pyqtSignal(int)
    finished  = pyqtSignal()
    error     = pyqtSignal(str)

    def __init__(self, dataset, angle_deg, method):
        super().__init__()
        self.dataset   = dataset
        self.angle_deg = angle_deg
        self.method    = method

    def run(self):
        try:
            apply_rotation(
                self.dataset, self.angle_deg, self.method,
                chunk_size=200,
                progress_callback=self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _hline():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color:#555;")
    return f


# ---------------------------------------------------------------------------
# Transform window
# ---------------------------------------------------------------------------

class TransformWindow(PickerMixin, QWidget):
    """
    Standalone window for grid rotation and origin shift.

    Parameters
    ----------
    dataset          : dict -- the shared dataset (modified in-place)
    on_transform_done: callable() -- called after each successful transform
                       so that MainWindow can refresh its plot and status strip
    """

    def __init__(self, dataset, on_transform_done=None, parent=None):
        super().__init__(parent)
        self.dataset           = dataset
        self._on_done_cb       = on_transform_done
        self._ref_line_artist  = None
        self._origin_dot       = None
        self._pick_origin_mode = False   # True while waiting for origin click
        self._clicked_x        = None    # last clicked coords (mode 2)
        self._clicked_y        = None
        self._worker           = None

        self.setWindowTitle("Transform / Align Dataset")
        self.resize(1000, 680)

        self._build_ui()
        self._draw_preview()
        self._connect_mouse()
        self._setup_picker(self.preview_canvas, self.preview_ax,
                           status_label=self.lbl_status)

    # -----------------------------------------------------------------------
    # Drawing active -- suppress PickerMixin cross-marker
    # -----------------------------------------------------------------------

    def _drawing_active(self):
        return True   # we handle all clicks ourselves

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ================================================================
        # LEFT: controls
        # ================================================================
        left = QWidget()
        left.setFixedWidth(420)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        # Title
        title = QLabel("Coordinate Transform")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        ll.addWidget(title)

        warn = QLabel(
            "\u25cf  Transforms overwrite data in memory and are NOT reversible.\n"
            "To undo, close this window and reload the dataset.")
        warn.setStyleSheet(
            "color:#e06c75; font-size:11px; background:#0e0e1a;"
            "padding:5px; border-left:3px solid #e06c75; border-radius:2px;")
        warn.setWordWrap(True)
        ll.addWidget(warn)

        ll.addWidget(_hline())

        # ----------------------------------------------------------------
        # Section 1: Rotation
        # ----------------------------------------------------------------
        rot_grp = QGroupBox("1.  Rotate  (correct calibration tilt)")
        rot_grp.setStyleSheet(
            "QGroupBox::title { font-size: 12px; font-weight: bold; }")
        rl = QVBoxLayout(rot_grp)

        _lbl_rot_instr = QLabel(
            "Draw a line on the preview (right panel) over a feature that\n"
            "should be horizontal or vertical. The correction angle is\n"
            "computed automatically. Max \u00b110\u00b0.")
        _lbl_rot_instr.setStyleSheet("font-size: 11px;")
        rl.addWidget(_lbl_rot_instr)

        ref_row = QHBoxLayout()
        _lbl_ref = QLabel("Reference line is:")
        _lbl_ref.setStyleSheet("font-size: 11px;")
        ref_row.addWidget(_lbl_ref)
        self.rb_horiz = QRadioButton("Horizontal")
        self.rb_vert  = QRadioButton("Vertical")
        self.rb_horiz.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_horiz)
        bg.addButton(self.rb_vert)
        ref_row.addWidget(self.rb_horiz)
        ref_row.addWidget(self.rb_vert)
        ref_row.addStretch()
        rl.addLayout(ref_row)

        self.lbl_angle_detected = QLabel("Detected angle:  --")
        self.lbl_angle_detected.setStyleSheet("font-size:11px;")
        rl.addWidget(self.lbl_angle_detected)

        angle_row = QHBoxLayout()
        _lbl_angle = QLabel("Correction angle [°]:")
        _lbl_angle.setStyleSheet("font-size: 11px;")
        angle_row.addWidget(_lbl_angle)
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-10.0, 10.0)
        self.spin_angle.setValue(0.0)
        self.spin_angle.setDecimals(2)
        self.spin_angle.setSingleStep(0.1)
        self.spin_angle.setFixedWidth(80)
        self.spin_angle.setToolTip(
            "Positive = counter-clockwise correction.\n"
            "Auto-filled from drawn line, but editable.")
        angle_row.addWidget(self.spin_angle)
        angle_row.addStretch()
        rl.addLayout(angle_row)

        interp_row = QHBoxLayout()
        _lbl_interp = QLabel("Interpolation:")
        _lbl_interp.setStyleSheet("font-size: 11px;")
        interp_row.addWidget(_lbl_interp)
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(["Linear", "Cubic"])
        self.combo_interp.setToolTip(
            "Linear: faster, slightly smoother edges.\n"
            "Cubic: more accurate interior, may ring at boundaries.")
        interp_row.addWidget(self.combo_interp)
        interp_row.addStretch()
        rl.addLayout(interp_row)

        self.progress_rot = QProgressBar()
        self.progress_rot.setVisible(False)
        self.progress_rot.setRange(0, 100)
        rl.addWidget(self.progress_rot)

        self.btn_apply_rot = QPushButton("Apply Rotation")
        self.btn_apply_rot.clicked.connect(self._on_apply_rotation)
        rl.addWidget(self.btn_apply_rot)

        ll.addWidget(rot_grp)
        ll.addWidget(_hline())

        # ----------------------------------------------------------------
        # Section 2: Shift
        # ----------------------------------------------------------------
        shift_grp = QGroupBox("2.  Shift Origin")
        shift_grp.setStyleSheet(
            "QGroupBox::title { font-size: 12px; font-weight: bold; }")
        sl = QVBoxLayout(shift_grp)
        sl.setSpacing(5)

        # Mode selector
        mode_row = QHBoxLayout()
        self.rb_shift_set_origin = QRadioButton("Set new origin")
        self.rb_shift_set_known  = QRadioButton("Set known point")
        self.rb_shift_set_origin.setChecked(True)
        self.rb_shift_set_origin.setStyleSheet("font-size: 11px;")
        self.rb_shift_set_known.setStyleSheet("font-size: 11px;")
        _shift_bg = QButtonGroup(self)
        _shift_bg.addButton(self.rb_shift_set_origin)
        _shift_bg.addButton(self.rb_shift_set_known)
        self.rb_shift_set_origin.toggled.connect(self._on_shift_mode_changed)
        mode_row.addWidget(self.rb_shift_set_origin)
        mode_row.addWidget(self.rb_shift_set_known)
        mode_row.addStretch()
        sl.addLayout(mode_row)

        self.btn_pick_origin = QPushButton("Activate click mode")
        self.btn_pick_origin.setCheckable(True)
        self.btn_pick_origin.clicked.connect(self._on_toggle_pick_origin)
        sl.addWidget(self.btn_pick_origin)

        # Mode 1: direct x/y shift spinboxes
        self._shift_mode1_widget = QWidget()
        m1 = QHBoxLayout(self._shift_mode1_widget)
        m1.setContentsMargins(0, 0, 0, 0)
        _lbl_dx = QLabel("X shift [mm]:")
        _lbl_dx.setStyleSheet("font-size: 11px;")
        m1.addWidget(_lbl_dx)
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(-5000.0, 5000.0)
        self.spin_dx.setValue(0.0)
        self.spin_dx.setDecimals(3)
        self.spin_dx.setSingleStep(0.1)
        self.spin_dx.setFixedWidth(90)
        self.spin_dx.setToolTip("Subtract this value from all x coordinates.")
        m1.addWidget(self.spin_dx)
        m1.addSpacing(12)
        _lbl_dy = QLabel("Y shift [mm]:")
        _lbl_dy.setStyleSheet("font-size: 11px;")
        m1.addWidget(_lbl_dy)
        self.spin_dy = QDoubleSpinBox()
        self.spin_dy.setRange(-5000.0, 5000.0)
        self.spin_dy.setValue(0.0)
        self.spin_dy.setDecimals(3)
        self.spin_dy.setSingleStep(0.1)
        self.spin_dy.setFixedWidth(90)
        self.spin_dy.setToolTip("Subtract this value from all y coordinates.")
        m1.addWidget(self.spin_dy)
        m1.addStretch()
        sl.addWidget(self._shift_mode1_widget)

        # Mode 2: real-world coordinate entry + status
        self._shift_mode2_widget = QWidget()
        m2 = QVBoxLayout(self._shift_mode2_widget)
        m2.setContentsMargins(0, 0, 0, 0)
        m2.setSpacing(4)
        real_row = QHBoxLayout()
        _lbl_real_x = QLabel("Real x [mm]:")
        _lbl_real_x.setStyleSheet("font-size: 11px;")
        real_row.addWidget(_lbl_real_x)
        self.spin_real_x = QDoubleSpinBox()
        self.spin_real_x.setRange(-5000.0, 5000.0)
        self.spin_real_x.setValue(0.0)
        self.spin_real_x.setDecimals(3)
        self.spin_real_x.setSingleStep(0.1)
        self.spin_real_x.setFixedWidth(90)
        self.spin_real_x.setToolTip(
            "The physical x-coordinate the clicked point should have after the shift.")
        self.spin_real_x.valueChanged.connect(self._update_known_point_status)
        real_row.addWidget(self.spin_real_x)
        real_row.addSpacing(12)
        _lbl_real_y = QLabel("Real y [mm]:")
        _lbl_real_y.setStyleSheet("font-size: 11px;")
        real_row.addWidget(_lbl_real_y)
        self.spin_real_y = QDoubleSpinBox()
        self.spin_real_y.setRange(-5000.0, 5000.0)
        self.spin_real_y.setValue(0.0)
        self.spin_real_y.setDecimals(3)
        self.spin_real_y.setSingleStep(0.1)
        self.spin_real_y.setFixedWidth(90)
        self.spin_real_y.setToolTip(
            "The physical y-coordinate the clicked point should have after the shift.")
        self.spin_real_y.valueChanged.connect(self._update_known_point_status)
        real_row.addWidget(self.spin_real_y)
        real_row.addStretch()
        m2.addLayout(real_row)
        self.lbl_known_point = QLabel("No point clicked yet.")
        self.lbl_known_point.setStyleSheet("font-size: 10px; color: #aaa;")
        self.lbl_known_point.setWordWrap(True)
        m2.addWidget(self.lbl_known_point)
        self._shift_mode2_widget.setVisible(False)
        sl.addWidget(self._shift_mode2_widget)

        self.btn_apply_shift = QPushButton("Apply Shift")
        self.btn_apply_shift.clicked.connect(self._on_apply_shift)
        sl.addWidget(self.btn_apply_shift)

        ll.addWidget(shift_grp)
        ll.addWidget(_hline())

        # ----------------------------------------------------------------
        # Section 3: Mirror
        # ----------------------------------------------------------------
        mirror_grp = QGroupBox("3.  Mirror")
        mirror_grp.setStyleSheet(
            "QGroupBox::title { font-size: 12px; font-weight: bold; }")
        ml = QVBoxLayout(mirror_grp)

        _lbl_mirror_instr = QLabel(
            "Flip the dataset along an axis through the current origin.")
        _lbl_mirror_instr.setStyleSheet("font-size: 11px;")
        ml.addWidget(_lbl_mirror_instr)

        mirror_btn_row = QHBoxLayout()
        self.btn_mirror_x = QPushButton("Mirror X  (flip left\u2013right)")
        self.btn_mirror_x.clicked.connect(self._on_apply_mirror_x)
        self.btn_mirror_y = QPushButton("Mirror Y  (flip top\u2013bottom)")
        self.btn_mirror_y.clicked.connect(self._on_apply_mirror_y)
        mirror_btn_row.addWidget(self.btn_mirror_x)
        mirror_btn_row.addWidget(self.btn_mirror_y)
        ml.addLayout(mirror_btn_row)

        ll.addWidget(mirror_grp)
        ll.addWidget(_hline())

        # ----------------------------------------------------------------
        # Transform history
        # ----------------------------------------------------------------
        self.lbl_history = QLabel("No transforms applied.")
        self.lbl_history.setStyleSheet(
            "font-size:11px; color:#aaa; padding:2px;")
        self.lbl_history.setWordWrap(True)
        ll.addWidget(self.lbl_history)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        ll.addWidget(_hline())

        self.btn_close = QPushButton("Done \u2014 Close Window")
        self.btn_close.setStyleSheet(
            "QPushButton { background: #1e5c1e; color: #c8f0c8;"
            " font-size: 11px; padding: 6px; border-radius: 3px; }"
            "QPushButton:hover { background: #276827; }"
            "QPushButton:pressed { background: #164416; }")
        self.btn_close.clicked.connect(self.close)
        ll.addWidget(self.btn_close)

        ll.addStretch(1)

        # ================================================================
        # RIGHT: field preview
        # ================================================================
        right = QWidget()
        rl2 = QVBoxLayout(right)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.setSpacing(2)

        self.preview_fig    = Figure()
        self.preview_canvas = FigureCanvas(self.preview_fig)
        self.preview_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                          QSizePolicy.Policy.Expanding)
        self.preview_toolbar = DrawAwareToolbar(self.preview_canvas, self)

        rl2.addWidget(QLabel(
            "Preview  (left-click+drag: draw reference line  |  "
            "left-click: set origin when active)",
            alignment=Qt.AlignmentFlag.AlignCenter))
        rl2.addWidget(self.preview_toolbar)
        rl2.addWidget(self.preview_canvas)

        root.addWidget(left)
        root.addWidget(right, stretch=1)

    # -----------------------------------------------------------------------
    # Field preview
    # -----------------------------------------------------------------------

    def _draw_preview(self):
        ds = self.dataset
        x, y = ds["x"], ds["y"]
        from core.dataset_utils import get_masked
        U_mean = np.nanmean(get_masked(ds, "U"), axis=2)
        U_mean[~ds["MASK"]] = np.nan

        # Constrained height for aspect ratio (same strategy as tke_budget)
        x_ext = float(np.nanmax(x) - np.nanmin(x))
        y_ext = float(np.nanmax(y) - np.nanmin(y))
        ratio = (y_ext / x_ext) if x_ext > 0 else 0.5

        self.preview_fig.clear()
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_ax.contourf(x, y, U_mean, levels=50,
                                 cmap="RdBu_r", extend="neither")
        self.preview_ax.set_xlabel("x [mm]", fontsize=9)
        self.preview_ax.set_ylabel("y [mm]", fontsize=9)
        self.preview_ax.set_title(
            "Mean U  --  draw line or click origin", fontsize=9)
        self.preview_ax.set_aspect("equal")
        self.preview_ax.set_facecolor("white")
        self.preview_ax.tick_params(labelsize=8)
        self.preview_fig.tight_layout(pad=0.3)
        self.preview_canvas.draw()

        self._x = x
        self._y = y
        self._last_field_values = U_mean
        self._ref_line_artist = None
        self._origin_dot      = None

    # -----------------------------------------------------------------------
    # Mouse
    # -----------------------------------------------------------------------

    def _connect_mouse(self):
        self._press_xy = None
        self.preview_canvas.mpl_connect("button_press_event",   self._on_press)
        self.preview_canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.preview_canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        if event.inaxes != self.preview_ax:
            return
        if self._toolbar_active(self.preview_toolbar):
            return

        if event.button == 1:
            if self._pick_origin_mode:
                # Click sets origin spinboxes
                self._set_origin_from_click(event.xdata, event.ydata)
            else:
                # Start drawing reference line
                self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None:
            return
        if event.inaxes != self.preview_ax or event.xdata is None:
            return
        if self._toolbar_active(self.preview_toolbar):
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._draw_ref_line(x0, y0, x1, y1)
        # Live angle update during drag
        if abs(x1 - x0) > 0.5 or abs(y1 - y0) > 0.5:
            self._compute_angle_from_line(x0, y0, x1, y1)

    def _on_release(self, event):
        if self._press_xy is None:
            return
        if self._toolbar_active(self.preview_toolbar):
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        self._press_xy = None

        if event.inaxes != self.preview_ax or event.xdata is None:
            return

        x1, y1 = event.xdata, event.ydata
        if abs(x1 - x0) < 0.1 and abs(y1 - y0) < 0.1:
            self.lbl_status.setText("Line too short -- try again.")
            return

        self._draw_ref_line(x0, y0, x1, y1)
        self._compute_angle_from_line(x0, y0, x1, y1)

    def _draw_ref_line(self, x0, y0, x1, y1):
        """Draw or update the reference line on the preview."""
        if self._ref_line_artist is not None:
            try:
                self._ref_line_artist.remove()
            except Exception:
                pass
        ln, = self.preview_ax.plot(
            [x0, x1], [y0, y1],
            color="#00ccff", linewidth=2, linestyle="-",
            marker="o", markersize=5, zorder=15)
        self._ref_line_artist = ln
        self.preview_canvas.draw()

    def _compute_angle_from_line(self, x0, y0, x1, y1):
        """
        Compute the correction angle from the drawn line.

        The key insight: a line drawn right-to-left gives arctan2 near ±180°
        but represents the SAME edge as left-to-right (near 0°). We fold the
        raw arctan2 output into (-90°, +90°] so the result is always the
        smallest deviation from horizontal or vertical, regardless of draw
        direction.

        Folding rule: if |raw| > 90°, subtract 180° (keeping sign of dy/dx).
        This maps, e.g., -178.68° → -178.68° + 180° = +1.32°.
        """
        dx = x1 - x0
        dy = y1 - y0

        if self.rb_horiz.isChecked():
            raw = float(np.degrees(np.arctan2(dy, dx)))
            # Fold into (-90°, +90°]
            if raw > 90.0:
                raw -= 180.0
            elif raw <= -90.0:
                raw += 180.0
            detected   = raw
            correction = -detected
            ref_name   = "horizontal"
        else:
            # For vertical reference, measure deviation from vertical.
            # arctan2(dx, dy) gives angle from vertical axis.
            raw = float(np.degrees(np.arctan2(dx, dy)))
            if raw > 90.0:
                raw -= 180.0
            elif raw <= -90.0:
                raw += 180.0
            detected   = raw
            correction = -detected
            ref_name   = "vertical"

        # Clamp correction to ±10°
        if abs(correction) > 10.0:
            clamped = float(np.clip(correction, -10.0, 10.0))
            clamp_note = f"  (clamped from {correction:+.2f}\u00b0)"
            correction = clamped
        else:
            clamp_note = ""

        # Direction label for correction
        if abs(correction) < 0.005:
            dir_str = "(no rotation needed)"
        elif correction > 0:
            dir_str = "(counter-clockwise)"
        else:
            dir_str = "(clockwise)"

        self.lbl_angle_detected.setText(
            f"Line tilt from {ref_name}: {detected:+.2f}\u00b0"
            f"{clamp_note}\n"
            f"Correction: {correction:+.2f}\u00b0  {dir_str}")
        self.spin_angle.setValue(correction)
        self.lbl_status.setText(
            f"Correction: {correction:+.2f}\u00b0 {dir_str}. "
            "Edit spinbox if needed, then click Apply Rotation.")

    def _set_origin_from_click(self, x_click, y_click):
        """Handle a preview click for either shift mode."""
        # Draw / update red-cross marker
        if self._origin_dot is not None:
            try:
                self._origin_dot.remove()
            except Exception:
                pass
        dot, = self.preview_ax.plot(
            x_click, y_click, "r+",
            markersize=16, markeredgewidth=2, zorder=20)
        self._origin_dot = dot
        self.preview_canvas.draw()

        if self.rb_shift_set_origin.isChecked():
            # Mode 1: clicked point becomes (0, 0) -- shift = click coords
            self.spin_dx.setValue(float(x_click))
            self.spin_dy.setValue(float(y_click))
            self.lbl_status.setText(
                f"Origin picked at ({x_click:.2f}, {y_click:.2f}) mm. "
                "Click \u2018Apply Shift\u2019 when ready.")
        else:
            # Mode 2: store click; shift computed from real-coord spinboxes
            self._clicked_x = float(x_click)
            self._clicked_y = float(y_click)
            self._update_known_point_status()
            self.lbl_status.setText(
                f"Clicked ({x_click:.2f}, {y_click:.2f}) mm. "
                "Enter real coordinates, then click \u2018Apply Shift\u2019.")

    def _on_toggle_pick_origin(self, checked):
        self._pick_origin_mode = checked
        if checked:
            self.btn_pick_origin.setText("Click mode  (active)")
            self.btn_pick_origin.setStyleSheet("background:#1a4f1a;")
            if self.rb_shift_set_origin.isChecked():
                self.lbl_status.setText("Click on the preview to set the new origin.")
            else:
                self.lbl_status.setText("Click on the preview to select the known point.")
        else:
            self.btn_pick_origin.setText("Activate click mode")
            self.btn_pick_origin.setStyleSheet("")
            self.lbl_status.setText("Ready.")

    def _on_shift_mode_changed(self, checked):
        """Show/hide mode-specific widgets when radio button toggles."""
        is_origin = self.rb_shift_set_origin.isChecked()
        self._shift_mode1_widget.setVisible(is_origin)
        self._shift_mode2_widget.setVisible(not is_origin)
        # Deactivate click mode when switching to avoid stale state
        self.btn_pick_origin.setChecked(False)
        self._on_toggle_pick_origin(False)

    def _update_known_point_status(self):
        """Refresh the 'Clicked → will become' status label in mode 2."""
        if self._clicked_x is None:
            return
        rx = self.spin_real_x.value()
        ry = self.spin_real_y.value()
        self.lbl_known_point.setText(
            f"Clicked: ({self._clicked_x:.2f}, {self._clicked_y:.2f}) mm"
            f"  \u2192  will become ({rx:.3f}, {ry:.3f}) mm")

    # -----------------------------------------------------------------------
    # Apply rotation
    # -----------------------------------------------------------------------

    def _on_apply_rotation(self):
        angle = self.spin_angle.value()

        if abs(angle) < 1e-3:
            QMessageBox.information(self, "No Rotation",
                "Correction angle is effectively zero. Nothing to apply.")
            return

        # Interpolation warning
        method = self.combo_interp.currentText().lower()
        if QMessageBox.warning(
            self, "Interpolation Warning",
            f"Rotating requires interpolating velocity onto a new grid "
            f"({method} interpolation).\n\n"
            "This introduces small errors, especially at boundaries.\n"
            "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        # Already-transformed warning
        log = self.dataset.get("transform_log", [])
        if log:
            if QMessageBox.information(
                self, "Dataset Already Transformed",
                "This dataset has already been transformed.\n"
                "Applying again will COMPOUND the effects.\n\n"
                "To start fresh, close this window and reload the dataset.\n\n"
                "Proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return

        # Final irreversibility warning
        if QMessageBox.warning(
            self, "Irreversible Operation",
            f"This will rotate the dataset by {angle:+.2f}\u00b0 "
            "and overwrite the data in memory.\n\n"
            "\u26a0  This is NOT reversible without reloading the dataset.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        # Run in background thread
        self.btn_apply_rot.setEnabled(False)
        self.btn_apply_shift.setEnabled(False)
        self.progress_rot.setVisible(True)
        self.progress_rot.setValue(0)
        self.lbl_status.setText(
            f"Rotating {angle:+.2f}\u00b0 ({method} interpolation)... "
            "This may take a while for large datasets.")

        self._worker = RotationWorker(self.dataset, angle, method)
        self._worker.progress.connect(self.progress_rot.setValue)
        self._worker.finished.connect(self._on_rotation_done)
        self._worker.error.connect(self._on_rotation_error)
        self._worker.start()

    def _on_rotation_done(self):
        self.progress_rot.setVisible(False)
        self.btn_apply_rot.setEnabled(True)
        self.btn_apply_shift.setEnabled(True)
        self._update_history()
        self._draw_preview()
        self.lbl_status.setText("Rotation applied.")
        if self._on_done_cb:
            self._on_done_cb()

    def _on_rotation_error(self, msg):
        self.progress_rot.setVisible(False)
        self.btn_apply_rot.setEnabled(True)
        self.btn_apply_shift.setEnabled(True)
        QMessageBox.critical(self, "Rotation Error", msg)
        self.lbl_status.setText(f"Error: {msg}")

    # -----------------------------------------------------------------------
    # Apply shift
    # -----------------------------------------------------------------------

    def _on_apply_shift(self):
        if self.rb_shift_set_origin.isChecked():
            # Mode 1: direct shift values
            dx = self.spin_dx.value()
            dy = self.spin_dy.value()
        else:
            # Mode 2: shift = clicked_pos - real_world_pos
            if self._clicked_x is None:
                QMessageBox.information(self, "No Point Clicked",
                    "Activate click mode and click a point on the preview first.")
                return
            dx = self._clicked_x - self.spin_real_x.value()
            dy = self._clicked_y - self.spin_real_y.value()

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            QMessageBox.information(self, "No Shift",
                "Both X and Y shifts are zero. Nothing to apply.")
            return

        # Already-transformed warning
        log = self.dataset.get("transform_log", [])
        if log:
            if QMessageBox.information(
                self, "Dataset Already Transformed",
                "This dataset has already been transformed.\n"
                "The shift will be applied on top of existing transforms.\n\n"
                "Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return

        # Irreversibility warning
        if QMessageBox.warning(
            self, "Irreversible Operation",
            f"This will shift the origin by "
            f"(\u0394x={dx:+.3f}, \u0394y={dy:+.3f}) mm "
            "and overwrite the coordinates.\n\n"
            "\u26a0  This is NOT reversible without reloading the dataset.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        apply_shift(self.dataset, dx, dy)
        self._update_history()
        self._draw_preview()
        self.lbl_status.setText(
            f"Shift applied: \u0394x={dx:+.3f}, \u0394y={dy:+.3f} mm.")

        # Reset mode-specific state
        if self.rb_shift_set_origin.isChecked():
            self.spin_dx.setValue(0.0)
            self.spin_dy.setValue(0.0)
        else:
            self._clicked_x = None
            self._clicked_y = None
            self.lbl_known_point.setText("No point clicked yet.")

        # Deactivate click mode
        self.btn_pick_origin.setChecked(False)
        self._on_toggle_pick_origin(False)

        if self._on_done_cb:
            self._on_done_cb()

    # -----------------------------------------------------------------------
    # Apply mirror
    # -----------------------------------------------------------------------

    def _on_apply_mirror_x(self):
        if QMessageBox.warning(
            self, "Irreversible Operation",
            "This will negate all x-coordinates and the U velocity component.\n\n"
            "\u26a0  This is NOT reversible without reloading the dataset.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        apply_mirror_x(self.dataset)
        self._update_history()
        self._draw_preview()
        self.lbl_status.setText("Mirror X applied.")
        if self._on_done_cb:
            self._on_done_cb()

    def _on_apply_mirror_y(self):
        if QMessageBox.warning(
            self, "Irreversible Operation",
            "This will negate all y-coordinates and the V velocity component.\n\n"
            "\u26a0  This is NOT reversible without reloading the dataset.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        apply_mirror_y(self.dataset)
        self._update_history()
        self._draw_preview()
        self.lbl_status.setText("Mirror Y applied.")
        if self._on_done_cb:
            self._on_done_cb()

    # -----------------------------------------------------------------------
    # History
    # -----------------------------------------------------------------------

    def _update_history(self):
        status = transform_status_string(self.dataset)
        if status:
            self.lbl_history.setText(f"Applied:  {status}")
        else:
            self.lbl_history.setText("No transforms applied.")
