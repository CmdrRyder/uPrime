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
from core.transform import apply_rotation, apply_shift, transform_status_string


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
            "color:#e06c75; font-size:10px; background:#0e0e1a;"
            "padding:5px; border-left:3px solid #e06c75; border-radius:2px;")
        warn.setWordWrap(True)
        ll.addWidget(warn)

        ll.addWidget(_hline())

        # ----------------------------------------------------------------
        # Section 1: Rotation
        # ----------------------------------------------------------------
        rot_grp = QGroupBox("1.  Rotate  (correct calibration tilt)")
        rl = QVBoxLayout(rot_grp)

        rl.addWidget(QLabel(
            "Draw a line on the preview (right panel) over a feature that\n"
            "should be horizontal or vertical. The correction angle is\n"
            "computed automatically. Max \u00b110\u00b0."))

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference line is:"))
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
        angle_row.addWidget(QLabel("Correction angle [°]:"))
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
        interp_row.addWidget(QLabel("Interpolation:"))
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
        sl = QVBoxLayout(shift_grp)

        sl.addWidget(QLabel(
            "Click on the preview to set a point as the new origin,\n"
            "or type x/y offset values directly."))

        self.btn_pick_origin = QPushButton("Click on preview to set origin")
        self.btn_pick_origin.setCheckable(True)
        self.btn_pick_origin.clicked.connect(self._on_toggle_pick_origin)
        sl.addWidget(self.btn_pick_origin)

        shift_val_row = QHBoxLayout()
        shift_val_row.addWidget(QLabel("X shift [mm]:"))
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(-5000.0, 5000.0)
        self.spin_dx.setValue(0.0)
        self.spin_dx.setDecimals(3)
        self.spin_dx.setSingleStep(0.1)
        self.spin_dx.setFixedWidth(90)
        self.spin_dx.setToolTip("Subtract this value from all x coordinates.")
        shift_val_row.addWidget(self.spin_dx)
        shift_val_row.addSpacing(12)
        shift_val_row.addWidget(QLabel("Y shift [mm]:"))
        self.spin_dy = QDoubleSpinBox()
        self.spin_dy.setRange(-5000.0, 5000.0)
        self.spin_dy.setValue(0.0)
        self.spin_dy.setDecimals(3)
        self.spin_dy.setSingleStep(0.1)
        self.spin_dy.setFixedWidth(90)
        self.spin_dy.setToolTip("Subtract this value from all y coordinates.")
        shift_val_row.addWidget(self.spin_dy)
        shift_val_row.addStretch()
        sl.addLayout(shift_val_row)

        self.btn_apply_shift = QPushButton("Apply Shift")
        self.btn_apply_shift.clicked.connect(self._on_apply_shift)
        sl.addWidget(self.btn_apply_shift)

        ll.addWidget(shift_grp)
        ll.addWidget(_hline())

        # ----------------------------------------------------------------
        # Transform history
        # ----------------------------------------------------------------
        self.lbl_history = QLabel("No transforms applied.")
        self.lbl_history.setStyleSheet(
            "font-size:10px; color:#aaa; padding:2px;")
        self.lbl_history.setWordWrap(True)
        ll.addWidget(self.lbl_history)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

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
        U_mean = np.nanmean(ds["U"], axis=2)
        vf     = np.mean(ds["valid"], axis=2)
        U_mean[vf < 0.5] = np.nan

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
        """Fill shift spinboxes from a click and show dot on preview."""
        self.spin_dx.setValue(float(x_click))
        self.spin_dy.setValue(float(y_click))

        # Update dot
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
        self.lbl_status.setText(
            f"Origin picked at ({x_click:.2f}, {y_click:.2f}) mm. "
            "Click 'Apply Shift' when ready.")

    def _on_toggle_pick_origin(self, checked):
        self._pick_origin_mode = checked
        if checked:
            self.btn_pick_origin.setText("Click preview to pick origin  (active)")
            self.btn_pick_origin.setStyleSheet("background:#1a4f1a;")
            self.lbl_status.setText("Click on the preview to set the new origin.")
        else:
            self.btn_pick_origin.setText("Click on preview to set origin")
            self.btn_pick_origin.setStyleSheet("")
            self.lbl_status.setText("Ready.")

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
        dx = self.spin_dx.value()
        dy = self.spin_dy.value()

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
        self.spin_dx.setValue(0.0)
        self.spin_dy.setValue(0.0)

        # Deactivate origin pick mode
        self.btn_pick_origin.setChecked(False)
        self._on_toggle_pick_origin(False)

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
