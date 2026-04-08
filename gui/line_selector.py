"""
gui/line_selector.py
--------------------
Reusable line selection widget for Reynolds stress, TKE, spectral,
and anisotropy windows.

Supports three modes:
  - Free      : user draws any line, nearest grid point used
  - Horizontal: snaps to nearest grid row, extends x0..x1
  - Vertical  : snaps to nearest grid column, extends y0..y1

Live snap: line updates while dragging to show the snapped position.
Spatial averaging band (grid points) available for horizontal/vertical.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout, QLabel,
    QRadioButton, QButtonGroup, QSpinBox, QWidget
)


class LineSelectorWidget(QWidget):
    """
    A control panel widget for line selection mode and averaging options.
    Embed this in any analysis window that needs line profiles.
    """

    def __init__(self, show_avg=True, parent=None):
        super().__init__(parent)
        self.show_avg = show_avg
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Mode selector
        mode_grp = QGroupBox("Line Mode")
        mode_lay = QHBoxLayout(mode_grp)

        self.rb_free  = QRadioButton("Free")
        self.rb_horiz = QRadioButton("Horizontal")
        self.rb_vert  = QRadioButton("Vertical")
        self.rb_free.setChecked(True)

        self._btn_grp = QButtonGroup()
        for rb in [self.rb_free, self.rb_horiz, self.rb_vert]:
            self._btn_grp.addButton(rb)
            mode_lay.addWidget(rb)

        layout.addWidget(mode_grp)

        # Spatial averaging (only shown when show_avg=True)
        if self.show_avg:
            avg_grp = QGroupBox("Spatial Averaging (H/V only)")
            avg_lay = QHBoxLayout(avg_grp)
            avg_lay.addWidget(QLabel("±  grid pts:"))
            self.spin_avg = QSpinBox()
            self.spin_avg.setRange(0, 20)
            self.spin_avg.setValue(0)
            self.spin_avg.setToolTip(
                "Average this many grid points either side of the line.\n"
                "0 = no averaging. Only applies to Horizontal/Vertical modes."
            )
            avg_lay.addWidget(self.spin_avg)
            layout.addWidget(avg_grp)
        else:
            self.spin_avg = None

    def get_mode(self):
        if self.rb_horiz.isChecked():
            return "horizontal"
        elif self.rb_vert.isChecked():
            return "vertical"
        return "free"

    def get_avg_band(self):
        if self.spin_avg is None:
            return 0
        mode = self.get_mode()
        if mode == "free":
            return 0
        return self.spin_avg.value()

    def hint_text(self):
        mode = self.get_mode()
        if mode == "horizontal":
            return "Click and drag horizontally to set y position and x range."
        elif mode == "vertical":
            return "Click and drag vertically to set x position and y range."
        return "Click and drag to draw a free line."


def compute_snapped_line(x, y, x0, y0, x1, y1, mode):
    """
    Return the display coordinates of the snapped line for live preview.

    Returns (lx0, ly0, lx1, ly1) -- the line to draw on the axes.
    """
    ny, nx = x.shape

    if mode == "horizontal":
        row0  = int(np.argmin(np.abs(y[:, 0] - y0)))
        snap_y = y[row0, 0]
        # Limit x to the dragged range
        lx0 = min(x0, x1)
        lx1 = max(x0, x1)
        return lx0, snap_y, lx1, snap_y

    elif mode == "vertical":
        col0  = int(np.argmin(np.abs(x[0, :] - x0)))
        snap_x = x[0, col0]
        ly0 = min(y0, y1)
        ly1 = max(y0, y1)
        return snap_x, ly0, snap_x, ly1

    return x0, y0, x1, y1
