"""
gui/arrow_toolbar.py
--------------------
Toolbar mixin that adds draw-mode awareness to any analysis window.

Usage:
    - Call self._toolbar_mode() to check if zoom/pan is active
    - If active, suppress mouse draw events
    - Arrow tool concept: when no zoom/pan active, mouse events work normally
"""

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import numpy as np
from PyQt6.QtCore import QTimer


class DrawAwareToolbar(NavigationToolbar2QT):
    """
    Standard matplotlib toolbar that exposes whether zoom/pan is active,
    so drawing interactions can be suppressed when zoom/pan is on.
    """

    def is_draw_mode(self):
        """Return True if user can draw (zoom/pan NOT active)."""
        return self.mode.name in ("NONE", "") if hasattr(self.mode, "name") \
               else str(self.mode) == ""


class PickerMixin:
    """
    Mixin for analysis windows that adds:
      - (x, y, value) display on hover
      - Temporary red cross marker on left-click (disappears after 3s)
      - Vertical crosshair on line plots
    
    To use: call self._setup_picker(field_canvas, field_ax, result_ax, status_label)
    """

    def _setup_picker(self, field_canvas, field_ax,
                      result_canvas=None, result_ax=None,
                      status_label=None):
        self._pick_field_canvas  = field_canvas
        self._pick_field_ax      = field_ax
        self._pick_result_canvas = result_canvas
        self._pick_result_ax     = result_ax
        self._pick_status        = status_label
        self._pick_dot           = None
        self._pick_crosshair     = None

        self._dot_timer = QTimer()
        self._dot_timer.setSingleShot(True)
        self._dot_timer.timeout.connect(self._remove_pick_markers)

        field_canvas.mpl_connect("motion_notify_event", self._on_pick_hover)
        field_canvas.mpl_connect("button_press_event",  self._on_pick_click)
        if result_canvas is not None and result_ax is not None:
            result_canvas.mpl_connect("motion_notify_event", self._on_result_hover)
            result_canvas.mpl_connect("button_press_event",  self._on_result_click)

    def _toolbar_active(self, toolbar):
        """Check if zoom or pan is active on a toolbar."""
        try:
            return toolbar.mode.name not in ("NONE", "")
        except Exception:
            return str(toolbar.mode) != ""

    def _on_pick_hover(self, event):
        if event.inaxes != self._pick_field_ax:
            return
        x, y = event.xdata, event.ydata
        val_str = self._get_field_value(x, y)
        if self._pick_status:
            msg = f"x = {x:.2f} mm   y = {y:.2f} mm"
            if val_str:
                msg += f"   {val_str}"
            self._pick_status.setText(msg)

    def _on_pick_click(self, event):
        """Left click on field: mark with temporary red cross."""
        if event.button != 1 or event.inaxes != self._pick_field_ax:
            return
        # Only mark if we're not in zoom/pan mode
        if hasattr(self, "field_toolbar") and self._toolbar_active(self.field_toolbar):
            return

        self._remove_pick_markers()
        x, y = event.xdata, event.ydata
        dot, = self._pick_field_ax.plot(x, y, "r+", markersize=14,
                                         markeredgewidth=2, zorder=20)
        self._pick_dot = (dot, self._pick_field_ax, self._pick_field_canvas)
        self._pick_field_canvas.draw()
        val_str = self._get_field_value(x, y)
        if self._pick_status:
            msg = f"Picked: x = {x:.2f} mm   y = {y:.2f} mm"
            if val_str:
                msg += f"   {val_str}"
            self._pick_status.setText(msg)
        self._dot_timer.start(3000)

    def _get_field_value(self, x, y):
        """Look up the value of the plotted field at the nearest grid point."""
        if not hasattr(self, '_pick_field_ax') or self._pick_field_ax is None:
            return ''
        ax = self._pick_field_ax
        # Try to find a QuadContourSet (contourf) on the axes
        # and look up nearest grid value using stored _x, _y, field arrays
        if hasattr(self, '_x') and hasattr(self, '_y'):
            dist2 = (self._x - x)**2 + (self._y - y)**2
            r, c  = np.unravel_index(np.argmin(dist2), dist2.shape)
            # Look for any stored field data
            if hasattr(self, '_last_field_values') and self._last_field_values is not None:
                v = self._last_field_values[r, c]
                if np.isfinite(v):
                    return f'value = {v:.4f}'
        return ''

    def _on_result_hover(self, event):
        """Hover on result (line plot): show vertical crosshair + values."""
        if event.inaxes != self._pick_result_ax:
            return
        ax = self._pick_result_ax
        x  = event.xdata

        # Remove old crosshair
        for line in ax.get_lines():
            if getattr(line, "_is_crosshair", False):
                line.remove()

        vline = ax.axvline(x, color="red", linewidth=0.8,
                           linestyle="--", alpha=0.6, zorder=15)
        vline._is_crosshair = True
        self._pick_result_canvas.draw()

        # Show values of all plotted lines at this x
        parts = []
        for line in ax.get_lines():
            if getattr(line, "_is_crosshair", False):
                continue
            xd = np.array(line.get_xdata())
            yd = np.array(line.get_ydata())
            label = line.get_label()
            if len(xd) == 0 or label.startswith("_"):
                continue
            idx = np.argmin(np.abs(xd - x))
            parts.append(f"{label} = {yd[idx]:.4e}")

        if self._pick_status and parts:
            self._pick_status.setText("  |  ".join(parts))

    def _on_result_click(self, event):
        """Left click on result plot: mark with vertical line."""
        if event.button != 1 or event.inaxes != self._pick_result_ax:
            return
        if hasattr(self, "result_toolbar") and self._toolbar_active(self.result_toolbar):
            return
        # crosshair already drawn on hover, nothing extra needed

    def _remove_pick_markers(self):
        if self._pick_dot is not None:
            try:
                artist, ax, canvas = self._pick_dot
                artist.remove()
                canvas.draw()
            except Exception:
                pass
            self._pick_dot = None
