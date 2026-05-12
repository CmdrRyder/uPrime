"""
gui/mask_window.py
------------------
Interactive spatial mask editor for uPrime.

Allows the user to define mask regions using Rectangle, Polygon, Circle,
or Ellipse shapes drawn on a field preview.  Each shape becomes a mask
layer (inside or outside, conservative or aggressive snapping).  Layers
are composited with logical-OR to produce the final boolean mask.

The final mask can be applied to the session (ds["mask"]), or saved /
loaded as mask.npy + mask_layers.json.
"""

import json
import time
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGroupBox, QPushButton, QComboBox, QRadioButton,
    QButtonGroup, QScrollArea, QSizePolicy, QMessageBox,
    QFileDialog, QApplication, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath

from scipy import ndimage

from gui.arrow_toolbar import DrawAwareToolbar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FONT_AX   = 9
_FONT_TICK = 8
_FONT_LEG  = 8

_LAYER_COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

_BG_FIELDS = ["Velocity Magnitude", "Mean U", "Mean V"]

# Double-click detection threshold (seconds).  Any two consecutive
# left-clicks within this interval while drawing a polygon are treated
# as a double-click close gesture.
_DBLCLICK_SEC = 0.40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hline():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color:#555;")
    return f


# ---------------------------------------------------------------------------
# SnapDialog  — used by all shapes; asks only for grid snapping
# ---------------------------------------------------------------------------

class SnapDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grid Snapping")
        self.setFixedWidth(300)
        self.snap = "conservative"

        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(QLabel("Grid snapping:"))

        self.rb_cons = QRadioButton("Conservative (mask strictly inside)")
        self.rb_aggr = QRadioButton("Aggressive (expand by 1 cell)")
        self.rb_cons.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_cons)
        bg.addButton(self.rb_aggr)
        lay.addWidget(self.rb_cons)
        lay.addWidget(self.rb_aggr)

        btn_row = QHBoxLayout()
        btn_apply  = QPushButton("Apply")
        btn_cancel = QPushButton("Cancel")
        btn_apply.clicked.connect(self._on_apply)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

    def _on_apply(self):
        self.snap = "aggressive" if self.rb_aggr.isChecked() else "conservative"
        self.accept()


# ---------------------------------------------------------------------------
# FullDialog  — alias for SnapDialog (inside/outside now lives on left panel)
# ---------------------------------------------------------------------------

class FullDialog(SnapDialog):
    """Snap-only dialog for Polygon / Circle / Ellipse shapes.

    Inside/outside is read from the left-panel Mask Region radio buttons,
    not from this dialog.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Options")


# ---------------------------------------------------------------------------
# MaskWindow
# ---------------------------------------------------------------------------

class MaskWindow(QDialog):
    """
    Interactive mask editor.

    Parameters
    ----------
    dataset    : dict         -- shared dataset dict (modified in-place on Apply)
    main_window: QMainWindow  -- parent main window for status bar updates
    """

    def __init__(self, dataset, main_window=None, parent=None):
        super().__init__(parent)
        self.dataset     = dataset
        self.main_win    = main_window
        self.mask_layers = []           # list of layer dicts

        # Drawing state
        self._draw_mode     = "Rectangle"
        self._drawing       = False     # True while a shape is being drawn
        self._press_xy      = None      # press coords (rect / circle / ellipse)
        self._poly_verts    = []        # polygon vertices so far
        self._preview_patch = None      # live preview patch
        self._preview_line  = None      # polygon edge preview line

        # Polygon state
        self._last_poly_click_t = 0.0   # time-based double-click fallback
        self._poly_hint         = None  # canvas annotation shown while drawing

        # Mpl connection IDs
        self._cid_press   = None
        self._cid_motion  = None
        self._cid_release = None
        self._cid_key     = None

        self.setWindowTitle("Mask Editor")
        self.setMinimumSize(1100, 680)
        self.resize(1300, 760)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)

        self._build_ui()
        self._draw_background()
        self._connect_events()
        self._restore_from_dataset()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ================================================================
        # LEFT panel
        # ================================================================
        left = QWidget()
        left.setFixedWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        title = QLabel("Mask Editor")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        ll.addWidget(title)

        # -- Drawing mode --
        mode_grp = QGroupBox("Drawing Mode")
        mode_lay = QVBoxLayout(mode_grp)
        self._mode_bg = QButtonGroup(self)
        for shape in ["Rectangle", "Polygon", "Circle", "Ellipse"]:
            rb = QRadioButton(shape)
            if shape == "Rectangle":
                rb.setChecked(True)
            self._mode_bg.addButton(rb)
            mode_lay.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        ll.addWidget(mode_grp)

        # -- Mask Region (inside / outside) --
        region_grp = QGroupBox("Mask Region")
        region_lay = QVBoxLayout(region_grp)
        self._region_bg  = QButtonGroup(self)
        self.rb_inside   = QRadioButton("Inside drawn shape")
        self.rb_outside  = QRadioButton("Outside drawn shape")
        self.rb_inside.setChecked(True)
        self._region_bg.addButton(self.rb_inside)
        self._region_bg.addButton(self.rb_outside)
        region_lay.addWidget(self.rb_inside)
        region_lay.addWidget(self.rb_outside)
        ll.addWidget(region_grp)

        # -- Background field --
        bg_grp = QGroupBox("Background Field")
        bg_lay = QVBoxLayout(bg_grp)
        self.combo_bg = QComboBox()
        self.combo_bg.addItems(_BG_FIELDS)
        self.combo_bg.currentIndexChanged.connect(self._on_bg_changed)
        bg_lay.addWidget(self.combo_bg)
        ll.addWidget(bg_grp)

        # -- Layers list --
        layers_grp     = QGroupBox("Mask Layers")
        layers_grp_lay = QVBoxLayout(layers_grp)
        self._layers_scroll = QScrollArea()
        self._layers_scroll.setWidgetResizable(True)
        self._layers_scroll.setMinimumHeight(140)
        self._layers_inner  = QWidget()
        self._layers_layout = QVBoxLayout(self._layers_inner)
        self._layers_layout.setContentsMargins(2, 2, 2, 2)
        self._layers_layout.setSpacing(2)
        self._layers_layout.addStretch()
        self._layers_scroll.setWidget(self._layers_inner)
        layers_grp_lay.addWidget(self._layers_scroll)
        ll.addWidget(layers_grp)

        # -- Cancel Drawing (enabled only during active draw) --
        self.btn_cancel_draw = QPushButton("Cancel Drawing")
        self.btn_cancel_draw.setEnabled(False)
        self.btn_cancel_draw.clicked.connect(self._cancel_drawing)
        self.btn_cancel_draw.setStyleSheet(
            "QPushButton         { background:#3e1e1e; color:#f08080; }"
            "QPushButton:hover   { background:#552828; }"
            "QPushButton:disabled{ color:#555; background:#222; }")
        ll.addWidget(self.btn_cancel_draw)

        ll.addWidget(_hline())

        # -- Action buttons --
        self.btn_undo = QPushButton("Undo Last Layer")
        self.btn_undo.clicked.connect(self._undo_layer)
        ll.addWidget(self.btn_undo)

        self.btn_reset = QPushButton("Reset All Masks")
        self.btn_reset.clicked.connect(self._reset_masks)
        self.btn_reset.setStyleSheet(
            "QPushButton       { background:#3e1e1e; color:#f08080; }"
            "QPushButton:hover { background:#552828; }")
        ll.addWidget(self.btn_reset)

        self.btn_save = QPushButton("Save Mask")
        self.btn_save.clicked.connect(self._save_mask)
        ll.addWidget(self.btn_save)

        self.btn_load_mask = QPushButton("Load Mask")
        self.btn_load_mask.clicked.connect(self._load_mask)
        ll.addWidget(self.btn_load_mask)

        self.btn_apply = QPushButton("Apply to Session")
        self.btn_apply.clicked.connect(self._apply_to_session)
        self.btn_apply.setStyleSheet(
            "QPushButton       { background:#1e3e1e; color:#90d890; }"
            "QPushButton:hover { background:#285228; }")
        ll.addWidget(self.btn_apply)

        ll.addStretch(1)

        # ================================================================
        # RIGHT panel: canvas
        # ================================================================
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)

        self.fig    = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.toolbar = DrawAwareToolbar(self.canvas, self)
        rl.addWidget(self.toolbar)
        rl.addWidget(self.canvas)

        root.addWidget(left)
        root.addWidget(right, stretch=1)

    # -----------------------------------------------------------------------
    # Mode change
    # -----------------------------------------------------------------------

    def _on_mode_changed(self, checked):
        if not checked:
            return
        for btn in self._mode_bg.buttons():
            if btn.isChecked():
                self._draw_mode = btn.text()
                break

    # -----------------------------------------------------------------------
    # Region mode helper
    # -----------------------------------------------------------------------

    def _get_region_mode(self):
        """Return 'inside' or 'outside' from the left-panel radio buttons."""
        return "inside" if self.rb_inside.isChecked() else "outside"

    # -----------------------------------------------------------------------
    # Background field drawing
    # -----------------------------------------------------------------------

    def _on_bg_changed(self):
        self._draw_background()

    def _draw_background(self):
        ds         = self.dataset
        x, y       = ds["x"], ds["y"]
        field_name = self.combo_bg.currentText()

        from core.dataset_utils import get_masked
        invalid = ~ds["MASK"]

        if field_name == "Mean U":
            field = np.nanmean(get_masked(ds, "U"), axis=2)
            field[invalid] = np.nan
            cmap  = "RdBu_r"
            label = "Mean U [m/s]"
        elif field_name == "Mean V":
            field = np.nanmean(get_masked(ds, "V"), axis=2)
            field[invalid] = np.nan
            cmap  = "RdBu_r"
            label = "Mean V [m/s]"
        else:                                   # Velocity Magnitude
            um = np.nanmean(get_masked(ds, "U"), axis=2)
            vm = np.nanmean(get_masked(ds, "V"), axis=2)
            _W = get_masked(ds, "W")
            if _W is not None:
                field = np.sqrt(um**2 + vm**2 + np.nanmean(_W, axis=2)**2)
            else:
                field = np.sqrt(um**2 + vm**2)
            field[invalid] = np.nan
            cmap  = "viridis"
            label = "Velocity Magnitude [m/s]"

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        cf = self.ax.contourf(x, y, np.ma.masked_invalid(field), levels=50, cmap=cmap, extend="neither")
        self.fig.colorbar(cf, ax=self.ax, label=label, shrink=0.8)
        self.ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        self.ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        self.ax.set_title(f"Mask Editor  —  {field_name}", fontsize=_FONT_AX)
        self.ax.tick_params(labelsize=_FONT_TICK)
        self.ax.set_aspect("equal")
        self.fig.tight_layout(pad=0.5)

        # Stored artists for overlay management
        self._layer_patch_artists = []

        self.canvas.draw()
        self._redraw_layers()

    # -----------------------------------------------------------------------
    # Redraw layers (called after every layer change)
    # -----------------------------------------------------------------------

    def _redraw_layers(self):
        if not hasattr(self, "ax"):
            return

        # Remove previous layer patches
        for p in self._layer_patch_artists:
            try:
                p.remove()
            except Exception:
                pass
        self._layer_patch_artists = []

        # Draw each layer as a hatched outline patch (no filled grey overlay)
        for layer in self.mask_layers:
            p = self._make_patch(layer)
            if p is not None:
                self.ax.add_patch(p)
                self._layer_patch_artists.append(p)

        self.canvas.draw_idle()

    def _make_patch(self, layer):
        """Return a matplotlib Patch for the given layer dict."""
        shape = layer["shape"]
        color = layer["color"]
        kw    = dict(linewidth=1.5, edgecolor=color,
                     facecolor=color, alpha=0.25,
                     hatch="///", zorder=10)
        if shape == "rectangle":
            verts = layer["vertices"]
            xs    = [v[0] for v in verts]
            ys    = [v[1] for v in verts]
            return mpatches.Rectangle(
                (min(xs), min(ys)), max(xs) - min(xs), max(ys) - min(ys), **kw)
        if shape == "polygon":
            return mpatches.Polygon(layer["vertices"], closed=True, **kw)
        if shape == "circle":
            cx, cy = layer["center"]
            return mpatches.Circle((cx, cy), layer["radii"][0], **kw)
        if shape == "ellipse":
            cx, cy  = layer["center"]
            rx, ry  = layer["radii"]
            return mpatches.Ellipse((cx, cy), 2 * rx, 2 * ry, **kw)
        return None

    # -----------------------------------------------------------------------
    # Event connections
    # -----------------------------------------------------------------------

    def _connect_events(self):
        self._cid_press   = self.canvas.mpl_connect(
            "button_press_event",   self._on_press)
        self._cid_motion  = self.canvas.mpl_connect(
            "motion_notify_event",  self._on_motion)
        self._cid_release = self.canvas.mpl_connect(
            "button_release_event", self._on_release)
        self._cid_key     = self.canvas.mpl_connect(
            "key_press_event",      self._on_key)

    def _disconnect_events(self):
        for attr in ("_cid_press", "_cid_motion", "_cid_release", "_cid_key"):
            cid = getattr(self, attr, None)
            if cid is not None:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
                setattr(self, attr, None)

    # -----------------------------------------------------------------------
    # Mouse: press
    # -----------------------------------------------------------------------

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self._toolbar_active():
            return

        mode = self._draw_mode

        if mode == "Polygon":
            self._handle_polygon_press(event)
            return

        # Rectangle / Circle / Ellipse: initiate drag on left press
        if event.button != 1:
            return
        self._drawing  = True
        self._press_xy = (event.xdata, event.ydata)
        self.btn_cancel_draw.setEnabled(True)

    def _handle_polygon_press(self, event):
        """Route a press event during polygon drawing."""
        if event.button == 3:
            # Right-click: close if enough vertices, else cancel
            if self._drawing and len(self._poly_verts) >= 3:
                self._close_polygon()
            elif self._drawing:
                self._cancel_drawing()
            return

        if event.button != 1:
            return

        now     = time.monotonic()
        elapsed = now - self._last_poly_click_t
        self._last_poly_click_t = now

        # mpl dblclick flag OR time-based fallback (covers slow clickers and
        # platforms where dblclick is not reliably set by the Qt backend)
        is_dblclick = bool(getattr(event, "dblclick", False)) or (elapsed < _DBLCLICK_SEC)

        if not self._drawing:
            # Start a new polygon
            self._drawing    = True
            self._poly_verts = [(event.xdata, event.ydata)]
            self.btn_cancel_draw.setEnabled(True)
            self._show_poly_hint()
            return

        # Already drawing
        if is_dblclick and len(self._poly_verts) >= 2:
            # Double-click: close polygon.  The first click of the pair already
            # appended a vertex; do NOT append again to avoid a duplicate point.
            self._close_polygon()
        else:
            # Single click: add vertex
            self._poly_verts.append((event.xdata, event.ydata))
            self._update_polygon_preview(event.xdata, event.ydata)

    # -----------------------------------------------------------------------
    # Mouse: motion
    # -----------------------------------------------------------------------

    def _on_motion(self, event):
        if not self._drawing:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self._toolbar_active():
            self._cancel_drawing()
            return

        mode = self._draw_mode

        if mode == "Polygon":
            if self._poly_verts:
                self._update_polygon_preview(event.xdata, event.ydata)
            return

        if self._press_xy is None:
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        color   = self._next_color()

        self._remove_preview()

        if mode == "Rectangle":
            p = mpatches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=1.5, edgecolor=color,
                facecolor=color, alpha=0.12,
                linestyle="--", zorder=15)
            self.ax.add_patch(p)
            self._preview_patch = p

        elif mode == "Circle":
            r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            p = mpatches.Circle(
                (x0, y0), r,
                linewidth=1.5, edgecolor=color,
                facecolor=color, alpha=0.12,
                linestyle="--", zorder=15)
            self.ax.add_patch(p)
            self._preview_patch = p

        elif mode == "Ellipse":
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            rx = abs(x1 - x0) / 2
            ry = abs(y1 - y0) / 2
            p = mpatches.Ellipse(
                (cx, cy), 2 * rx, 2 * ry,
                linewidth=1.5, edgecolor=color,
                facecolor=color, alpha=0.12,
                linestyle="--", zorder=15)
            self.ax.add_patch(p)
            self._preview_patch = p

        self.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Mouse: release
    # -----------------------------------------------------------------------

    def _on_release(self, event):
        mode = self._draw_mode

        if mode == "Polygon":
            return      # polygon lifecycle handled in press / key

        if not self._drawing or self._press_xy is None:
            return
        if self._toolbar_active():
            self._cancel_drawing()
            return
        if event.button != 1:
            return
        if event.inaxes != self.ax or event.xdata is None:
            self._cancel_drawing()
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata

        self._remove_preview()
        self._drawing  = False
        self._press_xy = None
        self.btn_cancel_draw.setEnabled(False)

        mode_str = self._get_region_mode()
        color    = self._next_color()

        if mode == "Rectangle":
            dlg = SnapDialog(self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            snap  = dlg.snap
            verts = [
                [min(x0, x1), min(y0, y1)],
                [max(x0, x1), min(y0, y1)],
                [max(x0, x1), max(y0, y1)],
                [min(x0, x1), max(y0, y1)],
            ]
            layer = {
                "shape":         "rectangle",
                "vertices":      verts,
                "center":        None,
                "radii":         None,
                "mode":          mode_str,
                "snap":          snap,
                "color":         color,
                "computed_mask": self._compute_mask_for_layer(
                    "rectangle", verts=verts, center=None, radii=None,
                    mode=mode_str, snap=snap),
            }

        elif mode == "Circle":
            r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            if r < 1e-9:
                return
            dlg = FullDialog(self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            layer = {
                "shape":         "circle",
                "vertices":      None,
                "center":        [float(x0), float(y0)],
                "radii":         [float(r), float(r)],
                "mode":          mode_str,
                "snap":          dlg.snap,
                "color":         color,
                "computed_mask": self._compute_mask_for_layer(
                    "circle", verts=None,
                    center=[x0, y0], radii=[r, r],
                    mode=mode_str, snap=dlg.snap),
            }

        elif mode == "Ellipse":
            rx = abs(x1 - x0) / 2
            ry = abs(y1 - y0) / 2
            if rx < 1e-9 or ry < 1e-9:
                return
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            dlg = FullDialog(self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            layer = {
                "shape":         "ellipse",
                "vertices":      None,
                "center":        [float(cx), float(cy)],
                "radii":         [float(rx), float(ry)],
                "mode":          mode_str,
                "snap":          dlg.snap,
                "color":         color,
                "computed_mask": self._compute_mask_for_layer(
                    "ellipse", verts=None,
                    center=[cx, cy], radii=[rx, ry],
                    mode=mode_str, snap=dlg.snap),
            }

        else:
            return

        self.mask_layers.append(layer)
        self._rebuild_layers_list()
        self._redraw_layers()

    # -----------------------------------------------------------------------
    # Keyboard
    # -----------------------------------------------------------------------

    def _on_key(self, event):
        """mpl key_press_event — cancel drawing on Escape."""
        if event.key == "escape":
            self._cancel_drawing()

    def keyPressEvent(self, event):
        """Qt keyPressEvent — intercept Escape before QDialog can act on it."""
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_drawing()
            # Accept the event and return WITHOUT calling super(), which would
            # trigger QDialog.reject() and close the window.
            event.accept()
            return
        super().keyPressEvent(event)

    # -----------------------------------------------------------------------
    # Polygon helpers
    # -----------------------------------------------------------------------

    def _update_polygon_preview(self, cur_x, cur_y):
        """Redraw the live polygon preview (edges + fill to current cursor)."""
        if not self._poly_verts:
            return

        self._remove_preview()

        color = self._next_color()
        verts = self._poly_verts

        # Edge line: existing vertices + cursor position
        xs = [v[0] for v in verts] + [cur_x]
        ys = [v[1] for v in verts] + [cur_y]
        ln, = self.ax.plot(xs, ys, color=color, linewidth=1.5,
                           linestyle="--", marker="o", markersize=4, zorder=15)
        self._preview_line = ln

        # Filled polygon preview (needs ≥ 2 committed vertices)
        if len(verts) >= 2:
            fill_verts = list(verts) + [(cur_x, cur_y)]
            p = mpatches.Polygon(
                fill_verts, closed=True,
                linewidth=0, facecolor=color, alpha=0.12, zorder=14)
            self.ax.add_patch(p)
            self._preview_patch = p

        self.canvas.draw_idle()

    def _close_polygon(self):
        """Finalise the polygon: clean up preview, show SnapDialog, add layer."""
        verts = self._poly_verts
        if len(verts) < 3:
            self._cancel_drawing()
            return

        # Remove preview artists and hint before opening the dialog
        self._remove_preview()
        self._remove_poly_hint()
        self._drawing = False
        self.btn_cancel_draw.setEnabled(False)
        self._last_poly_click_t = 0.0

        # Draw a temporary closed-polygon outline so the user can see their shape
        closed_xs = [v[0] for v in verts] + [verts[0][0]]
        closed_ys = [v[1] for v in verts] + [verts[0][1]]
        (closed_line,) = self.ax.plot(
            closed_xs, closed_ys,
            color=self._next_color(), linewidth=1.5,
            linestyle="-", marker="o", markersize=4, zorder=15)
        self.canvas.draw()

        # Disconnect canvas events while the modal dialog is open so that stale
        # mouse/key events don't fire into the drawing handlers.
        self._disconnect_events()
        try:
            dlg = FullDialog(self)
            accepted = dlg.exec() == QDialog.DialogCode.Accepted
            snap_val = dlg.snap if accepted else None
        finally:
            # Always reconnect, even if the dialog raised an exception
            self._connect_events()

        # Remove the temporary outline regardless of dialog result
        try:
            closed_line.remove()
        except Exception:
            pass

        if not accepted:
            self._poly_verts = []
            self._redraw_layers()
            return

        mode_str = self._get_region_mode()
        color    = self._next_color()
        layer = {
            "shape":         "polygon",
            "vertices":      [[float(v[0]), float(v[1])] for v in verts],
            "center":        None,
            "radii":         None,
            "mode":          mode_str,
            "snap":          snap_val,
            "color":         color,
            "computed_mask": self._compute_mask_for_layer(
                "polygon", verts=verts, center=None, radii=None,
                mode=mode_str, snap=snap_val),
        }
        self._poly_verts = []
        self.mask_layers.append(layer)
        self._rebuild_layers_list()
        self._redraw_layers()

    # -----------------------------------------------------------------------
    # Cancel in-progress drawing
    # -----------------------------------------------------------------------

    def _cancel_drawing(self):
        self._drawing    = False
        self._press_xy   = None
        self._poly_verts = []
        self._last_poly_click_t = 0.0
        self.btn_cancel_draw.setEnabled(False)
        self._remove_preview()
        self._remove_poly_hint()
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _remove_preview(self):
        for attr in ("_preview_patch", "_preview_line"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _show_poly_hint(self):
        """Add a canvas annotation telling the user how to close the polygon."""
        if not hasattr(self, "ax"):
            return
        self._remove_poly_hint()
        self._poly_hint = self.ax.text(
            0.5, 0.02,
            "Double-click or right-click to close",
            transform=self.ax.transAxes,
            ha="center", va="bottom",
            fontsize=_FONT_TICK, color="white",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#2b2b2b", alpha=0.75),
            zorder=20)

    def _remove_poly_hint(self):
        """Remove the polygon-close hint annotation if present."""
        obj = getattr(self, "_poly_hint", None)
        if obj is not None:
            try:
                obj.remove()
            except Exception:
                pass
            self._poly_hint = None

    # -----------------------------------------------------------------------
    # Mask computation per layer
    # -----------------------------------------------------------------------

    def _compute_mask_for_layer(self, shape, verts, center, radii, mode, snap):
        ds     = self.dataset
        x_grid = ds["x"]        # (ny, nx)
        y_grid = ds["y"]
        ny, nx = x_grid.shape

        pts = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        if shape in ("rectangle", "polygon"):
            path = MplPath(np.array(verts, dtype=float))
            raw  = path.contains_points(pts).reshape(ny, nx)

        elif shape == "circle":
            cx, cy = center
            r      = radii[0]
            raw    = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= r ** 2

        elif shape == "ellipse":
            cx, cy = center
            rx, ry = radii
            raw    = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2 <= 1.0

        else:
            return np.zeros((ny, nx), dtype=bool)

        if mode == "inside":
            if snap == "aggressive":
                result = ndimage.binary_dilation(raw, iterations=1)
            else:
                result = raw
        else:   # outside
            if snap == "aggressive":
                result = ~ndimage.binary_erosion(raw, iterations=1)
            else:
                result = ~raw

        return result.astype(bool)

    def _compute_final_mask(self):
        if not self.mask_layers:
            ny, nx = self.dataset["y"].shape
            return np.zeros((ny, nx), dtype=bool)
        return np.logical_or.reduce(
            [layer["computed_mask"] for layer in self.mask_layers])

    # -----------------------------------------------------------------------
    # Colour for next layer
    # -----------------------------------------------------------------------

    def _next_color(self):
        return _LAYER_COLORS[len(self.mask_layers) % len(_LAYER_COLORS)]

    # -----------------------------------------------------------------------
    # Layers list widget
    # -----------------------------------------------------------------------

    def _rebuild_layers_list(self):
        # Remove every row widget (all items except the trailing stretch).
        # setParent(None) detaches immediately; deleteLater schedules cleanup.
        while self._layers_layout.count() > 1:
            item = self._layers_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        for i, layer in enumerate(self.mask_layers):
            row     = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(2, 1, 2, 1)
            row_lay.setSpacing(4)

            # Colour swatch
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background:{layer['color']};"
                "border:1px solid #888; border-radius:2px;")
            row_lay.addWidget(swatch)

            # Info text
            shape    = layer["shape"].capitalize()
            mode_str = layer["mode"]
            snap_str = layer["snap"]
            parts    = [f"#{i + 1}", shape, mode_str, snap_str]
            lbl = QLabel("  ".join(parts))
            lbl.setStyleSheet("font-size:10px;")
            row_lay.addWidget(lbl, 1)

            # Remove button
            btn = QPushButton("\u2715")
            btn.setFixedSize(22, 22)
            btn.setStyleSheet(
                "QPushButton       { color:#e06c75; font-size:12px; padding:0; }"
                "QPushButton:hover { background:#3e1e1e; }")
            btn.clicked.connect(lambda _, idx=i: self._remove_layer(idx))
            row_lay.addWidget(btn)

            self._layers_layout.insertWidget(
                self._layers_layout.count() - 1, row)

    # -----------------------------------------------------------------------
    # Layer operations
    # -----------------------------------------------------------------------

    def _sync_mask_to_dataset(self):
        """Push current layers back to dataset["MASK"] (non-destructive)."""
        ds = self.dataset
        if "MASK_LOADED" not in ds:
            return
        drawn = self._compute_final_mask()
        ds["MASK"]       = ds["MASK_LOADED"] & ~drawn
        ds["valid"]      = ds["MASK"]
        ds["valid_frac"] = ds["MASK"].astype(np.float32)

    def _remove_layer(self, idx):
        if 0 <= idx < len(self.mask_layers):
            self.mask_layers.pop(idx)
            self._rebuild_layers_list()
            self._redraw_layers()
            self._sync_mask_to_dataset()

    def _undo_layer(self):
        if self.mask_layers:
            self.mask_layers.pop()
            self._rebuild_layers_list()
            self._redraw_layers()
            self._sync_mask_to_dataset()

    def _reset_masks(self):
        if not self.mask_layers:
            return
        reply = QMessageBox.question(
            self, "Reset All Masks",
            "Remove all mask layers and reset to an empty mask?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.mask_layers = []
            self._rebuild_layers_list()
            self._redraw_layers()
            self._sync_mask_to_dataset()

    # -----------------------------------------------------------------------
    # Save mask
    # -----------------------------------------------------------------------

    def _save_mask(self):
        if not self.mask_layers:
            QMessageBox.information(self, "No Mask",
                "Define at least one mask layer before saving.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "mask",
            "NumPy files (*.npy);;All files (*)")
        if not path:
            return

        import os
        base      = os.path.splitext(path)[0]
        npy_path  = base + ".npy"
        json_path = base + "_layers.json"

        np.save(npy_path, self._compute_final_mask())

        json_layers = []
        for layer in self.mask_layers:
            d = {k: v for k, v in layer.items() if k != "computed_mask"}
            if d.get("vertices") is not None:
                d["vertices"] = [[float(x), float(y)] for x, y in d["vertices"]]
            if d.get("center") is not None:
                d["center"] = [float(d["center"][0]), float(d["center"][1])]
            if d.get("radii") is not None:
                d["radii"] = [float(r) for r in d["radii"]]
            json_layers.append(d)

        with open(json_path, "w") as fh:
            json.dump(json_layers, fh, indent=2)

        QMessageBox.information(self, "Saved",
            f"Mask saved:\n  {npy_path}\n  {json_path}")

    # -----------------------------------------------------------------------
    # Load mask
    # -----------------------------------------------------------------------

    def _load_mask(self):
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mask Layers", "",
            "JSON files (*_layers.json *.json);;All files (*)")
        if not json_path:
            return

        try:
            with open(json_path, "r") as fh:
                json_layers = json.load(fh)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        loaded = []
        for d in json_layers:
            try:
                shape  = d["shape"]
                verts  = d.get("vertices")
                center = d.get("center")
                radii  = d.get("radii")
                mode   = d.get("mode", "inside")
                snap   = d.get("snap", "conservative")
                color  = d.get("color",
                               _LAYER_COLORS[len(loaded) % len(_LAYER_COLORS)])

                computed = self._compute_mask_for_layer(
                    shape, verts=verts, center=center, radii=radii,
                    mode=mode, snap=snap)

                loaded.append({
                    "shape":         shape,
                    "vertices":      verts,
                    "center":        center,
                    "radii":         radii,
                    "mode":          mode,
                    "snap":          snap,
                    "color":         color,
                    "computed_mask": computed,
                })
            except Exception as e:
                QMessageBox.warning(self, "Layer Warning",
                    f"Skipping a layer due to error:\n{e}")

        if loaded:
            self.mask_layers = loaded
            self._rebuild_layers_list()
            self._redraw_layers()

    # -----------------------------------------------------------------------
    # Apply to Session
    # -----------------------------------------------------------------------

    def _apply_to_session(self):
        ds = self.dataset

        try:
            QApplication.processEvents()

            # drawn_mask: True = grid points the user wants to exclude
            drawn_mask = self._compute_final_mask()

            # New active mask = original file mask AND NOT the drawn exclusion
            ds["MASK"]  = ds["MASK_LOADED"] & ~drawn_mask
            ds["valid"] = ds["MASK"]          # keep 2D alias in sync
            ds["valid_frac"] = ds["MASK"].astype(np.float32)

            # Serialise layers (no computed_mask) so reopening can restore them
            json_layers = []
            for layer in self.mask_layers:
                d = {k: v for k, v in layer.items() if k != "computed_mask"}
                if d.get("vertices") is not None:
                    d["vertices"] = [[float(c[0]), float(c[1])]
                                     for c in d["vertices"]]
                if d.get("center") is not None:
                    d["center"] = [float(d["center"][0]), float(d["center"][1])]
                if d.get("radii") is not None:
                    d["radii"] = [float(r) for r in d["radii"]]
                json_layers.append(d)
            ds["mask_layers"] = json_layers

            n_layers  = len(self.mask_layers)
            n_masked  = int(np.sum(drawn_mask))
            n_total   = int(drawn_mask.size)
            pct       = 100.0 * n_masked / n_total if n_total > 0 else 0.0
            msg = (f"Mask applied: {n_layers} layer"
                   f"{'s' if n_layers != 1 else ''}, "
                   f"{pct:.1f}% of grid masked")

            if self.main_win is not None:
                try:
                    self.main_win._plot_field()
                    self.main_win.lbl_status.setText(msg)
                except Exception:
                    pass

            QMessageBox.information(self, "Applied", msg)

        finally:
            QApplication.processEvents()

    # -----------------------------------------------------------------------
    # Restore mask layers from a previously applied session mask
    # -----------------------------------------------------------------------

    def _restore_from_dataset(self):
        """Reload layers stored in ds['mask_layers'] (written by _apply_to_session)."""
        json_layers = self.dataset.get("mask_layers")
        if not json_layers:
            return

        loaded = []
        for d in json_layers:
            try:
                shape  = d["shape"]
                verts  = d.get("vertices")
                center = d.get("center")
                radii  = d.get("radii")
                mode   = d.get("mode", "inside")
                snap   = d.get("snap", "conservative")
                color  = d.get("color",
                               _LAYER_COLORS[len(loaded) % len(_LAYER_COLORS)])
                computed = self._compute_mask_for_layer(
                    shape, verts=verts, center=center, radii=radii,
                    mode=mode, snap=snap)
                loaded.append({
                    "shape":         shape,
                    "vertices":      verts,
                    "center":        center,
                    "radii":         radii,
                    "mode":          mode,
                    "snap":          snap,
                    "color":         color,
                    "computed_mask": computed,
                })
            except Exception:
                pass

        if loaded:
            self.mask_layers = loaded
            self._rebuild_layers_list()
            self._redraw_layers()
            # Re-apply layers so MASK reflects the restored state
            drawn = self._compute_final_mask()
            ds = self.dataset
            ds["MASK"]       = ds["MASK_LOADED"] & ~drawn
            ds["valid"]      = ds["MASK"]
            ds["valid_frac"] = ds["MASK"].astype(np.float32)

    # -----------------------------------------------------------------------
    # Toolbar active guard  (mirrors PickerMixin._toolbar_active)
    # -----------------------------------------------------------------------

    def _toolbar_active(self):
        try:
            return self.toolbar.mode.name not in ("NONE", "")
        except Exception:
            return str(self.toolbar.mode) != ""

    # -----------------------------------------------------------------------
    # Close: disconnect mpl events
    # -----------------------------------------------------------------------

    def closeEvent(self, event):
        self._disconnect_events()
        super().closeEvent(event)
