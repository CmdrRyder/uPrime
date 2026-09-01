"""
gui/probability_window.py
-------------------------
Probability Analysis module (v0.8.0).

Container ``ProbabilityWindow`` holds a QTabWidget with four tabs, each a class in
this file (mirroring the container/sub-tab pattern of gui/mean_convergence_tab.py):

    1. PDF                 -- probability density of a quantity over point/line/ROI/field
    2. Flow Direction      -- per-point forward/reverse probability map (FFP / RFP)
    3. Binary Space-Time   -- one spatial coordinate vs time, two-colour by direction (TR only)
    4. Quadrant            -- quadrant analysis of a fluctuation pair (Lu & Willmarth)

All heavy time-axis work runs in core/probability.py through ProbabilityWorker so the
maps stay memmap-safe and NaN-correct. The forward/reverse convention (FFP: P(q>0) vs
RFP: P(q<0)) is surfaced in every title, colorbar, status line and export header.
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QRadioButton, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox,
    QButtonGroup, QFileDialog, QProgressBar, QApplication, QSizePolicy,
    QSplitter, QLineEdit, QMessageBox,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRect
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from gui.line_selector import LineSelectorWidget, compute_snapped_line
from core.export import (export_2d_tecplot, _settings_header)
from core import probability as prob
from core.workers import ProbabilityWorker

_FONT_AX   = 9
_FONT_TICK = 8
_FONT_LEG  = 8

# State colour map for the binary space-time plot (§3.5 codes 0..3)
_STATE_COLORS = ["#000000", "#E8A85C", "#808080", "#D9D9D9"]
_STATE_CMAP   = ListedColormap(_STATE_COLORS)
_STATE_NORM   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], _STATE_CMAP.N)
_STATE_NAMES  = ["reverse", "forward", "indeterminate", "invalid"]


def _case_name():
    try:
        return QApplication.instance()._session_case_name or "Data_1"
    except AttributeError:
        return "Data_1"


# --------------------------------------------------------------------------- #
# Memmap-safe adapters so |V| and fluctuations feed the chunked core functions
# without materialising the whole array.
# --------------------------------------------------------------------------- #

class _Fluct:
    """field - per-point 2D mean, evaluated lazily on any spatial/time slice."""
    def __init__(self, base, mean2d):
        self.base = base
        self.mean = mean2d
        self.shape = base.shape
        self.dtype = np.float64

    def __getitem__(self, key):
        sub = np.asarray(self.base[key], dtype=np.float64)
        m = self.mean[key[:2]] if isinstance(key, tuple) else self.mean[key]
        if sub.ndim > np.ndim(m):
            m = np.asarray(m)[..., None]
        return sub - m


class _Vmag:
    """sqrt(u^2 + v^2 [+ w^2]) evaluated lazily on any slice."""
    def __init__(self, U, V, W=None):
        self.U, self.V, self.W = U, V, W
        self.shape = U.shape
        self.dtype = np.float64

    def __getitem__(self, key):
        u = np.asarray(self.U[key], dtype=np.float64)
        v = np.asarray(self.V[key], dtype=np.float64)
        s = u * u + v * v
        if self.W is not None:
            w = np.asarray(self.W[key], dtype=np.float64)
            s = s + w * w
        return np.sqrt(s)


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #

class ProbabilityWindow(QWidget):
    def __init__(self, dataset, is_time_resolved=False, fs=1.0, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.is_time_resolved = bool(is_time_resolved)
        self.fs = float(fs) if fs else 1.0
        self.x = dataset["x"]
        self.y = dataset["y"]
        self.is_stereo = bool(dataset.get("is_stereo", False))
        self.mask2d = dataset["MASK"] if dataset.get("mask_active", True) else None
        self.Nt = dataset["U"].shape[2]
        self._means = None       # lazy {'U','V','W'}

        self.setWindowTitle("Probability Analysis")
        self.resize(1700, 900)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        self.pdf_tab = PDFTab(self)
        self.dir_tab = FlowDirectionTab(self)
        self.bin_tab = BinarySpaceTimeTab(self)
        self.quad_tab = QuadrantTab(self)

        self.tabs.addTab(self.pdf_tab, "PDF")
        self.tabs.addTab(self.dir_tab, "Flow Direction")
        self.tabs.addTab(self.bin_tab, "Binary Space-Time")
        self.tabs.addTab(self.quad_tab, "Quadrant")

        if not self.is_time_resolved:
            self.tabs.setTabEnabled(2, False)

    # -- lazy shared temporal means (chunked) --
    def get_means(self):
        if self._means is None:
            ds = self.dataset
            self._means = {
                "U": self._chunk_mean(ds["U"]),
                "V": self._chunk_mean(ds["V"]),
                "W": self._chunk_mean(ds["W"]) if ds["W"] is not None else None,
            }
        return self._means

    def _chunk_mean(self, field):
        ny, nx, Nt = field.shape
        s = np.zeros((ny, nx)); n = np.zeros((ny, nx))
        for t0, t1 in prob._iter_chunks(Nt, prob.CHUNK_DEFAULT):
            blk = np.asarray(field[:, :, t0:t1], dtype=np.float64)
            valid = np.isfinite(blk)
            if self.mask2d is not None:
                valid &= self.mask2d[:, :, None]
            s += np.where(valid, blk, 0.0).sum(axis=2)
            n += valid.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            n[n == 0] = np.nan
            return s / n

    def component_field(self, name):
        """Return an array-like [ny,nx,Nt] for a quantity name. Fluctuations and
        |V| are lazy adapters so a memmapped dataset is never fully realised."""
        ds = self.dataset
        if name == "u":
            return ds["U"]
        if name == "v":
            return ds["V"]
        if name == "w":
            return ds["W"]
        if name == "|V|":
            return _Vmag(ds["U"], ds["V"], ds["W"])
        m = self.get_means()
        if name == "u'":
            return _Fluct(ds["U"], m["U"])
        if name == "v'":
            return _Fluct(ds["V"], m["V"])
        if name == "w'":
            return _Fluct(ds["W"], m["W"])
        raise ValueError(f"unknown quantity {name!r}")

    def background_2d(self, name):
        """A 2D context field for the preview panels."""
        m = self.get_means()
        if name in ("u", "u'"):
            return m["U"]
        if name in ("v", "v'"):
            return m["V"]
        if name in ("w", "w'"):
            return m["W"]
        base = np.sqrt(np.nan_to_num(m["U"]) ** 2 + np.nan_to_num(m["V"]) ** 2)
        if m["W"] is not None:
            base = np.sqrt(base ** 2 + np.nan_to_num(m["W"]) ** 2)
        base = base.copy()
        if self.mask2d is not None:
            base[~self.mask2d] = np.nan
        return base


# --------------------------------------------------------------------------- #
# Shared field-preview / interaction mixin
# --------------------------------------------------------------------------- #

class _FieldSelectionMixin(PickerMixin):
    """Field-preview canvas with Reynolds-parity selection (see
    gui/reynolds_window.py:461-553).

    Left-drag draws a line (Line mode) or an ROI (ROI mode); left-click picks a
    grid point (Point mode); **right-drag always draws an ROI** in every mode. The
    single field axes object is created once and reused across replots via
    ``ax.clear()`` -- never reassigned -- so the axes cached by
    ``PickerMixin._setup_picker`` never goes stale (the bug that silently killed
    drawing after a replot). Tabs pick the active mode with a Selection Mode radio
    group (``_make_selection_group``); ``self._select_mode`` is the single control
    for both what left-drag does and what region is sampled.
    """

    # identical live/committed ROI style to reynolds_window.py:485-488
    ROI_KW = dict(linewidth=1.5, edgecolor="#e8a000", facecolor="#ffe066",
                  alpha=0.25, linestyle="--", zorder=10)
    _MODE_LABELS = {"point": "Point", "line": "Line", "roi": "ROI"}

    # ------------------------------------------------------------------ setup
    def _init_field(self, width=430):
        self.field_fig = Figure(constrained_layout=True)
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Fixed)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        self.field_ax = self.field_fig.add_subplot(111)   # created ONCE, never reassigned
        self._field_width = width
        self._roi = None          # (r0,r1,c0,c1) index space
        self._point = None        # (r,c)
        self._line = None         # (x0,y0,x1,y1) mm, snapped
        self._press_xy = None
        self._press_btn = None
        self._roi_artist = None
        self._line_artist = None
        self._point_artist = None
        if not hasattr(self, "_select_mode"):
            self._select_mode = "whole"

    def _draw_background(self, bg2d, title=""):
        # Replot on the SAME axes (ax.clear); field_ax is never reassigned so the
        # picker axes reference stays valid. Set the canvas height from the data
        # aspect ratio so set_aspect('equal') fills the widget and clicks are not
        # swallowed by blank margins (event.inaxes==None dead zone).
        x, y = self.win.x, self.win.y
        x_ext = float(np.nanmax(x) - np.nanmin(x))
        y_ext = float(np.nanmax(y) - np.nanmin(y))
        ratio = (y_ext / x_ext) if x_ext > 0 else 0.6
        self.field_canvas.setFixedHeight(
            max(150, min(420, int(self._field_width * ratio) + 10)))
        self.field_ax.clear()
        self.field_ax.contourf(x, y, np.ma.masked_invalid(bg2d), levels=40,
                               cmap="RdBu_r")
        self.field_ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        self.field_ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        if title:
            self.field_ax.set_title(title, fontsize=_FONT_AX - 1)
        self.field_ax.set_aspect("equal")
        self.field_ax.tick_params(labelsize=_FONT_TICK)
        self._x = x
        self._y = y
        self._last_field_values = bg2d
        # ax.clear() dropped the artists; recreate them from stored selection state
        self._roi_artist = self._line_artist = self._point_artist = None
        self._reattach_artists()
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()

    def _connect_field_mouse(self):
        self.field_canvas.mpl_connect("button_press_event", self._sel_press)
        self.field_canvas.mpl_connect("motion_notify_event", self._sel_motion)
        self.field_canvas.mpl_connect("button_release_event", self._sel_release)

    def _drawing_active(self):
        # Suppress the PickerMixin red-cross whenever we own the left button.
        return self._select_mode in ("point", "line", "roi")

    def _rc_from_xy(self, xd, yd):
        c = int(np.argmin(np.abs(self.win.x[0, :] - xd)))
        r = int(np.argmin(np.abs(self.win.y[:, 0] - yd)))
        return r, c

    # -------------------------------------------------- Selection Mode group
    def _make_selection_group(self, modes, default, whole_label="Whole field"):
        grp = QGroupBox("Selection Mode")
        lay = QVBoxLayout(grp)
        self._sel_btns = {}
        self._sel_grp = QButtonGroup(grp)
        self._whole_label = whole_label
        for m in modes:
            label = whole_label if m == "whole" else self._MODE_LABELS[m]
            rb = QRadioButton(label)
            self._sel_btns[m] = rb
            self._sel_grp.addButton(rb)
            lay.addWidget(rb)
            rb.toggled.connect(self._sel_mode_changed)
        # Set the default without firing the handler (the field canvas may not
        # exist yet; the real visibility/hint is applied in the tab's preview setup).
        self._select_mode = default
        self._block_sel(True)
        self._sel_btns[default].setChecked(True)
        self._block_sel(False)
        return grp

    def _block_sel(self, on):
        for rb in self._sel_btns.values():
            rb.blockSignals(on)

    def _sel_mode_changed(self, *_):
        for m, rb in self._sel_btns.items():
            if rb.isChecked():
                self._select_mode = m
                break
        # changing mode clears the current selection + artists (reynolds:362)
        self._clear_all_artists()
        self._roi = self._point = self._line = None
        self._apply_line_visibility()
        self._update_hint()
        if hasattr(self, "_on_selection_reset"):
            self._on_selection_reset()

    def _apply_line_visibility(self):
        line_mode = (self._select_mode == "line")
        if hasattr(self, "line_sel"):
            self.line_sel.setVisible(line_mode)
        if hasattr(self, "manual_grp"):
            self.manual_grp.setVisible(line_mode)

    def _set_mode_no_clear(self, mode):
        """Switch the radio + mode WITHOUT clearing the current selection (used
        when a right-drag ROI in Whole mode should reflect itself in the UI)."""
        self._select_mode = mode
        self._block_sel(True)
        self._sel_btns[mode].setChecked(True)
        self._block_sel(False)
        self._apply_line_visibility()

    def _update_hint(self):
        m = self._select_mode
        if m == "point":
            t = "Left-click to pick a grid point.  Right-click+drag for ROI."
        elif m == "line":
            t = "Left-click+drag to draw a line.  Right-click+drag for ROI."
        elif m == "roi":
            t = "Left-click+drag to draw an ROI.  Right-click+drag also works."
        else:
            t = "Sampling the whole field.  Right-click+drag for ROI."
        self._set_hint(t)

    def _set_hint(self, text):
        lbl = getattr(self, "lbl_hint", None) or getattr(self, "lbl_status", None)
        if lbl is not None:
            lbl.setText(text)

    # ---------------------------------------------------------- mouse handlers
    def _sel_press(self, event):
        if event.inaxes is not self.field_ax:
            return
        if self._toolbar_active(self.field_toolbar):
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button in (1, 3):
            self._press_xy = (event.xdata, event.ydata)
            self._press_btn = event.button

    def _sel_motion(self, event):
        if self._press_xy is None or event.inaxes is not self.field_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None
            return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        if self._press_btn == 3 or (self._press_btn == 1 and self._select_mode == "roi"):
            self._preview_roi(x0, y0, x1, y1)
        elif self._press_btn == 1 and self._select_mode == "line":
            lmode = self.line_sel.get_mode()
            lx0, ly0, lx1, ly1 = compute_snapped_line(
                self._x, self._y, x0, y0, x1, y1, lmode)
            self._preview_line(lx0, ly0, lx1, ly1)

    def _sel_release(self, event):
        if self._press_xy is None:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None
            return
        x0, y0 = self._press_xy
        x1 = event.xdata if event.xdata is not None else x0
        y1 = event.ydata if event.ydata is not None else y0
        btn = self._press_btn
        self._press_xy = None
        self._press_btn = None
        if btn == 3 or (btn == 1 and self._select_mode == "roi"):
            self._commit_roi(x0, y0, x1, y1, from_right=(btn == 3))
        elif btn == 1 and self._select_mode == "line":
            self._commit_line(x0, y0, x1, y1)
        elif btn == 1 and self._select_mode == "point":
            self._commit_point(x1, y1)

    # ------------------------------------------------------------- previews
    def _preview_roi(self, x0, y0, x1, y1):
        self._clear_roi_artist()
        p = MplRect((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
                    **self.ROI_KW)
        self.field_ax.add_patch(p)
        self._roi_artist = p
        self.field_canvas.draw()

    def _preview_line(self, lx0, ly0, lx1, ly1):
        self._clear_line_artist()
        ln, = self.field_ax.plot([lx0, lx1], [ly0, ly1], "r-", linewidth=2,
                                 zorder=10)
        self._line_artist = ln
        self.field_canvas.draw()

    # -------------------------------------------------------------- commits
    def _commit_roi(self, x0, y0, x1, y1, from_right=False):
        self._clear_roi_artist()
        if abs(x1 - x0) < 0.5 or abs(y1 - y0) < 0.5:
            self._set_hint("ROI too small -- try again.")
            self.field_canvas.draw()
            return
        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)
        r0, c0 = self._rc_from_xy(xlo, ylo)
        r1, c1 = self._rc_from_xy(xhi, yhi)
        self._roi = (min(r0, r1), max(r0, r1), min(c0, c1), max(c0, c1))
        p = MplRect((xlo, ylo), xhi - xlo, yhi - ylo, **self.ROI_KW)
        self.field_ax.add_patch(p)
        self._roi_artist = p
        self.field_canvas.draw()
        # right-drag ROI while in Whole mode -> reflect it in the radio (no clear)
        if (from_right and self._select_mode == "whole"
                and getattr(self, "_sel_btns", {}).get("roi") is not None):
            self._set_mode_no_clear("roi")
        self._set_hint(f"ROI: x=[{xlo:.1f},{xhi:.1f}]  y=[{ylo:.1f},{yhi:.1f}] mm")
        if hasattr(self, "_on_roi_committed"):
            self._on_roi_committed()

    def _commit_line(self, x0, y0, x1, y1):
        lmode = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(self._x, self._y, x0, y0, x1, y1,
                                                  lmode)
        if abs(lx1 - lx0) < 0.1 and abs(ly1 - ly0) < 0.1:
            self._set_hint("Line too short -- try again.")
            return
        self._set_line(lx0, ly0, lx1, ly1)

    def _commit_point(self, xd, yd):
        r, c = self._rc_from_xy(xd, yd)
        self._point = (r, c)
        self._clear_point_artist()
        self._point_artist, = self.field_ax.plot(
            self.win.x[0, c], self.win.y[r, 0], "r+", markersize=14,
            markeredgewidth=2, zorder=20)
        self.field_canvas.draw()
        self._set_hint(f"Point: r={r} c={c}  "
                       f"({self.win.x[0, c]:.2f}, {self.win.y[r, 0]:.2f}) mm")
        if hasattr(self, "_on_point_committed"):
            self._on_point_committed()

    def _set_line(self, x0, y0, x1, y1, sync_spins=True):
        """Single downstream path shared by drawn AND manual lines."""
        self._line = (x0, y0, x1, y1)
        self._clear_line_artist()
        ln, = self.field_ax.plot([x0, x1], [y0, y1], "r-", linewidth=2, zorder=10)
        self._line_artist = ln
        self.field_canvas.draw()
        if sync_spins and hasattr(self, "spin_x0"):
            for sp, v in ((self.spin_x0, x0), (self.spin_y0, y0),
                          (self.spin_x1, x1), (self.spin_y1, y1)):
                sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
        self._set_hint(f"Line: ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f}) mm")
        if hasattr(self, "_on_line_committed"):
            self._on_line_committed()

    def _apply_manual_line(self):
        x0, y0, x1, y1 = self._manual_line()
        lmode = self.line_sel.get_mode() if hasattr(self, "line_sel") else "free"
        lx0, ly0, lx1, ly1 = compute_snapped_line(self.win.x, self.win.y,
                                                  x0, y0, x1, y1, lmode)
        if abs(lx1 - lx0) < 0.1 and abs(ly1 - ly0) < 0.1:
            self._set_hint("Line too short -- try again.")
            return
        self._set_line(lx0, ly0, lx1, ly1, sync_spins=False)

    # -------------------------------------------------------------- artists
    def _clear_roi_artist(self):
        if self._roi_artist is not None:
            try:
                self._roi_artist.remove()
            except Exception:
                pass
            self._roi_artist = None

    def _clear_line_artist(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None

    def _clear_point_artist(self):
        if self._point_artist is not None:
            try:
                self._point_artist.remove()
            except Exception:
                pass
            self._point_artist = None

    def _clear_all_artists(self):
        self._clear_roi_artist()
        self._clear_line_artist()
        self._clear_point_artist()
        self.field_canvas.draw()

    def _reattach_artists(self):
        """Redraw persistent selection artists from stored state (after ax.clear)."""
        if self._roi is not None:
            r0, r1, c0, c1 = self._roi
            xlo, xhi = self.win.x[0, c0], self.win.x[0, c1]
            ylo, yhi = self.win.y[r0, 0], self.win.y[r1, 0]
            p = MplRect((min(xlo, xhi), min(ylo, yhi)), abs(xhi - xlo),
                        abs(yhi - ylo), **self.ROI_KW)
            self.field_ax.add_patch(p)
            self._roi_artist = p
        if self._point is not None:
            r, c = self._point
            self._point_artist, = self.field_ax.plot(
                self.win.x[0, c], self.win.y[r, 0], "r+", markersize=14,
                markeredgewidth=2, zorder=20)
        if self._line is not None:
            x0, y0, x1, y1 = self._line
            self._line_artist, = self.field_ax.plot([x0, x1], [y0, y1], "r-",
                                                    linewidth=2, zorder=10)

    def clear_roi(self):
        self._roi = None
        self._clear_roi_artist()
        self.field_canvas.draw()

    def _redraw_overlays(self):
        self._clear_all_artists()
        self._reattach_artists()
        self.field_canvas.draw()

    # -------------------- ROI-as-region helpers (restrict the computation) ----
    def _roi_region(self):
        """Region for the worker: ('roi', r0,r1,c0,c1) in ROI mode, else None
        (whole domain). The stored ``_roi`` is already inclusive grid indices."""
        if self._select_mode == "roi" and self._roi is not None:
            return ("roi", *self._roi)
        return None

    def _region_desc(self, region):
        """Status text: ROI extent in mm AND grid indices + points computed."""
        if region is None:
            return "whole domain"
        _, r0, r1, c0, c1 = region
        x = self.win.x; y = self.win.y
        npts = (r1 - r0 + 1) * (c1 - c0 + 1)
        return (f"ROI: x=[{x[0, c0]:.1f},{x[0, c1]:.1f}] "
                f"y=[{y[r0, 0]:.1f},{y[r1, 0]:.1f}] mm  "
                f"(rows {r0}-{r1}, cols {c0}-{c1}, {npts} pts)")

    def _region_header(self, region):
        """Region fields for an export settings header."""
        if region is None:
            return {"region": "whole domain"}
        _, r0, r1, c0, c1 = region
        x = self.win.x; y = self.win.y
        return {"region": "ROI",
                "roi_mm": (f"x=[{x[0, c0]:.3f},{x[0, c1]:.3f}] "
                           f"y=[{y[r0, 0]:.3f},{y[r1, 0]:.3f}]"),
                "roi_grid": f"rows {r0}-{r1}, cols {c0}-{c1}",
                "roi_points": (r1 - r0 + 1) * (c1 - c0 + 1)}

    def _shade_outside_roi(self, ax, region):
        """Grey the four bands outside the ROI (not computed), plus the dashed
        ROI outline, on a full-domain map. Does nothing for the whole domain."""
        if region is None:
            return
        _, r0, r1, c0, c1 = region
        x = self.win.x; y = self.win.y
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        dx = abs(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 0.0
        dy = abs(y[1, 0] - y[0, 0]) if y.shape[0] > 1 else 0.0
        xl, xr = x[0, c0] - dx / 2, x[0, c1] + dx / 2
        yb, yt = y[r0, 0] - dy / 2, y[r1, 0] + dy / 2
        grey = dict(facecolor="#BFBFBF", edgecolor="none", zorder=5)
        ax.add_patch(mpatches.Rectangle((xmin, ymin), xl - xmin, ymax - ymin, **grey))
        ax.add_patch(mpatches.Rectangle((xr, ymin), xmax - xr, ymax - ymin, **grey))
        ax.add_patch(mpatches.Rectangle((xl, ymin), xr - xl, yb - ymin, **grey))
        ax.add_patch(mpatches.Rectangle((xl, yt), xr - xl, ymax - yt, **grey))
        ax.add_patch(mpatches.Rectangle((xl, yb), xr - xl, yt - yb, fill=False,
                     edgecolor="#e8a000", linestyle="--", linewidth=1.5, zorder=10))

    # ------------------------ manual line entry group (x0/y0/x1/y1 + Set line)
    def _build_manual_group(self, with_draw=True):
        grp = QGroupBox("Line Entry (x0, y0, x1, y1)")
        lay = QVBoxLayout(grp)
        cl = QHBoxLayout()
        self.spin_x0 = QDoubleSpinBox(); self.spin_y0 = QDoubleSpinBox()
        self.spin_x1 = QDoubleSpinBox(); self.spin_y1 = QDoubleSpinBox()
        for sp in (self.spin_x0, self.spin_y0, self.spin_x1, self.spin_y1):
            sp.setDecimals(3); sp.setRange(-1e6, 1e6)
        xmin = float(np.nanmin(self.win.x)); xmax = float(np.nanmax(self.win.x))
        ymin = float(np.nanmin(self.win.y)); ymax = float(np.nanmax(self.win.y))
        self.spin_x0.setValue(xmin); self.spin_x1.setValue(xmax)
        self.spin_y0.setValue((ymin + ymax) / 2); self.spin_y1.setValue((ymin + ymax) / 2)
        for lbl, sp in (("x0:", self.spin_x0), ("y0:", self.spin_y0),
                        ("x1:", self.spin_x1), ("y1:", self.spin_y1)):
            cl.addWidget(QLabel(lbl)); cl.addWidget(sp)
        lay.addLayout(cl)
        self.btn_set_line = QPushButton("Set line")
        self.btn_set_line.clicked.connect(self._apply_manual_line)
        lay.addWidget(self.btn_set_line)
        return grp

    def _manual_line(self):
        return (self.spin_x0.value(), self.spin_y0.value(),
                self.spin_x1.value(), self.spin_y1.value())


# --------------------------------------------------------------------------- #
# Tab 1 -- PDF
# --------------------------------------------------------------------------- #

class PDFTab(_FieldSelectionMixin, QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._result = None
        self._preview_ready = False
        self._build_ui()

    def showEvent(self, e):
        super().showEvent(e)
        self._ensure_preview()

    def _ensure_preview(self):
        if self._preview_ready:
            return
        self._preview_ready = True
        self._init_field()
        self._connect_field_mouse()
        self._draw_background(self.win.background_2d(self.combo_q.currentText()),
                             "Sample source preview")
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)
        self._apply_line_visibility()
        self._update_hint()

    def _quantities(self):
        q = ["u", "v", "|V|", "u'", "v'"]
        if self.win.is_stereo:
            q = ["u", "v", "w", "|V|", "u'", "v'", "w'"]
        return q

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget(); left.setMinimumWidth(430); left.setMaximumWidth(540)
        ll = QVBoxLayout(left)
        self._left_layout = ll

        row = QHBoxLayout(); row.addWidget(QLabel("Quantity:"))
        self.combo_q = QComboBox(); self.combo_q.addItems(self._quantities())
        self.combo_q.currentTextChanged.connect(self._on_quantity)
        row.addWidget(self.combo_q); ll.addLayout(row)

        ll.addWidget(self._make_selection_group(
            ["point", "line", "roi", "whole"], "whole", "Whole field"))
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        self.line_sel = LineSelectorWidget(show_avg=False, show_free=False)
        self.line_sel.setVisible(False)
        ll.addWidget(self.line_sel)
        self.manual_grp = self._build_manual_group()
        self.manual_grp.setVisible(False)
        ll.addWidget(self.manual_grp)

        opt = QGroupBox("Histogram"); ol = QVBoxLayout(opt)
        r1 = QHBoxLayout(); r1.addWidget(QLabel("Bins:"))
        self.spin_bins = QSpinBox(); self.spin_bins.setRange(11, 501)
        self.spin_bins.setValue(101); r1.addWidget(self.spin_bins); ol.addLayout(r1)
        self.chk_auto = QCheckBox("Auto (robust) range"); self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self._on_auto); ol.addWidget(self.chk_auto)
        r2 = QHBoxLayout()
        self.spin_min = QDoubleSpinBox(); self.spin_max = QDoubleSpinBox()
        for sp in (self.spin_min, self.spin_max):
            sp.setDecimals(4); sp.setRange(-1e6, 1e6); sp.setEnabled(False)
        self.spin_max.setValue(1.0)
        r2.addWidget(QLabel("min:")); r2.addWidget(self.spin_min)
        r2.addWidget(QLabel("max:")); r2.addWidget(self.spin_max); ol.addLayout(r2)
        nrm = QHBoxLayout()
        self.rb_density = QRadioButton("Density"); self.rb_counts = QRadioButton("Counts")
        self.rb_density.setChecked(True)
        self._nrm_grp = QButtonGroup(opt)
        self._nrm_grp.addButton(self.rb_density); self._nrm_grp.addButton(self.rb_counts)
        nrm.addWidget(self.rb_density); nrm.addWidget(self.rb_counts); ol.addLayout(nrm)
        self.chk_gauss = QCheckBox("Gaussian overlay"); self.chk_gauss.setChecked(True)
        ol.addWidget(self.chk_gauss)
        self.chk_logy = QCheckBox("Log y"); ol.addWidget(self.chk_logy)
        ll.addWidget(opt)

        self.btn_compute = QPushButton("Compute PDF")
        self.btn_compute.clicked.connect(self._compute)
        ll.addWidget(self.btn_compute)
        self.progress = QProgressBar(); self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False); self.progress.setVisible(False)
        ll.addWidget(self.progress)
        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.clicked.connect(self._export); self.btn_export.setEnabled(False)
        ll.addWidget(self.btn_export)
        self.lbl_status = QLabel(""); self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_status)
        ll.addStretch()

        right = QWidget(); rl = QVBoxLayout(right)
        self.result_fig = Figure(); self.result_canvas = FigureCanvas(self.result_fig)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        rl.addWidget(self.result_canvas, stretch=1)
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("font-family: monospace; font-size: 11px;")
        rl.addWidget(self.lbl_stats)

        # Put the field preview under the controls on the left.
        # (created in _init_field, added here after)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([500, 1200])

    def field_canvas_holder(self):
        return None

    def _init_field(self, width=430):
        super()._init_field(width)
        # field preview goes at the top of the left column
        self._left_layout.insertWidget(0, self.field_toolbar)
        self._left_layout.insertWidget(1, self.field_canvas)

    # -- interaction wiring --
    def _on_quantity(self, *_):
        if not self._preview_ready:
            return
        try:
            self._draw_background(self.win.background_2d(self.combo_q.currentText()),
                                  "Sample source preview")
            self._redraw_overlays()
        except Exception:
            pass

    def _on_auto(self, on):
        self.spin_min.setEnabled(not on)
        self.spin_max.setEnabled(not on)

    def _current_region(self):
        """Region + description from the single Selection Mode control and the
        stored selection (point/ROI drawn on the field, line drawn or entered)."""
        m = self._select_mode
        if m == "point":
            if self._point is None:
                return None, "point (none picked)"
            return ("point", self._point[0], self._point[1]), \
                   f"point r={self._point[0]} c={self._point[1]}"
        if m == "roi":
            if self._roi is None:
                return None, "ROI (none drawn)"
            r0, r1, c0, c1 = self._roi
            return ("roi", r0, r1, c0, c1), f"ROI rows {r0}:{r1} cols {c0}:{c1}"
        if m == "line":
            if self._line is None:
                return None, "line (none set)"
            x0, y0, x1, y1 = self._line
            lmode = self.line_sel.get_mode()
            direction = "x" if lmode == "horizontal" else (
                "y" if lmode == "vertical" else None)
            rows, cols = self._line_indices(x0, y0, x1, y1, direction)
            return ("index", rows, cols), f"line ({len(rows)} pts)"
        return None, "whole field"

    def _line_indices(self, x0, y0, x1, y1, direction):
        x = self.win.x; y = self.win.y; ny, nx = x.shape
        if direction == "x":
            r = int(np.argmin(np.abs(y[:, 0] - y0)))
            cols = np.where((x[0, :] >= min(x0, x1)) & (x[0, :] <= max(x0, x1)))[0]
            if cols.size == 0:
                cols = np.arange(nx)
            return np.full(cols.size, r), cols
        if direction == "y":
            c = int(np.argmin(np.abs(x[0, :] - x0)))
            rows = np.where((y[:, 0] >= min(y0, y1)) & (y[:, 0] <= max(y0, y1)))[0]
            if rows.size == 0:
                rows = np.arange(ny)
            return rows, np.full(rows.size, c)
        # free line: march grid points nearest to the segment
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) /
                       max(abs(x[0, 1] - x[0, 0]), 1e-9)) + 1)
        xs = np.linspace(x0, x1, n); ys = np.linspace(y0, y1, n)
        rows = np.array([int(np.argmin(np.abs(y[:, 0] - yy))) for yy in ys])
        cols = np.array([int(np.argmin(np.abs(x[0, :] - xx))) for xx in xs])
        keep = np.concatenate([[True], (np.diff(rows) != 0) | (np.diff(cols) != 0)])
        return rows[keep], cols[keep]

    def _compute(self):
        region, desc = self._current_region()
        if region is None and self._select_mode != "whole":
            self.lbl_status.setText(f"No sample selected ({desc}).")
            return
        qty = self.combo_q.currentText()
        field = self.win.component_field(qty)
        edges = None
        if not self.chk_auto.isChecked():
            lo, hi = self.spin_min.value(), self.spin_max.value()
            if hi <= lo:
                self.lbl_status.setText("Manual range needs max > min.")
                return
            edges = np.linspace(lo, hi, self.spin_bins.value() + 1)
        self.btn_compute.setEnabled(False)
        self.progress.setVisible(True); self.progress.setRange(0, 100)
        self.lbl_status.setText(f"Computing PDF over {desc}…")
        self._worker = ProbabilityWorker(
            "histogram", field=field, mask_2d=self.win.mask2d, region=region,
            nbins=self.spin_bins.value(), robust=self.chk_auto.isChecked(),
            bin_edges=edges)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(lambda r: self._on_result(r, desc, qty))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_error(self, tb):
        self.btn_compute.setEnabled(True); self.progress.setVisible(False)
        QMessageBox.critical(self, "PDF Error", tb)

    def _on_result(self, r, desc, qty):
        self._result = dict(r, desc=desc, quantity=qty)
        self.btn_compute.setEnabled(True); self.progress.setVisible(False)
        self.btn_export.setEnabled(True)
        self._plot()

    def _plot(self):
        r = self._result
        counts = r["counts"].astype(float); edges = r["bin_edges"]
        centres = 0.5 * (edges[:-1] + edges[1:]); widths = np.diff(edges)
        density = self.rb_density.isChecked()
        n = counts.sum()
        yvals = counts / (n * widths) if (density and n > 0) else counts
        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        ax.bar(centres, yvals, width=widths, color="#4C78A8",
               edgecolor="none", alpha=0.85)
        st = r["stats"]
        if self.chk_gauss.isChecked() and np.isfinite(st["std"]) and st["std"] > 0:
            g = (np.exp(-0.5 * ((centres - st["mean"]) / st["std"]) ** 2)
                 / (st["std"] * np.sqrt(2 * np.pi)))
            if not density:
                g = g * n * widths
            ax.plot(centres, g, "k--", linewidth=1.2, label="Gaussian")
            ax.legend(fontsize=_FONT_LEG)
        if self.chk_logy.isChecked():
            ax.set_yscale("log")
        ax.set_xlabel(f"{r['quantity']} [m/s]", fontsize=_FONT_AX)
        ax.set_ylabel("density" if density else "counts", fontsize=_FONT_AX)
        ax.set_title(f"PDF of {r['quantity']} -- {r['desc']}", fontsize=_FONT_AX)
        ax.set_aspect("auto")
        ax.tick_params(labelsize=_FONT_TICK)
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.lbl_stats.setText(
            f"N={st['n']}   mean={st['mean']:.4g}   std={st['std']:.4g}   "
            f"skewness={st['skewness']:.3g}   kurtosis (Gaussian = 3)={st['kurtosis']:.3g}   "
            f"min={st['min']:.4g}   max={st['max']:.4g}")

    def _export(self):
        if self._result is None:
            return
        qty = self._result["quantity"].replace("'", "p").replace("|", "")
        fn = f"{_case_name()}_probability_pdf_{qty}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", fn,
                                              "CSV Files (*.csv)")
        if not path:
            return
        r = self._result; counts = r["counts"].astype(float); edges = r["bin_edges"]
        centres = 0.5 * (edges[:-1] + edges[1:]); widths = np.diff(edges)
        n = counts.sum(); density = counts / (n * widths) if n > 0 else counts * 0
        st = r["stats"]
        info = {
            "module": "Probability / PDF", "quantity": r["quantity"],
            "sample_source": r["desc"], "n_valid": r["n_valid"],
            "bins": len(counts), "normalisation":
            "density" if self.rb_density.isChecked() else "counts",
            "mean": st["mean"], "std": st["std"], "skewness": st["skewness"],
            "kurtosis_gaussian_is_3": st["kurtosis"],
            "note": "per-point valid-sample counting; NaN excluded per point",
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(_settings_header(info)); f.write("\n")
            f.write("bin_center,bin_left,bin_right,count,density\n")
            for i in range(len(counts)):
                f.write(f"{centres[i]:.8g},{edges[i]:.8g},{edges[i+1]:.8g},"
                        f"{int(counts[i])},{density[i]:.8g}\n")
        self.lbl_status.setText(f"Saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# Tab 2 -- Flow Direction
# --------------------------------------------------------------------------- #

class FlowDirectionTab(_FieldSelectionMixin, QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._result = None
        self._preview_ready = False
        self._build_ui()

    def showEvent(self, e):
        super().showEvent(e)
        self._ensure_preview()

    def _ensure_preview(self):
        if self._preview_ready:
            return
        self._preview_ready = True
        self._init_field(width=420)
        self._connect_field_mouse()
        self._draw_background(self.win.background_2d("u"), "Selection preview")
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)
        self._apply_line_visibility()
        self._update_hint()

    def _components(self):
        return ["u", "v", "w"] if self.win.is_stereo else ["u", "v"]

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal); root.addWidget(splitter)
        left = QWidget(); left.setMinimumWidth(420); left.setMaximumWidth(540)
        ll = QVBoxLayout(left); self._left_layout = ll

        r = QHBoxLayout(); r.addWidget(QLabel("Component:"))
        self.combo_c = QComboBox(); self.combo_c.addItems(self._components())
        r.addWidget(self.combo_c); ll.addLayout(r)

        conv = QGroupBox("Convention"); cl = QVBoxLayout(conv)
        self.rb_ffp = QRadioButton("Forward (FFP): P = 1 where q > 0")
        self.rb_rfp = QRadioButton("Reverse (RFP): P = 1 where q < 0")
        self.rb_ffp.setChecked(True)
        self._conv_grp = QButtonGroup(conv)
        self._conv_grp.addButton(self.rb_ffp); self._conv_grp.addButton(self.rb_rfp)
        self.rb_ffp.toggled.connect(self._replot_same_result)
        cl.addWidget(self.rb_ffp); cl.addWidget(self.rb_rfp); ll.addWidget(conv)

        r2 = QHBoxLayout(); r2.addWidget(QLabel("Deadband ε [m/s]:"))
        self.spin_eps = QDoubleSpinBox(); self.spin_eps.setRange(0.0, 10.0)
        self.spin_eps.setDecimals(3); self.spin_eps.setValue(0.0)
        self.spin_eps.setToolTip("0 gives the strict sign test. A non-zero value "
                                 "creates a third 'indeterminate' state |q|<=ε.")
        r2.addWidget(self.spin_eps); ll.addLayout(r2)

        vr = QHBoxLayout(); vr.addWidget(QLabel("View:"))
        self.combo_view = QComboBox()
        self.combo_view.addItems(["Probability", "Indeterminate fraction",
                                  "Valid samples N"])
        self.combo_view.currentIndexChanged.connect(self._replot_same_result)
        vr.addWidget(self.combo_view); ll.addLayout(vr)

        self.chk_simpson = QCheckBox("Show 0.01 / 0.20 contours")
        self.chk_simpson.toggled.connect(self._replot_same_result)
        ll.addWidget(self.chk_simpson)

        # Flow Direction is a 2D-field map only: ROI (statistics readout) or the
        # whole domain. No line mode -- there is no line profile on this tab.
        ll.addWidget(self._make_selection_group(
            ["roi", "whole"], "whole", "Whole domain"))
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        ll.addWidget(self.btn_clear_roi)

        self.btn_compute = QPushButton("Compute")
        self.btn_compute.clicked.connect(self._compute); ll.addWidget(self.btn_compute)
        self.progress = QProgressBar(); self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False); self.progress.setVisible(False)
        ll.addWidget(self.progress)
        self.btn_export2d = QPushButton("Export 2D Tecplot...")
        self.btn_export2d.clicked.connect(self._export_2d); self.btn_export2d.setEnabled(False)
        ll.addWidget(self.btn_export2d)
        self.lbl_status = QLabel(""); self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_status); ll.addStretch()

        right = QWidget(); rl = QVBoxLayout(right)
        self.result_fig = Figure(); self.result_canvas = FigureCanvas(self.result_fig)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar); rl.addWidget(self.result_canvas, stretch=1)
        self.lbl_roi = QLabel(""); self.lbl_roi.setStyleSheet("font-size: 11px;")
        self.lbl_roi.setWordWrap(True); rl.addWidget(self.lbl_roi)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([520, 1180])

    def _init_field(self, width=420):
        super()._init_field(width)
        self._left_layout.insertWidget(0, self.field_toolbar)
        self._left_layout.insertWidget(1, self.field_canvas)

    def _clear_roi(self):
        self.clear_roi(); self._update_roi_readout(); self._replot_same_result()

    # selection hooks: ANY ROI change or mode switch invalidates the result, so
    # the map is never left stale. Compute is the only way forward (no auto-run
    # on drag release -- that would fire a full pass on every nudge).
    def _on_roi_committed(self):
        self._invalidate()

    def _on_selection_reset(self):
        self._invalidate()

    def _invalidate(self):
        self._result = None
        self.btn_export2d.setEnabled(False)
        self.result_fig.clear(); self.result_canvas.draw()
        self.lbl_roi.setText("")
        self.lbl_status.setText("ROI changed -- press Compute.")

    def _conv_tag(self):
        c = self.combo_c.currentText()
        if self.rb_ffp.isChecked():
            return "FFP", f"FFP  P({c} > 0)", "p_forward", "ffp"
        return "RFP", f"RFP  P({c} < 0)", "p_reverse", "rfp"

    def _compute(self):
        c = self.combo_c.currentText()
        field = self.win.component_field(c)
        region = self._roi_region()
        self.btn_compute.setEnabled(False)
        self.progress.setVisible(True); self.progress.setRange(0, 100)
        self.lbl_status.setText(f"Computing {self._region_desc(region)}…")
        self._worker = ProbabilityWorker(
            "direction", field=field, mask_2d=self.win.mask2d,
            deadband=self.spin_eps.value(), region=region)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_error(self, tb):
        self.btn_compute.setEnabled(True); self.progress.setVisible(False)
        QMessageBox.critical(self, "Flow Direction Error", tb)

    def _on_result(self, r):
        self._result = r
        self.btn_compute.setEnabled(True); self.progress.setVisible(False)
        self.btn_export2d.setEnabled(True)
        has_ind = self.spin_eps.value() > 0
        self.combo_view.model().item(1).setEnabled(has_ind)
        self.lbl_status.setText(self._region_desc(r.get("region")))
        self._replot_same_result()

    def _displayed_map(self):
        r = self._result
        view = self.combo_view.currentIndex()
        if view == 1:
            return r["p_indeterminate"], "viridis", (None, None), "Indeterminate fraction"
        if view == 2:
            return r["n_valid"].astype(float), "viridis", (None, None), "Valid samples N"
        _, _, key, _ = self._conv_tag()
        return r[key], "RdBu_r", (0.0, 1.0), self._conv_tag()[1]

    def _replot_same_result(self, *_):
        if self._result is None:
            return
        region = self._result.get("region")
        arr, cmap, (vmin, vmax), label = self._displayed_map()
        x, y = self.win.x, self.win.y
        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        # computed-but-invalid points (masked) show as the lighter background grey
        ax.set_facecolor("#E6E6E6")
        data = np.ma.masked_invalid(arr)
        if vmin is not None:
            cf = ax.contourf(x, y, data, levels=21, cmap=cmap, vmin=vmin, vmax=vmax,
                             extend="neither")
        else:
            cf = ax.contourf(x, y, data, levels=21, cmap=cmap, extend="neither")
        cb = self.result_fig.colorbar(cf, ax=ax, extend="neither")
        cb.set_label(label, fontsize=_FONT_AX)
        # 0.50 (and optional 0.01/0.20) contours come from the NaN-outside array,
        # so they stop at the ROI boundary; contour() does not draw a spurious
        # edge along the NaN transition.
        if self.combo_view.currentIndex() == 0:
            ax.contour(x, y, data, levels=[0.5], colors="k", linewidths=1.5)
            if self.chk_simpson.isChecked():
                ax.contour(x, y, data, levels=[0.01, 0.20], colors="k",
                           linewidths=0.6)
        # darker grey over the not-computed bands outside the ROI + ROI outline
        self._shade_outside_roi(ax, region)
        ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        ax.set_title(label, fontsize=_FONT_AX)
        # axes limits stay at the FULL domain extent (do not zoom to the ROI)
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
        ax.set_aspect("equal"); ax.tick_params(labelsize=_FONT_TICK)
        self.result_fig.tight_layout(pad=0.5); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self._update_roi_readout()

    def _update_roi_readout(self):
        if self._result is None:
            return
        region = self._result.get("region")
        _, _, key, _ = self._conv_tag()
        p = self._result[key]
        # p is already NaN outside the ROI, so the finite set IS the computed
        # subregion -- the readout derives from the computed points, not a crop.
        vals = p[np.isfinite(p)]
        if vals.size == 0:
            self.lbl_roi.setText(""); return
        stat = (f"mean P={vals.mean():.3f}  area frac P>0.5={np.mean(vals > 0.5):.3f}  "
                f"min={vals.min():.3f}  max={vals.max():.3f}  (n={vals.size})")
        if region is not None:
            self.lbl_roi.setText("Grey: outside ROI, not computed.    " + stat)
        else:
            self.lbl_roi.setText("Whole domain.    " + stat)

    def _settings_info(self):
        _, label, _, _ = self._conv_tag()
        region = self._result.get("region") if self._result else self._roi_region()
        info = {"module": "Probability / Flow Direction",
                "component": self.combo_c.currentText(),
                "convention": label, "deadband_ms": self.spin_eps.value(),
                "Nt": self.win.Nt, "chunk": prob.CHUNK_DEFAULT,
                "denominator": "per-point n_valid (NaN excluded per point); "
                               "p_forward+p_reverse+p_indeterminate=1",
                "nan_note": "NaN outside the ROI = not computed; NaN inside = no "
                            "valid samples (see n_valid, 0 outside the ROI)"}
        info.update(self._region_header(region))
        return info

    def _export_2d(self):
        if self._result is None:
            return
        tag, _, key, short = self._conv_tag()
        c = self.combo_c.currentText()
        fn = f"{_case_name()}_probability_{short}_{c}.dat"
        path, _ = QFileDialog.getSaveFileName(self, "Export 2D", fn,
                                              "Tecplot DAT (*.dat)")
        if not path:
            return
        r = self._result
        fields = [r[key], r["p_indeterminate"], r["n_valid"].astype(float)]
        names = [f"P_{tag}_{c}", "P_indeterminate", "n_valid"]
        export_2d_tecplot(path, self.win.x, self.win.y, fields, names,
                          self._settings_info(), nan_repr="NaN")
        self.lbl_status.setText(f"Saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# Tab 3 -- Binary Space-Time
# --------------------------------------------------------------------------- #

class BinarySpaceTimeTab(_FieldSelectionMixin, QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._state = None
        self._q = None
        self._preview_ready = False
        self._build_ui()

    def showEvent(self, e):
        super().showEvent(e)
        self._ensure_preview()

    def _ensure_preview(self):
        if self._preview_ready or not self.win.is_time_resolved:
            return
        self._preview_ready = True
        self._init_field(width=380)
        self._connect_field_mouse()
        self._select_mode = "line"
        self._draw_background(self.win.background_2d("u"), "Line placement")
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    def _build_ui(self):
        root = QHBoxLayout(self)
        if not self.win.is_time_resolved:
            lbl = QLabel("Binary space-time maps require time-resolved data. "
                         "Select 'Time-Resolved' in Acquisition Type and set fs.")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#856404; background:#FFF3CD; padding:12px;")
            root.addWidget(lbl)
            return
        splitter = QSplitter(Qt.Orientation.Horizontal); root.addWidget(splitter)
        left = QWidget(); left.setMinimumWidth(400); left.setMaximumWidth(520)
        ll = QVBoxLayout(left); self._left_layout = ll

        r = QHBoxLayout(); r.addWidget(QLabel("Component:"))
        self.combo_c = QComboBox()
        self.combo_c.addItems(["u", "v", "w"] if self.win.is_stereo else ["u", "v"])
        r.addWidget(self.combo_c); ll.addLayout(r)

        # Line: only horizontal / vertical (a free line has non-uniform arc
        # spacing and would mis-meter the space axis).
        self.line_sel = LineSelectorWidget(show_avg=False, show_free=False)
        ll.addWidget(self.line_sel)
        self.manual_grp = self._build_manual_group()
        ll.addWidget(self.manual_grp)
        self.lbl_hint = QLabel("Draw a horizontal/vertical line or set x0,y0,x1,y1.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        rb = QHBoxLayout(); rb.addWidget(QLabel("± grid pts:"))
        self.spin_band = QSpinBox(); self.spin_band.setRange(0, 20)
        rb.addWidget(self.spin_band); ll.addLayout(rb)

        re = QHBoxLayout(); re.addWidget(QLabel("Deadband ε [m/s]:"))
        self.spin_eps = QDoubleSpinBox(); self.spin_eps.setRange(0.0, 10.0)
        self.spin_eps.setDecimals(3); self.spin_eps.setValue(0.0)
        re.addWidget(self.spin_eps); ll.addLayout(re)

        conv = QGroupBox("Convention"); cl = QVBoxLayout(conv)
        self.rb_ffp = QRadioButton("Forward (FFP)"); self.rb_rfp = QRadioButton("Reverse (RFP)")
        self.rb_ffp.setChecked(True)
        self._conv_grp = QButtonGroup(conv)
        self._conv_grp.addButton(self.rb_ffp); self._conv_grp.addButton(self.rb_rfp)
        self.rb_ffp.toggled.connect(self._plot)
        cl.addWidget(self.rb_ffp); cl.addWidget(self.rb_rfp); ll.addWidget(conv)

        nrm = QGroupBox("Normalisation"); nl = QVBoxLayout(nrm)
        self.chk_norm = QCheckBox("Use normalised axes"); nl.addWidget(self.chk_norm)
        self.chk_norm.toggled.connect(self._plot)
        gr = QHBoxLayout()
        self.spin_uref = QDoubleSpinBox(); self.spin_uref.setRange(1e-6, 1e6)
        self.spin_uref.setValue(1.0); self.spin_uref.setDecimals(4)
        self.spin_lref = QDoubleSpinBox(); self.spin_lref.setRange(1e-6, 1e6)
        self.spin_lref.setValue(1.0); self.spin_lref.setDecimals(4)
        self.spin_s0 = QDoubleSpinBox(); self.spin_s0.setRange(-1e6, 1e6)
        self.spin_s0.setDecimals(4)
        gr.addWidget(QLabel("U_ref")); gr.addWidget(self.spin_uref)
        gr.addWidget(QLabel("L_ref")); gr.addWidget(self.spin_lref)
        nl.addLayout(gr)
        gr2 = QHBoxLayout(); gr2.addWidget(QLabel("s origin offset [mm]:"))
        gr2.addWidget(self.spin_s0); nl.addLayout(gr2)
        ll.addWidget(nrm)

        rf = QGroupBox("Reference lines"); rfl = QVBoxLayout(rf)
        self.ed_ref1 = QLineEdit(); self.ed_ref1.setPlaceholderText("dashed: comma-separated")
        self.ed_ref2 = QLineEdit(); self.ed_ref2.setPlaceholderText("dotted: comma-separated")
        self.ed_ref1.editingFinished.connect(self._plot)
        self.ed_ref2.editingFinished.connect(self._plot)
        rfl.addWidget(self.ed_ref1); rfl.addWidget(self.ed_ref2); ll.addWidget(rf)

        self.btn_compute = QPushButton("Compute / Update")
        self.btn_compute.clicked.connect(self._compute); ll.addWidget(self.btn_compute)
        er = QHBoxLayout()
        self.btn_exp_state = QPushButton("Export state CSV...")
        self.btn_exp_state.clicked.connect(lambda: self._export("state"))
        self.btn_exp_q = QPushButton("Export q CSV...")
        self.btn_exp_q.clicked.connect(lambda: self._export("q"))
        for b in (self.btn_exp_state, self.btn_exp_q):
            b.setEnabled(False); er.addWidget(b)
        ll.addLayout(er)
        self.lbl_status = QLabel(
            "Streak slope = apparent convection velocity; positive/negative slope "
            "= downstream/upstream propagation.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_status); ll.addStretch()

        right = QWidget(); rl = QVBoxLayout(right)
        self.result_fig = Figure(); self.result_canvas = FigureCanvas(self.result_fig)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar); rl.addWidget(self.result_canvas, stretch=1)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([500, 1200])

    def _init_field(self, width=380):
        super()._init_field(width)
        self._left_layout.insertWidget(0, self.field_toolbar)
        self._left_layout.insertWidget(1, self.field_canvas)

    def _compute(self):
        c = self.combo_c.currentText()
        field = self.win.component_field(c)
        x, y = self.win.x, self.win.y
        # use the drawn line if there is one, else the manual x0/y0/x1/y1 entry
        x0, y0, x1, y1 = self._line if self._line is not None else self._manual_line()
        direction = "x" if self.line_sel.get_mode() == "horizontal" else "y"
        try:
            q_st, s_mm, info = prob.extract_space_time(
                field, x, y, x0, y0, x1, y1, direction, avg_band=self.spin_band.value())
        except Exception as exc:
            QMessageBox.critical(self, "Binary Space-Time Error", str(exc)); return
        state = prob.binarize_space_time(q_st, deadband=self.spin_eps.value())
        self._q = q_st; self._state = state; self._s = s_mm; self._info = info
        self._line = (x0, y0, x1, y1); self._redraw_overlays()
        self.btn_exp_state.setEnabled(True); self.btn_exp_q.setEnabled(True)
        self._plot()

    def _axes_vectors(self):
        Ns, Nt = self._state.shape
        t = np.arange(Nt) / self.win.fs
        s = self._s
        if self.chk_norm.isChecked():
            s_ax = (s - self.spin_s0.value()) / self.spin_lref.value()
            t_ax = t * self.spin_uref.value() / self.spin_lref.value()
            slab = "s* = (s - s0)/L_ref"; tlab = "t* = t U_ref/L_ref"
        else:
            s_ax = s; t_ax = t
            slab = "s [mm]"; tlab = "t [s]"
        return s_ax, t_ax, slab, tlab

    def _plot(self, *_):
        if self._state is None:
            return
        s_ax, t_ax, slab, tlab = self._axes_vectors()
        state = self._state
        self.result_fig.clear(); ax = self.result_fig.add_subplot(111)
        extent = [s_ax.min(), s_ax.max(), t_ax.min(), t_ax.max()]
        ax.imshow(state.T, origin="lower", aspect="auto", interpolation="nearest",
                  cmap=_STATE_CMAP, norm=_STATE_NORM, extent=extent)
        c = self.combo_c.currentText()
        if self.rb_ffp.isChecked():
            fwd_lbl, rev_lbl = f"{c} > 0", f"{c} < 0"
        else:
            fwd_lbl, rev_lbl = f"{c} > 0", f"{c} < 0"
        patches = [mpatches.Patch(color=_STATE_COLORS[0], label=rev_lbl),
                   mpatches.Patch(color=_STATE_COLORS[1], label=fwd_lbl)]
        if (state == 2).any():
            patches.append(mpatches.Patch(color=_STATE_COLORS[2], label="indeterminate"))
        if (state == 3).any():
            patches.append(mpatches.Patch(color=_STATE_COLORS[3], label="invalid"))
        ax.legend(handles=patches, ncol=len(patches), fontsize=_FONT_LEG,
                  loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        for ed, ls in ((self.ed_ref1, "--"), (self.ed_ref2, ":")):
            for v in self._parse_refs(ed.text()):
                ax.axvline(v, color="0.3", linewidth=0.8, linestyle=ls)
        ax.set_xlabel(slab, fontsize=_FONT_AX); ax.set_ylabel(tlab, fontsize=_FONT_AX)
        ax.tick_params(labelsize=_FONT_TICK); ax.set_aspect("auto")
        self.result_fig.tight_layout(pad=0.5); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

    @staticmethod
    def _parse_refs(text):
        out = []
        for tok in text.split(","):
            tok = tok.strip()
            if tok:
                try:
                    out.append(float(tok))
                except ValueError:
                    pass
        return out

    def _settings_info(self):
        return {"module": "Probability / Binary Space-Time",
                "component": self.combo_c.currentText(),
                "direction": self._info["direction"],
                "snapped_coord_mm": self._info["coord_mm"],
                "avg_band": self.spin_band.value(),
                "deadband_ms": self.spin_eps.value(),
                "convention": "FFP" if self.rb_ffp.isChecked() else "RFP",
                "fs_Hz": self.win.fs,
                "normalised": self.chk_norm.isChecked(),
                "U_ref": self.spin_uref.value(), "L_ref": self.spin_lref.value(),
                "s0_mm": self.spin_s0.value(),
                "state_codes": "0=reverse 1=forward 2=indeterminate 3=invalid"}

    def _export(self, which):
        if self._state is None:
            return
        c = self.combo_c.currentText()
        fn = f"{_case_name()}_probability_binary_{c}_{which}.csv"
        path, _ = QFileDialog.getSaveFileName(self, f"Export {which}", fn,
                                              "CSV Files (*.csv)")
        if not path:
            return
        s_ax, t_ax, _, _ = self._axes_vectors()
        arr = self._state if which == "state" else self._q
        with open(path, "w", encoding="utf-8") as f:
            f.write(_settings_header(self._settings_info())); f.write("\n")
            f.write("# row 0 = s axis; col 0 = t axis; body = "
                    + ("state codes" if which == "state" else "q(s,t)") + "\n")
            f.write("s\\t," + ",".join(f"{tv:.6g}" for tv in t_ax) + "\n")
            fmt = "%d" if which == "state" else "%.6g"
            for i in range(arr.shape[0]):
                row = ",".join((fmt % v) for v in arr[i, :])
                f.write(f"{s_ax[i]:.6g},{row}\n")
        self.lbl_status.setText(f"Saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# Tab 4 -- Quadrant
# --------------------------------------------------------------------------- #

class QuadrantTab(_FieldSelectionMixin, QWidget):
    _QNAMES = ["Q1 outward", "Q2 ejection", "Q3 inward", "Q4 sweep"]

    def __init__(self, win):
        super().__init__()
        self.win = win
        self._result = None
        self._sweep = None
        self._preview_ready = False
        self._build_ui()

    def showEvent(self, e):
        super().showEvent(e)
        self._ensure_preview()

    def _ensure_preview(self):
        if self._preview_ready:
            return
        self._preview_ready = True
        self._init_field(width=360)
        self._connect_field_mouse()
        self._draw_background(self.win.background_2d("u"), "Selection preview")
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)
        self._update_hint()

    def _pairs(self):
        return ["u'-v'", "u'-w'", "v'-w'"] if self.win.is_stereo else ["u'-v'"]

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal); root.addWidget(splitter)
        left = QWidget(); left.setMinimumWidth(380); left.setMaximumWidth(520)
        ll = QVBoxLayout(left); self._left_layout = ll

        r = QHBoxLayout(); r.addWidget(QLabel("Pair:"))
        self.combo_pair = QComboBox(); self.combo_pair.addItems(self._pairs())
        r.addWidget(self.combo_pair); ll.addLayout(r)

        r2 = QHBoxLayout(); r2.addWidget(QLabel("Hole size H:"))
        self.spin_hole = QDoubleSpinBox(); self.spin_hole.setRange(0.0, 10.0)
        self.spin_hole.setSingleStep(0.5); self.spin_hole.setValue(0.0)
        r2.addWidget(self.spin_hole); ll.addLayout(r2)

        ll.addWidget(self._make_selection_group(["roi", "whole"], "whole",
                                                "Whole field"))
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)
        note_reg = QLabel("Region applies to the joint histogram + hole sweep; "
                          "the 2D maps always use the full domain.")
        note_reg.setStyleSheet("color: gray; font-size: 10px;")
        note_reg.setWordWrap(True); ll.addWidget(note_reg)

        r3 = QHBoxLayout(); r3.addWidget(QLabel("Joint bins:"))
        self.spin_bins = QSpinBox(); self.spin_bins.setRange(11, 501)
        self.spin_bins.setValue(101); r3.addWidget(self.spin_bins); ll.addLayout(r3)

        self.btn_compute = QPushButton("Compute")
        self.btn_compute.clicked.connect(self._compute); ll.addWidget(self.btn_compute)
        self.btn_sweep = QPushButton("Hole sweep (Lu & Willmarth)")
        self.btn_sweep.clicked.connect(self._compute_sweep); ll.addWidget(self.btn_sweep)
        self.progress = QProgressBar(); self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False); self.progress.setVisible(False)
        ll.addWidget(self.progress)
        er = QHBoxLayout()
        self.btn_exp_frac = QPushButton("Export fractions CSV...")
        self.btn_exp_frac.clicked.connect(self._export_frac); self.btn_exp_frac.setEnabled(False)
        self.btn_exp_2d = QPushButton("Export 2D...")
        self.btn_exp_2d.clicked.connect(self._export_2d); self.btn_exp_2d.setEnabled(False)
        er.addWidget(self.btn_exp_frac); er.addWidget(self.btn_exp_2d); ll.addLayout(er)
        self.btn_exp_sweep = QPushButton("Export hole sweep CSV...")
        self.btn_exp_sweep.clicked.connect(self._export_sweep); self.btn_exp_sweep.setEnabled(False)
        ll.addWidget(self.btn_exp_sweep)

        note = QLabel("Quadrant labels assume x is streamwise and y is wall-normal. "
                      "Use the Transform module to align the dataset first.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#856404; background:#FFF3CD; padding:6px; font-size:11px;")
        ll.addWidget(note)
        self.lbl_status = QLabel(""); self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_status); ll.addStretch()

        right = QWidget(); rl = QVBoxLayout(right)
        self.result_fig = Figure(); self.result_canvas = FigureCanvas(self.result_fig)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar); rl.addWidget(self.result_canvas, stretch=1)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([500, 1200])

    def _init_field(self, width=360):
        super()._init_field(width)
        self._left_layout.insertWidget(0, self.field_toolbar)
        self._left_layout.insertWidget(1, self.field_canvas)

    def _pair_fields(self):
        pair = self.combo_pair.currentText()
        keymap = {"u'": "U", "v'": "V", "w'": "W"}
        a, b = pair.split("-")
        return self.win.dataset[keymap[a]], self.win.dataset[keymap[b]]

    def _region(self):
        if self._select_mode == "roi" and self._roi is not None:
            r0, r1, c0, c1 = self._roi
            return ("roi", r0, r1, c0, c1)
        return None

    # any ROI change / mode switch invalidates the current result
    def _on_roi_committed(self):
        self._invalidate()

    def _on_selection_reset(self):
        self._invalidate()

    def _invalidate(self):
        self._result = None
        self._sweep = None
        for b in (self.btn_exp_frac, self.btn_exp_2d, self.btn_exp_sweep):
            b.setEnabled(False)
        self.result_fig.clear(); self.result_canvas.draw()
        self.lbl_status.setText("ROI changed -- press Compute.")

    def _compute(self):
        fa, fb = self._pair_fields()
        self.btn_compute.setEnabled(False)
        self.progress.setVisible(True); self.progress.setRange(0, 100)
        self.lbl_status.setText("Computing quadrant analysis…")
        self._worker = ProbabilityWorker(
            "quadrant", field_a=fa, field_b=fb, mask_2d=self.win.mask2d,
            hole=self.spin_hole.value(), region=self._region(),
            joint_bins=self.spin_bins.value())
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_error(self, tb):
        self.btn_compute.setEnabled(True); self.progress.setVisible(False)
        QMessageBox.critical(self, "Quadrant Error", tb)

    def _on_result(self, r):
        self._result = r
        self.btn_compute.setEnabled(True); self.progress.setVisible(False)
        self.btn_exp_frac.setEnabled(True); self.btn_exp_2d.setEnabled(True)
        self._plot()

    def _plot(self):
        r = self._result
        self.result_fig.clear()
        gs = self.result_fig.add_gridspec(2, 2)
        a_lab, b_lab = self.combo_pair.currentText().split("-")

        # 1. Joint PDF
        ax1 = self.result_fig.add_subplot(gs[0, 0])
        jh = r["joint_hist"].astype(float)
        ea, eb = r["edges_a"], r["edges_b"]
        im = ax1.imshow(np.log10(jh.T + 1), origin="lower", aspect="auto",
                        extent=[ea[0], ea[-1], eb[0], eb[-1]], cmap="viridis")
        ax1.axhline(0, color="w", linewidth=0.6); ax1.axvline(0, color="w", linewidth=0.6)
        for (qx, qy, lab) in [(0.7, 0.9, "Q1"), (0.1, 0.9, "Q2"),
                              (0.1, 0.1, "Q3"), (0.7, 0.1, "Q4")]:
            ax1.text(qx, qy, lab, transform=ax1.transAxes, color="w", fontsize=8)
        H = self.spin_hole.value()
        if H > 0:
            arms = np.nanmean(r["a_rms"]); brms = np.nanmean(r["b_rms"])
            aa = np.linspace(ea[0], ea[-1], 200); aa = aa[np.abs(aa) > 1e-9]
            hyp = H * arms * brms / aa
            ax1.plot(aa, hyp, "w--", linewidth=0.7); ax1.plot(aa, -hyp, "w--", linewidth=0.7)
            ax1.set_ylim(eb[0], eb[-1])
        ax1.set_xlabel(a_lab, fontsize=_FONT_AX); ax1.set_ylabel(b_lab, fontsize=_FONT_AX)
        ax1.set_title("Joint PDF log10(N+1)", fontsize=_FONT_AX - 1)
        ax1.tick_params(labelsize=_FONT_TICK)

        # 2. Bar chart
        ax2 = self.result_fig.add_subplot(gs[0, 1])
        tf = [np.nanmean(r["time_frac"][i]) for i in range(4)]
        sf = [np.nanmean(r["stress_frac"][i]) for i in range(4)]
        xq = np.arange(4)
        ax2.bar(xq - 0.2, tf, width=0.4, label="time frac", color="#4C78A8")
        ax2.bar(xq + 0.2, sf, width=0.4, label="stress frac", color="#E8A85C")
        ax2.set_xticks(xq); ax2.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax2.axhline(0, color="k", linewidth=0.6)
        ax2.legend(fontsize=_FONT_LEG); ax2.tick_params(labelsize=_FONT_TICK)
        ax2.set_title("Quadrant contributions", fontsize=_FONT_AX - 1)
        ax2.set_aspect("auto")

        # 3. Q2 and Q4 stress-fraction maps (viridis limits are data-driven, so
        #    they come from the ROI data only; grey = outside ROI, not computed).
        region = r.get("region")
        x, y = self.win.x, self.win.y
        for idx, qi, ax in ((2, 1, self.result_fig.add_subplot(gs[1, 0])),
                            (4, 3, self.result_fig.add_subplot(gs[1, 1]))):
            ax.set_facecolor("#E6E6E6")
            cf = ax.contourf(x, y, np.ma.masked_invalid(r["stress_frac"][qi]),
                             levels=21, cmap="viridis", extend="neither")
            self.result_fig.colorbar(cf, ax=ax, extend="neither")
            self._shade_outside_roi(ax, region)
            ax.set_title(f"Q{idx} stress fraction", fontsize=_FONT_AX - 1)
            ax.set_xlabel("x [mm]", fontsize=_FONT_AX); ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
            ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
            ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
            ax.set_aspect("equal"); ax.tick_params(labelsize=_FONT_TICK)

        self.result_fig.tight_layout(pad=0.5); self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        cap = ("   Grey: outside ROI, not computed." if region is not None else "")
        self.lbl_status.setText(
            self._region_desc(region) + cap + "   |   time frac Q1..Q4 = " +
            ", ".join(f"{v:.3f}" for v in tf) + "   stress frac = " +
            ", ".join(f"{v:.3f}" for v in sf))

    def _compute_sweep(self):
        fa, fb = self._pair_fields()
        self.btn_sweep.setEnabled(False)
        self.progress.setVisible(True); self.progress.setRange(0, 100)
        self.lbl_status.setText("Computing hole sweep…")
        self._sworker = ProbabilityWorker(
            "hole_sweep", field_a=fa, field_b=fb, mask_2d=self.win.mask2d,
            region=self._region())
        self._sworker.progress.connect(self.progress.setValue)
        self._sworker.finished.connect(self._on_sweep)
        self._sworker.error.connect(lambda tb: (self.btn_sweep.setEnabled(True),
                                                self.progress.setVisible(False),
                                                QMessageBox.critical(self, "Hole Sweep", tb)))
        self._sworker.start()

    def _on_sweep(self, r):
        self._sweep = r
        self.btn_sweep.setEnabled(True); self.progress.setVisible(False)
        self.btn_exp_sweep.setEnabled(True)
        self.result_fig.clear(); ax = self.result_fig.add_subplot(111)
        holes = r["holes"]; sf = r["stress_frac"]
        for i, lab in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            ax.plot(holes, sf[i], "-o", markersize=3, label=lab)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xlabel("Hole size H", fontsize=_FONT_AX)
        ax.set_ylabel("fractional stress contribution", fontsize=_FONT_AX)
        ax.set_title("Hole-size sweep (Lu & Willmarth 1973)", fontsize=_FONT_AX)
        ax.legend(fontsize=_FONT_LEG); ax.set_aspect("auto")
        ax.tick_params(labelsize=_FONT_TICK)
        self.result_fig.tight_layout(pad=0.5); self.result_canvas.draw()

    def _settings_info(self):
        reg = (self._result.get("region") if self._result else None) or self._region()
        info = {"module": "Probability / Quadrant",
                "pair": self.combo_pair.currentText(),
                "hole_H": self.spin_hole.value(),
                "joint_bins": self.spin_bins.value(), "Nt": self.win.Nt,
                "quadrant_convention": "x streamwise, y wall-normal; "
                "Q1 outward, Q2 ejection, Q3 inward, Q4 sweep",
                "nan_note": "NaN outside the ROI = not computed"}
        info.update(self._region_header(reg))
        return info

    def _export_frac(self):
        if self._result is None:
            return
        fn = f"{_case_name()}_probability_quadrant_fractions.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export fractions", fn,
                                              "CSV Files (*.csv)")
        if not path:
            return
        r = self._result
        with open(path, "w", encoding="utf-8") as f:
            f.write(_settings_header(self._settings_info())); f.write("\n")
            f.write("quadrant,time_frac,stress_frac\n")
            for i, lab in enumerate(["Q1", "Q2", "Q3", "Q4"]):
                f.write(f"{lab},{np.nanmean(r['time_frac'][i]):.8g},"
                        f"{np.nanmean(r['stress_frac'][i]):.8g}\n")
        self.lbl_status.setText(f"Saved {os.path.basename(path)}")

    def _export_2d(self):
        if self._result is None:
            return
        fn = f"{_case_name()}_probability_quadrant_stressfrac.dat"
        path, _ = QFileDialog.getSaveFileName(self, "Export 2D", fn,
                                              "Tecplot DAT (*.dat)")
        if not path:
            return
        r = self._result
        fields = [r["stress_frac"][i] for i in range(4)]
        names = ["Q1_stress_frac", "Q2_stress_frac", "Q3_stress_frac", "Q4_stress_frac"]
        export_2d_tecplot(path, self.win.x, self.win.y, fields, names,
                          self._settings_info(), nan_repr="NaN")
        self.lbl_status.setText(f"Saved {os.path.basename(path)}")

    def _export_sweep(self):
        if self._sweep is None:
            return
        fn = f"{_case_name()}_probability_quadrant_holesweep.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export hole sweep", fn,
                                              "CSV Files (*.csv)")
        if not path:
            return
        r = self._sweep; holes = r["holes"]; sf = r["stress_frac"]
        with open(path, "w", encoding="utf-8") as f:
            f.write(_settings_header(self._settings_info())); f.write("\n")
            f.write("H,Q1,Q2,Q3,Q4\n")
            for j in range(len(holes)):
                f.write(f"{holes[j]:.6g}," +
                        ",".join(f"{sf[i, j]:.8g}" for i in range(4)) + "\n")
        self.lbl_status.setText(f"Saved {os.path.basename(path)}")
