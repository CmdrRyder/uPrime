"""
gui/anisotropy_window.py
------------------------
Anisotropy invariant analysis window.

Left  : mean field -- user draws a line (Lumley) or rectangle (barycentric)
Right : Lumley triangle (-II vs III) with points colored by distance along line
        OR barycentric RGB map overlaid on the flow domain
"""

import numpy as np
from scipy.ndimage import median_filter
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGroupBox, QPushButton, QRadioButton,
    QSizePolicy, QMessageBox, QSplitter, QTabWidget,
    QSpinBox, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from core.export import export_2d_tecplot, export_line_csv
from core.anisotropy import (
    compute_reynolds_tensor,
    compute_anisotropy_tensor,
    compute_invariants_fast,
    compute_barycentric,
    points_near_line,
)


class AnisotropyWindow(PickerMixin, QWidget):

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Anisotropy Invariant Analysis")
        self.resize(1400, 780)

        self._mode       = "line"    # "line" or "rect"
        self._press_xy   = None
        self._line_artist  = None
        self._rect_patch   = None
        self._selection    = None

        # Pre-compute Reynolds tensor and anisotropy tensor once
        self._precompute()
        self._build_ui()

        # Warn if snapshot count is low
        Nt = dataset["Nt"]
        if Nt < 2000:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Convergence Warning",
                f"Only {Nt} snapshots loaded.\n\n"
                "Reynolds stress statistics (and therefore anisotropy invariants) "
                "are typically not well-converged with fewer than ~2000 snapshots.\n\n"
                "Results should be interpreted with caution."
            )
        self._draw_field()
        self._connect_mouse()

    # ----------------------------------------------------------------------- #
    # Pre-computation
    # ----------------------------------------------------------------------- #

    def _precompute(self):
        ds = self.dataset
        self._R, self._k = compute_reynolds_tensor(ds["U"], ds["V"], ds["W"])
        self._b           = compute_anisotropy_tensor(self._R, self._k)
        self._neg_II, self._III = compute_invariants_fast(self._b)

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left: field canvas ----
        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        self.field_fig    = Figure(figsize=(6, 4), tight_layout=True)
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)

        # Mode selector
        mode_grp = QGroupBox("Selection Mode")
        mode_lay = QHBoxLayout(mode_grp)
        self.rb_line = QRadioButton("Line  (Lumley triangle)")
        self.rb_rect = QRadioButton("Rectangle  (Barycentric map)")
        self.rb_line.setChecked(True)
        self.rb_line.toggled.connect(self._on_mode_changed)
        mode_lay.addWidget(self.rb_line)
        mode_lay.addWidget(self.rb_rect)
        ll.addWidget(mode_grp)

        self.lbl_hint = QLabel("Click and drag to draw a line.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        ll.addWidget(self.lbl_hint)

        self.btn_compute = QPushButton("Compute")
        self.btn_compute.setEnabled(False)
        self.btn_compute.clicked.connect(self._on_compute)
        ll.addWidget(self.btn_compute)

        self.btn_export = QPushButton("Export Data...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        ll.addWidget(self.btn_export)

        # Median filter smoothing
        smooth_grp = QGroupBox("Smoothing")
        smooth_lay = QHBoxLayout(smooth_grp)
        self.chk_smooth = QCheckBox("Median filter")
        self.chk_smooth.setChecked(False)
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(1, 15)
        self.spin_smooth.setValue(3)
        self.spin_smooth.setSingleStep(2)
        self.spin_smooth.setToolTip("Filter kernel size (odd number, e.g. 3, 5, 7)")
        smooth_lay.addWidget(self.chk_smooth)
        smooth_lay.addWidget(QLabel("Kernel:"))
        smooth_lay.addWidget(self.spin_smooth)
        ll.addWidget(smooth_grp)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        # ---- Right: result tabs ----
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()

        # Tab 1: Lumley triangle
        self.lumley_fig    = Figure(figsize=(5, 5), tight_layout=True)
        self.lumley_canvas = FigureCanvas(self.lumley_fig)
        self.lumley_toolbar = NavToolbar(self.lumley_canvas, self)
        lumley_widget = QWidget()
        lw_lay = QVBoxLayout(lumley_widget)
        lw_lay.addWidget(self.lumley_toolbar)
        lw_lay.addWidget(self.lumley_canvas)
        self.tabs.addTab(lumley_widget, "Lumley Triangle")

        # Tab 2: Barycentric map
        self.bary_fig    = Figure(figsize=(7, 4), tight_layout=True)
        self.bary_canvas = FigureCanvas(self.bary_fig)
        self.bary_toolbar = NavToolbar(self.bary_canvas, self)
        bary_widget = QWidget()
        bw_lay = QVBoxLayout(bary_widget)
        bw_lay.addWidget(self.bary_toolbar)
        bw_lay.addWidget(self.bary_canvas)
        self.tabs.addTab(bary_widget, "Barycentric Map")

        rl.addWidget(self.tabs)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    # ----------------------------------------------------------------------- #
    # Draw mean speed field on left canvas
    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds = self.dataset
        x  = ds["x"]
        y  = ds["y"]

        speed = np.sqrt(
            np.nanmean(ds["U"], axis=2)**2 +
            np.nanmean(ds["V"], axis=2)**2 +
            np.nanmean(ds["W"], axis=2)**2
        )
        valid_frac = np.mean(ds["valid"], axis=2)
        speed[valid_frac < 0.5] = np.nan

        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        cf = self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
        self.field_fig.colorbar(cf, ax=self.field_ax,
                                label="Mean |V| [m/s]", shrink=0.8)
        self.field_ax.set_xlabel("x [mm]")
        self.field_ax.set_ylabel("y [mm]")
        self.field_ax.set_title("Draw a line or rectangle to analyze anisotropy")
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_canvas.draw()

        self._x = x
        self._y = y

    # ----------------------------------------------------------------------- #
    # Mouse
    # ----------------------------------------------------------------------- #

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_mode_changed(self):
        if self.rb_line.isChecked():
            self._mode = "line"
            self.lbl_hint.setText("Click and drag to draw a line.")
        else:
            self._mode = "rect"
            self.lbl_hint.setText("Click and drag to draw a rectangle.")
        self._clear_graphics()
        self._selection = None
        self.btn_compute.setEnabled(False)

    def _on_press(self, event):
        if event.inaxes != self.field_ax:
            return
        if hasattr(self, 'field_toolbar') and str(self.field_toolbar.mode) != '':
            return
        self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.field_ax:
            return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._clear_graphics()

        if self._mode == "line":
            self._line_artist, = self.field_ax.plot(
                [x0, x1], [y0, y1], "r-", linewidth=2, zorder=10
            )
        else:
            self._rect_patch = Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=1.5, edgecolor="red",
                facecolor="red", alpha=0.15, zorder=10
            )
            self.field_ax.add_patch(self._rect_patch)

        self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None:
            return
        if event.inaxes != self.field_ax:
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._press_xy = None

        if self._mode == "line":
            if abs(x1-x0) < 0.1 and abs(y1-y0) < 0.1:
                self.lbl_hint.setText("Line too short -- try again.")
                return
            self._selection = {"type": "line",
                               "x0": x0, "y0": y0, "x1": x1, "y1": y1}
            self.lbl_hint.setText(
                f"Line: ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f}) mm"
            )
        else:
            if abs(x1-x0) < 0.1 or abs(y1-y0) < 0.1:
                self.lbl_hint.setText("Rectangle too small -- try again.")
                return
            self._selection = {"type": "rect",
                               "x0": x0, "x1": x1, "y0": y0, "y1": y1}
            self.lbl_hint.setText(
                f"Rectangle: x=[{min(x0,x1):.1f},{max(x0,x1):.1f}] "
                f"y=[{min(y0,y1):.1f},{max(y0,y1):.1f}] mm"
            )

        self.btn_compute.setEnabled(True)

    def _clear_graphics(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None
        if self._rect_patch is not None:
            try:
                self._rect_patch.remove()
            except Exception:
                pass
            self._rect_patch = None
        self.field_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Compute
    # ----------------------------------------------------------------------- #

    def _get_smooth_fields(self):
        """Return neg_I2 and I3 fields, optionally smoothed."""
        neg_I2 = self._neg_II.copy()
        I3     = self._III.copy()
        if self.chk_smooth.isChecked():
            k = self.spin_smooth.value()
            # Median filter ignoring NaN: fill NaN, filter, restore NaN
            nan_mask = np.isnan(neg_I2) | np.isnan(I3)
            tmp = neg_I2.copy(); tmp[nan_mask] = 0
            neg_I2 = median_filter(tmp, size=k).astype(np.float32)
            tmp = I3.copy(); tmp[nan_mask] = 0
            I3 = median_filter(tmp, size=k).astype(np.float32)
            neg_I2[nan_mask] = np.nan
            I3[nan_mask]     = np.nan
        return neg_I2, I3

    def _on_compute(self):
        if self._selection is None:
            return

        sel = self._selection
        self.lbl_status.setText("Computing...")

        try:
            if sel["type"] == "line":
                self._compute_lumley_line(sel)
                self.tabs.setCurrentIndex(0)
            else:
                self._compute_barycentric_rect(sel)
                self.tabs.setCurrentIndex(1)

            suffix = ""
            if self.chk_smooth.isChecked():
                suffix = f"  [median filter k={self.spin_smooth.value()}]"
            self.lbl_status.setText(f"Done.{suffix}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText(f"Error: {e}")

    # ----------------------------------------------------------------------- #
    # Lumley triangle from line selection
    # ----------------------------------------------------------------------- #

    def _compute_lumley_line(self, sel):
        rows, cols, dist = points_near_line(
            self._x, self._y,
            sel["x0"], sel["y0"], sel["x1"], sel["y1"]
        )

        neg_I2_field, I3_field = self._get_smooth_fields()
        neg_II = neg_I2_field[rows, cols]
        III    = I3_field[rows, cols]

        # Remove NaN points
        valid  = np.isfinite(neg_II) & np.isfinite(III)
        neg_II = neg_II[valid]
        III    = III[valid]
        dist   = dist[valid]

        if len(neg_II) == 0:
            raise ValueError("No valid anisotropy data found along the selected line. "
                             "Check that the line passes through valid (non-masked) regions.")

        self.lumley_fig.clear()
        ax = self.lumley_fig.add_subplot(111)

        # Draw Lumley triangle boundary
        self._draw_lumley_boundary(ax)

        # Scatter points colored by distance along line
        sc = ax.scatter(III, neg_II, c=dist, cmap="plasma",
                        s=20, zorder=5, linewidths=0)
        self.lumley_fig.colorbar(sc, ax=ax, label="Distance along line [mm]")

        ax.set_xlabel("III  (third invariant)")
        ax.set_ylabel("-II  (second invariant)")
        ax.set_title(f"Lumley Triangle  ({len(neg_II)} points along line)")
        ax.grid(True, alpha=0.3)

        self._last_result = {"type": "line", "rows": rows, "cols": cols,
                              "dist": dist, "neg_II": neg_II, "III": III}
        self.btn_export.setEnabled(True)
        self.lumley_canvas.draw()

    def _draw_lumley_boundary(self, ax):
        """
        Draw Lumley triangle matching paper (Fig. 9c, 10):
          I2 = -0.5*bij*bji,  I3 = det(bij)
          Axisymmetric limits: I3 = +/-2*(-I2/3)^(3/2)

        Three boundary segments:
          1. Prolate (right):  l1=2a/3, l2=l3=-a/3,   a in [0,1]
             Tip: (I3=2/27, -I2=1/3)
          2. Oblate (left):    l1=l2=a/3, l3=-2a/3,   a in [0,0.5]
             Tip: (I3=-1/108, -I2=1/12)
          3. 2C top line:      straight from oblate tip to prolate tip
        """
        a = np.linspace(0, 1, 500)

        # --- Right boundary: axisymmetric expansion (prolate) ---
        l1 = 2*a/3;  l2 = -a/3;  l3 = -a/3
        negI2_p = 0.5*(l1**2 + l2**2 + l3**2)   # = 0.5*trace(b^2)
        I3_p    = l1 * l2 * l3                    # det of diagonal
        ax.plot(I3_p, negI2_p, "b-", linewidth=1.5,
                label="Axisym. expansion", zorder=4)

        # --- Left boundary: axisymmetric contraction (oblate) ---
        # Stop at a=0.5 where l3=-1/3 (two-component limit)
        a_o = np.linspace(0, 0.5, 500)
        l1 = a_o/3;  l2 = a_o/3;  l3 = -2*a_o/3
        negI2_o = 0.5*(l1**2 + l2**2 + l3**2)
        I3_o    = l1 * l2 * l3
        ax.plot(I3_o, negI2_o, "b-", linewidth=1.5,
                label="_nolegend_", zorder=4)

        # --- Top boundary: straight 2C limit line ---
        # From 2C-axisym (-1/108, 1/12) to 1-comp (2/27, 1/3)
        I3_2c   = np.array([-1/108, 2/27])
        negI2_2c = np.array([1/12,  1/3])
        ax.plot(I3_2c, negI2_2c, "b--", linewidth=1.5,
                label="Two-component limit", zorder=4)

        # --- Key limit points ---
        ax.plot(0, 0, "o", color="royalblue", markersize=8,
                label="Isotropic (3C)", zorder=6)
        ax.plot(2/27, 1/3, "r^", markersize=8,
                label="One-component (1C): I3=2/27, -I2=1/3", zorder=6)
        ax.plot(-1/108, 1/12, "gs", markersize=8,
                label="Two-component (2C): I3=-1/108, -I2=1/12", zorder=6)

        # I3=0 vertical line (plane-strain limit)
        ax.axvline(0, color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.6, label="Plane-strain (I3=0)")

        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.015, 0.08)
        ax.set_ylim(-0.005, 0.35)

    # ----------------------------------------------------------------------- #
    # Barycentric map from rectangle selection
    # ----------------------------------------------------------------------- #

    def _compute_barycentric_rect(self, sel):
        x  = self._x
        y  = self._y
        x0, x1 = min(sel["x0"], sel["x1"]), max(sel["x0"], sel["x1"])
        y0, y1 = min(sel["y0"], sel["y1"]), max(sel["y0"], sel["y1"])

        # Mask: only points inside rectangle
        mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

        if mask.sum() == 0:
            raise ValueError("No grid points found inside the rectangle.")

        # Extract sub-region bounding box indices for efficient plotting
        rows_in, cols_in = np.where(mask)
        r0, r1 = rows_in.min(), rows_in.max() + 1
        c0, c1 = cols_in.min(), cols_in.max() + 1

        b_sub   = self._b[r0:r1, c0:c1]
        x_sub   = x[r0:r1, c0:c1]
        y_sub   = y[r0:r1, c0:c1]
        mask_sub = mask[r0:r1, c0:c1]

        # Compute barycentric on sub-region
        C1c, C2c, C3c, RGB = compute_barycentric(b_sub)

        # Set outside-rectangle points to white
        RGB[~mask_sub] = 1.0

        # Also show mean speed as background context
        speed = np.sqrt(
            np.nanmean(self.dataset["U"], axis=2)**2 +
            np.nanmean(self.dataset["V"], axis=2)**2 +
            np.nanmean(self.dataset["W"], axis=2)**2
        )
        valid_frac = np.mean(self.dataset["valid"], axis=2)
        speed[valid_frac < 0.5] = np.nan

        self.bary_fig.clear()
        ax = self.bary_fig.add_subplot(111)

        # Full domain speed as grey background
        ax.contourf(x, y, speed, levels=40, cmap="Greys", alpha=0.35)

        # Barycentric RGB overlay on sub-region
        # pcolormesh needs RGB as [ny, nx, 3] with values in [0,1]
        ax.pcolormesh(x_sub, y_sub, RGB,
                      shading="auto", zorder=5)

        # Draw rectangle outline
        from matplotlib.patches import Rectangle as Rect
        ax.add_patch(Rect((x0, y0), x1-x0, y1-y0,
                          linewidth=1.5, edgecolor="red",
                          facecolor="none", zorder=6))

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Barycentric Map  (Red=1-comp, Green=2-comp, Blue=isotropic)")
        ax.set_aspect("equal")
        ax.set_facecolor("lightgrey")

        # Add colorbar legend as text annotations
        ax.text(0.02, 0.97, "Red = 1-component",
                transform=ax.transAxes, color="red",
                fontsize=8, va="top")
        ax.text(0.02, 0.91, "Green = 2-component",
                transform=ax.transAxes, color="green",
                fontsize=8, va="top")
        ax.text(0.02, 0.85, "Blue = Isotropic",
                transform=ax.transAxes, color="blue",
                fontsize=8, va="top")

        self.bary_canvas.draw()

        # Also populate the Lumley triangle with all points in the rectangle
        neg_I2_field, I3_field = self._get_smooth_fields()
        neg_II_sub = neg_I2_field[r0:r1, c0:c1][mask_sub]
        III_sub    = I3_field[r0:r1, c0:c1][mask_sub]
        valid      = np.isfinite(neg_II_sub) & np.isfinite(III_sub)

        if valid.sum() > 0:
            self.lumley_fig.clear()
            ax2 = self.lumley_fig.add_subplot(111)
            self._draw_lumley_boundary(ax2)

            # Color by x-position for spatial context
            x_pts = x_sub[mask_sub][valid]
            ax2.scatter(III_sub[valid], neg_II_sub[valid],
                        c=x_pts, cmap="coolwarm", s=10,
                        zorder=5, linewidths=0, alpha=0.7)
            ax2.set_xlabel("III")
            ax2.set_ylabel("-II")
            ax2.set_title(f"Lumley Triangle  ({valid.sum()} pts in rectangle)")
            ax2.grid(True, alpha=0.3)
            self.lumley_canvas.draw()

        self.lbl_status.setText(
            f"Done. {mask.sum()} points in rectangle."
        )

    def _on_export(self):
        if not hasattr(self, "_last_result"):
            return
        res = self._last_result

        if res["type"] == "line":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Lumley Data", "lumley_line.csv", "CSV Files (*.csv)"
            )
            if not path:
                return
            settings = {
                "Analysis"  : "Anisotropy Invariants - Line",
                "Snapshots" : self.dataset["Nt"],
                "Grid"      : f"{self.dataset['nx']} x {self.dataset['ny']}",
            }
            import numpy as np
            dist  = res["dist"]
            xpts  = self._x[res["rows"], res["cols"]]
            ypts  = self._y[res["rows"], res["cols"]]
            neg_II = res["neg_II"]
            I3     = res["III"]
            export_line_csv(path, dist, xpts, ypts,
                            {"-I2": neg_II, "I3": I3}, {}, settings)
            self.lbl_status.setText(f"Exported to {path}")

        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Barycentric Map", "barycentric.dat",
                "Tecplot DAT (*.dat)"
            )
            if not path:
                return
            settings = {
                "Analysis"  : "Anisotropy Invariants - Barycentric",
                "Snapshots" : self.dataset["Nt"],
                "Grid"      : f"{self.dataset['nx']} x {self.dataset['ny']}",
            }
            import numpy as np
            mask = res["mask"]
            r0, r1, c0, c1 = res["r0"], res["r1"], res["c0"], res["c1"]
            x_sub = self._x[r0:r1, c0:c1]
            y_sub = self._y[r0:r1, c0:c1]

            neg_I2_f, I3_f = self._get_smooth_fields()
            C1c, C2c, C3c, _ = compute_barycentric(self._b[r0:r1, c0:c1])

            neg_I2_sub = neg_I2_f[r0:r1, c0:c1].copy()
            I3_sub     = I3_f[r0:r1, c0:c1].copy()

            mask_sub = mask[r0:r1, c0:c1]
            for arr in [neg_I2_sub, I3_sub, C1c, C2c, C3c]:
                arr[~mask_sub] = np.nan

            export_2d_tecplot(
                path, x_sub, y_sub,
                [neg_I2_sub, I3_sub, C1c, C2c, C3c],
                ["-I2", "I3", "C1c_1comp", "C2c_2comp", "C3c_isotropic"],
                settings
            )
            self.lbl_status.setText(f"Exported to {path}")
