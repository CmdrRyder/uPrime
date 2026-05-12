# gui/mean_convergence_tab.py
# ----------------------------
# Mean & Convergence analysis window.
#
# Sub-tab 1 (Mean Velocity): field display + line profile of <U>, <V>, [<W>].
# Sub-tab 2 (Convergence): point-based cumulative statistics via Welford.
#
# Copyright (C) 2024  Jibu Tom Jose
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details. <https://www.gnu.org/licenses/>.

import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGroupBox, QPushButton, QRadioButton, QCheckBox,
    QSizePolicy, QMessageBox, QSplitter, QSpinBox,
    QDoubleSpinBox, QButtonGroup, QFileDialog,
    QApplication, QTabWidget,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRect

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from gui.comparison_mixin import ComparisonMixin
from gui.line_selector import LineSelectorWidget, compute_snapped_line
from core.reynolds_stress import extract_line_profile
from core.export import export_line_csv, _settings_header
from core.mean_convergence_core import (
    compute_mean_fields, compute_convergence, find_convergence_n,
)

_FONT_AX   = 9
_FONT_TICK = 8
_FONT_LEG  = 8

_COMP_COLOR = {"u": "tab:blue", "v": "tab:orange", "w": "tab:green"}
_ORDER_LS   = {1: "solid", 2: "dashed", 3: "dotted"}
_ORDER_LABEL = {1: "u′", 2: "u′u′", 3: "u′u′u′"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field_canvas_height(x, y, target_w=480):
    x_ext = float(np.nanmax(x) - np.nanmin(x))
    y_ext = float(np.nanmax(y) - np.nanmin(y))
    ratio = (y_ext / x_ext) if x_ext > 0 else 0.5
    return max(150, min(420, int(target_w * ratio) + 10))


def _draw_speed_field(ax, x, y, speed):
    """Draw viridis |<U>| contourf on ax and return the QuadContourSet."""
    cf = ax.contourf(x, y, np.ma.masked_invalid(speed), levels=40, cmap="viridis", extend="neither")
    ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
    ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=_FONT_TICK)
    return cf


# ===========================================================================
# Sub-tab 1 — Mean Velocity
# ===========================================================================

class MeanVelocityTab(PickerMixin, ComparisonMixin, QWidget):
    """
    Field display + line-profile for <U>, <V>, [<W>].
    Mirrors the Reynolds stress tab layout as closely as possible.
    """

    _module_name      = "Mean Velocity"
    _expected_columns = ["dist_mm", "mean_u", "mean_v"]
    _axis_columns     = ["dist_mm", "x_mm", "y_mm"]

    def __init__(self, dataset, mean_fields, parent=None):
        super().__init__(parent)
        self.dataset     = dataset
        self.mean_fields = mean_fields
        self.is_stereo   = dataset.get("is_stereo", False)

        self._mode         = "line"
        self._press_xy     = None
        self._press_button = None
        self._line_artist  = None
        self._selection    = None
        self._last_line_data = {"dist": None, "xpts": None, "ypts": None, "means": {}}

        self._build_ui()
        self._draw_field()
        self._connect_mouse()
        self._setup_picker(self.field_canvas, self.field_ax,
                           result_canvas=self.result_canvas,
                           result_ax=None,
                           status_label=self.lbl_status)

    # ------------------------------------------------------------------ #

    def _drawing_active(self):
        return True   # suppress red-cross — we always handle left-click

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left panel ----
        left = QWidget()
        left.setMinimumWidth(420)
        left.setMaximumWidth(560)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        self.field_fig    = Figure(constrained_layout=True)
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Fixed)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)

        self.line_sel = LineSelectorWidget(show_avg=True)
        ll.addWidget(self.line_sel)

        # Component checkboxes
        comp_grp = QGroupBox("Components to plot")
        comp_lay = QHBoxLayout(comp_grp)
        self.chk_U = QCheckBox("<U>"); self.chk_U.setChecked(True)
        self.chk_V = QCheckBox("<V>"); self.chk_V.setChecked(True)
        comp_lay.addWidget(self.chk_U)
        comp_lay.addWidget(self.chk_V)
        if self.is_stereo:
            self.chk_W = QCheckBox("<W>"); self.chk_W.setChecked(True)
            comp_lay.addWidget(self.chk_W)
        else:
            self.chk_W = None
        ll.addWidget(comp_grp)

        # U_m normalization — same style as Reynolds stress "Scale by Um²" row
        norm_grp = QGroupBox("Normalization")
        norm_lay = QHBoxLayout(norm_grp)
        self.chk_norm = QCheckBox("Normalize by U_m")
        self.chk_norm.setChecked(False)
        norm_lay.addWidget(self.chk_norm)
        norm_lay.addWidget(QLabel("U_m [m/s]:"))
        self.spin_um = QDoubleSpinBox()
        self.spin_um.setDecimals(6)
        self.spin_um.setRange(1e-6, 1e6)
        self.spin_um.setValue(1.0)
        self.spin_um.setSingleStep(0.1)
        self.spin_um.setFixedWidth(110)
        norm_lay.addWidget(self.spin_um)
        norm_lay.addStretch()
        ll.addWidget(norm_grp)

        self.lbl_hint = QLabel("Left-click+drag: draw line.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        self.btn_plot = QPushButton("Plot Line Profile")
        self.btn_plot.clicked.connect(self._on_plot)
        ll.addWidget(self.btn_plot)

        self.btn_export_csv = QPushButton("Export CSV…")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_csv.setEnabled(False)
        ll.addWidget(self.btn_export_csv)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        self._init_comparison_toolbar(ll)
        ll.addStretch()

        # ---- Right panel ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        self.result_fig    = Figure()
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([480, 1200])

    def _draw_field(self):
        ds    = self.dataset
        x, y  = ds["x"], ds["y"]
        speed = self.mean_fields["speed"].copy()

        self.field_canvas.setFixedHeight(_field_canvas_height(x, y))
        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        _draw_speed_field(self.field_ax, x, y, speed)
        self.field_ax.set_title(
            "Left-click+drag: line", fontsize=_FONT_AX - 1)
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()
        self._x = x
        self._y = y
        self._last_field_values = speed

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.field_ax:
            return
        if self._toolbar_active(self.field_toolbar):
            return
        if event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            self._press_xy     = (event.xdata, event.ydata)
            self._press_button = 1

    def _on_motion(self, event):
        if self._press_xy is None or self._press_button != 1:
            return
        if event.inaxes != self.field_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy     = None
            self._press_button = None
            return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        lmode = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, x1, y1, lmode)
        self._clear_line_artist()
        ln, = self.field_ax.plot(
            [lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10)
        self._line_artist = ln
        self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None or self._press_button != 1:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy     = None
            self._press_button = None
            return
        if event.xdata is None or event.ydata is None:
            self._press_xy     = None
            self._press_button = None
            return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._press_xy     = None
        self._press_button = None
        lmode = self.line_sel.get_mode()
        lx0, ly0, lx1, ly1 = compute_snapped_line(
            self._x, self._y, x0, y0, x1, y1, lmode)
        if abs(lx1 - lx0) < 0.1 and abs(ly1 - ly0) < 0.1:
            self._clear_line_artist()
            self.lbl_hint.setText("Line too short — try again.")
            return
        self._clear_line_artist()
        ln, = self.field_ax.plot(
            [lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10)
        self._line_artist = ln
        self.field_canvas.draw()
        self._selection = {"x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1}
        self.lbl_hint.setText(
            f"Line ({lmode}): ({lx0:.1f},{ly0:.1f}) → ({lx1:.1f},{ly1:.1f}) mm")

    def _clear_line_artist(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None

    # ---- Normalization helpers ----

    def _um_factor(self):
        """Return (scale_factor, y_label, um_header_str)."""
        if self.chk_norm.isChecked():
            um = self.spin_um.value()
            return 1.0 / um, "Mean velocity / U_m [-]", f"{um:.6f} m/s"
        return 1.0, "Mean velocity [m/s]", "N/A (raw values)"

    # ---- Plot ----

    def _on_plot(self):
        if self._selection is None:
            QMessageBox.information(self, "No Line",
                "Please draw a line first (left-click+drag).")
            return
        sel      = self._selection
        lmode    = self.line_sel.get_mode()
        avg_band = self.line_sel.get_avg_band()
        mf       = self.mean_fields
        scale, ylabel, _ = self._um_factor()

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)

        self._last_line_data = {"dist": None, "xpts": None, "ypts": None, "means": {}}
        plotted = False

        comp_label = {
            "u": ("<U>/U_m" if self.chk_norm.isChecked() else "<U>"),
            "v": ("<V>/U_m" if self.chk_norm.isChecked() else "<V>"),
            "w": ("<W>/U_m" if self.chk_norm.isChecked() else "<W>"),
        }
        for key, field, color in [
            ("u", mf["U_mean"],        _COMP_COLOR["u"]),
            ("v", mf["V_mean"],        _COMP_COLOR["v"]),
            ("w", mf.get("W_mean"),    _COMP_COLOR["w"]),
        ]:
            if field is None:
                continue
            chk = {"u": self.chk_U, "v": self.chk_V, "w": self.chk_W}.get(key)
            if chk is None or not chk.isChecked():
                continue
            vals, dist, xpts, ypts = extract_line_profile(
                field, self._x, self._y,
                sel["x0"], sel["y0"], sel["x1"], sel["y1"],
                mode=lmode, avg_band=avg_band)
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            ax.plot(dist[valid], (vals * scale)[valid], color=color,
                    label=comp_label[key], linewidth=1.5)
            self._last_line_data["dist"]       = dist
            self._last_line_data["xpts"]       = xpts
            self._last_line_data["ypts"]       = ypts
            self._last_line_data["means"][key] = vals   # raw m/s — scaled at export time
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No valid data along line",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            xlabel = {"horizontal": "x [mm]",
                      "vertical":   "y [mm]"}.get(lmode, "Arc length [mm]")
            ax.set_xlabel(xlabel, fontsize=_FONT_AX)
            ax.set_ylabel(ylabel, fontsize=_FONT_AX)
            ax.set_title("Mean Velocity Profile", fontsize=_FONT_AX)
            ax.legend(fontsize=_FONT_LEG)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=_FONT_TICK)
            self.btn_export_csv.setEnabled(True)

        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.lbl_status.setText("Line profile plotted.")

    # ---- ComparisonMixin interface ----

    def _validate_csv(self, df):
        cols = set(df.columns)
        return "dist_mm" in cols and bool(cols & {"mean_u", "mean_v", "mean_w"})

    def _get_case_um(self, case):
        """Read U_m string from source CSV comment header."""
        try:
            with open(case["source"], "r", encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("#"):
                        break
                    if "U_m" in line and ":" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return None

    def _plot_comparison(self, quantities, layout_mode):
        """Plot imported cases on the result canvas."""
        cases = [c for c in self._cases if c["data"] is not None]
        if not cases:
            return

        _, ylabel, cur_um_str = self._um_factor()
        warnings = []

        self.result_fig.clear()

        col_labels = {"mean_u": "<U>", "mean_v": "<V>", "mean_w": "<W>"}

        if layout_mode == "overlay":
            n_cols = len(quantities)
            for i, qty in enumerate(quantities):
                ax = self.result_fig.add_subplot(1, max(n_cols, 1), i + 1)
                for case in cases:
                    df = case["data"]
                    if "dist_mm" not in df.columns or qty not in df.columns:
                        continue
                    try:
                        x = np.asarray(df["dist_mm"].values, dtype=float)
                        y = np.asarray(df[qty].values, dtype=float)
                    except (ValueError, TypeError) as exc:
                        self.lbl_status.setText(
                            f"Compare error: '{case['name']}' has non-numeric values: {exc}")
                        continue
                    mask = np.isfinite(x) & np.isfinite(y)
                    if not np.any(mask):
                        continue
                    ax.plot(x[mask], y[mask],
                            color=case["color"], linestyle=case["linestyle"],
                            linewidth=1.5, label=case["name"])
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_xlabel("dist [mm]", fontsize=_FONT_AX)
                ax.set_ylabel(ylabel, fontsize=_FONT_AX)
                ax.set_title(col_labels.get(qty, qty), fontsize=_FONT_AX)
                ax.legend(fontsize=_FONT_LEG)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=_FONT_TICK)
        else:
            for i, case in enumerate(cases):
                ax = self.result_fig.add_subplot(1, len(cases), i + 1)
                df = case["data"]
                for qty in quantities:
                    if "dist_mm" not in df.columns or qty not in df.columns:
                        continue
                    try:
                        x = np.asarray(df["dist_mm"].values, dtype=float)
                        y = np.asarray(df[qty].values, dtype=float)
                    except (ValueError, TypeError) as exc:
                        self.lbl_status.setText(
                            f"Compare error: '{case['name']}' has non-numeric values: {exc}")
                        continue
                    mask = np.isfinite(x) & np.isfinite(y)
                    if not np.any(mask):
                        continue
                    ax.plot(x[mask], y[mask], linewidth=1.5,
                            label=col_labels.get(qty, qty))
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_xlabel("dist [mm]", fontsize=_FONT_AX)
                ax.set_title(case["name"], fontsize=_FONT_AX)
                ax.legend(fontsize=_FONT_LEG)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=_FONT_TICK)

        # Per-case U_m mismatch warnings
        cur_normalized = self.chk_norm.isChecked()
        for case in cases:
            um_header = self._get_case_um(case)
            imp_normalized = um_header is not None and "N/A" not in um_header
            mismatch = False
            if imp_normalized != cur_normalized:
                mismatch = True
            elif imp_normalized and cur_normalized:
                try:
                    imp_val = float(um_header.split()[0])
                    if abs(imp_val - self.spin_um.value()) > 1e-9:
                        mismatch = True
                except (ValueError, IndexError):
                    mismatch = True
            if mismatch:
                imp_um_display = um_header if um_header else "N/A"
                warnings.append(
                    f"Case '{case['name']}' U_m ({imp_um_display}) "
                    f"differs from current U_m ({cur_um_str}) — units may not match"
                )

        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

        if warnings:
            self.lbl_status.setText(" | ".join(warnings))
        else:
            self.lbl_status.setText(f"Comparison: {len(cases)} imported case(s).")

    # ---- Export / Import ----

    def _on_export_csv(self):
        ld = self._last_line_data
        if ld["dist"] is None:
            return
        try:
            cn = QApplication.instance()._session_case_name or "Data_1"
        except AttributeError:
            cn = "Data_1"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mean Velocity Profile",
            f"{cn}_mean_velocity_line.csv", "CSV Files (*.csv)")
        if not path:
            return

        scale, _, um_str = self._um_factor()
        settings = {
            "Analysis":  "Mean Velocity - Line Profile",
            "Snapshots": self.dataset["Nt"],
            "U_m":       um_str,
        }
        # Apply normalization to the exported values
        scaled_means = {k: v * scale for k, v in ld["means"].items()}
        export_line_csv(path, ld["dist"], ld["xpts"], ld["ypts"],
                        scaled_means, {}, settings)
        self.lbl_status.setText(f"Exported to {os.path.basename(path)}")


# ===========================================================================
# Sub-tab 2 — Convergence
# ===========================================================================

class ConvergenceTab(PickerMixin, QWidget):
    """
    Point-based cumulative convergence analysis using Welford's algorithm.
    """

    def __init__(self, dataset, mean_fields, parent=None):
        super().__init__(parent)
        self.dataset      = dataset
        self.mean_fields  = mean_fields
        self.is_stereo    = dataset.get("is_stereo", False)

        self._picked_ij   = None   # (row, col) grid indices of picked point
        self._picked_xy   = None   # (x_mm, y_mm) of picked point
        self._pick_marker = None   # matplotlib artist
        self._conv_result = None   # last convergence computation result

        self._build_ui()
        self._draw_field()
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    # ---- drawing_active: always suppress PickerMixin cross (we draw our own) ----

    def _drawing_active(self):
        return True

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root     = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left panel ----
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(420)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        self.field_fig    = Figure(constrained_layout=True)
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Fixed)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)
        self.field_canvas.mpl_connect("button_press_event", self._on_field_click)

        self.lbl_hint = QLabel("Left-click on field to pick a point.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        self.lbl_point = QLabel("Point: —")
        self.lbl_point.setStyleSheet("font-size: 11px;")
        ll.addWidget(self.lbl_point)

        # Kernel size
        kern_row = QHBoxLayout()
        kern_row.addWidget(QLabel("Kernel size:"))
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 51)
        self.spin_kernel.setValue(3)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.valueChanged.connect(self._ensure_odd_kernel)
        self.spin_kernel.valueChanged.connect(self._redraw_margin)
        kern_row.addWidget(self.spin_kernel)
        kern_row.addStretch()
        ll.addLayout(kern_row)

        # Thresholds
        thr_grp = QGroupBox("Convergence thresholds")
        thr_lay = QVBoxLayout(thr_grp)
        self._thr_spins = {}
        for order, symbol in [(1, "u′"), (2, "u′u′"), (3, "u′u′u′")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"ε ({symbol}):"))
            sp = QDoubleSpinBox()
            sp.setDecimals(5)
            sp.setRange(1e-6, 1.0)
            sp.setValue(1e-3)
            sp.setSingleStep(1e-4)
            sp.setFixedWidth(90)
            self._thr_spins[order] = sp
            row.addWidget(sp)
            row.addStretch()
            thr_lay.addLayout(row)
        ll.addWidget(thr_grp)

        # Order checkboxes
        order_grp = QGroupBox("Orders to plot")
        order_lay = QHBoxLayout(order_grp)
        self._order_chks = {}
        for order, label in [(1, "1st"), (2, "2nd"), (3, "3rd")]:
            chk = QCheckBox(label)
            chk.setChecked(True)
            self._order_chks[order] = chk
            order_lay.addWidget(chk)
        ll.addWidget(order_grp)

        # Layout radio
        layout_grp = QGroupBox("Plot layout")
        layout_lay = QHBoxLayout(layout_grp)
        self.rb_grid   = QRadioButton("Grid")
        self.rb_single = QRadioButton("Single (normalized)")
        self.rb_grid.setChecked(True)
        self._layout_bg = QButtonGroup()
        self._layout_bg.addButton(self.rb_grid)
        self._layout_bg.addButton(self.rb_single)
        layout_lay.addWidget(self.rb_grid)
        layout_lay.addWidget(self.rb_single)
        ll.addWidget(layout_grp)

        self.btn_compute = QPushButton("Compute Convergence")
        self.btn_compute.clicked.connect(self._on_compute)
        ll.addWidget(self.btn_compute)

        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        ll.addWidget(self.btn_export)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)
        ll.addStretch()

        # ---- Right panel ----
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        self.result_fig    = Figure()
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 1200])

    def _ensure_odd_kernel(self, val):
        if val % 2 == 0:
            self.spin_kernel.blockSignals(True)
            self.spin_kernel.setValue(val + 1)
            self.spin_kernel.blockSignals(False)

    def _draw_field(self):
        ds    = self.dataset
        x, y  = ds["x"], ds["y"]
        speed = self.mean_fields["speed"].copy()

        # Use actual canvas width when available (widget already shown); fall
        # back to 380 which matches Tab 2's narrower left panel (maxWidth=420).
        w = self.field_canvas.width()
        target_w = w if w > 50 else 380
        self.field_canvas.setFixedHeight(_field_canvas_height(x, y, target_w=target_w))
        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        _draw_speed_field(self.field_ax, x, y, speed)
        self.field_ax.set_title("Left-click to pick point", fontsize=_FONT_AX - 1)
        # Must assign _x/_y before _draw_margin_patches, which reads them
        self._x = x
        self._y = y
        self._draw_margin_patches()
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()
        self._last_field_values = speed

    def _get_margin_px(self):
        kernel = self.spin_kernel.value()
        return (kernel - 1) // 2

    def _draw_margin_patches(self):
        """Shade the boundary margin grey on the field axes."""
        ax     = self.field_ax
        x, y   = self._x, self._y
        margin = self._get_margin_px()
        if margin == 0:
            return

        ny, nx = x.shape
        # Grid spacing (uniform grid)
        dx = abs(float(x[0, 1] - x[0, 0]))
        dy = abs(float(y[1, 0] - y[0, 0]))
        mx = margin * dx
        my = margin * dy

        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))

        kw = dict(facecolor="gray", alpha=0.35, zorder=5, linewidth=0)
        ax.add_patch(MplRect((xmin, ymin), mx,           ymax - ymin, **kw))  # left
        ax.add_patch(MplRect((xmax - mx, ymin), mx,      ymax - ymin, **kw))  # right
        ax.add_patch(MplRect((xmin + mx, ymin), xmax - xmin - 2*mx, my, **kw))  # bottom
        ax.add_patch(MplRect((xmin + mx, ymax - my), xmax - xmin - 2*mx, my, **kw))  # top

    def _redraw_margin(self):
        self._draw_field()
        if self._picked_ij is not None:
            self._redraw_pick_marker()

    def _redraw_pick_marker(self):
        if self._picked_xy is None:
            return
        if self._pick_marker is not None:
            try:
                self._pick_marker.remove()
            except Exception:
                pass
        mkr, = self.field_ax.plot(
            self._picked_xy[0], self._picked_xy[1],
            "r+", markersize=14, markeredgewidth=2, zorder=20)
        self._pick_marker = mkr
        self.field_canvas.draw()

    # ------------------------------------------------------------------ #
    # Point picker
    # ------------------------------------------------------------------ #

    def _on_field_click(self, event):
        if event.button != 1 or event.inaxes != self.field_ax:
            return
        if self._toolbar_active(self.field_toolbar):
            return
        if event.xdata is None or event.ydata is None:
            return

        x_mm, y_mm = event.xdata, event.ydata
        x, y = self._x, self._y

        col = int(np.argmin(np.abs(x[0, :] - x_mm)))
        row = int(np.argmin(np.abs(y[:, 0] - y_mm)))
        ny, nx = x.shape

        margin = self._get_margin_px()
        if row < margin or row >= ny - margin or col < margin or col >= nx - margin:
            kernel = self.spin_kernel.value()
            self.lbl_status.setText(
                f"Point too close to boundary for kernel size {kernel}")
            return

        self._picked_ij = (row, col)
        self._picked_xy = (float(x[row, col]), float(y[row, col]))

        if self._pick_marker is not None:
            try:
                self._pick_marker.remove()
            except Exception:
                pass
        mkr, = self.field_ax.plot(
            self._picked_xy[0], self._picked_xy[1],
            "r+", markersize=14, markeredgewidth=2, zorder=20)
        self._pick_marker = mkr
        self.field_canvas.draw()

        self.lbl_point.setText(
            f"Point: i={row}, j={col}  "
            f"({self._picked_xy[0]:.2f}, {self._picked_xy[1]:.2f}) mm")
        self.lbl_status.setText("")

    # ------------------------------------------------------------------ #
    # Convergence computation
    # ------------------------------------------------------------------ #

    def _on_compute(self):
        if self._picked_ij is None:
            QMessageBox.information(self, "No Point",
                "Please click on the field to pick a point first.")
            return

        row, col = self._picked_ij
        kernel   = self.spin_kernel.value()
        self.lbl_status.setText("Computing…")
        QApplication.processEvents()
        try:
            result = compute_convergence(self.dataset, row, col, kernel)
            self._conv_result = result
            self._plot_convergence(result)
            self.btn_export.setEnabled(True)
            self.lbl_status.setText("Done.")
        except Exception as e:
            self.lbl_status.setText(f"Error: {e}")
        finally:
            QApplication.processEvents()

    def _plot_convergence(self, result):
        N_total    = result["N_total"]
        comps      = result["components"]
        stats      = result["stats"]
        N_arr      = np.arange(1, N_total + 1)
        orders     = [o for o in (1, 2, 3) if self._order_chks[o].isChecked()]
        thresholds = {o: self._thr_spins[o].value() for o in (1, 2, 3)}

        self.result_fig.clear()

        if self.rb_grid.isChecked():
            self._plot_grid(N_arr, comps, stats, orders, thresholds)
        else:
            self._plot_single(N_arr, comps, stats, orders, thresholds)

        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()

    def _scale_for(self, stat, order):
        fv = stat["final_var"]
        if order == 1:
            return float(np.sqrt(max(fv, 1e-30)))
        if order == 2:
            return float(max(fv, 1e-30))
        return float(max(fv, 1e-30) ** 1.5)

    def _q_for_order(self, stat, order):
        return {1: stat["mean1"], 2: stat["mean2"], 3: stat["mean3"]}[order]

    def _plot_grid(self, N_arr, comps, stats, orders, thresholds):
        n_rows = len(comps)   # 2 for planar, 3 for stereo
        n_cols = 3            # always 3 order columns
        comp_labels = {"u": "u′", "v": "v′", "w": "w′"}
        order_titles = {1: "1st moment", 2: "2nd moment", 3: "3rd moment"}

        axes = self.result_fig.subplots(n_rows, n_cols, sharex=True)
        if n_rows == 1:
            axes = [axes]

        for ri, comp in enumerate(comps):
            stat  = stats[comp]
            color = _COMP_COLOR.get(comp, "tab:gray")
            for ci, order in enumerate(range(1, 4)):
                ax = axes[ri][ci]
                scale   = self._scale_for(stat, order)
                q       = self._q_for_order(stat, order)
                q_final = float(q[-1])
                thr     = thresholds[order]
                N_star  = find_convergence_n(q, scale, thr) if order in orders else None

                if order in orders:
                    ax.plot(N_arr, q, color=color, linewidth=1.2,
                            label=f"{comp_labels.get(comp, comp)}")
                    ax.axhline(q_final + thr * scale, color="gray", linewidth=0.8,
                               linestyle="--", alpha=0.6)
                    ax.axhline(q_final - thr * scale, color="gray", linewidth=0.8,
                               linestyle="--", alpha=0.6)
                    if N_star is not None:
                        ax.axvline(N_star, color="red", linewidth=1.0,
                                   linestyle=":", alpha=0.8)
                        ax.text(N_star, 0.97, f"N*={N_star}", fontsize=6,
                                color="red", ha="left", va="top",
                                transform=ax.get_xaxis_transform(), clip_on=True)

                sym = comp_labels.get(comp, comp)
                moment_ylabels = {1: f"<{sym}>_N", 2: f"<{sym}²>_N",
                                  3: f"<{sym}³>_N"}
                ax.set_ylabel(moment_ylabels[order], fontsize=_FONT_AX - 1)
                ax.tick_params(labelsize=_FONT_TICK)
                ax.grid(True, alpha=0.25)
                if ri == 0:
                    ax.set_title(order_titles.get(order, ""), fontsize=_FONT_AX)
                if ri == n_rows - 1:
                    ax.set_xlabel("N (snapshots)", fontsize=_FONT_AX)

    def _plot_single(self, N_arr, comps, stats, orders, thresholds):
        ax = self.result_fig.add_subplot(111)
        comp_labels = {"u": "u′", "v": "v′", "w": "w′"}

        for comp in comps:
            stat  = stats[comp]
            color = _COMP_COLOR.get(comp, "tab:gray")
            for order in orders:
                scale   = self._scale_for(stat, order)
                q       = self._q_for_order(stat, order)
                q_final = float(q[-1])
                q_norm  = q / scale if scale != 0 else q
                thr     = thresholds[order]
                N_star  = find_convergence_n(q, scale, thr)

                lbl_suffix = "Not converged" if N_star is None else f"N*={N_star}"
                label = (f"{comp_labels.get(comp, comp)} ord{order} "
                         f"[{lbl_suffix}]")
                ax.plot(N_arr, q_norm, color=color,
                        linestyle=_ORDER_LS[order],
                        linewidth=1.2, label=label)

                if N_star is not None:
                    ax.axvline(N_star, color=color, linewidth=0.8,
                               linestyle=":", alpha=0.6)

                # Threshold band around this component's asymptote
                asym = q_final / scale if scale != 0 else 0.0
                ax.axhline(asym + thr, color=color, linewidth=0.8,
                           linestyle="--", alpha=0.4)
                ax.axhline(asym - thr, color=color, linewidth=0.8,
                           linestyle="--", alpha=0.4)

        ax.set_xlabel("N (snapshots)", fontsize=_FONT_AX)
        ax.set_ylabel("Running moment / scale", fontsize=_FONT_AX)
        ax.set_title("Convergence (normalized)", fontsize=_FONT_AX)
        ax.legend(fontsize=_FONT_LEG - 1)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=_FONT_TICK)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _on_export(self):
        if self._conv_result is None:
            return
        try:
            cn = QApplication.instance()._session_case_name or "Data_1"
        except AttributeError:
            cn = "Data_1"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Convergence Data",
            f"{cn}_convergence.csv", "CSV Files (*.csv)")
        if not path:
            return

        result  = self._conv_result
        N_total = result["N_total"]
        comps   = result["components"]
        stats   = result["stats"]
        kernel  = self.spin_kernel.value()
        row, col = self._picked_ij
        x_pt, y_pt = self._picked_xy
        thresholds = {o: self._thr_spins[o].value() for o in (1, 2, 3)}

        # Compute N* per quantity for header
        nstar_info = {}
        for comp in comps:
            stat = stats[comp]
            for order in (1, 2, 3):
                scale = self._scale_for(stat, order)
                q     = self._q_for_order(stat, order)
                ns    = find_convergence_n(q, scale, thresholds[order])
                nstar_info[f"N* {comp} order{order}"] = (
                    str(ns) if ns is not None else "Not converged")

        header_info = {
            "Analysis":          "Convergence Analysis",
            "Point index (i,j)": f"({row}, {col})",
            "Point coord (x,y)": f"({x_pt:.3f}, {y_pt:.3f}) mm",
            "Kernel size":       kernel,
            "Threshold ord1":    thresholds[1],
            "Threshold ord2":    thresholds[2],
            "Threshold ord3":    thresholds[3],
            "N_total":           N_total,
        }
        header_info.update(nstar_info)

        # Build columns
        col_names = ["N"]
        arrays    = [np.arange(1, N_total + 1)]
        for comp in comps:
            stat = stats[comp]
            for key, arr in [("mean1", stat["mean1"]),
                             ("mean2", stat["mean2"]),
                             ("mean3", stat["mean3"])]:
                order = {"mean1": 1, "mean2": 2, "mean3": 3}[key]
                symbol = {"u": "u'", "v": "v'", "w": "w'"}.get(comp, comp)
                moment = {1: f"<{symbol}>", 2: f"<{symbol}{symbol}>",
                          3: f"<{symbol}{symbol}{symbol}>"}[order]
                col_names.append(moment)
                arrays.append(arr)

        from datetime import datetime
        with open(path, "w", encoding="utf-8") as f:
            f.write(_settings_header(header_info))
            f.write("\n")
            f.write(",".join(col_names) + "\n")
            for i in range(N_total):
                row_vals = [str(arrays[0][i])]
                for arr in arrays[1:]:
                    v = arr[i]
                    row_vals.append(f"{v:.8g}" if np.isfinite(v) else "")
                f.write(",".join(row_vals) + "\n")

        self.lbl_status.setText(f"Exported to {os.path.basename(path)}")


# ===========================================================================
# Main container window
# ===========================================================================

class MeanConvergenceWindow(QWidget):
    """
    Standalone window: QTabWidget containing MeanVelocityTab and ConvergenceTab.
    Mean fields are computed once here and shared with both sub-tabs.
    """

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Mean & Convergence Analysis")
        self.resize(1700, 900)

        # Compute mean fields once; both sub-tabs share this dict
        self._mean_fields = compute_mean_fields(dataset)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        self._mean_tab = MeanVelocityTab(dataset, self._mean_fields)
        self._conv_tab = ConvergenceTab(dataset, self._mean_fields)
        tabs.addTab(self._mean_tab, "Mean Velocity")
        tabs.addTab(self._conv_tab, "Convergence")
        root.addWidget(tabs)
