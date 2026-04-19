"""
gui/reynolds_window.py
----------------------
Reynolds stress analysis window.

Mouse protocol (consistent across all uPrime windows)
------------------------------------------------------
Left-click + drag  : draw line (in line-profile mode)
Right-click + drag : draw yellow ROI rectangle (in any mode -- reserved for
                     future ROI averaging; currently sets selection for
                     line-profile mode as a rectangle region)
Zoom / pan toolbar : suppresses all draw events
"""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGroupBox, QPushButton, QRadioButton, QCheckBox,
    QSizePolicy, QMessageBox, QSplitter, QComboBox,
    QDoubleSpinBox, QButtonGroup, QFileDialog, QProgressBar
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRect

from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin
from core.reynolds_stress import extract_line_profile
from core.export import export_2d_tecplot, export_line_csv
from gui.line_selector import LineSelectorWidget, compute_snapped_line


COMP_LABELS = {
    "uu": "<u'u'>",
    "vv": "<v'v'>",
    "ww": "<w'w'>",
    "uv": "<u'v'>",
    "uw": "<u'w'>",
    "vw": "<v'w'>",
}
COMP_COLORS = {
    "uu": "tab:blue", "vv": "tab:orange", "ww": "tab:green",
    "uv": "tab:red",  "uw": "tab:purple", "vw": "tab:brown",
}

_FONT_AX   = 9
_FONT_TICK = 8
_FONT_LEG  = 8


class ReynoldsWindow(PickerMixin, QWidget):

    def __init__(self, dataset, is_time_resolved=False, Nt_warn=2000,
                 duration_warn=2.0, parent=None):
        super().__init__(parent)
        self.dataset   = dataset
        self.setWindowTitle("Reynolds Stress Analysis")
        self.resize(1700, 900)

        self._mode         = "contour"
        self._press_xy     = None
        self._line_artist  = None
        self._roi_artist   = None
        self._selection    = None

        self._show_convergence_warning(is_time_resolved, Nt_warn, duration_warn)

        self._stresses  = None
        self._k         = None
        self._std       = None
        self._available = (["uu", "vv", "ww", "uv", "uw", "vw"]
                           if dataset.get("is_stereo", False)
                           else ["uu", "vv", "uv"])

        self._build_ui()
        self._draw_field()
        self._connect_mouse()
        # Setup picker AFTER _connect_mouse so it does not eat left-clicks
        self._setup_picker(self.field_canvas, self.field_ax,
                           result_canvas=self.result_canvas,
                           result_ax=None,
                           status_label=self.lbl_status)

        self._start_stress_computation()

    # ----------------------------------------------------------------------- #

    def _show_convergence_warning(self, is_tr, Nt, duration):
        if is_tr:
            if duration < 2.0:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Dataset is {duration:.2f} s (< 2 s).\n"
                    "Reynolds stress statistics may not be converged.")
        else:
            if Nt < 2000:
                QMessageBox.warning(self, "Convergence Warning",
                    f"Only {Nt} snapshots (< 2000 recommended).\n"
                    "Reynolds stress statistics may not be converged.")

    # ----------------------------------------------------------------------- #
    # Background stress computation
    # ----------------------------------------------------------------------- #

    def _start_stress_computation(self):
        from core.workers import ReynoldsWorker
        from core.dataset_utils import get_masked
        U = get_masked(self.dataset, "U")
        V = get_masked(self.dataset, "V")
        W = get_masked(self.dataset, "W")
        self._worker = ReynoldsWorker(U, V, W)
        self._worker.finished.connect(self._on_stress_result)
        self._worker.error.connect(self._on_stress_error)
        self._worker.start()

    def _on_stress_result(self, result):
        self._stresses = result['stresses']
        self._k        = result['k']
        self._std      = result['std']
        self.btn_plot.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Ready.")

    def _on_stress_error(self, tb_str):
        QMessageBox.critical(self, "Reynolds Stress Error", tb_str)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Error computing stresses — see dialog.")

    # ----------------------------------------------------------------------- #
    # PickerMixin override: suppress red-cross when we are drawing
    # ----------------------------------------------------------------------- #

    def _drawing_active(self):
        return self._mode == "line"

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left ----
        left = QWidget()
        left.setMinimumWidth(440)
        left.setMaximumWidth(540)
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        self.field_fig    = Figure(constrained_layout=True)
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Fixed)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas)

        mode_grp = QGroupBox("Plot Mode")
        mode_lay = QHBoxLayout(mode_grp)
        self.rb_contour = QRadioButton("2D Contour Map")
        self.rb_line    = QRadioButton("Line Profile")
        self.rb_contour.setChecked(True)
        self.rb_contour.toggled.connect(self._on_mode_changed)
        mode_lay.addWidget(self.rb_contour)
        mode_lay.addWidget(self.rb_line)
        ll.addWidget(mode_grp)

        self.line_sel = LineSelectorWidget(show_avg=True)
        self.line_sel.setVisible(False)
        ll.addWidget(self.line_sel)

        self.lbl_hint = QLabel(
            "Left-click+drag: line.   Select component and click 'Plot Contour'.")
        self.lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint)

        opt_grp = QGroupBox("Options")
        opt_lay = QVBoxLayout(opt_grp)

        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("Component:"))
        self.combo_comp = QComboBox()
        for c in self._available:
            self.combo_comp.addItem(COMP_LABELS.get(c, c), c)
        comp_row.addWidget(self.combo_comp)
        opt_lay.addLayout(comp_row)

        scale_row = QHBoxLayout()
        self.chk_scale = QCheckBox("Scale by Um\u00b2")
        self.chk_scale.setChecked(False)
        self.spin_um = QDoubleSpinBox()
        self.spin_um.setRange(0.001, 1000.0)
        self.spin_um.setValue(1.0)
        self.spin_um.setDecimals(3)
        self.spin_um.setSingleStep(0.1)
        scale_row.addWidget(self.chk_scale)
        scale_row.addWidget(QLabel("Um [m/s]:"))
        scale_row.addWidget(self.spin_um)
        opt_lay.addLayout(scale_row)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["RdBu_r", "hot_r", "viridis", "plasma", "seismic"])
        cmap_row.addWidget(self.combo_cmap)
        opt_lay.addLayout(cmap_row)

        ll.addWidget(opt_grp)

        self.line_comp_grp = QGroupBox("Components to overlay (line mode)")
        self.line_comp_grp.setVisible(False)
        lc_lay = QHBoxLayout(self.line_comp_grp)
        self.comp_chks = {}
        for c in self._available:
            chk = QCheckBox(COMP_LABELS.get(c, c))
            chk.setChecked(True)
            self.comp_chks[c] = chk
            lc_lay.addWidget(chk)
        ll.addWidget(self.line_comp_grp)

        self.btn_plot = QPushButton("Plot Contour")
        self.btn_plot.clicked.connect(self._on_plot)
        self.btn_plot.setEnabled(False)
        ll.addWidget(self.btn_plot)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        ll.addWidget(self.progress_bar)

        self.chk_std_band = QCheckBox("Show \u00b11\u03c3 band on line plot")
        self.chk_std_band.setChecked(True)
        ll.addWidget(self.chk_std_band)

        self.btn_export = QPushButton("Export Data...")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        ll.addWidget(self.btn_export)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status)

        # ---- Right ----
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self.result_fig    = Figure()
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = DrawAwareToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        chk_row = QHBoxLayout()
        chk_row.addStretch()
        self.chk_hide_axes = QCheckBox("Hide axes")
        self.chk_hide_axes.stateChanged.connect(self._on_plot)
        chk_row.addWidget(self.chk_hide_axes)
        self.chk_hide_colorbar = QCheckBox("Hide colorbar")
        self.chk_hide_colorbar.stateChanged.connect(self._on_plot)
        chk_row.addWidget(self.chk_hide_colorbar)
        rl.addLayout(chk_row)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([500, 1200])

    # ----------------------------------------------------------------------- #

    def _draw_field(self):
        ds = self.dataset
        x, y = ds["x"], ds["y"]
        from core.dataset_utils import get_masked
        _U = get_masked(ds, "U"); _V = get_masked(ds, "V"); _W = get_masked(ds, "W")
        speed = np.sqrt(np.nanmean(_U, axis=2)**2 + np.nanmean(_V, axis=2)**2)
        if _W is not None:
            speed = np.sqrt(speed**2 + np.nanmean(_W, axis=2)**2)
        speed[~ds["MASK"]] = np.nan

        # Fix canvas height to match data aspect ratio so set_aspect("equal")
        # fills the widget without white margins (same pattern as tke_budget_window).
        x_ext = float(np.nanmax(x) - np.nanmin(x))
        y_ext = float(np.nanmax(y) - np.nanmin(y))
        ratio = (y_ext / x_ext) if x_ext > 0 else 0.5
        target_w = 480   # left panel ~540px minus margins/toolbar
        target_h = max(150, min(420, int(target_w * ratio) + 10))
        self.field_canvas.setFixedHeight(target_h)

        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
        self.field_ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        self.field_ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        self.field_ax.set_title("Left-click+drag: line    Right-click+drag: ROI",
                                fontsize=_FONT_AX - 1)
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_ax.tick_params(labelsize=_FONT_TICK)
        self.field_canvas.draw()
        self.field_toolbar.set_home_limits()
        self._x = x
        self._y = y
        self._last_field_values = speed

    # ----------------------------------------------------------------------- #
    # Mode
    # ----------------------------------------------------------------------- #

    def _on_mode_changed(self):
        if self.rb_contour.isChecked():
            self._mode = "contour"
            self.btn_plot.setText("Plot Contour")
            self.lbl_hint.setText("Select component and click 'Plot Contour'.")
            self.line_comp_grp.setVisible(False)
            self.combo_comp.setVisible(True)
            self.line_sel.setVisible(False)
        else:
            self._mode = "line"
            self.btn_plot.setText("Plot Line Profile")
            self.lbl_hint.setText(
                "Left-click+drag to draw a line.  Right-click+drag for ROI.")
            self.line_comp_grp.setVisible(True)
            self.combo_comp.setVisible(False)
            self.line_sel.setVisible(True)
        self._clear_all_artists()
        self._selection = None

    # ----------------------------------------------------------------------- #
    # Mouse
    # ----------------------------------------------------------------------- #

    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.field_ax:
            return
        if self._toolbar_active(self.field_toolbar):
            return
        if event.button in (1, 3):
            self._press_xy = (event.xdata, event.ydata)
            self._press_button = event.button

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.field_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata

        if self._press_button == 3:
            # Right-drag: yellow ROI rectangle
            self._clear_roi_artist()
            p = MplRect(
                (min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
                linewidth=1.5, edgecolor="#e8a000", facecolor="#ffe066",
                alpha=0.25, linestyle="--", zorder=10)
            self.field_ax.add_patch(p)
            self._roi_artist = p
            self.field_canvas.draw()

        elif self._press_button == 1 and self._mode == "line":
            # Left-drag: red line
            lmode = self.line_sel.get_mode()
            lx0, ly0, lx1, ly1 = compute_snapped_line(
                self._x, self._y, x0, y0, x1, y1, lmode)
            self._clear_line_artist()
            ln, = self.field_ax.plot(
                [lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10)
            self._line_artist = ln
            self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None:
            return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None
            return
        if event.xdata is None or event.ydata is None:
            self._press_xy = None
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        btn    = self._press_button
        self._press_xy = None

        if btn == 3:
            # Commit ROI
            self._clear_roi_artist()
            if abs(x1 - x0) < 0.5 or abs(y1 - y0) < 0.5:
                self.lbl_hint.setText("ROI too small -- try again.")
                return
            xlo, xhi = min(x0, x1), max(x0, x1)
            ylo, yhi = min(y0, y1), max(y0, y1)
            p = MplRect(
                (xlo, ylo), xhi - xlo, yhi - ylo,
                linewidth=1.5, edgecolor="#e8a000", facecolor="#ffe066",
                alpha=0.25, linestyle="--", zorder=10)
            self.field_ax.add_patch(p)
            self._roi_artist = p
            self.field_canvas.draw()
            self.lbl_hint.setText(
                f"ROI: x=[{xlo:.1f},{xhi:.1f}]  y=[{ylo:.1f},{yhi:.1f}] mm")
            self.lbl_status.setText("ROI selected.")

        elif btn == 1 and self._mode == "line":
            # Commit line
            lmode = self.line_sel.get_mode()
            lx0, ly0, lx1, ly1 = compute_snapped_line(
                self._x, self._y, x0, y0, x1, y1, lmode)
            if abs(lx1 - lx0) < 0.1 and abs(ly1 - ly0) < 0.1:
                self.lbl_hint.setText("Line too short -- try again.")
                return
            self._clear_line_artist()
            ln, = self.field_ax.plot(
                [lx0, lx1], [ly0, ly1], "r-", linewidth=2, zorder=10)
            self._line_artist = ln
            self.field_canvas.draw()
            self._selection = {"x0": lx0, "y0": ly0, "x1": lx1, "y1": ly1}
            self.lbl_hint.setText(
                f"Line ({lmode}): ({lx0:.1f},{ly0:.1f}) -> ({lx1:.1f},{ly1:.1f}) mm")

    def _clear_line_artist(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None

    def _clear_roi_artist(self):
        if self._roi_artist is not None:
            try:
                self._roi_artist.remove()
            except Exception:
                pass
            self._roi_artist = None

    def _clear_all_artists(self):
        self._clear_line_artist()
        self._clear_roi_artist()
        self.field_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Scale
    # ----------------------------------------------------------------------- #

    def _scale_factor(self):
        if self.chk_scale.isChecked():
            um = self.spin_um.value()
            return 1.0 / (um ** 2), f"/ Um\u00b2 (Um={um:.2f} m/s)"
        return 1.0, "[m\u00b2/s\u00b2]"

    # ----------------------------------------------------------------------- #
    # Plot
    # ----------------------------------------------------------------------- #

    def _on_plot(self):
        if self._stresses is None:
            return
        if self._mode == "contour":
            self._plot_contour()
        else:
            if self._selection is None:
                QMessageBox.information(self, "No Line",
                    "Please draw a line on the field first (left-click+drag).")
                return
            self._plot_line()

    def _plot_contour(self):
        comp  = self.combo_comp.currentData()
        field = self._stresses[comp].copy()
        scale, unit_str = self._scale_factor()
        field *= scale
        field[~self.dataset["MASK"]] = np.nan
        cmap  = self.combo_cmap.currentText()

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)
        cf = ax.contourf(self._x, self._y, field, levels=50, cmap=cmap)
        cb = self.result_fig.colorbar(cf, ax=ax,
                                      label=f"{COMP_LABELS.get(comp, comp)} {unit_str}",
                                      shrink=0.8)
        if self.chk_hide_colorbar.isChecked():
            cb.remove()
            self.result_fig.tight_layout(pad=0.5)
        ax.set_xlabel("x [mm]", fontsize=_FONT_AX)
        ax.set_ylabel("y [mm]", fontsize=_FONT_AX)
        ax.set_title(f"Reynolds Stress: {COMP_LABELS.get(comp, comp)}",
                     fontsize=_FONT_AX)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=_FONT_TICK)
        if self.chk_hide_axes.isChecked():
            ax.axis('off')
            ax.set_title('')
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self._last_contour_comp = comp
        self.btn_export.setEnabled(True)
        self.lbl_status.setText(f"Contour: {comp}")

    def _plot_line(self):
        sel   = self._selection
        scale, unit_str = self._scale_factor()
        show_band = self.chk_std_band.isChecked()
        lmode     = self.line_sel.get_mode()
        avg_band  = self.line_sel.get_avg_band()

        self.result_fig.clear()
        # Landscape figure: wider than tall
        ax = self.result_fig.add_subplot(111)

        plotted = False
        self._last_line_data = {
            "dist": None, "xpts": None, "ypts": None,
            "means": {}, "stds": {}}

        for comp, chk in self.comp_chks.items():
            if not chk.isChecked():
                continue
            field = self._stresses[comp]
            if field is None:
                continue

            vals, dist, xpts, ypts = extract_line_profile(
                field * scale, self._x, self._y,
                sel["x0"], sel["y0"], sel["x1"], sel["y1"],
                mode=lmode, avg_band=avg_band)

            std_field = self._std.get(comp)
            std_vals  = None
            if std_field is not None:
                std_vals, _, _, _ = extract_line_profile(
                    std_field * scale, self._x, self._y,
                    sel["x0"], sel["y0"], sel["x1"], sel["y1"],
                    mode=lmode, avg_band=avg_band)

            valid = np.isfinite(vals)
            if not np.any(valid):
                continue

            color = COMP_COLORS[comp]
            ax.plot(dist[valid], vals[valid], color=color,
                    label=COMP_LABELS.get(comp, comp), linewidth=1.5)

            if show_band and std_vals is not None:
                ax.fill_between(dist[valid],
                                vals[valid] - std_vals[valid],
                                vals[valid] + std_vals[valid],
                                alpha=0.2, color=color)

            self._last_line_data["dist"]  = dist
            self._last_line_data["xpts"]  = xpts
            self._last_line_data["ypts"]  = ypts
            self._last_line_data["means"][comp] = vals
            self._last_line_data["stds"][comp]  = std_vals
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No valid data along line",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            xlabel = {"horizontal": "x [mm]",
                      "vertical":   "y [mm]"}.get(lmode, "Distance from origin [mm]")
            ax.set_xlabel(xlabel, fontsize=_FONT_AX)
            ax.set_ylabel(f"Reynolds Stress {unit_str}", fontsize=_FONT_AX)
            ax.set_title("Reynolds Stress Profile", fontsize=_FONT_AX)
            ax.legend(fontsize=_FONT_LEG)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=_FONT_TICK)
            self.btn_export.setEnabled(True)

        if self.chk_hide_axes.isChecked():
            ax.axis('off')
        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()
        self.result_toolbar.set_home_limits()
        self.lbl_status.setText("Line profile plotted.")

    # ----------------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------------- #

    def _on_export(self):
        scale, unit_str = self._scale_factor()
        is_line = self._mode == "line"

        if is_line:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Line Profile", "reynolds_line.csv",
                "CSV Files (*.csv)")
            if not path:
                return
            ld = self._last_line_data
            settings = {
                "Analysis"    : "Reynolds Stress - Line Profile",
                "Snapshots"   : self.dataset["Nt"],
                "Scaled by Um": (f"Yes, Um={self.spin_um.value():.3f} m/s"
                                 if self.chk_scale.isChecked() else "No"),
            }
            export_line_csv(path, ld["dist"], ld["xpts"], ld["ypts"],
                            ld["means"], ld["stds"], settings)
            self.lbl_status.setText(f"Exported to {path}")
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export 2D Field", "reynolds_stresses_all.dat",
                "Tecplot DAT (*.dat);;CSV Files (*.csv)")
            if not path:
                return
            fields, labels = [], []
            for comp in self._available:
                f = self._stresses[comp].copy() * scale
                f[~self.dataset["MASK"]] = np.nan
                fields.append(f)
                labels.append(f"{COMP_LABELS.get(comp, comp)} {unit_str}")
            settings = {
                "Analysis"    : "Reynolds Stresses - All Components",
                "Snapshots"   : self.dataset["Nt"],
                "Scaled by Um": (f"Yes, Um={self.spin_um.value():.3f} m/s"
                                 if self.chk_scale.isChecked() else "No"),
            }
            export_2d_tecplot(path, self._x, self._y, fields, labels, settings)
            self.lbl_status.setText(
                f"Exported {len(fields)} stress components to {os.path.basename(path)}")
