"""
gui/spatial_spectra_window.py
Spatial + Spatiotemporal Spectral Analysis (tabbed)
"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QPushButton, QRadioButton, QCheckBox, QSizePolicy,
    QMessageBox, QSplitter, QSpinBox, QButtonGroup,
    QFileDialog, QApplication, QTabWidget, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRect
from core.spatial_spectra import spatial_psd_line, spatial_psd_roi
from core.spatiotemporal_spectra import compute_st_spectra
from core.export import export_spectra_csv
from gui.line_selector import compute_snapped_line
from gui.arrow_toolbar import DrawAwareToolbar, PickerMixin


class SpatialSpectraWindow(PickerMixin, QWidget):
    def __init__(self, dataset, is_time_resolved=False, fs=1000.0, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self._is_tr  = is_time_resolved
        self._fs     = fs
        self.setWindowTitle("Spatial & Spatiotemporal Spectral Analysis")
        self.resize(1700, 900)

        Nt = dataset["Nt"]
        if Nt < 2:
            QMessageBox.critical(self, "Insufficient Data",
                "Spatial spectra require multiple snapshots.")
            return
        if Nt < 1000:
            QMessageBox.warning(self, "Convergence Warning",
                f"Only {Nt} snapshots -- results may not be converged.")

        self._build_ui()
        self._draw_field()
        self._connect_mouse()
        self._setup_picker(self.field_canvas, self.field_ax,
                           status_label=self.lbl_status)

    # ----------------------------------------------------------------------- #
    # UI
    # ----------------------------------------------------------------------- #
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- LEFT: field + controls ----
        left = QWidget()
        left.setMinimumWidth(500); left.setMaximumWidth(600)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)

        self.field_fig    = Figure()
        self.field_canvas = FigureCanvas(self.field_fig)
        self.field_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.field_toolbar = DrawAwareToolbar(self.field_canvas, self)
        ll.addWidget(self.field_toolbar)
        ll.addWidget(self.field_canvas, stretch=6)

        # Tabs: Spatial / Spatiotemporal
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_spatial_tab(),        "Spatial E(k)")
        st_tab = self._build_st_tab()
        self.tabs.addTab(st_tab, "Spatiotemporal E(k,f)")
        if not self._is_tr:
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabToolTip(1, "Requires time-resolved data")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        ll.addWidget(self.tabs, stretch=0)

        self.lbl_hint = QLabel("Select a mode and draw on the field.")
        self.lbl_hint.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_hint.setWordWrap(True)
        ll.addWidget(self.lbl_hint, stretch=0)

        self.btn_compute = QPushButton("Compute")
        self.btn_compute.setEnabled(False)
        self.btn_compute.clicked.connect(self._on_compute)
        ll.addWidget(self.btn_compute, stretch=0)

        self.btn_export = QPushButton("Export Data...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        ll.addWidget(self.btn_export, stretch=0)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("color:gray;font-size:11px;")
        self.lbl_status.setWordWrap(True)
        ll.addWidget(self.lbl_status, stretch=0)

        # ---- RIGHT: result canvas ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.result_fig    = Figure()
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.result_toolbar = NavToolbar(self.result_canvas, self)
        rl.addWidget(self.result_toolbar)
        rl.addWidget(self.result_canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([920, 780])

        # Internal state
        self._mode        = "horizontal"
        self._press_xy    = None
        self._artist      = None
        self._selection   = None
        self._last_result = None

    def _build_spatial_tab(self):
        w  = QWidget()
        ll = QVBoxLayout(w)
        ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)

        # Selection mode
        sg = QGroupBox("Selection Mode")
        sl = QVBoxLayout(sg)
        mr = QHBoxLayout()
        self.rb_horiz = QRadioButton("Horizontal (kx)")
        self.rb_vert  = QRadioButton("Vertical (ky)")
        self.rb_roi   = QRadioButton("Rectangle ROI")
        self.rb_horiz.setChecked(True)
        bg = QButtonGroup(self)
        for rb in [self.rb_horiz, self.rb_vert, self.rb_roi]:
            bg.addButton(rb); mr.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        sl.addLayout(mr)
        ar = QHBoxLayout()
        ar.addWidget(QLabel("Spatial avg ± pts:"))
        self.spin_avg = QSpinBox()
        self.spin_avg.setRange(0, 20); self.spin_avg.setValue(0)
        ar.addWidget(self.spin_avg)
        sl.addLayout(ar)
        ll.addWidget(sg)

        # Welch params
        pg = QGroupBox("Welch Parameters")
        pl = QVBoxLayout(pg)
        Nt = self.dataset["Nt"]
        r1 = QHBoxLayout(); r1.addWidget(QLabel("Segment (pts):"))
        self.spin_nperseg = QSpinBox(); self.spin_nperseg.setRange(4, 10000)
        self.spin_nperseg.setValue(max(16, min(256, Nt // 4)))
        r1.addWidget(self.spin_nperseg); pl.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Overlap (pts):"))
        self.spin_overlap = QSpinBox(); self.spin_overlap.setRange(0, 10000)
        self.spin_overlap.setValue(max(8, min(128, Nt // 8)))
        r2.addWidget(self.spin_overlap); pl.addLayout(r2)
        self.chk_subtract = QCheckBox("Subtract spatial mean")
        self.chk_subtract.setChecked(True)
        pl.addWidget(self.chk_subtract)
        ll.addWidget(pg)

        # Display options
        dg = QGroupBox("Display")
        dl = QVBoxLayout(dg)
        self.chk_kolmogorov = QCheckBox("Show -5/3 slope")
        self.chk_kolmogorov.setChecked(True)
        dl.addWidget(self.chk_kolmogorov)
        cr = QHBoxLayout()
        self.chk_compensate = QCheckBox("Compensate k^α, α =")
        self.chk_compensate.setChecked(False)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 5.0)
        self.spin_alpha.setValue(5/3)
        self.spin_alpha.setDecimals(2)
        self.spin_alpha.setSingleStep(0.1)
        self.spin_alpha.setFixedWidth(60)
        cr.addWidget(self.chk_compensate)
        cr.addWidget(self.spin_alpha)
        dl.addLayout(cr)
        ll.addWidget(dg)

        # Components
        cg = QGroupBox("Components")
        cl = QHBoxLayout(cg)
        self.chk_u = QCheckBox("u"); self.chk_u.setChecked(True)
        self.chk_v = QCheckBox("v"); self.chk_v.setChecked(True)
        self.chk_w = QCheckBox("w")
        self.chk_w.setChecked(self.dataset["is_stereo"])
        self.chk_w.setEnabled(self.dataset["is_stereo"])
        cl.addWidget(self.chk_u); cl.addWidget(self.chk_v); cl.addWidget(self.chk_w)
        ll.addWidget(cg)
        return w

    def _build_st_tab(self):
        w  = QWidget()
        ll = QVBoxLayout(w)
        ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)

        # Line direction
        dg = QGroupBox("Line Direction")
        dl = QHBoxLayout(dg)
        self.rb_st_horiz = QRadioButton("Horizontal (kx)")
        self.rb_st_vert  = QRadioButton("Vertical (ky)")
        self.rb_st_horiz.setChecked(True)
        bg2 = QButtonGroup(self)
        for rb in [self.rb_st_horiz, self.rb_st_vert]:
            bg2.addButton(rb); dl.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        ll.addWidget(dg)

        # Avg band
        ar = QHBoxLayout()
        ar.addWidget(QLabel("Spatial avg ± pts:"))
        self.spin_st_avg = QSpinBox()
        self.spin_st_avg.setRange(0, 20); self.spin_st_avg.setValue(0)
        ar.addWidget(self.spin_st_avg)
        ll.addLayout(ar)

        # fs
        fsr = QHBoxLayout()
        fsr.addWidget(QLabel("fs [Hz]:"))
        self.spin_st_fs = QDoubleSpinBox()
        self.spin_st_fs.setRange(1, 1e6)
        self.spin_st_fs.setValue(self._fs)
        self.spin_st_fs.setDecimals(1)
        fsr.addWidget(self.spin_st_fs)
        ll.addLayout(fsr)

        # Convection velocity overlay
        cg = QGroupBox("Convection Velocity Overlay")
        cl = QVBoxLayout(cg)
        self.chk_uc = QCheckBox("Show Uc line")
        self.chk_uc.setChecked(False)
        cl.addWidget(self.chk_uc)
        ucr = QHBoxLayout()
        ucr.addWidget(QLabel("Uc [m/s]:"))
        self.spin_uc = QDoubleSpinBox()
        self.spin_uc.setRange(0.001, 1000)
        self.spin_uc.setValue(1.0)
        self.spin_uc.setDecimals(3)
        ucr.addWidget(self.spin_uc)
        cl.addLayout(ucr)
        ll.addWidget(cg)

        # Component
        cg2 = QGroupBox("Component")
        cl2 = QHBoxLayout(cg2)
        self.rb_st_u = QRadioButton("u"); self.rb_st_u.setChecked(True)
        self.rb_st_v = QRadioButton("v")
        self.rb_st_w = QRadioButton("w")
        self.rb_st_w.setEnabled(self.dataset["is_stereo"])
        bg3 = QButtonGroup(self)
        for rb in [self.rb_st_u, self.rb_st_v, self.rb_st_w]:
            bg3.addButton(rb); cl2.addWidget(rb)
        ll.addWidget(cg2)
        return w

    # ----------------------------------------------------------------------- #
    # Field plot
    # ----------------------------------------------------------------------- #
    def _draw_field(self):
        ds   = self.dataset
        x, y = ds["x"], ds["y"]
        from core.dataset_utils import get_masked
        speed = np.sqrt(np.nanmean(get_masked(ds, "U"), axis=2)**2 +
                        np.nanmean(get_masked(ds, "V"), axis=2)**2)
        speed[~ds["MASK"]] = np.nan
        self.field_fig.clear()
        self.field_ax = self.field_fig.add_subplot(111)
        self.field_ax.contourf(x, y, speed, levels=40, cmap="RdBu_r")
        self.field_ax.set_xlabel("x [mm]", fontsize=9)
        self.field_ax.set_ylabel("y [mm]", fontsize=9)
        self.field_ax.set_aspect("equal")
        self.field_ax.set_facecolor("white")
        self.field_ax.tick_params(labelsize=8)
        self.field_fig.tight_layout(pad=0.3)
        self.field_canvas.draw()
        self._x = x; self._y = y
        self._last_field_values = speed

    # ----------------------------------------------------------------------- #
    # Mode
    # ----------------------------------------------------------------------- #
    def _on_tab_changed(self, idx):
        self._clear_artist()
        self._selection = None
        self.btn_compute.setEnabled(False)
        self._update_hint()

    def _on_mode_changed(self):
        self._clear_artist()
        self._selection = None
        self.btn_compute.setEnabled(False)
        self._update_hint()

    def _update_hint(self):
        tab = self.tabs.currentIndex()
        if tab == 0:
            if self.rb_horiz.isChecked():
                self._mode = "horizontal"
                self.lbl_hint.setText("Drag horizontally to set y position and x range.")
            elif self.rb_vert.isChecked():
                self._mode = "vertical"
                self.lbl_hint.setText("Drag vertically to set x position and y range.")
            else:
                self._mode = "roi"
                self.lbl_hint.setText("Drag to draw a rectangle ROI.")
        else:
            if self.rb_st_horiz.isChecked():
                self._mode = "st_horizontal"
                self.lbl_hint.setText("Drag horizontally to select a row for E(kx,f).")
            else:
                self._mode = "st_vertical"
                self.lbl_hint.setText("Drag vertically to select a column for E(ky,f).")

    # ----------------------------------------------------------------------- #
    # Mouse
    # ----------------------------------------------------------------------- #
    def _connect_mouse(self):
        self.field_canvas.mpl_connect("button_press_event",   self._on_press)
        self.field_canvas.mpl_connect("button_release_event", self._on_release)
        self.field_canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.field_ax: return
        if self._toolbar_active(self.field_toolbar): return
        self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press_xy is None or event.inaxes != self.field_ax: return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None; return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._clear_artist()

        if self._mode == "roi":
            p = MplRect((min(x0,x1),min(y0,y1)),abs(x1-x0),abs(y1-y0),
                        linewidth=1.5,edgecolor="red",facecolor="red",
                        alpha=0.15,zorder=10)
            self.field_ax.add_patch(p); self._artist = p
        else:
            lm = ("horizontal" if self._mode in ("horizontal","st_horizontal")
                  else "vertical")
            lx0,ly0,lx1,ly1 = compute_snapped_line(self._x,self._y,x0,y0,x1,y1,lm)
            ln, = self.field_ax.plot([lx0,lx1],[ly0,ly1],"r-",linewidth=2,zorder=10)
            self._artist = ln
        self.field_canvas.draw()

    def _on_release(self, event):
        if self._press_xy is None: return
        if self._toolbar_active(self.field_toolbar):
            self._press_xy = None; return
        if event.inaxes != self.field_ax:
            self._press_xy = None; return
        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata
        self._press_xy = None

        if self._mode == "roi":
            if abs(x1-x0)<1 or abs(y1-y0)<1:
                self.lbl_hint.setText("Too small -- try again."); return
            self._selection = {"type":"roi","x0":x0,"x1":x1,"y0":y0,"y1":y1}
            self.lbl_hint.setText(
                f"ROI: x=[{min(x0,x1):.1f},{max(x0,x1):.1f}] "
                f"y=[{min(y0,y1):.1f},{max(y0,y1):.1f}] mm")
        else:
            lm = ("horizontal" if self._mode in ("horizontal","st_horizontal")
                  else "vertical")
            lx0,ly0,lx1,ly1 = compute_snapped_line(self._x,self._y,x0,y0,x1,y1,lm)
            if abs(lx1-lx0)<1 and abs(ly1-ly0)<1:
                self.lbl_hint.setText("Too short -- try again."); return
            self._selection = {"type":lm,"x0":lx0,"y0":ly0,"x1":lx1,"y1":ly1}
            self.lbl_hint.setText(
                f"Line ({lm}): ({lx0:.1f},{ly0:.1f})->({lx1:.1f},{ly1:.1f}) mm")

        # Redraw committed selection
        self._clear_artist()
        if self._selection["type"] == "roi":
            sel = self._selection
            x0r,x1r = min(sel["x0"],sel["x1"]),max(sel["x0"],sel["x1"])
            y0r,y1r = min(sel["y0"],sel["y1"]),max(sel["y0"],sel["y1"])
            p = MplRect((x0r,y0r),x1r-x0r,y1r-y0r,linewidth=2,
                        edgecolor="red",facecolor="red",alpha=0.15,zorder=10)
            self.field_ax.add_patch(p); self._artist = p
        else:
            sel = self._selection
            ln, = self.field_ax.plot([sel["x0"],sel["x1"]],
                                     [sel["y0"],sel["y1"]],
                                     "r-",linewidth=2,zorder=10)
            self._artist = ln
        self.field_canvas.draw()
        self.btn_compute.setEnabled(True)

    def _clear_artist(self):
        if self._artist is not None:
            try: self._artist.remove()
            except: pass
            self._artist = None
        self.field_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Compute
    # ----------------------------------------------------------------------- #
    def _on_compute(self):
        if self._selection is None: return
        tab = self.tabs.currentIndex()
        self.result_fig.clear()
        self.result_canvas.draw()
        self.lbl_status.setText("⏳ Busy: computing...")
        self.btn_compute.setEnabled(False)
        QApplication.processEvents()

        try:
            if tab == 0:
                self._compute_spatial()
            else:
                self._compute_st()
            self.btn_export.setEnabled(True)
            self.lbl_status.setText("✓ Done. Draw new selection to recompute.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText(f"Error: {e}")
        finally:
            self.btn_compute.setEnabled(True)

    def _compute_spatial(self):
        ds       = self.dataset
        nperseg  = self.spin_nperseg.value()
        noverlap = self.spin_overlap.value()
        subtract = self.chk_subtract.isChecked()
        avg_band = self.spin_avg.value()
        sel      = self._selection

        if noverlap >= nperseg:
            raise ValueError("Overlap must be less than segment length.")

        from core.dataset_utils import get_masked
        _U = get_masked(ds, "U"); _V = get_masked(ds, "V"); _W = get_masked(ds, "W")
        if sel["type"] == "roi":
            results, n_lines = spatial_psd_roi(
                _U, _V, _W, self._x, self._y,
                sel["x0"],sel["x1"],sel["y0"],sel["y1"],
                nperseg,noverlap,subtract)
            self._last_result = {"tab":"spatial","type":"roi",
                                 "results":results,"n_lines":n_lines}
            self._plot_spatial_roi(results, n_lines)
        else:
            direction = "x" if sel["type"]=="horizontal" else "y"
            k, psds = spatial_psd_line(
                _U, _V, _W, self._x, self._y,
                sel["x0"],sel["y0"],sel["x1"],sel["y1"],
                direction,avg_band,nperseg,noverlap,subtract)
            self._last_result = {"tab":"spatial","type":"line",
                                 "direction":direction,"k":k,"psds":psds}
            self._plot_spatial_line(k, psds, direction)

    def _compute_st(self):
        ds   = self.dataset
        sel  = self._selection
        fs   = self.spin_st_fs.value()
        avg  = self.spin_st_avg.value()
        direction = "x" if self._mode == "st_horizontal" else "y"

        from core.dataset_utils import get_masked
        k, f, psds = compute_st_spectra(
            get_masked(ds, "U"), get_masked(ds, "V"), get_masked(ds, "W"),
            self._x, self._y,
            sel["x0"],sel["y0"],sel["x1"],sel["y1"],
            direction,avg,fs)
        self._last_result = {"tab":"st","direction":direction,
                             "k":k,"f":f,"psds":psds}
        self._plot_st(k, f, psds, direction)

    # ----------------------------------------------------------------------- #
    # Plot: Spatial
    # ----------------------------------------------------------------------- #
    def _active_comps(self):
        c = []
        if self.chk_u.isChecked(): c.append("u")
        if self.chk_v.isChecked(): c.append("v")
        if self.chk_w.isChecked() and self.dataset["is_stereo"]: c.append("w")
        return c

    def _plot_psd_ax(self, ax, k, psd, label, color):
        if k is None or psd is None:
            ax.text(0.5,0.5,"No data",transform=ax.transAxes,
                    ha="center",va="center",fontsize=9)
            ax.set_title(label,fontsize=9); return

        mask  = k > 0
        kp    = k[mask]; p = psd[mask]
        comp  = self.chk_compensate.isChecked()
        alpha = self.spin_alpha.value()

        if comp and np.any(kp > 0):
            p_plot = p * kp**alpha
            ylabel = f"k^{alpha:.2f} · PSD [(m/s)²·(rad/m)^{alpha-1:.2f}]"
        else:
            p_plot = p
            ylabel = "PSD [(m/s)²/(rad/m)]"

        valid = np.isfinite(p_plot) & (p_plot > 0)
        if not np.any(valid):
            ax.text(0.5,0.5,"No valid data",transform=ax.transAxes,
                    ha="center",va="center"); return

        ax.loglog(kp[valid],p_plot[valid],color=color,linewidth=1.2,label=label)

        if self.chk_kolmogorov.isChecked():
            nv  = np.sum(valid); ilo = max(0,int(nv*0.10)); ihi = min(nv-1,int(nv*0.60))
            if ihi > ilo+2:
                klo = kp[valid][ilo]; khi = kp[valid][ihi]
                kl  = np.logspace(np.log10(klo),np.log10(khi),50)
                pa  = p_plot[valid][ilo]
                if comp:
                    # compensated: -5/3 + alpha slope -> flat if alpha=5/3
                    ref_slope = -5/3 + alpha
                    ax.loglog(kl,pa*(kl/klo)**ref_slope,"k--",
                              linewidth=1.5,alpha=0.7,
                              label=r"$k^{" + f"{ref_slope:.2f}" + r"}$")
                else:
                    ax.loglog(kl,pa*(kl/klo)**(-5/3),"k--",
                              linewidth=1.5,alpha=0.7,label=r"$k^{-5/3}$")

        ax.set_xlabel("k [rad/m]",fontsize=8)
        ax.set_ylabel(ylabel,fontsize=7)
        ax.set_title(label,fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True,which="both",alpha=0.3)
        ax.legend(fontsize=7)

    def _plot_spatial_line(self, k, psds, direction):
        comps = [c for c in self._active_comps() if psds.get(c) is not None]
        if not comps:
            self.lbl_status.setText("No valid spectra."); return
        colors = {"u":"tab:blue","v":"tab:orange","w":"tab:green"}
        dl = "x" if direction=="x" else "y"
        n=len(comps); ncols=min(n,2); nrows=(n+ncols-1)//ncols
        for i,comp in enumerate(comps):
            ax = self.result_fig.add_subplot(nrows,ncols,i+1)
            self._plot_psd_ax(ax,k,psds[comp],f"E_{comp}(k_{dl})",colors[comp])
        self.result_fig.tight_layout(pad=1.2)
        self.result_canvas.draw()

    def _plot_spatial_roi(self, results, n_lines):
        comps  = self._active_comps()
        colors = {"u":"tab:blue","v":"tab:orange","w":"tab:green"}
        pairs  = []
        for d in ["x","y"]:
            k = results[d]["k"]
            for c in comps:
                p = results[d]["psds"].get(c)
                if p is not None and k is not None:
                    pairs.append((k,p,f"E_{c}(k_{d}) [{n_lines[d]}L]",colors[c]))
        if not pairs:
            self.lbl_status.setText("No valid spectra."); return
        n=len(pairs); ncols=min(n,2); nrows=(n+ncols-1)//ncols
        for i,(k_,p_,lbl,col) in enumerate(pairs):
            ax = self.result_fig.add_subplot(nrows,ncols,i+1)
            self._plot_psd_ax(ax,k_,p_,lbl,col)
        self.result_fig.tight_layout(pad=1.2)
        self.result_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Plot: Spatiotemporal
    # ----------------------------------------------------------------------- #
    def _st_comp(self):
        if self.rb_st_u.isChecked(): return "u"
        if self.rb_st_v.isChecked(): return "v"
        return "w"

    def _plot_st(self, k, f, psds, direction):
        comp = self._st_comp()
        E    = psds.get(comp)
        if k is None or f is None or E is None:
            self.lbl_status.setText("No valid spatiotemporal spectrum."); return

        dl = "x" if direction=="x" else "y"

        # Trim DC (k=0, f=0) bins -- they dominate and obscure the plot
        k_plot = k[1:]; f_plot = f[1:]; E_plot = E[1:, 1:]

        # Clip extremely small values before log
        E_plot = np.where(E_plot > 0, E_plot, np.nan)
        logE   = np.log10(E_plot)

        self.result_fig.clear()
        ax = self.result_fig.add_subplot(111)

        # pcolormesh: x=k, y=f (transpose so k on x-axis)
        # Use log-spaced visual by plotting on log axes with pcolormesh on linear grid
        pcm = ax.pcolormesh(k_plot, f_plot, logE.T,
                            cmap="inferno", shading="nearest")
        self.result_fig.colorbar(pcm, ax=ax,
                                 label=r"$\log_{10}$ E(k,f) [(m/s)²/(rad/m·Hz)]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"k_{dl} [rad/m]", fontsize=10)
        ax.set_ylabel("f [Hz]", fontsize=10)
        ax.set_title(f"Spatiotemporal spectrum E_{comp}(k_{dl}, f)", fontsize=10)
        ax.grid(True, which="both", alpha=0.2)

        # Convection velocity line: f = Uc * k / (2*pi)
        if self.chk_uc.isChecked():
            Uc    = self.spin_uc.value()
            k_line = k_plot[k_plot > 0]
            f_line = Uc * k_line / (2 * np.pi)
            valid  = (f_line >= f_plot[0]) & (f_line <= f_plot[-1])
            if np.any(valid):
                ax.plot(k_line[valid], f_line[valid],
                        "w--", linewidth=1.5, alpha=0.8,
                        label=f"Uc = {Uc:.2f} m/s")
                ax.legend(fontsize=9)

        self.result_fig.tight_layout(pad=0.5)
        self.result_canvas.draw()

    # ----------------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------------- #
    def _on_export(self):
        if self._last_result is None: return
        res = self._last_result
        settings = {"Analysis": "Spectral", "Snapshots": self.dataset["Nt"]}

        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "spectrum.csv", "CSV (*.csv)")
        if not path: return

        if res["tab"] == "spatial":
            if res["type"] == "line":
                export_spectra_csv(path, res["k"], res["psds"], settings)
            else:
                combined = {}; k_ref = None
                for d in ["x","y"]:
                    k = res["results"][d]["k"]
                    if k is not None and k_ref is None: k_ref = k
                    for c,p in res["results"][d]["psds"].items():
                        if p is not None: combined[f"{c}_k{d}"] = p
                if k_ref is not None:
                    export_spectra_csv(path, k_ref, combined, settings)
        else:
            # ST: export k, f, E as CSV
            k = res["k"][1:]; f = res["f"][1:]
            comp = self._st_comp()
            E = res["psds"].get(comp)
            if E is not None:
                E = E[1:, 1:]
                rows = []
                rows.append("# " + "; ".join(f"{k_}={v}" for k_,v in settings.items()))
                rows.append(f"# k [rad/m] | f [Hz] | E_{comp}")
                for ik, kv in enumerate(k):
                    for jf, fv in enumerate(f):
                        rows.append(f"{kv:.6e},{fv:.6e},{E[ik,jf]:.6e}")
                with open(path, "w") as fp:
                    fp.write("\n".join(rows))

        self.lbl_status.setText(f"✓ Exported to {path}")
