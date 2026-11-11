"""
gui/lumley_plot.py
-------------------
Single source of truth for the Lumley-triangle boundary artwork. Shared by
gui.anisotropy_window (draws it live under a computed scatter) and
gui.compare_window (draws it under exported-CSV comparisons across cases),
so both views render byte-for-byte the same limit lines.
"""

import numpy as np


def draw_lumley_boundary(ax, legend=True, set_limits=True):
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
    nolegend = "_nolegend_"
    a = np.linspace(0, 1, 500)

    # --- Right boundary: axisymmetric expansion (prolate) ---
    l1 = 2*a/3;  l2 = -a/3;  l3 = -a/3
    negI2_p = 0.5*(l1**2 + l2**2 + l3**2)   # = 0.5*trace(b^2)
    I3_p    = l1 * l2 * l3                    # det of diagonal
    ax.plot(I3_p, negI2_p, "b-", linewidth=1.5,
            label=("Axisym. expansion" if legend else nolegend), zorder=4)

    # --- Left boundary: axisymmetric contraction (oblate) ---
    # Stop at a=0.5 where l3=-1/3 (two-component limit)
    a_o = np.linspace(0, 0.5, 500)
    l1 = a_o/3;  l2 = a_o/3;  l3 = -2*a_o/3
    negI2_o = 0.5*(l1**2 + l2**2 + l3**2)
    I3_o    = l1 * l2 * l3
    ax.plot(I3_o, negI2_o, "b-", linewidth=1.5,
            label=nolegend, zorder=4)

    # --- Top boundary: straight 2C limit line ---
    # From 2C-axisym (-1/108, 1/12) to 1-comp (2/27, 1/3)
    I3_2c   = np.array([-1/108, 2/27])
    negI2_2c = np.array([1/12,  1/3])
    ax.plot(I3_2c, negI2_2c, "b--", linewidth=1.5,
            label=("Two-component limit" if legend else nolegend), zorder=4)

    # --- Key limit points ---
    ax.plot(0, 0, "o", color="royalblue", markersize=8,
            label=("Isotropic (3C)" if legend else nolegend), zorder=6)
    ax.plot(2/27, 1/3, "r^", markersize=8,
            label=("One-component (1C): I3=2/27, -I2=1/3" if legend else nolegend),
            zorder=6)
    ax.plot(-1/108, 1/12, "gs", markersize=8,
            label=("Two-component (2C): I3=-1/108, -I2=1/12" if legend else nolegend),
            zorder=6)

    # I3=0 vertical line (plane-strain limit)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
               label=("Plane-strain (I3=0)" if legend else nolegend))

    if legend:
        ax.legend(fontsize=7, loc="upper left")
    if set_limits:
        ax.set_xlim(-0.015, 0.08)
        ax.set_ylim(-0.005, 0.35)
