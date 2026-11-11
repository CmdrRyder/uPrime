"""
uPrime - Open-source fluid velocity field analysis
Because u' matters
v0.9.1

Developed by Jibu Tom Jose
Transient Fluid Mechanics Lab, Technion
"""

import sys

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family']  = 'serif'
matplotlib.rcParams['font.serif']   = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

from PyQt6.QtWidgets import QApplication
from uprime.gui.main_window import MainWindow, apply_neutral_control_theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("uPrime")
    app.setStyle("Fusion")
    # Neutral-grey radio/checkbox indicators (no orange accent) app-wide.
    apply_neutral_control_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
