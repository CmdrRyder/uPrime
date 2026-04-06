"""
uPrime - Open-source fluid velocity field analysis
Because u' matters
v0.3 - alpha

Developed by Jibu Tom Jose
Transient Fluid Mechanics Lab, Technion
"""

import sys
import os

# When running as a PyInstaller bundle, add the bundle path to sys.path
# so that 'gui' and 'core' subfolders are importable
if getattr(sys, "frozen", False):
    # Running as compiled exe -- base path is the temp extraction folder
    base_path = sys._MEIPASS
else:
    # Running as normal Python script
    base_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, base_path)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family']  = 'serif'
matplotlib.rcParams['font.serif']   = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("uPrime")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()