"""
PIV Toolkit - Main entry point
Run this file to launch the application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("uPrime")
    app.setStyle("Fusion")          # clean cross-platform look
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
