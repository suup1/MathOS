"""
main.py
Entry Point — Quant Research Terminal

Run with:
    python main.py
"""
#ABHI MAJA AYEGA NA BHIDUUU
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui.QuantApp import QuantApp


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Quant Research Terminal")
    app.setOrganizationName("MathOS")

    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = QuantApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
