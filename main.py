"""Punto de entrada de LocalScribe."""

import sys

from PySide6.QtWidgets import QApplication

from localscribe.ui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LocalScribe")
    app.setOrganizationName("LocalScribe")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
