"""Application entry point for LocalScribe."""

import sys

from PySide6.QtWidgets import QApplication

from localscribe import __version__
from localscribe.theme import app_icon
from localscribe.ui import MainWindow


def main() -> None:
    if sys.platform == "win32":
        # Own taskbar identity so Windows shows the LocalScribe icon, not Python's.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"LocalScribe.LiveTranscriber.{__version__}"
        )
    app = QApplication(sys.argv)
    app.setApplicationName("LocalScribe")
    app.setOrganizationName("LocalScribe")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
