import sys
from contextlib import suppress
from PyQt5.Qt import QApplication, QTimer
from sscanss.config import STATIC_PATH, IMAGES_PATH
from sscanss.ui.window.view import MainWindow


def execute():
    # Create the GUI event loop
    app = QApplication(sys.argv)

    # Load global style
    with suppress(FileNotFoundError):
        with open(STATIC_PATH / 'style.css', 'rt') as stylesheet:
            style = stylesheet.read().replace('@Path', IMAGES_PATH.as_posix())
            app.setStyleSheet(style)

    window = MainWindow()
    window.show()
    # Wait for 0.5 seconds before opening project dialog
    QTimer.singleShot(500, lambda: window.showNewProjectDialog())

    return app.exec()
