import sys
from contextlib import suppress
from PyQt5.Qt import QApplication, QTimer
from sscanss.ui.windows.main.view import MainWindow


def execute():
    # Create the GUI event loop
    app = QApplication(sys.argv)

    # Load global style
    style = ''
    with suppress(FileNotFoundError):
        with open('../static/style.css', 'rt') as stylesheet:
            style = stylesheet.read()

    app.setStyleSheet(style)
    window = MainWindow()
    window.show()
    # Wait for 0.5 seconds before opening project dialog
    QTimer.singleShot(500, lambda: window.showNewProjectDialog())

    return app.exec()
