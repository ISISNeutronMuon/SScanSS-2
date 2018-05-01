import sys

from PyQt5.Qt import QApplication
from sscanss.ui.windows.main.view import MainWindow


def execute():
    # create the GUI event loop
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()
