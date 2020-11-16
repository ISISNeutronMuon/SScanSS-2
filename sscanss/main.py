from contextlib import suppress
import logging
import multiprocessing
import sys
from PyQt5 import QtCore, QtWidgets
from sscanss.config import setup_logging, STATIC_PATH, IMAGES_PATH
from sscanss.ui.window.view import MainWindow


def ui_execute():
    # Create the GUI event loop
    app = QtWidgets.QApplication(sys.argv)

    # Load global style
    with suppress(FileNotFoundError):
        with open(STATIC_PATH / 'style.css', 'rt') as stylesheet:
            style = stylesheet.read().replace('@Path', IMAGES_PATH.as_posix())
            app.setStyleSheet(style)

    window = MainWindow()
    window.show()
    window.updater.check(True)

    # Wait for 0.5 seconds before opening project dialog
    QtCore.QTimer.singleShot(500, window.showNewProjectDialog)

    return app.exec()


def main():
    multiprocessing.freeze_support()
    setup_logging('main.log')
    logging.info('Started the application...')
    exit_code = ui_execute()
    logging.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
