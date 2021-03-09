from contextlib import suppress
import logging
import multiprocessing
import pathlib
import sys
from PyQt5 import QtCore, QtWidgets
from sscanss.config import setup_logging, STATIC_PATH, IMAGES_PATH
from sscanss.ui.window.view import MainWindow


def ui_execute():
    # Create the GUI event loop
    app = QtWidgets.QApplication(sys.argv[:1])

    # Load global style
    with suppress(FileNotFoundError):
        with open(STATIC_PATH / 'style.css', 'rt') as stylesheet:
            style = stylesheet.read().replace('@Path', IMAGES_PATH.as_posix())
            app.setStyleSheet(style)

    window = MainWindow()

    wait_time = 500  # time for main window to show
    if sys.argv[1:]:
        filename = sys.argv[1]
        if pathlib.PurePath(filename).suffix == '.h5':
            window.openProject(filename)
        else:
            msg = f'{filename} could not be opened because it has an unknown file type'
            QtCore.QTimer.singleShot(wait_time, lambda: window.showMessage(msg))
    else:
        QtCore.QTimer.singleShot(wait_time, window.showNewProjectDialog)

    window.show()
    window.updater.check(True)
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
