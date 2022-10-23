import logging
import multiprocessing
import pathlib
import sys
from sys import platform
from PyQt5 import QtCore, QtWidgets, QtGui
from sscanss.config import setup_logging, load_stylesheet, ProcessServer, handle_scaling, path_for
from sscanss.app.window.view import MainWindow


def ui_execute():
    """Creates main window and executes GUI event loop

    :return: exit code
    :rtype: int
    """
    handle_scaling()

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)
    app.setWindowIcon(QtGui.QIcon(path_for('logo.png')))

    if platform == 'darwin':
        style = load_stylesheet("mac_style.css")
    else:
        style = load_stylesheet("style.css")
    # Load global style
    if style:
        app.setStyleSheet(style)

    window = MainWindow()

    wait_time = 500  # time for main window to show
    if sys.argv[1:]:
        filename = sys.argv[1]
        if pathlib.PurePath(filename).suffix == '.h5':
            window.presenter.openProject(filename)
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
    ProcessServer()  # initialize process server
    setup_logging('main.log')
    logging.info('Started the application...')
    exit_code = ui_execute()
    logging.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
