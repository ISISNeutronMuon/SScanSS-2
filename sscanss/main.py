from contextlib import suppress
import logging
import multiprocessing
import sys
from PyQt5.Qt import QLocale, QApplication, QTimer
from OpenGL.plugins import FormatHandler
from sscanss.config import setup_logging, STATIC_PATH, IMAGES_PATH
from sscanss.ui.window.view import MainWindow


def ui_execute():
    # Create the GUI event loop
    app = QApplication(sys.argv)

    # Load global style
    with suppress(FileNotFoundError):
        with open(STATIC_PATH / 'style.css', 'rt') as stylesheet:
            style = stylesheet.read().replace('@Path', IMAGES_PATH.as_posix())
            app.setStyleSheet(style)

    window = MainWindow()
    window.show()
    window.updater.check(True)

    # Wait for 0.5 seconds before opening project dialog
    QTimer.singleShot(500, window.showNewProjectDialog)

    return app.exec()


def set_locale():
    locale = QLocale(QLocale.C)
    locale.setNumberOptions(QLocale.RejectGroupSeparator)
    QLocale.setDefault(locale)


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """
    Qt slots swallows exceptions but this ensures exceptions are logged
    """
    logging.error('An unhandled exception occurred!', exc_info=(exc_type, exc_value, exc_traceback))
    sys.exit(1)


def main():
    multiprocessing.freeze_support()
    setup_logging('main.log')
    set_locale()
    # Tells OpenGL to use the NumpyHandler for the Matrix44 objects
    FormatHandler('sscanss', 'OpenGL.arrays.numpymodule.NumpyHandler', ['sscanss.core.math.matrix.Matrix44'])
    logger = logging.getLogger(__name__)
    sys.excepthook = log_uncaught_exceptions

    logger.info('Started the application...')
    sys.exit(ui_execute())


if __name__ == '__main__':
    main()
