from view import *
from sscanss.config import setup_logging, STATIC_PATH, IMAGES_PATH, ProcessServer
from contextlib import suppress
import sys


def setStyleSheet(app):  # Copied method from the appMain, I think something like this should be a common function
    with suppress(FileNotFoundError):
        with open(STATIC_PATH / 'style.css', 'rt') as stylesheet:
            style = stylesheet.read().replace('@Path', IMAGES_PATH.as_posix())

        app.setStyleSheet(style)


def main():
    setup_logging('editor.log')
    app = QtWidgets.QApplication([])
    app.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)
    setStyleSheet(app)

    window = EditorWindow()

    if sys.argv[1:]:
        file_path = sys.argv[1]
        if pathlib.PurePath(file_path).suffix == '.json':
            window.openFile(file_path)
        else:
            window.message.setText(f'{file_path} could not be opened because it has an unknown file type')

    window.show()
    exit_code = app.exec_()
    logging.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
