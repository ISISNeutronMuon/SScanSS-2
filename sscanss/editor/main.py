import logging
import pathlib
import sys
from PyQt6 import QtWidgets
from sscanss.editor.view import EditorWindow
from sscanss.config import setup_logging, load_stylesheet, handle_scaling


def main():
    handle_scaling()
    setup_logging('editor.log')
    app = QtWidgets.QApplication([])

    style = load_stylesheet("style.css")
    if style:
        app.setStyleSheet(style)

    window = EditorWindow()

    if sys.argv[1:]:
        file_path = sys.argv[1]
        if pathlib.PurePath(file_path).suffix == '.json':
            window.openFile(file_path)
        else:
            window.showMessage(f'{file_path} could not be opened because it has an unknown file type')

    window.show()
    exit_code = app.exec()
    logging.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
