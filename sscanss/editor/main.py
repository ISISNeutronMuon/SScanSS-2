from sscanss.editor.view import EditorWindow
from sscanss.config import setup_logging, load_stylesheet, turn_on_scaling
import logging
from PyQt5 import QtCore, QtWidgets
import sys


def main():
    turn_on_scaling()
    setup_logging('editor.log')
    app = QtWidgets.QApplication([])
    app.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)
    style = load_stylesheet("styleEditor.css")
    if style:
        app.setStyleSheet(style)

    window = EditorWindow()
    window.show()
    exit_code = app.exec_()
    logging.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
