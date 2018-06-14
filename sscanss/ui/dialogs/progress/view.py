from PyQt5 import QtCore, QtWidgets


class ProgressDialog(QtWidgets.QDialog):

    def __init__(self, message, parent=None):
        super().__init__(parent)

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setTextVisible(False)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        message = QtWidgets.QLabel(message)
        message.setAlignment(QtCore.Qt.AlignCenter)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addStretch(1)
        main_layout.addWidget(progress_bar)
        main_layout.addWidget(message)
        main_layout.addStretch(1)

        self.setLayout(main_layout)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setMinimumSize(300, 120)

    def keyPressEvent(self, _):
        """
        This ensure the user cannot close the dialog box with the Esc key
        """
        pass
