from PyQt5 import QtWidgets


class Preferences(QtWidgets.QDialog):

    def __init__(self, parent):
        super().__init__(parent)

        self.widgets = {}
        layout = QtWidgets.QVBoxLayout()
        self.main_layout = QtWidgets.QHBoxLayout()

        self.category_list = QtWidgets.QTreeWidget()
        self.category_list.header().setVisible(False)
        self.category_list.itemClicked.connect(self.switchCategory)
        self.category_list.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.populateCategories()

        self.stack = QtWidgets.QStackedLayout()
        self.createForms()

        self.main_layout.addWidget(self.category_list)
        self.main_layout.addLayout(self.stack)

        layout.addLayout(self.main_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.accept_button = QtWidgets.QPushButton('Accept')
        self.accept_button.clicked.connect(self.accept)
        self.default_button = QtWidgets.QPushButton('Reset to Default')
        self.default_button.clicked.connect(self.resetToDefaults)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setDefault(True)
        button_layout.addStretch(1)

        button_layout.addWidget(self.default_button)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setWindowTitle('Preferences')
        self.setMinimumSize(640, 480)

    def createForms(self):
        pass

    def populateCategories(self):
        pass

    def switchCategory(self, item):
        pass

    def resetToDefaults(self):
        super().accept()

    def accept(self):
        super().accept()
