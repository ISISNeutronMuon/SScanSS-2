from PyQt5 import QtCore, QtWidgets, QtGui


class SampleManager(QtWidgets.QDockWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.parent = parent
        self.parent_model = parent.presenter.model

        layout = QtWidgets.QHBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_widget.setMinimumHeight(350)
        self.list_widget.setSpacing(2)
        self.getSamples()

        layout.addWidget(self.list_widget)

        button_layout = QtWidgets.QVBoxLayout()
        self.delete_button = QtWidgets.QToolButton()
        self.delete_button.setIcon(QtGui.QIcon('../static/images/cross.png'))
        self.delete_button.clicked.connect(self.removeSample)
        button_layout.addWidget(self.delete_button)

        self.merge_button = QtWidgets.QToolButton()
        self.merge_button.setIcon(QtGui.QIcon('../static/images/merge.png'))
        button_layout.addWidget(self.merge_button)

        self.up_button = QtWidgets.QToolButton()
        self.up_button.setIcon(QtGui.QIcon('../static/images/arrow-up.png'))
        button_layout.addWidget(self.up_button)

        self.down_button = QtWidgets.QToolButton()
        self.down_button.setIcon(QtGui.QIcon('../static/images/arrow-down.png'))
        button_layout.addWidget(self.down_button)
        layout.addSpacing(10)
        layout.addLayout(button_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addStretch(1)
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(main_layout)
        self.setWidget(main_widget)

        parent.presenter.model.sample_changed.connect(self.getSamples)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.setWindowTitle('Samples')
        self.setMinimumWidth(350)

    def getSamples(self):
        self.list_widget.clear()
        samples = self.parent_model.project_data['sample'].keys()
        for sample in samples:
            item = QtWidgets.QListWidgetItem(sample)
            self.list_widget.addItem(item)

    def removeSample(self):
        items = []
        for item in self.list_widget.selectedItems():
            items.append(item.text())

        self.parent_model.removeMeshFromProject(items)


