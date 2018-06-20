from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np


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
        self.list_widget.setMinimumHeight(300)
        self.list_widget.setSpacing(2)
        self.list_widget.itemSelectionChanged.connect(self.selectionChanged)
        self.getSamples()

        layout.addWidget(self.list_widget)

        button_layout = QtWidgets.QVBoxLayout()
        self.delete_button = QtWidgets.QToolButton()
        self.delete_button.setIcon(QtGui.QIcon('../static/images/cross.png'))
        self.delete_button.clicked.connect(self.removeSamples)
        button_layout.addWidget(self.delete_button)

        self.merge_button = QtWidgets.QToolButton()
        self.merge_button.setIcon(QtGui.QIcon('../static/images/merge.png'))
        self.merge_button.clicked.connect(self.mergeSamples)
        button_layout.addWidget(self.merge_button)

        self.priority_button = QtWidgets.QToolButton()
        self.priority_button.setIcon(QtGui.QIcon('../static/images/check.png'))
        self.priority_button.clicked.connect(self.makeFirstSample)
        button_layout.addWidget(self.priority_button)

        layout.addSpacing(10)
        layout.addLayout(button_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addStretch(1)
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(main_layout)
        self.setWidget(main_widget)

        self.parent_model.sample_changed.connect(self.getSamples)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.setWindowTitle('Samples')
        self.setMinimumWidth(350)

    def getSamples(self):
        self.list_widget.clear()
        samples = self.parent_model.sample.keys()
        if samples:
            self.list_widget.addItems(samples)
            self.list_widget.item(0).setIcon(QtGui.QIcon('../static/images/check.png'))

    def removeSamples(self):
        items = [item.text() for item in self.list_widget.selectedItems()]
        self.parent_model.removeMeshFromProject(items)

    def mergeSamples(self):
        items = [item.text() for item in self.list_widget.selectedItems()]
        if items and len(items) < 2:
            return
        samples = self.parent_model.sample
        new_mesh = samples.pop(items[0], None)
        for i in range(1, len(items)):
            old_mesh = samples.pop(items[i], None)
            count = new_mesh['vertices'].shape[0]
            new_mesh['vertices'] = np.vstack((new_mesh['vertices'], old_mesh['vertices']))
            new_mesh['indices'] = np.concatenate((new_mesh['indices'], old_mesh['indices'] + count))
            new_mesh['normals'] = np.vstack((new_mesh['normals'], old_mesh['normals']))

        name = self.parent_model.create_unique_key('merged')
        samples[name] = new_mesh
        self.getSamples()

    def makeFirstSample(self):
        item = self.list_widget.currentItem()
        if item:
            key = item.text()
            samples = self.parent_model.sample
            samples.move_to_end(key, last=False)
            self.getSamples()
            self.list_widget.setCurrentRow(0)

    def selectionChanged(self):
        if len(self.list_widget.selectedItems()) > 1:
            self.priority_button.setDisabled(True)
        else:
            self.priority_button.setEnabled(True)
