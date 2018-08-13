from PyQt5 import QtWidgets, QtGui
from sscanss.core.util import DockFlag, PointType
from sscanss.ui.widgets import NumpyModel


class SampleManager(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        layout = QtWidgets.QHBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_widget.setMinimumHeight(300)
        self.list_widget.setSpacing(2)
        self.list_widget.itemSelectionChanged.connect(self.onMultiSelection)
        self.updateSampleList()

        layout.addWidget(self.list_widget)

        button_layout = QtWidgets.QVBoxLayout()
        self.delete_button = QtWidgets.QToolButton()
        self.delete_button.setObjectName('ToolButton')
        self.delete_button.setIcon(QtGui.QIcon('../static/images/cross.png'))
        self.delete_button.clicked.connect(self.removeSamples)
        button_layout.addWidget(self.delete_button)

        self.merge_button = QtWidgets.QToolButton()
        self.merge_button.setObjectName('ToolButton')
        self.merge_button.setIcon(QtGui.QIcon('../static/images/merge.png'))
        self.merge_button.clicked.connect(self.mergeSamples)
        button_layout.addWidget(self.merge_button)

        self.priority_button = QtWidgets.QToolButton()
        self.priority_button.setObjectName('ToolButton')
        self.priority_button.setIcon(QtGui.QIcon('../static/images/check.png'))
        self.priority_button.clicked.connect(self.changeMainSample)
        button_layout.addWidget(self.priority_button)

        layout.addSpacing(10)
        layout.addLayout(button_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

        self.parent_model.sample_changed.connect(self.updateSampleList)
        self.title = 'Samples'
        self.dock_flag = DockFlag.Bottom
        self.setMinimumWidth(350)

    def updateSampleList(self):
        self.list_widget.clear()
        samples = self.parent_model.sample.keys()
        if samples:
            self.list_widget.addItems(samples)
            self.list_widget.item(0).setIcon(QtGui.QIcon('../static/images/check.png'))

    def removeSamples(self):
        keys = [item.text() for item in self.list_widget.selectedItems()]
        if keys:
            self.parent.presenter.deleteSample(keys)

    def mergeSamples(self):
        keys = [item.text() for item in self.list_widget.selectedItems()]
        if keys and len(keys) < 2:
            return

        self.parent.presenter.mergeSample(keys)
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def changeMainSample(self):
        item = self.list_widget.currentItem()

        if not item:
            return

        key = item.text()
        self.parent.presenter.changeMainSample(key)
        self.list_widget.setCurrentRow(0)

    def onMultiSelection(self):
        if len(self.list_widget.selectedItems()) > 1:
            self.priority_button.setDisabled(True)
        else:
            self.priority_button.setEnabled(True)


class PointManager(QtWidgets.QWidget):
    def __init__(self, point_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.point_type = point_type

        self.selected = None

        layout = QtWidgets.QHBoxLayout()
        self.table_view = QtWidgets.QTableView()
        self.updateTable()
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setMinimumHeight(300)
        self.table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_view.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        self.table_view.horizontalHeader().setMinimumSectionSize(40)
        self.table_view.horizontalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.table_view)

        button_layout = QtWidgets.QVBoxLayout()
        self.delete_button = QtWidgets.QToolButton()
        self.delete_button.setObjectName('ToolButton')
        self.delete_button.setIcon(QtGui.QIcon('../static/images/cross.png'))
        self.delete_button.clicked.connect(self.deletePoints)
        button_layout.addWidget(self.delete_button)

        self.move_up_button = QtWidgets.QToolButton()
        self.move_up_button.setObjectName('ToolButton')
        self.move_up_button.setIcon(QtGui.QIcon('../static/images/arrow-up.png'))
        self.move_up_button.clicked.connect(lambda: self.movePoint(-1))
        button_layout.addWidget(self.move_up_button)

        self.move_down_button = QtWidgets.QToolButton()
        self.move_down_button.setObjectName('ToolButton')
        self.move_down_button.setIcon(QtGui.QIcon('../static/images/arrow-down.png'))
        self.move_down_button.clicked.connect(lambda: self.movePoint(1))
        button_layout.addWidget(self.move_down_button)

        layout.addSpacing(10)
        layout.addLayout(button_layout)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = '{} Points'.format(self.point_type.value)
        self.dock_flag = DockFlag.Bottom
        self.setMinimumWidth(350)
        self.table_view.clicked.connect(self.onMultiSelection)
        if self.point_type == PointType.Fiducial:
            self.parent_model.fiducials_changed.connect(self.updateTable)
        elif self.point_type == PointType.Measurement:
            self.parent_model.measurement_points_changed.connect(self.updateTable)

    def updateTable(self):
        if self.point_type == PointType.Fiducial:
            points = self.parent_model.fiducials
        elif self.point_type == PointType.Measurement:
            points = self.parent_model.measurement_points

        self.table_model = NumpyModel(points, parent=self.table_view)
        self.table_model.editCompleted.connect(self.editPoints)
        self.table_view.setModel(self.table_model)
        if self.selected is not None:
            self.table_view.setCurrentIndex(self.selected)

    def deletePoints(self):
        selection_model = self.table_view.selectionModel()
        indices = [item.row() for item in selection_model.selectedRows()]
        if indices:
            self.parent.presenter.deletePoints(indices, self.point_type)
            self.selected = None

    def movePoint(self, offset):
        selection_model = self.table_view.selectionModel()
        index = [item.row() for item in selection_model.selectedRows()]

        if not index:
            return

        index_from = index[0]
        index_to = index_from + offset

        if 0 <= index_to < self.table_model.rowCount():
            self.selected = self.table_model.index(index_to, 0)
            self.parent.presenter.movePoints(index_from, index_to, self.point_type)

    def editPoints(self, row, new_value):
        self.parent.presenter.editPoints(row, new_value, self.point_type)

    def onMultiSelection(self):
        selection_model = self.table_view.selectionModel()
        indices = [item.row() for item in selection_model.selectedRows()]
        if len(indices) > 1:
            self.move_down_button.setDisabled(True)
            self.move_up_button.setDisabled(True)
        else:
            self.move_down_button.setEnabled(True)
            self.move_up_button.setEnabled(True)
