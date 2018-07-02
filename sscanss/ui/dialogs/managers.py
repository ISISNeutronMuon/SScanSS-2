from PyQt5 import QtCore, QtWidgets, QtGui
from sscanss.core.util import TransformType, DockFlag
from sscanss.ui.widgets import FormControl, FormGroup


class SampleManager(QtWidgets.QWidget):
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


class TransformDialog(QtWidgets.QWidget):
    def __init__(self, transform_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.type = transform_type

        self.main_layout = QtWidgets.QVBoxLayout()
        unit = 'mm' if self.type == TransformType.Translate else 'degrees'
        title_label = QtWidgets.QLabel('{} sample around X, Y, Z axis'.format(self.type.value))
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(10)

        self.createSampleComboBox()

        self.form_group = FormGroup()
        self.x_axis = FormControl('X', 0.0, required=True, unit=unit)
        self.x_axis.number = True
        self.y_axis = FormControl('Y', 0.0, required=True, unit=unit)
        self.y_axis.number = True
        self.z_axis = FormControl('Z', 0.0, required=True, unit=unit)
        self.z_axis.number = True
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.groupValidation.connect(self.formValidation)
        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(self.type.value)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addWidget(self.form_group)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.title = '{} Sample'.format(self.type.value)
        self.dock_flag = DockFlag.Upper
        self.setMinimumWidth(350)
        self.parent_model.sample_changed.connect(self.updateSampleList)

    def createSampleComboBox(self):
        self.combobox_container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel('Sample:')
        self.combobox = QtWidgets.QComboBox()
        view = self.combobox.view()
        view.setSpacing(4)  # Add spacing between list items

        layout.addWidget(label)
        layout.addWidget(self.combobox)
        layout.addSpacing(5)

        self.combobox_container.setLayout(layout)
        self.main_layout.addWidget(self.combobox_container)
        self.updateSampleList()

    def updateSampleList(self):
        self.combobox.clear()
        sample_list = ['All', *self.parent_model.sample.keys()]
        self.combobox.addItems(sample_list)
        if len(self.parent_model.sample) > 1:
            self.combobox_container.setVisible(True)
        else:
            self.combobox_container.setVisible(False)

    def formValidation(self, is_valid):
        if is_valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        angles_or_offset = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        selected_sample = self.combobox.currentText()
        self.parent.presenter.transformSample(angles_or_offset, selected_sample, self.type)
