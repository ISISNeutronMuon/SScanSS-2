from PyQt5 import QtCore, QtWidgets
from sscanss.core.util import TransformType, DockFlag, to_float
from sscanss.ui.widgets import FormControl, FormGroup


class TransformDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper

    def __init__(self, transform_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.type = transform_type

        self.main_layout = QtWidgets.QVBoxLayout()

        if self.type == TransformType.Custom:
            self.createCustomTransformWidget()
        else:
            self.createDefaultWidget()


        self.setMinimumWidth(350)
        self.parent_model.sample_changed.connect(self.updateSampleList)

    def createCustomTransformWidget(self):
        self.matrix = [[1.0000, 0.0000, 0.0000, 0.0000],
                       [0.0000, 1.0000, 0.0000, 0.0000],
                       [0.0000, 0.0000, 1.0000, 0.0000],
                       [0.0000, 0.0000, 0.0000, 1.0000]]

        self.main_layout.addSpacing(10)

        self.createSampleComboBox()

        self.load_matrix = QtWidgets.QPushButton('Load Matrix')
        self.load_matrix.clicked.connect(self.loadMatrix)
        self.show_matrix = QtWidgets.QPlainTextEdit(self.matrixToString())
        self.show_matrix.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.show_matrix.setReadOnly(True)
        self.main_layout.addWidget(self.show_matrix)

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton('Apply Transform')
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.load_matrix)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)
        self.title = 'Transform Sample with Matrix'

    def createDefaultWidget(self):
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

    def loadMatrix(self):
        matrix = self.parent.presenter.importTransformMatrix()
        if matrix:
            self.validateMatrix(matrix)
            self.show_matrix.setPlainText(self.matrixToString())

    def validateMatrix(self, matrix):
        values = []
        is_valid = []
        for row in matrix:
            temp = []
            for col in row:
                value, valid = to_float(col)
                temp.append(value)
                is_valid.append(valid)
            values.append(temp)

        if False in is_valid:
            self.formValidation(False)
            self.matrix = []
        else:
            self.formValidation(True)
            self.matrix = values

    def matrixToString(self):
        result = []
        for row in self.matrix:
            for col in row:
                result.append('{:.4f}\t'.format(col))
            result.append('\n')
        return ''.join(result)

    def createSampleComboBox(self):
        self.combobox_container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel('Sample:')
        self.combobox = QtWidgets.QComboBox()
        self.combobox.setView(QtWidgets.QListView())

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
        selected_sample = self.combobox.currentText()
        if self.type == TransformType.Custom:
            self.parent.presenter.transformSample(self.matrix, selected_sample, self.type)
        else:
            angles_or_offset = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
            self.parent.presenter.transformSample(angles_or_offset, selected_sample, self.type)
