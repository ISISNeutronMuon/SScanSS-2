from enum import Enum, unique
from PyQt5 import QtWidgets
from sscanss.core.geometry import BoundingBox
from sscanss.core.math import is_close, Matrix44
from sscanss.core.util import TransformType, DockFlag
from sscanss.ui.widgets import FormControl, FormGroup


class TransformDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper

    def __init__(self, transform_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.type = transform_type

        self.main_layout = QtWidgets.QVBoxLayout()

        title_label = QtWidgets.QLabel()
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(10)
        self.tool = None

        self.createSampleComboBox()

        if self.type == TransformType.Rotate:
            title_label.setText('{} sample around X, Y, Z axis'.format(self.type.value))
            self.tool = RotateTool(self.combobox.currentText(), parent)
            self.main_layout.addWidget(self.tool)
            self.title = '{} Sample'.format(self.type.value)
        elif self.type == TransformType.Translate:
            title_label.setText('{} sample around X, Y, Z axis'.format(self.type.value))
            self.tool = TranslateTool(self.combobox.currentText(), parent)
            self.main_layout.addWidget(self.tool)
            self.title = '{} Sample'.format(self.type.value)
        elif self.type == TransformType.Custom:
            title_label.setText('Transform sample with arbitrary matrix')
            self.tool = CustomTransformTool(self.combobox.currentText(), parent)
            self.main_layout.addWidget(self.tool)
            self.title = 'Transform Sample with Matrix'
        else:
            title_label.setText('Move origin with respect to sample bounds')
            self.tool = MoveOriginTool(self.combobox.currentText(), parent)
            self.main_layout.addWidget(self.tool)
            self.title = 'Move Origin to Sample'

        self.setLayout(self.main_layout)
        self.setMinimumWidth(350)

        self.combobox.activated[str].connect(self.changeSample)
        self.parent_model.sample_changed.connect(self.updateSampleList)

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
        self.updateSampleList()

        self.combobox_container.setLayout(layout)
        self.main_layout.addWidget(self.combobox_container)

    def updateSampleList(self):
        self.combobox.clear()
        sample_list = ['All', *self.parent_model.sample.keys()]
        self.combobox.addItems(sample_list)
        self.changeSample(self.combobox.currentText())
        if len(self.parent_model.sample) > 1:
            self.combobox_container.setVisible(True)
        else:
            self.combobox_container.setVisible(False)

    def changeSample(self, new_sample):
        if self.tool is not None:
            self.tool.selected_sample = new_sample


class RotateTool(QtWidgets.QWidget):
    def __init__(self, sample, parent):
        super().__init__()

        self.parent = parent

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        unit = 'degrees'

        self.form_group = FormGroup()
        self.x_rotation = FormControl('X', 0.0, required=True, desc=unit, number=True)
        self.x_rotation.range(-360.0, 360.0)
        self.y_rotation = FormControl('Y', 0.0, required=True, desc=unit, number=True)
        self.y_rotation.range(-360.0, 360.0)
        self.z_rotation = FormControl('Z', 0.0, required=True, desc=unit, number=True)
        self.z_rotation.range(-360.0, 360.0)
        self.form_group.addControl(self.x_rotation)
        self.form_group.addControl(self.y_rotation)
        self.form_group.addControl(self.z_rotation)
        self.form_group.groupValidation.connect(self.formValidation)
        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(TransformType.Rotate.value)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addWidget(self.form_group)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.valid_sample = False
        self.selected_sample = sample

    @property
    def selected_sample(self):
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        self.valid_sample = True if self.parent.presenter.model.sample else False
        self.form_group.validateGroup()

    def formValidation(self, is_valid):
        if is_valid and self.valid_sample:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        angles = [self.x_rotation.value, self.y_rotation.value, self.z_rotation.value]
        if not is_close(angles, [0.0, 0.0, 0.0]):
            self.parent.presenter.transformSample(angles, self.selected_sample, TransformType.Rotate)


class TranslateTool(QtWidgets.QWidget):
    def __init__(self, sample, parent):
        super().__init__()

        self.parent = parent

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        unit = 'mm'

        self.form_group = FormGroup()
        self.x_position = FormControl('X', 0.0, required=True, desc=unit, number=True)
        self.x_position.range(-10000.0, 10000.0)
        self.y_position = FormControl('Y', 0.0, required=True, desc=unit, number=True)
        self.y_position.range(-10000.0, 10000.0)
        self.z_position = FormControl('Z', 0.0, required=True, desc=unit, number=True)
        self.z_position.range(-10000.0, 10000.0)
        self.form_group.addControl(self.x_position)
        self.form_group.addControl(self.y_position)
        self.form_group.addControl(self.z_position)
        self.form_group.groupValidation.connect(self.formValidation)
        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(TransformType.Translate.value)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addWidget(self.form_group)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.valid_sample = False
        self.selected_sample = sample

    @property
    def selected_sample(self):
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        self.valid_sample = True if self.parent.presenter.model.sample else False
        self.form_group.validateGroup()

    def formValidation(self, is_valid):
        if is_valid and self.valid_sample:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        offset = [self.x_position.value, self.y_position.value, self.z_position.value]
        if not is_close(offset, [0.0, 0.0, 0.0]):
            self.parent.presenter.transformSample(offset, self.selected_sample, TransformType.Translate)


class CustomTransformTool(QtWidgets.QWidget):
    def __init__(self, sample, parent):
        super().__init__()

        self.parent = parent

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.matrix =  Matrix44.identity()
        self.show_matrix = QtWidgets.QPlainTextEdit(self.matrixToString())
        self.show_matrix.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.show_matrix.setReadOnly(True)
        self.main_layout.addWidget(self.show_matrix)

        self.invert_checkbox = QtWidgets.QCheckBox('Invert Transformation Matrix')
        self.main_layout.addWidget(self.invert_checkbox)

        button_layout = QtWidgets.QHBoxLayout()
        self.load_matrix = QtWidgets.QPushButton('Load Matrix')
        self.load_matrix.clicked.connect(self.loadMatrix)
        self.execute_button = QtWidgets.QPushButton('Apply Transform')
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.load_matrix)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.selected_sample = sample

    @property
    def selected_sample(self):
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        if self.parent.presenter.model.sample:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def loadMatrix(self):
        matrix = self.parent.presenter.importTransformMatrix()
        if matrix is None:
            return
        self.matrix = matrix
        self.show_matrix.setPlainText(self.matrixToString())

    def matrixToString(self):
        result = []
        for row in self.matrix:
            for col in row:
                result.append('    {:>20.8f}'.format(col))
            result.append('\n')
        return ''.join(result)

    def executeButtonClicked(self):
        matrix = self.matrix.inverse() if self.invert_checkbox.isChecked() else self.matrix
        if not is_close(matrix, Matrix44.identity()):
            self.parent.presenter.transformSample(matrix, self.selected_sample, TransformType.Custom)


class MoveOriginTool(QtWidgets.QWidget):
    @unique
    class MoveOptions(Enum):
        Center = 'Bound Center'
        Minimum = 'Bound Minimum'
        Maximum = 'Bound Maximum'

    @unique
    class IgnoreOptions(Enum):
        No_change = 'None'
        X = 'X'
        Y = 'Y'
        Z = 'Z'
        YZ = 'YZ'
        XY = 'XY'
        XZ = 'XZ'

    def __init__(self, sample, parent):
        super().__init__()

        self.parent = parent
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel('Move To:')
        self.move_combobox = QtWidgets.QComboBox()
        self.move_combobox.setView(QtWidgets.QListView())
        self.move_combobox.addItems([option.value for option in MoveOriginTool.MoveOptions])
        self.move_combobox.currentTextChanged.connect(self.move_options)
        self.main_layout.addWidget(label)
        self.main_layout.addWidget(self.move_combobox)
        self.main_layout.addSpacing(5)

        label = QtWidgets.QLabel('Ignore Axis:')
        self.ignore_combobox = QtWidgets.QComboBox()
        self.ignore_combobox.setView(QtWidgets.QListView())
        self.ignore_combobox.addItems([option.value for option in MoveOriginTool.IgnoreOptions])
        self.ignore_combobox.currentTextChanged.connect(self.ignore_options)
        self.main_layout.addWidget(label)
        self.main_layout.addWidget(self.ignore_combobox)
        self.main_layout.addSpacing(5)

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton('Move Origin')
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.selected_sample = sample
        self.ignore_options(self.ignore_combobox.currentText())

    @property
    def selected_sample(self):
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value

        sample = self.parent.presenter.model.sample
        if not sample:
            self.bounding_box = None
            self.execute_button.setDisabled(True)
            return

        if value == 'All':
            self.bounding_box = BoundingBox.merge([s.bounding_box for s in sample.values()])
        else:
            self.bounding_box = self.parent.presenter.model.sample[value].bounding_box
        self.move_options(self.move_combobox.currentText())
        self.execute_button.setEnabled(True)

    def move_options(self, text):
        if self.bounding_box is None:
            return

        option = MoveOriginTool.MoveOptions(text)
        if option == MoveOriginTool.MoveOptions.Center:
            self.move_to = self.bounding_box.center
        elif option == MoveOriginTool.MoveOptions.Minimum:
            self.move_to = self.bounding_box.min
        else:
            self.move_to = self.bounding_box.max

    def ignore_options(self, text):
        option = MoveOriginTool.IgnoreOptions(text)
        if option == MoveOriginTool.IgnoreOptions.No_change:
            self.ignore = [False, False, False]
        elif option == MoveOriginTool.IgnoreOptions.X:
            self.ignore = [True, False, False]
        elif option == MoveOriginTool.IgnoreOptions.Y:
            self.ignore = [False, True, False]
        elif option == MoveOriginTool.IgnoreOptions.Z:
            self.ignore = [False, False, True]
        elif option == MoveOriginTool.IgnoreOptions.XY:
            self.ignore = [True, True, False]
        elif option == MoveOriginTool.IgnoreOptions.YZ:
            self.ignore = [False, True, True]
        else:
            self.ignore = [True, False, True]

    def executeButtonClicked(self):
        offset = [0.0 if ignore else -value for value, ignore in zip(self.move_to, self.ignore)]
        if not is_close(offset, [0.0, 0.0, 0.0]):
            self.parent.presenter.transformSample(offset, self.selected_sample, TransformType.Translate)


class PlaneAlignmentTool(QtWidgets.QWidget):
    def __init__(self, sample, parent):
        super().__init__()

        self.parent = parent

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.selected_sample = sample
