from enum import Enum, unique
from PyQt5 import QtWidgets
from sscanss.config import path_for
from sscanss.core.geometry import BoundingBox, segment_triangle_intersection
from sscanss.core.math import is_close, Matrix44, Plane, rotation_btw_vectors, Vector3
from sscanss.core.util import TransformType, DockFlag, PlaneOptions
from sscanss.ui.widgets import FormControl, FormGroup, create_tool_button, Banner


class TransformDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper

    def __init__(self, transform_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.type = transform_type

        self.main_layout = QtWidgets.QVBoxLayout()
        self.banner = Banner(Banner.Type.Info, self)
        self.main_layout.addWidget(self.banner)
        self.banner.hide()
        self.main_layout.addSpacing(10)

        title_label = QtWidgets.QLabel()
        title_label.setWordWrap(True)
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
        elif self.type == TransformType.Origin:
            title_label.setText('Move origin with respect to sample bounds')
            self.tool = MoveOriginTool(self.combobox.currentText(), parent)
            self.main_layout.addWidget(self.tool)
            self.title = 'Move Origin to Sample'
        else:
            title_label.setText(('Define initial plane by selecting a minimum of 3 points using '
                                 'the pick tool, then select final plane to rotate initial plane to.'))
            self.tool = PlaneAlignmentTool(self.combobox.currentText(), parent)
            self.main_layout.addWidget(self.tool)
            self.title = 'Rotate Sample by Plane Alignment'

        self.setLayout(self.main_layout)
        self.setMinimumWidth(350)

        self.combobox.activated[str].connect(self.changeSample)
        self.parent_model.sample_changed.connect(self.updateSampleList)

        if self.parent_model.sample and self.parent_model.fiducials.size == 0:
            self.banner.showMessage('It is recommended to add fiducial points before transforming the sample.',
                                    Banner.Type.Info)

    def closeEvent(self, event):
        self.tool.close()
        event.accept()

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
        self.vertices = None
        self.initial_plane = None
        self.final_plane_normal = None

        layout = QtWidgets.QHBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_widget.setFixedHeight(150)
        self.list_widget.setSpacing(2)
        self.list_widget.itemSelectionChanged.connect(self.selection)

        layout.addWidget(self.list_widget)

        button_layout = QtWidgets.QVBoxLayout()
        self.select_button = create_tool_button(icon_path=path_for('select.png'), checkable=True, checked=True,
                                                style_name='ToolButton')
        self.select_button.clicked.connect(lambda: self.togglePicking(False))
        button_layout.addWidget(self.select_button)

        self.pick_button = create_tool_button(icon_path=path_for('point.png'), checkable=True,
                                              style_name='ToolButton')

        self.pick_button.clicked.connect(lambda: self.togglePicking(True))
        button_layout.addWidget(self.pick_button)

        self.delete_button = create_tool_button(icon_path=path_for('cross.png'), style_name='ToolButton')
        self.delete_button.clicked.connect(self.removePicks)
        button_layout.addWidget(self.delete_button)

        layout.addSpacing(10)
        layout.addLayout(button_layout)
        self.main_layout.addLayout(layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(QtWidgets.QLabel('Select Final Plane:'))
        self.plane_combobox = QtWidgets.QComboBox()
        self.plane_combobox.setView(QtWidgets.QListView())
        self.plane_combobox.addItems([p.value for p in PlaneOptions])
        self.plane_combobox.currentTextChanged.connect(self.setPlane)
        self.main_layout.addWidget(self.plane_combobox)
        self.createCustomPlaneBox()
        self.setPlane(self.plane_combobox.currentText())

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton('Align Planes')
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)
        self.parent.gl_widget.pick_added.connect(self.addPicks)

        self.selected_sample = sample

    def togglePicking(self, value):
        self.parent.gl_widget.picking = value
        if value:
            self.select_button.setChecked(False)
            self.pick_button.setChecked(True)
        else:
            self.select_button.setChecked(True)
            self.pick_button.setChecked(False)

    def createCustomPlaneBox(self):
        self.custom_plane_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout()

        self.form_group = FormGroup(FormGroup.Layout.Horizontal)
        self.x_axis = FormControl('X', 1.0, required=True, number=True)
        self.x_axis.range(-1.0, 1.0)
        self.y_axis = FormControl('Y', 0.0, required=True, number=True)
        self.y_axis.range(-1.0, 1.0)
        self.z_axis = FormControl('Z', 0.0, required=True, number=True)
        self.z_axis.range(-1.0, 1.0)
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.groupValidation.connect(self.setCustomPlane)

        layout.addWidget(self.form_group)
        self.custom_plane_widget.setLayout(layout)
        self.custom_plane_widget.setVisible(False)
        self.main_layout.addWidget(self.custom_plane_widget)

    def setCustomPlane(self, is_valid):
        if is_valid:
            normal = Vector3([self.x_axis.value, self.y_axis.value, self.z_axis.value])
            length = normal.length
            if length > 1e-5:
                self.final_plane_normal = normal / length
                return
            else:
                self.x_axis.validation_label.setText('Bad Normal')

        self.final_plane_normal = None

    def setPlane(self, selected_text):
        if selected_text == PlaneOptions.Custom.value:
            self.custom_plane_widget.setVisible(True)
            return
        elif selected_text == PlaneOptions.XY.value:
            self.final_plane_normal = Vector3([0., 0., 1.])
        elif selected_text == PlaneOptions.XZ.value:
            self.final_plane_normal = Vector3([0., 1., 0.])
        else:
            self.final_plane_normal = Vector3([1., 0., 0.])

        self.custom_plane_widget.setVisible(False)

    @property
    def selected_sample(self):
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value

        sample = self.parent.presenter.model.sample
        if not sample:
            self.vertices = None
            self.execute_button.setDisabled(True)
            self.clearPicks()
            return

        if value == 'All':
            mesh = None
            for s in sample.values():
                if mesh is None:
                    mesh = s.copy()
                else:
                    mesh.append(s)
        else:
            mesh = self.parent.presenter.model.sample[value]

        self.vertices = mesh.vertices[mesh.indices].reshape(-1, 9)
        self.plane_size = mesh.bounding_box.radius
        self.sample_center = mesh.bounding_box.center
        self.execute_button.setEnabled(True)

    def selection(self):
        picks = self.parent.gl_widget.picks
        for i in range(self.list_widget.count()):
            picks[i][1] = self.list_widget.item(i).isSelected()
        self.parent.gl_widget.update()

    def addPicks(self, start, end):
        direction = end - start
        length = direction.length
        if length < 1e-5 or self.vertices is None:
            return
        direction /= length

        distances = segment_triangle_intersection(start, direction, length, self.vertices)
        if not distances:
            return

        point = start + direction * distances[0]
        self.list_widget.addItem('X: {:12.3f} Y: {:12.3f} Z: {:12.3f}'.format(*point))
        self.parent.gl_widget.picks.append([point, False])
        self.updateInitialPlane()

    def removePicks(self):
        model_index = [index.row() for index in self.list_widget.selectionModel().selectedIndexes()]
        model_index.sort(reverse=True)
        self.list_widget.selectionModel().reset()
        for index in model_index:
            del self.parent.gl_widget.picks[index]
            self.list_widget.takeItem(index)

        self.updateInitialPlane()

    def updateInitialPlane(self):
        if len(self.parent.gl_widget.picks) > 2:
            points = list(zip(*self.parent.gl_widget.picks))[0]
            self.initial_plane = Plane.fromBestFit(points)
            d = self.initial_plane.normal.dot(self.initial_plane.point - self.sample_center)
            self.initial_plane.point = self.sample_center + self.initial_plane.normal * d
            self.parent.scenes.drawPlane(self.initial_plane, 2 * self.plane_size, 2 * self.plane_size)
        else:
            self.initial_plane = None
            self.parent.scenes.removePlane()

    def closeEvent(self, event):
        self.parent.gl_widget.picking = False
        self.clearPicks()
        event.accept()

    def executeButtonClicked(self):
        if self.final_plane_normal is None or self.initial_plane is None:
            return

        matrix = Matrix44.identity()
        matrix[0:3, 0:3] = rotation_btw_vectors(self.initial_plane.normal, self.final_plane_normal)
        if not is_close(matrix, Matrix44.identity()):
            self.parent.presenter.transformSample(matrix, self.selected_sample, TransformType.Custom)
            self.clearPicks()

    def clearPicks(self):
        self.list_widget.clear()
        self.initial_plane = None
        self.parent.gl_widget.picks.clear()
        self.parent.scenes.removePlane()
