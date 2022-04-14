from enum import Enum, unique
from PyQt5 import QtCore, QtWidgets
from sscanss.config import path_for
from sscanss.core.geometry import point_selection, Mesh
from sscanss.core.math import is_close, Matrix44, Plane, rotation_btw_vectors, Vector3
from sscanss.core.util import (TransformType, DockFlag, PlaneOptions, create_tool_button, FormControl, FormGroup,
                               Banner, MessageType)


class TransformDialog(QtWidgets.QWidget):
    """Creates a container widget for sample transformation tools

    :param transform_type: transform type
    :type transform_type: TransformType
    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Upper

    def __init__(self, transform_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.type = transform_type

        self.main_layout = QtWidgets.QVBoxLayout()
        self.banner = Banner(MessageType.Information, self)
        self.main_layout.addWidget(self.banner)
        self.banner.hide()
        self.main_layout.addSpacing(10)

        title_label = QtWidgets.QLabel()
        title_label.setWordWrap(True)
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(10)
        self.tool = None

        if self.type == TransformType.Rotate:
            title_label.setText(f'{self.type.value} sample around X, Y, Z axis')
            self.tool = RotateTool(parent)
            self.main_layout.addWidget(self.tool)
            self.title = f'{self.type.value} Sample'
        elif self.type == TransformType.Translate:
            title_label.setText(f'{self.type.value} sample along X, Y, Z axis')
            self.tool = TranslateTool(parent)
            self.main_layout.addWidget(self.tool)
            self.title = f'{self.type.value} Sample'
        elif self.type == TransformType.Custom:
            title_label.setText('Transform sample with arbitrary matrix')
            self.tool = CustomTransformTool(parent)
            self.main_layout.addWidget(self.tool)
            self.title = 'Transform Sample with Matrix'
        elif self.type == TransformType.Origin:
            title_label.setText('Move origin with respect to sample bounds')
            self.tool = MoveOriginTool(parent)
            self.main_layout.addWidget(self.tool)
            self.title = 'Move Origin to Sample'
        else:
            title_label.setText(('Define initial plane by selecting a minimum of 3 points using '
                                 'the pick tool, then select final plane to rotate initial plane to. '
                                 '<b>This method will not work for volumes</b>'))
            self.tool = PlaneAlignmentTool(parent)
            self.main_layout.addWidget(self.tool)
            self.title = 'Rotate Sample by Plane Alignment'

        self.setLayout(self.main_layout)
        self.setMinimumWidth(450)
        self.tool.selected_sample = self.parent_model.sample
        self.parent_model.sample_changed.connect(self.changeSample)

        if self.parent_model.sample and self.parent_model.fiducials.size == 0:
            self.banner.showMessage('It is recommended to add fiducial points before transforming the sample.',
                                    MessageType.Information)

    def closeEvent(self, event):
        self.tool.close()
        event.accept()

    def changeSample(self, ):
        """Changes the selected sample"""
        if self.tool is not None:
            self.tool.selected_sample = self.parent_model.sample


class RotateTool(QtWidgets.QWidget):
    """Creates a UI for applying simple rotation around the 3 principal axes

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
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
        self.form_group.group_validation.connect(self.formValidation)
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

    @property
    def selected_sample(self):
        """Gets and sets the selected sample key

        :returns: selected sample key
        :rtype: str
        """
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        self.valid_sample = True if self._selected_sample is not None else False
        self.form_group.validateGroup()

    def formValidation(self, is_valid):
        if is_valid and self.valid_sample:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        angles = [self.z_rotation.value, self.y_rotation.value, self.x_rotation.value]
        if not is_close(angles, [0.0, 0.0, 0.0]):
            self.parent.presenter.transformSample(angles, TransformType.Rotate)


class TranslateTool(QtWidgets.QWidget):
    """Creates a UI for applying simple translation along the 3 principal axes

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        unit = 'mm'

        self.form_group = FormGroup()
        self.x_position = FormControl('X', 0.0, required=True, desc=unit, number=True)
        self.y_position = FormControl('Y', 0.0, required=True, desc=unit, number=True)
        self.z_position = FormControl('Z', 0.0, required=True, desc=unit, number=True)
        self.form_group.addControl(self.x_position)
        self.form_group.addControl(self.y_position)
        self.form_group.addControl(self.z_position)
        self.form_group.group_validation.connect(self.formValidation)
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

    @property
    def selected_sample(self):
        """Gets and sets the selected sample key

        :returns: selected sample key
        :rtype: str
        """
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        self.valid_sample = True if self._selected_sample else False
        self.form_group.validateGroup()

    def formValidation(self, is_valid):
        if is_valid and self.valid_sample:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        offset = [self.x_position.value, self.y_position.value, self.z_position.value]
        if not is_close(offset, [0.0, 0.0, 0.0]):
            self.parent.presenter.transformSample(offset, TransformType.Translate)


class CustomTransformTool(QtWidgets.QWidget):
    """Creates a UI for applying rigid transformation with arbitrary homogeneous matrix

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.matrix = Matrix44.identity()
        self.table_widget = QtWidgets.QTableWidget(4, 4)
        self.table_widget.setFixedHeight(120)
        self.table_widget.setShowGrid(False)
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.horizontalHeader().setVisible(False)
        self.table_widget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.main_layout.addWidget(self.table_widget)
        self.main_layout.addSpacing(10)
        self.updateTable()

        self.invert_checkbox = QtWidgets.QCheckBox('Invert Transformation Matrix')
        self.main_layout.addWidget(self.invert_checkbox)
        self.main_layout.addSpacing(10)

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

    @property
    def selected_sample(self):
        """Gets and sets the selected sample key

        :returns: selected sample key
        :rtype: str
        """
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        self.execute_button.setEnabled(self._selected_sample is not None)

    def loadMatrix(self):
        """Loads a transformation matrix from file and displays it"""
        matrix = self.parent.presenter.importTransformMatrix()
        if matrix is None:
            return
        self.matrix = matrix
        self.updateTable()

    def updateTable(self):
        """Displays matrix in table widget"""
        for i in range(4):
            for j in range(4):
                item = QtWidgets.QTableWidgetItem(f'{self.matrix[i, j]:.8f}')
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table_widget.setItem(i, j, item)

    def executeButtonClicked(self):
        matrix = self.matrix.inverse() if self.invert_checkbox.isChecked() else self.matrix
        if not is_close(matrix, Matrix44.identity()):
            self.parent.presenter.transformSample(matrix, TransformType.Custom)


class MoveOriginTool(QtWidgets.QWidget):
    """Creates a UI to translate origin of the coordinate system to the sample bounds

    :param parent: main window instance
    :type parent: MainWindow
    """
    @unique
    class MoveOptions(Enum):
        """Options for moving origin"""
        Center = 'Bound Center'
        Minimum = 'Bound Minimum'
        Maximum = 'Bound Maximum'

    @unique
    class IgnoreOptions(Enum):
        """Options to indicate which axis to ignore when moving origin"""
        No_change = 'None'
        X = 'X'
        Y = 'Y'
        Z = 'Z'
        YZ = 'YZ'
        XY = 'XY'
        XZ = 'XZ'

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel('Move To:')
        self.move_combobox = QtWidgets.QComboBox()
        self.move_combobox.setView(QtWidgets.QListView())
        self.move_combobox.addItems([option.value for option in MoveOriginTool.MoveOptions])
        self.move_combobox.currentTextChanged.connect(self.setMoveOptions)
        self.main_layout.addWidget(label)
        self.main_layout.addWidget(self.move_combobox)
        self.main_layout.addSpacing(5)

        label = QtWidgets.QLabel('Ignore Axis:')
        self.ignore_combobox = QtWidgets.QComboBox()
        self.ignore_combobox.setView(QtWidgets.QListView())
        self.ignore_combobox.addItems([option.value for option in MoveOriginTool.IgnoreOptions])
        self.ignore_combobox.currentTextChanged.connect(self.setIgnoreOptions)
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

        self.bounding_box = None
        self.setIgnoreOptions(self.ignore_combobox.currentText())

    @property
    def selected_sample(self):
        """Gets and sets the selected sample key

        :returns: selected sample key
        :rtype: str
        """
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        if self._selected_sample is None:
            self.bounding_box = None
            self.execute_button.setDisabled(True)
            return

        self.bounding_box = self._selected_sample.bounding_box
        self.setMoveOptions(self.move_combobox.currentText())
        self.execute_button.setEnabled(True)

    def setMoveOptions(self, text):
        """Sets the move option of the tool

        :param text: MoveOptions
        :type text: str
        """
        if self.bounding_box is None:
            return

        option = MoveOriginTool.MoveOptions(text)
        if option == MoveOriginTool.MoveOptions.Center:
            self.move_to = self.bounding_box.center
        elif option == MoveOriginTool.MoveOptions.Minimum:
            self.move_to = self.bounding_box.min
        else:
            self.move_to = self.bounding_box.max

    def setIgnoreOptions(self, text):
        """Sets the ignore option of the tool

        :param text: IgnoreOptions
        :type text: str
        """
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
            self.parent.presenter.transformSample(offset, TransformType.Translate)


class PlaneAlignmentTool(QtWidgets.QWidget):
    """Creates a UI to rotate a selected plane on the sample so that it is aligned with an arbitrary
    plane or a plane formed by any 2 axes of the coordinate system. The plane on the sample is specified
    by picking 3 or more points on the surface of the sample.

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.vertices = None
        self.initial_plane = None
        self.final_plane_normal = None

        layout = QtWidgets.QHBoxLayout()
        self.table_widget = QtWidgets.QTableWidget()
        self.table_widget.setShowGrid(False)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_widget.setColumnCount(3)
        self.table_widget.setFixedHeight(150)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setHorizontalHeaderLabels(['X', 'Y', 'Z'])
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table_widget.selectionModel().selectionChanged.connect(self.selection)
        self.table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table_widget)

        button_layout = QtWidgets.QVBoxLayout()
        self.select_button = create_tool_button(icon_path=path_for('select.png'),
                                                checkable=True,
                                                checked=True,
                                                status_tip='Normal scene manipulation with the mouse',
                                                tooltip='Normal Mode',
                                                style_name='ToolButton')
        self.select_button.clicked.connect(lambda: self.togglePicking(False))
        button_layout.addWidget(self.select_button)

        self.pick_button = create_tool_button(icon_path=path_for('point.png'),
                                              checkable=True,
                                              status_tip='Select 3D points that define the plane',
                                              tooltip='Pick Point Mode',
                                              style_name='ToolButton')

        self.pick_button.clicked.connect(lambda: self.togglePicking(True))
        button_layout.addWidget(self.pick_button)

        self.delete_button = create_tool_button(icon_path=path_for('cross.png'),
                                                style_name='ToolButton',
                                                status_tip='Remove selected points from the scene',
                                                tooltip='Delete Points')
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

    def togglePicking(self, value):
        """Toggles between point picking and scene manipulation in the graphics widget

        :param value: indicates if picking is enabled
        :type value: bool
        """
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
        self.form_group.group_validation.connect(self.setCustomPlane)

        layout.addWidget(self.form_group)
        self.custom_plane_widget.setLayout(layout)
        self.custom_plane_widget.setVisible(False)
        self.main_layout.addWidget(self.custom_plane_widget)

    def setCustomPlane(self, is_valid):
        """Sets custom destination/final plane normal. Error is shown if normal is invalid

        :param is_valid: indicate if the custom plane normal is valid
        :type is_valid: bool
        """
        if is_valid:
            normal = Vector3([self.x_axis.value, self.y_axis.value, self.z_axis.value])
            length = normal.length
            if length > 1e-5:
                self.final_plane_normal = normal / length
                self.x_axis.validation_label.setText('')
                return
            else:
                self.x_axis.validation_label.setText('Bad Normal')

        self.final_plane_normal = None

    def setPlane(self, selected_text):
        """Sets destination/final plane normal or shows inputs for custom normal

        :param selected_text: PlaneOptions
        :type selected_text: str
        """
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
        """Gets and sets the selected sample key

        :returns: selected sample key
        :rtype: str
        """
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, value):
        self._selected_sample = value
        if not isinstance(self._selected_sample, Mesh):
            self.vertices = None
            self.execute_button.setDisabled(True)
            self.clearPicks()
            return

        mesh = self._selected_sample
        self.vertices = mesh.vertices[mesh.indices].reshape(-1, 9)
        self.plane_size = mesh.bounding_box.radius
        self.sample_center = mesh.bounding_box.center
        self.execute_button.setEnabled(True)

    def selection(self):
        """Highlights selected picks in the graphics widget"""
        picks = self.parent.gl_widget.picks
        sm = self.table_widget.selectionModel()
        for i in range(self.table_widget.rowCount()):
            picks[i][1] = sm.isRowSelected(i, QtCore.QModelIndex())
        self.parent.gl_widget.update()

    def addPicks(self, start, end):
        """Computes pick coordinates and adds pick to table and graphics widgets. Pick coordinates are
        computed as the first intersection between start and end coordinates

        :param start: start point of pick line segment
        :type start: Vector3
        :param end: end point of pick line segment
        :type end: Vector3
        """
        points = point_selection(start, end, self.vertices)
        if points.size == 0:
            return

        point = points[0]

        last_index = self.table_widget.rowCount()
        self.table_widget.insertRow(last_index)
        for i in range(3):
            item = QtWidgets.QTableWidgetItem(f'{point[i]:.3f}')
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table_widget.setItem(last_index, i, item)
        self.parent.gl_widget.picks.append([list(point), False])
        self.updateInitialPlane()

    def removePicks(self):
        """Removes picks selected in the table widget and updates the initial plane"""
        model_index = [m.row() for m in self.table_widget.selectionModel().selectedRows()]
        model_index.sort(reverse=True)
        self.table_widget.selectionModel().reset()
        for index in model_index:
            del self.parent.gl_widget.picks[index]
            self.table_widget.removeRow(index)

        self.updateInitialPlane()

    def updateInitialPlane(self):
        """Computes the initial plane when 3 or more points have been picked"""
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
            self.parent.presenter.transformSample(matrix, TransformType.Custom)
            self.clearPicks()

    def clearPicks(self):
        """Clears picks from tool and graphics widget"""
        self.table_widget.clear()
        self.initial_plane = None
        self.parent.gl_widget.picks.clear()
        self.parent.scenes.removePlane()
