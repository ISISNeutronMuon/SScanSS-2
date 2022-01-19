import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.config import path_for, settings
from sscanss.core.math import Plane, Matrix33, Vector3, clamp, map_range, trunc, VECTOR_EPS, POS_EPS
from sscanss.core.geometry import mesh_plane_intersection
from sscanss.core.util import (Primitives, DockFlag, StrainComponents, PointType, PlaneOptions, Attributes,
                               create_tool_button, create_scroll_area, create_icon, FormTitle, CompareValidator,
                               FormGroup, FormControl, FilePicker)
from sscanss.app.widgets import GraphicsView, GraphicsScene, GraphicsPointItem, Grid
from .managers import PointManager


class InsertPrimitiveDialog(QtWidgets.QWidget):
    """Creates a UI for typing in measurement/fiducial points

    :param primitive: primitive type
    :type primitive: Primitives
    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Upper

    def __init__(self, primitive, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = self.parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.primitive = primitive

        self.main_layout = QtWidgets.QVBoxLayout()

        self.textboxes = {}
        name = self.parent_model.uniqueKey(self.primitive.value)
        self.mesh_args = {'name': name}
        if self.primitive == Primitives.Tube:
            self.mesh_args.update({'outer_radius': 100.000, 'inner_radius': 50.000, 'height': 200.000})
        elif self.primitive == Primitives.Sphere:
            self.mesh_args.update({'radius': 100.000})
        elif self.primitive == Primitives.Cylinder:
            self.mesh_args.update({'radius': 100.000, 'height': 200.000})
        else:
            self.mesh_args.update({'width': 50.000, 'height': 100.000, 'depth': 200.000})

        self.createPrimitiveSwitcher()
        self.createFormInputs()

        button_layout = QtWidgets.QHBoxLayout()
        self.create_primitive_button = QtWidgets.QPushButton('Create')
        self.create_primitive_button.clicked.connect(self.createPrimiviteButtonClicked)
        button_layout.addWidget(self.create_primitive_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = f'Insert {self.primitive.value}'
        self.setMinimumWidth(450)
        self.textboxes['name'].setFocus()

    def createPrimitiveSwitcher(self):
        """Creates a button to switch primitive type"""
        switcher_layout = QtWidgets.QHBoxLayout()
        switcher = create_tool_button(style_name='MenuButton', status_tip='Open dialog for a different primitive')
        switcher.setArrowType(QtCore.Qt.DownArrow)
        switcher.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        switcher.setMenu(self.parent.primitives_menu)
        switcher_layout.addStretch(1)
        switcher_layout.addWidget(switcher)
        self.main_layout.addLayout(switcher_layout)

    def createFormInputs(self):
        """Creates inputs for primitive arguments"""
        self.form_group = FormGroup()
        for key, value in self.mesh_args.items():
            pretty_label = key.replace('_', ' ').title()

            if key == 'name':
                control = FormControl(pretty_label, value, required=True)
                control.form_lineedit.textChanged.connect(self.nameCheck)
            else:
                control = FormControl(pretty_label, value, desc='mm', required=True, number=True)
                control.range(0, None, min_exclusive=True)

            self.textboxes[key] = control
            self.form_group.addControl(control)

        if self.primitive == Primitives.Tube:
            outer_radius = self.textboxes['outer_radius']
            inner_radius = self.textboxes['inner_radius']

            outer_radius.compareWith(inner_radius, CompareValidator.Operator.Greater)
            inner_radius.compareWith(outer_radius, CompareValidator.Operator.Less)

        self.main_layout.addWidget(self.form_group)
        self.form_group.group_validation.connect(self.formValidation)

    def nameCheck(self, value):
        """Checks the name given to the primitive is not reserved"""
        if self.parent_model.all_sample_key == value:
            self.textboxes['name'].isInvalid(f'"{self.parent_model.all_sample_key}" is a reserved name')

    def formValidation(self, is_valid):
        if is_valid:
            self.create_primitive_button.setEnabled(True)
        else:
            self.create_primitive_button.setDisabled(True)

    def createPrimiviteButtonClicked(self):
        for key, textbox in self.textboxes.items():
            value = textbox.value
            self.mesh_args[key] = value

        self.parent.presenter.addPrimitive(self.primitive, self.mesh_args)
        new_name = self.parent_model.uniqueKey(self.primitive.value)
        self.textboxes['name'].value = new_name


class InsertPointDialog(QtWidgets.QWidget):
    """Creates a UI for typing in measurement/fiducial points

    :param point_type: point type
    :type point_type: PointType
    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Upper

    def __init__(self, point_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.point_type = point_type
        self.title = f'Add {point_type.value} Point'
        self.main_layout = QtWidgets.QVBoxLayout()
        unit = 'mm'
        self.form_group = FormGroup()
        self.x_axis = FormControl('X', 0.0, required=True, desc=unit, number=True)
        self.y_axis = FormControl('Y', 0.0, required=True, desc=unit, number=True)
        self.z_axis = FormControl('Z', 0.0, required=True, desc=unit, number=True)
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.group_validation.connect(self.formValidation)
        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(self.title)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addWidget(self.form_group)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.setMinimumWidth(450)

    def formValidation(self, is_valid):
        if is_valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        point = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        self.parent.presenter.addPoints([(point, True)], self.point_type)


class InsertVectorDialog(QtWidgets.QWidget):
    """Creates a UI for adding measurement vectors using a variety of methods

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Upper

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.title = 'Add Measurement Vectors'
        self.main_layout = QtWidgets.QVBoxLayout()
        spacing = 10
        self.main_layout.addSpacing(spacing)
        self.main_layout.addWidget(QtWidgets.QLabel('Measurement Point:'))
        self.points_combobox = QtWidgets.QComboBox()
        self.points_combobox.setView(QtWidgets.QListView())
        self.main_layout.addWidget(self.points_combobox)
        self.updatePointList()
        self.main_layout.addSpacing(spacing)

        layout = QtWidgets.QHBoxLayout()
        alignment_layout = QtWidgets.QVBoxLayout()
        alignment_layout.addWidget(QtWidgets.QLabel('Alignment:'))
        self.alignment_combobox = QtWidgets.QComboBox()
        self.alignment_combobox.setView(QtWidgets.QListView())
        self.alignment_combobox.setInsertPolicy(QtWidgets.QComboBox.InsertAtCurrent)
        self.updateAlignment()
        self.alignment_combobox.activated.connect(self.addNewAlignment)
        self.alignment_combobox.currentIndexChanged.connect(self.changeRenderedAlignment)
        alignment_layout.addWidget(self.alignment_combobox)
        alignment_layout.addSpacing(spacing)
        layout.addLayout(alignment_layout)

        self.detector_combobox = QtWidgets.QComboBox()
        self.detector_combobox.setView(QtWidgets.QListView())
        self.detector_combobox.addItems(list(self.parent_model.instrument.detectors.keys()))
        if len(self.parent_model.instrument.detectors) > 1:
            detector_layout = QtWidgets.QVBoxLayout()
            detector_layout.addWidget(QtWidgets.QLabel('Detector:'))
            detector_layout.addWidget(self.detector_combobox)
            size = self.detector_combobox.iconSize()
            self.detector_combobox.setItemIcon(0, create_icon(settings.value(settings.Key.Vector_1_Colour), size))
            self.detector_combobox.setItemIcon(1, create_icon(settings.value(settings.Key.Vector_2_Colour), size))
            detector_layout.addSpacing(spacing)
            layout.addSpacing(spacing)
            layout.addLayout(detector_layout)

        self.main_layout.addLayout(layout)

        self.main_layout.addWidget(QtWidgets.QLabel('Strain Component:'))
        self.component_combobox = QtWidgets.QComboBox()
        self.component_combobox.setView(QtWidgets.QListView())
        strain_components = [s.value for s in StrainComponents]
        self.component_combobox.addItems(strain_components)
        self.component_combobox.currentTextChanged.connect(self.toggleKeyInBox)
        self.main_layout.addWidget(self.component_combobox)
        self.main_layout.addSpacing(spacing)

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(self.title)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.createKeyInBox()

        self.reverse_checkbox = QtWidgets.QCheckBox('Reverse Direction of Vector')
        self.main_layout.addWidget(self.reverse_checkbox)
        self.main_layout.addSpacing(spacing)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)
        self.parent_model.measurement_points_changed.connect(self.updatePointList)
        self.parent_model.measurement_vectors_changed.connect(self.updateAlignment)
        self.parent.scenes.rendered_alignment_changed.connect(self.alignment_combobox.setCurrentIndex)
        self.setMinimumWidth(450)

    def updatePointList(self):
        """Updates the list of measurement points"""
        self.points_combobox.clear()
        point_list = ['All Points']
        point_list.extend([f'{i + 1}' for i in range(self.parent_model.measurement_points.size)])
        self.points_combobox.addItems(point_list)

    def updateAlignment(self):
        """Updates the list of alignments after selection change or vector update"""
        align_count = self.parent_model.measurement_vectors.shape[2]
        if align_count != self.alignment_combobox.count() - 1:
            self.alignment_combobox.clear()
            alignment_list = [f'{i + 1}' for i in range(align_count)]
            alignment_list.append('Add New...')
            self.alignment_combobox.addItems(alignment_list)

        self.alignment_combobox.setCurrentIndex(self.parent.scenes.rendered_alignment)

    def addNewAlignment(self, index):
        """Adds a new alignment to the alignment list"""
        if index == self.alignment_combobox.count() - 1:
            self.alignment_combobox.insertItem(index, f'{index + 1}')
            self.alignment_combobox.setCurrentIndex(index)

    def changeRenderedAlignment(self, index):
        """Changes the alignment that is rendered in the scene

        :param index: index of alignment to render
        :type index: int
        """
        align_count = self.parent_model.measurement_vectors.shape[2]
        if 0 <= index < align_count:
            self.parent.scenes.changeRenderedAlignment(index)
        elif index >= align_count:
            self.parent.scenes.changeVisibility(Attributes.Vectors, False)

    def toggleKeyInBox(self, selected_text):
        """Shows/Hides the inputs for key-in vector when appropriate strain component is selected

        :param selected_text: strain component
        :type selected_text: str
        """
        strain_component = StrainComponents(selected_text)
        if strain_component == StrainComponents.custom:
            self.key_in_box.setVisible(True)
            self.form_group.validateGroup()
        else:
            self.key_in_box.setVisible(False)
            self.execute_button.setEnabled(True)

    def createKeyInBox(self):
        """Creates the inputs for key-in vector """
        self.key_in_box = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout()

        self.form_group = FormGroup(FormGroup.Layout.Horizontal)
        self.x_axis = FormControl('X', 1.0, required=True, number=True, decimals=7)
        self.x_axis.range(-1.0, 1.0)
        self.y_axis = FormControl('Y', 0.0, required=True, number=True, decimals=7)
        self.y_axis.range(-1.0, 1.0)
        self.z_axis = FormControl('Z', 0.0, required=True, number=True, decimals=7)
        self.z_axis.range(-1.0, 1.0)
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.group_validation.connect(self.formValidation)

        layout.addWidget(self.form_group)
        self.key_in_box.setLayout(layout)
        self.main_layout.addWidget(self.key_in_box)
        self.toggleKeyInBox(self.component_combobox.currentText())

    def formValidation(self, is_valid):
        self.execute_button.setDisabled(True)
        if is_valid:
            if np.linalg.norm([self.x_axis.value, self.y_axis.value, self.z_axis.value]) > VECTOR_EPS:
                self.x_axis.validation_label.setText('')
                self.execute_button.setEnabled(True)
            else:
                self.x_axis.validation_label.setText('Bad Normal')

    def executeButtonClicked(self):
        points = self.points_combobox.currentIndex() - 1

        selected_text = self.component_combobox.currentText()
        strain_component = StrainComponents(selected_text)

        alignment = self.alignment_combobox.currentIndex()
        detector = self.detector_combobox.currentIndex()
        check_state = self.reverse_checkbox.checkState()
        reverse = (check_state == QtCore.Qt.Checked)

        if strain_component == StrainComponents.custom:
            vector = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        else:
            vector = None

        self.parent.presenter.addVectors(points, strain_component, alignment, detector, key_in=vector, reverse=reverse)
        # New vectors are drawn by the scene manager after function ends
        self.parent.scenes._rendered_alignment = alignment

    def closeEvent(self, event):
        self.parent.scenes.changeRenderedAlignment(0)
        event.accept()


class PickPointDialog(QtWidgets.QWidget):
    """Creates a UI for selecting measurement points on a cross-section of the sample

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Full

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.title = 'Add Measurement Points Graphically'
        self.setMinimumWidth(500)

        self.plane_offset_range = (-1., 1.)
        self.slider_range = (-10000000, 10000000)

        self.sample_scale = 20
        self.path_pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 0)
        self.point_pen = QtGui.QPen(QtGui.QColor(200, 0, 0), 0)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        button_layout = QtWidgets.QHBoxLayout()
        self.help_button = create_tool_button(tooltip='Help',
                                              style_name='ToolButton',
                                              status_tip='Display shortcuts for the cross-section view',
                                              icon_path=path_for('question.png'))
        self.help_button.clicked.connect(self.showHelp)

        self.reset_button = create_tool_button(tooltip='Reset View',
                                               style_name='ToolButton',
                                               status_tip='Reset camera transformation of the cross-section view',
                                               icon_path=path_for('refresh.png'))
        self.execute_button = QtWidgets.QPushButton('Add Points')
        self.execute_button.clicked.connect(self.addPoints)
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.execute_button)
        self.main_layout.addLayout(button_layout)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.main_layout.addWidget(self.splitter)
        self.createGraphicsView()
        self.reset_button.clicked.connect(self.view.reset)
        self.createControlPanel()

        self.prepareMesh()
        self.parent_model.sample_changed.connect(self.prepareMesh)
        self.parent_model.measurement_points_changed.connect(self.updateCrossSection)
        self.initializing = True

    def showEvent(self, event):
        if self.initializing:
            self.view.fitInView(self.view.anchor, QtCore.Qt.KeepAspectRatio)
            self.initializing = False

        super().showEvent(event)

    def closeEvent(self, event):
        self.parent.scenes.removePlane()
        event.accept()

    def prepareMesh(self):
        """Merges the sample meshes and initialize UI. UI is disabled if no mesh is present"""
        self.mesh = None
        samples = self.parent_model.sample
        for _, sample in samples.items():
            if self.mesh is None:
                self.mesh = sample.copy()
            else:
                self.mesh.append(sample)

        self.scene.clear()
        self.tabs.setEnabled(self.mesh is not None)
        if self.mesh is not None:
            self.setPlane(self.plane_combobox.currentText())
        else:
            self.parent.scenes.removePlane()
        self.view.reset()

    def updateStatusBar(self, point):
        """Updates the status bar with the position of the mouse cursor in 3D world coordinates

        :param point: mouse cursor position in widget coordinates
        :type point: QtCore.QPoint
        """
        if self.view.rect().contains(point):
            transform = self.view.scene_transform.inverted()[0]
            scene_pt = transform.map(self.view.mapToScene(point)) / self.sample_scale
            world_pt = [scene_pt.x(), scene_pt.y(), -self.old_distance] @ self.matrix.transpose()
            cursor_text = f'X:   {world_pt[0]:.3f}        Y:   {world_pt[1]:.3f}        Z:   {world_pt[2]:.3f}'
            self.parent.cursor_label.setText(cursor_text)
        else:
            self.parent.cursor_label.clear()

    def createGraphicsView(self):
        """Creates the graphics view and scene"""
        self.scene = GraphicsScene(self.sample_scale, self)
        self.view = GraphicsView(self.scene)
        self.view.mouse_moved.connect(self.updateStatusBar)
        self.view.setMinimumHeight(350)
        self.splitter.addWidget(self.view)

    def createControlPanel(self):
        """Creates the control panel widgets"""
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumHeight(250)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.South)
        self.splitter.addWidget(self.tabs)

        self.createPlaneTab()
        self.createSelectionToolsTab()
        self.createGridOptionsTab()
        point_manager = PointManager(PointType.Measurement, self.parent)
        self.tabs.addTab(create_scroll_area(point_manager), 'Point Manager')

    def createPlaneTab(self):
        """Creates the plane widget"""
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Specify Plane:'))
        self.plane_combobox = QtWidgets.QComboBox()
        self.plane_combobox.setView(QtWidgets.QListView())
        self.plane_combobox.addItems([p.value for p in PlaneOptions])
        self.plane_combobox.currentTextChanged.connect(self.setPlane)
        self.createCustomPlaneBox()
        layout.addWidget(self.plane_combobox)
        layout.addWidget(self.custom_plane_widget)
        layout.addSpacing(20)

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel('Plane Distance from Origin (mm):'))
        self.plane_lineedit = QtWidgets.QLineEdit()
        validator = QtGui.QDoubleValidator(self.plane_lineedit)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator.setDecimals(3)
        self.plane_lineedit.setValidator(validator)
        self.plane_lineedit.textEdited.connect(self.updateSlider)
        self.plane_lineedit.editingFinished.connect(self.movePlane)
        slider_layout.addStretch(1)
        slider_layout.addWidget(self.plane_lineedit)
        layout.addLayout(slider_layout)
        self.plane_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.plane_slider.setMinimum(self.slider_range[0])
        self.plane_slider.setMaximum(self.slider_range[1])
        self.plane_slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.plane_slider.setSingleStep(1)
        self.plane_slider.sliderMoved.connect(self.updateLineEdit)
        self.plane_slider.sliderReleased.connect(self.movePlane)
        layout.addWidget(self.plane_slider)
        layout.addStretch(1)

        plane_tab = QtWidgets.QWidget()
        plane_tab.setLayout(layout)
        self.tabs.addTab(create_scroll_area(plane_tab), 'Define Plane')

    def createSelectionToolsTab(self):
        """Creates the point selection widget"""
        layout = QtWidgets.QVBoxLayout()
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel('Select Geometry of Points: '))
        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.buttonClicked[int].connect(self.changeSceneMode)

        self.object_selector = create_tool_button(checkable=True,
                                                  checked=True,
                                                  tooltip='Select Points',
                                                  status_tip='Select movable points from the cross-section view',
                                                  style_name='MidToolButton',
                                                  icon_path=path_for('select.png'))
        self.point_selector = create_tool_button(checkable=True,
                                                 tooltip='Draw a Point',
                                                 status_tip='Draw a single point at the selected position',
                                                 style_name='MidToolButton',
                                                 icon_path=path_for('point.png'))
        self.line_selector = create_tool_button(checkable=True,
                                                tooltip='Draw Points on Line',
                                                status_tip='Draw equally spaced points on the selected line',
                                                style_name='MidToolButton',
                                                icon_path=path_for('line_tool.png'))
        self.area_selector = create_tool_button(checkable=True,
                                                tooltip='Draw Points on Area',
                                                status_tip='Draw a grid of points on the selected area',
                                                style_name='MidToolButton',
                                                icon_path=path_for('area_tool.png'))

        self.button_group.addButton(self.object_selector, GraphicsScene.Mode.Select.value)
        self.button_group.addButton(self.point_selector, GraphicsScene.Mode.Draw_point.value)
        self.button_group.addButton(self.line_selector, GraphicsScene.Mode.Draw_line.value)
        self.button_group.addButton(self.area_selector, GraphicsScene.Mode.Draw_area.value)
        selector_layout.addWidget(self.object_selector)
        selector_layout.addWidget(self.point_selector)
        selector_layout.addWidget(self.line_selector)
        selector_layout.addWidget(self.area_selector)
        selector_layout.addStretch(1)

        self.createLineToolWidget()
        self.createAreaToolWidget()

        layout.addLayout(selector_layout)
        layout.addWidget(self.line_tool_widget)
        layout.addWidget(self.area_tool_widget)
        layout.addStretch(1)

        select_tab = QtWidgets.QWidget()
        select_tab.setLayout(layout)
        self.tabs.addTab(create_scroll_area(select_tab), 'Selection Tools')

    def createGridOptionsTab(self):
        """Creates the grid option widget"""
        layout = QtWidgets.QVBoxLayout()
        self.show_grid_checkbox = QtWidgets.QCheckBox('Show Grid')
        self.show_grid_checkbox.stateChanged.connect(self.showGrid)
        self.snap_to_grid_checkbox = QtWidgets.QCheckBox('Snap Selection to Grid')
        self.snap_to_grid_checkbox.stateChanged.connect(self.snapToGrid)
        self.snap_to_grid_checkbox.setEnabled(self.view.show_grid)
        layout.addWidget(self.show_grid_checkbox)
        layout.addWidget(self.snap_to_grid_checkbox)
        self.createGridWidget()
        layout.addWidget(self.grid_widget)
        layout.addStretch(1)

        grid_tab = QtWidgets.QWidget()
        grid_tab.setLayout(layout)
        self.tabs.addTab(create_scroll_area(grid_tab), 'Grid Options')

    def createCustomPlaneBox(self):
        """Creates inputs for custom plane axis"""
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

    def createLineToolWidget(self):
        """Creates the input for number of points when using the line tool"""
        self.line_tool_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 20, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Number of Points: '))
        self.line_point_count_spinbox = QtWidgets.QSpinBox()
        self.line_point_count_spinbox.setValue(self.scene.line_tool_size)
        self.line_point_count_spinbox.setRange(2, 100)
        self.line_point_count_spinbox.valueChanged.connect(self.scene.setLineToolSize)

        layout.addWidget(self.line_point_count_spinbox)
        self.line_tool_widget.setVisible(False)
        self.line_tool_widget.setLayout(layout)

    def createAreaToolWidget(self):
        """Creates the inputs for number of points when using the area tool"""
        self.area_tool_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 20, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Number of Points: '))
        self.area_x_spinbox = QtWidgets.QSpinBox()
        self.area_x_spinbox.setValue(self.scene.area_tool_size[0])
        self.area_x_spinbox.setRange(2, 100)
        self.area_y_spinbox = QtWidgets.QSpinBox()
        self.area_y_spinbox.setValue(self.scene.area_tool_size[1])
        self.area_y_spinbox.setRange(2, 100)

        stretch_factor = 3
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('X: '))
        self.area_x_spinbox.valueChanged.connect(
            lambda: self.scene.setAreaToolSize(self.area_x_spinbox.value(), self.area_y_spinbox.value()))
        layout.addWidget(self.area_x_spinbox, stretch_factor)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('Y: '))
        self.area_y_spinbox.valueChanged.connect(
            lambda: self.scene.setAreaToolSize(self.area_x_spinbox.value(), self.area_y_spinbox.value()))
        layout.addWidget(self.area_y_spinbox, stretch_factor)
        self.area_tool_widget.setVisible(False)
        self.area_tool_widget.setLayout(layout)

    def createGridWidget(self):
        """Creates the inputs for grid size"""
        self.grid_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 20, 0, 0)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Grid Type: '))
        grid_combobox = QtWidgets.QComboBox()
        grid_combobox.setView(QtWidgets.QListView())
        grid_combobox.addItems([g.value for g in Grid.Type])
        grid_combobox.currentTextChanged.connect(lambda value: self.setGridType(Grid.Type(value)))
        layout.addWidget(grid_combobox)
        main_layout.addLayout(layout)
        main_layout.addSpacing(20)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Grid Size: '))
        self.grid_x_label = QtWidgets.QLabel('')
        self.grid_x_spinbox = QtWidgets.QDoubleSpinBox()
        self.grid_x_spinbox.setDecimals(1)
        self.grid_x_spinbox.setSingleStep(0.1)
        self.grid_x_spinbox.valueChanged.connect(self.changeGridSize)
        self.grid_y_label = QtWidgets.QLabel('')
        self.grid_y_spinbox = QtWidgets.QDoubleSpinBox()
        self.grid_y_spinbox.setDecimals(1)
        self.grid_y_spinbox.setSingleStep(0.1)
        self.grid_y_spinbox.valueChanged.connect(self.changeGridSize)
        stretch_factor = 3
        layout.addStretch(1)
        layout.addWidget(self.grid_x_label)
        layout.addWidget(self.grid_x_spinbox, stretch_factor)
        layout.addStretch(1)
        layout.addWidget(self.grid_y_label)
        layout.addWidget(self.grid_y_spinbox, stretch_factor)
        main_layout.addLayout(layout)
        self.setGridType(self.view.grid.type)
        self.grid_widget.setVisible(False)
        self.grid_widget.setLayout(main_layout)

    def changeGridSize(self):
        """Changes the grid size in the scene"""
        if self.view.grid.type == Grid.Type.Box:
            grid_x = int(self.grid_x_spinbox.value() * self.sample_scale)
            grid_y = int(self.grid_y_spinbox.value() * self.sample_scale)
        else:
            grid_x = int(self.grid_x_spinbox.value() * self.sample_scale)
            grid_y = self.grid_y_spinbox.value()
        self.view.setGridSize((grid_x, grid_y))

    def setGridType(self, grid_type):
        """Sets the grid type

        :param grid_type: type of grid
        :type grid_type: Grid.Type
        """
        self.view.setGridType(grid_type)
        size = self.view.grid.size
        if grid_type == Grid.Type.Box:
            self.grid_x_label.setText('X (mm): ')
            self.grid_y_label.setText('Y (mm): ')
            self.grid_x_spinbox.setValue(size[0])
            self.grid_y_spinbox.setValue(size[1])
            self.grid_x_spinbox.setRange(0.1, 1000)
            self.grid_y_spinbox.setRange(0.1, 1000)
        else:
            self.grid_x_label.setText('Radius (mm): ')
            self.grid_y_label.setText('Angle (degree): ')
            self.grid_x_spinbox.setValue(size[0])
            self.grid_y_spinbox.setValue(size[1])
            self.grid_x_spinbox.setRange(0.1, 1000)
            self.grid_y_spinbox.setRange(0.1, 360)

    def changeSceneMode(self, button_id):
        """Changes the scene's mode based on the active selection tool

        :param button_id: index of active selection tool
        :type button_id: int
        """
        self.scene.mode = GraphicsScene.Mode(button_id)
        self.line_tool_widget.setVisible(self.scene.mode == GraphicsScene.Mode.Draw_line)
        self.area_tool_widget.setVisible(self.scene.mode == GraphicsScene.Mode.Draw_area)

    def showHelp(self):
        """Toggles the help overlay in the scene"""
        self.view.show_help = not self.view.has_foreground
        self.scene.update()

    def showGrid(self, state):
        """Shows/Hides the grid in the scene

        :param state: indicated the state if the checkbox
        :type state: Qt.CheckState
        """
        self.view.show_grid = (state == QtCore.Qt.Checked)
        self.snap_to_grid_checkbox.setEnabled(self.view.show_grid)
        self.grid_widget.setVisible(self.view.show_grid)
        self.scene.update()

    def snapToGrid(self, state):
        """Enables/Disables snap to grid

        :param state: indicated the state if the checkbox
        :type state: Qt.CheckState
        """
        self.view.snap_to_grid = (state == QtCore.Qt.Checked)

    def updateSlider(self, value):
        """Updates the cross-section plane position in the slider when position is changed via
         the line edit

        :param value: value in the line edit
        :type value: str
        """
        if not self.plane_lineedit.hasAcceptableInput():
            return

        new_distance = clamp(float(value), *self.plane_offset_range)
        slider_value = int(map_range(*self.plane_offset_range, *self.slider_range, new_distance))
        self.plane_slider.setValue(slider_value)

        offset = new_distance - self.old_distance
        self.parent.scenes.movePlane(offset * self.plane.normal)
        self.old_distance = new_distance

    def updateLineEdit(self, value):
        """Updates the cross-section plane position in the line edit when position is changed via
         the slider

        :param value: value in the slider
        :type value: int
        """
        new_distance = trunc(map_range(*self.slider_range, *self.plane_offset_range, value), 3)
        self.plane_lineedit.setText(f'{new_distance:.3f}')

        offset = new_distance - self.old_distance
        self.parent.scenes.movePlane(offset * self.plane.normal)
        self.old_distance = new_distance

    def movePlane(self):
        """Updates the position of the plane when the value is changed via the line edit or slider"""
        distance = clamp(float(self.plane_lineedit.text()), *self.plane_offset_range)
        self.plane_lineedit.setText(f'{distance:.3f}')
        point = distance * self.plane.normal
        self.plane = Plane(self.plane.normal, point)
        self.updateCrossSection()

    def setCustomPlane(self, is_valid):
        """Initializes the cross-section plane with a custom normal

        :param is_valid: indicates if custom normal inputs are valid
        :type is_valid: bool
        """
        if is_valid:
            normal = np.array([self.x_axis.value, self.y_axis.value, self.z_axis.value])
            try:
                self.initializePlane(normal, self.mesh.bounding_box.center)
            except ValueError:
                self.x_axis.validation_label.setText('Bad Normal')

    def setPlane(self, selected_text):
        """Sets the cross-section plane orientation

        :param selected_text: plane options
        :type selected_text: str
        """
        if selected_text == PlaneOptions.Custom.value:
            self.custom_plane_widget.setVisible(True)
            self.form_group.validateGroup()
            return
        else:
            self.custom_plane_widget.setVisible(False)

        if selected_text == PlaneOptions.XY.value:
            plane_normal = np.array([0., 0., 1.])
        elif selected_text == PlaneOptions.XZ.value:
            plane_normal = np.array([0., 1., 0.])
        else:
            plane_normal = np.array([1., 0., 0.])

        self.initializePlane(plane_normal, self.mesh.bounding_box.center)

    def initializePlane(self, plane_normal, plane_point):
        """Creates the cross-section plane

        :param plane_normal: plane normal
        :type plane_normal: Union[numpy.ndarray, Vector3]
        :param plane_point: point on the plane
        :type plane_point: Union[numpy.ndarray, Vector3]
        """
        self.plane = Plane(plane_normal, plane_point)
        plane_size = self.mesh.bounding_box.radius

        self.parent.scenes.drawPlane(self.plane, 2 * plane_size, 2 * plane_size)
        distance = self.plane.distanceFromOrigin()
        self.plane_offset_range = (distance - plane_size, distance + plane_size)
        slider_value = int(map_range(*self.plane_offset_range, *self.slider_range, distance))
        self.plane_slider.setValue(slider_value)
        self.plane_lineedit.setText(f'{distance:.3f}')
        self.old_distance = distance
        # inverted the normal so that the y-axis is flipped
        self.matrix = self.__lookAt(-Vector3(self.plane.normal))
        self.view.resetTransform()
        self.updateCrossSection()

    def updateCrossSection(self):
        """Creates the mesh cross-section and displays the cross-section and points in the scene"""
        self.scene.clear()
        segments = mesh_plane_intersection(self.mesh, self.plane)
        if len(segments) == 0:
            return
        segments = np.array(segments)

        item = QtWidgets.QGraphicsPathItem()
        cross_section_path = QtGui.QPainterPath()
        rotated_segments = self.sample_scale * (segments @ self.matrix)
        for i in range(0, rotated_segments.shape[0], 2):
            start = rotated_segments[i, :]
            cross_section_path.moveTo(start[0], start[1])
            end = rotated_segments[i + 1, :]
            cross_section_path.lineTo(end[0], end[1])
        item.setPath(cross_section_path)
        item.setPen(self.path_pen)
        item.setTransform(self.view.scene_transform)
        self.scene.addItem(item)
        rect = item.boundingRect()
        anchor = rect.center()

        ab = self.plane.point - self.parent_model.measurement_points.points
        d = np.einsum('ij,ij->i', np.expand_dims(self.plane.normal, axis=0), ab)
        index = np.where(np.abs(d) < POS_EPS)[0]
        rotated_points = self.parent_model.measurement_points.points[index, :]
        rotated_points = rotated_points @ self.matrix

        for i, p in zip(index, rotated_points):
            point = QtCore.QPointF(p[0], p[1]) * self.sample_scale
            point = self.view.scene_transform.map(point)
            item = GraphicsPointItem(point, size=self.scene.point_size)
            item.setToolTip(f'Point {i + 1}')
            item.fixed = True
            item.makeControllable(self.scene.mode == GraphicsScene.Mode.Select)
            item.setPen(self.point_pen)
            self.scene.addItem(item)
            rect = rect.united(item.boundingRect().translated(point))

        # calculate new rectangle that encloses original rect with a different anchor
        rect.united(rect.translated(anchor - rect.center()))
        self.view.setSceneRect(rect)
        self.view.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        self.view.anchor = rect

    @staticmethod
    def __lookAt(forward):
        """Computes the matrix for the scene camera"""
        rot_matrix = Matrix33.identity()
        up = Vector3([0., -1., 0.]) if -VECTOR_EPS < forward[1] < VECTOR_EPS else Vector3([0., 0., 1.])
        left = up ^ forward
        left.normalize()
        up = forward ^ left

        rot_matrix.c1[:3] = left
        rot_matrix.c2[:3] = up
        rot_matrix.c3[:3] = forward

        return rot_matrix

    def addPoints(self):
        """Adds the points in the scene into the measurement points of the  project"""
        if len(self.scene.items()) < 2:
            return

        points_2d = []
        transform = self.view.scene_transform.inverted()[0]
        for item in self.scene.items():
            if isinstance(item, GraphicsPointItem) and not item.fixed:
                pos = transform.map(item.pos()) / self.sample_scale
                # negate distance due to inverted normal when creating matrix
                points_2d.append([pos.x(), pos.y(), -self.old_distance])
                self.scene.removeItem(item)

        if not points_2d:
            return

        points = points_2d[::-1] @ self.matrix.transpose()
        enabled = [True] * points.shape[0]
        self.parent.presenter.addPoints(list(zip(points, enabled)), PointType.Measurement, False)


class AlignSample(QtWidgets.QWidget):
    """Creates a UI for aligning sample on instrument with 6D pose

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Upper

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.scenes.switchToInstrumentScene()
        self.title = 'Align Sample with 6D pose'
        self.setMinimumWidth(450)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addSpacing(20)
        self.main_layout.addWidget(FormTitle('Create Transformation for Alignment'))
        self.main_layout.addSpacing(10)

        self.main_layout.addWidget(QtWidgets.QLabel('Translation along the X, Y, and Z axis (mm):'))
        self.position_form_group = FormGroup(FormGroup.Layout.Horizontal)
        self.x_position = FormControl('X', 0.0, required=True, number=True)
        self.y_position = FormControl('Y', 0.0, required=True, number=True)
        self.z_position = FormControl('Z', 0.0, required=True, number=True)
        self.position_form_group.addControl(self.x_position)
        self.position_form_group.addControl(self.y_position)
        self.position_form_group.addControl(self.z_position)
        self.position_form_group.group_validation.connect(self.formValidation)
        self.main_layout.addWidget(self.position_form_group)

        self.main_layout.addWidget(QtWidgets.QLabel('Rotation around the X, Y, and Z axis (degrees):'))
        self.orientation_form_group = FormGroup(FormGroup.Layout.Horizontal)
        self.x_rotation = FormControl('X', 0.0, required=True, number=True)
        self.x_rotation.range(-360.0, 360.0)
        self.y_rotation = FormControl('Y', 0.0, required=True, number=True)
        self.y_rotation.range(-360.0, 360.0)
        self.z_rotation = FormControl('Z', 0.0, required=True, number=True)
        self.z_rotation.range(-360.0, 360.0)
        self.orientation_form_group.addControl(self.x_rotation)
        self.orientation_form_group.addControl(self.y_rotation)
        self.orientation_form_group.addControl(self.z_rotation)
        self.orientation_form_group.group_validation.connect(self.formValidation)
        self.main_layout.addWidget(self.orientation_form_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton('Align Sample')
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)

    def formValidation(self):
        if self.position_form_group.valid and self.orientation_form_group.valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        pose = [
            self.x_position.value, self.y_position.value, self.z_position.value, self.z_rotation.value,
            self.y_rotation.value, self.x_rotation.value
        ]

        self.parent.presenter.alignSampleWithPose(pose)


class TomoTiffLoaderDialog(QtWidgets.QDialog):
    """Creates a dialog which allows a stack of TIFF files to be
        :param parent: main window instance
        :type parent: MainWindow
        """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.setWindowTitle('Load series of TIFF files')

        self.filepath_box = QtWidgets.QLineEdit()
        self.filepath_box.setPlaceholderText("Folder path")
        self.filepath_browse_button = QtWidgets.QPushButton('Browse...')
        self.filepath_browse_button.clicked.connect(self.search)

        self.x_pitch_box = QtWidgets.QLineEdit()
        self.x_pitch_box.setValidator(QtGui.QDoubleValidator(self))
        self.x_pitch_box.setPlaceholderText("x")
        self.x_label = QtWidgets.QLabel("Pitch of x pixels in mm")

        self.y_pitch_box = QtWidgets.QLineEdit()
        self.y_pitch_box.setValidator(QtGui.QDoubleValidator(self))
        self.y_pitch_box.setPlaceholderText("y")
        self.y_label = QtWidgets.QLabel("Pitch of y pixels in mm")

        self.z_pitch_box = QtWidgets.QLineEdit()
        self.z_pitch_box.setValidator(QtGui.QDoubleValidator(self))
        self.z_pitch_box.setPlaceholderText("z")
        self.z_label = QtWidgets.QLabel("Pitch of z pixels in mm")

        self.ok_button = QtWidgets.QPushButton('OK')
        self.ok_button.clicked.connect(self.executeButtonClicked)

        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.close)
        self.status_box = QtWidgets.QLabel()
        self.status_box.setStyleSheet('color: red')

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.filepath_box, 0, 0, 1, 3)
        layout.addWidget(self.filepath_browse_button, 0, 3)
        layout.addWidget(self.x_pitch_box, 1, 0)
        layout.addWidget(self.x_label, 1, 3)
        layout.addWidget(self.y_pitch_box, 2, 0)
        layout.addWidget(self.y_label, 2, 3)
        layout.addWidget(self.z_pitch_box, 3, 0)
        layout.addWidget(self.z_label, 3, 3)
        layout.addWidget(self.ok_button, 4, 2)
        layout.addWidget(self.cancel_button, 4, 3)
        layout.addWidget(self.status_box, 5, 0)
        self.setLayout(layout)

    def search(self):
        filepath = self.parent.showOpenTomographyDialog(hdf_flag=False)
        self.filepath_box.setText(str(filepath))

    def executeButtonClicked(self):
        if self.x_pitch_box.text() and self.y_pitch_box.text() and self.z_pitch_box.text() and self.filepath_box.text():
            filepath = self.filepath_box.text()
            x_pitch = self.x_pitch_box.text()
            y_pitch = self.y_pitch_box.text()
            z_pitch = self.z_pitch_box.text()

            array_of_data_and_axes = [filepath, x_pitch, y_pitch, z_pitch]

            self.parent.presenter.importTomography(array_of_data_and_axes)
            self.close()
        else:
            self.status_box.setText("Please enter filepath and all pitches")


class TomoTiffLoader(QtWidgets.QDialog):
    """Creates a dialog which allows a stack of TIFF files to be
        :param parent: main window instance
        :type parent: MainWindow
        """
    dock_flag = DockFlag.Upper

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title = 'Load tomography from TIFFs'
        self.setMinimumWidth(450)
        #self.parent.scenes.switchToSampleScene()
        self.main_layout = QtWidgets.QVBoxLayout()
        spacing = 10
        unit = 'mm'

        self.filepath_layout = FormGroup(FormGroup.Layout.Vertical)
        self.pixel_size_group = FormGroup(FormGroup.Layout.Horizontal)
        self.pixel_centre_group = FormGroup(FormGroup.Layout.Horizontal)


        self.filepath_box = FormControl('Filepath', '', number=False)
        self.file_is_valid = False
        self.filepath_picker = FilePicker(path='', select_folder=True)
        self.filepath_picker.value_changed.connect(self.filepath_validation)


        pixel_size_layout = QtWidgets.QVBoxLayout()
        pixel_size_layout.addWidget(QtWidgets.QLabel('Size of voxel (mm):'))
        self.x_pixel_box = FormControl('X', 1.0, required=True, desc=unit, number=True, decimals=4)
        self.y_pixel_box = FormControl('Y', 1.0, required=True, desc=unit, number=True, decimals=4)
        self.z_pixel_box = FormControl('Z', 1.0, required=True, desc=unit, number=True, decimals=4)
        for box in [self.x_pixel_box, self.y_pixel_box, self.z_pixel_box]:
            box.range(minimum=0.0001, maximum=1000, min_exclusive=True)
        self.pixel_size_group.addControl(self.x_pixel_box)
        self.pixel_size_group.addControl(self.y_pixel_box)
        self.pixel_size_group.addControl(self.z_pixel_box)
        self.pixel_size_group.group_validation.connect(self.formValidation)

        self.x_centre_box = FormControl('X', 0.0, required=True, desc=unit, number=True)
        self.y_centre_box = FormControl('Y', 0.0, required=True, desc=unit, number=True)
        self.z_centre_box = FormControl('Z', 0.0, required=True, desc=unit, number=True)
        self.pixel_centre_group.addControl(self.x_centre_box)
        self.pixel_centre_group.addControl(self.y_centre_box)
        self.pixel_centre_group.addControl(self.z_centre_box)
        self.pixel_centre_group.group_validation.connect(self.formValidation)

        execute_button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(self.title)
        self.execute_button.setDisabled(True)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        execute_button_layout.addWidget(self.execute_button)
        execute_button_layout.addStretch(1)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(FormTitle('Filepath of folder containing TIFFs:'))
        self.main_layout.addWidget(self.filepath_picker)
        self.main_layout.addWidget(FormTitle('Size of voxel (mm):'))
        self.main_layout.addWidget(self.pixel_size_group)
        self.main_layout.addWidget(FormTitle('Centre of image coordinates (mm):'))
        self.main_layout.addWidget(self.pixel_centre_group)
        self.main_layout.addLayout(execute_button_layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

    def executeButtonClicked(self):
        self.filepath = self.filepath_box.text
        x_size = self.x_pixel_box.text
        y_size = self.y_pixel_box.text
        z_size = self.z_pixel_box.text
        x_centre = self.x_centre_box.text
        y_centre = self.y_centre_box.text
        z_centre = self.z_centre_box.text

        self.sizes_and_centres = [x_size, y_size, z_size, x_centre, y_centre, z_centre]

        self.parent.presenter.importTomography(self.filepath, self.sizes_and_centres)
        self.close()

    def filepath_validation(self):
        if not self.filepath_picker.value == '':
            self.file_is_valid = True
            self.formValidation()

    def formValidation(self):
        if self.pixel_centre_group.valid and self.pixel_size_group.valid and self.file_is_valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)
