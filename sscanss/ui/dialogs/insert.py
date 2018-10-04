from enum import Enum, unique
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.math import Plane, Matrix33, Vector3, clamp, map_range
from sscanss.core.mesh import mesh_plane_intersection
from sscanss.core.util import (Primitives, CompareOperator, DockFlag, StrainComponents, PointType)
from sscanss.ui.widgets import FormGroup, FormControl, GraphicsView, GraphicsScene
from .managers import PointManager


class InsertPrimitiveDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper
    formSubmitted = QtCore.pyqtSignal(Primitives, dict)

    def __init__(self, primitive, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = self.parent.presenter.model

        self.primitive = primitive
        self.formSubmitted.connect(parent.presenter.addPrimitive)

        self.minimum = 0
        self.maximum = 10000

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

        self.title = 'Insert {}'.format(self.primitive.value)
        self.setMinimumWidth(350)
        self.textboxes['name'].setFocus()

    def createPrimitiveSwitcher(self):
        switcher_layout = QtWidgets.QHBoxLayout()
        switcher = QtWidgets.QToolButton()
        switcher.setObjectName('ToolButton')
        switcher.setArrowType(QtCore.Qt.DownArrow)
        switcher.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        switcher.setMenu(self.parent.primitives_menu)
        switcher_layout.addStretch(1)
        switcher_layout.addWidget(switcher)
        self.main_layout.addLayout(switcher_layout)

    def createFormInputs(self):
        self.form_group = FormGroup()
        for key, value in self.mesh_args.items():
            pretty_label = key.replace('_', ' ').title()

            if key == 'name':
                control = FormControl(pretty_label, value, required=True)
            else:
                control = FormControl(pretty_label, value, unit='mm', required=True)
                control.range(self.minimum, self.maximum, min_exclusive=True)

            self.textboxes[key] = control
            self.form_group.addControl(control)

        if self.primitive == Primitives.Tube:
            outer_radius = self.textboxes['outer_radius']
            inner_radius = self.textboxes['inner_radius']

            outer_radius.compareWith(inner_radius, CompareOperator.Greater)
            inner_radius.compareWith(outer_radius, CompareOperator.Less)

        self.main_layout.addWidget(self.form_group)
        self.form_group.groupValidation.connect(self.formValidation)

    def formValidation(self, is_valid):
        if is_valid:
            self.create_primitive_button.setEnabled(True)
        else:
            self.create_primitive_button.setDisabled(True)

    def createPrimiviteButtonClicked(self):
        for key, textbox in self.textboxes.items():
            value = textbox.value
            self.mesh_args[key] = value

        self.formSubmitted.emit(self.primitive, self.mesh_args)
        new_name = self.parent_model.uniqueKey(self.primitive.value)
        self.textboxes['name'].value = new_name


class InsertPointDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper

    def __init__(self, point_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.point_type = point_type
        self.title = 'Add {} Point'.format(point_type.value)
        self.main_layout = QtWidgets.QVBoxLayout()
        unit = 'mm'
        self.form_group = FormGroup()
        self.x_axis = FormControl('X', 0.0, required=True, unit=unit)
        self.x_axis.range(-10000, 10000)
        self.y_axis = FormControl('Y', 0.0, required=True, unit=unit)
        self.y_axis.range(-10000, 10000)
        self.z_axis = FormControl('Z', 0.0, required=True, unit=unit)
        self.z_axis.range(-10000, 10000)
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.groupValidation.connect(self.formValidation)
        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(self.title)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addWidget(self.form_group)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.setMinimumWidth(350)

    def formValidation(self, is_valid):
        if is_valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        point = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        self.parent.presenter.addPoints([(point, True)], self.point_type)


class InsertVectorDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
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
        align_count = self.parent_model.measurement_vectors.shape[2]
        alignment_list = ['{}'.format(i+1) for i in range(align_count)]
        alignment_list.append('Add New...')
        self.alignment_combobox.addItems(alignment_list)
        self.alignment_combobox.activated.connect(self.addNewAlignment)
        self.alignment_combobox.currentIndexChanged.connect(self.changeRenderedAlignment)
        alignment_layout.addWidget(self.alignment_combobox)
        alignment_layout.addSpacing(spacing)
        layout.addLayout(alignment_layout)
        layout.addSpacing(spacing)

        detector_layout = QtWidgets.QVBoxLayout()
        detector_layout.addWidget(QtWidgets.QLabel('Detector:'))
        self.detector_combobox = QtWidgets.QComboBox()
        self.detector_combobox.setView(QtWidgets.QListView())
        self.detector_combobox.addItems(['1', '2'])
        detector_layout.addWidget(self.detector_combobox)
        detector_layout.addSpacing(spacing)
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
        self.setMinimumWidth(350)

    def updatePointList(self):
        self.points_combobox.clear()
        point_list = ['All Points']
        point_list.extend(['{}'.format(i+1) for i in range(self.parent_model.measurement_points.size)])
        self.points_combobox.addItems(point_list)

    def addNewAlignment(self, index):
        if index == self.alignment_combobox.count() - 1:
            self.alignment_combobox.insertItem(index, '{}'.format(index + 1))
            self.alignment_combobox.setCurrentIndex(index)

    def changeRenderedAlignment(self, index):
        if index < self.alignment_combobox.count() - 1:
            self.parent_model.rendered_alignment = index
            self.parent_model.updateSampleScene('measurement_vectors')

    def toggleKeyInBox(self, selected_text):
        strain_component = StrainComponents(selected_text)
        if strain_component == StrainComponents.custom:
            self.key_in_box.setVisible(True)
            self.form_group.validateGroup()
        else:
            self.key_in_box.setVisible(False)
            self.execute_button.setEnabled(True)

    def createKeyInBox(self):
        self.key_in_box = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout()

        self.form_group = FormGroup(FormGroup.Layout.Horizontal)
        self.x_axis = FormControl('X', 0.0, required=True)
        self.x_axis.range(-1.0, 1.0)
        self.y_axis = FormControl('Y', 0.0, required=True)
        self.y_axis.range(-1.0, 1.0)
        self.z_axis = FormControl('Z', 0.0, required=True)
        self.z_axis.range(-1.0, 1.0)
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.groupValidation.connect(self.formValidation)

        layout.addWidget(self.form_group)
        self.key_in_box.setLayout(layout)
        self.main_layout.addWidget(self.key_in_box)
        self.toggleKeyInBox(self.component_combobox.currentText())

    def formValidation(self, is_valid):
        if is_valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        points = self.points_combobox.currentIndex() - 1

        selected_text = self.component_combobox.currentText()
        strain_component = StrainComponents(selected_text)

        alignment = self.alignment_combobox.currentIndex()
        detector = self.detector_combobox.currentIndex()
        check_state = self.reverse_checkbox.checkState()
        reverse = True if check_state == QtCore.Qt.Checked else False

        if strain_component == StrainComponents.custom:
            vector = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        else:
            vector = None

        self.parent.presenter.addVectors(points, strain_component, alignment, detector,
                                         key_in=vector, reverse=reverse)


class PickPointDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Full

    @unique
    class PlaneOptions(Enum):
        XY = 'XY plane'
        XZ = 'XZ plane'
        YZ = 'YZ plane'
        Custom = 'Custom Normal'

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.title = 'Add Measurement Points Graphically'
        self.setMinimumWidth(500)

        self.plane_offset_range = (-1., 1.)
        self.slider_range = (-10000000, 10000000)
        self.path_pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 1)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        button_layout = QtWidgets.QHBoxLayout()
        self.help_button = self.__createToolButton(tooltip='Help', style_name='ToolButton',
                                                   icon=QtGui.QIcon('../static/images/question.png'))
        self.help_button.clicked.connect(self.showHelp)

        self.reset_button = self.__createToolButton(tooltip='Reset View', style_name='ToolButton',
                                                    icon=QtGui.QIcon('../static/images/refresh.png'))
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

    def hide(self):
        self.parent_model.removePlane()
        super().hide()

    def prepareMesh(self):
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
            self.parent_model.removePlane()

    def createGraphicsView(self):
        self.scene = GraphicsScene(self)
        self.view = GraphicsView(self.scene)
        self.view.setMinimumHeight(350)
        self.splitter.addWidget(self.view)

    def createControlPanel(self):
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumHeight(250)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.South)
        self.splitter.addWidget(self.tabs)

        self.createPlaneTab()
        self.createSelectionToolsTab()
        self.createGridOptionsTab()
        point_manager = PointManager(PointType.Measurement, self.parent)
        self.tabs.addTab(self.__createScrollArea(point_manager), 'Point Manager')

    def createPlaneTab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Specify Plane:'))
        self.plane_combobox = QtWidgets.QComboBox()
        self.plane_combobox.setView(QtWidgets.QListView())
        self.plane_combobox.addItems([p.value for p in self.PlaneOptions])
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
        self.tabs.addTab(self.__createScrollArea(plane_tab), 'Define Plane')

    def createSelectionToolsTab(self):
        layout = QtWidgets.QVBoxLayout()
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel('Select Geometry of Points: '))
        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.buttonClicked[int].connect(self.changeSceneMode)

        self.object_selector = self.__createToolButton(checkable=True, checked=True, tooltip='Select Points',
                                                       style_name='PickDialogToolButton',
                                                       icon=QtGui.QIcon('../static/images/select.png'))
        self.point_selector = self.__createToolButton(checkable=True, tooltip='Draw a Point',
                                                      style_name='PickDialogToolButton',
                                                      icon=QtGui.QIcon('../static/images/point.png'))
        self.line_selector = self.__createToolButton(checkable=True, tooltip='Draw Points on Line',
                                                     style_name='PickDialogToolButton',
                                                     icon=QtGui.QIcon('../static/images/line_tool.png'))
        self.area_selector = self.__createToolButton(checkable=True, tooltip='Draw Points on Area',
                                                     style_name='PickDialogToolButton',
                                                     icon=QtGui.QIcon('../static/images/area_tool.png'))

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
        self.tabs.addTab(self.__createScrollArea(select_tab), 'Selection Tools')

    @staticmethod
    def __createScrollArea(content):
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setViewportMargins(0, 10, 0, 0)
        scroll_area.setAutoFillBackground(True)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        return scroll_area

    @staticmethod
    def __createToolButton(checkable=False, checked=False, tooltip='', style_name='', icon=None):
        button = QtWidgets.QToolButton()
        button.setCheckable(checkable)
        button.setChecked(checked)
        if tooltip:
            button.setToolTip(tooltip)
        if style_name:
            button.setObjectName(style_name)
        if icon is not None:
            button.setIcon(icon)

        return button

    def createGridOptionsTab(self):
        layout = QtWidgets.QVBoxLayout()
        self.show_grid_checkbox = QtWidgets.QCheckBox('Show Grid')
        self.show_grid_checkbox.stateChanged.connect(self.showGrid)
        self.snap_to_grid_checkbox = QtWidgets.QCheckBox('Snap Selection to Grid')
        self.snap_to_grid_checkbox.stateChanged.connect(self.snapToGrid)
        self.snap_to_grid_checkbox.setEnabled(self.view.show_grid)
        layout.addWidget(self.show_grid_checkbox)
        layout.addWidget(self.snap_to_grid_checkbox)
        self.createGridSizeWidget()
        layout.addWidget(self.grid_size_widget)
        layout.addStretch(1)

        grid_tab = QtWidgets.QWidget()
        grid_tab.setLayout(layout)
        self.tabs.addTab(self.__createScrollArea(grid_tab), 'Grid Options')

    def createCustomPlaneBox(self):
        self.custom_plane_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout()

        self.form_group = FormGroup(FormGroup.Layout.Horizontal)
        self.x_axis = FormControl('X', 1.0, required=True)
        self.x_axis.range(-1.0, 1.0)
        self.y_axis = FormControl('Y', 0.0, required=True)
        self.y_axis.range(-1.0, 1.0)
        self.z_axis = FormControl('Z', 0.0, required=True)
        self.z_axis.range(-1.0, 1.0)
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.groupValidation.connect(self.setCustomPlane)

        layout.addWidget(self.form_group)
        self.custom_plane_widget.setLayout(layout)
        self.main_layout.addWidget(self.custom_plane_widget)

    def createLineToolWidget(self):
        self.line_tool_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 20, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Number of Points: '))
        self.line_point_count_spinbox = QtWidgets.QSpinBox()
        self.line_point_count_spinbox.setRange(2, 1000)
        self.line_point_count_spinbox.valueChanged.connect(self.scene.setLineToolPointCount)

        layout.addWidget(self.line_point_count_spinbox)
        self.line_tool_widget.setVisible(False)
        self.line_tool_widget.setLayout(layout)

    def createAreaToolWidget(self):
        self.area_tool_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 20, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Number of Points: '))
        self.area_x_spinbox = QtWidgets.QSpinBox()
        self.area_x_spinbox.setValue(self.scene.area_tool_size[0])
        self.area_x_spinbox.setRange(2, 1000)
        self.area_y_spinbox = QtWidgets.QSpinBox()
        self.area_y_spinbox.setValue(self.scene.area_tool_size[1])
        self.area_y_spinbox.setRange(2, 1000)

        stretch_factor = 3
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('X: '))
        self.area_x_spinbox.valueChanged.connect(lambda: self.scene.setAreaToolPointCount(self.area_x_spinbox.value(),
                                                                                          self.area_y_spinbox.value()))
        layout.addWidget(self.area_x_spinbox, stretch_factor)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('Y: '))
        self.area_y_spinbox.valueChanged.connect(lambda: self.scene.setAreaToolPointCount(self.area_x_spinbox.value(),
                                                                                          self.area_y_spinbox.value()))
        layout.addWidget(self.area_y_spinbox, stretch_factor)
        self.area_tool_widget.setVisible(False)
        self.area_tool_widget.setLayout(layout)

    def createGridSizeWidget(self):
        self.grid_size_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 20, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Grid Size: '))
        self.grid_x_spinbox = QtWidgets.QSpinBox()
        self.grid_x_spinbox.setValue(self.view.grid_x_size)
        self.grid_x_spinbox.setRange(2, 1000)
        self.grid_y_spinbox = QtWidgets.QSpinBox()
        self.grid_y_spinbox.setValue(self.view.grid_y_size)
        self.grid_y_spinbox.setRange(2, 1000)

        stretch_factor = 3
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('X: '))
        self.grid_x_spinbox.valueChanged.connect(lambda: self.view.setGridSize(self.grid_x_spinbox.value(),
                                                                               self.grid_y_spinbox.value()))
        layout.addWidget(self.grid_x_spinbox, stretch_factor)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('Y: '))
        self.grid_y_spinbox.valueChanged.connect(lambda: self.view.setGridSize(self.grid_x_spinbox.value(),
                                                                               self.grid_y_spinbox.value()))
        layout.addWidget(self.grid_y_spinbox, stretch_factor)
        self.grid_size_widget.setVisible(False)
        self.grid_size_widget.setLayout(layout)

    def changeSceneMode(self, buttonid):
        self.scene.mode = GraphicsScene.Mode(buttonid)
        self.line_tool_widget.setVisible(self.scene.mode == GraphicsScene.Mode.Draw_line)
        self.area_tool_widget.setVisible(self.scene.mode == GraphicsScene.Mode.Draw_area)

    def showHelp(self):
        self.view.show_help = False if self.view.has_foreground else True
        self.scene.update()

    def showGrid(self, state):
        self.view.show_grid = True if state == QtCore.Qt.Checked else False
        self.snap_to_grid_checkbox.setEnabled(self.view.show_grid)
        self.grid_size_widget.setVisible(self.view.show_grid)
        self.scene.update()

    def snapToGrid(self, state):
        self.view.snap_to_grid = True if state == QtCore.Qt.Checked else False

    def updateSlider(self, value):
        if not self.plane_lineedit.hasAcceptableInput():
            return

        new_distance = clamp(float(value), *self.plane_offset_range)
        slider_value = int(map_range(*self.plane_offset_range, *self.slider_range, new_distance))
        self.plane_slider.setValue(slider_value)

        offset = new_distance - self.old_distance
        self.parent_model.addPlane(shift_by=offset * self.plane.normal)
        self.old_distance = new_distance

    def updateLineEdit(self, value):
        new_distance = map_range(*self.slider_range, *self.plane_offset_range, value)
        self.plane_lineedit.setText('{:.3f}'.format(new_distance))

        offset = new_distance - self.old_distance
        self.parent_model.addPlane(shift_by=offset * self.plane.normal)
        self.old_distance = new_distance

    def movePlane(self):
        distance = clamp(float(self.plane_lineedit.text()), *self.plane_offset_range)
        self.plane_lineedit.setText('{:.3f}'.format(distance))
        point = distance * self.plane.normal
        self.plane = Plane(self.plane.normal, point)
        self.updateCrossSection()

    def setCustomPlane(self, is_valid):
        if is_valid:
            normal = np.array([self.x_axis.value, self.y_axis.value, self.z_axis.value])
            try:
                self.initializePlane(normal, self.mesh.bounding_box.center)
            except ValueError:
                self.x_axis.validation_label.setText('Bad Normal')

    def setPlane(self, selected_text):
        if selected_text == self.PlaneOptions.Custom.value:
            self.custom_plane_widget.setVisible(True)
            self.form_group.validateGroup()
            return
        else:
            self.custom_plane_widget.setVisible(False)

        if selected_text == self.PlaneOptions.XY.value:
            plane_normal = np.array([0., 0., 1.])
        elif selected_text == self.PlaneOptions.XZ.value:
            plane_normal = np.array([0., 1., 0.])
        else:
            plane_normal = np.array([1., 0., 0.])

        self.initializePlane(plane_normal, self.mesh.bounding_box.center)

    def initializePlane(self, plane_normal, plane_point):
        self.plane = Plane(plane_normal, plane_point)
        plane_size = self.mesh.bounding_box.radius

        self.parent_model.addPlane(self.plane, 2 * plane_size, 2 * plane_size)
        distance = self.plane.distanceFromOrigin()
        self.plane_offset_range = (distance - plane_size, distance + plane_size)
        slider_value = int(map_range(*self.plane_offset_range, *self.slider_range, distance))
        self.plane_slider.setValue(slider_value)
        self.plane_lineedit.setText('{:.3f}'.format(distance))
        self.old_distance = distance
        # inverted the normal so that the y-axis is flipped
        self.matrix = self.__lookAt(-Vector3(self.plane.normal))
        self.view.reset()
        self.updateCrossSection()

    def updateCrossSection(self):
        self.scene.clear()
        segments = mesh_plane_intersection(self.mesh, self.plane)
        if len(segments) == 0:
            return
        segments = np.array(segments)

        item = QtWidgets.QGraphicsPathItem()
        cross_section_path = QtGui.QPainterPath()
        rotated_segments = segments.dot(self.matrix[:])
        for i in range(0, rotated_segments.shape[0], 2):
            start = rotated_segments[i, :]
            cross_section_path.moveTo(start[0], start[1])
            end = rotated_segments[i + 1, :]
            cross_section_path.lineTo(end[0], end[1])
        item.setPath(cross_section_path)
        item.setPen(self.path_pen)
        item.setTransform(self.view.scene_transform)
        self.scene.addItem(item)
        self.view.setSceneRect(item.boundingRect())
        self.scene.clearSelection()
        self.scene.update()

    @staticmethod
    def __lookAt(forward):
        eps = 1e-6
        rot_matrix = Matrix33.identity()
        up = Vector3([0., -1., 0.]) if -eps < forward[1] < eps else Vector3([0., 0., 1.])
        left = up ^ forward
        left.normalize()
        up = forward ^ left

        rot_matrix.c1[:3] = left
        rot_matrix.c2[:3] = up
        rot_matrix.c3[:3] = forward

        return rot_matrix

    def addPoints(self):
        if len(self.scene.items()) < 2:
            return

        points_2d = []
        transform = self.view.scene_transform.inverted()[0]
        for item in self.scene.items():
            if not isinstance(item, QtWidgets.QGraphicsPathItem):
                pos = transform.map(item.pos())
                # negate distance due to inverted normal when creating matrix
                point = np.array([pos.x(), pos.y(), -self.old_distance])
                points_2d.append(point)
                self.scene.removeItem(item)

        _matrix = self.matrix.transpose()
        points = np.array(points_2d).dot(_matrix[:])
        enabled = [True] * points.shape[0]
        self.parent.presenter.addPoints(list(zip(points, enabled)), PointType.Measurement, False)
