import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.config import path_for, settings
from sscanss.core.math import Plane, Matrix33, Vector3, clamp, map_range, trunc
from sscanss.core.geometry import mesh_plane_intersection
from sscanss.core.util import Primitives, DockFlag, StrainComponents, PointType, PlaneOptions, Attributes
from sscanss.ui.widgets import (FormGroup, FormControl, GraphicsView, GraphicsScene, create_tool_button, FormTitle,
                                create_scroll_area, CompareValidator, GraphicsPointItem, Grid, create_icon)
from .managers import PointManager


class InsertPrimitiveDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper
    formSubmitted = QtCore.pyqtSignal(Primitives, dict)

    def __init__(self, primitive, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = self.parent.presenter.model
        self.parent.scenes.switchToSampleScene()

        self.primitive = primitive
        self.formSubmitted.connect(parent.presenter.addPrimitive)

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
        switcher = create_tool_button(style_name='MenuButton', status_tip='Open dialog for a different primitive')
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
                control = FormControl(pretty_label, value, desc='mm', required=True, number=True)
                control.range(0, 10000, min_exclusive=True)

            self.textboxes[key] = control
            self.form_group.addControl(control)

        if self.primitive == Primitives.Tube:
            outer_radius = self.textboxes['outer_radius']
            inner_radius = self.textboxes['inner_radius']

            outer_radius.compareWith(inner_radius, CompareValidator.Operator.Greater)
            inner_radius.compareWith(outer_radius, CompareValidator.Operator.Less)

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
        self.parent.scenes.switchToSampleScene()
        self.point_type = point_type
        self.title = 'Add {} Point'.format(point_type.value)
        self.main_layout = QtWidgets.QVBoxLayout()
        unit = 'mm'
        self.form_group = FormGroup()
        self.x_axis = FormControl('X', 0.0, required=True, desc=unit, number=True)
        self.x_axis.range(-10000, 10000)
        self.y_axis = FormControl('Y', 0.0, required=True, desc=unit, number=True)
        self.y_axis.range(-10000, 10000)
        self.z_axis = FormControl('Z', 0.0, required=True, desc=unit, number=True)
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
        layout.addSpacing(spacing)

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
        self.setMinimumWidth(350)

    def updatePointList(self):
        self.points_combobox.clear()
        point_list = ['All Points']
        point_list.extend(['{}'.format(i+1) for i in range(self.parent_model.measurement_points.size)])
        self.points_combobox.addItems(point_list)

    def updateAlignment(self):
        align_count = self.parent_model.measurement_vectors.shape[2]
        if align_count != self.alignment_combobox.count() - 1:
            self.alignment_combobox.clear()
            alignment_list = ['{}'.format(i + 1) for i in range(align_count)]
            alignment_list.append('Add New...')
            self.alignment_combobox.addItems(alignment_list)

        self.alignment_combobox.setCurrentIndex(self.parent.scenes.rendered_alignment)

    def addNewAlignment(self, index):
        if index == self.alignment_combobox.count() - 1:
            self.alignment_combobox.insertItem(index, '{}'.format(index + 1))
            self.alignment_combobox.setCurrentIndex(index)

    def changeRenderedAlignment(self, index):
        align_count = self.parent_model.measurement_vectors.shape[2]
        if 0 <= index < align_count:
            self.parent.scenes.changeRenderedAlignment(index)
        elif index >= align_count:
            self.parent.scenes.toggleVisibility(Attributes.Vectors, False)

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
        self.x_axis = FormControl('X', 1.0, required=True, number=True)
        self.x_axis.range(-1.0, 1.0)
        self.y_axis = FormControl('Y', 0.0, required=True, number=True)
        self.y_axis.range(-1.0, 1.0)
        self.z_axis = FormControl('Z', 0.0, required=True, number=True)
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
        self.execute_button.setDisabled(True)
        if is_valid:
            if np.linalg.norm([self.x_axis.value, self.y_axis.value, self.z_axis.value]) > 1e-6:
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
        reverse = True if check_state == QtCore.Qt.Checked else False

        if strain_component == StrainComponents.custom:
            vector = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        else:
            vector = None

        self.parent.presenter.addVectors(points, strain_component, alignment, detector,
                                         key_in=vector, reverse=reverse)
        # New vectors are drawn by the scene manager after function ends
        self.parent.scenes._rendered_alignment = alignment

    def closeEvent(self, event):
        self.parent.scenes.changeRenderedAlignment(0)
        event.accept()


class PickPointDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Full

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.title = 'Add Measurement Points Graphically'
        self.setMinimumWidth(500)

        self.plane_offset_range = (-1., 1.)
        self.sample_scale = 2
        self.slider_range = (-10000000, 10000000)
        self.path_pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 1)
        self.point_pen = QtGui.QPen(QtGui.QColor(127, 0, 0), 1)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        button_layout = QtWidgets.QHBoxLayout()
        self.help_button = create_tool_button(tooltip='Help', style_name='ToolButton',
                                              status_tip='Display shortcuts for the cross-section view',
                                              icon_path=path_for('question.png'))
        self.help_button.clicked.connect(self.showHelp)

        self.reset_button = create_tool_button(tooltip='Reset View', style_name='ToolButton',
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

    def closeEvent(self, event):
        self.parent.scenes.removePlane()
        event.accept()

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
            self.parent.scenes.removePlane()

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
        self.tabs.addTab(create_scroll_area(point_manager), 'Point Manager')

    def createPlaneTab(self):
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
        layout = QtWidgets.QVBoxLayout()
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel('Select Geometry of Points: '))
        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.buttonClicked[int].connect(self.changeSceneMode)

        self.object_selector = create_tool_button(checkable=True, checked=True, tooltip='Select Points',
                                                  status_tip='Select movable points from the cross-section view',
                                                  style_name='MidToolButton', icon_path=path_for('select.png'))
        self.point_selector = create_tool_button(checkable=True, tooltip='Draw a Point',
                                                 status_tip='Draw a single point at the selected position',
                                                 style_name='MidToolButton', icon_path=path_for('point.png'))
        self.line_selector = create_tool_button(checkable=True, tooltip='Draw Points on Line',
                                                status_tip='Draw equally spaced points on the selected line',
                                                style_name='MidToolButton', icon_path=path_for('line_tool.png'))
        self.area_selector = create_tool_button(checkable=True, tooltip='Draw Points on Area',
                                                status_tip='Draw a grid of points on the selected area',
                                                style_name='MidToolButton', icon_path=path_for('area_tool.png'))

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

    def createGridWidget(self):
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
        self.grid_x_spinbox = QtWidgets.QSpinBox()
        self.grid_x_spinbox.valueChanged.connect(self.changeGridSize)
        self.grid_y_label = QtWidgets.QLabel('')
        self.grid_y_spinbox = QtWidgets.QSpinBox()
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
        grid_x = self.grid_x_spinbox.value()
        grid_y = self.grid_y_spinbox.value()
        if self.view.grid.type == Grid.Type.Box:
            grid_x *= self.sample_scale
            grid_y *= self.sample_scale
        else:
            grid_x *= self.sample_scale
        self.view.setGridSize((grid_x, grid_y))

    def setGridType(self, grid_type):
        self.view.setGridType(grid_type)
        size = self.view.grid.size
        if grid_type == Grid.Type.Box:
            self.grid_x_label.setText('X (mm): ')
            self.grid_y_label.setText('Y (mm): ')
            self.grid_x_spinbox.setValue(size[0])
            self.grid_y_spinbox.setValue(size[1])
            self.grid_x_spinbox.setRange(1, 1000)
            self.grid_y_spinbox.setRange(1, 1000)
        else:
            self.grid_x_label.setText('Radial (mm): ')
            self.grid_y_label.setText('Angular (degree): ')
            self.grid_x_spinbox.setValue(size[0])
            self.grid_y_spinbox.setValue(size[1])
            self.grid_x_spinbox.setRange(1, 1000)
            self.grid_y_spinbox.setRange(1, 360)

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
        self.grid_widget.setVisible(self.view.show_grid)
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
        self.parent.scenes.drawPlane(shift_by=offset * self.plane.normal)
        self.old_distance = new_distance

    def updateLineEdit(self, value):
        new_distance = trunc(map_range(*self.slider_range, *self.plane_offset_range, value), 3)
        self.plane_lineedit.setText('{:.3f}'.format(new_distance))

        offset = new_distance - self.old_distance
        self.parent.scenes.drawPlane(shift_by=offset * self.plane.normal)
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
        self.plane = Plane(plane_normal, plane_point)
        plane_size = self.mesh.bounding_box.radius

        self.parent.scenes.drawPlane(self.plane, 2 * plane_size, 2 * plane_size)
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

        ab = self.plane.point - self.parent_model.measurement_points.points
        d = np.einsum('ij,ij->i', np.expand_dims(self.plane.normal, axis=0), ab)
        index = np.where(np.abs(d) < 1e-5)[0]
        rotated_points = self.parent_model.measurement_points.points[index, :]
        rotated_points = rotated_points @ self.matrix

        for i, p in zip(index, rotated_points):
            point = QtCore.QPointF(p[0], p[1]) * self.sample_scale
            point = self.view.scene_transform.map(point)
            item = GraphicsPointItem(point)
            item.setToolTip(f'Point {i + 1}')
            item.fixed = True
            item.makeControllable(self.scene.mode == GraphicsScene.Mode.Select)
            item.setPen(self.point_pen)
            self.scene.addItem(item)
            rect = rect.united(item.boundingRect().translated(point))

        self.view.fitInView(rect, QtCore.Qt.KeepAspectRatio)
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
            if isinstance(item, GraphicsPointItem) and not item.fixed:
                pos = transform.map(item.pos()) / self.sample_scale
                # negate distance due to inverted normal when creating matrix
                points_2d.append([pos.x(), pos.y(), -self.old_distance])
                self.scene.removeItem(item)

        if not points_2d:
            return

        points = points_2d @ self.matrix.transpose()
        enabled = [True] * points.shape[0]
        self.parent.presenter.addPoints(list(zip(points, enabled)), PointType.Measurement, False)


class AlignSample(QtWidgets.QWidget):
    dock_flag = DockFlag.Upper

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.scenes.switchToInstrumentScene()
        self.title = 'Align Sample with 6D pose'
        self.setMinimumWidth(350)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addSpacing(20)
        self.main_layout.addWidget(FormTitle('Create Transformation for Alignment'))
        self.main_layout.addSpacing(10)

        self.main_layout.addWidget(QtWidgets.QLabel('Translation along the X, Y, and Z axis (mm):'))
        self.position_form_group = FormGroup(FormGroup.Layout.Horizontal)
        self.x_position = FormControl('X', 0.0, required=True, number=True)
        self.x_position.range(-10000.0, 10000.0)
        self.y_position = FormControl('Y', 0.0, required=True, number=True)
        self.y_position.range(-10000.0, 10000.0)
        self.z_position = FormControl('Z', 0.0, required=True, number=True)
        self.z_position.range(-10000.0, 10000.0)
        self.position_form_group.addControl(self.x_position)
        self.position_form_group.addControl(self.y_position)
        self.position_form_group.addControl(self.z_position)
        self.position_form_group.groupValidation.connect(self.formValidation)
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
        self.orientation_form_group.groupValidation.connect(self.formValidation)
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
        pose = [self.x_position.value, self.y_position.value, self.z_position.value,
                self.x_rotation.value, self.y_rotation.value, self.z_rotation.value]

        self.parent.presenter.alignSampleWithPose(pose)
