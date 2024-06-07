import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from sscanss.config import settings
from sscanss.core.math import Plane, clamp, map_range, trunc, view_from_plane, VECTOR_EPS, POS_EPS, Line
from sscanss.core.geometry import mesh_plane_intersection, Mesh, volume_plane_intersection, line_box_intersection
from sscanss.core.util import (Primitives, DockFlag, StrainComponents, PointType, PlaneOptions, create_tool_button,
                               create_scroll_area, create_icon, FormTitle, CompareValidator, FormGroup, FormControl,
                               FilePicker, Anchor)
from sscanss.app.widgets import GraphicsView, GraphicsScene, GraphicsPointItem, Grid, GraphicsImageItem
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
        if self.primitive == Primitives.Tube:
            self.mesh_args = {'outer_radius': 100.000, 'inner_radius': 50.000, 'height': 200.000}
        elif self.primitive == Primitives.Sphere:
            self.mesh_args = {'radius': 100.000}
        elif self.primitive == Primitives.Cylinder:
            self.mesh_args = {'radius': 100.000, 'height': 200.000}
        elif self.primitive == Primitives.Cuboid:
            self.mesh_args = {'width': 50.000, 'height': 100.000, 'depth': 200.000}
        else:
            self.mesh_args = {'radius': 100.000, 'height': 200.000}

        self.createPrimitiveSwitcher()
        self.createFormInputs()

        button_layout = QtWidgets.QHBoxLayout()
        self.create_primitive_button = QtWidgets.QPushButton('Create', objectName='GreyTextPushButton')
        self.create_primitive_button.clicked.connect(self.createPrimiviteButtonClicked)
        button_layout.addWidget(self.create_primitive_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = f'Insert {self.primitive.value}'
        self.setMinimumWidth(450)
        list(self.textboxes.values())[0].setFocus()

    def createPrimitiveSwitcher(self):
        """Creates a button to switch primitive type"""
        switcher_layout = QtWidgets.QHBoxLayout()
        switcher = create_tool_button(style_name='MenuButton', status_tip='Open dialog for a different primitive')
        switcher.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        switcher.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        switcher.setMenu(self.parent.primitives_menu)
        switcher_layout.addStretch(1)
        switcher_layout.addWidget(switcher)
        self.main_layout.addLayout(switcher_layout)

    def createFormInputs(self):
        """Creates inputs for primitive arguments"""
        self.form_group = FormGroup()
        for key, value in self.mesh_args.items():
            pretty_label = key.replace('_', ' ').title()

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
        self.execute_button = QtWidgets.QPushButton(self.title, objectName='GreyTextPushButton')
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

        self.add_new_text = 'Add New...'
        layout = QtWidgets.QHBoxLayout()
        alignment_layout = QtWidgets.QVBoxLayout()
        alignment_layout.addWidget(QtWidgets.QLabel('Alignment:'))
        self.alignment_combobox = QtWidgets.QComboBox()
        self.alignment_combobox.setView(QtWidgets.QListView())
        self.alignment_combobox.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAtCurrent)
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
        self.updateStrainComponents()
        self.component_combobox.currentTextChanged.connect(self.toggleKeyInBox)
        self.main_layout.addWidget(self.component_combobox)
        self.main_layout.addSpacing(spacing)

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton(self.title, objectName='GreyTextPushButton')
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
        self.parent_model.sample_changed.connect(self.updateStrainComponents)
        self.parent_model.measurement_points_changed.connect(self.updatePointList)
        self.parent_model.measurement_vectors_changed.connect(self.updateAlignment)
        self.parent.scenes.rendered_alignment_changed.connect(self.alignment_combobox.setCurrentIndex)
        self.setMinimumWidth(450)

    def updateStrainComponents(self):
        self.component_combobox.clear()
        if isinstance(self.parent_model.sample, Mesh):
            strain_components = [s.value for s in StrainComponents]
        else:
            strain_components = [
                StrainComponents.ParallelX.value, StrainComponents.ParallelY.value, StrainComponents.ParallelZ.value,
                StrainComponents.Custom.value
            ]
        self.component_combobox.addItems(strain_components)

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
            alignment_list.append(self.add_new_text)
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
        if self.add_new_text == self.alignment_combobox.currentText():
            return

        self.parent.scenes.changeRenderedAlignment(index)

    def toggleKeyInBox(self, selected_text):
        """Shows/Hides the inputs for key-in vector when appropriate strain component is selected

        :param selected_text: strain component
        :type selected_text: str
        """
        if not selected_text:
            return

        strain_component = StrainComponents(selected_text)
        if strain_component == StrainComponents.Custom:
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
        reverse = (check_state == QtCore.Qt.CheckState.Checked)

        if strain_component == StrainComponents.Custom:
            vector = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        else:
            vector = None

        self.parent.presenter.addVectors(points, strain_component, alignment, detector, key_in=vector, reverse=reverse)
        # New vectors are drawn by the scene manager after function ends
        self.parent.scenes.changeRenderedAlignment(alignment)

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

        self.old_distance = None
        self.cross_section_rect = QtCore.QRectF()
        self.plane_offset_range = (-1., 1.)
        self.slider_range = (-10000000, 10000000)
        self.reference_planes = []
        self.sample_id = 'sample'

        self.sample_scale = 20  # This is necessary to allow sub-pixel values on the grid sizes
        self.path_pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 0)
        self.ref_path_pen = QtGui.QPen(QtGui.QColor(159, 43, 104), 0)
        self.point_pen = QtGui.QPen(QtGui.QColor(200, 0, 0), 0)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        button_layout = QtWidgets.QHBoxLayout()
        self.help_button = create_tool_button(tooltip='Help',
                                              style_name='ToolButton',
                                              status_tip='Display shortcuts for the cross-section view',
                                              icon='question.png')
        self.help_button.clicked.connect(self.showHelp)

        self.reset_button = create_tool_button(tooltip='Reset View',
                                               style_name='ToolButton',
                                               status_tip='Reset camera transformation of the cross-section view',
                                               icon='refresh.png')

        self.bounds_button = create_tool_button(checkable=True,
                                                checked=False,
                                                tooltip='Show Bounds',
                                                style_name='ToolButton',
                                                status_tip='Show bounding box of the cross-section',
                                                icon='bounding-box.png')
        self.bounds_button.clicked.connect(self.showBounds)
        self.execute_button = QtWidgets.QPushButton('Add Points', objectName='GreyTextPushButton')
        self.execute_button.clicked.connect(self.addPoints)
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.bounds_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.execute_button)
        self.main_layout.addLayout(button_layout)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.main_layout.addWidget(self.splitter)
        self.createGraphicsView()
        self.reset_button.clicked.connect(self.view.reset)
        self.createControlPanel()

        self.prepareMesh()
        self.parent_model.sample_changed.connect(self.prepareMesh)
        self.parent_model.measurement_points_changed.connect(self.updateCrossSection)
        self.initializing = True
        self.updatePenColour()
        self.parent.themes.theme_changed.connect(self.updatePenColour)

    def updatePenColour(self):
        """Changes the pen colours and updates scene"""
        self.scene.path_pen.setColor(self.parent.themes.scene_path)
        self.scene.highlight_pen.setColor(self.parent.themes.scene_highlight)
        self.scene.anchor_item.default_pen.setColor(self.parent.themes.scene_anchor)
        self.scene.bounds_item.default_pen.setColor(self.parent.themes.scene_bounds)
        self.view.grid_pen.setColor(self.parent.themes.scene_grid)
        for item in self.scene.items():
            item.update()
        self.scene.update()

    def showEvent(self, event):
        if self.initializing:
            self.view.fitInView(self.view.viewport_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.initializing = False

        super().showEvent(event)

    def closeEvent(self, event):
        self.parent.scenes.removePlane()
        self.parent.size_label.clear()
        self.parent.size_label.setToolTip('')
        self.parent.cursor_label.clear()
        event.accept()

    def prepareMesh(self):
        """Setup sample mesh and initialize UI. UI is disabled if no mesh is present"""
        self.mesh = None
        sample = self.parent_model.sample
        if sample is not None:
            self.mesh = sample

        self.scene.clear()
        self.tabs.setEnabled(self.mesh is not None)
        if self.mesh is not None:
            self.setPlane(self.plane_combobox.currentText())
        else:
            self.old_distance = None
            self.parent.scenes.removePlane()
        self.view.reset()

    def updateCursorStatus(self, point):
        """Updates the status bar with the position of the mouse cursor in 3D world coordinates

        :param point: mouse cursor position in widget coordinates
        :type point: QtCore.QPoint
        """
        if self.old_distance is not None and self.view.rect().contains(point):
            transform = self.scene.transform.inverted()[0]
            scene_pt = transform.map(self.view.mapToScene(point)) / self.sample_scale
            world_pt = [scene_pt.x(), scene_pt.y(), -self.old_distance] @ self.matrix.transpose()
            cursor_text = (f'<p style="white-space: pre;"><b>Cursor Position:</b>  ({world_pt[0]:.3f},  '
                           f'{world_pt[1]:.3f},  {world_pt[2]:.3f})  </p>')
            self.parent.cursor_label.setText(cursor_text)
        else:
            self.parent.cursor_label.clear()

    def updateDimensionStatus(self):
        """Updates the status bar with the dimension of bounding box of the sample cross-section. The dimension
        include centre, width, and height of the bounding box.
        """
        if self.cross_section_rect.isValid():
            x, y = self.cross_section_rect.center().x(), self.cross_section_rect.center().y()
            w, h = self.cross_section_rect.width(), self.cross_section_rect.height()
            size_text = (f'<p style="white-space: pre;"><span><b>Center:</b>  ({x:.3f},  {y:.3f})    </span>'
                         f'<span><b>W:</b>  {w:.3f}    </span> <span><b>H:</b>  {h:.3f}    </span></p>')
            self.parent.size_label.setText(size_text)
            self.parent.size_label.setToolTip('Dimension of bounding box (center, width and height)')
        else:
            self.parent.size_label.clear()
            self.parent.size_label.setToolTip('')

    def createGraphicsView(self):
        """Creates the graphics view and scene"""
        self.scene = GraphicsScene(self.sample_scale, self)
        self.view = GraphicsView(self.scene)
        self.view.interaction.mouse_moved.connect(self.updateCursorStatus)
        self.view.setMinimumHeight(350)
        self.splitter.addWidget(self.view)

    def createControlPanel(self):
        """Creates the control panel widgets"""
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumHeight(250)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.South)
        self.splitter.addWidget(self.tabs)

        self.createPlaneTab()
        self.createSelectionToolsTab()
        self.createGridOptionsTab()
        point_manager = PointManager(PointType.Measurement, self.parent)
        point_manager.rows_highlighted.connect(self.highlightPoints)
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
        validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        validator.setDecimals(3)
        self.plane_lineedit.setValidator(validator)
        self.plane_lineedit.textEdited.connect(self.updateSlider)
        self.plane_lineedit.editingFinished.connect(self.movePlane)
        slider_layout.addStretch(1)
        slider_layout.addWidget(self.plane_lineedit)
        layout.addLayout(slider_layout)
        self.plane_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.plane_slider.setMinimum(self.slider_range[0])
        self.plane_slider.setMaximum(self.slider_range[1])
        self.plane_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plane_slider.setSingleStep(1)
        self.plane_slider.sliderMoved.connect(self.updateLineEdit)
        self.plane_slider.sliderReleased.connect(self.movePlane)
        layout.addWidget(self.plane_slider)

        layout.addSpacing(20)
        ref_plane_layout = QtWidgets.QHBoxLayout()
        ref_plane_layout.addWidget(QtWidgets.QLabel('Reference Planes: '))
        self.add_ref_plane = QtWidgets.QPushButton('Add Plane')
        self.add_ref_plane.clicked.connect(self.addReferencePlane)
        self.clear_ref_plane = QtWidgets.QPushButton('Remove Planes')
        self.clear_ref_plane.clicked.connect(self.clearReferencePlanes)
        ref_plane_layout.addStretch(1)
        ref_plane_layout.addWidget(self.add_ref_plane)
        ref_plane_layout.addWidget(self.clear_ref_plane)
        layout.addLayout(ref_plane_layout)
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
        self.button_group.idClicked.connect(self.changeSceneMode)

        self.object_selector = create_tool_button(checkable=True,
                                                  checked=True,
                                                  tooltip='Select Points',
                                                  status_tip='Select movable points from the cross-section view',
                                                  style_name='MidToolButton',
                                                  icon='select.png')
        self.point_selector = create_tool_button(checkable=True,
                                                 tooltip='Draw a Point',
                                                 status_tip='Draw a single point at the selected position',
                                                 style_name='MidToolButton',
                                                 icon='point.png')
        self.line_selector = create_tool_button(checkable=True,
                                                tooltip='Draw Points on Line',
                                                status_tip='Draw equally spaced points on the selected line',
                                                style_name='MidToolButton',
                                                icon='line-tool.png')
        self.area_selector = create_tool_button(checkable=True,
                                                tooltip='Draw Points on Area',
                                                status_tip='Draw a grid of points on the selected area',
                                                style_name='MidToolButton',
                                                icon='area-tool.png')

        self.button_group.addButton(self.object_selector, GraphicsView.DrawMode.None_.value)
        self.button_group.addButton(self.point_selector, GraphicsView.DrawMode.Point.value)
        self.button_group.addButton(self.line_selector, GraphicsView.DrawMode.Line.value)
        self.button_group.addButton(self.area_selector, GraphicsView.DrawMode.Rectangle.value)
        selector_layout.addWidget(self.object_selector)
        selector_layout.addWidget(self.point_selector)
        selector_layout.addWidget(self.line_selector)
        selector_layout.addWidget(self.area_selector)
        selector_layout.addStretch(1)

        self.createLineToolWidget()
        self.createAreaToolWidget()
        self.createPositionWidget()

        self.key_in_button = create_tool_button(checkable=True,
                                                tooltip='Key-in Points for Shape',
                                                status_tip='Key-in the Start and End Points on Shape',
                                                style_name='MidToolButton',
                                                icon='edit.png')
        self.key_in_button.setVisible(False)
        self.key_in_button.toggled.connect(self.toggleShapeEditMode)
        layout.addLayout(selector_layout)
        key_in_edit_layout = QtWidgets.QHBoxLayout()
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setContentsMargins(0, 17, 0, 0)
        key_in_edit_layout.addLayout(left_layout)
        key_in_edit_layout.addLayout(right_layout)
        layout.addLayout(key_in_edit_layout)
        left_layout.addWidget(self.line_tool_widget)
        left_layout.addWidget(self.area_tool_widget)
        left_layout.addStretch(1)
        right_layout.addWidget(self.key_in_button)
        right_layout.addStretch(1)
        layout.addWidget(self.position_widget)
        layout.addStretch(1)

        select_tab = QtWidgets.QWidget()
        select_tab.setLayout(layout)
        self.tabs.addTab(create_scroll_area(select_tab), 'Selection Tools')

    def createGridOptionsTab(self):
        """Creates the grid option widget"""
        layout = QtWidgets.QVBoxLayout()
        self.show_grid_checkbox = QtWidgets.QCheckBox('Show Grid')
        self.show_grid_checkbox.toggled.connect(self.showGrid)
        self.snap_select_to_grid_checkbox = QtWidgets.QCheckBox('Snap Selection to Grid')
        self.snap_select_to_grid_checkbox.toggled.connect(self.snapToGrid)
        self.snap_select_to_grid_checkbox.setEnabled(self.view.show_grid)
        self.snap_object_to_grid_checkbox = QtWidgets.QCheckBox('Snap Cross-Section to Grid')
        self.snap_object_to_grid_checkbox.toggled.connect(self.snapObjectToGrid)
        self.snap_object_to_grid_checkbox.setEnabled(self.view.show_grid)
        layout.addWidget(self.show_grid_checkbox)
        layout.addWidget(self.snap_select_to_grid_checkbox)
        layout.addWidget(self.snap_object_to_grid_checkbox)
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

    def createPositionWidget(self):
        """Creates inputs for typing the start and stop position of a shape"""
        self.position_widget = QtWidgets.QGroupBox('Key-In', self)
        layout = QtWidgets.QVBoxLayout()

        layout_1 = QtWidgets.QHBoxLayout()
        layout_1.addWidget(QtWidgets.QLabel('Start Position: '))
        self.start_x_spinbox = QtWidgets.QDoubleSpinBox()
        self.start_x_spinbox.setDecimals(3)
        self.start_x_spinbox.setValue(0)
        self.start_y_spinbox = QtWidgets.QDoubleSpinBox()
        self.start_y_spinbox.setDecimals(3)
        self.start_y_spinbox.setValue(0)

        layout_2 = QtWidgets.QHBoxLayout()
        layout_2.addWidget(QtWidgets.QLabel('Stop Position: '))
        self.stop_x_spinbox = QtWidgets.QDoubleSpinBox()
        self.stop_x_spinbox.setDecimals(3)
        self.stop_x_spinbox.setValue(0)
        self.stop_y_spinbox = QtWidgets.QDoubleSpinBox()
        self.stop_y_spinbox.setDecimals(3)
        self.stop_y_spinbox.setValue(0)

        self.start_x_spinbox.valueChanged.connect(self.setShapePosition)
        self.start_y_spinbox.valueChanged.connect(self.setShapePosition)
        self.stop_x_spinbox.valueChanged.connect(self.setShapePosition)
        self.stop_y_spinbox.valueChanged.connect(self.setShapePosition)

        button_layout = QtWidgets.QHBoxLayout()
        accept_button = create_tool_button(tooltip='Accept Shape',
                                           status_tip='Accepts shape and coverts to points',
                                           style_name='MidToolButton',
                                           icon='check.png')
        accept_button.clicked.connect(self.acceptShape)
        clear_button = create_tool_button(tooltip='Clear Shape',
                                          status_tip='Clears shape outline',
                                          style_name='MidToolButton',
                                          icon='cross.png')
        clear_button.clicked.connect(self.clearShape)
        button_layout.addStretch(1)
        button_layout.addWidget(accept_button)
        button_layout.addWidget(clear_button)

        stretch_factor = 3
        layout_1.addStretch(1)
        layout_1.addWidget(QtWidgets.QLabel('X: '))
        layout_1.addWidget(self.start_x_spinbox, stretch_factor)
        layout_1.addStretch(1)
        layout_1.addWidget(QtWidgets.QLabel('Y: '))
        layout_1.addWidget(self.start_y_spinbox, stretch_factor)
        layout_2.addStretch(1)
        layout_2.addWidget(QtWidgets.QLabel('X: '))
        layout_2.addWidget(self.stop_x_spinbox, stretch_factor)
        layout_2.addStretch(1)
        layout_2.addWidget(QtWidgets.QLabel('Y: '))
        layout_2.addWidget(self.stop_y_spinbox, stretch_factor)

        layout.addLayout(layout_1)
        layout.addLayout(layout_2)
        layout.addLayout(button_layout)
        self.position_widget.setVisible(False)
        self.position_widget.setLayout(layout)

    def setShapePosition(self):
        """Updates the start and stop position of the shape"""
        if self.view.draw_tool is None:
            return

        self.view.draw_tool.start_pos = self.scene.transform.map(
            QtCore.QPointF(self.start_x_spinbox.value(), self.start_y_spinbox.value()) * self.sample_scale)
        self.view.draw_tool.stop_pos = self.scene.transform.map(
            QtCore.QPointF(self.stop_x_spinbox.value(), self.stop_y_spinbox.value()) * self.sample_scale)

        self.view.draw_tool.updateOutline(True)
        self.view.draw_tool.updateOutline()

    def acceptShape(self):
        """Accepts shape and generates points"""
        if self.view.draw_tool is None:
            return

        self.setShapePosition()
        self.view.draw_tool.drawPoints()
        self.clearShape()

    def clearShape(self):
        """Clears shape and reset start and stop position inputs"""
        self.start_x_spinbox.blockSignals(True)
        self.start_x_spinbox.setRange(self.cross_section_rect.left(), self.cross_section_rect.right())
        self.start_x_spinbox.setValue(self.cross_section_rect.center().x())
        self.start_x_spinbox.blockSignals(False)

        self.start_y_spinbox.blockSignals(True)
        self.start_y_spinbox.setRange(self.cross_section_rect.top(), self.cross_section_rect.bottom())
        self.start_y_spinbox.setValue(self.cross_section_rect.center().y())
        self.start_y_spinbox.blockSignals(False)

        self.stop_x_spinbox.blockSignals(True)
        self.stop_x_spinbox.setRange(self.cross_section_rect.left(), self.cross_section_rect.right())
        self.stop_x_spinbox.setValue(self.cross_section_rect.center().x())
        self.stop_x_spinbox.blockSignals(False)

        self.stop_y_spinbox.blockSignals(True)
        self.stop_y_spinbox.setRange(self.cross_section_rect.top(), self.cross_section_rect.bottom())
        self.stop_y_spinbox.setValue(self.cross_section_rect.center().y())
        self.stop_y_spinbox.blockSignals(False)

        self.scene.updateOutlineItem(None)

    def toggleShapeEditMode(self, state):
        """Toggle the shape edit mode

        :param state: indicates if the edit mode is active
        :type state: bool
        """
        if state:
            self.view.interaction.mouse_moved.connect(self.setShapePosition)
        else:
            self.view.interaction.mouse_moved.disconnect()
            self.view.interaction.mouse_moved.connect(self.updateCursorStatus)
        self.position_widget.setVisible(state)
        self.clearShape()
        if self.view.draw_tool is not None:
            self.view.draw_tool.mouse_enabled = not state

    def createLineToolWidget(self):
        """Creates the input for number of points when using the line tool"""
        self.line_tool_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 20, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Number of Points: '))
        self.line_point_count_spinbox = QtWidgets.QSpinBox()
        self.line_point_count_spinbox.setValue(2)
        self.line_point_count_spinbox.setRange(2, 100)
        self.line_point_count_spinbox.valueChanged.connect(lambda x: self.view.setDrawToolPointCount((x, )))

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
        self.area_x_spinbox.setValue(2)
        self.area_x_spinbox.setRange(2, 100)
        self.area_y_spinbox = QtWidgets.QSpinBox()
        self.area_y_spinbox.setValue(2)
        self.area_y_spinbox.setRange(2, 100)

        stretch_factor = 3
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('X: '))
        self.area_x_spinbox.valueChanged.connect(lambda: self.view.setDrawToolPointCount(
            (self.area_x_spinbox.value(), self.area_y_spinbox.value())))
        layout.addWidget(self.area_x_spinbox, stretch_factor)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel('Y: '))
        self.area_y_spinbox.valueChanged.connect(lambda: self.view.setDrawToolPointCount(
            (self.area_x_spinbox.value(), self.area_y_spinbox.value())))
        layout.addWidget(self.area_y_spinbox, stretch_factor)
        self.area_tool_widget.setVisible(False)
        self.area_tool_widget.setLayout(layout)

    def createSnapAnchorWidget(self):
        """Creates the snap anchor selection widget"""
        self.snap_anchor_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Snap Anchor: '))
        self.snap_anchor_combobox = QtWidgets.QComboBox()
        self.snap_anchor_combobox.setView(QtWidgets.QListView())
        self.snap_anchor_combobox.addItems([anchor.value for anchor in Anchor])
        self.snap_anchor_combobox.currentTextChanged.connect(self.updateObjectAnchor)
        layout.addWidget(self.snap_anchor_combobox)
        self.snap_anchor_widget.setVisible(False)
        self.snap_anchor_widget.setLayout(layout)

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
        main_layout.addSpacing(20)

        self.createSnapAnchorWidget()
        main_layout.addWidget(self.snap_anchor_widget)
        main_layout.addStretch(1)

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
        size = ()
        mode = GraphicsView.DrawMode(button_id)
        if mode == GraphicsView.DrawMode.Line:
            size = (self.line_point_count_spinbox.value(), )
        elif mode == GraphicsView.DrawMode.Rectangle:
            size = (self.area_x_spinbox.value(), self.area_y_spinbox.value())

        self.line_tool_widget.setVisible(mode == GraphicsView.DrawMode.Line)
        self.area_tool_widget.setVisible(mode == GraphicsView.DrawMode.Rectangle)
        self.key_in_button.setVisible(mode in (GraphicsView.DrawMode.Line, GraphicsView.DrawMode.Rectangle))
        self.key_in_button.setChecked(False)
        self.view.draw_tool = self.view.createDrawTool(mode, size)

    def showBounds(self, state):
        """Shows/Hides the bounds in the scene

        :param state: indicated if the bounds should be shown
        :type state: bool
        """
        rect = QtCore.QRectF()
        for item in self.scene.items():
            if item.data(0) == self.sample_id:
                rect = item.boundingRect()
                break

        if state and rect.isValid():
            self.scene.bounds_item.rect = rect
            if self.scene.bounds_item not in self.scene.items():
                self.scene.bounds_item.setPos(QtCore.QPointF())
                self.scene.bounds_item.setTransform(self.scene.transform)
                self.scene.addItem(self.scene.bounds_item)
        else:
            if self.scene.bounds_item in self.scene.items():
                self.scene.removeItem(self.scene.bounds_item)

    def showHelp(self):
        """Toggles the help overlay in the scene"""
        self.view.show_help = not self.view.has_foreground
        self.scene.update()

    def showGrid(self, state):
        """Shows/Hides the grid in the scene

        :param state: indicates if the grid should be shown
        :type state: bool
        """
        self.view.show_grid = state
        self.grid_widget.setVisible(self.view.show_grid)
        self.snap_select_to_grid_checkbox.setEnabled(self.view.show_grid)
        self.snap_object_to_grid_checkbox.setEnabled(self.view.show_grid)
        self.snapToGrid(self.snap_select_to_grid_checkbox.isChecked())
        self.snapObjectToGrid(self.snap_object_to_grid_checkbox.isChecked())
        self.scene.update()

    def snapToGrid(self, state):
        """Enables/Disables snap point to grid

        :param state: indicates if snap point to grid is enabled
        :type state: bool
        """
        self.view.snap_to_grid = self.view.show_grid and state

    def snapObjectToGrid(self, state):
        """Enables/Disables snap object to grid

        :param state: indicates if snap object to grid is enabled
        :type state: bool
        """
        self.view.snap_object_to_grid = self.view.show_grid and state
        self.snap_anchor_widget.setVisible(self.view.snap_object_to_grid)
        self.updateObjectAnchor(self.snap_anchor_combobox.currentText())

    def updateObjectAnchor(self, value):
        """Sets the anchor point for snap to grid

        :param value: anchor name
        :type value: str
        """
        box = QtCore.QRectF()
        for item in self.scene.items():
            if item.data(0) == self.sample_id:
                box = item.boundingRect()
                break
        if value == Anchor.Center.value:
            anchor = box.center()
        elif value == Anchor.TopLeft.value:
            anchor = box.topLeft()
        elif value == Anchor.TopRight.value:
            anchor = box.topRight()
        elif value == Anchor.BottomLeft.value:
            anchor = box.bottomLeft()
        else:
            anchor = box.bottomRight()

        self.view.object_anchor = anchor

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
        self.plane = self.plane.moveToDistance(distance)
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
                self.x_axis.validation_label.setText('')
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
        plane_size = 2 * self.mesh.bounding_box.radius

        self.matrix = view_from_plane(self.plane.normal)
        extent = [[*self.mesh.bounding_box.min], [*self.mesh.bounding_box.max]] @ self.matrix
        plane_offset = (abs(extent[0][2] - extent[1][2]) / 2) + 0.01

        self.parent.scenes.drawPlane(self.plane, (plane_size, plane_size))
        distance = self.plane.distanceFromOrigin()
        self.plane_offset_range = (distance - plane_offset, distance + plane_offset)
        slider_value = int(map_range(*self.plane_offset_range, *self.slider_range, distance))
        self.plane_slider.setValue(slider_value)
        self.plane_lineedit.setText(f'{distance:.3f}')
        self.old_distance = distance
        self.view.resetTransform()
        self.updateCrossSection()

    def addReferencePlane(self):
        """Add a fixed reference plane at the location of the main cross-sectional plane"""
        self.reference_planes.append(self.plane)
        plane_size = 2 * self.mesh.bounding_box.radius
        self.parent.scenes.drawPlane(self.plane, (plane_size, plane_size), True)

    def clearReferencePlanes(self):
        """Removes all fixed reference planes"""
        self.reference_planes.clear()
        self.parent.scenes.removePlane(True)
        self.updateCrossSection()

    def updateCrossSection(self):
        """Creates the mesh cross-section and displays the cross-section and points in the scene"""
        self.scene.clear()
        if self.mesh is None:
            return

        if isinstance(self.mesh, Mesh):
            segments = mesh_plane_intersection(self.mesh, self.plane)
            if len(segments) == 0:
                return
            segments = np.array(segments)

            cross_section_item = QtWidgets.QGraphicsPathItem()
            cross_section_path = QtGui.QPainterPath()
            rotated_segments = self.sample_scale * (segments @ self.matrix)
            for i in range(0, rotated_segments.shape[0], 2):
                start = rotated_segments[i, :]
                cross_section_path.moveTo(start[0], start[1])
                end = rotated_segments[i + 1, :]
                cross_section_path.lineTo(end[0], end[1])
            cross_section_item.setPath(cross_section_path)
            cross_section_item.setPen(self.path_pen)
            cross_section_item.setTransform(self.scene.transform)
        else:
            volume_slice = volume_plane_intersection(self.mesh, self.plane)
            if volume_slice is None:
                return

            rect = QtCore.QRectF(*volume_slice.rect)
            transform = QtGui.QTransform.fromScale(self.sample_scale, self.sample_scale)
            rect = transform.mapRect(rect)
            cross_section_item = GraphicsImageItem(rect, volume_slice.image)
            cross_section_item.setTransform(self.scene.transform)
        cross_section_item.setData(0, self.sample_id)
        self.scene.addItem(cross_section_item)

        rect = cross_section_item.boundingRect()
        anchor = rect.center()
        reference_lines_item = QtWidgets.QGraphicsPathItem()
        reference_lines_path = QtGui.QPainterPath()
        min_bound = np.array([rect.bottomLeft().x(), rect.bottomLeft().y(), 0])
        max_bound = np.array([rect.topRight().x(), rect.topRight().y(), 0])
        for reference_plane in self.reference_planes:
            line = self.plane.intersectPlane(reference_plane)
            if line is None:
                continue

            line_point = self.sample_scale * (line.point @ self.matrix)
            line_point[2] = 0
            line = Line(line.axis @ self.matrix, line_point)
            value = line_box_intersection(line, min_bound, max_bound)
            if value is None:
                continue

            reference_lines_path.moveTo(value[0][0], value[0][1])
            reference_lines_path.lineTo(value[1][0], value[1][1])
        reference_lines_item.setPath(reference_lines_path)
        reference_lines_item.setPen(self.ref_path_pen)
        reference_lines_item.setTransform(self.scene.transform)
        self.scene.addItem(reference_lines_item)

        ab = self.plane.point - self.parent_model.measurement_points.points
        d = np.einsum('ij,ij->i', np.expand_dims(self.plane.normal, axis=0), ab)
        index = np.where(np.abs(d) < POS_EPS)[0]
        rotated_points = self.parent_model.measurement_points.points[index, :]
        rotated_points = rotated_points @ self.matrix

        for i, p in zip(index, rotated_points):
            point = QtCore.QPointF(p[0], p[1]) * self.sample_scale
            point = self.scene.transform.map(point)
            point_item = GraphicsPointItem(point, size=self.scene.point_size)
            point_item.rank = i
            point_item.setToolTip(f'Point {i + 1}')
            point_item.fixed = True
            point_item.makeControllable(self.view.draw_tool is None)
            point_item.default_pen = self.point_pen
            self.scene.addItem(point_item)
            rect = rect.united(point_item.boundingRect().translated(point))

        # calculate new rectangle that encloses original rect with a different anchor
        rect.united(rect.translated(anchor - rect.center()))
        self.view.setSceneRect(rect)
        self.view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.view.viewport_rect = rect

        self.showGrid(self.show_grid_checkbox.isChecked())
        self.updateObjectAnchor(self.snap_anchor_combobox.currentText())

        scale = QtGui.QTransform.fromScale(1 / self.sample_scale, 1 / self.sample_scale)
        self.cross_section_rect = scale.mapRect(cross_section_item.boundingRect())
        self.updateDimensionStatus()
        self.showBounds(self.bounds_button.isChecked())
        self.clearShape()

    def highlightPoints(self, highlighted_rows):
        """Highlights the points corresponding to the rows currently selected in the point manager
        
        :param highlighted_rows: the currently highlighted rows
        :type highlighted_rows: List[bool]
        """
        items = self.scene.items()
        fixed_points = {item.rank: item for item in items if isinstance(item, GraphicsPointItem) and item.fixed}

        for i, is_highlighted in enumerate(highlighted_rows):
            if fixed_points.get(i):
                fixed_points[i].highlighted = is_highlighted
        self.scene.update()

    def addPoints(self):
        """Adds the points in the scene into the measurement points of the  project"""
        if len(self.scene.items()) < 2:
            return

        points_2d = []
        transform = self.scene.transform.inverted()[0]
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
        self.execute_button = QtWidgets.QPushButton('Align Sample', objectName='GreyTextPushButton')
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


class VolumeLoader(QtWidgets.QDialog):
    """Creates a UI for loading a volume from a stack of TIFF files into memory

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.main_layout = QtWidgets.QVBoxLayout()
        unit = 'mm'

        self.filepath_layout = FormGroup(FormGroup.Layout.Vertical)
        self.pixel_size_group = FormGroup(FormGroup.Layout.Horizontal)
        self.pixel_centre_group = FormGroup(FormGroup.Layout.Horizontal)

        self.filepath_picker = FilePicker(path='', select_folder=True)
        self.filepath_picker.value_changed.connect(self.formValidation)

        pixel_size_layout = QtWidgets.QVBoxLayout()
        pixel_size_layout.addWidget(QtWidgets.QLabel('Size of voxel: '))
        self.x_pixel_box = FormControl('X', 1.0, required=True, desc=unit, number=True, decimals=3)
        self.y_pixel_box = FormControl('Y', 1.0, required=True, desc=unit, number=True, decimals=3)
        self.z_pixel_box = FormControl('Z', 1.0, required=True, desc=unit, number=True, decimals=3)
        self.x_pixel_box.range(minimum=0.001, maximum=1000)
        self.y_pixel_box.range(minimum=0.001, maximum=1000)
        self.z_pixel_box.range(minimum=0.001, maximum=1000)
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
        self.execute_button = QtWidgets.QPushButton('Load Volume', objectName='WhiteTextPushButton')
        self.execute_button.setDisabled(True)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        cancel_button = QtWidgets.QPushButton('Cancel', objectName='BlueTextPushButton')
        cancel_button.clicked.connect(self.reject)
        execute_button_layout.addStretch(1)
        execute_button_layout.addWidget(self.execute_button)
        execute_button_layout.addWidget(cancel_button)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(FormTitle('Select Folder with TIFFs: '))
        self.main_layout.addWidget(QtWidgets.QLabel('Image directory: '))
        self.main_layout.addWidget(self.filepath_picker)
        self.main_layout.addSpacing(15)
        self.main_layout.addWidget(QtWidgets.QLabel('Size of voxel: '))
        self.main_layout.addWidget(self.pixel_size_group)
        self.main_layout.addSpacing(5)
        self.main_layout.addWidget(QtWidgets.QLabel('Centre of volume: '))
        self.main_layout.addWidget(self.pixel_centre_group)
        self.main_layout.addStretch(1)
        self.main_layout.addLayout(execute_button_layout)

        self.setLayout(self.main_layout)
        self.setMinimumSize(560, 420)
        self.setWindowTitle('Load Volume')
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    def formValidation(self):
        if self.pixel_centre_group.valid and self.pixel_size_group.valid and self.filepath_picker.value:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        filepath = self.filepath_picker.value
        voxel_size = [self.x_pixel_box.value, self.y_pixel_box.value, self.z_pixel_box.value]
        centre = [self.x_centre_box.value, self.y_centre_box.value, self.z_centre_box.value]
        self.parent.presenter.importVolume(filepath, voxel_size, centre)
        self.accept()
