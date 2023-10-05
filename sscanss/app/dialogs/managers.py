import math
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from sscanss.app.widgets import PointModel, LimitTextDelegate
from sscanss.core.geometry.mesh import Mesh
from sscanss.core.geometry.volume import Volume
from sscanss.core.math import is_close, POS_EPS
from sscanss.core.instrument import Link
from sscanss.core.util import (DockFlag, PointType, CommandID, Attributes, create_tool_button, create_scroll_area,
                               create_icon, FormControl, FormGroup, FormTitle, IconEngine)
from sscanss.config import settings


class PointManager(QtWidgets.QWidget):
    """Creates a UI for viewing, deleting, reordering, and editing measurement/fiducial points

    :param point_type: point type
    :type point_type: PointType
    :param parent: main window instance
    :type parent: MainWindow
    """
    rows_highlighted = QtCore.pyqtSignal(list)
    dock_flag = DockFlag.Bottom

    def __init__(self, point_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()
        self.point_type = point_type
        self.attribute = Attributes.Fiducials if point_type == PointType.Fiducial else Attributes.Measurements

        self.selected = None

        layout = QtWidgets.QHBoxLayout()
        self.table_view = QtWidgets.QTableView()
        self.table_model = PointModel(self.points.copy())
        self.table_model.edit_completed.connect(self.editPoints)
        self.table_view.setItemDelegate(LimitTextDelegate())
        self.table_view.horizontalHeader().sectionClicked.connect(self.table_model.toggleCheckState)
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setMinimumHeight(300)
        self.table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_view.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table_view.horizontalHeader().setMinimumSectionSize(40)
        self.table_view.horizontalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.table_view)

        button_layout = QtWidgets.QVBoxLayout()
        self.delete_button = create_tool_button(icon='cross.png',
                                                style_name='ToolButton',
                                                tooltip='Delete Points',
                                                status_tip='Remove selected points from project')
        self.delete_button.clicked.connect(self.deletePoints)
        button_layout.addWidget(self.delete_button)

        self.move_up_button = create_tool_button(icon='arrow-up.png',
                                                 style_name='ToolButton',
                                                 tooltip='Move Point Index Up',
                                                 status_tip='Swap point index with previous point')
        self.move_up_button.clicked.connect(lambda: self.movePoint(-1))
        button_layout.addWidget(self.move_up_button)

        self.move_down_button = create_tool_button(icon='arrow-down.png',
                                                   style_name='ToolButton',
                                                   tooltip='Move Point Index Down',
                                                   status_tip='Swap point index with next point')
        self.move_down_button.clicked.connect(lambda: self.movePoint(1))
        button_layout.addWidget(self.move_down_button)

        layout.addSpacing(10)
        layout.addLayout(button_layout)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = f'{self.point_type.value} Points'
        self.setMinimumWidth(450)
        self.table_view.selectionModel().selectionChanged.connect(self.onMultiSelection)
        if self.point_type == PointType.Fiducial:
            self.parent_model.fiducials_changed.connect(self.updateTable)
        elif self.point_type == PointType.Measurement:
            self.parent_model.measurement_points_changed.connect(self.updateTable)

    @property
    def points(self):
        """Gets the measurement or fiducial point depending on widget's point type"""
        if self.point_type == PointType.Fiducial:
            return self.parent_model.fiducials
        else:
            return self.parent_model.measurement_points

    def updateTable(self):
        """Updates the table view model"""
        self.table_model.update(self.points.copy())
        self.table_view.update()
        if self.selected is not None:
            self.table_view.setCurrentIndex(self.selected)

    def deletePoints(self):
        """Deletes the points if rows are selected in the table widget"""
        selection_model = self.table_view.selectionModel()
        indices = [item.row() for item in selection_model.selectedRows()]
        if indices:
            self.parent.presenter.deletePoints(indices, self.point_type)
            self.selected = None

    def movePoint(self, offset):
        """Moves the point indices by offset if rows are selected in the table widget

        :param offset: offset to shift point index by
        :type offset: int
        """
        selection_model = self.table_view.selectionModel()
        index = [item.row() for item in selection_model.selectedRows()]

        if not index:
            return

        index_from = index[0]
        index_to = index_from + offset

        if 0 <= index_to < self.table_model.rowCount():
            self.selected = self.table_model.index(index_to, 0)
            self.parent.presenter.movePoints(index_from, index_to, self.point_type)

    def editPoints(self, new_values):
        """Edits the points in table widget

        :param new_values: new values
        :type new_values: numpy.ndarray
        """
        self.selected = self.table_view.currentIndex()
        self.table_view.selectionModel().reset()
        self.parent.presenter.editPoints(new_values, self.point_type)

    def onMultiSelection(self):
        """Handles when multiple rows are selected in the table widget """
        sm = self.table_view.selectionModel()
        selections = [sm.isRowSelected(i, QtCore.QModelIndex()) for i in range(self.table_model.rowCount())]
        self.parent.scenes.changeSelected(self.attribute, selections)
        if len(sm.selectedRows()) > 1:
            self.move_down_button.setDisabled(True)
            self.move_up_button.setDisabled(True)
        else:
            self.move_down_button.setEnabled(True)
            self.move_up_button.setEnabled(True)
        self.rows_highlighted.emit(selections)

    def closeEvent(self, event):
        selections = [False] * self.table_model.rowCount()
        self.parent.scenes.changeSelected(self.attribute, selections)
        event.accept()


class VectorManager(QtWidgets.QWidget):
    """Creates a UI for viewing, and deleting measurement vectors or alignments

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Bottom

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()

        self.main_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Alignment:'))
        self.alignment_combobox = QtWidgets.QComboBox()
        self.alignment_combobox.setView(QtWidgets.QListView())
        self.alignment_combobox.activated.connect(self.onComboBoxActivated)
        layout.addWidget(self.alignment_combobox, 4)
        layout.addSpacing(10)

        self.detector_combobox = QtWidgets.QComboBox()
        self.detector_combobox.setView(QtWidgets.QListView())
        self.detector_combobox.addItems(list(self.parent_model.instrument.detectors.keys()))
        self.detector_combobox.activated.connect(self.onComboBoxActivated)
        if len(self.parent_model.instrument.detectors) > 1:
            layout.addWidget(QtWidgets.QLabel('Detector:'))
            layout.addWidget(self.detector_combobox, 4)
            size = self.detector_combobox.iconSize()
            self.detector_combobox.setItemIcon(0, create_icon(settings.value(settings.Key.Vector_1_Colour), size))
            self.detector_combobox.setItemIcon(1, create_icon(settings.value(settings.Key.Vector_2_Colour), size))

        self.delete_vector_action = QtGui.QAction("Delete Vectors", self)
        self.delete_vector_action.triggered.connect(self.deleteVectors)
        self.delete_vector_action.setStatusTip('Remove (zero) selected vectors or all vectors in current alignment')

        self.delete_alignment_action = QtGui.QAction("Delete Alignment", self)
        self.delete_alignment_action.triggered.connect(self.deleteAlignment)
        self.delete_alignment_action.setStatusTip('Remove selected vector alignment')
        delete_menu = QtWidgets.QMenu(self)
        delete_menu.addAction(self.delete_vector_action)
        delete_menu.addAction(self.delete_alignment_action)

        self.delete_alignment_button = create_tool_button(icon='cross.png',
                                                          style_name='ToolButton',
                                                          status_tip='Remove measurement vectors or alignments')
        self.delete_alignment_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.delete_alignment_button.setMenu(delete_menu)

        layout.addStretch(1)
        layout.addWidget(self.delete_alignment_button)
        layout.setAlignment(self.delete_alignment_button, QtCore.Qt.AlignmentFlag.AlignBottom)
        self.main_layout.addLayout(layout, 0)

        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['X', 'Y', 'Z'])
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(300)
        self.table.setMaximumHeight(350)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(40)
        self.table.horizontalHeader().setDefaultSectionSize(40)
        self.updateWidget()
        self.main_layout.addWidget(self.table, 20)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = 'Measurement Vectors'
        self.setMinimumWidth(450)
        self.parent_model.measurement_vectors_changed.connect(self.updateWidget)
        self.parent.scenes.rendered_alignment_changed.connect(self.updateWidget)

    def updateWidget(self):
        """Updates the table, and displays current detector and alignment"""
        detector = self.detector_combobox.currentIndex()
        alignment = self.parent.scenes.rendered_alignment
        alignment = self.updateComboBoxes(alignment)
        self.updateTable(detector, alignment)

    def updateTable(self, detector, alignment):
        """Updates the table to show measurement vectors for specified detector and
        alignment

        :param detector: index of detector
        :type detector: int
        :param alignment: index of alignment
        :type alignment: int
        """
        detector_index = slice(detector * 3, detector * 3 + 3)
        vectors = self.parent_model.measurement_vectors[:, detector_index, alignment]
        self.table.setRowCount(vectors.shape[0])

        for row, vector in enumerate(vectors):
            x = QtWidgets.QTableWidgetItem(f'{vector[0]:.7f}')
            x.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            y = QtWidgets.QTableWidgetItem(f'{vector[1]:.7f}')
            y.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            z = QtWidgets.QTableWidgetItem(f'{vector[2]:.7f}')
            z.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            self.table.setItem(row, 0, x)
            self.table.setItem(row, 1, y)
            self.table.setItem(row, 2, z)

    def onComboBoxActivated(self):
        """Handles detector or alignment selection from the drop down"""
        detector = self.detector_combobox.currentIndex()
        alignment = self.alignment_combobox.currentIndex()
        self.parent.scenes.changeRenderedAlignment(alignment)
        self.updateTable(detector, alignment)

    def updateComboBoxes(self, alignment):
        """Updates the widget to reflect changes to number of measurement vector alignments

        :param alignment: index of alignment
        :type alignment: int
        """
        align_count = self.parent_model.measurement_vectors.shape[2]
        self.delete_alignment_action.setEnabled(align_count > 1)

        alignment_list = [f'{i + 1}' for i in range(align_count)]
        self.alignment_combobox.clear()
        self.alignment_combobox.addItems(alignment_list)
        current_alignment = alignment if 0 < alignment < len(alignment_list) else 0
        self.alignment_combobox.setCurrentIndex(current_alignment)

        return current_alignment

    def deleteAlignment(self):
        """Deletes the selected measurement vector alignment"""
        self.parent.presenter.removeVectorAlignment(self.alignment_combobox.currentIndex())

    def deleteVectors(self):
        """Sets the measurement vector to zeros if rows are selected in the table widget"""
        selection_model = self.table.selectionModel()
        indices = [item.row() for item in selection_model.selectedRows()]

        if not indices:
            indices = list(range(self.table.rowCount()))

        detector_index = self.detector_combobox.currentIndex()
        alignment_index = self.alignment_combobox.currentIndex()

        self.parent.presenter.removeVectors(indices, detector_index, alignment_index)

    def closeEvent(self, event):
        self.parent.scenes.changeRenderedAlignment(0)
        event.accept()


class JawControl(QtWidgets.QWidget):
    """Creates a UI for controlling the jaw's positioner and aperture size

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Full

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToInstrumentScene()

        self.instrument = self.parent_model.instrument
        self.main_layout = QtWidgets.QVBoxLayout()

        if self.instrument.jaws.positioner is not None:
            self.createPositionerForm()
            self.main_layout.addSpacing(40)
        self.createApertureForm()
        self.main_layout.addStretch(1)

        widget = QtWidgets.QWidget()
        widget.setLayout(self.main_layout)
        scroll_area = create_scroll_area(widget)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.title = f'Configure {self.instrument.jaws.name}'
        self.setMinimumWidth(450)
        self.parent_model.instrument_controlled.connect(self.updateForms)
        self.parent.scenes.changeVisibility(Attributes.Beam, True)
        self.formValidation()

    def updateForms(self, command_id):
        """Updates the form inputs in response to a command

        :param command_id: ID of command
        :type command_id: CommandID
        """
        if command_id == CommandID.ChangeJawAperture:
            self.aperture_form_group.form_controls[0].value = self.instrument.jaws.aperture[0]
            self.aperture_form_group.form_controls[1].value = self.instrument.jaws.aperture[1]
        elif command_id == CommandID.MovePositioner:
            positioner = self.instrument.jaws.positioner
            set_points = positioner.toUserFormat(positioner.set_points)
            for value, control in zip(set_points, self.position_form_group.form_controls):
                control.value = value
        elif command_id == CommandID.IgnoreJointLimits:
            order = self.instrument.jaws.positioner.order
            for index, control in zip(order, self.position_form_group.form_controls):
                link = self.instrument.jaws.positioner.links[index]
                toggle_button = control.extra[0]
                toggle_button.setChecked(link.ignore_limits)

                if link.ignore_limits:
                    control.range(None, None)
                else:
                    if link.type == Link.Type.Revolute:
                        lower_limit = math.degrees(link.lower_limit)
                        upper_limit = math.degrees(link.upper_limit)
                    else:
                        lower_limit = link.lower_limit
                        upper_limit = link.upper_limit

                    control.range(lower_limit, upper_limit)

    def createPositionerForm(self):
        """Creates the form inputs for the jaw positioner"""
        title = FormTitle(f'{self.instrument.jaws.name} Position')
        self.main_layout.addWidget(title)
        self.position_form_group = FormGroup(FormGroup.Layout.Grid)

        for index in self.instrument.jaws.positioner.order:
            link = self.instrument.jaws.positioner.links[index]
            if link.type == Link.Type.Revolute:
                unit = 'degrees'
                offset = math.degrees(link.set_point)
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                unit = 'mm'
                offset = link.set_point
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            control = FormControl(link.name.title(),
                                  offset,
                                  desc=unit,
                                  required=True,
                                  number=True,
                                  tooltip=link.description)
            control.form_lineedit.setDisabled(link.locked)
            if not link.ignore_limits:
                control.range(lower_limit, upper_limit)

            limits_button = create_tool_button(tooltip='Disable Joint Limits',
                                               style_name='MidToolButton',
                                               icon='limit.png',
                                               checkable=True,
                                               status_tip=f'Ignore joint limits of {link.name}',
                                               checked=link.ignore_limits)
            limits_button.clicked.connect(self.ignoreJointLimits)
            limits_button.setProperty('link_index', index)

            control.extra = [limits_button]
            self.position_form_group.addControl(control)
            self.position_form_group.group_validation.connect(self.formValidation)

        self.main_layout.addWidget(self.position_form_group)
        button_layout = QtWidgets.QHBoxLayout()
        self.move_jaws_button = QtWidgets.QPushButton('Move Jaws', objectName='GreyTextPushButton')
        self.move_jaws_button.clicked.connect(self.moveJawsButtonClicked)
        button_layout.addWidget(self.move_jaws_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def createApertureForm(self):
        """Creates the form inputs for changing the aperture size"""
        title = FormTitle(f'{self.instrument.jaws.name} Aperture Size')
        self.main_layout.addWidget(title)
        aperture = self.instrument.jaws.aperture
        upper_limit = self.instrument.jaws.aperture_upper_limit
        lower_limit = self.instrument.jaws.aperture_lower_limit
        self.aperture_form_group = FormGroup(FormGroup.Layout.Grid)
        control = FormControl('Horizontal Aperture Size', aperture[0], desc='mm', required=True, number=True)
        control.range(lower_limit[0], upper_limit[0])
        self.aperture_form_group.addControl(control)
        control = FormControl('Vertical Aperture Size', aperture[1], desc='mm', required=True, number=True)
        control.range(lower_limit[1], upper_limit[1])
        self.aperture_form_group.addControl(control)
        self.main_layout.addWidget(self.aperture_form_group)
        self.aperture_form_group.group_validation.connect(self.formValidation)
        self.main_layout.addSpacing(10)

        button_layout = QtWidgets.QHBoxLayout()
        self.change_aperture_button = QtWidgets.QPushButton('Change Aperture Size', objectName='GreyTextPushButton')
        self.change_aperture_button.clicked.connect(self.changeApertureButtonClicked)
        button_layout.addWidget(self.change_aperture_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def ignoreJointLimits(self, check_state):
        """Sets the ignore joint limit state for joints in the jaw positioner

        :param check_state: indicates joint limit should be ignored
        :type check_state: bool
        """
        index = self.sender().property('link_index')
        name = self.instrument.jaws.positioner.name
        self.parent.presenter.ignorePositionerJointLimits(name, index, check_state)

    def formValidation(self):
        if self.position_form_group.valid:
            self.move_jaws_button.setEnabled(True)
        else:
            self.move_jaws_button.setDisabled(True)

        if self.aperture_form_group.valid:
            self.change_aperture_button.setEnabled(True)
        else:
            self.change_aperture_button.setDisabled(True)

    def moveJawsButtonClicked(self):
        q = [control.value for control in self.position_form_group.form_controls]
        q = self.instrument.jaws.positioner.fromUserFormat(q)

        if not is_close(q, self.instrument.jaws.positioner.set_points, POS_EPS):
            name = self.instrument.jaws.positioner.name
            self.parent.presenter.movePositioner(name, q)

    def changeApertureButtonClicked(self):
        aperture = [self.aperture_form_group.form_controls[0].value, self.aperture_form_group.form_controls[1].value]

        if aperture != self.instrument.jaws.aperture:
            self.parent.presenter.changeJawAperture(aperture)

    def closeEvent(self, event):
        self.parent.scenes.changeVisibility(Attributes.Beam, False)
        event.accept()


class PositionerControl(QtWidgets.QWidget):
    """Creates a UI for selecting positioning stack, setting joint
    positions, locking joints and removing joint limits

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Full

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToInstrumentScene()

        self.instrument = self.parent_model.instrument
        self.main_layout = QtWidgets.QVBoxLayout()

        if len(self.instrument.positioning_stacks) > 1:
            stack_layout = QtWidgets.QHBoxLayout()
            stack_layout.addStretch(1)
            stack_layout.addWidget(QtWidgets.QLabel('Positioning Stack:'))
            self.stack_combobox = QtWidgets.QComboBox()
            self.stack_combobox.setView(QtWidgets.QListView())
            self.stack_combobox.addItems(self.instrument.positioning_stacks.keys())
            self.stack_combobox.setCurrentText(self.instrument.positioning_stack.name)
            self.stack_combobox.textActivated.connect(self.changeStack)
            stack_layout.addWidget(self.stack_combobox)
            self.main_layout.addLayout(stack_layout)
            self.main_layout.addSpacing(5)

        self.positioner_forms_layout = QtWidgets.QVBoxLayout()
        self.positioner_forms_layout.setContentsMargins(0, 0, 0, 0)

        self.createForms()
        self.main_layout.addLayout(self.positioner_forms_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.move_joints_button = QtWidgets.QPushButton('Move Joints', objectName='GreyTextPushButton')
        self.move_joints_button.clicked.connect(self.moveJointsButtonClicked)
        button_layout.addWidget(self.move_joints_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)

        widget = QtWidgets.QWidget()
        widget.setLayout(self.main_layout)
        scroll_area = create_scroll_area(widget)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.title = 'Configure Positioning System'
        self.setMinimumWidth(450)
        self.parent_model.instrument_controlled.connect(self.updateForms)
        self.formValidation()

    def updateForms(self, command_id):
        """Updates the form inputs in response to a command

        :param command_id: ID of command
        :type command_id: CommandID
        """
        if command_id == CommandID.ChangePositioningStack:
            self.stack_combobox.setCurrentText(self.instrument.positioning_stack.name)
            self.createForms()
            self.formValidation()
        elif command_id == CommandID.ChangePositionerBase:
            for aux in self.instrument.positioning_stack.auxiliary:
                button = self.base_reset_buttons[aux.name]
                button.setVisible(aux.base is not aux.default_base)
        elif command_id == CommandID.MovePositioner:
            positioner = self.instrument.positioning_stack
            set_points = positioner.toUserFormat(positioner.set_points)
            for value, control in zip(set_points, self.positioner_form_controls):
                control.value = value
        elif command_id == CommandID.LockJoint:
            order = self.instrument.positioning_stack.order
            for index, control in zip(order, self.positioner_form_controls):
                link = self.instrument.positioning_stack.links[index]
                control.form_lineedit.setDisabled(link.locked)
                toggle_button = control.extra[0]
                toggle_button.setChecked(link.locked)
                if link.locked:
                    control.value = link.set_point if link.type == Link.Type.Prismatic else math.degrees(link.set_point)
        elif command_id == CommandID.IgnoreJointLimits:
            order = self.instrument.positioning_stack.order
            for index, control in zip(order, self.positioner_form_controls):
                link = self.instrument.positioning_stack.links[index]
                toggle_button = control.extra[1]
                toggle_button.setChecked(link.ignore_limits)

                if link.ignore_limits:
                    control.range(None, None)
                else:
                    if link.type == Link.Type.Revolute:
                        lower_limit = math.degrees(link.lower_limit)
                        upper_limit = math.degrees(link.upper_limit)
                    else:
                        lower_limit = link.lower_limit
                        upper_limit = link.upper_limit

                    control.range(lower_limit, upper_limit)

    def changeStack(self, stack_name):
        """Changes the positioning stack to specified

        :param stack_name: name of stack
        :type stack_name: str
        """
        if stack_name != self.instrument.positioning_stack.name:
            self.parent.presenter.changePositioningStack(stack_name)

    def createForms(self):
        """Creates the form inputs for each positioner in stack"""
        for _ in range(self.positioner_forms_layout.count()):
            widget = self.positioner_forms_layout.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()

        self.positioner_forms = []
        self.positioner_form_controls = []
        self.base_reset_buttons = {}

        positioner = self.instrument.positioning_stack.fixed
        widget = self.createPositionerWidget(positioner)
        self.positioner_forms_layout.addWidget(widget)

        for aux in self.instrument.positioning_stack.auxiliary:
            widget = self.createPositionerWidget(aux, True)
            self.positioner_forms_layout.addWidget(widget)

    def createPositionerWidget(self, positioner, add_base_button=False):
        """Creates the form widget for the given positioner

        :param positioner: serial manipulator
        :type positioner: SerialManipulator
        :param add_base_button: indicates base button should be shown
        :type add_base_button: bool
        :return: positioner widget
        :rtype: QtWidgets.QWidget
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        title = FormTitle(positioner.name)
        layout.addWidget(title)
        if add_base_button:
            base_button = create_tool_button(tooltip='Change Base Matrix', style_name='MidToolButton')

            import_base_action = QtGui.QAction('Import Base Matrix', self)
            import_base_action.setStatusTip(f'Import base transformation matrix for {positioner.name}')
            import_base_action.triggered.connect(lambda ignore, n=positioner.name: self.importPositionerBase(n))

            compute_base_action = QtGui.QAction('Calculate Base Matrix', self)
            compute_base_action.setStatusTip(f'Calculate base transformation matrix for {positioner.name}')
            compute_base_action.triggered.connect(lambda ignore, n=positioner.name: self.computePositionerBase(n))

            export_base_action = QtGui.QAction('Export Base Matrix', self)
            export_base_action.setStatusTip(f'Export base transformation matrix for {positioner.name}')
            export_base_action.triggered.connect(lambda ignore, n=positioner.name: self.exportPositionerBase(n))

            base_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            base_button.addActions([import_base_action, compute_base_action, export_base_action])
            base_button.setDefaultAction(import_base_action)
            base_button.setIcon(QtGui.QIcon(IconEngine('base.png')))
            title.addHeaderControl(base_button)

            reset_button = create_tool_button(tooltip='Reset Base Matrix',
                                              style_name='MidToolButton',
                                              status_tip=f'Reset base transformation matrix of {positioner.name}',
                                              icon='refresh.png',
                                              hide=True)
            reset_button.clicked.connect(lambda ignore, n=positioner.name: self.resetPositionerBase(n))
            title.addHeaderControl(reset_button)
            reset_button.setVisible(positioner.base is not positioner.default_base)
            self.base_reset_buttons[positioner.name] = reset_button

        form_group = FormGroup(FormGroup.Layout.Grid)
        order_offset = len(self.positioner_form_controls)
        for index in positioner.order:
            link = positioner.links[index]
            if link.type == Link.Type.Revolute:
                unit = 'degrees'
                offset = math.degrees(link.set_point)
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                unit = 'mm'
                offset = link.set_point
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            control = FormControl(link.name.title(),
                                  offset,
                                  desc=unit,
                                  required=True,
                                  number=True,
                                  tooltip=link.description)
            control.form_lineedit.setDisabled(link.locked)
            if not link.ignore_limits:
                control.range(lower_limit, upper_limit)

            lock_button = create_tool_button(tooltip='Lock Joint',
                                             style_name='MidToolButton',
                                             status_tip=f'Lock {link.name} joint in {positioner.name}',
                                             icon='lock.png',
                                             checkable=True,
                                             checked=link.locked)
            lock_button.clicked.connect(self.lockJoint)
            lock_button.setProperty('link_index', index + order_offset)

            limits_button = create_tool_button(tooltip='Disable Joint Limits',
                                               style_name='MidToolButton',
                                               status_tip=f'Ignore joint limits of {link.name}',
                                               icon='limit.png',
                                               checkable=True,
                                               checked=link.ignore_limits)
            limits_button.clicked.connect(self.ignoreJointLimits)
            limits_button.setProperty('link_index', index + order_offset)

            control.extra = [lock_button, limits_button]
            form_group.addControl(control)

            form_group.group_validation.connect(self.formValidation)
            self.positioner_form_controls.append(control)

        layout.addWidget(form_group)
        widget.setLayout(layout)
        self.positioner_forms.append(form_group)

        return widget

    def lockJoint(self, check_state):
        """Sets the lock joint state for a joint in the positioning stack

        :param check_state: indicates joint should be locked
        :type check_state: bool
        """
        index = self.sender().property('link_index')
        name = self.instrument.positioning_stack.name
        self.parent.presenter.lockPositionerJoint(name, index, check_state)

    def ignoreJointLimits(self, check_state):
        """Sets the ignore joint limit state for a joint in the positioning stack

        :param check_state: indicates joint limit should be ignored
        :type check_state: bool
        """
        index = self.sender().property('link_index')
        name = self.instrument.positioning_stack.name
        self.parent.presenter.ignorePositionerJointLimits(name, index, check_state)

    def formValidation(self):
        for form in self.positioner_forms:
            if not form.valid:
                self.move_joints_button.setDisabled(True)
                return
        self.move_joints_button.setEnabled(True)

    def importPositionerBase(self, name):
        """Imports and sets the base matrix for specified positioner

        :param name: name of positioner
        :type name: str
        """
        matrix = self.parent.presenter.importTransformMatrix()
        if matrix is None:
            return

        positioner = self.instrument.positioners[name]
        self.parent.presenter.changePositionerBase(positioner, matrix)

    def computePositionerBase(self, name):
        """Computes the base matrix for specified positioner

        :param name: name of positioner
        :type name: str
        """
        positioner = self.instrument.positioners[name]
        matrix = self.parent.presenter.computePositionerBase(positioner)
        if matrix is None:
            return

        self.parent.presenter.changePositionerBase(positioner, matrix)

    def exportPositionerBase(self, name):
        """Exports the base matrix for specified positioner

        :param name: name of positioner
        :type name: str
        """
        positioner = self.instrument.positioners[name]
        self.parent.presenter.exportBaseMatrix(positioner.base)

    def resetPositionerBase(self, name):
        """Resets the base matrix for specified positioner

        :param name: name of positioner
        :type name: str
        """
        positioner = self.instrument.positioners[name]
        self.parent.presenter.changePositionerBase(positioner, positioner.default_base)

    def moveJointsButtonClicked(self):
        q = [control.value for control in self.positioner_form_controls]
        q = self.instrument.positioning_stack.fromUserFormat(q)

        if not is_close(q, self.instrument.positioning_stack.set_points, POS_EPS):
            name = self.instrument.positioning_stack.name
            self.parent.presenter.movePositioner(name, q)


class DetectorControl(QtWidgets.QWidget):
    """Creates a UI for controlling a detector's positioner

    :param detector: name of the detector
    :type detector: str
    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Full

    def __init__(self, detector, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToInstrumentScene()

        self.main_layout = QtWidgets.QVBoxLayout()

        self.name = detector
        self.detector = self.parent_model.instrument.detectors[detector]
        if self.detector.positioner is not None:
            self.createPositionerForm()
            self.formValidation()
        self.main_layout.addStretch(1)

        widget = QtWidgets.QWidget()
        widget.setLayout(self.main_layout)
        scroll_area = create_scroll_area(widget)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        detector_count = len(self.parent_model.instrument.detectors)
        self.title = 'Configure Detector' if detector_count == 1 else f'Configure {detector} Detector'
        self.setMinimumWidth(450)
        self.parent_model.instrument_controlled.connect(self.updateForms)
        self.parent.scenes.changeVisibility(Attributes.Beam, True)

    def updateForms(self, command_id):
        """Updates the form inputs in response to a command

        :param command_id: ID of command
        :type command_id: CommandID
        """
        if command_id == CommandID.MovePositioner:
            positioner = self.detector.positioner
            set_points = positioner.toUserFormat(positioner.set_points)
            for value, control in zip(set_points, self.position_form_group.form_controls):
                control.value = value
        elif command_id == CommandID.IgnoreJointLimits:
            order = self.detector.positioner.order
            for index, control in zip(order, self.position_form_group.form_controls):
                link = self.detector.positioner.links[index]
                toggle_button = control.extra[0]
                toggle_button.setChecked(link.ignore_limits)

                if link.ignore_limits:
                    control.range(None, None)
                else:
                    if link.type == Link.Type.Revolute:
                        lower_limit = math.degrees(link.lower_limit)
                        upper_limit = math.degrees(link.upper_limit)
                    else:
                        lower_limit = link.lower_limit
                        upper_limit = link.upper_limit

                    control.range(lower_limit, upper_limit)

    def createPositionerForm(self):
        """Creates the form inputs for the detector positioner"""
        title = FormTitle('Detector Position')
        self.main_layout.addWidget(title)
        self.position_form_group = FormGroup(FormGroup.Layout.Grid)

        for index, link in enumerate(self.detector.positioner.links):
            if link.type == Link.Type.Revolute:
                unit = 'degrees'
                offset = math.degrees(link.set_point)
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                unit = 'mm'
                offset = link.set_point
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            pretty_label = link.name.replace('_', ' ').title()
            control = FormControl(pretty_label, offset, desc=unit, required=True, number=True, tooltip=link.description)
            control.form_lineedit.setDisabled(link.locked)
            if not link.ignore_limits:
                control.range(lower_limit, upper_limit)

            limits_button = create_tool_button(tooltip='Disable Joint Limits',
                                               style_name='MidToolButton',
                                               icon='limit.png',
                                               checkable=True,
                                               status_tip=f'Ignore joint limits of {pretty_label}',
                                               checked=link.ignore_limits)
            limits_button.clicked.connect(self.ignoreJointLimits)
            limits_button.setProperty('link_index', index)

            control.extra = [limits_button]
            self.position_form_group.addControl(control)
            self.position_form_group.group_validation.connect(self.formValidation)

        self.main_layout.addWidget(self.position_form_group)
        button_layout = QtWidgets.QHBoxLayout()
        self.move_detector_button = QtWidgets.QPushButton('Move Detector', objectName='GreyTextPushButton')
        self.move_detector_button.clicked.connect(self.moveDetectorsButtonClicked)
        button_layout.addWidget(self.move_detector_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def ignoreJointLimits(self, check_state):
        """Sets the ignore joint limit state for joints in the detector positioner

        :param check_state: indicates joint limit should be ignored
        :type check_state: bool
        """
        index = self.sender().property('link_index')
        name = self.detector.positioner.name
        self.parent.presenter.ignorePositionerJointLimits(name, index, check_state)

    def formValidation(self):
        if self.position_form_group.valid:
            self.move_detector_button.setEnabled(True)
        else:
            self.move_detector_button.setDisabled(True)

    def moveDetectorsButtonClicked(self):
        q = [control.value for control in self.position_form_group.form_controls]
        q = self.detector.positioner.fromUserFormat(q)

        if not is_close(q, self.detector.positioner.set_points, POS_EPS):
            name = self.detector.positioner.name
            self.parent.presenter.movePositioner(name, q)

    def closeEvent(self, event):
        self.parent.scenes.changeVisibility(Attributes.Beam, False)
        event.accept()


class SampleProperties(QtWidgets.QWidget):
    """Creates a UI for viewing sample properties

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Bottom

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.main_layout = QtWidgets.QVBoxLayout()
        self.sample_property_table = QtWidgets.QTableWidget()
        self.sample_property_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.sample_property_table.verticalHeader().hide()
        self.sample_property_table.setColumnCount(2)
        self.sample_property_table.setRowCount(5)
        self.sample_property_table.setAlternatingRowColors(True)
        self.sample_property_table.setMinimumHeight(100)
        self.sample_property_table.setMaximumHeight(165)
        self.sample_property_table.setHorizontalHeaderLabels(['Property', 'Value'])
        self.sample_property_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.addSampleProperties()
        self.sample_property_table.resizeRowsToContents()
        self.main_layout.addWidget(self.sample_property_table, 20)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)
        self.title = 'Sample Properties'
        self.setMinimumWidth(450)
        self.parent_model.sample_changed.connect(self.updateTable)

    def updateTable(self):
        """Updates the contents of the table when sample changes"""
        self.sample_property_table.clearContents()
        self.addSampleProperties()

    def addSampleProperties(self):
        """Adds the sample property labels to the table"""
        self._addItem(0, 'Memory (MB)', self._getMemoryUsed())
        if isinstance(self.parent_model.sample, Mesh):
            self._addMeshProperties()
        elif isinstance(self.parent_model.sample, Volume):
            self._addVolumeProperties()

    def _addMeshProperties(self):
        """Adds the mesh property values to the table"""
        num_faces = self.parent_model.sample.indices.shape[0] // 3
        self._addItem(1, 'Faces', num_faces)

        num_vertices = self.parent_model.sample.vertices.shape[0]
        self._addItem(2, 'Vertices', num_vertices)

    def _getMemoryUsed(self):
        """Calculates the memory used by the sample

        :return memory_str: memory consumed in mb
        :rtype: string
        """
        if isinstance(self.parent_model.sample, Mesh):
            bytes_used = self.parent_model.sample.vertices.nbytes
        elif isinstance(self.parent_model.sample, Volume):
            bytes_used = self.parent_model.sample.data.nbytes
        elif self.parent_model.sample is None:
            bytes_used = 0
        memory_str = self._bytesToMb(bytes_used)
        return memory_str

    def _bytesToMb(self, bytes_used):
        """Converts the bytes consumed to mb

        :param bytes_used: the bytes used by the sample
        :type bytes_used: int
        :return memory: consumed in mb
        :rtype: string
        """
        bytes_to_mb_factor = 1 / (1024**2)
        memory = bytes_used * bytes_to_mb_factor
        memory_str = '0' if memory == 0 else f'{memory:.4f}'
        return memory_str

    def _addVolumeProperties(self):
        """Adds the Volume property values to the table"""
        x, y, z = self.parent_model.sample.data.shape
        dimension_str = f'{x} x {y} x {z}'
        self._addItem(1, 'Dimension', dimension_str)

        voxel_size = self.parent_model.sample.voxel_size
        voxel_size_str = f'[x: {voxel_size[0]},  y: {voxel_size[1]},  z: {voxel_size[2]}]'
        self._addItem(2, 'Voxel size', voxel_size_str)

        min_intensity = np.min(self.parent_model.sample.data)
        self._addItem(3, 'Minimum Intensity', min_intensity)

        max_intensity = np.max(self.parent_model.sample.data)
        self._addItem(4, 'Maximum Intensity', max_intensity)

    def _addItem(self, row, label, value):
        """Adds a Property label or Value to the table

        :param row: the row in table to add the label and value
        :type row: int
        :param label: the sample property label
        :type label: string
        :param value: the value of the sample property  
        :type value: Union[string, int]
        """
        for ix, item in enumerate([label, value]):
            table_item = QtWidgets.QTableWidgetItem(str(item))
            table_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.sample_property_table.setItem(row, ix, table_item)
