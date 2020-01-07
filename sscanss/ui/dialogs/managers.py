import math
from PyQt5 import QtWidgets, QtGui, QtCore
from sscanss.config import path_for, settings
from sscanss.core.instrument import Link
from sscanss.core.util import DockFlag, PointType, CommandID, Attributes
from sscanss.ui.widgets import (PointModel, FormControl, FormGroup, FormTitle, create_tool_button,
                                create_scroll_area, create_icon)


class SampleManager(QtWidgets.QWidget):
    dock_flag = DockFlag.Bottom

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.parent.scenes.switchToSampleScene()

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
        self.delete_button = create_tool_button(icon_path=path_for('cross.png'), style_name='ToolButton',
                                                tooltip='Delete Sample',
                                                status_tip='Remove selected sample from project')
        self.delete_button.clicked.connect(self.removeSamples)
        button_layout.addWidget(self.delete_button)

        self.merge_button = create_tool_button(icon_path=path_for('merge.png'), style_name='ToolButton',
                                               tooltip='Merge Samples',
                                               status_tip='Combine vertices of selected samples to form single sample')
        self.merge_button.clicked.connect(self.mergeSamples)
        button_layout.addWidget(self.merge_button)

        self.priority_button = create_tool_button(icon_path=path_for('check.png'), style_name='ToolButton',
                                                  tooltip='Set Main Sample',
                                                  status_tip='Set selected sample as main sample')
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
        self.setMinimumWidth(450)

    def updateSampleList(self):
        self.list_widget.clear()
        samples = self.parent_model.sample.keys()
        if samples:
            self.list_widget.addItems(samples)
            self.list_widget.item(0).setIcon(QtGui.QIcon(path_for('check.png')))

    def removeSamples(self):
        keys = [item.text() for item in self.list_widget.selectedItems()]
        if keys:
            self.parent.presenter.deleteSample(keys)

    def mergeSamples(self):
        keys = [item.text() for item in self.list_widget.selectedItems()]
        if len(keys) > 1:
            self.parent.presenter.mergeSample(keys)
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def changeMainSample(self):
        item = self.list_widget.currentItem()

        if item and item.text() != list(self.parent_model.sample.keys())[0]:
            key = item.text()
            self.parent.presenter.changeMainSample(key)
            self.list_widget.setCurrentRow(0)

    def onMultiSelection(self):
        selections = [self.list_widget.item(i).isSelected() for i in range(self.list_widget.count())]
        self.parent.scenes.changeSelected(Attributes.Sample, selections)
        if len(self.list_widget.selectedItems()) > 1:
            self.priority_button.setDisabled(True)
        else:
            self.priority_button.setEnabled(True)

    def closeEvent(self, event):
        selections = [False] * self.list_widget.count()
        self.parent.scenes.changeSelected(Attributes.Sample, selections)
        event.accept()


class PointManager(QtWidgets.QWidget):
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
        self.table_model = PointModel(self.points)
        self.table_model.editCompleted.connect(self.editPoints)
        self.table_view.horizontalHeader().sectionClicked.connect(self.table_model.toggleCheckState)
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setMinimumHeight(300)
        self.table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_view.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        self.table_view.horizontalHeader().setMinimumSectionSize(40)
        self.table_view.horizontalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.table_view)

        button_layout = QtWidgets.QVBoxLayout()
        self.delete_button = create_tool_button(icon_path=path_for('cross.png'), style_name='ToolButton',
                                                tooltip='Delete Points',
                                                status_tip='Remove selected points from project')
        self.delete_button.clicked.connect(self.deletePoints)
        button_layout.addWidget(self.delete_button)

        self.move_up_button = create_tool_button(icon_path=path_for('arrow-up.png'), style_name='ToolButton',
                                                 tooltip='Move Point Index Up',
                                                 status_tip='Swap point index with previous point')
        self.move_up_button.clicked.connect(lambda: self.movePoint(-1))
        button_layout.addWidget(self.move_up_button)

        self.move_down_button = create_tool_button(icon_path=path_for('arrow-down.png'), style_name='ToolButton',
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

        self.title = '{} Points'.format(self.point_type.value)
        self.setMinimumWidth(450)
        self.table_view.selectionModel().selectionChanged.connect(self.onMultiSelection)
        if self.point_type == PointType.Fiducial:
            self.parent_model.fiducials_changed.connect(self.updateTable)
        elif self.point_type == PointType.Measurement:
            self.parent_model.measurement_points_changed.connect(self.updateTable)

    @property
    def points(self):
        if self.point_type == PointType.Fiducial:
            return self.parent_model.fiducials
        else:
            return self.parent_model.measurement_points

    def updateTable(self):
        self.table_model.update(self.points)
        self.table_view.update()
        if self.selected is not None:
            self.table_view.setCurrentIndex(self.selected)

    def deletePoints(self):
        selection_model = self.table_view.selectionModel()
        indices = [item.row() for item in selection_model.selectedRows()]
        if indices:
            self.parent.presenter.deletePoints(indices, self.point_type)
            self.selected = None

    def movePoint(self, offset):
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
        self.selected = self.table_view.currentIndex()
        self.table_view.selectionModel().reset()
        self.parent.presenter.editPoints(new_values, self.point_type)

    def onMultiSelection(self):
        sm = self.table_view.selectionModel()
        selections = [sm.isRowSelected(i, QtCore.QModelIndex()) for i in range(self.table_model.rowCount())]
        self.parent.scenes.changeSelected(self.attribute, selections)
        if len(sm.selectedRows()) > 1:
            self.move_down_button.setDisabled(True)
            self.move_up_button.setDisabled(True)
        else:
            self.move_down_button.setEnabled(True)
            self.move_up_button.setEnabled(True)

    def closeEvent(self, event):
        selections = [False] * self.table_model.rowCount()
        self.parent.scenes.changeSelected(self.attribute, selections)
        event.accept()


class VectorManager(QtWidgets.QWidget):
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

        self.delete_vector_action = QtWidgets.QAction("Delete Vectors", self)
        self.delete_vector_action.triggered.connect(self.delete_vectors)
        self.delete_vector_action.setStatusTip('Remove (zero) selected vectors or all vectors in current alignment')

        self.delete_alignment_action = QtWidgets.QAction("Delete Alignment", self)
        self.delete_alignment_action.triggered.connect(self.delete_alignment)
        self.delete_alignment_action.setStatusTip('Remove selected vector alignment')
        delete_menu = QtWidgets.QMenu()
        delete_menu.addAction(self.delete_vector_action)
        delete_menu.addAction(self.delete_alignment_action)

        self.delete_alignment_button = create_tool_button(icon_path=path_for('cross.png'), style_name='ToolButton',
                                                          status_tip='Remove measurement vectors or alignments')
        self.delete_alignment_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.delete_alignment_button.setMenu(delete_menu)

        layout.addStretch(1)
        layout.addWidget(self.delete_alignment_button)
        layout.setAlignment(self.delete_alignment_button, QtCore.Qt.AlignBottom)
        self.main_layout.addLayout(layout, 0)

        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['X', 'Y', 'Z'])
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(300)
        self.table.setMaximumHeight(500)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
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
        detector = self.detector_combobox.currentIndex()
        alignment = self.parent.scenes.rendered_alignment
        alignment = self.updateComboBoxes(alignment)
        self.updateTable(detector, alignment)

    def updateTable(self, detector, alignment):
        detector_index = slice(detector * 3, detector * 3 + 3)
        vectors = self.parent_model.measurement_vectors[:, detector_index, alignment]
        self.table.setRowCount(vectors.shape[0])

        for row, vector in enumerate(vectors):
            x = QtWidgets.QTableWidgetItem('{:.3f}'.format(vector[0]))
            x.setTextAlignment(QtCore.Qt.AlignCenter)
            y = QtWidgets.QTableWidgetItem('{:.3f}'.format(vector[1]))
            y.setTextAlignment(QtCore.Qt.AlignCenter)
            z = QtWidgets.QTableWidgetItem('{:.3f}'.format(vector[2]))
            z.setTextAlignment(QtCore.Qt.AlignCenter)

            self.table.setItem(row, 0, x)
            self.table.setItem(row, 1, y)
            self.table.setItem(row, 2, z)

    def onComboBoxActivated(self):
        detector = self.detector_combobox.currentIndex()
        alignment = self.alignment_combobox.currentIndex()
        self.parent.scenes.changeRenderedAlignment(alignment)
        self.updateTable(detector, alignment)

    def updateComboBoxes(self, alignment):
        align_count = self.parent_model.measurement_vectors.shape[2]
        self.delete_alignment_action.setEnabled(align_count > 1)

        alignment_list = ['{}'.format(i + 1) for i in range(align_count)]
        self.alignment_combobox.clear()
        self.alignment_combobox.addItems(alignment_list)
        current_alignment = alignment if 0 < alignment < len(alignment_list) else 0
        self.alignment_combobox.setCurrentIndex(alignment)

        return current_alignment

    def delete_alignment(self):
        self.parent.presenter.removeVectorAlignment(self.alignment_combobox.currentIndex())

    def delete_vectors(self):
        selection_model = self.table.selectionModel()
        indices = [item.row() for item in selection_model.selectedRows()]

        if not indices:
            indices = list(range(self.table.rowCount()))

        detector_index = self.detector_combobox.currentIndex()
        alignment_index = self.alignment_combobox.currentIndex()

        vectors = self.parent_model.measurement_vectors[indices, slice(detector_index * 3, detector_index * 3 + 3),
                                                        alignment_index]
        if (vectors < 0.001).all():
            return
        self.parent.presenter.removeVectors(indices, detector_index, alignment_index)

    def closeEvent(self, event):
        self.parent.scenes.changeRenderedAlignment(0)
        event.accept()


class JawControl(QtWidgets.QWidget):
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
        self.parent.scenes.toggleVisibility(Attributes.Beam, True)

    def updateForms(self, id):
        if id == CommandID.ChangeJawAperture:
            self.aperture_form_group.form_controls[0].value = self.instrument.jaws.aperture[0]
            self.aperture_form_group.form_controls[1].value = self.instrument.jaws.aperture[1]
        elif id == CommandID.MovePositioner:
            positioner = self.instrument.jaws.positioner
            set_points = positioner.toUserFormat(positioner.set_points)
            for value, control in zip(set_points, self.position_form_group.form_controls):
                control.value = value
        elif id == CommandID.IgnoreJointLimits:
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

            control = FormControl(link.name.title(), offset, desc=unit, required=True, number=True)
            control.form_lineedit.setDisabled(link.locked)
            if not link.ignore_limits:
                control.range(lower_limit, upper_limit)

            limits_button = create_tool_button(tooltip='Disable Joint Limits',  style_name='MidToolButton',
                                               icon_path=path_for('limit.png'), checkable=True,
                                               status_tip=f'Ignore joint limits of {link.name}',
                                               checked=link.ignore_limits)
            limits_button.clicked.connect(self.adjustJointLimits)
            limits_button.setProperty('link_index', index)

            control.extra = [limits_button]
            self.position_form_group.addControl(control)
            self.position_form_group.groupValidation.connect(self.formValidation)

        self.main_layout.addWidget(self.position_form_group)
        button_layout = QtWidgets.QHBoxLayout()
        self.move_jaws_button = QtWidgets.QPushButton('Move Jaws')
        self.move_jaws_button.clicked.connect(self.moveJawsButtonClicked)
        button_layout.addWidget(self.move_jaws_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def createApertureForm(self):
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
        self.aperture_form_group.groupValidation.connect(self.formValidation)
        self.main_layout.addSpacing(10)

        button_layout = QtWidgets.QHBoxLayout()
        self.change_aperture_button = QtWidgets.QPushButton('Change Aperture Size')
        self.change_aperture_button.clicked.connect(self.changeApertureButtonClicked)
        button_layout.addWidget(self.change_aperture_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def adjustJointLimits(self, check_state):
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

        if q != self.instrument.jaws.positioner.set_points:
            name = self.instrument.jaws.positioner.name
            self.parent.presenter.movePositioner(name, q)

    def changeApertureButtonClicked(self):
        aperture = [self.aperture_form_group.form_controls[0].value,
                    self.aperture_form_group.form_controls[1].value]
        self.parent.presenter.changeJawAperture(aperture)

    def closeEvent(self, event):
        self.parent.scenes.toggleVisibility(Attributes.Beam, False)
        event.accept()


class PositionerControl(QtWidgets.QWidget):
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
            self.stack_combobox.activated[str].connect(self.changeStack)
            stack_layout.addWidget(self.stack_combobox)
            self.main_layout.addLayout(stack_layout)
            self.main_layout.addSpacing(5)

        self.positioner_forms_layout = QtWidgets.QVBoxLayout()
        self.positioner_forms_layout.setContentsMargins(0, 0, 0, 0)

        self.createForms()
        self.main_layout.addLayout(self.positioner_forms_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.move_joints_button = QtWidgets.QPushButton('Move Joints')
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

    def updateForms(self, id):
        if id == CommandID.ChangePositioningStack:
            self.stack_combobox.setCurrentText(self.instrument.positioning_stack.name)
            self.createForms()
        elif id == CommandID.ChangePositionerBase:
            for aux in self.instrument.positioning_stack.auxiliary:
                button = self.base_reset_buttons[aux.name]
                button.setVisible(aux.base is not aux.default_base)
        elif id == CommandID.MovePositioner:
            positioner = self.instrument.positioning_stack
            set_points = positioner.toUserFormat(positioner.set_points)
            for value, control in zip(set_points, self.positioner_form_controls):
                control.value = value
        elif id == CommandID.LockJoint:
            order = self.instrument.positioning_stack.order
            for index, control in zip(order, self.positioner_form_controls):
                link = self.instrument.positioning_stack.links[index]
                control.form_lineedit.setDisabled(link.locked)
                toggle_button = control.extra[0]
                toggle_button.setChecked(link.locked)
                if link.locked:
                    control.value = link.set_point if link.type == Link.Type.Prismatic else math.degrees(link.set_point)
        elif id == CommandID.IgnoreJointLimits:
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

    def changeStack(self, selected):
        if selected != self.instrument.positioning_stack.name:
            self.parent.presenter.changePositioningStack(selected)

    def createForms(self):
        for i in range(self.positioner_forms_layout.count()):
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
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        title = FormTitle(positioner.name)
        layout.addWidget(title)
        if add_base_button:
            base_button = create_tool_button(tooltip='Change Base Matrix', style_name='MidToolButton',
                                             status_tip=f'Import base transformation matrix for {positioner.name}',
                                             icon_path=path_for('base.png'))
            base_button.clicked.connect(lambda ignore, n=positioner.name: self.changePositionerBase(n))
            title.addHeaderControl(base_button)

            reset_button = create_tool_button(tooltip='Reset Base Matrix', style_name='MidToolButton',
                                              status_tip=f'Reset base transformation matrix of {positioner.name}',
                                              icon_path=path_for('refresh.png'), hide=True)
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

            control = FormControl(link.name.title(), offset, desc=unit, required=True, number=True)
            control.form_lineedit.setDisabled(link.locked)
            if not link.ignore_limits:
                control.range(lower_limit, upper_limit)

            lock_button = create_tool_button(tooltip='Lock Joint',  style_name='MidToolButton',
                                             status_tip=f'Lock {link.name} joint in {positioner.name}',
                                             icon_path=path_for('lock.png'), checkable=True,
                                             checked=link.locked)
            lock_button.clicked.connect(self.lockJoint)
            lock_button.setProperty('link_index', index + order_offset)

            limits_button = create_tool_button(tooltip='Disable Joint Limits',  style_name='MidToolButton',
                                               status_tip=f'Ignore joint limits of {link.name}',
                                               icon_path=path_for('limit.png'), checkable=True,
                                               checked=link.ignore_limits)
            limits_button.clicked.connect(self.adjustJointLimits)
            limits_button.setProperty('link_index', index + order_offset)

            control.extra = [lock_button, limits_button]
            form_group.addControl(control)

            form_group.groupValidation.connect(self.formValidation)
            self.positioner_form_controls.append(control)

        layout.addWidget(form_group)
        widget.setLayout(layout)
        self.positioner_forms.append(form_group)

        return widget

    def lockJoint(self, check_state):
        index = self.sender().property('link_index')
        name = self.instrument.positioning_stack.name
        self.parent.presenter.lockPositionerJoint(name, index, check_state)

    def adjustJointLimits(self, check_state):
        index = self.sender().property('link_index')
        name = self.instrument.positioning_stack.name
        self.parent.presenter.ignorePositionerJointLimits(name, index, check_state)

    def formValidation(self):
        for form in self.positioner_forms:
            if not form.valid:
                self.move_joints_button.setDisabled(True)
                return
        self.move_joints_button.setEnabled(True)

    def changePositionerBase(self, name):
        matrix = self.parent.presenter.importTransformMatrix()
        if matrix is None:
            return

        positioner = self.instrument.positioners[name]
        self.parent.presenter.changePositionerBase(positioner, matrix)

    def resetPositionerBase(self, name):
        positioner = self.instrument.positioners[name]
        self.parent.presenter.changePositionerBase(positioner, positioner.default_base)

    def moveJointsButtonClicked(self):
        q = [control.value for control in self.positioner_form_controls]
        q = self.instrument.positioning_stack.fromUserFormat(q)

        if q != self.instrument.positioning_stack.set_points:
            name = self.instrument.positioning_stack.name
            self.parent.presenter.movePositioner(name, q)


class DetectorControl(QtWidgets.QWidget):
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
        self.parent.scenes.toggleVisibility(Attributes.Beam, True)

    def updateForms(self, id):
        if id == CommandID.MovePositioner:
            positioner = self.detector.positioner
            set_points = positioner.toUserFormat(positioner.set_points)
            for value, control in zip(set_points, self.position_form_group.form_controls):
                control.value = value
        elif id == CommandID.IgnoreJointLimits:
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
            control = FormControl(pretty_label, offset, desc=unit, required=True, number=True)
            control.form_lineedit.setDisabled(link.locked)
            if not link.ignore_limits:
                control.range(lower_limit, upper_limit)

            limits_button = create_tool_button(tooltip='Disable Joint Limits',  style_name='MidToolButton',
                                               icon_path=path_for('limit.png'), checkable=True,
                                               status_tip=f'Ignore joint limits of {pretty_label}',
                                               checked=link.ignore_limits)
            limits_button.clicked.connect(self.adjustJointLimits)
            limits_button.setProperty('link_index', index)

            control.extra = [limits_button]
            self.position_form_group.addControl(control)
            self.position_form_group.groupValidation.connect(self.formValidation)

        self.main_layout.addWidget(self.position_form_group)
        button_layout = QtWidgets.QHBoxLayout()
        self.move_detector_button = QtWidgets.QPushButton('Move Detector')
        self.move_detector_button.clicked.connect(self.moveDetectorsButtonClicked)
        button_layout.addWidget(self.move_detector_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def adjustJointLimits(self, check_state):
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

        if q != self.detector.positioner.set_points:
            name = self.detector.positioner.name
            self.parent.presenter.movePositioner(name, q)

    def closeEvent(self, event):
        self.parent.scenes.toggleVisibility(Attributes.Beam, False)
        event.accept()
