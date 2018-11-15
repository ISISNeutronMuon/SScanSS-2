import math
from collections import OrderedDict
from PyQt5 import QtWidgets, QtGui, QtCore
from sscanss.core.instrument import Link
from sscanss.core.util import DockFlag, PointType
from sscanss.ui.widgets import (NumpyModel, FormControl, FormGroup, FormTitle, create_tool_button,
                                create_scroll_area)


class SampleManager(QtWidgets.QWidget):
    dock_flag = DockFlag.Bottom

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

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
        self.delete_button = create_tool_button(icon_path='../static/images/cross.png', style_name='ToolButton')
        self.delete_button.clicked.connect(self.removeSamples)
        button_layout.addWidget(self.delete_button)

        self.merge_button = create_tool_button(icon_path='../static/images/merge.png', style_name='ToolButton')
        self.merge_button.clicked.connect(self.mergeSamples)
        button_layout.addWidget(self.merge_button)

        self.priority_button = create_tool_button(icon_path='../static/images/check.png', style_name='ToolButton')
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
        self.setMinimumWidth(350)

    def updateSampleList(self):
        self.list_widget.clear()
        samples = self.parent_model.sample.keys()
        if samples:
            self.list_widget.addItems(samples)
            self.list_widget.item(0).setIcon(QtGui.QIcon('../static/images/check.png'))

    def removeSamples(self):
        keys = [item.text() for item in self.list_widget.selectedItems()]
        if keys:
            self.parent.presenter.deleteSample(keys)

    def mergeSamples(self):
        keys = [item.text() for item in self.list_widget.selectedItems()]
        if keys and len(keys) < 2:
            return

        self.parent.presenter.mergeSample(keys)
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def changeMainSample(self):
        item = self.list_widget.currentItem()

        if not item:
            return

        key = item.text()
        self.parent.presenter.changeMainSample(key)
        self.list_widget.setCurrentRow(0)

    def onMultiSelection(self):
        if len(self.list_widget.selectedItems()) > 1:
            self.priority_button.setDisabled(True)
        else:
            self.priority_button.setEnabled(True)


class PointManager(QtWidgets.QWidget):
    dock_flag = DockFlag.Bottom

    def __init__(self, point_type, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model
        self.point_type = point_type

        self.selected = None

        layout = QtWidgets.QHBoxLayout()
        self.table_view = QtWidgets.QTableView()
        self.updateTable()
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setMinimumHeight(300)
        self.table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_view.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        self.table_view.horizontalHeader().setMinimumSectionSize(40)
        self.table_view.horizontalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.table_view)

        button_layout = QtWidgets.QVBoxLayout()
        self.delete_button = create_tool_button(icon_path='../static/images/cross.png', style_name='ToolButton')
        self.delete_button.clicked.connect(self.deletePoints)
        button_layout.addWidget(self.delete_button)

        self.move_up_button = create_tool_button(icon_path='../static/images/arrow-up.png', style_name='ToolButton')
        self.move_up_button.clicked.connect(lambda: self.movePoint(-1))
        button_layout.addWidget(self.move_up_button)

        self.move_down_button = create_tool_button(icon_path='../static/images/arrow-down.png', style_name='ToolButton')
        self.move_down_button.clicked.connect(lambda: self.movePoint(1))
        button_layout.addWidget(self.move_down_button)

        layout.addSpacing(10)
        layout.addLayout(button_layout)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = '{} Points'.format(self.point_type.value)
        self.setMinimumWidth(350)
        self.table_view.clicked.connect(self.onMultiSelection)
        if self.point_type == PointType.Fiducial:
            self.parent_model.fiducials_changed.connect(self.updateTable)
        elif self.point_type == PointType.Measurement:
            self.parent_model.measurement_points_changed.connect(self.updateTable)

    def updateTable(self):
        if self.point_type == PointType.Fiducial:
            points = self.parent_model.fiducials
        elif self.point_type == PointType.Measurement:
            points = self.parent_model.measurement_points

        self.table_model = NumpyModel(points, parent=self.table_view)
        self.table_model.editCompleted.connect(self.editPoints)
        self.table_view.setModel(self.table_model)
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

    def editPoints(self, row, new_value):
        self.parent.presenter.editPoints(row, new_value, self.point_type)

    def onMultiSelection(self):
        selection_model = self.table_view.selectionModel()
        indices = [item.row() for item in selection_model.selectedRows()]
        if len(indices) > 1:
            self.move_down_button.setDisabled(True)
            self.move_up_button.setDisabled(True)
        else:
            self.move_down_button.setEnabled(True)
            self.move_up_button.setEnabled(True)


class VectorManager(QtWidgets.QWidget):
    dock_flag = DockFlag.Bottom

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.main_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        alignment_layout = QtWidgets.QVBoxLayout()
        alignment_layout.addWidget(QtWidgets.QLabel('Alignment:'))
        self.alignment_combobox = QtWidgets.QComboBox()
        self.alignment_combobox.setView(QtWidgets.QListView())
        self.alignment_combobox.addItem('1')
        self.alignment_combobox.activated.connect(self.updateWidget)
        alignment_layout.addWidget(self.alignment_combobox)
        layout.addLayout(alignment_layout)
        layout.addSpacing(10)

        detector_layout = QtWidgets.QVBoxLayout()
        detector_layout.addWidget(QtWidgets.QLabel('Detector:'))
        self.detector_combobox = QtWidgets.QComboBox()
        self.detector_combobox.setView(QtWidgets.QListView())
        self.detector_combobox.addItem('1')
        self.detector_combobox.activated.connect(self.updateWidget)
        detector_layout.addWidget(self.detector_combobox)
        layout.addLayout(detector_layout)
        self.main_layout.addLayout(layout)

        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['X', 'Y', 'Z'])
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(300)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(40)
        self.table.horizontalHeader().setDefaultSectionSize(40)
        self.updateWidget()
        self.main_layout.addWidget(self.table)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = 'Measurement Vectors'
        self.setMinimumWidth(350)
        self.parent_model.measurement_vectors_changed.connect(self.updateWidget)

    def updateWidget(self):
        detector = self.detector_combobox.currentIndex()
        alignment = self.alignment_combobox.currentIndex()

        detector, alignment = self.updateComboBoxes(detector, alignment)

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

    def updateComboBoxes(self, detector, alignment):
        align_count = self.parent_model.measurement_vectors.shape[2]
        alignment_list = ['{}'.format(i + 1) for i in range(align_count)]
        self.alignment_combobox.clear()
        self.alignment_combobox.addItems(alignment_list)
        current_alignment = alignment if alignment < len(alignment_list) else 0
        self.alignment_combobox.setCurrentIndex(current_alignment)

        detector_count = self.parent_model.measurement_vectors.shape[1] // 3
        detector_list = ['{}'.format(i + 1) for i in range(detector_count)]
        self.detector_combobox.clear()
        self.detector_combobox.addItems(detector_list)
        current_detector = detector if detector < len(detector_list) else 0
        self.detector_combobox.setCurrentIndex(current_detector)

        return current_detector, current_alignment


class JawControl(QtWidgets.QWidget):
    dock_flag = DockFlag.Full

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.instrument = self.parent_model.active_instrument
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(QtWidgets.QLabel(
            'Change {} position'.format('incident jaws')))
        self.main_layout.addSpacing(10)

        self.position_form_group = FormGroup()
        for axis in self.instrument.jaws.axes:
            if axis.type == Link.Type.Revolute:
                unit = 'degrees'
                offset = math.radians(axis.offset)
                lower_limit = math.radians(axis.lower_limit)
                upper_limit = math.radians(axis.upper_limit)
            else:
                unit = 'mm'
                offset = axis.offset
                lower_limit = axis.lower_limit
                upper_limit = axis.upper_limit

            pretty_label = axis.name.replace('_', ' ').title()
            control = FormControl(pretty_label, offset, unit=unit, required=True, number=True)
            control.range(lower_limit, upper_limit)
            self.position_form_group.addControl(control)

        self.main_layout.addWidget(self.position_form_group)
        self.position_form_group.groupValidation.connect(self.formValidation)
        self.position_limit_checkbox = QtWidgets.QCheckBox('Ignore Jaw Axes Limits')
        self.position_limit_checkbox.stateChanged.connect(self.adjustPositionLimits)
        self.main_layout.addWidget(self.position_limit_checkbox)

        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(QtWidgets.QLabel('Change incident jaws aperture size'))
        self.main_layout.addSpacing(10)
        aperture = self.instrument.jaws.aperture
        aperture_upper_limit = self.instrument.jaws.aperture_upper_limit
        aperture_lower_limit = self.instrument.jaws.aperture_lower_limit
        self.aperture_form_group = FormGroup(FormGroup.Layout.Horizontal)
        control = FormControl('Horizontal Aperture Size', aperture[0], unit='mm', required=True, number=True)
        control.range(aperture_lower_limit[0], aperture_upper_limit[0])
        self.aperture_form_group.addControl(control)
        control = FormControl('Vertical Aperture Size', aperture[1], unit='mm', required=True, number=True)
        control.range(aperture_lower_limit[1], aperture_upper_limit[1])
        self.aperture_form_group.addControl(control)
        self.main_layout.addWidget(self.aperture_form_group)
        self.aperture_form_group.groupValidation.connect(self.formValidation)
        self.aperture_limit_checkbox = QtWidgets.QCheckBox('Ignore Aperture Limits')
        self.aperture_limit_checkbox.stateChanged.connect(self.adjustApetureLimits)
        self.main_layout.addWidget(self.aperture_limit_checkbox)
        self.main_layout.addSpacing(10)

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton('Apply')
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.title = 'Control Jaws'
        self.setMinimumWidth(350)
        # self.parent_model.measurement_vectors_changed.connect(self.updateWidget)

    def adjustPositionLimits(self, check_state):
        axes = self.instrument.jaws.axes
        for i, control in enumerate(self.position_form_group.form_controls):
            if check_state == QtCore.Qt.Checked:
                control.range(None, None)
            else:
                if axes[i].type == Link.Type.Revolute:
                    lower_limit = math.radians(axes[i].lower_limit)
                    upper_limit = math.radians(axes[i].upper_limit)
                else:
                    lower_limit = axes[i].lower_limit
                    upper_limit = axes[i].upper_limit

                control.range(lower_limit, upper_limit)

    def adjustApetureLimits(self, check_state):
        controls = self.aperture_form_group.form_controls
        aperture_upper_limit = self.instrument.jaws.aperture_upper_limit
        aperture_lower_limit = self.instrument.jaws.aperture_lower_limit
        if check_state == QtCore.Qt.Checked:
            controls[0].range(0, None)
            controls[1].range(0, None)
        else:
            controls[0].range(aperture_lower_limit[0], aperture_upper_limit[0])
            controls[1].range(aperture_lower_limit[1], aperture_upper_limit[1])

    def formValidation(self):
        if self.position_form_group.valid and self.aperture_form_group.valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):

        end_q = []
        for control in self.position_form_group.form_controls:
            end_q.append(control.value)

        self.instrument.jaws.aperture[0] = self.aperture_form_group.form_controls[0].value
        self.instrument.jaws.aperture[1] = self.aperture_form_group.form_controls[1].value

        q = self.instrument.jaws.positioner.configuration
        self.parent_model.animateInstrument(self.instrument.jaws.move,
                                            q, end_q, 500, 10)


class PositionerControl(QtWidgets.QWidget):
    dock_flag = DockFlag.Full

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.instrument = self.parent_model.active_instrument
        self.main_layout = QtWidgets.QVBoxLayout()

        self.positioner_forms = OrderedDict()
        self.positioner_widgets = {}
        self.add_auxiliary = []
        self.remove_auxiliary = []

        positioner = self.instrument.positioning_stack.fixed
        widget = self.createPositionerWidget(positioner)
        self.main_layout.addWidget(widget)

        for name in list(self.instrument.auxiliary_positioners):
            positioner = self.instrument.positioners[name]
            if positioner in self.instrument.positioning_stack.auxiliary:
                self.remove_auxiliary.append(name)
                widget = self.createPositionerWidget(positioner, True)
                self.main_layout.addWidget(widget)
            else:
                self.add_auxiliary.append(name)

        self.main_layout.addSpacing(10)

        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton('Apply')
        self.execute_button.clicked.connect(self.executeButtonClicked)

        self.add_positioner_button = create_tool_button(style_name='DropDownButton', text='Add Positioner')
        self.add_positioner_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.remove_positioner_button = create_tool_button(style_name='DropDownButton', text='Remove Positioner')
        self.remove_positioner_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        button_layout.addWidget(self.execute_button)
        button_layout.addWidget(self.add_positioner_button)
        button_layout.addWidget(self.remove_positioner_button)

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

        self.title = 'Control Positioning System'
        self.setMinimumWidth(450)
        self.__updateButtons()

    def createPositionerWidget(self, positioner, add_base_button=False):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        title = FormTitle(positioner.name)
        layout.addWidget(title)
        if add_base_button:
            base_button = create_tool_button(tooltip='Change Base Matrix', style_name='MidToolButton',
                                        icon_path='../static/images/base.png')
            base_button.clicked.connect(lambda ignore, n=positioner.name: self.changePositionerBase(n))
            title.addHeaderControl(base_button)

            reset_button = create_tool_button(tooltip='Reset Base Matrix', style_name='MidToolButton',
                                        icon_path='../static/images/refresh.png', hide=True)
            reset_button.clicked.connect(lambda ignore, n=positioner.name: self.resetPositionerBase(n))
            title.addHeaderControl(reset_button)
            base_button.setProperty('reset', reset_button)

        form_group = FormGroup(FormGroup.Layout.Grid)
        for link in positioner.links:
            if link.type == Link.Type.Revolute:
                unit = 'degrees'
                offset = math.degrees(link.offset)
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                unit = 'mm'
                offset = link.offset
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            pretty_label = link.name.replace('_', ' ').title()
            control = FormControl(pretty_label, offset, unit=unit, required=True, number=True)
            control.range(lower_limit, upper_limit)
            control.setProperty('link', link)
            locking_checkbox = QtWidgets.QCheckBox("Locked")
            locking_checkbox.stateChanged.connect(self.lockJoint)
            locking_checkbox.setProperty('form', control)
            locking_checkbox.setProperty('link', link)
            limits_checkbox = QtWidgets.QCheckBox("Ignore Limits")
            limits_checkbox.setProperty('form', control)
            limits_checkbox.setProperty('link', link)
            limits_checkbox.stateChanged.connect(self.adjustJointLimits)

            extras = [locking_checkbox, limits_checkbox]
            form_group.addControl(control, extras)

            form_group.groupValidation.connect(self.formValidation)

        layout.addWidget(form_group)
        widget.setLayout(layout)
        self.positioner_forms[positioner.name] = form_group
        self.positioner_widgets[positioner.name] = widget

        return widget

    def lockJoint(self, check_state):
        sender = self.sender()
        form = sender.property('form')
        form.form_control.setDisabled(check_state == QtCore.Qt.Checked)
        link = sender.property('link')
        link.locked = check_state == QtCore.Qt.Checked
        form.value = link.offset if link.type == Link.Type.Prismatic else math.degrees(link.offset)

    def adjustJointLimits(self, check_state):
        sender = self.sender()
        form = sender.property('form')
        link = sender.property('link')
        link.ignore_limits = check_state == QtCore.Qt.Checked

        if check_state == QtCore.Qt.Checked:
            form.range(None, None)
        else:
            if link.type == Link.Type.Revolute:
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            form.range(lower_limit, upper_limit)

    def formValidation(self):
        for form in self.positioner_forms.values():
            if not form.valid:
                self.execute_button.setDisabled(True)
                return
        self.execute_button.setEnabled(True)

    def __createAuxiliary(self, add_positioner=True):
        menu = QtWidgets.QMenu(self)
        if add_positioner:
            menu_function = self.addPositioner
            positioner_names = self.add_auxiliary
        else:
            menu_function = self.removePositioner
            positioner_names = self.remove_auxiliary

        for name in positioner_names:
            action = QtWidgets.QAction(name, self)
            action.triggered.connect(lambda ignore, n=name: menu_function(n))
            menu.addAction(action)

        return menu

    def removePositioner(self, name):
        positioner = self.instrument.positioners[name]
        self.instrument.positioning_stack.removePositioner(positioner)
        self.parent_model.updateInstrumentScene()
        self.positioner_forms.pop(name)
        widget = self.positioner_widgets.pop(name)
        self.main_layout.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

        self.remove_auxiliary.remove(name)
        self.add_auxiliary.append(name)
        self.__updateButtons()

    def __updateButtons(self):
        self.add_positioner_button.setMenu(self.__createAuxiliary())
        self.remove_positioner_button.setMenu(self.__createAuxiliary(False))
        self.add_positioner_button.setVisible(len(self.add_auxiliary) != 0)
        self.remove_positioner_button.setVisible(len(self.remove_auxiliary) != 0)

    def addPositioner(self, name):
        positioner = self.instrument.positioners[name]
        self.instrument.positioning_stack.addPositioner(positioner)
        self.parent_model.updateInstrumentScene()

        widget = self.createPositionerWidget(positioner, True)
        count = self.main_layout.count()
        self.main_layout.insertWidget(count-2, widget)

        self.add_auxiliary.remove(name)
        self.remove_auxiliary.append(name)
        self.__updateButtons()

    def changePositionerBase(self, name):
        matrix = self.parent.presenter.importTransformMatrix()
        if matrix is None:
            return

        positioner = self.instrument.positioners[name]
        self.instrument.positioning_stack.changeBaseMatrix(positioner, matrix)
        self.parent_model.updateInstrumentScene()
        reset_button = self.sender().property('reset')
        reset_button.setVisible(True)

    def resetPositionerBase(self, name):
        positioner = self.instrument.positioners[name]
        self.instrument.positioning_stack.changeBaseMatrix(positioner, reset=True)
        self.parent_model.updateInstrumentScene()
        self.sender().setVisible(False)

    def executeButtonClicked(self):
        end_q = []
        for form in self.positioner_forms.values():
            for control in form.form_controls:
                if control.property('link').type == Link.Type.Revolute:
                    end_q.append(math.radians(control.value))
                else:
                    end_q.append(control.value)

        positioner = self.instrument.positioning_stack
        q = positioner.configuration
        self.parent_model.animateInstrument(positioner.fkine,
                                            q, end_q, 500, 10)
