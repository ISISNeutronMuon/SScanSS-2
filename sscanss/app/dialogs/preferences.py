from PyQt6 import QtWidgets, QtGui, QtCore
from sscanss.core.util import Attributes, ColourPicker, create_scroll_area, create_header, FilePicker, SliderTextInput
from sscanss.config import settings


class Preferences(QtWidgets.QDialog):
    """Creates a UI for modifying global and project specific preferences

    :param parent: main window instance
    :type parent: MainWindow
    """
    prop_name = 'key-value'

    def __init__(self, parent):
        super().__init__(parent)

        project_created = parent.presenter.model.project_data is not None

        self.changed_settings = {}
        self.group = []
        self.global_names = []
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.main_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.main_layout)

        self.category_list = QtWidgets.QTreeWidget()
        self.main_layout.addWidget(self.category_list)
        self.category_list.setFixedWidth(150)
        self.category_list.header().setVisible(False)
        self.category_list.currentItemChanged.connect(self.changePage)
        self.category_list.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)

        self.stack = QtWidgets.QStackedLayout()
        self.main_layout.addLayout(self.stack)
        self.createForms()

        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setObjectName('DropDownButton')
        reset_action = QtGui.QAction('Reset', self)
        reset_action.triggered.connect(self.resetToDefaults)
        reset_default_action = QtGui.QAction('Reset Default', self)
        reset_default_action.triggered.connect(lambda: self.resetToDefaults(True))
        if project_created:
            self.reset_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            self.reset_button.addActions([reset_action, reset_default_action])
            self.reset_button.setDefaultAction(reset_action)
        else:
            self.reset_button.setText(reset_default_action.text())
            self.reset_button.clicked.connect(reset_default_action.trigger)

        self.accept_button = QtWidgets.QToolButton()
        self.accept_button.setObjectName('DropDownButton')

        accept_action = QtGui.QAction('Accept', self)
        accept_action.triggered.connect(self.accept)
        set_default_action = QtGui.QAction('Set As Default', self)
        set_default_action.triggered.connect(lambda: self.accept(True))
        if project_created:
            self.accept_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            self.accept_button.addActions([accept_action, set_default_action])
            self.accept_button.setDefaultAction(accept_action)
        else:
            self.accept_button.setText(set_default_action.text())
            self.accept_button.clicked.connect(set_default_action.trigger)
        self.accept_button.setDisabled(True)

        self.cancel_button = QtWidgets.QPushButton('Cancel', objectName='BlueTextPushButton')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setDefault(True)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setWindowTitle('Preferences')
        self.setMinimumSize(640, 520)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    def createForms(self):
        """Creates the setting forms"""
        self.generalForm()
        self.graphicsForm()
        self.simulationForm()

    def addGroup(self, group):
        """Adds a group to tree widget

        :param: setting group
        :type: Setting.Group
        """
        QtWidgets.QTreeWidgetItem(self.category_list, [group.value])
        self.group.append(group)

    def simulationForm(self):
        """Creates the form inputs for simulation settings"""
        self.addGroup(settings.Group.Simulation)

        frame = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(10)

        layout = QtWidgets.QVBoxLayout()
        key = settings.Key.Skip_Zero_Vectors
        value = settings.value(key)
        layout.addWidget(create_header('Zero Measurement Vector:'))
        group = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout()
        group_layout.setContentsMargins(0, 0, 0, 0)
        radio_button_1 = QtWidgets.QRadioButton('Skip the measurement')
        radio_button_1.setChecked(value)
        radio_button_1.setProperty(self.prop_name, (key, value))
        radio_button_1.toggled.connect(lambda: self.changeSetting(True))
        radio_button_2 = QtWidgets.QRadioButton('Perform translation but no rotation')
        radio_button_2.setChecked(not value)
        radio_button_2.setProperty(self.prop_name, (key, value))
        radio_button_2.toggled.connect(lambda: self.changeSetting(False))
        group.setLayout(group_layout)
        group_layout.addWidget(radio_button_1)
        group_layout.addWidget(radio_button_2)
        layout.addWidget(group)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QVBoxLayout()
        key = settings.Key.Align_First
        value = settings.value(key)
        layout.addWidget(create_header('Execution Order:'))
        group = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout()
        group_layout.setContentsMargins(0, 0, 0, 0)
        radio_button_1 = QtWidgets.QRadioButton('Run alignments before next point', frame)
        radio_button_1.setChecked(value)
        radio_button_1.setProperty(self.prop_name, (key, value))
        radio_button_1.toggled.connect(lambda: self.changeSetting(True))
        radio_button_2 = QtWidgets.QRadioButton('Run next point before alignments', frame)
        radio_button_2.setChecked(not value)
        radio_button_2.setProperty(self.prop_name, (key, value))
        radio_button_2.toggled.connect(lambda: self.changeSetting(False))
        group.setLayout(group_layout)
        group_layout.addWidget(radio_button_1)
        group_layout.addWidget(radio_button_2)
        layout.addWidget(group)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        main_layout.addWidget(create_header('Inverse Kinematics'))
        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Position_Stop_Val
        value = settings.value(key)
        lim = settings.default(key).limits
        layout.addWidget(QtWidgets.QLabel('Position termination tolerance (mm): '))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(*lim)
        spin.setValue(settings.value(key))
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)

        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Angular_Stop_Val
        value = settings.value(key)
        lim = settings.default(key).limits
        layout.addWidget(QtWidgets.QLabel('Orientation termination tolerance (degrees): '))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(*lim)
        spin.setValue(settings.value(key))
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)

        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Global_Max_Eval
        value = settings.value(key)
        lim = settings.default(key).limits
        layout.addWidget(QtWidgets.QLabel(f'Number of evaluations for global optimization ({lim[0]} - {lim[1]}): '))
        spin = QtWidgets.QSpinBox()
        spin.setRange(*lim)
        spin.setValue(settings.value(key))
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)

        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Local_Max_Eval
        value = settings.value(key)
        lim = settings.default(key).limits
        layout.addWidget(QtWidgets.QLabel(f'Number of evaluations for local optimization ({lim[0]} - {lim[1]}): '))
        spin = QtWidgets.QSpinBox()
        spin.setRange(*lim)
        spin.setValue(value)
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)
        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addStretch(1)

        frame.setLayout(main_layout)
        self.stack.addWidget(create_scroll_area(frame))

    def generalForm(self):
        """Creates the form inputs for general settings"""
        self.addGroup(settings.Group.General)

        frame = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Check_Update
        self.global_names.append(key)
        value = settings.value(key)
        checkbox = QtWidgets.QCheckBox('Check for updates on startup')
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(lambda ignore, c=checkbox: self.changeSetting(c.isChecked()))
        checkbox.setProperty(self.prop_name, (key, value))
        layout.addWidget(checkbox)
        main_layout.addLayout(layout)
        main_layout.addSpacing(10)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Custom_Instruments_Path
        self.global_names.append(key)
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Custom Instruments: '))
        path_picker = FilePicker(value, select_folder=True)
        path_picker.setProperty(self.prop_name, (key, value))
        path_picker.value_changed.connect(self.changeSetting)
        layout.addWidget(path_picker)
        main_layout.addLayout(layout)

        main_layout.addStretch(1)
        frame.setLayout(main_layout)
        self.stack.addWidget(create_scroll_area(frame))

    def graphicsForm(self):
        """Creates form inputs for graphics settings"""
        self.addGroup(settings.Group.Graphics)

        frame = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(10)

        main_layout.addWidget(create_header('Rendering Size'))

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Fiducial_Size
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Fiducials:'))
        fiducials = SliderTextInput(self, value, settings.default(key).limits)
        fiducials.slider.setProperty(self.prop_name, (key, value))
        fiducials.slider.valueChanged.connect(self.changeSetting)
        layout.addWidget(fiducials)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addSpacing(5)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Measurement_Size
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Points:'))
        measurement_points = SliderTextInput(self, value, settings.default(key).limits)
        measurement_points.slider.setProperty(self.prop_name, (key, value))
        measurement_points.slider.valueChanged.connect(self.changeSetting)
        layout.addWidget(measurement_points)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addSpacing(5)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Vector_Size
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Vectors:'))
        measurement_vectors = SliderTextInput(self, value, settings.default(key).limits)
        measurement_vectors.slider.setProperty(self.prop_name, (key, value))
        measurement_vectors.slider.valueChanged.connect(self.changeSetting)
        layout.addWidget(measurement_vectors)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addSpacing(5)

        main_layout.addWidget(create_header('Rendering Colour'))
        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Sample_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Sample:'))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Fiducial_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Fiducials (Enabled): '))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Fiducial_Disabled_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Fiducials (Disabled): '))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Measurement_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Point (Enabled): '))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Measurement_Disabled_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Point (Disabled): '))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Vector_1_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Vector 1: '))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Vector_2_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Vector 2: '))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Cross_Sectional_Plane_Colour
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Cross-Sectional Plane: '))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        main_layout.addStretch(1)
        frame.setLayout(main_layout)
        self.stack.addWidget(create_scroll_area(frame))

    def setActiveGroup(self, group):
        """Shows the form widget for the given group

        :param: setting group
        :type: Setting.Group
        """
        index = 0 if group is None else self.group.index(group)
        item = self.category_list.topLevelItem(index)
        self.category_list.setCurrentItem(item)

    def changePage(self, item):
        """Changes the page to that of the clicked item

        :param: widget tree item
        :type: QtWidgets.QTreeWidgetItem
        """
        index = self.category_list.indexOfTopLevelItem(item)
        self.stack.setCurrentIndex(index)

    def changeSetting(self, new_value):
        """Changes the value of the setting associated with the calling widget

        :param new_value: new value
        :type new_value: Any
        """
        key, old_value = self.sender().property(self.prop_name)
        if old_value == new_value:
            self.changed_settings.pop(key, None)
        else:
            self.changed_settings[key] = new_value

        if self.changed_settings:
            self.accept_button.setEnabled(True)
        else:
            self.accept_button.setEnabled(False)

    def resetToDefaults(self, set_global=False):
        """Resets the settings to default values

        :param set_global: indicates the global setting should also be reset
        :type set_global: bool
        """
        reset_undo_stack = True if settings.local else False
        settings.reset(set_global)
        if reset_undo_stack:
            self.parent().undo_stack.resetClean()
        self.notify()
        super().accept()

    def accept(self, set_global=False):
        """Saves the changes made to setting

        :param set_global: indicates the global setting should also be set
        :type set_global: bool
        """
        reset_undo_stack = False
        for key, value in self.changed_settings.items():
            # For now general setting are considered as system settings
            if key in self.global_names:
                settings.system.setValue(key.value, value)
            else:
                settings.setValue(key, value, set_global)
                reset_undo_stack = True
        if reset_undo_stack:
            self.parent().undo_stack.resetClean()
        self.notify()
        super().accept()

    def notify(self):
        """Notifies the listeners of setting changes"""
        view = self.parent()
        model = view.presenter.model
        model.updateInstrumentList()
        view.updateChangeInstrumentMenu()

        if model.project_data is None:
            return

        model.notifyChange(Attributes.Sample)
        model.notifyChange(Attributes.Fiducials)
        model.notifyChange(Attributes.Measurements)
        model.notifyChange(Attributes.Vectors)
