from PyQt5 import QtWidgets, QtGui, QtCore
from sscanss.ui.widgets import ColourPicker, create_scroll_area, create_header, FilePicker
from sscanss.core.util import Attributes
from sscanss.config import settings


class Preferences(QtWidgets.QDialog):
    """Provides UI for modifying global and project specific preferences

    :param parent: Main window
    :type parent: MainWindow
    """
    prop_name = 'key-value'

    def __init__(self, parent):
        super().__init__(parent)

        project_created = parent.presenter.model.project_data is not None

        self.changed_settings = {}
        self.group = []
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.main_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.main_layout)

        self.category_list = QtWidgets.QTreeWidget()
        self.main_layout.addWidget(self.category_list)
        self.category_list.setFixedWidth(150)
        self.category_list.header().setVisible(False)
        self.category_list.currentItemChanged.connect(self.changePage)
        self.category_list.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.stack = QtWidgets.QStackedLayout()
        self.main_layout.addLayout(self.stack)
        self.createForms()

        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setObjectName('DropDownButton')
        reset_action = QtWidgets.QAction('Reset', self)
        reset_action.triggered.connect(self.resetToDefaults)
        reset_default_action = QtWidgets.QAction('Reset Default', self)
        reset_default_action.triggered.connect(lambda: self.resetToDefaults(True))
        if project_created:
            self.reset_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
            self.reset_button.addActions([reset_action, reset_default_action])
            self.reset_button.setDefaultAction(reset_action)
        else:
            self.reset_button.setText(reset_default_action.text())
            self.reset_button.clicked.connect(reset_default_action.trigger)

        self.accept_button = QtWidgets.QToolButton()
        self.accept_button.setObjectName('DropDownButton')

        accept_action = QtWidgets.QAction('Accept', self)
        accept_action.triggered.connect(self.accept)
        set_default_action = QtWidgets.QAction('Set As Default', self)
        set_default_action.triggered.connect(lambda: self.accept(True))
        if project_created:
            self.accept_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
            self.accept_button.addActions([accept_action, set_default_action])
            self.accept_button.setDefaultAction(accept_action)
        else:
            self.accept_button.setText(set_default_action.text())
            self.accept_button.clicked.connect(set_default_action.trigger)
        self.accept_button.setDisabled(True)

        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setDefault(True)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setWindowTitle('Preferences')
        self.setMinimumSize(640, 480)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def createForms(self):
        self.generalForm()
        self.graphicsForm()
        self.simulationForm()

    def addGroup(self, group):
        QtWidgets.QTreeWidgetItem(self.category_list, [group.value])
        self.group.append(group)

    def simulationForm(self):
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
        self.addGroup(settings.Group.General)

        frame = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Custom_Instruments_Path
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
        self.addGroup(settings.Group.Graphics)

        frame = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addWidget(create_header('Rendering Size'))

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Fiducial_Size
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Fiducials:'))
        size = (5, 15, 30)
        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(['Small', 'Medium', 'Large'])
        combo_box.setProperty(self.prop_name, (key, value))
        combo_box.setCurrentIndex(size.index(value) if value in size else 0)
        combo_box.currentIndexChanged.connect(lambda i, v=size: self.changeSetting(v[i]))
        layout.addWidget(combo_box)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addSpacing(5)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Measurement_Size
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Points:'))
        size = (5, 15, 30)
        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(['Small', 'Medium', 'Large'])
        combo_box.setProperty(self.prop_name, (key, value))
        combo_box.setCurrentIndex(size.index(value) if value in size else 0)
        combo_box.currentIndexChanged.connect(lambda i, v=size: self.changeSetting(v[i]))
        layout.addWidget(combo_box)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addSpacing(5)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Vector_Size
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Measurement Vectors:'))
        size = (10, 35, 50)
        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(['Small', 'Medium', 'Large'])
        combo_box.setProperty(self.prop_name, (key, value))
        combo_box.setCurrentIndex(size.index(value) if value in size else 0)
        combo_box.currentIndexChanged.connect(lambda i, v=size: self.changeSetting(v[i]))
        layout.addWidget(combo_box)
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
        index = 0 if group is None else self.group.index(group)
        item = self.category_list.topLevelItem(index)
        self.category_list.setCurrentItem(item)

    def changePage(self, item):
        index = self.category_list.indexOfTopLevelItem(item)
        self.stack.setCurrentIndex(index)

    def changeSetting(self, new_value):
        key, old_value = self.sender().property(self.prop_name)
        if old_value == new_value:
            self.changed_settings.pop(key, None)
        else:
            self.changed_settings[key] = new_value

        if self.changed_settings:
            self.accept_button.setEnabled(True)
        else:
            self.accept_button.setEnabled(False)

    def resetToDefaults(self, default=False):
        reset_undo_stack = True if settings.local else False
        settings.reset(default)
        if reset_undo_stack:
            self.parent().undo_stack.resetClean()
        self.notify()
        super().accept()

    def accept(self, default=False):
        reset_undo_stack = False
        for key, value in self.changed_settings.items():
            # For now general setting are considered as system settings
            if key.value.startswith(settings.Group.General.value):
                settings.system.setValue(key.value, value)
            else:
                settings.setValue(key, value, default)
                reset_undo_stack = True
        if reset_undo_stack:
            self.parent().undo_stack.resetClean()
        self.notify()
        super().accept()

    def notify(self):
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
