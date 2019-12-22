from PyQt5 import QtWidgets, QtGui, QtCore
from sscanss.ui.widgets import ColourPicker, create_scroll_area, create_header
from sscanss.core.util import Attributes
from sscanss.config import settings


class Preferences(QtWidgets.QDialog):
    prop_name = 'key-value'

    def __init__(self, parent):
        super().__init__(parent)

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

        self.default_button = QtWidgets.QToolButton()
        self.default_button.setObjectName('DropDownButton')
        self.default_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        reset_action = QtWidgets.QAction('Reset', self)
        reset_action.triggered.connect(self.resetToDefaults)
        reset_default_action = QtWidgets.QAction('Reset Default', self)
        reset_default_action.triggered.connect(lambda: self.resetToDefaults(True))
        self.default_button.addActions([reset_action, reset_default_action])
        self.default_button.setDefaultAction(reset_action)

        self.accept_button = QtWidgets.QToolButton()
        self.accept_button.setObjectName('DropDownButton')
        self.accept_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        accept_action = QtWidgets.QAction('Accept', self)
        accept_action.triggered.connect(self.accept)
        set_default_action = QtWidgets.QAction('Set As Default', self)
        set_default_action.triggered.connect(lambda: self.accept(True))
        self.accept_button.addActions([accept_action, set_default_action])
        self.accept_button.setDefaultAction(accept_action)
        self.accept_button.setDisabled(True)

        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setDefault(True)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.default_button)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setWindowTitle('Preferences')
        self.setMinimumSize(640, 480)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)

    def createForms(self):
        self.graphicsForm()
        self.simulationForm()

    def simulationForm(self):
        self.addGroup(settings.Group.Simulation)

        frame = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(15)

        layout = QtWidgets.QVBoxLayout()
        key = settings.Key.Align_First
        value = settings.value(key)
        layout.addWidget(create_header('Execution Order:'))
        radio_button_1 = QtWidgets.QRadioButton('Run alignments before next point', frame)
        radio_button_1.setChecked(value)
        radio_button_1.setProperty(self.prop_name, (key, value))
        radio_button_1.toggled.connect(lambda: self.changeSetting(True))

        radio_button_2 = QtWidgets.QRadioButton('Run next point before alignments', frame)
        radio_button_2.setChecked(not value)
        radio_button_2.setProperty(self.prop_name, (key, value))
        radio_button_2.toggled.connect(lambda: self.changeSetting(False))
        layout.addWidget(radio_button_1)
        layout.addWidget(radio_button_2)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        main_layout.addWidget(create_header('Inverse Kinematics'))
        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Position_Stop_Val
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Position termination tolerance (mm): '))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setMinimum(0.000)
        spin.setValue(settings.value(key))
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)

        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Angular_Stop_Val
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Orientation termination tolerance (degrees): '))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.000, 360.000)
        spin.setValue(settings.value(key))
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)

        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Global_Max_Eval
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Number of evaluations for global optimization (50 - 500): '))
        spin = QtWidgets.QSpinBox()
        spin.setRange(50, 500)
        spin.setValue(settings.value(key))
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)

        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Local_Max_Eval
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Number of evaluations for local optimization (500 - 5000): '))
        spin = QtWidgets.QSpinBox()
        spin.setRange(500, 5000)
        spin.setValue(value)
        spin.setProperty(self.prop_name, (key, value))
        spin.valueChanged.connect(self.changeSetting)
        layout.addWidget(spin)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addStretch(1)

        frame.setLayout(main_layout)
        self.stack.addWidget(create_scroll_area(frame))

    def addGroup(self, group):
        QtWidgets.QTreeWidgetItem(self.category_list, [group.value])
        self.group.append(group)

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
        settings.reset(default)
        self.parent().undo_stack.resetClean()
        self.notify()
        super().accept()

    def accept(self, default=False):
        for key, value in self.changed_settings.items():
            settings.setValue(key, value, default)
        self.parent().undo_stack.resetClean()
        self.notify()
        super().accept()

    def notify(self):
        model = self.parent().presenter.model
        if not model.project_data:
            return

        model.notifyChange(Attributes.Sample)
        model.notifyChange(Attributes.Fiducials)
        model.notifyChange(Attributes.Measurements)
        model.notifyChange(Attributes.Vectors)
