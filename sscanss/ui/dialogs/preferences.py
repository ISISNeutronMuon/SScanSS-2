from PyQt5 import QtWidgets, QtGui, QtCore
from sscanss.ui.widgets import ColourPicker
from sscanss.config import settings


class Preferences(QtWidgets.QDialog):
    prop_name = 'key-value'

    def __init__(self, group, parent):
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
        self.category_list.currentItemChanged.connect(self.switchCategory)
        self.category_list.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.stack = QtWidgets.QStackedLayout()
        self.main_layout.addLayout(self.stack)
        self.createForms()
        index = 0 if group is None else self.group.index(group)
        self.category_list.topLevelItem(index).setSelected(True)
        self.stack.setCurrentIndex(index)

        self.default_button = QtWidgets.QPushButton('Reset to Default')
        self.default_button.clicked.connect(self.resetToDefaults)
        self.accept_button = QtWidgets.QPushButton('Accept')
        self.accept_button.setDisabled(True)
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button = QtWidgets.QPushButton('Close')
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

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Order:'))
        combo = QtWidgets.QComboBox()
        combo.addItems(['Run alignments before next point', 'Run alignments after all points'])
        layout.addWidget(combo)
        layout.addStretch(1)
        main_layout.addLayout(layout)

        main_layout.addWidget(QtWidgets.QLabel('Inverse Kinematics'))
        layout = QtWidgets.QHBoxLayout()
        key = settings.Key.Stop_Val
        value = settings.value(key)
        layout.addWidget(QtWidgets.QLabel('Termination tolerance (0.000 - 10.000): '))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.000, 10.000)
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
        self.stack.addWidget(frame)

    def addGroup(self, group):
        QtWidgets.QTreeWidgetItem(self.category_list, [group.value])
        self.group.append(group)

    def graphicsForm(self):
        self.addGroup(settings.Group.Graphics)

        frame = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addWidget(QtWidgets.QLabel('Rendering Colour'))
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
        layout.addWidget(QtWidgets.QLabel('Fiducials'))
        colour_picker = ColourPicker(QtGui.QColor.fromRgbF(*value))
        colour_picker.value_changed.connect(self.changeSetting)
        colour_picker.setProperty(self.prop_name, (key, value))
        layout.addWidget(colour_picker)
        layout.addStretch(1)
        main_layout.addLayout(layout)
        main_layout.addStretch(1)
        frame.setLayout(main_layout)
        self.stack.addWidget(frame)

    def switchCategory(self, item):
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

    def resetToDefaults(self):
        settings.reset()
        super().accept()

    def accept(self):
        for key, value in self.changed_settings.items():
            settings.setValue(key, value)
        super().accept()
