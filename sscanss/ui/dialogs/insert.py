from PyQt5 import QtCore, QtWidgets
from sscanss.core.util import Primitives, CompareOperator, DockFlag
from sscanss.ui.widgets import FormGroup, FormControl


class InsertPrimitiveDialog(QtWidgets.QWidget):
    formSubmitted = QtCore.pyqtSignal(Primitives, dict)

    def __init__(self, primitive, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = self.parent.presenter.model

        self.primitive = primitive
        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.formSubmitted.connect(parent.presenter.addPrimitive)

        self.minimum = 0
        self.maximum = 10000

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setSpacing(1)

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
        self.createprimitive_button = QtWidgets.QPushButton('Create')
        self.createprimitive_button.clicked.connect(self.createPrimiviteButtonClicked)
        button_layout.addWidget(self.createprimitive_button)
        button_layout.addStretch(1)

        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)

        self.setLayout(self.main_layout)

        self.title = 'Insert {}'.format(self.primitive.value)
        self.dock_flag = DockFlag.Upper
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
            self.createprimitive_button.setEnabled(True)
        else:
            self.createprimitive_button.setDisabled(True)

    def createPrimiviteButtonClicked(self):
        for key, textbox in self.textboxes.items():
            value = textbox.value
            self.mesh_args[key] = value

        self.formSubmitted.emit(self.primitive, self.mesh_args)
        new_name = self.parent_model.uniqueKey(self.primitive.value)
        self.textboxes['name'].value = new_name


class InsertPointDialog(QtWidgets.QWidget):
    def __init__(self, parent, transform_type=None):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.main_layout = QtWidgets.QVBoxLayout()
        unit = 'mm'
        self.form_group = FormGroup()
        self.x_axis = FormControl('X', 0.0, required=True, unit=unit)
        self.x_axis.range(0, 10000)
        self.y_axis = FormControl('Y', 0.0, required=True, unit=unit)
        self.y_axis.range(0, 10000)
        self.z_axis = FormControl('Z', 0.0, required=True, unit=unit)
        self.z_axis.range(0, 10000)
        self.form_group.addControl(self.x_axis)
        self.form_group.addControl(self.y_axis)
        self.form_group.addControl(self.z_axis)
        self.form_group.groupValidation.connect(self.formValidation)
        button_layout = QtWidgets.QHBoxLayout()
        self.execute_button = QtWidgets.QPushButton('Add Fiducial Point')
        self.execute_button.clicked.connect(self.executeButtonClicked)
        button_layout.addWidget(self.execute_button)
        button_layout.addStretch(1)

        self.main_layout.addWidget(self.form_group)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.title = 'Add Fiducial Point'
        self.dock_flag = DockFlag.Upper
        self.setMinimumWidth(350)

    def formValidation(self, is_valid):
        if is_valid:
            self.execute_button.setEnabled(True)
        else:
            self.execute_button.setDisabled(True)

    def executeButtonClicked(self):
        point = [self.x_axis.value, self.y_axis.value, self.z_axis.value]
        self.parent.presenter.addFiducial(point)
