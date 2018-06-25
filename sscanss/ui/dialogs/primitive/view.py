from PyQt5 import QtCore, QtWidgets
from sscanss.core.util import Primitives, CompareOperator
from sscanss.ui.widgets import FormGroup, FormControl


class InsertPrimitiveDialog(QtWidgets.QDockWidget):
    formSubmitted = QtCore.pyqtSignal(Primitives, dict)

    def __init__(self, primitive, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = self.parent.presenter.model

        self._primitive = primitive
        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.formSubmitted.connect(parent.presenter.addPrimitive)

        self.minimum = 0
        self.maximum = 10000

        self.createForm()

    @property
    def primitive(self):
        return self._primitive

    @primitive.setter
    def primitive(self, value):
        if self._primitive != value:
            self._primitive = value
            self.createForm()

    def createForm(self):
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setSpacing(1)

        self.textboxes = {}
        name = self.parent_model.create_unique_key(self._primitive.value)
        self.mesh_args = {'name': name}
        if self._primitive == Primitives.Tube:
            self.mesh_args.update({'outer_radius': 100.000, 'inner_radius': 50.000, 'height': 200.000})
        elif self._primitive == Primitives.Sphere:
            self.mesh_args.update({'radius': 100.000})
        elif self._primitive == Primitives.Cylinder:
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

        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(self.main_layout)
        self.setWidget(main_widget)

        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.setWindowTitle('Insert {}'.format(self._primitive.value))
        self.setMinimumWidth(350)
        self.textboxes['name'].setFocus()

    def createPrimitiveSwitcher(self):
        switcher_layout = QtWidgets.QHBoxLayout()
        switcher = QtWidgets.QToolButton()
        switcher.setArrowType(QtCore.Qt.DownArrow)
        switcher.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        switcher.setStyleSheet('::menu-indicator { image: none; }')
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
                control.range(self.minimum, self.maximum, min_inclusive=True)

            self.textboxes[key] = control
            control.validate()
            self.form_group.addControl(control)

        if self._primitive == Primitives.Tube:
            outer_radius = self.textboxes['outer_radius']
            inner_radius = self.textboxes['inner_radius']

            outer_radius.compareWith(inner_radius, CompareOperator.Greater)
            inner_radius.compareWith(outer_radius, CompareOperator.Less)

        self.main_layout.addWidget(self.form_group)
        self.form_group.groupValidation.connect(self.formValidation)

    def formValidation(self, is_valid):
        if is_valid:
            self.create_primitive_button.setEnabled(True)
        else:
            self.create_primitive_button.setDisabled(True)

    def createPrimiviteButtonClicked(self):
        for key, textbox in self.textboxes.items():
            value = textbox.value if key == 'name' else float(textbox.value)
            self.mesh_args[key] = value

        self.formSubmitted.emit(self._primitive, self.mesh_args)
        new_name = self.parent_model.create_unique_key(self._primitive.value)
        self.textboxes['name'].value = new_name
