from enum import Enum, unique
from PyQt5 import QtCore, QtWidgets, QtGui
from sscanss.core.util import Primitives


@unique
class Compare(Enum):
    Equal = 1
    Not_Equal = 2
    Greater = 3
    Less = 4


def to_float(text):
    try:
        value = float(text)
    except ValueError:
        value = None
    return value


class InsertPrimitiveDialog(QtWidgets.QDockWidget):
    formSubmitted = QtCore.pyqtSignal(Primitives, dict)

    def __init__(self, primitive, parent):
        super().__init__(parent)

        self.primitive = primitive

        self.value_too_large = '{} should be less than {}{}.'
        self.value_too_small = '{} should be greater than {}{}.'
        self.value_is_empty = '{} should not be empty.'
        self.validation_errors = {}

        self.minimum = 0
        self.maximum = 10000

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setSpacing(10)

        self.validator = QtGui.QDoubleValidator()
        self.validator.setDecimals(3)
        self.validator.setNotation(QtGui.QDoubleValidator.StandardNotation)

        self.validator_label = QtWidgets.QLabel('')
        self.validator_label.setObjectName('Error')
        self.main_layout.addWidget(self.validator_label)

        self.textboxes = {}
        name = parent.presenter.model.create_unique_key(self.primitive.value)
        self.mesh_args = {'name': name}

        if self.primitive == Primitives.Tube:
            self.mesh_args.update({'outer_radius': 100.000, 'inner_radius': 50.000, 'height': 200.000})
        elif self.primitive == Primitives.Sphere:
            self.mesh_args.update({'radius': 100.000})
        elif self.primitive == Primitives.Cylinder:
            self.mesh_args.update({'radius': 100.000, 'height': 200.000})
        else:
            self.mesh_args.update({'width': 50.000, 'height': 100.000, 'depth': 200.000})

        self.createForm()

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
        self.setWindowTitle('Insert {}'.format(primitive.value))
        self.setMinimumWidth(350)

        self.formSubmitted.connect(parent.presenter.addPrimitive)

    def createForm(self):
        for key, value in self.mesh_args.items():
            pretty_label = key.replace('_', ' ').title()
            label = QtWidgets.QLabel('{}:'.format(pretty_label))

            textbox = QtWidgets.QLineEdit(str(value))
            textbox.textChanged.connect(self.inputValidation)

            # Line-edit properties are used to facilitate input validation
            textbox.setProperty('key', key)
            textbox.setProperty('label', pretty_label)
            textbox.setProperty('required', True)  # All inputs are required
            if key != 'name':
                textbox.setValidator(self.validator)
                textbox.setProperty('number', True)  # All except name are numbers
                textbox.setProperty('range', True)  # max and min range check

            self.main_layout.addWidget(label)
            self.main_layout.addWidget(textbox)
            self.textboxes[key] = textbox

        if self.primitive == Primitives.Tube:
            outer_radius = self.textboxes['outer_radius']
            inner_radius = self.textboxes['inner_radius']

            # This properties are used to perform comparision validation
            # i.e. outer radius of a tube should be greater than inner
            outer_radius.setProperty('compare', True)
            outer_radius.setProperty('compare_with', inner_radius)
            outer_radius.setProperty('compare_op', Compare.Greater)
            inner_radius.setProperty('compare', True)
            inner_radius.setProperty('compare_with', outer_radius)
            inner_radius.setProperty('compare_op', Compare.Less)

    def inputValidation(self, input_text):

        textbox = self.sender()
        key = textbox.property('key')
        input_label = textbox.property('label')
        required = textbox.property('required')
        number = textbox.property('number')
        check_range = textbox.property('range')
        compare = textbox.property('compare')

        if required and not input_text:
            self.validation_errors[key] = self.value_is_empty.format(input_label)
            self.onInvalid(textbox)
            return
        else:
            self.validation_errors.pop(key, None)

        if number:
            value = to_float(input_text)
            if value is None:
                return

            if check_range and value <= self.minimum:
                self.validation_errors[key] = self.value_too_small.format(input_label, self.minimum, 'mm')
                self.onInvalid(textbox)
                return
            else:
                self.validation_errors.pop(key, None)

            if check_range and value > self.maximum:
                self.validation_errors[key] = self.value_too_large.format(input_label, self.maximum, 'mm')
                self.onInvalid(textbox)
                return
            else:
                self.validation_errors.pop(key, None)

            if compare:
                textbox_2 = textbox.property('compare_with')
                label_2 = textbox_2.property('label')
                compare_text = textbox_2.text()
                op = textbox.property('compare_op')
                if compare_text:
                    value_2 = to_float(compare_text)
                    if value_2 is None:
                        return
                    if op == Compare.Less and value >= value_2:
                        self.validation_errors['compare'] = self.value_too_large.format(input_label, label_2, '')
                        self.onInvalid([textbox, textbox_2])
                        return
                    elif op == Compare.Greater and value <= value_2:
                        self.validation_errors['compare'] = self.value_too_small.format(input_label, label_2, '')
                        self.onInvalid([textbox, textbox_2])
                        return

                    self.validation_errors.pop('compare', None)
                    textbox_2.setStyleSheet('')

        textbox.setStyleSheet('')
        self.onValid()
        return

    def loadErrorStyleSheet(self, textboxes):
        try:
            textboxes.setStyleSheet('border: 1px solid red;')
        except AttributeError:
            for textbox in textboxes:
                textbox.setStyleSheet('border: 1px solid red;')

    def onInvalid(self, textboxes):
        self.loadErrorStyleSheet(textboxes)

        self.validator_label.setText('\n\n'.join([val for _, val in self.validation_errors.items()]))
        self.create_primitive_button.setDisabled(True)

    def onValid(self,):
        self.validator_label.setText('\n\n'.join([val for _, val in self.validation_errors.items()]))
        if self.validation_errors:
                return
        self.create_primitive_button.setDisabled(False)

    def createPrimiviteButtonClicked(self):

        for key, textbox in self.textboxes.items():
            value = textbox.text() if key == 'name' else float(textbox.text())

            self.mesh_args[key] = value

        self.formSubmitted.emit(self.primitive, self.mesh_args)
