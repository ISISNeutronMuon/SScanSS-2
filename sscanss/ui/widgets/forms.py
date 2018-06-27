from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.util import CompareOperator, to_float


class FormGroup(QtWidgets.QWidget):
    groupValidation = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        self.form_controls = []
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

    def addControl(self, control):
        if type(control) == FormControl:
            self.form_controls.append(control)
            self.main_layout.addWidget(control)
            control.inputValidation.connect(self.validateGroup)
        else:
            raise ValueError('could not add object of type {}'.format(type(control)))

    def validateGroup(self, _):
        for control in self.form_controls:
            if not control.valid:
                self.groupValidation.emit(False)
                return

        self.groupValidation.emit(True)


class FormControl(QtWidgets.QWidget):
    inputValidation = QtCore.pyqtSignal(bool)

    def __init__(self, title, value='', unit=None, required=False):
        super().__init__()

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.form_label = QtWidgets.QLabel()
        self.form_control = QtWidgets.QLineEdit(str(value))
        self.validation_label = QtWidgets.QLabel()
        self.validation_label.setStyleSheet('color: red')
        self._validator = None

        control_layout.addWidget(self.form_label)
        control_layout.addWidget(self.form_control)
        control_layout.addWidget(self.validation_label)

        self.setLayout(control_layout)

        # Validation Flags and Errors
        if unit:
            self.title = (title, unit)
        else:
            self.title = title

        self.required = required
        self.required_error = '{} is required.'

        self._number = False
        self.number_error = '{} should be a number.'

        self._minimum = None
        self._min_inclusive = False
        self.min_error = '{} should be higher than {}.'
        self._maximum = None
        self._max_inclusive = False
        self.max_error = '{} should be lower than {}.'

        self.compare_with = None
        self.compare_op = CompareOperator.Equal
        self.compare_error = 'Could not compare {} with {}.'
        self.compare_equality_error = '{} should be equal to {}.'
        self.compare_notequal_error = '{} should not be equal to {}.'
        self.compare_greater_error = '{} should be greater than {}.'
        self.compare_less_error = '{} should be less than {}.'

        self.valid = False

        self.form_control.textChanged.connect(self.validate)
        self.validate()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        try:
            self._title, units = title
            self.form_label.setText('{} ({}):'.format(self._title, units))
        except ValueError:
            self._title = title
            self.form_label.setText('{}:'.format(self._title))

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, value):
        self._number = value
        if self._number:
            self._validator = QtGui.QDoubleValidator()
            self._validator.setDecimals(3)
            self._validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            self.form_control.setValidator(self._validator)

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        self._maximum = value
        self.number = True

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, value):
        self._minimum = value
        self.number = True

    def range(self, minimum, maximum, min_inclusive=False, max_inclusive=False):
        self.minimum = minimum
        self.maximum = maximum
        self._max_inclusive = max_inclusive
        self._min_inclusive = min_inclusive

    @property
    def value(self):
        text = self.form_control.text()
        if not self.number:
            return text

        val, ok = to_float(text)
        if not ok:
            raise ValueError('FormControl value is not a number')

        return val

    @value.setter
    def value(self, value):
        return self.form_control.setText(str(value))

    def compareWith(self, form_control, operation):
        if type(form_control) == FormControl:
            self.compare_op = operation
            self.compare_with = form_control
            self.form_control.textChanged.connect(lambda ignore: form_control.validate())
            if self.compare_op == CompareOperator.Equal:
                self.compare_error = self.compare_equality_error.format(self.title, self.compare_with.title)
            if self.compare_op == CompareOperator.Not_Equal:
                self.compare_error = self.compare_notequal_error.format(self.title, self.compare_with.title)
            if self.compare_op == CompareOperator.Greater:
                self.compare_error = self.compare_greater_error.format(self.title, self.compare_with.title)
            if self.compare_op == CompareOperator.Less:
                self.compare_error = self.compare_less_error.format(self.title, self.compare_with.title)
            else:
                self.compare_error = self.compare_error.format(self.title, self.compare_with.title)
        else:
            raise ValueError('could not add object of type {}'.format(type(form_control)))

    def validate(self, input_text=None):
        text = self.value.strip()if input_text is None else input_text.strip()

        if self.required and not text:
            self.isInvalid(self.required_error.format(self.title))
            return

        if not self.rangeValid(text):
            return

        if not self.compareValid(text):
            return

        self.isValid()

    def rangeValid(self, input_text):
        if not self.number:
            return True

        value, ok = to_float(input_text)
        if not ok:
            self.isInvalid(self.number_error.format(self.title))
            return False

        max_logic = None
        if self.maximum is not None and self._max_inclusive:
            max_logic = value >= self.maximum
        elif self.maximum is not None and not self._max_inclusive:
            max_logic = value > self.maximum

        if max_logic:
            self.isInvalid(self.max_error.format(self.title, self.maximum))
            return False

        min_logic = None
        if self.minimum is not None and self._min_inclusive:
            min_logic = value <= self.minimum
        elif self.minimum is not None and not self._min_inclusive:
            min_logic = value < self.minimum
        if min_logic:
            self.isInvalid(self.min_error.format(self.title, self.minimum))
            return False

        return True

    def compareValid(self, input_text):
        if self.compare_with is None:
            return True

        input_text_2 = self.compare_with.value.strip()

        if self.number:
            value, ok = to_float(input_text)
            value_2, ok_2 = to_float(input_text_2)
            if not ok or not ok_2:
                self.isInvalid(self.compare_error)
                return False
        else:
            value = input_text
            value_2 = input_text_2

        if self.compare_op == CompareOperator.Equal and value != value_2:
            self.isInvalid(self.compare_equality_error.format(self.title, self.compare_with.title))
            return False
        if self.compare_op == CompareOperator.Not_Equal and value == value_2:
            self.isInvalid(self.compare_notequal_error.format(self.title, self.compare_with.title))
            return False
        if self.compare_op == CompareOperator.Greater and value <= value_2:
            self.isInvalid(self.compare_greater_error.format(self.title, self.compare_with.title))
            return False
        if self.compare_op == CompareOperator.Less and value >= value_2:
            self.isInvalid(self.compare_less_error.format(self.title, self.compare_with.title))
            return False

        return True

    def isValid(self):
        self.form_control.setStyleSheet('')
        self.validation_label.setText('')
        self.valid = True
        self.inputValidation.emit(True)

    def isInvalid(self, error):
        self.form_control.setStyleSheet('border: 1px solid red;')
        self.validation_label.setText(error)
        self.valid = False
        self.inputValidation.emit(False)

    def setFocus(self):
        self.form_control.setFocus()
