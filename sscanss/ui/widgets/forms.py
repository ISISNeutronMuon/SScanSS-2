from enum import Enum, unique
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.util import CompareOperator, to_float


class FormTitle(QtWidgets.QWidget):
    def __init__(self, text, divider=True, name='form-title'):
        super().__init__()
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

        self.title_layout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(text)
        self.label.setObjectName(name)
        self.title_layout.addWidget(self.label)
        self.title_layout.addStretch(1)
        self.main_layout.addLayout(self.title_layout)
        if divider:
            self.line = QtWidgets.QFrame()
            self.line.setFrameShape(QtWidgets.QFrame.HLine)
            self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.main_layout.addWidget(self.line)

    def addHeaderControl(self, control):
        self.title_layout.addWidget(control)


class FormGroup(QtWidgets.QWidget):
    groupValidation = QtCore.pyqtSignal(bool)

    @unique
    class Layout(Enum):
        Vertical = 0
        Horizontal = 1
        Grid = 2

    def __init__(self, layout=Layout.Vertical):
        """ Manages validation for a group of Form Controls """
        super().__init__()

        self.form_controls = []
        self.valid = False
        if layout == FormGroup.Layout.Vertical:
            self.main_layout = QtWidgets.QVBoxLayout()
        elif layout == FormGroup.Layout.Horizontal:
            self.main_layout = QtWidgets.QHBoxLayout()
        else:
            self.main_layout = QtWidgets.QGridLayout()

        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

    def addControl(self, control, extra_widgets=None):
        """ Adds a form control to group.

        :param control: control to add to group
        :type control: sscanss.ui.widgets.forms.FormControl
        """
        if type(control) != FormControl:
            raise ValueError('could not add object of type {}'.format(type(control)))

        self.form_controls.append(control)
        extra_widgets = [] if extra_widgets is None else extra_widgets
        if isinstance(self.main_layout, QtWidgets.QGridLayout):
            index = 2 * (len(self.form_controls) - 1)
            self.main_layout.addWidget(control.form_label, index, 0)
            self.main_layout.addWidget(control.form_control, index, 1)
            self.main_layout.addWidget(control.validation_label, index+1, 1)
            for i, widget in enumerate(extra_widgets):
                self.main_layout.addWidget(widget, index, i+2)
        else:
            self.main_layout.addWidget(control)
            for widget in extra_widgets:
                self.main_layout.addWidget(widget)

        control.inputValidation.connect(self.validateGroup)
        if len(self.form_controls) == 1 and control.valid:
            self.valid = True
        else:
            self.valid &= control.valid


    def validateGroup(self):
        """ Checks if all controls in the group are valid if so returns True

        :return: group validation state
        :rtype: bool
        """
        for control in self.form_controls:
            if not control.valid:
                self.valid = False
                self.groupValidation.emit(self.valid)
                return self.valid

        self.valid = True
        self.groupValidation.emit(self.valid)
        return self.valid


class FormControl(QtWidgets.QWidget):
    inputValidation = QtCore.pyqtSignal(bool)

    def __init__(self, title, value='', unit=None, required=False, number=False):
        """ Creates a form widget that provides input validation

        :param title: title to display in Label
        :type title: str
        :param value: input value
        :type value: Union[str, float]
        :param unit: Units to display in Label
        :type unit: Union[None, str]
        :param required: indicate if input is required
        :type required: bool
        """
        super().__init__()

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.form_label = QtWidgets.QLabel()
        self.form_control = QtWidgets.QLineEdit()
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

        self._number = number
        self.__addDoubleValidator(number)
        self.number_error = '{} should be a number.'

        self._minimum = None
        self._min_exclusive = False
        self.min_error = '{} should be higher or equal to {}.'
        self.min_exc_error = '{} should be higher than {}.'
        self._maximum = None
        self._max_exclusive = False
        self.max_error = '{} should be lower or equal to {}.'
        self.max_exc_error = '{} should be lower than {}.'

        self.compare_with = None
        self.compare_op = CompareOperator.Equal
        self.compare_error = 'Could not compare {} with {}.'
        self.compare_equality_error = '{} should be equal to {}.'
        self.compare_notequal_error = '{} should not be equal to {}.'
        self.compare_greater_error = '{} should be greater than {}.'
        self.compare_less_error = '{} should be less than {}.'

        self.valid = False

        self.value = value
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
        self.__addDoubleValidator(value)
        self.validate()

    def __addDoubleValidator(self, add=False):
        if add:
            self._validator = QtGui.QDoubleValidator()
            self._validator.setDecimals(3)
            self._validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            self.form_control.setValidator(self._validator)
        else:
            self.form_control.setValidator(None)

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        self._maximum = value
        if not self._number:
            self._number = True
            self.__addDoubleValidator(True)

        self.validate()

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, value):
        self._minimum = value
        if not self._number:
            self._number = True
            self.__addDoubleValidator(True)

        self.validate()

    def range(self, minimum, maximum, min_exclusive=False, max_exclusive=False):
        """ Sets a range within which the control's input must lie to be valid.
        By default, the minimum and maximum are included, but can be excluded using
        the corresponding exclusive flag.

        :param minimum: min value of the control
        :type minimum: numbers.Number
        :param maximum: max value of the control
        :type maximum: numbers.Number
        :param min_exclusive: indicates min should be excluded
        :type min_exclusive: bool
        :param max_exclusive: indicates min should be excluded
        :type max_exclusive: bool
        """
        self._minimum = minimum
        self._maximum = maximum
        self._max_exclusive = max_exclusive
        self._min_exclusive = min_exclusive
        if not self._number:
            self._number = True
            self.__addDoubleValidator(True)

        self.validate()

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
        if self.number:
            value = '{:.{decimal}f}'.format(value, decimal=self._validator.decimals())
        else:
            value = str(value)
        self.form_control.setText(value)

    @property
    def text(self):
        return self.form_control.text()

    @text.setter
    def text(self, value):
        self.form_control.setText(value)

    def compareWith(self, form_control, operation):
        """ Specifies which control's input to compare with this control's input
         and which comparision operation to perform. The comparision operation must
         pass for the control to be valid.

        :param form_control: control to compare with
        :type form_control: sscanss.ui.widgets.forms.FormControl
        :param operation: comparision operation to perform
        :type operation: sscanss.core.util.misc.CompareOperator
        """
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

            self.validate()
        else:
            raise ValueError('could not add object of type {}'.format(type(form_control)))

    def validate(self, input_text=None):
        """ Performs validation for a specified text or the text in the control.
        The validation performed (required, compare, range, or number) is dependent on
        which flags have bee set.

        :param input_text: text to validate
        :type input_text: Union[str, None]
        """
        text = self.text.strip() if input_text is None else input_text.strip()

        if self.required and not text:
            self.isInvalid(self.required_error.format(self.title))
            return

        if not self.rangeValid(text):
            return

        if not self.compareValid(text):
            return

        self.isValid()

    def rangeValid(self, input_text):
        """ Performs range checks on the control's input

        :param input_text: text to validate
        :type input_text: str
        :return: indicates if the control passed checks
        :rtype: bool
        """
        if not self.number:
            return True

        value, ok = to_float(input_text)
        if not ok:
            self.isInvalid(self.number_error.format(self.title))
            return False

        max_logic = None
        max_error = ''
        if self.maximum is not None and self._max_exclusive:
            max_logic = value >= self.maximum
            max_error = self.max_exc_error
        elif self.maximum is not None and not self._max_exclusive:
            max_logic = value > self.maximum
            max_error = self.max_error

        if max_logic:
            self.isInvalid(max_error.format(self.title, self.maximum))
            return False

        min_logic = None
        min_error = ''
        if self.minimum is not None and self._min_exclusive:
            min_logic = value <= self.minimum
            min_error = self.min_exc_error
        elif self.minimum is not None and not self._min_exclusive:
            min_logic = value < self.minimum
            min_error = self.min_error
        if min_logic:
            self.isInvalid(min_error.format(self.title, self.minimum))
            return False

        return True

    def compareValid(self, input_text):
        """ Compares the control's input with another specified control

        :param input_text: text to validate
        :type input_text: str
        :return: indicates if the control passed checks
        :rtype: bool
        """
        if self.compare_with is None:
            return True

        input_text_2 = self.compare_with.text.strip()

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
        """ Puts the control to an valid state """
        self.form_control.setStyleSheet('')
        self.validation_label.setText('')
        self.valid = True
        self.inputValidation.emit(True)

    def isInvalid(self, error):
        """ Puts the control to an invalid state

        :param error: error message
        :type error: str
        """
        self.form_control.setStyleSheet('border: 1px solid red;')
        self.validation_label.setText(error)
        self.valid = False
        self.inputValidation.emit(False)

    def setFocus(self):
        """ Sets the focus on this control """
        self.form_control.setFocus()
