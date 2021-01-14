from abc import ABC, abstractmethod
from enum import Enum, unique
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.util import to_float


class Validator(ABC):
    """base class for form control validators """
    @abstractmethod
    def valid(self):
        """return valid state of the control"""


class RequiredValidator(Validator):
    def __init__(self, control):
        """validator for form controls with required inputs

        :param control: control to validate
        :type control: sscanss.ui.widgets.forms.FormControl
        """
        self.control = control
        self.error = '{} is required.'

    def valid(self):
        """validates control's input

        :return: indicates input are valid
        :rtype: bool
        """
        if not self.control.text.strip():
            self.control.isInvalid(self.error.format(self.control.title))
            return False
        return True


class RangeValidator(Validator):
    def __init__(self, control, minimum=None, maximum=None, min_exclusive=False, max_exclusive=False):
        """validator for form controls with required inputs

        :param control: control to validate
        :type control: sscanss.ui.widgets.forms.FormControl
        :param minimum: min value of the control
        :type minimum: Union[None, float]
        :param maximum: max value of the control
        :type maximum: Union[None, float]
        :param min_exclusive: min value of the control
        :type min_exclusive: bool
        :param max_exclusive: max value of the control
        :type max_exclusive: bool
        """
        self.control = control

        self.minimum = minimum
        self.maximum = maximum
        self.max_exclusive = max_exclusive
        self.min_exclusive = min_exclusive

        self.number_error = '{} should be a number.'
        self.min_error = '{} should be higher or equal to {}.'
        self.min_exc_error = '{} should be higher than {}.'
        self.max_error = '{} should be lower or equal to {}.'
        self.max_exc_error = '{} should be lower than {}.'

    def valid(self):
        """validates control's input

        :return: indicates input are valid
        :rtype: bool
        """
        if not self.control.number:
            raise ValueError('RangeValidator requires the control to contain numbers only.')

        text = self.control.text.strip()
        if not self.control.required and not text:
            return True

        title = self.control.title

        value, ok = to_float(text)
        if not ok:
            self.control.isInvalid(self.number_error.format(title))
            return False

        max_logic = None
        max_error = ''
        if self.maximum is not None and self.max_exclusive:
            max_logic = value >= self.maximum
            max_error = self.max_exc_error
        elif self.maximum is not None and not self.max_exclusive:
            max_logic = value > self.maximum
            max_error = self.max_error

        if max_logic:
            self.control.isInvalid(max_error.format(title, self.maximum))
            return False

        min_logic = None
        min_error = ''
        if self.minimum is not None and self.min_exclusive:
            min_logic = value <= self.minimum
            min_error = self.min_exc_error
        elif self.minimum is not None and not self.min_exclusive:
            min_logic = value < self.minimum
            min_error = self.min_error
        if min_logic:
            self.control.isInvalid(min_error.format(title, self.minimum))
            return False

        return True


class CompareValidator(Validator):
    @unique
    class Operator(Enum):
        Equal = 1
        Not_Equal = 2
        Greater = 3
        Less = 4

    def __init__(self, control, compare_with, operation=Operator.Equal):
        """Compares control's input with input in another control using specified operation.
         The comparision operation must pass for the control to be valid.

        :param control: first control
        :type control: sscanss.ui.widgets.forms.FormControl
        :param compare_with: second control to compare with
        :type compare_with: sscanss.ui.widgets.forms.FormControl
        :param operation: comparison operation
        :type operation: sscanss.ui.widgets.forms.CompareValidator.Operator
        """
        self.compare_error = ''
        self.compare_equality_error = '{} should be equal to {}.'
        self.compare_notequal_error = '{} should not be equal to {}.'
        self.compare_greater_error = '{} should be greater than {}.'
        self.compare_less_error = '{} should be less than {}.'

        self.control = control
        self.compare_op = operation
        self.compare_with = compare_with
        self.control.form_lineedit.textChanged.connect(lambda ignore: compare_with.validate())
        if self.compare_op == CompareValidator.Operator.Equal:
            self.compare_error = self.compare_equality_error
        elif self.compare_op == CompareValidator.Operator.Not_Equal:
            self.compare_error = self.compare_notequal_error
        elif self.compare_op == CompareValidator.Operator.Greater:
            self.compare_error = self.compare_greater_error
        elif self.compare_op == CompareValidator.Operator.Less:
            self.compare_error = self.compare_less_error
        else:
            raise ValueError('Invalid Compare Operator with type:{}'.format(type(operation)))

    def valid(self):
        """validates control's input

        :return: indicates input are valid
        :rtype: bool
        """
        error = self.compare_error.format(self.control.title, self.compare_with.title)
        value = self.control.text.strip()
        value_2 = self.compare_with.text.strip()
        if not self.control.required and not value and not value_2:
            return True

        if self.control.number:
            value, ok = to_float(value)
            value_2, ok_2 = to_float(value_2)
            if not ok or not ok_2:
                self.control.isInvalid(error)
                return False

        if self.compare_op == CompareValidator.Operator.Equal and value != value_2:
            self.control.isInvalid(error)
            return False
        if self.compare_op == CompareValidator.Operator.Not_Equal and value == value_2:
            self.control.isInvalid(error)
            return False
        if self.compare_op == CompareValidator.Operator.Greater and value <= value_2:
            self.control.isInvalid(error)
            return False
        if self.compare_op == CompareValidator.Operator.Less and value >= value_2:
            self.control.isInvalid(error)
            return False

        return True


class FormTitle(QtWidgets.QWidget):
    def __init__(self, text, divider=True, name='form-title'):
        """This widget provides a header for the form with divider and style name

        :param text: text in header
        :type text: str
        :param divider: indicates if divider is required
        :type divider: bool
        :param name: style name
        :type name: str
        """
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
        """Adds extra widgets such as buttons to the header area

        :param control: widget to add
        :type control: PyQt5.QtWidgets.*
        """
        self.title_layout.addWidget(control)


class FormGroup(QtWidgets.QWidget):
    groupValidation = QtCore.pyqtSignal(bool)

    @unique
    class Layout(Enum):
        Vertical = 0
        Horizontal = 1
        Grid = 2

    def __init__(self, layout=Layout.Vertical):
        """Manages arrangement and validation for a group of Form Controls

        :param layout: layout of Form Controls
        :type layout: sscanss.ui.widgets.forms.Layout
        """
        super().__init__()

        self.top_level = []
        self.form_controls = []
        self.valid = True
        if layout == FormGroup.Layout.Vertical:
            self.main_layout = QtWidgets.QVBoxLayout()
        elif layout == FormGroup.Layout.Horizontal:
            self.main_layout = QtWidgets.QHBoxLayout()
        else:
            self.main_layout = QtWidgets.QGridLayout()

        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

    def addControl(self, control):
        """Adds a form control to group.

        :param control: control to add to group
        :type control: sscanss.ui.widgets.forms.FormControl
        """
        if type(control) != FormControl:
            raise ValueError('could not add object of type {}'.format(type(control)))

        self.form_controls.append(control)
        extra_widgets = control.extra
        if isinstance(self.main_layout, QtWidgets.QGridLayout):
            index = 2 * (len(self.form_controls) - 1)
            self.main_layout.addWidget(control.form_label, index, 0)
            self.main_layout.addWidget(control.form_lineedit, index, 1)
            self.main_layout.addWidget(control.validation_label, index+1, 1)
            for i, widget in enumerate(extra_widgets):
                self.main_layout.addWidget(widget, index, i+2)
        else:
            self.main_layout.addWidget(control)
            for widget in extra_widgets:
                self.main_layout.addWidget(widget)

        control.inputValidation.connect(self.validateGroup)
        self.valid &= control.valid

    def validateGroup(self):
        """Checks if all controls in the group are valid if so returns True

        :return: group validation state
        :rtype: bool
        """
        for control in self.form_controls:
            if not control.valid:
                self.valid = False
                break
        else:
            self.valid = True

        self.groupValidation.emit(self.valid)
        return self.valid


class FormControl(QtWidgets.QWidget):
    inputValidation = QtCore.pyqtSignal(bool)

    def __init__(self, title, value, desc='', required=False, number=False, decimals=3):
        """Creates a form widget that provides input validation

        :param title: title to display in label
        :type title: str
        :param value: input value
        :type value: Union[str, float]
        :param desc: description or units to display in label
        :type desc: str
        :param required: indicate if input is required
        :type required: bool
        :param number: indicate if input is numeric
        :type number: bool
        """
        super().__init__()

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)

        self._title = title
        self._desc = desc

        self.form_label = QtWidgets.QLabel(self.label)
        self.form_lineedit = QtWidgets.QLineEdit()
        self.form_lineedit.setMaxLength(255)
        self.validation_label = QtWidgets.QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet('color: red')
        self._validator = None
        self.extra = []

        control_layout.addWidget(self.form_label)
        control_layout.addWidget(self.form_lineedit)
        control_layout.addWidget(self.validation_label)

        self.setLayout(control_layout)

        # Validators
        self.required_validator = RequiredValidator(self) if required else None
        self.range_validator = None
        self.compare_validator = None
        self._number = number

        if number:
            self.range_validator = RangeValidator(self, None, None)
            self._validator = QtGui.QDoubleValidator()
            self._validator.setDecimals(decimals)
            self._validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            self.form_lineedit.setValidator(self._validator)
            self.form_lineedit.setMaxLength(12)

        self.valid = False

        self.value = value
        self.form_lineedit.textChanged.connect(self.validate)
        self.validate()

    @property
    def title(self):
        return self._title

    @property
    def label(self):
        return f'{self._title} ({self._desc}):' if self._desc else f'{self._title}:'

    @property
    def required(self):
        return self.required_validator is not None

    @property
    def number(self):
        return self._number

    def range(self, minimum, maximum, min_exclusive=False, max_exclusive=False):
        """Sets a range within which the control's input is valid.
        By default, the minimum and maximum are included, but can be excluded using
        the corresponding exclusive flag.

        :param minimum: min value of the control
        :type minimum: Union[None, float]
        :param maximum: max value of the control
        :type maximum: Union[None, float]
        :param min_exclusive: indicates min should be excluded
        :type min_exclusive: bool
        :param max_exclusive: indicates max should be excluded
        :type max_exclusive: bool
        """
        self.range_validator = RangeValidator(self, self._fixup(minimum), self._fixup(maximum), min_exclusive,
                                              max_exclusive)

        self.validate()

    def _fixup(self, value):
        if value is None:
            return

        return round(value, self._validator.decimals())

    @property
    def value(self):
        """value return a float if the control is a number otherwise return a string.
        A valueError is thrown if the conversion to float is not possible.

        :return: value in the form control
        :rtype: Union[str, float]
        """
        text = self.form_lineedit.text()
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
        self.form_lineedit.setText(value)

    @property
    def text(self):
        return self.form_lineedit.text()

    @text.setter
    def text(self, value):
        self.form_lineedit.setText(value)

    def compareWith(self, form_control, operation):
        """Specifies which control's input to compare with this control's input
         and which comparision operation to perform. The comparision operation must
         pass for the control to be valid.

        :param form_control: control to compare with
        :type form_control: sscanss.ui.widgets.forms.FormControl
        :param operation: comparision operation to perform
        :type operation: sscanss.ui.widgets.forms.CompareValidator.Operator
        """
        self.compare_validator = CompareValidator(self, form_control, operation)
        self.validate()

    def validate(self):
        """Performs validation for the value in the control. The validation performed (required,
        compare, range, or number) is dependent on which flags have been set.
        """
        if self.required and not self.required_validator.valid():
            return

        if self.range_validator is not None and not self.range_validator.valid():
            return

        if self.compare_validator is not None and not self.compare_validator.valid():
            return

        self.isValid()

    def isValid(self):
        """Puts the control to an valid state """
        self.form_lineedit.setStyleSheet('')
        self.validation_label.setText('')
        self.valid = True
        self.inputValidation.emit(True)

    def isInvalid(self, error):
        """Puts the control to an invalid state

        :param error: error message
        :type error: str
        """
        self.form_lineedit.setStyleSheet('border: 1px solid red;')
        self.validation_label.setText(error)
        self.valid = False
        self.inputValidation.emit(False)

    def setFocus(self):
        """Sets the focus on this control """
        self.form_lineedit.setFocus()


class Banner(QtWidgets.QWidget):
    @unique
    class Type(Enum):
        Info = 1
        Warn = 2
        Error = 3

    def __init__(self, ntype, parent):
        super().__init__(parent)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)

        self.setAutoFillBackground(True)
        layout = QtWidgets.QHBoxLayout()
        self.message_label = QtWidgets.QLabel('')
        self.message_label.setWordWrap(True)
        self.close_button = QtWidgets.QPushButton('DISMISS')
        self.close_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        self.close_button.clicked.connect(self.hide)
        self.action_button = QtWidgets.QPushButton('ACTION')
        self.action_button.hide()
        self.action_button.clicked.connect(lambda: self.action_button.hide())
        self.action_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        layout.addWidget(self.message_label)
        layout.addWidget(self.action_button)
        layout.addWidget(self.close_button)
        self.setLayout(layout)
        self.setType(ntype)

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)

        super().paintEvent(event)

    def setType(self, ntype):
        if ntype == self.Type.Error:
            style = 'Error-Banner'
        elif ntype == self.Type.Warn:
            style = 'Warning-Banner'
        else:
            style = 'Info-Banner'

        self.setObjectName(style)
        self.setStyle(self.style())
        self.setStyleSheet(self.styleSheet())

    def showMessage(self, message, ntype=None, no_action=True):
        self.message_label.setText(message)
        if ntype is not None:
            self.setType(ntype)

        if not no_action and self.action_button.isHidden():
            self.action_button.show()
        elif no_action:
            self.action_button.hide()

        if self.isHidden():
            self.show()

    def actionButton(self, text, slot):
        self.action_button.setText(text)
        if self.action_button.receivers(self.action_button.clicked) > 0:
            self.action_button.clicked.disconnect()
        self.action_button.clicked.connect(slot)
