import os
from functools import partial
from PyQt5 import QtWidgets


class JsonAttribute:
    def __init__(self, mandatory=True, unique=False):
        self.mandatory = mandatory
        self.value = None

    def setValue(self, new_value):
        self.value = new_value

    def createWidget(self):
        pass

    def isObject(self):
        pass


class JsonString(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = ''

    def createWidget(self):
        widget = QtWidgets.QLineEdit()
        widget.setText(self.value)
        widget.textEdited.connect(self.setValue)
        return widget


class JsonFile(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = ''

    def setValue(self, file_path):
        try:
            os.open(file_path)
        except (OSError, ValueError):
            raise ValueError("New value is not a valid filepath")

    def createWidget(self):
        widget = QtWidgets.QLineEdit()
        widget.setText(self.value)
        widget.textEdited.connect(self.setValue)
        return widget


class JsonFloat(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = 0.0

    def setValue(self, new_float):
        self.value = float(new_float)

    def createWidget(self):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setValue(self.value)
        widget.valueChanged.connect(self.setValue)
        return widget


class JsonAttributeArray(JsonAttribute):
    def __init__(self, attributes, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.attributes = attributes

    def createWidget(self):
        list_widget = QtWidgets.QWidget()
        list_widget.layout = QtWidgets.QHBoxLayout()
        list_widget.setLayout(list_widget.layout)

        for attribute in self.attributes:
            list_widget.layout.addWidget(attribute.createWidget())

        return list_widget


class JsonObjReference(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = 0

    def setValue(self, new_float):
        self.value = int(new_float)


class JsonObject(JsonAttribute):
    def __init__(self, title, object_stack, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.object_stack = object_stack
        self.title = title

    def createStackWidget(self):
        button = QtWidgets.QPushButton()
        button.setText(self.formatTitle(self.title))
        button.clicked.connect(partial(self.object_stack.goDown, self))
        return button

    def formatTitle(self, string):
        return ' '.join([word.capitalize() for word in string.split('_')])

    def createPanel(self):
        pass


class JsonObjArray(JsonObject):
    def __init__(self, title, object_stack, mandatory=True, unique=False):
        super().__init__(title, object_stack, mandatory, unique)

        self.array = []
        self.selected = None


class JsonObject(JsonObject):
    def __init__(self, title, attributes, object_stack, mandatory=True, unique=False):
        super().__init__(title, object_stack, mandatory, unique)
        self.attributes = attributes

    def createPanel(self):
        attributes_panel = QtWidgets.QWidget()
        attributes_panel.layout = QtWidgets.QGridLayout()

        for count, attribute_pair in enumerate(self.attributes.items()):
            name, attribute = attribute_pair
            attributes_panel.layout.addWidget(QtWidgets.QLabel(self.formatTitle(name)), count, 0)
            attributes_panel.layout.addWidget(attribute.createWidget(), count, 1)

        attributes_panel.setLayout(attributes_panel.layout)
        return attributes_panel

    def createWidget(self):
        button = QtWidgets.QPushButton()
        button.setText("Edit " + self.title)
        button.clicked.connect(partial(self.object_stack.addObject, self))
        return button
