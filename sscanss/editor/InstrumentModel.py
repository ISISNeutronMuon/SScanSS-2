import os
from functools import partial
from PyQt5 import QtWidgets


class JsonAttribute:
    def __init__(self, mandatory=True, unique=False):
        self.mandatory = mandatory
        self.unique = unique
        self.value = None

    def setValue(self, new_value):
        self.value = new_value

    def createWidget(self):
        pass

    def defaultCopy(self):
        return type(self)()


class JsonVariable(JsonAttribute):
    def __init__(self, value=None, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = value


class JsonString(JsonVariable):
    def __init__(self, value='', mandatory=True, unique=False):
        super().__init__(value, mandatory, unique)

    def createWidget(self):
        widget = QtWidgets.QLineEdit()
        widget.setText(self.value)
        widget.textEdited.connect(self.setValue)
        return widget


class JsonFile(JsonVariable):
    def __init__(self, value='', mandatory=True, unique=False):
        super().__init__(value, mandatory, unique)

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


class JsonFloat(JsonVariable):
    def __init__(self, value=0.0, mandatory=True, unique=False):
        super().__init__(value, mandatory, unique)

    def setValue(self, new_float):
        self.value = float(new_float)

    def createWidget(self):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setValue(self.value)
        widget.valueChanged.connect(self.setValue)
        return widget


class JsonAttributeArray(JsonVariable):
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


class JsonObjReference(JsonVariable):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = 0

    def setValue(self, new_float):
        self.value = int(new_float)


class JsonObjectAttribute(JsonVariable):
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

    def createWidget(self):
        button = QtWidgets.QPushButton()
        button.setText("Edit " + self.title)
        button.clicked.connect(partial(self.object_stack.addObject, self))  # !-!Storing state in the gui feels bad
        return button

    def createPanel(self):
        pass


class JsonObjectArray(JsonObjectAttribute):
    def __init__(self, title, objects, object_stack, mandatory=True, unique=False):
        super().__init__(title, object_stack, mandatory, unique)

        self.objects = objects
        self.selected = self.objects[0]

    def newObject(self):
        self.objects.append(self.objects[0].defaultCopy())
        self.combo_box.addItem(self.objects[-1].title)
        self.selected = self.objects[-1]
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def newSelection(self, new_index):
        self.selected = self.objects[new_index]
        self.panel.layout.itemAtPosition(1, 0).widget().setParent(None)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def createPanel(self):
        self.panel = QtWidgets.QWidget()
        self.panel.layout = QtWidgets.QGridLayout()
        self.panel.setLayout(self.panel.layout)

        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems([obj.title for obj in self.objects])
        self.combo_box.currentIndexChanged.connect(self.newSelection)
        button = QtWidgets.QPushButton()
        button.setText("Add")
        button.clicked.connect(self.newObject)
        self.panel.layout.addWidget(self.combo_box, 0, 0)
        self.panel.layout.addWidget(button, 0, 1)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

        return self.panel


class JsonObject(JsonObjectAttribute):
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

    def defaultCopy(self):
        new_attributes = {}

        for key, attribute in self.attributes.items():
            new_attributes[key] = attribute.defaultCopy()
        return JsonObject(self.title, new_attributes, self.object_stack,
                          mandatory=self.mandatory, unique=self.unique)
