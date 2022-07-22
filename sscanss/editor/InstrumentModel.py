import os
from functools import partial
from PyQt5 import QtWidgets
from sscanss.core.util.widgets import FileDialog

class JsonAttribute:
    def __init__(self):
        pass

    def createWidget(self):
        pass

    def defaultCopy(self):
        return type(self)()


class JsonVariable(JsonAttribute):
    def __init__(self, value=None):
        self.value = value

    def setValue(self, new_value):
        self.value = new_value


class JsonString(JsonVariable):
    def __init__(self, value=''):
        super().__init__(value)

    def createWidget(self):
        widget = QtWidgets.QLineEdit()
        widget.setText(self.value)
        widget.textEdited.connect(self.setValue)
        return widget


class JsonFile(JsonVariable):
    def __init__(self, parent, value=''):
        super().__init__(value)
        self.parent = parent

    def pickFile(self):
        dialog = FileDialog(self.parent, "Open dir", "", "")
        self.value = dialog.getOpenFileName(self.parent, "Open dir", "", "")
        self.label.setText(self.value)

    def createWidget(self):
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QHBoxLayout()
        widget.setLayout(widget.layout)
        self.label = QtWidgets.QLabel()
        button = QtWidgets.QPushButton()
        button.setText("Pick file...")
        button.clicked.connect(self.pickFile)
        widget.layout.addWidget(self.label)
        widget.layout.addWidget(button)

        return widget


class JsonFloat(JsonVariable):
    def __init__(self, value=0.0):
        super().__init__(value)

    def setValue(self, new_float):
        self.value = float(new_float)

    def createWidget(self):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setValue(self.value)
        widget.valueChanged.connect(self.setValue)
        return widget


class JsonAttributeArray(JsonAttribute):
    def __init__(self, attributes):
        self.attributes = attributes

    @property
    def value(self):
        return [attribute.value for attribute in self.attributes]

    def createWidget(self):
        list_widget = QtWidgets.QWidget()
        list_widget.layout = QtWidgets.QHBoxLayout()
        list_widget.setLayout(list_widget.layout)

        for attribute in self.attributes:
            list_widget.layout.addWidget(attribute.createWidget())

        return list_widget


class JsonColour(JsonVariable):
    def __init__(self, value = ''):
        super().__init__(value)

    def chooseColour(self):
        self.value = QtWidgets.QColorDialog.getColor()

    def createWidget(self):
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QHBoxLayout()
        widget.setLayout(widget.layout)
        button = QtWidgets.QPushButton()
        button.setText("Choose colour")
        button.clicked.connect(self.chooseColour)
        widget.layout.addWidget(button)
        return widget


class JsonObjReference(JsonVariable):
    def __init__(self, mandatory=True):
        super().__init__(mandatory)
        self.value = 0

    def setValue(self, new_float):
        self.value = int(new_float)


class JsonObjectAttribute(JsonVariable):
    def __init__(self, object_stack):
        self.object_stack = object_stack

    def createWidget(self):
        button = QtWidgets.QPushButton()
        button.setText("Edit...")
        button.clicked.connect(partial(self.object_stack.addObject, "Title", self))  #!-! Storing state in the gui feels bad
        return button

    def formatTitle(self, string):
        return ' '.join([word.capitalize() for word in string.split('_')])

    def createPanel(self):
        pass


class JsonObjectArray(JsonObjectAttribute):
    def __init__(self, objects, key, object_stack):
        super().__init__(object_stack)

        self.objects = objects
        self.selected = self.objects[0]
        self.key = key

    def newSelection(self, new_index):
        self.selected = self.objects[new_index]
        self.panel.layout.itemAtPosition(1, 0).widget().setParent(None)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def newObject(self):
        self.objects.append(self.objects[0].defaultCopy())
        self.combo_box.addItem(self.objects[-1].title)
        self.selected = self.objects[-1]
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def deleteObject(self):
        pass

    def moveObjectForward(self):
        pass

    def createPanel(self):
        self.panel = QtWidgets.QWidget()
        self.panel.layout = QtWidgets.QGridLayout()
        self.panel.setLayout(self.panel.layout)

        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems([obj.attributes[self.key].value for obj in self.objects])
        self.combo_box.currentIndexChanged.connect(self.newSelection)
        button = QtWidgets.QPushButton()
        button.setText("Add")
        button.clicked.connect(self.newObject)
        self.panel.layout.addWidget(self.combo_box, 0, 0)
        self.panel.layout.addWidget(button, 0, 1)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

        return self.panel


class JsonObject(JsonObjectAttribute):
    def __init__(self, attributes, object_stack):
        super().__init__(object_stack)
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


class JsonDirectlyEditableObject(JsonObject):
    def createWidget(self):
        return self.createPanel()
