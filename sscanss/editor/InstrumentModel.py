from functools import partial
from PyQt5 import QtWidgets, QtCore, QtGui
from sscanss.core.util.widgets import FileDialog, FilePicker, ColourPicker


class JsonAttribute(QtCore.QObject):
    def __init__(self):
        super().__init__()

    def createWidget(self):
        pass

    def defaultCopy(self):
        return type(self)()


class JsonVariable(JsonAttribute):
    has_changed = QtCore.pyqtSignal(object)

    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def setValue(self, new_value):
        self.value = new_value
        self.has_changed.emit(self.value)


class JsonString(JsonVariable):
    def __init__(self, value=''):
        super().__init__(value)

    def createWidget(self, title=''):
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

    def createWidget(self, title=''):
        widget = FilePicker("", "", "")
        widget.value_changed.connect(self.setValue)
        """
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QHBoxLayout()
        widget.setLayout(widget.layout)
        self.label = QtWidgets.QLabel(" "*160)
        button = QtWidgets.QPushButton()
        button.setText("Pick file...")
        button.clicked.connect(self.pickFile)
        widget.layout.addWidget(self.label)
        widget.layout.addWidget(button)
        """
        return widget


class JsonFloat(JsonVariable):
    def __init__(self, value=0.0):
        super().__init__(value)

    def setValue(self, new_float):
        self.value = float(new_float)

    def createWidget(self, title=''):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setValue(self.value)
        widget.valueChanged.connect(self.setValue)
        return widget


class JsonAttributeArray(JsonAttribute):
    def __init__(self, attributes):
        super().__init__()
        self.attributes = attributes

    @property
    def value(self):
        return [attribute.value for attribute in self.attributes]

    def createWidget(self, title=''):
        list_widget = QtWidgets.QWidget()
        list_widget.layout = QtWidgets.QHBoxLayout()
        list_widget.setLayout(list_widget.layout)

        for attribute in self.attributes:
            list_widget.layout.addWidget(attribute.createWidget())

        list_widget.layout.setContentsMargins(0, 0, 0, 0)

        return list_widget


class JsonColour(JsonVariable):
    def __init__(self, value=QtGui.QColor()):
        super().__init__(value)

    def setValue(self, new_value):
        self.value = QtGui.QColor(new_value[0] * 255, new_value[1] * 255, new_value[2] * 255, new_value[3] * 255)

    def createWidget(self, title=''):
        widget = ColourPicker(self.value)
        widget.value_changed.connect(self.setValue) #Check the colour is actually picked
        return widget


class JsonObjReference(JsonVariable):
    def __init__(self, object_array, value=''):
        super().__init__(value)
        self.object_array = object_array

    def setValue(self, new_value):
        self.value = self.combo_box.itemText(new_value)

    def createWidget(self, title = ''):
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(self.object_array.getObjectKeys())
        self.combo_box.currentIndexChanged.connect(self.setValue)

        return self.combo_box


class JsonEnum(JsonVariable):
    def __init__(self, value=0):
        super().__init__(value)

    def createWidget(self):
        pass


class JsonObjectAttribute(JsonVariable):
    def __init__(self, object_stack):
        super().__init__()
        self.object_stack = object_stack

    def formatTitle(self, string):
        return ' '.join([word.capitalize() for word in string.split('_')])

    def createWidget(self, title=''):
        button = QtWidgets.QPushButton()
        button.setText("Edit " + self.formatTitle(title) + "...")
        button.clicked.connect(partial(self.object_stack.addObject, self.formatTitle(title), self))  #!-! Storing state in the gui feels bad
        return button

    def createPanel(self):
        pass


class JsonObjectArray(JsonObjectAttribute):
    def __init__(self, objects, key, object_stack):
        super().__init__(object_stack)

        self.objects = objects
        self.current_index = 0
        self.key_attribute = key

        for obj in objects:
            obj.attributes[self.key_attribute].has_changed.connect(self.updateComboBox)

    @property
    def selected(self):
        return self.objects[self.current_index]

    def newSelection(self, new_index):
        self.current_index = new_index
        self.swapSelectedObject()

    def swapSelectedObject(self):
        self.panel.layout.itemAtPosition(1, 0).widget().setParent(None)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def newObject(self):
        self.objects.append(self.objects[0].defaultCopy())
        self.current_index = len(self.objects) - 1
        self.selected.attributes[self.key_attribute].has_changed.connect(self.updateComboBox)
        self.swapSelectedObject()
        self.updateComboBox()

    def getObjectKeys(self):
        return [obj.attributes[self.key_attribute].value for obj in self.objects]

    def deleteObject(self):
        if(len(self.objects) == 1):
            self.objects[0] = self.selected.defaultCopy()
            self.objects[0].attributes[self.key_attribute].has_changed.connect(self.updateComboBox)
        else:
            self.objects.pop(self.current_index)
            if(self.current_index > 1):
                self.current_index -= 1
        self.swapSelectedObject()
        self.updateComboBox()

    def moveObject(self):
        moved_object = self.objects.pop(self.current_index)
        if(self.current_index < len(self.objects) - 1):
            self.current_index += 1
        self.objects.insert(self.current_index, moved_object)
        self.updateComboBox()

    def updateComboBox(self):
        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(self.getObjectKeys())
        combo_box.setCurrentIndex(self.current_index)
        combo_box.currentIndexChanged.connect(self.newSelection)

        if self.panel.layout.itemAtPosition(0, 0):
            self.panel.layout.itemAtPosition(0, 0).widget().setParent(None)

        self.panel.layout.addWidget(combo_box, 0, 0)

    def createPanel(self):
        self.panel = QtWidgets.QWidget()
        self.panel.layout = QtWidgets.QGridLayout()
        self.panel.setLayout(self.panel.layout)
        self.updateComboBox()

        add_button = QtWidgets.QPushButton()
        add_button.setText("Add")
        add_button.clicked.connect(self.newObject)
        delete_button = QtWidgets.QPushButton()
        delete_button.setText("Delete")
        delete_button.clicked.connect(self.deleteObject)
        swap_button = QtWidgets.QPushButton()
        swap_button.setText("Swap")
        swap_button.clicked.connect(self.moveObject)

        self.panel.layout.addWidget(add_button, 0, 1)
        self.panel.layout.addWidget(delete_button, 1, 1)
        self.panel.layout.addWidget(swap_button, 2, 1)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)
        self.panel.layout.setRowStretch(3, 1)

        return self.panel


class JsonObject(JsonObjectAttribute):
    def __init__(self, attributes, object_stack):
        super().__init__(object_stack)
        self.attributes = attributes

    def createPanel(self):
        attributes_panel = QtWidgets.QWidget()
        attributes_panel.layout = QtWidgets.QGridLayout()

        for row, attribute_pair in enumerate(self.attributes.items()):
            key, attribute = attribute_pair
            title = self.formatTitle(key)
            attributes_panel.layout.addWidget(QtWidgets.QLabel(title), row, 0)
            attributes_panel.layout.addWidget(attribute.createWidget(title=title), row, 1)

        attributes_panel.setLayout(attributes_panel.layout)
        return attributes_panel

    def defaultCopy(self):
        new_attributes = {}
        for key, attribute in self.attributes.items():
            new_attributes[key] = attribute.defaultCopy()
        return JsonObject(new_attributes, self.object_stack)


class JsonDirectlyEditableObject(JsonObject):
    def createWidget(self, title=''):
        return self.createPanel()
