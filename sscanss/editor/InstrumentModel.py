from functools import partial
from PyQt5 import QtWidgets, QtCore, QtGui
from sscanss.core.util.widgets import FilePicker, ColourPicker


class JsonAttribute(QtCore.QObject):
    def __init__(self):
        """The parent class of all the nodes in the tree
        Contains the methods which should be overriden by the child classes
        """
        super().__init__()

    def createWidget(self):
        """Creates the widget which should be used in the GUI to edit the data.
        It is linked with the object using events
        """
        pass

    def defaultCopy(self):
        """Creates a copy of the node, it is needed to be able to create an object,
        following the same schema as another object
        """
        return type(self)()

    def setJsonValue(self, json_value):
        """Sets the value in the attribute from the value passed from a Json file"""
        self.value = json_value

    def getJsonValue(self):
        """Returns the value from the attribute, suitable for use in a Json file"""
        return self.value

class JsonVariable(JsonAttribute):
    """The parent class of all leafs of the tree. It represents single
    attributes which contain one value and should be edited directly by created widget.
    Have the has_changed event which should be triggered every time value is set
    :param value: initial value of the attribute
    :type value: object
    """
    has_changed = QtCore.pyqtSignal(object)

    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def setValue(self, new_value):
        self.value = new_value
        self.has_changed.emit(self.value)


class JsonString(JsonVariable):
    """Attribute which contains a simple string
    :param value: initial string
    :type value: str
    """
    def __init__(self, value=''):
        super().__init__(value)

    def createWidget(self, title=''):
        """Creates a line edit to modify the string in the attribute
        :return: the line edit
        :rtype: QLineEdit
        """
        widget = QtWidgets.QLineEdit()
        widget.setText(self.value)
        widget.textEdited.connect(self.setValue)
        return widget


class JsonFile(JsonVariable):
    """Attribute which should allow to modify a file path
    :param directory: the initial directory which will be suggested to a user
    :type directory: str
    :param filter: which files should be allowed to be open
    :type filter: str
    :param value: the initial address in the attribute
    :type value: str
    """
    def __init__(self, directory='', filter='', value=''):
        super().__init__(value)
        self.directory = directory
        self.filter = filter

    def createWidget(self, title=''):
        """Creates the file picker to choose the filepath in the attribute
        :return: the file picker
        :rtype: FilePicker
        """
        widget = FilePicker(self.directory, False, self.filter)
        widget.value = self.value
        widget.value_changed.connect(self.setValue)

        return widget

    def defaultCopy(self):
        return JsonFile(self.directory, self.filter)

class JsonFloat(JsonVariable):
    """Attribute manages the float values in objects"""
    def __init__(self, value=0.0):
        super().__init__(value)

    def setValue(self, new_float):
        self.value = float(new_float)

    def createWidget(self, title=''):
        """Creates a spin box to enter a float value
        :return: The spin box
        :rtype: QDoubleSpinBox
        """
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(float("-inf"), float("inf"))
        widget.setValue(self.value)
        widget.valueChanged.connect(self.setValue)
        return widget


class JsonAttributeArray(JsonAttribute):
    """Array of several attributes allowing to edit them on the same line"""
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

    def defaultCopy(self):
        return JsonAttributeArray([attribute.defaultCopy() for attribute in self.attributes])

    def getJsonValue(self):
        return [attr.getJsonValue() for attr in self.attributes]

class JsonColour(JsonVariable):
    rgbSize = 255

    def __init__(self, value=QtGui.QColor()):
        super().__init__(value)

    def setValue(self, new_value):
        self.value = QtGui.QColor(new_value[0] * self.rgbSize, new_value[1] * self.rgbSize, new_value[2] * self.rgbSize, new_value[3] * self.rgbSize)

    def createWidget(self, title=''):
        widget = ColourPicker(self.value)
        widget.value_changed.connect(self.setValue)
        return widget

    def getJsonValue(self):
        return [self.value.value(0) / self.rgbSize, self.value.value(0) / self.rgbSize, self.value.value(0) / self.rgbSize]


class JsonObjReference(JsonVariable):
    """Attribute which contains name of an object from an object array
    """
    def __init__(self, object_array, value=0):
        super().__init__(value)
        self.object_array = object_array

    def createWidget(self, title=''):
        """Creates a combobox to choose one object from already existing ones in the list
        :return: the combobox where the user can select the object
        :rtype: QComboBox
        """
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(self.object_array.getObjectKeys())
        self.combo_box.setCurrentIndex(self.value)
        self.combo_box.currentIndexChanged.connect(self.setValue)

        return self.combo_box

    def defaultCopy(self):
        return JsonObjReference(self.object_array)

    def getJsonValue(self):
        return self.object_array.getObjectKeys()[self.value]

class JsonEnum(JsonVariable):
    def __init__(self, enum_class, value=0):
        super().__init__(value)
        self.enum = enum_class

    def enumList(self):
        return [option.name for option in self.enum]

    def createWidget(self, title=''):
        widget = QtWidgets.QComboBox()
        widget.addItems(self.enumList())
        widget.setCurrentIndex(self.value)
        widget.currentIndexChanged.connect(self.setValue)

        return widget

    def defaultCopy(self):
        return JsonEnum(self.enum)

    def getJsonValue(self):
        return self.enumList()[self.value]

class ObjectOrder(JsonVariable):
    def __init__(self, object_array, value=''):
        super().__init__()
        self.object_array = object_array
        self.value = value


class JsonObjectAttribute(JsonVariable):
    """Parent class of all the node attributes - classes should be able to be added to the object stack and
    allow to modify their attributes by getting their widgets
    :param object_stack: the reference to the object stack from the designer widget
    :type object_stack: ObjectStack
    """
    def __init__(self, object_stack):
        super().__init__()
        self.object_stack = object_stack

    def formatTitle(self, string):
        return ' '.join([word.capitalize() for word in string.split('_')])

    def createWidget(self, title=''):
        title = self.formatTitle(title)
        button = QtWidgets.QPushButton("Edit " + title + "...")
        button.clicked.connect(partial(self.object_stack.addObject, title, self))
        return button

    def createPanel(self):
        """"""
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

    @selected.setter
    def selected(self, new_obj):
        self.objects[self.current_index] = new_obj

    @property
    def prototype(self):
        return self.objects[0]

    def newSelection(self, new_index):
        self.current_index = new_index
        self.updateSelectedPanel()

    def updateSelectedPanel(self):
        self.panel.layout.itemAtPosition(1, 0).widget().setParent(None)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def newObject(self):
        self.objects.append(self.prototype.defaultCopy())
        self.current_index = len(self.objects) - 1
        self.selected.attributes[self.key_attribute].has_changed.connect(self.updateComboBox)
        self.updateSelectedPanel()
        self.updateComboBox()

    def getObjectKeys(self):
        return [obj.attributes[self.key_attribute].value for obj in self.objects]

    def deleteObject(self):
        if len(self.objects) == 1:
            self.selected = self.prototype.defaultCopy()
            self.selected.attributes[self.key_attribute].has_changed.connect(self.updateComboBox)
        else:
            self.objects.pop(self.current_index)
            if self.current_index > 0:
                self.current_index -= 1
        self.updateSelectedPanel()
        self.updateComboBox()

    def moveObject(self):
        moved_object = self.objects.pop(self.current_index)
        if self.current_index < len(self.objects):
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

        button_widget = QtWidgets.QWidget()
        button_widget.layout = QtWidgets.QVBoxLayout()
        button_widget.setLayout(button_widget.layout)
        add_button = QtWidgets.QPushButton()
        add_button.setText("Add")
        add_button.clicked.connect(self.newObject)
        delete_button = QtWidgets.QPushButton()
        delete_button.setText("Delete")
        delete_button.clicked.connect(self.deleteObject)
        swap_button = QtWidgets.QPushButton()
        swap_button.setText("Swap")
        swap_button.clicked.connect(self.moveObject)
        button_widget.layout.addWidget(add_button)
        button_widget.layout.addWidget(delete_button)
        button_widget.layout.addWidget(swap_button)

        self.panel.layout.addWidget(button_widget, 0, 1, 2, 1)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)
        self.panel.layout.setRowStretch(3, 0)
        self.panel.layout.setRowStretch(3, 1)

        return self.panel

    def defaultCopy(self):
        return JsonObjectArray([self.prototype.defaultCopy()], self.key_attribute, self.object_stack)

    def getJsonValue(self):
        return [obj.getJsonValue() for obj in self.objects]

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

    def getJsonValue(self):
        return {title: value.getJsonValue() for (title, value) in self.attributes}

class JsonDirectlyEditableObject(JsonObject):
    def createWidget(self, title=''):
        return self.createPanel()

    def defaultCopy(self):
        new_attributes = {}
        for key, attribute in self.attributes.items():
            new_attributes[key] = attribute.defaultCopy()
        return JsonDirectlyEditableObject(new_attributes, self.object_stack)