from functools import partial
from PyQt5 import QtWidgets, QtCore, QtGui
from sscanss.core.util.widgets import FilePicker, ColourPicker


class JsonAttribute(QtCore.QObject):
    def __init__(self):
        """The parent class of all the nodes in the tree
        Contains the methods which should be overriden by the child classes
        """
        super().__init__()

    def createWidget(self, title=''):
        """Creates the widget which should be used in the GUI to edit the data.
        It is linked with the object using events
        :param title: title the which should be used inside the widget if it is needed
        :type title: str
        """
        pass

    def defaultCopy(self):
        """Creates a copy of the node, it is needed to be able to create an object,
        following the same schema as another object
        """
        return type(self)()

    def setJsonValue(self, json_value):
        """Sets the value in the attribute from the value passed from a Json file"""
        pass

    def getJsonValue(self):
        """Returns the value from the attribute, suitable for use in a Json file"""
        return None


class JsonVariable(JsonAttribute):
    """The parent class of all leafs of the tree. It represents single
    attributes which contain one value and should be edited directly by created widget.
    Have the has_changed event which should be triggered every time value is set
    :param value: initial value of the attribute
    """
    has_changed = QtCore.pyqtSignal(object)

    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def setValue(self, new_value):
        self.value = new_value
        self.has_changed.emit(self.value)

    def setJsonValue(self, json_value):
        self.value = json_value

    def getJsonValue(self):
        return self.value


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
        widget = QtWidgets.QLineEdit(self.value)
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
        """Returns list of values collected from attributes in the array
        :return: list of values
        :rtype: list
        """
        return [attribute.value for attribute in self.attributes]

    def createWidget(self, title=''):
        """Creates a widget which itself contains widgets of the attributes
        :return: the widget with attributes' widgets to be displayed
        :rtype: QWidget
        """
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

    def setJsonValue(self, json_value):
        for i, value in enumerate(json_value):
            self.attributes[i].setJsonValue(value)


class JsonColour(JsonVariable):
    """Attribute which manages attribute responsible for colour. The output should be
    normalised colour while it contains normalised version
    :param value: the initial colour
    :type value: QColour
    """
    rgbSize = 255

    def __init__(self, value=QtGui.QColor()):
        super().__init__(value)

    def setValue(self, new_value):
        super().setValue(QtGui.QColor(new_value[0] * self.rgbSize, new_value[1] * self.rgbSize,
                                      new_value[2] * self.rgbSize, new_value[3] * self.rgbSize))

    def createWidget(self, title=''):
        """Creates a custom picker widget which allows user to pick a colour and then displays it
        :return: the colour picker
        :rtype: ColourPicker
        """
        widget = ColourPicker(self.value)
        widget.value_changed.connect(self.setValue)
        return widget

    def getJsonValue(self):
        return [self.value.value(0) / self.rgbSize, self.value.value(0) / self.rgbSize, self.value.value(0) / self.rgbSize]

    def setJsonValue(self, json_value):
        self.value = QtGui.QColor(json_value[0] * self.rgbSize, json_value[1] * self.rgbSize,
                                  json_value[2] * self.rgbSize, 1)

class JsonObjReference(JsonVariable):
    """Attribute which contains name of an object from an object array
    :param value: the initial selected index
    :type value: str
    """
    def __init__(self, object_array, value=''):
        super().__init__(value)
        self.object_array = object_array

    def setValue(self, new_value):
        self.value = new_value

    def newIndex(self, new_index):
        self.value = self.box_items[new_index]
        self.object_array.objects[new_index].attributes["name"].has_changed.connect(self.setValue)

    def createWidget(self, title=''):
        """Creates a combobox to choose one object from already existing ones in the list
        :return: the combobox where the user can select the object
        :rtype: QComboBox
        """
        self.box_items = self.object_array.getObjectKeys()
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(self.box_items)
        if self.value in self.box_items:
            self.combo_box.setCurrentText(self.value)
        self.combo_box.currentIndexChanged.connect(self.newIndex)

        return self.combo_box

    def defaultCopy(self):
        return JsonObjReference(self.object_array)

    def getJsonValue(self):
        return self.value


class JsonEnum(JsonVariable):
    """Attribute which allows to select a value from an enum
    :param enum_class: the class to chose value from
    :type enum_class: Enum
    :param value: the initial selected index
    :type value: int
    """
    def __init__(self, enum_class, value=0):
        super().__init__(value)
        self.enum = enum_class

    def enumList(self):
        return [option.value for option in self.enum]

    def createWidget(self, title=''):
        """Creates combobox with the options as all possible values of the enum
        :return: the combo box to edit enum values
        :rtype: QComboBox
        """
        widget = QtWidgets.QComboBox()
        widget.addItems(self.enumList())
        widget.setCurrentIndex(self.value)
        widget.currentIndexChanged.connect(self.setValue)

        return widget

    def defaultCopy(self):
        return JsonEnum(self.enum)

    def getJsonValue(self):
        return self.enumList()[self.value]

    def setJsonValue(self, json_value):
        self.enumList().index(json_value)


class ObjectOrder(JsonVariable):
    def __init__(self, object_array, value=[]):
        super().__init__(value)
        self.object_array = object_array
        #self.value = value
        #self.stack = []
        #self.remaining = self.object_array.getObjectKeys()

    def updateWidget(self):
        for obj_name in self.stack:
            return_button = QtWidgets.QPushButton()

    def pushNew(self, new_object):
        pass

    def returnTo(self, string):
        pass

    def createWidget(self, title=''):
        return QtWidgets.QWidget()
        self.orderWidget = QtWidgets.QWidget()
        self.orderWidget.layout = QtWidgets.QHBoxLayout()
        self.orderWidget.setLayout(self.orderWidget.layout)
        self.updateWidget()
        return self.orderWidget

    def defaultCopy(self):
        return ObjectOrder(self.object_array)

    def setJsonValue(self, json_value):
        pass

    def getJsonValue(self):
        return None

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
        """Formats string into title by separating words on '_' and capitalising first letter
        :pram string: the string to be formatted
        :type string: str
        :return: the original string in title format
        :rtype: str
        """
        return ' '.join([word.capitalize() for word in string.split('_')])

    def createWidget(self, title=''):
        """Creates the button which would switch the UI of the designer to the current object by
        updating pushing itself on top of the object stack
        :return: The button liked to stack to push the object on top of it
        :rtype: QButton
        """
        title = self.formatTitle(title)
        button = QtWidgets.QPushButton("Edit " + title + "...")
        button.clicked.connect(partial(self.object_stack.addObject, title, self))
        return button

    def createPanel(self):
        """The panel is a widget which should display other attributes generated by the attributes
        of the object.
        :return: the panel widget
        :rtype: QWidget
        """
        return None


class JsonObjectArray(JsonObjectAttribute):
    """The array of other objects which should allow the user to add, delete and edit each of them individually
        :param object: the array of the objects which must be non-empty
        :type object: list
        :param key: the name of the attribute which should be used to identify objects in the list
        :type key: str
        :param object_stack: the reference to the object stack from the designer widget
        :type object_stack: ObjectStack
        """
    def __init__(self, parent_object, path, key, object_stack):
        super().__init__(object_stack)

        self.parent_object = object
        self.path = path
        self.current_index = 0
        self.key_attribute = key
        self.panel = None

        for obj in objects:
            obj.attributes[self.key_attribute].has_changed.connect(self.updateComboBox)

    def setUpObject(self):
        current_object = self.object

    @property
    def selected(self):
        return self.objects[self.current_index]

    @selected.setter
    def selected(self, new_obj):
        self.objects[self.current_index] = new_obj

    @property
    def prototype(self):
        """Returns the first object in the list to be copied when new objects are added"""
        return self.objects[0]

    def newSelection(self, new_index):
        self.current_index = new_index
        self.updateSelectedPanel()

    def updateSelectedPanel(self):
        self.panel.layout.itemAtPosition(1, 0).widget().setParent(None)
        self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def getObjectKeys(self):
        """
        The method returns the list with all the objects' keys
        :return: list of keys
        :rtype: list
        """
        return [obj.attributes[self.key_attribute].value for obj in self.objects]

    def newObject(self):
        """Creates the new object in the end of the list and selects it"""
        self.objects.append(self.prototype.defaultCopy())
        self.current_index = len(self.objects) - 1
        self.selected.attributes[self.key_attribute].has_changed.connect(self.updateComboBox)
        if self.panel:
            self.updateSelectedPanel()
            self.updateComboBox()

    def deleteObject(self):
        """Deletes the current object, if it was the last remaining replaces it with a default copy"""
        if len(self.objects) == 1:
            self.selected = self.prototype.defaultCopy()
            self.selected.attributes[self.key_attribute].has_changed.connect(self.updateComboBox)
        else:
            self.objects.pop(self.current_index)
            if self.current_index > 0:
                self.current_index -= 1

    def moveObject(self):
        """Moves the currently selected object by 1 in the list if it was not the last one"""
        moved_object = self.objects.pop(self.current_index)
        if self.current_index < len(self.objects):
            self.current_index += 1
        self.objects.insert(self.current_index, moved_object)
        self.updateComboBox()

    def updateComboBox(self):
        """Recreates the combobox when any information on it has changed"""
        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(self.getObjectKeys())
        combo_box.setCurrentIndex(self.current_index)
        combo_box.currentIndexChanged.connect(self.newSelection)

        if self.panel.layout.itemAtPosition(0, 0):
            self.panel.layout.itemAtPosition(0, 0).widget().setParent(None)

        self.panel.layout.addWidget(combo_box, 0, 0)

    def createPanel(self):
        """Creates the panel to be displayed with the combobox to select current object, button to add, delete and move objects,
        and the panel to edit the selected object
        :return: the panel widget
        :rtype: QWidget
        """
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

    def setJsonValue(self, json_value):
        while len(self.objects) < len(json_value):
            self.newObject()

        for i, obj in enumerate(json_value):
            self.objects[i].setJsonValue(obj)


class JsonObject(JsonObjectAttribute):
    """The json object which contains attributes, including variables and other objects
    :param attributes: dictionary with all the attributes of the object
    :type attributes: dict{str: JsonAttribute}
    :param object_stack: reference to the object stack from the designer
    :type object_stack: ObjectStack
    """
    def __init__(self, attributes, object_stack):
        super().__init__(object_stack)
        self.attributes = attributes

    def createPanel(self):
        """Creates the panel widget by getting widgets from each attribute
        :return: the panel with other widgets
        :rtype: QWidget
        """
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
        return type(self)(new_attributes, self.object_stack)

    def getJsonValue(self):
        return {title: value.getJsonValue() for (title, value) in self.attributes}

    def setJsonValue(self, json_value):
        for key, value in json_value.items():
            print("key: " + key + ", value: " + str(value))
            self.attributes[key].setJsonValue(value)


class JsonDirectlyEditableObject(JsonObject):
    """Class is the same as JsonObject but instead of creating button it allows to edit itself directly inside
    panel of owning object
    """
    def createWidget(self, title=''):
        return self.createPanel()
