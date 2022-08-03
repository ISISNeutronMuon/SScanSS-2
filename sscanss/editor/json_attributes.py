from functools import partial
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from sscanss.core.util.widgets import FilePicker, ColourPicker


class AttributeTitle:
    """Is a title used as a key for each attribute
    :param json_title: the title used by the json schema
    :type: str
    :param actual_title: the title which should be used in the gui for better description
    :type actual_title: str
    """
    def __init__(self, json_title, actual_title=''):
        self.json_title = json_title
        if not actual_title:
            self.actual_title = json_title


class JsonAttribute(QtCore.QObject):
    been_set = QtCore.pyqtSignal(object)

    def __init__(self):
        """The parent class of all the nodes in the tree
        Contains the methods which should be overridden by the child classes
        """
        super().__init__()
        self.tree_parent = None

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

    @property
    def json_value(self):
        """Returns the value from the attribute, suitable for use in a Json file"""
        return None

    @json_value.setter
    def json_value(self, value):
        """Sets the value in the attribute from the value passed from a Json file"""
        pass


class JsonVariable(JsonAttribute):
    """The parent class of all leafs of the tree. It represents single
    attributes which contain one value and should be edited directly by created widget.
    Have the has_changed event which should be triggered every time value is set
    :param value: initial value of the attribute
    """
    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def setValue(self, new_value):
        self.value = new_value
        self.been_set.emit(self.value)

    @property
    def json_value(self):
        return self.value

    @json_value.setter
    def json_value(self, value):
        self.value = value


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

        for attribute in self.attributes:
            attribute.been_set.connect(self.been_set.emit)

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

    @property
    def json_value(self):
        return [attr.json_value for attr in self.attributes]

    @json_value.setter
    def json_value(self, value):
        for i, attr_value in enumerate(value):
            self.attributes[i].json_value = attr_value


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
        super().setValue(new_value)

    def createWidget(self, title=''):
        """Creates a custom picker widget which allows user to pick a colour and then displays it
        :return: the colour picker
        :rtype: ColourPicker
        """
        widget = ColourPicker(self.value)
        widget.value_changed.connect(self.setValue)
        return widget

    @property
    def json_value(self):
        return [self.value.redF() / self.rgbSize, self.value.greenF() / self.rgbSize, self.value.blueF() / self.rgbSize]

    @json_value.setter
    def json_value(self, value):
        self.value = QtGui.QColor(int(value[0] * self.rgbSize), int(value[1] * self.rgbSize),
                                   int(value[2] * self.rgbSize))


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

    @property
    def json_value(self):
        return self.enumList()[self.value]

    @json_value.setter
    def json_value(self, value):
        self.value = self.enumList().index(value)


class JsonListReference(JsonVariable):
    """Attribute which contains name of an object from an object array
    :param object_array_path: the relative path to the list to take references from in the tree
    :type object_array_path: str
    :param value: the initial selected index
    :type value: str
    """
    def __init__(self, object_array_path, value=''):
        super().__init__(value)
        self.object_array_path = object_array_path
        self.object_names = self.object_array.getObjectKeys()

    @property
    def object_array(self):
        """Returns the array at set path, needed to always reference the relevant list in case parent object got cloned
        :return: the objects list
        :rtype: JsonObjectArray
        """
        curr_object = self.tree_parent
        commands = self.object_array_path.split("/")
        for command in commands:
            if command == ".":
                curr_object = curr_object.tree_parent
            else:
                curr_object = curr_object.attributes[command]

        return curr_object

    def defaultCopy(self):
        return type(self)(self.object_array_path)


class JsonObjectReference(JsonListReference):
    """Attribute which contains name of an object from an object array
    :param object_array_path: the relative path to the list to take references from in the tree
    :type object_array_path: str
    :param value: the initial selected index
    :type value: str
    """
    def __init__(self, object_array_path, value=''):
        super().__init__(value, object_array_path)
        if self.value not in self.object_names:
            self.value = self.object_names[0]

    def newIndex(self, new_index):
        self.value = self.object_names[new_index]
        self.object_array.objects[new_index].attributes["name"].been_set.connect(self.setValue)

    def createWidget(self, title=''):
        """Creates a combobox to choose one object from already existing ones in the list
        :return: the combobox where the user can select the object
        :rtype: QComboBox
        """
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(self.object_names)
        if self.value in self.object_names:
            self.combo_box.setCurrentText(self.value)
        self.combo_box.currentIndexChanged.connect(self.newIndex)

        return self.combo_box


class DropList(QtWidgets.QListWidget):
    itemDropped = QtCore.pyqtSignal()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.itemDropped.emit()


class ObjectOrder(JsonListReference):
    """Attribute is supposed to contain order of an object list
    :param object_array_path: the relative path to the list to take references from in the tree
    :type object_array_path: str
    :param value: the initial selected index
    :type value: list
    """
    def __init__(self, object_array_path, value=[]):
        super().__init__(object_array_path, value)

    def itemDropped(self):
        self.value = [self.obj_list.item(x).text() for x in range(self.obj_list.count())]
        print(self.value)

    def createWidget(self, title=''):
        if not self.value:
            self.value = self.object_names

        self.obj_list = DropList()
        self.obj_list.addItems(self.value)
        self.obj_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.obj_list.itemDropped.connect(self.itemDropped)

        return self.obj_list


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
    def __init__(self, objects, key, object_stack):
        super().__init__(object_stack)

        self.objects = objects
        self.current_index = 0
        self.key_attribute = key
        self.panel = None

        for obj in self.objects:
            obj.attributes[self.key_attribute].been_set.connect(self.updateComboBox)
            obj.been_set.connect(self.been_set.emit)
            obj.tree_parent = self

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
        self.selected.tree_parent = self
        self.selected.attributes[self.key_attribute].been_set.connect(self.updateComboBox)
        if self.panel:
            self.updateSelectedPanel()
            self.updateComboBox()

    def deleteObject(self):
        """Deletes the current object, if it was the last remaining replaces it with a default copy"""
        if len(self.objects) == 1:
            self.selected = self.prototype.defaultCopy()
            self.selected.attributes[self.key_attribute].been_set.connect(self.updateComboBox)
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

        if self.panel:
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

    @property
    def json_value(self):
        return [obj.json_value for obj in self.objects]

    @json_value.setter
    def json_value(self, value):
        while len(self.objects) < len(value):
            self.newObject()

        for i, obj in enumerate(value):
            self.objects[i].json_value = obj


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

        for key, attribute in self.attributes.items():
            attribute.tree_parent = self
            attribute.been_set.connect(self.been_set.emit)

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

    @property
    def json_value(self):
        return {title: value.json_value for (title, value) in self.attributes.items()}

    @json_value.setter
    def json_value(self, value):
        for key, attr_value in value.items():
            print("key: " + key + ", value: " + str(attr_value))
            self.attributes[key].json_value = attr_value


class JsonDirectlyEditableObject(JsonObject):
    """Class is the same as JsonObject but instead of creating button it allows to edit itself directly inside
    panel of owning object
    """
    def createWidget(self, title=''):
        return self.createPanel()
