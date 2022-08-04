from functools import partial
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from sscanss.core.util.widgets import FilePicker, ColourPicker


class RelativeReference:
    def __init__(self, parent_list, path_to_list, path_to_attribute):
        self.parent_list = parent_list
        self.path_to_list = path_to_list
        self.path_to_attribute = path_to_attribute

    @staticmethod
    def getAttribute(relative_path, parent_object):
        attributes_to_visit = relative_path.split("/")
        curr_object = parent_object
        for attribute in attributes_to_visit:
            curr_object = curr_object.value[attribute]

        return curr_object

    def getRelativeReference(self, caller):
        for obj in self.parent_list.objects:
            if self.getAttribute(self.path_to_attribute, obj) is caller:
                return self.getAttribute(self.path_to_list, obj)


class JsonAttributes:
    def __init__(self):
        self.attributes = {}

    @staticmethod
    def formatTitle(key):
        """Formats key into a title by splitting words on '_' and capitalising first letters
        :pram key: the string to be formatted
        :type key: str
        :return: the title obtained from the key
        :rtype: str
        """
        return ' '.join([word.capitalize() for word in key.split('_')])

    def addAttribute(self, key, json_value, title='', mandatory=False):
        if not title:
            title = self.formatTitle(key)

        self.attributes[key] = JsonAttribute(json_value, title, mandatory)

    def defaultCopy(self):
        copy = JsonAttributes()
        for key, attribute in self.attributes:
            copy.attributes[key] = attribute.defaultCopy()

        return copy

    def __getitem__(self, key):
        return self.attributes[key].value

    def __iter__(self):
        for key, attribute in self.attributes.items():
            if attribute.turned_on:
                yield key, attribute


class JsonAttribute(QtCore.QObject):
    been_set = QtCore.pyqtSignal(object)

    def __init__(self, json_value, title, mandatory):
        super().__init__()
        self.value = json_value
        self.value.been_set.connect(self.been_set.emit)
        self.title = title
        self.mandatory = mandatory
        self.turned_on = True

    def setTurnedOn(self, new_state):
        self.turned_on = new_state
        self.been_set.emit(self.turned_on)

    def defaultCopy(self):
        return JsonAttribute(self.value.defaultCopy(), self.title, self.mandatory)

    def createWidget(self):
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QHBoxLayout()
        widget.setLayout(widget.layout)

        label = QtWidgets.QLabel(self.title)
        widget.layout.addWidget(label)
        edit_widget = self.value.createEditWidget(self.title)
        widget.layout.addWidget(edit_widget)

        if not self.mandatory:
            checkbox = QtWidgets.QCheckBox("Include")
            checkbox.setChecked(self.turned_on)
            checkbox.stateChanged.connect(self.setTurnedOn)
            widget.layout.addWidget(checkbox)

        return widget

    @property
    def json_value(self):
        return self.value.json_value

    @json_value.setter
    def json_value(self, value):
        self.value.json_value = value


class JsonValue(QtCore.QObject):
    been_set = QtCore.pyqtSignal(object)
    default_value = None

    def __init__(self, initial_value=None):
        """The parent class of all the nodes in the tree
        Contains the methods which should be overridden by the child classes
        """
        super().__init__()
        if initial_value:
            self.value = initial_value
        else:
            self.value = self.default_value

    def createEditWidget(self, title=''):
        """Creates the widget which should be used in the GUI to edit the data.
        It is linked with the object using events
        :param title: title the which should be used inside the widget if it is needed
        :type title: str
        """
        return QtWidgets.QWidget()

    def defaultCopy(self):
        """Creates a copy of the node, it is needed to be able to create an object,
        following the same schema as another object
        """
        return type(self)()

    def setValue(self, new_value):
        self.value = new_value
        self.been_set.emit(new_value)

    @property
    def json_value(self):
        """Returns the value from the attribute, suitable for use in a Json file"""
        return self.value

    @json_value.setter
    def json_value(self, value):
        """Sets the value in the attribute from the value passed from a Json file"""
        self.value = value


class StringValue(JsonValue):
    """Attribute which contains a simple string
    :param value: initial string
    :type value: str
    """
    default_value = ''

    def createEditWidget(self, title=''):
        """Creates a line edit to modify the string in the attribute
        :return: the line edit
        :rtype: QLineEdit
        """
        widget = QtWidgets.QLineEdit(self.value)
        widget.textEdited.connect(self.setValue)
        return widget


class FileValue(JsonValue):
    """Attribute which should allow to modify a file path
    :param directory: the initial directory which will be suggested to a user
    :type directory: str
    :param filter: which files should be allowed to be open
    :type filter: str
    :param value: the initial address in the attribute
    :type value: str
    """
    default_value = ''

    def __init__(self, directory='', filter='', initial_value=None):
        super().__init__(initial_value)
        self.directory = directory
        self.filter = filter

    def createEditWidget(self, title=''):
        """Creates the file picker to choose the filepath in the attribute
        :return: the file picker
        :rtype: FilePicker
        """
        widget = FilePicker(self.directory, False, self.filter)
        widget.value = self.value
        widget.value_changed.connect(self.setValue)

        return widget

    def defaultCopy(self):
        return type(self)(self.directory, self.filter)


class FloatValue(JsonValue):
    """Attribute manages the float values in objects"""
    default_value = 0.0

    def createEditWidget(self, title=''):
        """Creates a spin box to enter a float value
        :return: The spin box
        :rtype: QDoubleSpinBox
        """
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(float("-inf"), float("inf"))
        widget.setValue(self.value)
        widget.valueChanged.connect(self.setValue)
        return widget


class ValueArray(JsonValue):
    """Array of several attributes allowing to edit them on the same line"""
    default_value = []

    def __init__(self, values):
        super().__init__(values)

        for individual_value in self.value:
            individual_value.been_set.connect(self.been_set.emit)

    def createEditWidget(self, title=''):
        """Creates a widget which itself contains widgets of the attributes
        :return: the widget with attributes' widgets to be displayed
        :rtype: QWidget
        """
        array_widget = QtWidgets.QWidget()
        array_widget.layout = QtWidgets.QHBoxLayout()
        array_widget.setLayout(array_widget.layout)

        for attribute in self.value:
            array_widget.layout.addWidget(attribute.createEditWidget())

        array_widget.layout.setContentsMargins(0, 0, 0, 0)

        return array_widget

    def defaultCopy(self):
        return type(self)([individual_value.defaultCopy() for individual_value in self.value])

    @property
    def json_value(self):
        return [individual_value.json_value for individual_value in self.value]

    @json_value.setter
    def json_value(self, value):
        for i, new_json_value in enumerate(value):
            self.value[i].json_value = new_json_value


class ColourValue(JsonValue):
    """Attribute which manages attribute responsible for colour. The output should be
    normalised colour while it contains normalised version
    """
    rgbSize = 255
    default_value = [0.0, 0.0, 0.0]

    def createEditWidget(self, title=''):
        """Creates a custom picker widget which allows user to pick a colour and then displays it
        :return: the colour picker
        :rtype: ColourPicker
        """
        widget = ColourPicker(QtGui.QColor(self.value[0] * self.rgbSize, self.value[1] * self.rgbSize,
                              self.value[2] * self.rgbSize))
        widget.value_changed.connect(self.setValue)
        return widget

    def setValue(self, new_value):
        super().setValue((new_value[0], new_value[1], new_value[2]))


class EnumValue(JsonValue):
    """Attribute which allows to select a value from an enum
    :param enum_class: the class to chose value from
    :type enum_class: Enum
    :param value: the initial selected index
    :type value: int
    """
    default_value = 0

    def __init__(self, enum_class, initial_value=None):
        super().__init__(initial_value)
        self.enum = enum_class
        self.enum_list = [option.value for option in self.enum]

    def createEditWidget(self, title=''):
        """Creates combobox with the options as all possible values of the enum
        :return: the combo box to edit enum values
        :rtype: QComboBox
        """
        widget = QtWidgets.QComboBox()
        widget.addItems(self.enum_list)
        widget.setCurrentIndex(self.value)
        widget.currentIndexChanged.connect(self.setValue)

        return widget

    def defaultCopy(self):
        return type(self)(self.enum)

    @property
    def json_value(self):
        return self.enum_list[self.value]

    @json_value.setter
    def json_value(self, value):
        self.value = self.enum_list.index(value)


class ListReference(JsonValue):
    """Attribute which contains name of an object from an object array
    :param path_to_list: the relative path to the list to take references from in the tree
    :type path_to_list: str
    :param value: the initial selected index
    :type value: str
    """
    def __init__(self, list_path, initial_value=None):
        super().__init__(initial_value)
        self.list_path = list_path
        self.list_reference = self.list_path.getRelativeReference().been_set.connect(self.updateOnListChange)

    def updateOnListChange(self):
        pass

    def defaultCopy(self):
        return type(self)(self.list_path)


class SelectedObject(ListReference):
    """Attribute which contains name of an object from an object array
    :param object_array_path: the relative path to the list to take references from in the tree
    :type object_array_path: str
    :param value: the initial selected index
    :type value: str
    """
    default_value = ''

    def newIndex(self, new_index):
        self.value = self.list_reference.getObjectKeys()[new_index]

    def updateOnListChange(self):
        if self.value:
            if self.value not in self.list_reference.getObjectKeys():
                if len(self.list_reference.getObjectKeys()) > 0:
                    self.value = self.list_reference.getObjectKeys()[0]
                else:
                    self.value = ''
        else:
            if len(self.list_reference.getObjectKeys()) > 0:
                self.value = self.list_reference.getObjectKeys()[0]

    def createEditWidget(self, title=''):
        """Creates a combobox to choose one object from already existing ones in the list
        :return: the combobox where the user can select the object
        :rtype: QComboBox
        """
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(self.list_reference.getObjectKeys())
        self.combo_box.currentIndexChanged.connect(self.newIndex)

        return self.combo_box


class DropList(QtWidgets.QListWidget):
    """The object is like the normal QList but fires an event when object is dragged and dropped"""
    itemDropped = QtCore.pyqtSignal()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.itemDropped.emit()


class ObjectOrder(ListReference):
    """Attribute is supposed to contain order of an object list
    :param object_array_path: the relative path to the list to take references from in the tree
    :type object_array_path: str
    :param value: the initial selected index
    :type value: list
    """
    def itemDropped(self):
        self.value = [self.obj_list.item(x).text() for x in range(self.obj_list.count())]
        print(self.value)

    def updateOnListChange(self):
        self.value = [item for item in self.value if item in self.object_array.getObjectKeys()]
        self.value += [item for item in self.object_array.getObjectKeys() if item not in self.value]

    def createEditWidget(self, title=''):
        if not self.value:
            self.value = self.object_array.getObjectKeys()

        self.obj_list = DropList()
        self.obj_list.addItems(self.value)
        self.obj_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.obj_list.itemDropped.connect(self.itemDropped)

        return self.obj_list


class ObjectAttribute(JsonValue):
    """Parent class of all the node attributes - classes should be able to be added to the object stack and
    allow to modify their attributes by getting their widgets
    :param object_stack: the reference to the object stack from the designer widget
    :type object_stack: ObjectStack
    """
    def __init__(self, object_stack, initial_value=None):
        super().__init__(initial_value)
        self.object_stack = object_stack

    def createEditWidget(self, title=''):
        """Creates the button which would switch the UI of the designer to the current object by
        updating pushing itself on top of the object stack
        :return: The button liked to stack to push the object on top of it
        :rtype: QButton
        """
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


class JsonObject(ObjectAttribute):
    default_value = JsonAttributes()

    def __init__(self, object_stack, initial_value):
        super().__init__(object_stack, initial_value)

        for attribute in self.value.attributes.values():
            attribute.been_set.connect(self.been_set.emit)

    def createPanel(self):
        """Creates the panel widget by getting widgets from each attribute
        :return: the panel with other widgets
        :rtype: QWidget
        """
        attributes_panel = QtWidgets.QWidget()
        attributes_panel.layout = QtWidgets.QVBoxLayout()
        attributes_panel.setLayout(attributes_panel.layout)

        for attribute in self.value.attributes.values():
            attributes_panel.layout.addWidget(attribute.createWidget())

        return attributes_panel

    def defaultCopy(self):
        return type(self)(self.object_stack, self.value.defaultCopy())

    @property
    def json_value(self):
        return {key: attribute.json_value for key, attribute in self.value}

    @json_value.setter
    def json_value(self, value):
        for key, attr_value in value.items():
            self.value[key].json_value = attr_value


class DirectlyEditableObject(JsonObject):
    """Class is the same as JsonObject but instead of creating button it allows to edit itself directly inside
    panel of owning object - object array
    """
    def createEditWidget(self, title=''):
        return self.createPanel()


class ObjectList(ObjectAttribute):
    """The array of other objects which should allow the user to add, delete and edit each of them individually
        :param object: the array of the objects which must be non-empty
        :type object: list
        :param key: the name of the attribute which should be used to identify objects in the list
        :type key: str
        :param object_stack: the reference to the object stack from the designer widget
        :type object_stack: ObjectStack
        """
    def __init__(self, key, object_stack, initial_value=None):
        super().__init__(object_stack, initial_value)

        self.objects = []
        self.current_index = 0
        self.key_attribute = key
        self.panel = QtWidgets.QWidget()

    @property
    def selected(self):
        return self.objects[self.current_index]

    @selected.setter
    def selected(self, new_obj):
        self.objects[self.current_index] = new_obj

    def comboboxNewSelected(self, new_index):
        self.current_index = new_index
        self.updateSelectedPanel()

    def updateSelectedPanel(self):
        if self.panel.layout.itemAtPosition(1, 0):
            self.panel.layout.itemAtPosition(1, 0).widget().setParent(None)

        if len(self.objects) > 0:
            self.panel.layout.addWidget(self.selected.createPanel(), 1, 0)

    def updateComboBox(self):
        """Recreates the combobox when any information on it has changed"""
        combo_box = QtWidgets.QComboBox()
        combo_box.addItems(self.getObjectKeys())
        combo_box.setCurrentIndex(self.current_index)
        combo_box.currentIndexChanged.connect(self.comboboxNewSelected)

        if self.panel.layout.itemAtPosition(0, 0):
            self.panel.layout.itemAtPosition(0, 0).widget().setParent(None)

        self.panel.layout.addWidget(combo_box, 0, 0)

    def getObjectKeys(self):
        """
        The method returns the list with all the objects' keys
        :return: list of keys
        :rtype: list
        """
        return [obj.value[self.key_attribute].value for obj in self.objects]

    def newObject(self):
        """Creates the new object in the end of the list and selects it"""
        self.objects.append(self.value.defaultCopy())
        self.current_index = len(self.objects) - 1
        self.selected.values[self.key_attribute].been_set.connect(self.updateComboBox)
        if self.panel:
            self.updateSelectedPanel()
            self.updateComboBox()

    def deleteObject(self):
        """Deletes the current object, if it was the last remaining replaces it with a default copy"""
        if len(self.objects) > 0:
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
        self.updateSelectedPanel()

        self.buttons_widget = QtWidgets.QWidget()
        self.buttons_widget.layout = QtWidgets.QVBoxLayout()
        self.buttons_widget.setLayout(self.buttons_widget.layout)
        add_button = QtWidgets.QPushButton()
        add_button.setText("Add")
        add_button.clicked.connect(self.newObject)
        delete_button = QtWidgets.QPushButton()
        delete_button.setText("Delete")
        delete_button.clicked.connect(self.deleteObject)
        swap_button = QtWidgets.QPushButton()
        swap_button.setText("Swap")
        swap_button.clicked.connect(self.moveObject)
        self.buttons_widget.layout.addWidget(add_button)
        self.buttons_widget.layout.addWidget(delete_button)
        self.buttons_widget.layout.addWidget(swap_button)
        self.panel.layout.addWidget(self.buttons_widget, 0, 1, 2, 1)

        self.panel.layout.setRowStretch(3, 0)
        self.panel.layout.setRowStretch(3, 1)

        return self.panel

    def defaultCopy(self):
        return type(self)(self.key_attribute, self.object_stack, self.value.defaultCopy())

    @property
    def json_value(self):
        return [obj.json_value for obj in self.objects]

    @json_value.setter
    def json_value(self, value):
        self.objects = []
        self.current_index = 0

        for json_obj in value:
            self.newObject()
            self.selected.json_value = json_obj

