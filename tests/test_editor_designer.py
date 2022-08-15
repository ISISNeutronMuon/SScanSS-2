import unittest
from unittest import mock
from itertools import cycle
from PyQt5 import QtTest
from PyQt5 import QtCore, QtWidgets, QtGui
import sscanss.editor.json_attributes as ja
import sscanss.editor.designer as d
from sscanss.core.util.widgets import FilePicker, ColourPicker
from enum import Enum
from helpers import APP, TestSignal


class TestEnum(Enum):
    ZERO = "Zero"
    ONE = "One"
    TWO = "Two"
    THREE = "Three"


class TestDesignerTree(unittest.TestCase):
    def testStringValue(self):
        string_val = ja.StringValue()

        self.assertEqual(string_val.value, '')
        test_string = "This is a string"
        string_val = ja.StringValue(test_string)
        parent = mock.MagicMock()
        parent.been_set.emit = mock.Mock()
        string_val.connectParent(parent)
        self.assertIs(string_val.parent, parent)
        self.assertEqual(string_val.value, test_string)

        # Here test the get/set properties
        new_string = "New string"
        string_val.setValue(new_string)
        self.assertEqual(string_val.value, new_string)
        edit_widget = string_val.createEditWidget()
        self.assertIsInstance(edit_widget, QtWidgets.QLineEdit)
        self.assertEqual(edit_widget.text(), new_string)
        self.assertEqual(string_val.json_value, new_string)
        parent.been_set.emit.assert_called_with(new_string)

        # Here test the created widget
        string_val.setValue("")
        event_handler = mock.Mock()
        string_val.been_set.connect(event_handler)
        ui_string = "Ui text"
        edit_widget = string_val.createEditWidget()
        QtTest.QTest.keyClicks(edit_widget, ui_string)
        self.assertEqual(len(ui_string), event_handler.call_count)
        self.assertEqual(string_val.value, ui_string)

        # Test the json getter/setter
        event_handler = mock.Mock()
        string_val.been_set.connect(event_handler)
        json_string = "Json string"
        string_val.json_value = json_string
        self.assertEqual(string_val.value, json_string)
        parent.assert_not_called()

        copy_string = string_val.defaultCopy()
        self.assertEqual(copy_string.value, '')
        self.assertIsInstance(copy_string, ja.StringValue)

    def testFloatAttribute(self):
        float_attr = ja.FloatValue()

        self.assertEqual(float_attr.value, 0.0)
        test_float = 5.0
        float_attr = ja.FloatValue(test_float)
        parent = mock.Mock()
        parent.been_set.emit = mock.Mock()
        float_attr.connectParent(parent)
        self.assertEqual(float_attr.value, test_float)
        self.assertEqual(float_attr.parent, parent)
        new_float = 7.32
        float_attr.setValue(new_float)
        self.assertEqual(float_attr.value, new_float)
        parent.been_set.emit.assert_called_with(new_float)
        control_widget = float_attr.createEditWidget()
        self.assertIsInstance(control_widget, QtWidgets.QDoubleSpinBox)
        self.assertEqual(control_widget.value(), new_float)

        ui_float = 5.236
        event_handler = mock.Mock()
        float_attr.been_set.connect(event_handler)

        json_float = -1.53
        float_attr.json_value = json_float
        self.assertEqual(float_attr.value, json_float)

        copy_float = float_attr.defaultCopy()
        self.assertEqual(copy_float.value, 0.0)
        self.assertIsInstance(copy_float, ja.FloatValue)

    def testFileAttribute(self):
        file_val = ja.FileValue()

        self.assertEqual(file_val.value, '')
        parent = mock.MagicMock()
        parent.been_set.emit = mock.Mock()
        file_val.connectParent(parent)
        self.assertIs(file_val.parent, parent)

        # Here test the get/set properties
        new_filepath = "filepath"
        file_val.setValue(new_filepath)
        self.assertEqual(file_val.value, new_filepath)
        edit_widget = file_val.createEditWidget()
        self.assertIsInstance(edit_widget, FilePicker)
        self.assertEqual(edit_widget.value, new_filepath)
        self.assertEqual(file_val.json_value, new_filepath)
        parent.been_set.emit.assert_called_with(new_filepath)

        # Test the json getter/setter
        event_handler = mock.Mock()
        file_val.been_set.connect(event_handler)
        json_string = "Json string"
        file_val.json_value = json_string
        self.assertEqual(file_val.value, json_string)
        parent.assert_not_called()

        copy_string = file_val.defaultCopy()
        self.assertEqual(copy_string.value, '')
        self.assertIsInstance(copy_string, ja.FileValue)

    def testAttributeArray(self):
        string_val = "String"
        float_val = -53.2
        int_val = 423

        mock_arr = []
        copy_arr = []
        for i in range(3):
            m = mock.MagicMock()
            m.been_set = TestSignal()
            m.connectParent = mock.Mock()
            copy_attr = mock.Mock()
            m.defaultCopy = mock.Mock(return_value=copy_attr)
            copy_arr.append(copy_attr)
            mock_arr.append(m)

        mock_arr[0].createEditWidget = mock.Mock(return_value=QtWidgets.QLineEdit())
        mock_arr[0].json_value = string_val
        mock_arr[1].createEditWidget = mock.Mock(return_value=QtWidgets.QDoubleSpinBox())
        mock_arr[1].json_value = float_val
        mock_arr[2].createEditWidget = mock.Mock(return_value=QtWidgets.QLabel())
        mock_arr[2].json_value = int_val

        array_value = ja.ValueArray(mock_arr)
        parent = mock.Mock()
        parent.been_set.emit = mock.Mock()
        array_value.connectParent(parent)
        self.assertListEqual(array_value.value, mock_arr)
        for m in mock_arr:
            m.connectParent.assert_called_with(parent)
        widget = array_value.createEditWidget()
        self.assertEqual(widget.layout.count(), 3)
        self.assertListEqual(array_value.json_value, [string_val, float_val, int_val])
        parent.been_set.emit.assert_not_called()

        array_value.json_value = ["new", 3.2, -4]
        self.assertEqual(mock_arr[0].json_value, "new")
        self.assertEqual(mock_arr[1].json_value, 3.2)
        self.assertEqual(mock_arr[2].json_value, -4)

        copy_value = array_value.defaultCopy()
        self.assertIsInstance(copy_value, ja.ValueArray)
        self.assertListEqual(copy_value.value, copy_arr)

    def testEnumAttribute(self):
        enum_attr = ja.EnumValue(TestEnum)

        self.assertEqual(enum_attr.value, 0)
        test_index = 1
        enum_attr = ja.EnumValue(TestEnum, test_index)
        parent = mock.Mock()
        parent.been_set.emit = mock.Mock()
        enum_attr.connectParent(parent)
        self.assertEqual(enum_attr.value, test_index)
        new_index = 0
        enum_attr.setValue(0)
        self.assertEqual(enum_attr.value, new_index)
        control_widget = enum_attr.createEditWidget()
        self.assertIsInstance(control_widget, QtWidgets.QComboBox)
        self.assertEqual(control_widget.currentIndex(), new_index)
        self.assertEqual(enum_attr.json_value, "Zero")
        parent.been_set.emit.assert_called_with(new_index)
        json_value = "Three"
        enum_attr.json_value = json_value
        self.assertEqual(enum_attr.value, 3)
        parent.been_set.emit.assert_called_with(new_index)

        copy_enum = enum_attr.defaultCopy()
        self.assertEqual(copy_enum.value, 0.0)
        self.assertIsInstance(copy_enum, ja.EnumValue)

    def testColourAttribute(self):
        colour_attr = ja.ColourValue()
        self.assertEqual(colour_attr.value, [0.0, 0.0, 0.0])
        test_colour = [0.12, 0.35, 0.58]
        colour_attr = ja.ColourValue(test_colour)
        parent = mock.Mock()
        parent.been_set.emit = mock.Mock()
        colour_attr.connectParent(parent)
        self.assertEqual(colour_attr.value, test_colour)
        new_colour = [0.52, 0.86, 0.03]
        colour_attr.setValue(new_colour)
        parent.been_set.emit.assert_called_with(new_colour)
        self.assertEqual(new_colour, colour_attr.value)
        control_widget = colour_attr.createEditWidget()
        self.assertIsInstance(control_widget, ColourPicker)
        rgb_size = 255
        self.assertEqual(
            control_widget.value,
            QtGui.QColor(int(new_colour[0] * rgb_size), int(new_colour[1] * rgb_size), int(new_colour[2] * rgb_size)))
        json_colour = [0.4, 0.8, 0.06]
        colour_attr.json_value = json_colour
        self.assertEqual(colour_attr.value, json_colour)

        copy_colour = colour_attr.defaultCopy()
        self.assertEqual(copy_colour.value, [0.0, 0.0, 0.0])
        self.assertIsInstance(copy_colour, ja.ColourValue)

    def testSelectedObject(self):
        list_mock = mock.Mock()
        list_mock.getObjectKeys = mock.Mock(return_value=["key1", "key2", "key3"])
        mock_objects = []
        for i in range(3):
            m = mock.Mock()
            m.been_set = TestSignal()
            mock_objects.append(m)
        list_mock.value = mock_objects
        list_mock.been_set = TestSignal()
        parent = mock.Mock()
        parent.been_set.emit = mock.Mock()
        list_path = mock.Mock()
        list_path.getRelativeReference = mock.Mock(return_value=list_mock)
        attribute = ja.SelectedObject(list_path)
        attribute.connectParent(parent)

        attribute.resolveReferences()
        self.assertIs(attribute.list_reference, list_mock)
        self.assertEqual(attribute.value, "key1")
        self.assertEqual(attribute.json_value, "key1")
        attribute.json_value = "key2"
        self.assertEqual(attribute.value, "key2")

        combobox = attribute.createEditWidget()
        self.assertIsInstance(combobox, QtWidgets.QComboBox)
        self.assertEqual("key2", combobox.currentText())
        self.assertEqual("key1", combobox.itemText(0))
        self.assertEqual("key2", combobox.itemText(1))
        self.assertEqual("key3", combobox.itemText(2))

        combobox.setCurrentIndex(2)
        combobox.currentIndexChanged.emit(2)
        parent.been_set.emit.assert_called_with("key3")
        self.assertEqual("key3", attribute.value)

        mock_objects[2].been_set.emit("new key")
        self.assertEqual("new key", attribute.value)
        event_handler = mock.Mock()
        attribute.been_set.connect(event_handler)
        list_mock.getObjectKeys = mock.Mock(return_value=["key1", "new key"])
        list_mock.value = [mock_objects[0], mock_objects[2]]
        list_mock.been_set.emit()
        self.assertEqual("new key", attribute.value)
        event_handler.assert_not_called()

        list_mock.getObjectKeys = mock.Mock(return_value=["key1", "new key2"])
        list_mock.value = [mock_objects[0], mock_objects[2]]
        list_mock.been_set.emit()
        self.assertEqual("key1", attribute.value)

        copy_attr = attribute.defaultCopy()
        self.assertIsInstance(copy_attr, ja.SelectedObject)
        self.assertEqual(copy_attr.list_path, attribute.list_path)

    def testObjectOrderAttribute(self):
        list_mock = mock.Mock()
        list_mock.getObjectKeys = mock.Mock(return_value=["key1", "key2", "key3"])
        mock_objects = []
        for i in range(3):
            m = mock.Mock()
            m.been_set = TestSignal()
            mock_objects.append(m)
        list_mock.value = mock_objects
        list_mock.been_set = TestSignal()
        parent = mock.Mock()
        parent.been_set.emit = mock.Mock()
        list_path = mock.Mock()
        list_path.getRelativeReference = mock.Mock(return_value=list_mock)
        attribute = ja.ObjectOrder(list_path)
        attribute.connectParent(parent)

        attribute.resolveReferences()
        self.assertIs(attribute.list_reference, list_mock)
        self.assertListEqual(attribute.value,
                             [ja.OrderItem("key1", True),
                              ja.OrderItem("key2", True),
                              ja.OrderItem("key3", True)])
        self.assertEqual(attribute.json_value, ["key1", "key2", "key3"])
        attribute.json_value = ["key2", "key3", "key1"]
        self.assertListEqual(attribute.value,
                             [ja.OrderItem("key2", True),
                              ja.OrderItem("key3", True),
                              ja.OrderItem("key1", True)])

        list_widget = attribute.createEditWidget()
        self.assertIsInstance(list_widget, QtWidgets.QListWidget)
        self.assertEqual("key2", list_widget.item(0).text())
        self.assertEqual("key3", list_widget.item(1).text())
        self.assertEqual("key1", list_widget.item(2).text())

        list_mock.getObjectKeys = mock.Mock(return_value=["key3", "new key", "key2"])
        list_mock.been_set.emit()
        self.assertEqual([ja.OrderItem("key2", True),
                          ja.OrderItem("key3", True),
                          ja.OrderItem("new key", True)], attribute.value)
        parent.been_set.emit.assert_not_called()

        list_mock.getObjectKeys = mock.Mock(return_value=["key3", "new key2"])
        list_mock.value = [mock_objects[1], mock_objects[2]]
        list_mock.been_set.emit()
        self.assertEqual([ja.OrderItem("key3", True), ja.OrderItem("new key2", True)], attribute.value)

        copy_attr = attribute.defaultCopy()
        self.assertIsInstance(copy_attr, ja.ObjectOrder)
        self.assertEqual(copy_attr.list_path, attribute.list_path)

    def testJsonObject(self):
        mock_stack = mock.Mock()
        mock_stack.addObject = mock.Mock()
        mock_dict = {}
        mock_copies = [mock.Mock(), mock.Mock(), mock.Mock()]
        for i in range(1, 4):
            attr = mock.Mock()
            attr.connectParent = mock.Mock()
            attr.resolveReferences = mock.Mock()
            attr.json_value = "j" + str(i)
            if i == 1 or i == 2:
                attr.turned_on = True
                attr.mandatory = True
            else:
                attr.turned_on = False
                attr.mandatory = False
            mock_dict["key" + str(i)] = attr
            attr.createWidget = mock.Mock(return_value=QtWidgets.QWidget())
            attr.defaultCopy = mock.Mock(return_value=mock_copies[i - 1])

        mock_attributes = mock.MagicMock()
        mock_attributes.attributes = mock_dict
        mock_attributes.__iter__.return_value = mock_attributes.attributes.items()
        mock_attributes.__getitem__ = mock_dict.__getitem__
        json_object = ja.JsonObject(mock_stack, mock_attributes)
        self.assertIs(mock_attributes, json_object.value)
        for attr in mock_dict.values():
            attr.connectParent.assert_called_with(json_object)

        json_object.resolveReferences()
        for attr in mock_dict.values():
            attr.resolveReferences.assert_called()

        self.assertEqual({"key1": "j1", "key2": "j2"}, json_object.json_value)
        json_object.json_value = {"key1": "n1", "key2": "n2", "key3": "n3"}
        for i, m in enumerate(mock_dict.values()):
            self.assertEqual("n" + str(i + 1), m.json_value)
            self.assertEqual(True, m.turned_on)

        json_object.json_value = {"key2": "m2"}
        self.assertTrue(mock_dict["key1"].turned_on)
        self.assertTrue(mock_dict["key2"].turned_on)
        self.assertFalse(mock_dict["key3"].turned_on)

        widget = json_object.createEditWidget("Object")
        widget.click()
        mock_stack.addObject.assert_called_with("Object", json_object, mock.ANY)

        panel = json_object.createPanel()
        self.assertIsInstance(panel, QtWidgets.QWidget)
        for attr in mock_dict.values():
            attr.createWidget.assert_called_once()

        copy_object = json_object.defaultCopy()
        self.assertIsInstance(copy_object, ja.JsonObject)
        for i, key_attr in enumerate(copy_object.value()):
            key, attr = key_attr
            self.assertIsInstance(attr, mock_copies[i])

    def testDirectlyEditableObject(self):
        mock_stack = mock.Mock()
        mock_stack.addObject = mock.Mock()
        mock_dict = {}
        mock_copies = [mock.Mock(), mock.Mock(), mock.Mock()]
        for i in range(1, 4):
            attr = mock.Mock()
            attr.connectParent = mock.Mock()
            attr.resolveReferences = mock.Mock()
            attr.json_value = "j" + str(i)
            if i == 1 or i == 2:
                attr.turned_on = True
                attr.mandatory = True
            else:
                attr.turned_on = False
                attr.mandatory = False
            mock_dict["key" + str(i)] = attr
            attr.createWidget = mock.Mock(return_value=QtWidgets.QWidget())
            attr.defaultCopy = mock.Mock(return_value=mock_copies[i - 1])

        mock_attributes = mock.MagicMock()
        mock_attributes.attributes = mock_dict
        mock_attributes.__iter__.return_value = mock_attributes.attributes.items()
        mock_attributes.__getitem__ = mock_dict.__getitem__
        json_object = ja.DirectlyEditableObject(mock_stack, mock_attributes)
        self.assertIs(mock_attributes, json_object.value)
        for attr in mock_dict.values():
            attr.connectParent.assert_called_with(json_object)

        json_object.resolveReferences()
        for attr in mock_dict.values():
            attr.resolveReferences.assert_called()

        self.assertEqual({"key1": "j1", "key2": "j2"}, json_object.json_value)
        json_object.json_value = {"key1": "n1", "key2": "n2", "key3": "n3"}
        for i, m in enumerate(mock_dict.values()):
            self.assertEqual("n" + str(i + 1), m.json_value)
            self.assertEqual(True, m.turned_on)

        json_object.json_value = {"key2": "m2"}
        self.assertTrue(mock_dict["key1"].turned_on)
        self.assertTrue(mock_dict["key2"].turned_on)
        self.assertFalse(mock_dict["key3"].turned_on)

        panel = json_object.createPanel()
        self.assertIsInstance(panel, QtWidgets.QWidget)
        for attr in mock_dict.values():
            attr.createWidget.assert_called_once()

        for attr in mock_dict.values():
            attr.createWidget = mock.Mock(return_value=QtWidgets.QWidget())
        widget = json_object.createEditWidget()
        self.assertIsInstance(widget, QtWidgets.QWidget)
        for attr in mock_dict.values():
            attr.createWidget.assert_called_once()

        copy_object = json_object.defaultCopy()
        self.assertIsInstance(copy_object, ja.JsonObject)
        for i, key_attr in enumerate(copy_object.value()):
            key, attr = key_attr
            self.assertIsInstance(attr, mock_copies[i])

    def testObjectList(self):
        mock_stack = mock.Mock()
        mock_stack.addObject = mock.Mock()
        mock_object = mock.Mock()
        mock_object.connectParent = mock.Mock()
        mock_object.resolveReferences = mock.Mock()
        mock_object.been_set = TestSignal()

        mock_dict = {}
        for i in range(3):
            mock_attr = mock.Mock()
            mock_attr.been_set = TestSignal()
            mock_attr.value = "val" + str(i + 1)
            mock_dict["key" + str(i + 1)] = mock_attr
        mock_object.value = mock_dict
        mock_panel = QtWidgets.QWidget()
        mock_object.createPanel = mock.Mock(return_value=mock_panel)
        copy_object = mock.Mock()
        copy_obj_attr = mock.Mock()
        copy_obj_attr.value = "other val"
        copy_object.value = {"key1": copy_obj_attr}
        copy_object.createPanel = mock.Mock(return_value=QtWidgets.QWidget())
        mock_object.defaultCopy = mock.Mock(return_value=copy_object)

        mock_object.json_value = "json val"
        copy_object.json_value = "copy val"
        copy_object.defaultCopy = mock.Mock(return_value=mock_object)

        parent = mock.Mock()
        parent.been_set.emit = mock.Mock()

        obj_list = ja.ObjectList("key1", mock_stack, mock_object)
        self.assertIsInstance(obj_list, ja.ObjectList)
        mock_object.connectParent.assert_called_with(obj_list)
        mock_object.been_set.connect(obj_list.been_set.emit)

        obj_list.connectParent(parent)
        self.assertIs(parent, obj_list.parent)

        obj_list.resolveReferences()
        mock_object.resolveReferences.assert_called()
        self.assertListEqual(["val1"], obj_list.getObjectKeys())

        panel = obj_list.createPanel()
        self.assertIsInstance(panel, QtWidgets.QWidget)
        mock_object.createPanel.assert_called()
        self.assertListEqual(["json val"], obj_list.json_value)

        add_button = obj_list.buttons_widget.layout.itemAt(0).widget()
        self.assertIsInstance(add_button, QtWidgets.QPushButton)
        add_button.click()
        self.assertEqual(["val1", "other val"], obj_list.getObjectKeys())
        self.assertEqual([mock_object, copy_object], obj_list.value)
        self.assertEqual(copy_object, obj_list.selected)
        parent.been_set.emit.assert_called_with(copy_object)

        test_list = [mock_object, copy_object]
        for i, obj in enumerate(obj_list):
            self.assertIs(obj, test_list[i])

        event_handler = mock.Mock()
        obj_list.been_set.connect(event_handler)
        combobox = panel.layout.itemAtPosition(0, 0).widget()
        combobox.setCurrentIndex(0)
        combobox.currentIndexChanged.emit(0)
        self.assertIsInstance(combobox, QtWidgets.QComboBox)
        self.assertIs(mock_object, obj_list.selected)
        event_handler.assert_not_called()

        mock_dict["key1"].value = "Changed"
        mock_dict["key1"].been_set.emit()
        mock_object.been_set.emit("Changed")
        combobox = panel.layout.itemAtPosition(0, 0).widget()
        self.assertEqual(combobox.itemText(0), "Changed")
        event_handler.assert_called_with("Changed")

        event_handler = mock.Mock()
        obj_list.been_set.connect(event_handler)
        move_button = obj_list.buttons_widget.layout.itemAt(2).widget()
        self.assertIsInstance(move_button, QtWidgets.QPushButton)
        move_button.click()
        self.assertIs(mock_object, obj_list.selected)
        self.assertListEqual([copy_object, mock_object], obj_list.value)
        event_handler.assert_not_called()
        move_button.click()
        self.assertIs(mock_object, obj_list.selected)
        self.assertListEqual([copy_object, mock_object], obj_list.value)
        self.assertListEqual(["copy val", "json val"], obj_list.json_value)

        delete_button = obj_list.buttons_widget.layout.itemAt(1).widget()
        self.assertIsInstance(move_button, QtWidgets.QPushButton)
        delete_button.click()
        self.assertIs(copy_object, obj_list.selected)
        self.assertListEqual([copy_object], obj_list.value)
        event_handler.assert_called()

        obj_list.json_value = ["new json", "new json2"]
        self.assertListEqual(obj_list.json_value, ["new json", "new json2"])
        self.assertListEqual(obj_list.value, [copy_object, mock_object])

        obj_list.selected = copy_object
        self.assertListEqual(obj_list.value, [copy_object, copy_object])

        copy_list = obj_list.defaultCopy()
        self.assertIsInstance(copy_list, ja.ObjectList)
        self.assertEqual(copy_list.key_attribute, obj_list.key_attribute)
        self.assertIs(copy_list.object_stack, mock_stack)

    def testJsonAttribute(self):
        num = 642
        test_title = "title"
        mandatory = True
        json_value = mock.Mock()
        edit_widget = QtWidgets.QWidget()
        json_value.createEditWidget = mock.Mock(return_value=edit_widget)
        json_value.json_value = num
        json_value.been_set = TestSignal()
        event_handler = mock.Mock()
        attribute = ja.JsonAttribute(json_value, test_title, mandatory)
        attribute.been_set.connect(event_handler)
        self.assertTrue(attribute.turned_on)
        self.assertTrue(attribute.mandatory)
        self.assertEqual(attribute.json_value, num)
        self.assertEqual(attribute.title, test_title)
        event_handler.assert_not_called()

        new_num = 45
        attribute.json_value = new_num
        self.assertEqual(json_value.json_value, new_num)
        event_handler.assert_not_called()
        widget = attribute.createWidget()
        self.assertEqual(len(widget.layout), 3)

        mandatory = False
        attribute = ja.JsonAttribute(json_value, test_title, mandatory)
        event_handler = mock.Mock()
        attribute.been_set.connect(event_handler)
        self.assertTrue(attribute.turned_on)
        widget = attribute.createWidget()
        self.assertEqual(len(widget.layout), 3)
        checkbox = widget.layout.itemAt(2).widget()
        self.assertTrue(checkbox.isChecked())
        event_handler = mock.Mock()
        attribute.been_set.connect(event_handler)
        QtTest.QTest.mouseClick(checkbox, QtCore.Qt.LeftButton, pos=QtCore.QPoint(2, int(checkbox.height() / 2)))
        self.assertFalse(attribute.turned_on)
        event_handler.assert_called_with(False)

        parent = mock.Mock()
        m = mock.Mock()
        connect_parent = mock.Mock()
        m.connectParent = connect_parent
        the_copy = mock.Mock()
        default_copy = mock.Mock(return_value=the_copy)
        m.defaultCopy = default_copy
        attribute = ja.JsonAttribute(m, test_title, True)
        attribute.connectParent(parent)
        connect_parent.assert_called_with(parent)

        copy = attribute.defaultCopy()
        default_copy.assert_called()
        self.assertEqual(copy.value, the_copy)
        self.assertEqual(copy.title, attribute.title)
        self.assertEqual(copy.mandatory, attribute.mandatory)

    def testJsonAttributes(self):
        json_attributes = ja.JsonAttributes()
        self.assertEqual(json_attributes.formatTitle("mm_mm"), "Mm Mm")
        self.assertEqual(json_attributes.formatTitle("word"), "Word")
        self.assertEqual(json_attributes.formatTitle("Nothing"), "Nothing")
        self.assertEqual(json_attributes.formatTitle("Split_split"), "Split Split")

        json_value = mock.Mock()
        the_copy = mock.Mock()
        json_value.defaultCopy = mock.Mock(return_value=the_copy)
        json_attributes.addAttribute("key", json_value)
        self.assertEqual(json_attributes.attributes["key"].value, json_value)
        self.assertEqual(json_attributes.attributes["key"].title, 'Key')
        self.assertEqual(json_attributes.attributes["key"].mandatory, True)

        json_value2 = mock.Mock()
        the_copy2 = mock.Mock()
        json_value2.defaultCopy = mock.Mock(return_value=the_copy2)
        json_attributes.addAttribute("key2", json_value2, "Title", False)
        self.assertEqual(json_attributes.attributes["key2"].value, json_value2)
        self.assertEqual(json_attributes.attributes["key2"].title, 'Title')
        self.assertEqual(json_attributes.attributes["key2"].mandatory, False)

        self.assertEqual(json_attributes["key"], json_value)
        key_list = []
        attr_list = []
        for key, item in json_attributes:
            key_list.append(key)
            attr_list.append(item)
        self.assertListEqual(key_list, ["key", "key2"])
        self.assertListEqual(attr_list, [json_attributes.attributes["key"], json_attributes.attributes["key2"]])

        default_copy = json_attributes.defaultCopy()
        self.assertIsInstance(default_copy, ja.JsonAttributes)
        self.assertEqual(default_copy["key"], the_copy)
        self.assertEqual(default_copy["key2"], the_copy2)

    def testRelativeReference(self):
        left_child = mock.Mock()
        left_child_attr = mock.Mock()
        left_child_attr.parent = left_child
        right_child = mock.Mock()
        parent = mock.Mock()
        parent.value = {"child": right_child}
        right_child.parent = parent
        left_child.parent = parent

        relative_reference = ja.RelativeReference("./child")
        self.assertIs(right_child, relative_reference.getRelativeReference(left_child_attr))
        relative_reference = ja.RelativeReference("./child/./child")
        self.assertIs(right_child, relative_reference.getRelativeReference(left_child_attr))

        left_child_child = mock.Mock
        left_child.value = {"child": left_child_child}
        relative_reference = ja.RelativeReference("child")
        self.assertIs(left_child_child, relative_reference.getRelativeReference(left_child_attr))


class TestDesignerWidget(unittest.TestCase):
    def testObjectStack(self):
        parent = QtWidgets.QWidget()
        stack = d.ObjectStack(parent)
        event_handler = mock.Mock()
        stack.stack_changed.connect(event_handler)
        event_handler.assert_not_called()
        obj1 = mock.Mock()
        obj2 = mock.Mock()
        obj3 = mock.Mock()
        obj4 = mock.Mock()
        stack.addObject("First", obj1)
        self.assertEqual(event_handler.call_count, 1)
        self.assertEqual(stack.top(), obj1)
        stack.addObject("Second", obj2)
        stack.addObject("Third", obj3)
        self.assertEqual(stack.top(), obj3)
        self.assertEqual(event_handler.call_count, 3)
        stack.goDown(obj2)
        self.assertEqual(stack.top(), obj2)
        self.assertEqual(event_handler.call_count, 4)
        stack.addObject("Fourth", obj4)
        self.assertEqual(stack.top(), obj4)

        stack.goDown(obj1)
        stack.createUi()
        self.assertEqual(stack.layout.count(), 2)
        stack.addObject("New object", obj2)
        self.assertEqual(stack.layout.count(), 4)
        return_button = stack.layout.itemAt(0).widget()
        return_button.click()
        self.assertEqual(stack.top(), obj1)

    def testDesigner(self):
        parent = QtWidgets.QWidget()
        mock_schema = mock.Mock()
        mock_schema.createPanel = mock.Mock(return_value=QtWidgets.QWidget())
        mock_schema.json_value = mock.Mock(return_value="{'name': 'engine-x'}")
        mock_schema.been_set = TestSignal()
        mock_schema.resolveReferences = mock.Mock()
        d.Designer.createSchema = mock.Mock(return_value=mock_schema)
        designer = d.Designer(parent)
        mock_schema.resolveReferences.assert_called()

        new_path_handler = mock.Mock()
        data_changed_handler = mock.Mock()

        designer.new_relative_path.connect(new_path_handler)
        designer.data_changed.connect(data_changed_handler)
        array_val = designer.createAttributeArray(ja.FloatValue(), 3)
        self.assertIsInstance(array_val, ja.ValueArray)
        self.assertEqual(3, len(array_val.value))
        self.assertIsInstance(array_val.value[0], ja.FloatValue)
        self.assertIsInstance(array_val.value[1], ja.FloatValue)
        self.assertIsInstance(array_val.value[2], ja.FloatValue)
