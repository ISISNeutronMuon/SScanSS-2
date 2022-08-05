import unittest
from unittest import mock
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
        m = mock.Mock()
        string_val.been_set.connect(m)
        self.assertEqual(string_val.value, test_string)

        new_string = "New string"
        string_val.setValue(new_string)
        self.assertEqual(string_val.value, new_string)
        edit_widget = string_val.createEditWidget()
        self.assertIsInstance(edit_widget, QtWidgets.QLineEdit)
        self.assertEqual(edit_widget.text(), new_string)
        self.assertEqual(string_val.json_value, new_string)
        m.assert_called_with(new_string)

        string_val.setValue("")
        m = mock.Mock()
        string_val.been_set.connect(m)
        ui_string = "Ui text"
        edit_widget = string_val.createEditWidget()
        QtTest.QTest.keyClicks(edit_widget, ui_string)
        self.assertEqual(len(ui_string), m.call_count)
        self.assertEqual(string_val.value, ui_string)

        m = mock.Mock()
        string_val.been_set.connect(m)
        json_string = "Json string"
        string_val.json_value = json_string
        self.assertEqual(string_val.value, json_string)
        m.assert_not_called()

        copy_string = string_val.defaultCopy()
        self.assertEqual(copy_string.value, '')
        self.assertIsInstance(copy_string, ja.StringValue)

    def testFloatAttribute(self):
        float_attr = ja.FloatValue()

        self.assertEqual(float_attr.value, 0.0)
        test_float = 5.0
        float_attr = ja.FloatValue(test_float)
        m = mock.Mock()
        float_attr.been_set.connect(m)
        self.assertEqual(float_attr.value, test_float)
        new_float = 7.32
        float_attr.setValue(new_float)
        self.assertEqual(float_attr.value, new_float)
        control_widget = float_attr.createEditWidget()
        self.assertIsInstance(control_widget, QtWidgets.QDoubleSpinBox)
        self.assertEqual(control_widget.value(), new_float)
        self.assertEqual(float_attr.json_value, new_float)
        m.assert_called_with(new_float)
        json_float = -1.53
        float_attr.json_value = json_float
        self.assertEqual(float_attr.value, json_float)
        m.assert_called_with(new_float)

        copy_float = float_attr.defaultCopy()
        self.assertEqual(copy_float.value, 0.0)
        self.assertIsInstance(copy_float, ja.FloatValue)

    def testAttributeArray(self):
        string_val = "String"
        float_val = -53.2
        int_val = 423

        m1 = mock.Mock()
        m1.createEditWidget = mock.Mock(return_value=QtWidgets.QLineEdit())
        m1.been_set = TestSignal()
        m1.json_value = string_val
        m2 = mock.MagicMock()
        m2.createEditWidget = mock.Mock(return_value=QtWidgets.QDoubleSpinBox())
        m2.been_set = TestSignal()
        m2.json_value = float_val
        m3 = mock.MagicMock()
        m3.createEditWidget = mock.Mock(return_value=QtWidgets.QLabel())
        m3.been_set = TestSignal()
        m3.json_value = int_val
        array_attribute = ja.ValueArray([m1, m2, m3])
        mock_event_handler = mock.Mock()
        array_attribute.been_set.connect(mock_event_handler)
        self.assertListEqual(array_attribute.value, [m1, m2, m3])
        widget = array_attribute.createEditWidget()
        self.assertEqual(widget.layout.count(), 3)
        self.assertListEqual(array_attribute.json_value, [string_val, float_val, int_val])
        mock_event_handler.assert_not_called()
        m1.been_set.emit("New")
        m2.been_set.emit(976.3)
        m3.been_set.emit(-53)
        self.assertEqual(mock_event_handler.call_count, 3)
        self.assertListEqual(mock_event_handler.call_args_list, [mock.call("New"), mock.call(976.3), mock.call(-53)])

    def testEnumAttribute(self):
        enum_attr = ja.EnumValue(TestEnum)

        self.assertEqual(enum_attr.value, 0)
        test_index = 1
        enum_attr = ja.EnumValue(TestEnum, test_index)
        m = mock.Mock()
        enum_attr.been_set.connect(m)
        self.assertEqual(enum_attr.value, test_index)
        new_index = 0
        enum_attr.setValue(0)
        self.assertEqual(enum_attr.value, new_index)
        control_widget = enum_attr.createEditWidget()
        self.assertIsInstance(control_widget, QtWidgets.QComboBox)
        self.assertEqual(control_widget.currentIndex(), new_index)
        self.assertEqual(enum_attr.json_value, "Zero")
        m.assert_called_with(new_index)
        json_value = "Three"
        enum_attr.json_value = json_value
        self.assertEqual(enum_attr.value, 3)
        m.assert_called_with(new_index)

        copy_enum = enum_attr.defaultCopy()
        self.assertEqual(copy_enum.value, 0.0)
        self.assertIsInstance(copy_enum, ja.EnumValue)

    def testColourAttribute(self):

        colour_attr = ja.ColourValue()
        self.assertEqual(colour_attr.value, [0.0, 0.0, 0.0])
        test_colour = [0.12, 0.35, 0.58]
        colour_attr = ja.ColourValue(test_colour)
        event_handler = mock.Mock()
        colour_attr.been_set.connect(event_handler)
        self.assertEqual(colour_attr.value, test_colour)
        new_colour = [0.52, 0.86, 0.03]
        colour_attr.setValue(new_colour)
        event_handler.assert_called_with(new_colour)
        self.assertEqual(colour_attr.value, new_colour)
        control_widget = colour_attr.createEditWidget()
        self.assertIsInstance(control_widget, ColourPicker)
        rgb_size = 255
        self.assertEqual(control_widget.value, QtGui.QColor(int(new_colour[0]*rgb_size), int(new_colour[1]*rgb_size),
                                                            int(new_colour[2]*rgb_size)))
        json_colour = [0.4, 0.8, 0.06]
        colour_attr.json_value = json_colour
        self.assertEqual(colour_attr.value, json_colour)

        copy_colour = colour_attr.defaultCopy()
        self.assertEqual(copy_colour.value, [0.0, 0.0, 0.0])
        self.assertIsInstance(copy_colour, ja.ColourValue)

    def testObjectReferenceAttribute(self):
        mock_parent = mock.Mock()
        mock_child1 = mock.Mock()
        mock_child1.tree_parent = mock_parent
        mock_child2 = mock.Mock()
        mock_child2.tree_parent = mock_parent
        key_list = ["key1", "key2", "key3"]
        mock_child2.getObjectKeys = mock.Mock(return_value=key_list)
        mock_child2.objects = mock.MagicMock()
        mock_parent.attributes = {"child": mock_child2}

        reference_attr = ja.SelectedObject("./child")
        reference_attr.tree_parent = mock_child1
        event_handler = mock.MagicMock()
        reference_attr.been_set = event_handler
        self.assertEqual(reference_attr.object_array, mock_child2)
        widget = reference_attr.createEditWidget()
        self.assertIsInstance(widget, QtWidgets.QComboBox)
        self.assertEqual(reference_attr.value, key_list[0])
        self.assertEqual(widget.currentIndex(), 0)
        self.assertListEqual([widget.itemText(i) for i in range(widget.count())], key_list)
        self.assertEqual(widget.currentText(), reference_attr.value)
        widget.setCurrentIndex(2)
        event_handler.assert_called_with(2)
        self.assertEqual(widget.currentText(), "key2")
        self.assertEqual(reference_attr.value, "key2")

        copy_ref = reference_attr.defaultCopy()
        mock_parent_copy = mock.Mock()
        mock_child1_copy = mock.Mock()
        mock_child1_copy.tree_parent = mock_parent_copy
        mock_child2_copy = mock.Mock()
        mock_child2_copy.tree_parent = mock_parent_copy
        mock_child2_copy.getObjectKeys = mock.Mock(return_value=key_list)
        mock_parent_copy.attributes = {"child": mock_child2_copy}

        copy_ref.tree_parent = mock_child1_copy
        self.assertIsInstance(copy_ref, ja.SelectedObject)
        self.assertEqual(copy_ref.object_array, mock_child2_copy)

    def testObjectOrderAttribute(self):
        pass

    def testObjectList(self):
        pass

    def testObject(self):
        pass

    def testDirectlyEditableObject(self):
        pass

    def testJsonAttribute(self):
        num = 642
        test_title = "title"
        mandatory = True
        json_value = mock.Mock()
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

        another_num = -65
        json_value.been_set.emit(another_num)
        event_handler.assert_called_with(another_num)




    def testJsonAttributes(self):
        pass

    def testPathToObject(self):
        pass


class TestDesignerWidget(unittest.TestCase):
    def testObjectStack(self):
        stack = d.ObjectStack(None)
        event_handler = mock.Mock()
        stack.stackChanged.connect(event_handler)
        event_handler.assert_not_called()
        obj1 = mock.Mock()
        obj2 = mock.Mock()
        obj3 = mock.Mock()
        obj4 = mock.Mock()
        stack.addObject(obj1, "First")
        self.assertEqual(event_handler.call_count, 1)
        self.assertEqual(stack.top(), obj1)
        stack.addObject(obj2, "Second")
        stack.addObject(obj3, "Third")
        self.assertEqual(stack.top(), obj3)
        self.assertEqual(event_handler.call_count, 3)
        stack.goDown(obj2)
        self.assertEqual(stack.top(), obj2)
        self.assertEqual(event_handler.call_count, 4)
        stack.addObject(obj4)
        self.assertEqual(stack.top(), obj4)

    def testDesigner(self):
        pass
