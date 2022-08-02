import unittest
from unittest import mock
from PyQt5 import QtCore, QtWidgets, QtGui
import sscanss.editor.json_attributes as im
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
    def testStringAttribute(self):
        string_attr = im.JsonString()

        self.assertEqual(string_attr.value, '')
        test_string = "This is a string"
        string_attr = im.JsonString(test_string)
        m = mock.Mock()
        string_attr.been_set.connect(m)
        self.assertEqual(string_attr.value, test_string)
        new_string = "New string"
        string_attr.setValue(new_string)
        self.assertEqual(string_attr.value, new_string)
        control_widget = string_attr.createWidget()
        self.assertIsInstance(control_widget, QtWidgets.QLineEdit)
        self.assertEqual(control_widget.text(), new_string)
        self.assertEqual(string_attr.json_value, new_string)
        m.assert_called_with(new_string)
        json_string = "Json string"
        string_attr.json_value = json_string
        self.assertEqual(string_attr.value, json_string)
        self.assertEqual(m.call_count, 2)
        self.assertListEqual(m.call_args_list, [mock.call(new_string), mock.call(json_string)])

        copy_string = string_attr.defaultCopy()
        self.assertEqual(copy_string.value, '')
        self.assertIsInstance(copy_string, im.JsonString)

    def testFloatAttribute(self):
        float_attr = im.JsonFloat()

        self.assertEqual(float_attr.value, 0.0)
        test_float = 5.0
        float_attr = im.JsonFloat(test_float)
        m = mock.Mock()
        float_attr.been_set.connect(m)
        self.assertEqual(float_attr.value, test_float)
        new_float = 7.32
        float_attr.setValue(new_float)
        self.assertEqual(float_attr.value, new_float)
        control_widget = float_attr.createWidget()
        self.assertIsInstance(control_widget, QtWidgets.QDoubleSpinBox)
        self.assertEqual(control_widget.value(), new_float)
        self.assertEqual(float_attr.json_value, new_float)
        m.assert_called_with(new_float)
        json_float = -1.53
        float_attr.json_value = json_float
        self.assertEqual(float_attr.value, json_float)
        self.assertEqual(m.call_count, 2)
        self.assertListEqual(m.call_args_list, [mock.call(new_float), mock.call(json_float)])

        copy_float = float_attr.defaultCopy()
        self.assertEqual(copy_float.value, 0.0)
        self.assertIsInstance(copy_float, im.JsonFloat)

    def testAttributeArray(self):
        string_val = "String"
        float_val = -53.2
        int_val = 423

        m1 = mock.Mock()
        m1.createWidget = mock.Mock(return_value=QtWidgets.QLineEdit())
        m1.has_changed = TestSignal()
        m1.value = string_val
        m2 = mock.MagicMock()
        m2.createWidget = mock.Mock(return_value=QtWidgets.QDoubleSpinBox())
        m2.has_changed = TestSignal()
        m2.value = float_val
        m3 = mock.MagicMock()
        m3.createWidget = mock.Mock(return_value=QtWidgets.QLabel())
        m3.has_changed = TestSignal()
        m3.value = int_val
        array_attribute = im.JsonAttributeArray([m1, m2, m3])
        mock_event_handler = mock.Mock()
        array_attribute.been_set.connect(mock_event_handler)
        self.assertListEqual(array_attribute.attributes, [m1, m2, m3])
        widget = array_attribute.createWidget()
        self.assertEqual(widget.layout.count(), 3)
        self.assertListEqual(array_attribute.value, [string_val, float_val, int_val])
        mock_event_handler.assert_not_called()
        m1.has_changed.emit("New")
        m2.has_changed.emit(976.3)
        m3.has_changed.emit(-53)
        self.assertEqual(mock_event_handler.call_count, 3)
        self.assertListEqual(mock_event_handler.call_args_list, [mock.call("New"), mock.call(976.3), mock.call(-53)])

    def testEnumAttribute(self):
        enum_attr = im.JsonEnum(TestEnum)

        self.assertEqual(enum_attr.value, 0)
        test_index = 1
        enum_attr = im.JsonEnum(TestEnum, test_index)
        m = mock.Mock()
        enum_attr.been_set.connect(m)
        self.assertEqual(enum_attr.value, test_index)
        new_index = 0
        enum_attr.setValue(0)
        self.assertEqual(enum_attr.value, new_index)
        control_widget = enum_attr.createWidget()
        self.assertIsInstance(control_widget, QtWidgets.QComboBox)
        self.assertEqual(control_widget.currentIndex(), new_index)
        self.assertEqual(enum_attr.json_value, "Zero")
        m.assert_called_with(new_index)
        json_value = "Three"
        enum_attr.json_value = json_value
        self.assertEqual(enum_attr.value, 3)
        self.assertEqual(m.call_count, 2)
        self.assertListEqual(m.call_args_list, [mock.call(new_index), mock.call(3)])

        copy_enum = enum_attr.defaultCopy()
        self.assertEqual(copy_enum.value, 0.0)
        self.assertIsInstance(copy_enum, im.JsonEnum)

    def testColourAttribute(self):

        colour_attr = im.JsonColour()
        self.assertEqual(colour_attr.value, QtGui.QColor())
        test_colour = QtGui.QColor(53, 12, 154)
        colour_attr = im.JsonColour(test_colour)
        event_handler = mock.Mock()
        colour_attr.been_set.connect(event_handler)
        self.assertEqual(colour_attr.value, test_colour)
        new_colour = QtGui.QColor(43, 11, 211)
        colour_attr.setValue(new_colour)
        self.assertEqual(colour_attr.value, new_colour)
        control_widget = colour_attr.createWidget()
        self.assertIsInstance(control_widget, ColourPicker)
        self.assertEqual(control_widget.value, new_colour)

        rgb_size = 255

        self.assertEqual(colour_attr.json_value, [new_colour.redF()/rgb_size, new_colour.greenF()/rgb_size,
                                                  new_colour.blueF()/rgb_size])
        json_colour = [0.4, 0.8, 0.06]
        colour_attr.json_value = json_colour
        actual_json_colour = QtGui.QColor(int(json_colour[0]*rgb_size), int(json_colour[1]*rgb_size),
                                          int(json_colour[2]*rgb_size))
        self.assertEqual(colour_attr.value, actual_json_colour)
        self.assertEqual(event_handler.call_count, 2)
        self.assertListEqual(event_handler.call_args_list, [mock.call(new_colour), mock.call(actual_json_colour)])

        copy_colour = colour_attr.defaultCopy()
        self.assertEqual(copy_colour.value, QtGui.QColor())
        self.assertIsInstance(copy_colour, im.JsonColour)

    def testObjectReferenceAttribute(self):
        reference_attr = im.JsonObjectReference("/./.")

    def testObjectOrderAttribute(self):
        pass

    def testObjectList(self):
        pass

    def testObject(self):
        pass

    def testDirectlyEditableObject(self):
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
