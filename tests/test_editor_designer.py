import unittest
from unittest import mock
from PyQt5 import QtCore, QtWidgets
import sscanss.editor.json_attributes as im
from enum import Enum
from helpers import APP


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
        string_attr.has_changed.connect(m)
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
        float_attr.has_changed.connect(m)
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
        m1 = mock.MagicMock()
        m1.createWidget = mock.Mock(return_value=QtWidgets.QLineEdit())
        m1.has_changed = mock.MagicMock.return_value = "String"
        m1.value = "String2"
        m2 = mock.MagicMock()
        m2.createWidget = mock.Mock(return_value=QtWidgets.QDoubleSpinBox())
        m2.has_changed = mock.MagicMock.return_value = 42.3
        m2.has_changed = -5.6
        m3 = mock.MagicMock()
        m3.createWidget = mock.Mock(return_value=QtWidgets.QLabel())
        m3.has_changed = mock.Mock(return_value=6)
        m3.has_changed = 3
        arrayAttribute = im.JsonAttributeArray([m1, m2, m3])
        self.assertListEqual(arrayAttribute.attributes, [m1, m2, m3])
        widget = arrayAttribute.createWidget()
        self.assertIsInstance(widget.layout[0], QtWidgets.QLineEdit)
        self.assertIsInstance(widget.layout[1], QtWidgets.QDoubleSpinBox)
        self.assertIsInstance(widget.layout[2], QtWidgets.QLabel)


    def testAttributeEnum(self):
        enum_attr = im.JsonEnum(TestEnum)

        self.assertEqual(enum_attr.value, 0)
        test_index = 1
        enum_attr = im.JsonEnum(TestEnum, test_index)
        m = mock.Mock()
        enum_attr.has_changed.connect(m)
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



class TestDesignerWidget(unittest.TestCase):
    def testObjectStack(self):
        pass
