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

        # Here test the get/set properties
        new_string = "New string"
        string_val.setValue(new_string)
        self.assertEqual(string_val.value, new_string)
        edit_widget = string_val.createEditWidget()
        self.assertIsInstance(edit_widget, QtWidgets.QLineEdit)
        self.assertEqual(edit_widget.text(), new_string)
        self.assertEqual(string_val.json_value, new_string)
        m.assert_called_with(new_string)

        # Here test the created widget
        string_val.setValue("")
        m = mock.Mock()
        string_val.been_set.connect(m)
        ui_string = "Ui text"
        edit_widget = string_val.createEditWidget()
        QtTest.QTest.keyClicks(edit_widget, ui_string)
        self.assertEqual(len(ui_string), m.call_count)
        self.assertEqual(string_val.value, ui_string)

        # Test the json getter/setter
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
        m.assert_called_with(new_float)

        control_widget = float_attr.createEditWidget()
        self.assertIsInstance(control_widget, QtWidgets.QDoubleSpinBox)
        self.assertEqual(control_widget.value(), new_float)
        ui_float = 5.236
        m = mock.Mock()
        """
        float_attr.been_set.connect(m)
        self.assertAlmostEqual(float_attr.value, ui_float, 3)
        m.assert_called()
        self.assertEqual(float_attr.json_value, ui_float)"""
        json_float = -1.53
        float_attr.json_value = json_float
        self.assertEqual(float_attr.value, json_float)

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
        list_reference = mock.Mock()
        list_path = mock.Mock()
        list_path.getRelativeReference = mock.Mock(return_value=list_reference)
        attribute = ja.SelectedObject(list_path)


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
        another_num = -65
        json_value.been_set.emit(another_num)
        event_handler.assert_called_with(another_num)
        widget = attribute.createWidget()
        self.assertEqual(len(widget.layout), 2)

        mandatory = False
        attribute = ja.JsonAttribute(json_value, test_title, mandatory)
        event_handler = mock.Mock()
        attribute.been_set.connect(event_handler)
        self.assertTrue(attribute.turned_on)
        widget = attribute.createWidget()
        self.assertEqual(len(widget.layout), 3)
        checkbox = widget.layout.itemAt(2).widget()
        self.assertTrue(checkbox.isChecked())
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
        self.assertListEqual(attr_list, [json_attributes.attributes["key"],
                                         json_attributes.attributes["key2"]])

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
