import unittest
from unittest import mock
from PyQt5 import QtCore, QtWidgets
import sscanss.editor.json_attributes as im


class TestDesignerTree(unittest.TestCase):
    def setUp(self):
        pass

    def testStringAttribute(self):
        string_attr = im.JsonString()

        self.assertEqual(string_attr.value, '')
        test_string = "This is a string"
        string_attr = im.JsonString(test_string)
        m = mock.Mock()
        string_attr.has_changed = m
        self.assertEqual(string_attr.value, test_string)
        self.assertIsInstance(string_attr.createWidget(), QtWidgets.QLineEdit)
        new_string = "New string"
        string_attr.setValue(new_string)
        self.assertEqual(string_attr.value, new_string)
        self.assertEqual(string_attr.getJsonValue(), new_string)
        json_string = "Json string"
        string_attr.setJsonValue(json_string)
        self.assertEqual(string_attr.value, json_string)
        m.assert_called_with(new_string, json_string)

        copy_string = string_attr.defaultCopy()
        self.assertEqual(copy_string.value, '')
        self.assertIsInstance(copy_string, im.JsonString)
