import unittest
from PyQt5 import QtCore, QtWidgets
import sscanss.editor.InstrumentModel as im

class TestDesignerTree(unittest.TestCase):
    def setUp(self):
        pass

    def test_string_attribute(self):
        string_attr = im.JsonString()
        self.assertEqual(string_attr.value, '')
        test_string = "This is a string"
        string_attr = im.JsonString(test_string)
        self.assertEqual(string_attr.value, test_string)
        label = QtWidgets.QLabel(test_string)
        self.assertEqual(string_attr.createWidget(), label)

