import unittest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from sscanss.ui.widgets import FormGroup, FormControl
from sscanss.core.util import CompareOperator


class TestFormWidgets(unittest.TestCase):
    app = QApplication([])

    def setUp(self):
        self.form_group = FormGroup()

        self.name = FormControl('Name', ' ', required=True)
        self.email = FormControl('Email')

        self.height = FormControl('Height', 0.0, required=True, unit='cm')
        self.weight = FormControl('Weight', 0.0, required=True, unit='kg')
        self.height.number = True
        self.weight.number = True

        self.form_group.addControl(self.name)
        self.form_group.addControl(self.email)
        self.form_group.addControl(self.height)
        self.form_group.addControl(self.weight)

    def testRequiredValidation(self):
        self.assertEqual(self.name.value, ' ')
        self.assertFalse(self.name.valid)
        self.assertTrue(self.email.valid)
        self.assertTrue(self.weight.valid)
        self.assertTrue(self.height.valid)

    def testGroupValidation(self):
        self.assertFalse(self.form_group.validateGroup())
        self.name.text = 'Space'
        self.assertTrue(self.form_group.validateGroup())

    def testRangeValidation(self):
        self.weight.minimum = 80
        self.weight.maximum = 100
        self.assertFalse(self.weight.valid)
        self.weight.value = 81
        self.assertTrue(self.weight.valid)
        self.weight.value = 100
        self.assertTrue(self.weight.valid)
        self.weight.value = 80
        self.assertTrue(self.weight.valid)

        self.weight.range(80, 100, True, True)
        self.weight.value = 100
        self.assertFalse(self.weight.valid)
        self.weight.value = 80
        self.assertFalse(self.weight.valid)

    def testCompareValidation(self):
        self.weight.compareWith(self.height, CompareOperator.Less)
        self.assertFalse(self.weight.valid)
        self.weight.value = -1
        self.assertTrue(self.weight.valid)

        self.weight.compareWith(self.height, CompareOperator.Greater)
        self.assertFalse(self.weight.valid)
        self.weight.value = 5
        self.assertTrue(self.weight.valid)

        self.weight.compareWith(self.height, CompareOperator.Not_Equal)
        self.assertTrue(self.weight.valid)
        self.weight.value = 0.0
        self.assertFalse(self.weight.valid)

        self.weight.compareWith(self.height, CompareOperator.Equal)
        self.assertTrue(self.weight.valid)
        self.weight.value = -1
        self.assertFalse(self.weight.valid)

    def testNumberValidation(self):
        with self.assertRaises(ValueError):
            self.weight.value = '.'

        self.height.text = '.'
        self.assertFalse(self.height.valid)
        with self.assertRaises(ValueError):
            self.height.value
