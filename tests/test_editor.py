from collections import namedtuple
import unittest
import unittest.mock as mock
import numpy as np
from PyQt5.QtWidgets import QLineEdit, QComboBox, QDoubleSpinBox
from sscanss.core.instrument.instrument import Instrument, PositioningStack, Detector, Script, Jaws
from sscanss.core.instrument.robotics import Link, SerialManipulator
from sscanss.editor.main import Window
from sscanss.editor.widgets import PositionerWidget, JawsWidget, ScriptWidget, DetectorWidget
from sscanss.editor.dialogs import CalibrationWidget, Controls, FindWidget
from tests.helpers import TestSignal, APP


Collimator = namedtuple("Collimator", ["name"])


class TestEditor(unittest.TestCase):
    @mock.patch("sscanss.editor.main.SceneManager", autospec=True)
    def setUp(self, scene_mock):
        self.view = Window()
        self.view.animate_instrument = TestSignal()
        self.view.filename = ""
        self.view.instrument = mock.create_autospec(Instrument)

        q1 = Link("", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q3 = Link("", [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)

        s1 = PositioningStack("a", SerialManipulator("a", [q1, q2]))
        s1.addPositioner(SerialManipulator("c", [q3]))
        s2 = PositioningStack("b", SerialManipulator("b", [q2, q1]))

        self.view.instrument.positioning_stacks = {s1.name: s1, s2.name: s2}
        self.view.instrument.positioning_stack = s1
        self.view.instrument.jaws = Jaws("", [0, 0, 0], [1, 0, 0], [1.0, 1.0], [0.5, 0.5], [4.0, 4.0], None)
        self.view.instrument.detectors = {"East": Detector("East", [1, 0, 0])}
        self.view.instrument.script = Script("{{header}}\n{{#script}}\n{{position}}    {{mu_amps}}\n{{/script}}")

    def testPositionerWidget(self):
        widget = PositionerWidget(self.view)
        widget.changeStack("b")
        self.view.instrument.loadPositioningStack.assert_called_with("b")

        self.assertEqual(self.view.instrument.positioning_stack.set_points[0], 0)
        widget.positioner_form_controls[0].setValue(50)
        widget.move_joints_button.click()
        self.assertEqual(self.view.instrument.positioning_stack.set_points[0], 50)

    def testFindInText(self):
        # Testing search works, and only finds one occurrence
        window = self.view
        window.editor.setText("The Text")
        widget = FindWidget(window)
        widget.search_box.setText("text")
        widget.search()
        self.assertEqual(window.editor.selectedText(), "Text")
        widget.search()
        self.assertEqual(widget.status_box.text(), "No more entries found.")
        # Testing match case
        window.editor.selectAll(False)
        widget.match_case.click()
        widget.search()
        self.assertNotEqual(window.editor.selectedText(), "Text")
        widget.search_box.setText("Text")
        window.editor.selectAll(False)
        widget.search()
        self.assertEqual(window.editor.selectedText(), "Text")
        widget.match_case.click()
        # Testing whole word
        widget.whole_word.click()
        window.editor.selectAll(False)
        widget.search()
        self.assertEqual(window.editor.selectedText(), "Text")
        widget.search_box.setText("Tex")
        window.editor.selectAll(False)
        widget.search()
        self.assertNotEqual(window.editor.selectedText(), "Text")
        widget.whole_word.click()
        # Testing empty search string
        window.editor.setText("The Text")
        widget.search_box.setText("")
        window.editor.selectAll(False)
        widget.search()
        self.assertEqual(widget.status_box.text(), "No more entries found.")
        # Testing empty search string and empty window text
        window.editor.setText("")
        widget.search_box.setText("")
        window.editor.selectAll(False)
        widget.search()
        self.assertEqual(widget.status_box.text(), "No more entries found.")


    def testJawsWidget(self):
        widget = JawsWidget(self.view)
        self.assertEqual(widget.aperture_forms[0].value(), 1)
        widget.aperture_forms[0].setValue(3)
        widget.change_aperture_button.click()
        self.assertEqual(widget.aperture_forms[0].value(), 3)

        q1 = Link("", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link("", [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        self.view.instrument.jaws.positioner = SerialManipulator("a", [q1, q2])
        widget = JawsWidget(self.view)
        self.assertEqual(self.view.instrument.jaws.positioner.set_points[0], 0)
        widget.position_forms[0].setValue(50)
        widget.move_jaws_button.click()
        self.assertEqual(self.view.instrument.jaws.positioner.set_points[0], 50)

    def testDetectorWidget(self):
        widget = DetectorWidget(self.view, "East")
        self.assertEqual(widget.collimator_name, "None")

        c = {"1": Collimator("1"), "2": Collimator("2")}
        self.view.instrument.detectors = {"East": Detector("East", [1, 0, 0], c)}

        widget = DetectorWidget(self.view, "East")
        widget.combobox.setCurrentIndex(1)
        widget.changeCollimator()
        self.assertEqual(widget.collimator_name, "1")

        q1 = Link("", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link("", [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        positioner = SerialManipulator("a", [q1, q2])
        self.view.instrument.detectors = {"East": Detector("East", [1, 0, 0], c, positioner)}
        widget = DetectorWidget(self.view, "East")

        self.assertEqual(self.view.instrument.detectors["East"].positioner.set_points[0], 0)
        widget.position_forms[0].setValue(50)
        widget.move_detector_button.click()
        self.assertEqual(self.view.instrument.detectors["East"].positioner.set_points[0], 50)

    def testScriptWidget(self):
        widget = ScriptWidget(self.view)
        self.assertNotEqual(widget.preview_label.toPlainText(), "")
        widget.updateScript()

        self.view.instrument.script = Script("{{header}}\n{{#script}}\n{{position}}\n{{/script}}")
        self.widget = ScriptWidget(self.view)
        self.assertNotEqual(widget.preview_label.toPlainText(), "")

    def testControlsDialog(self):
        widget = Controls(self.view)
        widget.createWidgets()
        self.assertEqual(self.view.instrument.positioning_stack.name, "a")
        self.assertEqual(widget.last_stack_name, "")
        widget.setStack("b")
        widget.createWidgets()
        self.assertEqual(widget.last_stack_name, "b")

        self.assertDictEqual(widget.last_collimator_name, {})
        widget.setCollimator("East", "1")
        widget.createWidgets()
        self.assertDictEqual(widget.last_collimator_name, {"East": "1"})

        self.assertEqual(widget.last_tab_index, 0)
        self.view.instrument.script = Script("{{header}}\n{{#script}}\n{{position}}\n{{/script}}")
        text = widget.script_widget.preview_label.toPlainText()
        widget.updateTabs(3)
        widget.createWidgets()
        self.assertEqual(widget.last_tab_index, 3)
        self.assertNotEqual(text, widget.script_widget.preview_label.toPlainText())

        widget.reset()
        self.assertEqual(widget.last_stack_name, "")
        self.assertDictEqual(widget.last_collimator_name, {})
        self.assertEqual(widget.last_tab_index, 0)

    @mock.patch("sscanss.editor.dialogs.QtWidgets.QFileDialog", autospec=True)
    def testCalibrationDialog(self, file_dialog):
        points = [
            np.array([[12.0, 0.0, 2.5], [10.0, 2.0, 1.5], [8.0, 0.0, 1.5]]),
            np.array([[10.0, 0.0, 1.5], [11.0, -1.0, 1.5], [12.0, 0.0, 1.5]]),
        ]

        offsets = [np.array([0.0, 90.0, 180.0]), np.array([-180.0, -90.0, 0.0])]
        types = [Link.Type.Revolute, Link.Type.Revolute]
        homes = np.array([10.0, 0.0])

        widget = CalibrationWidget(self.view, points, types, offsets, homes)
        widget.calibrate_button.click()

        line_edits = widget.findChildren(QLineEdit)

        line_edits[0].setText("")
        self.assertFalse(widget.calibrate_button.isEnabled())
        self.assertEqual(widget.robot_name, "")
        line_edits[0].setText("Two link")
        self.assertTrue(widget.calibrate_button.isEnabled())
        self.assertEqual(widget.robot_name, "Two link")

        line_edits[1].setText("")
        self.assertFalse(widget.calibrate_button.isEnabled())
        line_edits[1].setText("3, 1")
        self.assertListEqual(widget.order, [])
        self.assertFalse(widget.calibrate_button.isEnabled())
        line_edits[1].setText("b, a")
        self.assertListEqual(widget.order, [])
        self.assertFalse(widget.calibrate_button.isEnabled())
        line_edits[1].setText("3, 1, 2")
        self.assertListEqual(widget.order, [])
        self.assertFalse(widget.calibrate_button.isEnabled())
        line_edits[1].setText("2, 1")
        self.assertListEqual(widget.order, [1, 0])

        line_edits[2].setText("")
        self.assertFalse(widget.calibrate_button.isEnabled())
        self.assertEqual(widget.names[0], "")
        line_edits[2].setText("a")
        self.assertTrue(widget.calibrate_button.isEnabled())
        line_edits[4].setText("a")
        self.assertFalse(widget.calibrate_button.isEnabled())
        line_edits[4].setText("b")
        self.assertTrue(widget.calibrate_button.isEnabled())
        self.assertEqual(widget.names[0], "a")

        self.assertEqual(widget.types[1], Link.Type.Revolute)
        widget.findChildren(QComboBox)[1].setCurrentIndex(1)
        self.assertEqual(widget.types[1], Link.Type.Prismatic)

        self.assertEqual(widget.homes[0], 10.0)
        widget.findChildren(QDoubleSpinBox)[0].setValue(0.0)
        self.assertEqual(widget.homes[0], 0.0)

        self.assertEqual(widget.model_error_table.rowCount(), 6)
        widget.filter_combobox.setCurrentIndex(1)
        self.assertEqual(widget.model_error_table.rowCount(), 3)
        widget.tabs.setCurrentIndex(1)
        self.assertEqual(widget.fit_error_table.rowCount(), 3)
        widget.filter_combobox.setCurrentIndex(0)
        self.assertEqual(widget.fit_error_table.rowCount(), 6)

        widget.json = {"name": "Json"}
        widget.copy_model_button.click()
        self.assertEqual("".join(APP.clipboard().text().split()), '{"name":"Json"}')

        file_dialog.getSaveFileName.return_value = ("", "")
        widget.save_model_button.click()
        file_dialog.getSaveFileName.return_value = ("file.json", "")
        m = mock.mock_open()
        with mock.patch("sscanss.editor.dialogs.open", m):
            widget.save_model_button.click()
            m.assert_called_once()
