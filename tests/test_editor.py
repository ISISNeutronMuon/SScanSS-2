from collections import namedtuple
import unittest
import unittest.mock as mock
from PyQt5.QtWidgets import QApplication
from editor.main import Controls, Window
from editor.ui.scene_manager import SceneManager
from editor.ui.widgets import PositionerWidget, JawsWidget, ScriptWidget, DetectorWidget
from sscanss.core.instrument.instrument import Instrument, PositioningStack, Detector, Script, Jaws
from sscanss.core.instrument.robotics import Link, SerialManipulator
from tests.helpers import TestSignal


Collimator = namedtuple('Collimator', ['name'])


class TestEditor(unittest.TestCase):
    app = QApplication([])

    def setUp(self):
        self.view = Window()
        self.view.animate_instrument = TestSignal()
        self.view.filename = ''
        self.view.instrument = mock.create_autospec(Instrument)

        q1 = Link('', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link('', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q3 = Link('', [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)

        s1 = PositioningStack('a', SerialManipulator('a', [q1, q2]))
        s1.addPositioner(SerialManipulator('c', [q3]))
        s2 = PositioningStack('b', SerialManipulator('b', [q2, q1]))

        self.view.instrument.positioning_stacks = {s1.name: s1, s2.name: s2}
        self.view.instrument.positioning_stack = s1
        self.view.instrument.jaws = Jaws('', [0, 0, 0], [1, 0, 0], [1.0, 1.0], [0.5, 0.5], [4.0, 4.0], None)
        self.view.instrument.detectors = {'East': Detector('East', [1, 0, 0])}
        self.view.instrument.script = Script('{{header}}\n{{#script}}\n{{position}}    {{mu_amps}}\n{{/script}}')

        self.view.manager = mock.create_autospec(SceneManager)

    def testPositionerWidget(self):
        widget = PositionerWidget(self.view)
        widget.changeStack('b')
        self.view.instrument.loadPositioningStack.assert_called_with('b')

        self.assertEqual(self.view.instrument.positioning_stack.set_points[0], 0)
        widget.positioner_form_controls[0].setValue(50)
        widget.move_joints_button.click()
        self.assertEqual(self.view.instrument.positioning_stack.set_points[0], 50)

    def testJawsWidget(self):
        widget = JawsWidget(self.view)
        self.assertEqual(widget.aperture_forms[0].value(), 1)
        widget.aperture_forms[0].setValue(3)
        widget.change_aperture_button.click()
        self.assertEqual(widget.aperture_forms[0].value(), 3)

        q1 = Link('', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link('', [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        self.view.instrument.jaws.positioner = SerialManipulator('a', [q1, q2])
        widget = JawsWidget(self.view)
        self.assertEqual(self.view.instrument.jaws.positioner.set_points[0], 0)
        widget.position_forms[0].setValue(50)
        widget.move_jaws_button.click()
        self.assertEqual(self.view.instrument.jaws.positioner.set_points[0], 50)

    def testDetectorWidget(self):
        widget = DetectorWidget(self.view, 'East')
        self.assertEqual(widget.collimator_name, 'None')

        c = {'1': Collimator('1'), '2': Collimator('2')}
        self.view.instrument.detectors = {'East': Detector('East', [1, 0, 0], c)}

        widget = DetectorWidget(self.view, 'East')
        widget.combobox.setCurrentIndex(1)
        widget.changeCollimator()
        self.assertEqual(widget.collimator_name, '1')

        q1 = Link('', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link('', [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        positioner = SerialManipulator('a', [q1, q2])
        self.view.instrument.detectors = {'East': Detector('East', [1, 0, 0], c, positioner)}
        widget = DetectorWidget(self.view, 'East')

        self.assertEqual(self.view.instrument.detectors['East'].positioner.set_points[0], 0)
        widget.position_forms[0].setValue(50)
        widget.move_detector_button.click()
        self.assertEqual(self.view.instrument.detectors['East'].positioner.set_points[0], 50)

    def testScriptWidget(self):
        widget = ScriptWidget(self.view)
        self.assertNotEqual(widget.preview_label.toPlainText(), '')
        widget.updateScript()

        self.view.instrument.script = Script('{{header}}\n{{#script}}\n{{position}}\n{{/script}}')
        self.widget = ScriptWidget(self.view)
        self.assertNotEqual(widget.preview_label.toPlainText(), '')

    def testControlsDialog(self):
        widget = Controls(self.view)
        widget.createWidgets()
        self.assertEqual(self.view.instrument.positioning_stack.name, 'a')
        self.assertEqual(widget.last_stack_name, '')
        widget.setStack('b')
        widget.createWidgets()
        self.assertEqual(widget.last_stack_name, 'b')

        self.assertDictEqual(widget.last_collimator_name, {})
        widget.setCollimator('East', '1')
        widget.createWidgets()
        self.assertDictEqual(widget.last_collimator_name, {'East': '1'})

        self.assertEqual(widget.last_tab_index, 0)
        self.view.instrument.script = Script('{{header}}\n{{#script}}\n{{position}}\n{{/script}}')
        text = widget.script_widget.preview_label.toPlainText()
        widget.updateTabs(4)
        widget.createWidgets()
        self.assertEqual(widget.last_tab_index, 4)
        self.assertNotEqual(text, widget.script_widget.preview_label.toPlainText())

        widget.reset()
        self.assertEqual(widget.last_stack_name, '')
        self.assertDictEqual(widget.last_collimator_name, {})
        self.assertEqual(widget.last_tab_index, 0)
