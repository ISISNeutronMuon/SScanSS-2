import unittest
import unittest.mock as mock
from PyQt5.QtWidgets import QApplication
from sscanss.core.instrument.simulation import SimulationResult, Simulation
from sscanss.core.instrument.robotics import IKSolver, IKResult
from sscanss.core.instrument.instrument import Script
from sscanss.ui.dialogs import SimulationDialog, ScriptExportDialog
from sscanss.ui.widgets import FormGroup, FormControl, CompareValidator
from sscanss.ui.window.scene_manager import SceneManager
from sscanss.ui.window.presenter import MainWindowPresenter
from tests.helpers import TestView, TestSignal


class TestFormWidgets(unittest.TestCase):
    app = QApplication([])

    def setUp(self):
        self.form_group = FormGroup()

        self.name = FormControl('Name', ' ', required=True)
        self.email = FormControl('Email', '')

        self.height = FormControl('Height', 0.0, required=True, desc='cm', number=True)
        self.weight = FormControl('Weight', 0.0, required=True, desc='kg', number=True)

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
        self.weight.range(80, 100)
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
        self.weight.compareWith(self.height, CompareValidator.Operator.Less)
        self.assertFalse(self.weight.valid)
        self.weight.value = -1
        self.assertTrue(self.weight.valid)

        self.weight.compareWith(self.height, CompareValidator.Operator.Greater)
        self.assertFalse(self.weight.valid)
        self.weight.value = 5
        self.assertTrue(self.weight.valid)

        self.weight.compareWith(self.height, CompareValidator.Operator.Not_Equal)
        self.assertTrue(self.weight.valid)
        self.weight.value = 0.0
        self.assertFalse(self.weight.valid)

        self.weight.compareWith(self.height, CompareValidator.Operator.Equal)
        self.assertTrue(self.weight.valid)
        self.weight.value = -1
        self.assertFalse(self.weight.valid)

    def testNumberValidation(self):
        with self.assertRaises(ValueError):
            self.weight.value = '.'

        self.height.text = '.'
        self.assertFalse(self.height.valid)
        self.assertRaises(ValueError, lambda: self.height.value)


class TestSimulationDialog(unittest.TestCase):
    app = QApplication([])
    dummy = 'dummy'

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [self.dummy]
        self.model_mock.return_value.instrument.positioning_stack.name = self.dummy
        self.model_mock.return_value.simulation = None
        self.model_mock.return_value.simulation_created = TestSignal()
        self.presenter = MainWindowPresenter(self.view)

        self.simulation_mock = mock.create_autospec(Simulation)
        self.simulation_mock.positioner.name = self.dummy
        self.simulation_mock.validateInstrumentParameters.return_value = True
        self.simulation_mock.isRunning.return_value = True
        self.simulation_mock.detector_names = ['East']
        self.simulation_mock.result_updated = TestSignal()
        self.simulation_mock.render_graphics = True

        self.view.presenter = self.presenter
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.showSelectChoiceMessage = mock.Mock(return_value='Cancel')
        self.dialog = SimulationDialog(self.view)

    def testSimulationResult(self):
        converged = IKResult([90], IKSolver.Status.Converged, (0., 0.1, 0.), (0.1, 0., 0.), True, True)
        not_converged = IKResult([87.8], IKSolver.Status.Converged, (0., 0., 0.), (1., 1., 0.), True, False)
        non_fatal = IKResult([45], IKSolver.Status.Failed, (-1., -1., -1.), (-1., -1., -1.), False, False)
        self.simulation_mock.results = [SimulationResult('1', converged, (['X'], [90]), 0, (120,), [False, True]),
                                        SimulationResult('2', not_converged, (['X'], [87.8]), 0, (25,), [True, True]),
                                        SimulationResult('3', non_fatal, (['X'], [45]), 0, None, None)]
        self.simulation_mock.count = len(self.simulation_mock.results)
        self.simulation_mock.scene_size = 2

        self.model_mock.return_value.simulation = self.simulation_mock
        self.model_mock.return_value.simulation_created.emit()
        self.simulation_mock.result_updated.emit(False)
        self.assertEqual(len(self.dialog.result_list.panes), 3)
        actions = self.dialog.result_list.panes[0].context_menu.actions()
        actions[0].trigger()  # copy action
        self.assertEqual(self.app.clipboard().text(), '90.000')

        self.model_mock.return_value.moveInstrument.reset_mock()
        self.view.scenes.renderCollision.reset_mock()
        actions[1].trigger()  # visualize action
        self.model_mock.moveInstrument.assert_not_called()
        self.view.scenes.renderCollision.assert_not_called()
        self.simulation_mock.isRunning.return_value = False
        actions[1].trigger()
        self.model_mock.return_value.moveInstrument.assert_called()
        self.view.scenes.renderCollision.assert_called()

        self.model_mock.return_value.moveInstrument.reset_mock()
        self.view.scenes.renderCollision.reset_mock()
        self.simulation_mock.positioner.name = 'new'
        actions[1].trigger()
        self.model_mock.return_value.moveInstrument.assert_not_called()
        self.view.scenes.renderCollision.assert_not_called()
        self.simulation_mock.positioner.name = self.dummy
        self.simulation_mock.validateInstrumentParameters.return_value = False
        actions[1].trigger()
        self.model_mock.return_value.moveInstrument.assert_called()
        self.view.scenes.renderCollision.assert_not_called()

        self.simulation_mock.result_updated.emit(True)
        self.simulation_mock.isRunning.return_value = True
        self.dialog.close()
        self.simulation_mock.abort.assert_not_called()
        self.view.showSelectChoiceMessage.return_value = 'Stop'
        self.dialog.close()
        self.simulation_mock.abort.assert_called_once()


class TestScriptExportDialog(unittest.TestCase):
    app = QApplication([])
    dummy = 'dummy'

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [self.dummy]
        self.model_mock.return_value.save_path = self.dummy

        self.template_mock = mock.create_autospec(Script)
        self.template_mock.Key = Script.Key
        self.template_mock.keys = {key.value: '' for key in Script.Key}
        self.template_mock.header_order = [Script.Key.mu_amps.value, Script.Key.position.value]
        self.template_mock.render.return_value = self.dummy

        self.simulation_mock = mock.create_autospec(Simulation)
        converged = IKResult([90], IKSolver.Status.Converged, (0., 0.1, 0.), (0.1, 0., 0.), True, True)
        not_converged = IKResult([87.8], IKSolver.Status.Converged, (0., 0., 0.), (1., 1., 0.), True, False)
        non_fatal = IKResult([45], IKSolver.Status.Failed, (-1., -1., -1.), (-1., -1., -1.), False, False)
        self.model_mock.return_value.instrument.script = self.template_mock
        self.simulation_mock.results = [SimulationResult('1', converged, (['X'], [90]), 0, (120,), [False, True]),
                                        SimulationResult('2', not_converged, (['X'], [87.8]), 0, (25,), [True, True]),
                                        SimulationResult('3', non_fatal, (['X'], [45]), 0, None, None)]

        self.presenter = MainWindowPresenter(self.view)
        self.view.presenter = self.presenter
        self.dialog = ScriptExportDialog(self.simulation_mock, self.view)

    def testRendering(self):
        self.assertEqual(self.dialog.preview_label.toPlainText(), self.dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.filename.value], self.dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.mu_amps.value], '0.000')
        self.assertEqual(self.template_mock.keys[Script.Key.header.value], f'{Script.Key.mu_amps.value}\tX')
        self.assertEqual(self.template_mock.keys[Script.Key.count.value], 3)
        self.assertEqual(self.template_mock.keys[Script.Key.position.value], '')
        self.assertEqual(self.template_mock.keys[Script.Key.script.value],
                         [{'position': '90.000'}, {'position': '87.800'}, {'position': '45.000'}])

        for _ in range(8):
            self.simulation_mock.results.append(self.simulation_mock.results[0])

        self.dialog = ScriptExportDialog(self.simulation_mock, self.view)
        self.dialog.micro_amp_textbox.setText('4.500')
        self.dialog.micro_amp_textbox.textEdited.emit(self.dialog.micro_amp_textbox.text())
        self.assertEqual(self.template_mock.keys[Script.Key.mu_amps.value], self.dialog.micro_amp_textbox.text())

        self.assertEqual(self.template_mock.keys[Script.Key.count.value], 11)
        self.assertNotEqual(self.dialog.preview_label.toPlainText(), self.dummy)
        self.assertTrue(self.dialog.preview_label.toPlainText().endswith(self.dummy))

        self.presenter.exportScript = mock.Mock(return_value=False)
        self.dialog.export()
        self.presenter.exportScript.assert_called_once()
        self.assertEqual(self.dialog.result(), 0)

        self.presenter.exportScript = mock.Mock(return_value=True)
        self.dialog.export()
        self.presenter.exportScript.assert_called()
        self.assertEqual(self.dialog.result(), 1)
