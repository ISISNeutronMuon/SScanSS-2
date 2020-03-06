import unittest
import unittest.mock as mock
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QEvent
from PyQt5.QtGui import QColor, QMouseEvent
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QLabel, QAction
from sscanss.core.instrument.simulation import SimulationResult, Simulation
from sscanss.core.instrument.robotics import IKSolver, IKResult
from sscanss.core.instrument.instrument import Script
from sscanss.ui.dialogs import SimulationDialog, ScriptExportDialog, PathLengthPlotter, SampleExportDialog
from sscanss.ui.widgets import (FormGroup, FormControl, CompareValidator, StatusBar, ColourPicker, FileDialog,
                                FilePicker, Accordion, Pane)
from sscanss.ui.window.scene_manager import SceneManager
from sscanss.ui.window.presenter import MainWindowPresenter
from tests.helpers import TestView, TestSignal

dummy = 'dummy'


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

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument.positioning_stack.name = dummy
        self.model_mock.return_value.simulation = None
        self.model_mock.return_value.simulation_created = TestSignal()
        self.presenter = MainWindowPresenter(self.view)

        self.simulation_mock = mock.create_autospec(Simulation)
        self.simulation_mock.positioner.name = dummy
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
        self.simulation_mock.positioner.name = dummy
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

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.save_path = dummy

        self.template_mock = mock.create_autospec(Script)
        self.template_mock.Key = Script.Key
        self.template_mock.keys = {key.value: '' for key in Script.Key}
        del self.template_mock.keys[Script.Key.mu_amps.value]
        self.template_mock.header_order = [Script.Key.mu_amps.value, Script.Key.position.value]
        self.template_mock.render.return_value = dummy

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
        self.assertEqual(self.dialog.preview_label.toPlainText(), dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.filename.value], dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.header.value], f'{Script.Key.mu_amps.value}\tX')
        self.assertEqual(self.template_mock.keys[Script.Key.count.value], 3)
        self.assertEqual(self.template_mock.keys[Script.Key.position.value], '')
        self.assertEqual(self.template_mock.keys[Script.Key.script.value],
                         [{'position': '90.000'}, {'position': '87.800'}, {'position': '45.000'}])

        self.assertFalse(self.dialog.show_mu_amps)
        self.assertFalse(hasattr(self.dialog, 'micro_amp_textbox'))

        for _ in range(8):
            self.simulation_mock.results.append(self.simulation_mock.results[0])

        self.template_mock.keys[Script.Key.mu_amps.value] = ''
        self.dialog = ScriptExportDialog(self.simulation_mock, self.view)
        self.assertEqual(self.template_mock.keys[Script.Key.mu_amps.value], '0.000')
        self.dialog.micro_amp_textbox.setText('4.500')
        self.dialog.micro_amp_textbox.textEdited.emit(self.dialog.micro_amp_textbox.text())
        self.assertEqual(self.template_mock.keys[Script.Key.mu_amps.value], self.dialog.micro_amp_textbox.text())

        self.assertEqual(self.template_mock.keys[Script.Key.count.value], 11)
        self.assertNotEqual(self.dialog.preview_label.toPlainText(), dummy)
        self.assertTrue(self.dialog.preview_label.toPlainText().endswith(dummy))

        self.presenter.exportScript = mock.Mock(return_value=False)
        self.dialog.export()
        self.presenter.exportScript.assert_called_once()
        self.assertEqual(self.dialog.result(), 0)

        self.presenter.exportScript = mock.Mock(return_value=True)
        self.dialog.export()
        self.presenter.exportScript.assert_called()
        self.assertEqual(self.dialog.result(), 1)


class TestPathLengthPlotter(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.presenter = MainWindowPresenter(self.view)

        self.simulation_mock = mock.create_autospec(Simulation)
        self.model_mock.return_value.simulation = self.simulation_mock
        self.simulation_mock.shape = (0, 0, 0)
        self.simulation_mock.compute_path_length = False
        self.simulation_mock.detector_names = ['East', 'West']

        self.view.presenter = self.presenter
        self.dialog = PathLengthPlotter(self.view)

    def testPlotting(self):
        self.assertFalse(hasattr(self.dialog, 'alignment_combobox'))
        shape = (2, 2, 4)
        self.simulation_mock.shape = shape
        self.simulation_mock.path_lengths = np.zeros(shape)
        self.simulation_mock.path_lengths[:, :, 0] = [[1, 2], [3, 4]]
        self.simulation_mock.path_lengths[:, :, 3] = [[5, 6], [7, 8]]
        self.simulation_mock.compute_path_length = True
        self.dialog = PathLengthPlotter(self.view)

        combo = self.dialog.alignment_combobox
        self.assertListEqual(['1', '2', '3', '4'], [combo.itemText(i) for i in range(combo.count())])
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2], line.get_xdata().tolist())
        self.assertListEqual([1, 3], line.get_ydata().tolist())
        line = self.dialog.axes.lines[1]
        self.assertListEqual([1, 2], line.get_xdata().tolist())
        self.assertListEqual([2, 4], line.get_ydata().tolist())

        combo.activated.emit(3)
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2], line.get_xdata().tolist())
        self.assertListEqual([5, 7], line.get_ydata().tolist())
        line = self.dialog.axes.lines[1]
        self.assertListEqual([1, 2], line.get_xdata().tolist())
        self.assertListEqual([6, 8], line.get_ydata().tolist())


class TestSampleExportDialog(unittest.TestCase):
    app = QApplication([])

    def testSampleSelection(self):
        dialog = SampleExportDialog([], None)
        self.assertEqual(dialog.selected, '')
        dialog = SampleExportDialog(['first', 'second'], None)
        self.assertEqual(dialog.selected, 'first')
        list_widget = dialog.list_widget
        list_widget.itemClicked.emit(list_widget.item(1))
        self.assertEqual(dialog.selected, 'second')
        # Check that it is not deselected by another click
        list_widget.itemClicked.emit(list_widget.item(1))
        self.assertEqual(dialog.selected, 'second')

        self.assertTrue(dialog.close())
        self.assertEqual(dialog.selected, 'second')


class TestStatusBar(unittest.TestCase):
    app = QApplication([])

    def testWidgetManagement(self):
        widget = StatusBar()
        compound_widget_1 = FormControl('Name', dummy)
        self.assertEqual(widget.left_layout.count(), 0)
        self.assertEqual(widget.right_layout.count(), 0)
        widget.addPermanentWidget(compound_widget_1, alignment=Qt.AlignRight)
        self.assertEqual(widget.left_layout.count(), 0)
        self.assertEqual(widget.right_layout.count(), 1)
        compound_widget_2 = FormControl('Age', dummy)
        widget.addPermanentWidget(compound_widget_2, alignment=Qt.AlignLeft)
        self.assertEqual(widget.left_layout.count(), 1)
        self.assertEqual(widget.right_layout.count(), 1)
        widget.removeWidget(compound_widget_1)
        self.assertEqual(widget.left_layout.count(), 1)
        self.assertEqual(widget.right_layout.count(), 0)
        widget.removeWidget(compound_widget_2)
        self.assertEqual(widget.left_layout.count(), 0)
        self.assertEqual(widget.right_layout.count(), 0)

    def testTempMessage(self):
        widget = StatusBar()
        widget.timer.singleShot = mock.Mock()
        message = 'Hello, World!'
        widget.showMessage(message)
        self.assertEqual(widget.currentMessage(), message)
        widget.clearMessage()
        self.assertEqual(widget.currentMessage(), '')

        widget.showMessage(message, 2)
        self.assertEqual(widget.currentMessage(), message)
        self.assertEqual(widget.timer.singleShot.call_args[0][0], 2)
        clear_function = widget.timer.singleShot.call_args[0][1]
        clear_function()
        self.assertEqual(widget.currentMessage(), '')


class TestFileDialog(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()

        self.mock_select_filter = self.createMock('sscanss.ui.widgets.helpers.QtWidgets.QFileDialog.selectedNameFilter')
        self.mock_select_file = self.createMock('sscanss.ui.widgets.helpers.QtWidgets.QFileDialog.selectedFiles')
        self.mock_isfile = self.createMock('sscanss.ui.widgets.helpers.os.path.isfile', True)
        self.mock_dialog_exec = self.createMock('sscanss.ui.widgets.helpers.QtWidgets.QFileDialog.exec')
        self.mock_message_box = self.createMock('sscanss.ui.widgets.helpers.QtWidgets.QMessageBox.warning')

    def createMock(self, module, autospec=False):
        patcher = mock.patch(module, autospec=autospec)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def testOpenFileDialog(self):
        d = FileDialog(self.view, '', '', 'All Files (*);;Python Files (*.py);;3D Files (*.stl *.obj)')
        self.assertListEqual(d.filters, ['', '.py', '.stl', '.obj'])

        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)', )
        self.assertEqual(filename, '')
        self.mock_dialog_exec.assert_called_once()
        self.assertEqual(len(self.mock_dialog_exec.call_args[0]), 0)
        self.mock_message_box.assert_not_called()

        self.mock_select_filter.return_value = '3D Files (*.stl *.obj)'
        self.mock_select_file.return_value = ['unknown_file']
        self.mock_dialog_exec.return_value = QFileDialog.Accepted
        self.mock_isfile.return_value = False
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)', )
        self.assertEqual(filename, '')
        self.mock_message_box.assert_called()
        self.assertEqual(len(self.mock_message_box.call_args[0]), 5)
        self.assertEqual(self.mock_dialog_exec.call_count, 2)

        self.mock_isfile.return_value = True
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)', )
        self.assertEqual(filename, 'unknown_file.stl')
        self.assertEqual(self.mock_message_box.call_count, 1)
        self.assertEqual(self.mock_dialog_exec.call_count, 3)
        self.assertEqual(len(self.mock_select_file.call_args[0]), 0)
        self.assertEqual(len(self.mock_select_filter.call_args[0]), 0)

    def testSaveFileDialog(self):
        filename = FileDialog.getSaveFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)', )
        self.assertEqual(filename, '')
        self.mock_dialog_exec.assert_called_once()
        self.assertEqual(len(self.mock_dialog_exec.call_args[0]), 0)
        self.mock_message_box.assert_not_called()

        self.mock_select_filter.return_value = '3D Files (*.stl *.obj)'
        self.mock_select_file.return_value = ['unknown_file']
        self.mock_dialog_exec.return_value = QFileDialog.Accepted
        self.mock_message_box.return_value = QMessageBox.No
        self.mock_isfile.return_value = True
        filename = FileDialog.getSaveFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)', )
        self.assertEqual(filename, '')
        self.mock_message_box.assert_called()
        self.assertEqual(len(self.mock_message_box.call_args[0]), 5)
        self.assertEqual(self.mock_dialog_exec.call_count, 2)

        self.mock_isfile.return_value = False
        filename = FileDialog.getSaveFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)', )
        self.assertEqual(filename, 'unknown_file.stl')
        self.assertEqual(self.mock_message_box.call_count, 1)
        self.assertEqual(self.mock_dialog_exec.call_count, 3)
        self.assertEqual(len(self.mock_select_file.call_args[0]), 0)
        self.assertEqual(len(self.mock_select_filter.call_args[0]), 0)


class TestSelectionWidgets(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.widgets.helpers.QtWidgets.QColorDialog', autospec=True)
    def testColourPicker(self, color_dialog):
        colour = QColor(Qt.black)
        widget = ColourPicker(colour)
        self.assertEqual(widget.value, colour)
        self.assertEqual(widget.colour_name.text(), colour.name())

        colour = QColor(Qt.red)
        color_dialog.getColor.return_value = colour
        widget.mousePressEvent(None)
        self.assertEqual(color_dialog.getColor.call_count, 1)
        self.assertEqual(len(color_dialog.getColor.call_args[0]), 1)
        self.assertEqual(widget.value, colour)
        self.assertEqual(widget.colour_name.text(), colour.name())

        invalid_colour = QColor()
        color_dialog.getColor.return_value = invalid_colour
        widget.mousePressEvent(None)
        self.assertEqual(color_dialog.getColor.call_count, 2)
        self.assertEqual(widget.value, colour)
        self.assertEqual(widget.colour_name.text(), colour.name())

    @mock.patch('sscanss.ui.widgets.helpers.FileDialog', autospec=True)
    def testFilePicker(self, file_dialog):
        path = 'some_file.txt'
        widget = FilePicker(path, False)
        self.assertEqual(widget.value, path)

        new_path = 'some_other_file.txt'
        file_dialog.getOpenFileName.return_value = new_path
        widget.openFileDialog()
        self.assertEqual(widget.value, new_path)
        self.assertEqual(file_dialog.getOpenFileName.call_count, 1)
        self.assertEqual(file_dialog.getExistingDirectory.call_count, 0)

        new_path = 'yet another_file.txt'
        file_dialog.getExistingDirectory.return_value = new_path
        widget.select_folder = True
        widget.openFileDialog()
        self.assertEqual(widget.value, new_path)
        self.assertEqual(file_dialog.getOpenFileName.call_count, 1)
        self.assertEqual(file_dialog.getExistingDirectory.call_count, 1)


class TestAccordion(unittest.TestCase):
    app = QApplication([])

    def setUp(self):
        self.pane_content_visible = False
        self.context_menu_visible = False
        self.context_menu_point = None

    def contentShow(self):
        self.pane_content_visible = True

    def contentHide(self):
        self.pane_content_visible = False

    def contextMenuPopup(self, _):
        self.context_menu_visible = True

    def testAddAndClear(self):
        pane_1 = Pane(QLabel(), QLabel(), Pane.Type.Warn)
        pane_2 = Pane(QLabel(), QLabel(), Pane.Type.Error)

        accordion = Accordion()
        accordion.addPane(pane_1)
        self.assertEqual(len(accordion.panes), 1)
        accordion.addPane(pane_2)
        self.assertEqual(len(accordion.panes), 2)
        self.assertRaises(TypeError, accordion.addPane, 'Pane')
        self.assertIs(accordion.panes[0], pane_1)
        self.assertIs(accordion.panes[1], pane_2)
        accordion.clear()
        self.assertEqual(len(accordion.panes), 0)

    def testPanes(self):
        pane = Pane(QLabel(), QLabel())
        pane.content.show = mock.Mock(side_effect=self.contentShow)
        pane.content.hide = mock.Mock(side_effect=self.contentHide)
        pane.context_menu.popup = mock.Mock(side_effect=self.contextMenuPopup)
        pane.toggle(False)
        self.assertTrue(self.pane_content_visible)
        pane.toggle(True)
        self.assertFalse(self.pane_content_visible)
        self.app.sendEvent(pane, QMouseEvent(QEvent.MouseButtonPress, QPoint(),
                                             Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertTrue(self.pane_content_visible)

        a = QAction('Some Action!')
        self.assertEqual(len(pane.context_menu.actions()), 0)
        pane.addContextMenuAction(a)
        self.assertEqual(len(pane.context_menu.actions()), 1)
        pane.customContextMenuRequested.emit(QPoint(100, 250))
        self.assertTrue(self.context_menu_visible)
        pane.paintEvent(None)
