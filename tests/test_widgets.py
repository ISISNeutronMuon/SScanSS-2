import unittest
import unittest.mock as mock
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QEvent
from PyQt5.QtGui import QColor, QMouseEvent, QBrush
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QLabel, QAction
from sscanss.core.util import PointType, POINT_DTYPE, CommandID, TransformType
from sscanss.core.geometry import Mesh
from sscanss.core.instrument.simulation import SimulationResult, Simulation
from sscanss.core.instrument.robotics import IKSolver, IKResult, SerialManipulator, Link
from sscanss.core.instrument.instrument import Script, PositioningStack
from sscanss.ui.dialogs import (SimulationDialog, ScriptExportDialog, PathLengthPlotter, SampleExportDialog,
                                SampleManager, PointManager, VectorManager, DetectorControl, JawControl,
                                PositionerControl, TransformDialog)
from sscanss.ui.widgets import (FormGroup, FormControl, CompareValidator, StatusBar, ColourPicker, FileDialog,
                                FilePicker, Accordion, Pane, PointModel, AlignmentErrorModel, ErrorDetailModel,
                                GLWidget)
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
        self.simulation_mock.stopped = TestSignal()
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
                                        SimulationResult('3', non_fatal, (['X'], [45]), 0),
                                        SimulationResult('4', skipped=True, note='something happened')]
        self.simulation_mock.count = len(self.simulation_mock.results)
        self.simulation_mock.scene_size = 2

        self.model_mock.return_value.simulation = self.simulation_mock
        self.assertFalse(self.dialog._hide_skipped_results)
        self.dialog.hide_skipped_button.toggle()
        self.assertTrue(self.dialog._hide_skipped_results)
        self.model_mock.return_value.simulation_created.emit()
        self.simulation_mock.result_updated.emit(False)
        self.assertEqual(len(self.dialog.result_list.panes), 4)
        actions = self.dialog.result_list.panes[0].context_menu.actions()
        actions[0].trigger()  # copy action
        self.assertEqual(self.app.clipboard().text(), '90.000')

        self.assertTrue(self.dialog.result_list.panes[0].isEnabled())
        self.assertTrue(self.dialog.result_list.panes[1].isEnabled())
        self.assertFalse(self.dialog.result_list.panes[2].isEnabled())
        self.assertFalse(self.dialog.result_list.panes[3].isEnabled())

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


class TestSampleManger(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.sample_changed = TestSignal()

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        mesh = Mesh(vertices, indices, normals)

        self.model_mock.return_value.sample = {'m': mesh, 't': mesh}
        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

        self.dialog = SampleManager(self.view)

    def testRemoveSample(self):
        self.presenter.deleteSample = mock.Mock()
        self.dialog.delete_button.click()
        self.presenter.deleteSample.assert_not_called()

        self.dialog.list_widget.setCurrentRow(1)
        self.dialog.delete_button.click()
        self.presenter.deleteSample.assert_called_with(['t'])

        self.dialog.list_widget.item(0).setSelected(True)
        self.dialog.list_widget.item(1).setSelected(True)
        self.dialog.delete_button.click()
        self.presenter.deleteSample.assert_called_with(['m', 't'])

    def testChangeMainSample(self):
        self.presenter.changeMainSample = mock.Mock()
        self.dialog.priority_button.click()
        self.presenter.changeMainSample.assert_not_called()

        self.dialog.list_widget.item(0).setSelected(True)
        self.dialog.list_widget.item(1).setSelected(True)
        self.assertFalse(self.dialog.priority_button.isEnabled())
        self.dialog.list_widget.item(1).setSelected(False)
        self.assertTrue(self.dialog.priority_button.isEnabled())
        self.dialog.priority_button.click()
        self.presenter.changeMainSample.assert_not_called()

        self.dialog.list_widget.item(0).setSelected(False)
        self.dialog.list_widget.item(1).setSelected(False)
        self.dialog.list_widget.setCurrentRow(1)
        self.dialog.priority_button.click()
        self.presenter.changeMainSample.assert_called_with('t')

    def testMergeSample(self):
        self.presenter.mergeSample = mock.Mock()
        self.dialog.merge_button.click()
        self.presenter.mergeSample.assert_not_called()

        self.dialog.list_widget.item(0).setSelected(True)
        self.dialog.merge_button.click()
        self.presenter.mergeSample.assert_not_called()

        self.dialog.list_widget.item(1).setSelected(True)
        self.dialog.merge_button.click()
        self.presenter.mergeSample.assert_called_with(['m', 't'])

    def testMisc(self):
        self.assertEqual(self.dialog.list_widget.count(), 2)
        self.model_mock.return_value.sample = {'m': None}
        self.model_mock.return_value.sample_changed.emit()
        self.assertEqual(self.dialog.list_widget.count(), 1)
        self.model_mock.return_value.sample = {}
        self.model_mock.return_value.sample_changed.emit()
        self.assertEqual(self.dialog.list_widget.count(), 0)

        self.dialog.close()
        self.view.scenes.changeSelected.assert_called_once()


class TestPointManger(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.fiducials_changed = TestSignal()
        self.model_mock.return_value.measurement_points_changed = TestSignal()
        points = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], True)],
                              dtype=POINT_DTYPE)

        self.model_mock.return_value.fiducials = points
        self.model_mock.return_value.measurement_points = points

        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

        self.dialog1 = PointManager(PointType.Fiducial, self.view)
        self.dialog2 = PointManager(PointType.Measurement, self.view)

    def testDeletePoints(self):
        self.presenter.deletePoints = mock.Mock()
        self.dialog1.delete_button.click()
        self.presenter.deletePoints.assert_not_called()

        self.dialog1.table_view.selectRow(0)
        self.dialog1.delete_button.click()
        self.presenter.deletePoints.assert_called_with([0], PointType.Fiducial)

        self.presenter.deletePoints.reset_mock()
        self.dialog2.delete_button.click()
        self.presenter.deletePoints.assert_not_called()

        table = self.dialog2.table_view
        table.setSelectionMode(table.MultiSelection)
        table.selectRow(1)
        table.selectRow(2)
        self.dialog2.delete_button.click()
        self.presenter.deletePoints.assert_called_with([1, 2], PointType.Measurement)

    def testMovePoints(self):
        self.presenter.movePoints = mock.Mock()
        self.dialog1.move_up_button.click()
        self.presenter.movePoints.assert_not_called()

        self.dialog1.table_view.selectRow(0)
        self.dialog1.move_up_button.click()
        self.presenter.movePoints.assert_not_called()

        self.dialog1.table_view.selectRow(2)
        self.dialog1.move_up_button.click()
        self.presenter.movePoints.assert_called_with(2, 1, PointType.Fiducial)

        self.presenter.movePoints.reset_mock()
        self.dialog2.move_down_button.click()
        self.presenter.movePoints.assert_not_called()

        self.dialog2.table_view.selectRow(2)
        self.dialog2.move_down_button.click()
        self.presenter.movePoints.assert_not_called()

        self.dialog2.table_view.selectRow(0)
        self.dialog2.move_down_button.click()
        self.presenter.movePoints.assert_called_with(0, 1, PointType.Measurement)

        table = self.dialog2.table_view
        table.setSelectionMode(table.MultiSelection)
        table.selectRow(1)
        table.selectRow(2)
        self.assertFalse(self.dialog2.move_up_button.isEnabled())
        self.assertFalse(self.dialog2.move_down_button.isEnabled())
        table.selectionModel().clearSelection()
        self.assertTrue(self.dialog2.move_up_button.isEnabled())
        self.assertTrue(self.dialog2.move_down_button.isEnabled())

    def testEditPoints(self):
        self.presenter.editPoints = mock.Mock()
        points = np.rec.array([([1.0, 2.0, 3.0], True), ([4.0, 5.0, 6.0], False), ([7.0, 8.0, 9.0], False)],
                              dtype=POINT_DTYPE)

        self.dialog1.table_model.editCompleted.emit(points)
        self.presenter.editPoints.assert_called_with(points, PointType.Fiducial)

        self.dialog2.table_model.editCompleted.emit(points)
        self.presenter.editPoints.assert_called_with(points, PointType.Measurement)

        self.dialog2.selected = self.dialog2.table_model.index(1, 0)
        self.model_mock.return_value.measurement_points = points
        self.model_mock.return_value.measurement_points_changed.emit()
        np.testing.assert_array_almost_equal(points.points, self.dialog2.table_model._data.points, decimal=5)
        np.testing.assert_array_almost_equal(points.enabled, self.dialog2.table_model._data.enabled, decimal=5)
        self.assertEqual(self.dialog2.table_view.currentIndex().row(), 1)

        self.view.scenes.reset_mock()
        self.dialog1.close()
        self.view.scenes.changeSelected.assert_called_once()


class TestVectorManger(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]

        detectors = []
        for i in range(2):
            collimator = mock.Mock()
            collimator.name = i * 2
            detector = mock.Mock()
            detector.current_collimator = collimator
            detectors.append(detector)
        detectors = {'East': detectors[0], 'West': detectors[1]}
        self.model_mock.return_value.instrument.detectors = detectors
        self.model_mock.return_value.measurement_vectors_changed = TestSignal()
        vectors = np.ones((4, 6, 2))

        self.model_mock.return_value.measurement_vectors = vectors

        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.scenes.rendered_alignment = 0
        self.view.scenes.rendered_alignment_changed = TestSignal()
        self.view.presenter = self.presenter

        self.dialog = VectorManager(self.view)

    def testDeleteAlignment(self):
        self.presenter.removeVectorAlignment = mock.Mock()
        self.dialog.delete_alignment_action.trigger()
        self.presenter.removeVectorAlignment.assert_called_with(0)

        self.dialog.alignment_combobox.setCurrentIndex(1)
        self.dialog.delete_alignment_action.trigger()
        self.presenter.removeVectorAlignment.assert_called_with(1)

    def testDeleteVectors(self):
        self.presenter.removeVectors = mock.Mock()
        self.dialog.delete_vector_action.trigger()
        self.presenter.removeVectors.assert_called_with([0, 1, 2, 3], 0, 0)

        self.dialog.table.selectRow(2)
        self.dialog.alignment_combobox.setCurrentIndex(1)
        self.dialog.detector_combobox.setCurrentIndex(1)
        self.dialog.delete_vector_action.trigger()
        self.presenter.removeVectors.assert_called_with([2], 1, 1)

    def testMisc(self):
        self.dialog.detector_combobox.setCurrentIndex(1)
        self.assertEqual(self.dialog.alignment_combobox.currentIndex(), 0)
        self.view.scenes.rendered_alignment = 1
        self.view.scenes.rendered_alignment_changed.emit()
        self.assertEqual(self.dialog.alignment_combobox.currentIndex(), 1)

        self.dialog.onComboBoxActivated()
        self.view.scenes.changeRenderedAlignment.assert_called_with(1)

        self.dialog.close()
        self.view.scenes.changeRenderedAlignment.assert_called_with(0)


class TestJawControl(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument_controlled = TestSignal()

        jaws = mock.Mock()
        jaws.aperture = [2, 2]
        jaws.aperture_upper_limit = [10, 10]
        jaws.aperture_lower_limit = [1, 1]
        q1 = Link('X', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200., 200., 0)
        q2 = Link('Z', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi / 3., np.pi / 3., 0)
        jaws.positioner = SerialManipulator(f'jaw', [q1, q2])
        self.model_mock.return_value.instrument.jaws = jaws

        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

        self.dialog = JawControl(self.view)

    def testMoveJaws(self):
        self.presenter.movePositioner = mock.Mock()

        control = self.dialog.position_form_group.form_controls[1]
        control.value = -70  # value is in degrees
        self.assertFalse(self.dialog.move_jaws_button.isEnabled())
        control.value = 0
        self.assertTrue(self.dialog.move_jaws_button.isEnabled())

        control = self.dialog.position_form_group.form_controls[0]
        control.value = 201
        self.assertFalse(self.dialog.move_jaws_button.isEnabled())
        control.value = 0
        self.assertTrue(self.dialog.move_jaws_button.isEnabled())
        self.dialog.move_jaws_button.click()
        self.presenter.movePositioner.assert_not_called()

        control.value = 10
        self.assertTrue(self.dialog.move_jaws_button.isEnabled())
        self.dialog.move_jaws_button.click()
        self.presenter.movePositioner.assert_called_with('jaw', [10, 0])

    def testChangeAperture(self):
        self.presenter.changeJawAperture = mock.Mock()

        control = self.dialog.aperture_form_group.form_controls[1]
        control.value = 0  # value is in degrees
        self.assertFalse(self.dialog.change_aperture_button.isEnabled())
        control.value = 2
        self.assertTrue(self.dialog.change_aperture_button.isEnabled())

        control = self.dialog.aperture_form_group.form_controls[0]
        control.value = 12
        self.assertFalse(self.dialog.change_aperture_button.isEnabled())
        control.value = 2
        self.assertTrue(self.dialog.change_aperture_button.isEnabled())
        self.dialog.change_aperture_button.click()
        self.presenter.changeJawAperture.assert_not_called()

        control.value = 5
        self.dialog.change_aperture_button.click()
        self.presenter.changeJawAperture.assert_called_with([5, 2])

    def testIgnoreLimits(self):
        self.presenter.ignorePositionerJointLimits = mock.Mock()

        control = self.dialog.position_form_group.form_controls[0]
        control.extra[0].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with('jaw', 0, True)

        control.extra[0].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with('jaw', 0, False)

    def testUpdate(self):
        control = self.dialog.position_form_group.form_controls[0]
        control.value = 10
        self.assertTrue(control.valid)

        self.model_mock.return_value.instrument_controlled.emit(CommandID.MovePositioner)
        self.assertEqual(control.value, 0)

        self.dialog.instrument.jaws.positioner.links[0].ignore_limits = True
        self.model_mock.return_value.instrument_controlled.emit(CommandID.IgnoreJointLimits)
        self.assertIsNone(control.range_validator.minimum)
        self.assertIsNone(control.range_validator.maximum)

        self.dialog.instrument.jaws.positioner.links[0].ignore_limits = False
        link = self.dialog.instrument.jaws.positioner.links[0]
        self.model_mock.return_value.instrument_controlled.emit(CommandID.IgnoreJointLimits)
        self.assertEqual(control.range_validator.minimum, link.lower_limit)
        self.assertEqual(control.range_validator.maximum, link.upper_limit)

        control = self.dialog.aperture_form_group.form_controls[0]
        control.value = 10
        self.assertTrue(control.valid)

        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangeJawAperture)
        self.assertEqual(control.value, 2)

        self.dialog.close()
        self.view.scenes.changeVisibility.assert_called()


class TestTransformDialog(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.all_sample_key = 'all the samples'
        self.model_mock.return_value.sample_changed = TestSignal()

        vertices = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        self.model_mock.return_value.sample = {'m': Mesh(vertices, indices, normals)}
        self.model_mock.return_value.fiducials = np.array([])
        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

    def testRotateTool(self):
        dialog = TransformDialog(TransformType.Rotate, self.view)
        self.presenter.transformSample = mock.Mock()
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        dialog.tool.x_rotation.value = 4.0
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, 0.0, 4.0], None, TransformType.Rotate)

        self.model_mock.return_value.sample = {'m': None, 't': None}
        self.model_mock.return_value.sample_changed.emit()
        dialog.combobox.setCurrentIndex(2)
        dialog.combobox.activated[str].emit('t')
        dialog.tool.y_rotation.value = 4.0
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, 4.0, 4.0], 't', TransformType.Rotate)

        dialog.tool.y_rotation.value = 361.0
        self.assertFalse(dialog.tool.execute_button.isEnabled())
        dialog.tool.y_rotation.value = 360.0
        self.assertTrue(dialog.tool.execute_button.isEnabled())
        self.model_mock.return_value.sample = {}
        self.model_mock.return_value.sample_changed.emit()
        self.assertFalse(dialog.tool.execute_button.isEnabled())

    def testTranslateTool(self):
        dialog = TransformDialog(TransformType.Translate, self.view)
        self.presenter.transformSample = mock.Mock()
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        dialog.tool.x_position.value = 4.0
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([4.0, 0.0, 0.0], None, TransformType.Translate)

        self.model_mock.return_value.sample = {'m': None, 't': None}
        self.model_mock.return_value.sample_changed.emit()
        dialog.combobox.setCurrentIndex(2)
        dialog.combobox.activated[str].emit('t')
        dialog.tool.y_position.value = 4.0
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([4.0, 4.0, 0.0], 't', TransformType.Translate)

        dialog.tool.y_position.text = 'a'
        self.assertFalse(dialog.tool.execute_button.isEnabled())
        dialog.tool.y_position.value = 4.0
        self.assertTrue(dialog.tool.execute_button.isEnabled())
        self.model_mock.return_value.sample = {}
        self.model_mock.return_value.sample_changed.emit()
        self.assertFalse(dialog.tool.execute_button.isEnabled())

    def testCustomTransformTool(self):
        dialog = TransformDialog(TransformType.Custom, self.view)
        self.presenter.transformSample = mock.Mock()
        self.presenter.importTransformMatrix = mock.Mock()
        self.presenter.importTransformMatrix.return_value = None

        dialog.tool.load_matrix.click()
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        self.presenter.importTransformMatrix.return_value = np.eye(4)
        dialog.tool.load_matrix.click()
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        random_matrix = np.random.rand(4, 4)
        self.presenter.importTransformMatrix.return_value = random_matrix
        dialog.tool.load_matrix.click()
        self.model_mock.return_value.sample = {'m': None, 't': None}
        self.model_mock.return_value.sample_changed.emit()
        dialog.combobox.setCurrentIndex(2)
        dialog.combobox.activated[str].emit('t')
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with(random_matrix, 't', TransformType.Custom)

        self.model_mock.return_value.sample = {}
        self.model_mock.return_value.sample_changed.emit()
        self.assertFalse(dialog.tool.execute_button.isEnabled())

    def testMoveOriginTool(self):
        dialog = TransformDialog(TransformType.Origin, self.view)
        self.presenter.transformSample = mock.Mock()
        dialog.tool.move_combobox.setCurrentIndex(1)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        dialog.tool.move_combobox.setCurrentIndex(2)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([-1.0, -1.0, -1.0], None, TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(1)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, -1.0, -1.0], None, TransformType.Translate)

        vertices = np.array([[1, 1, 1], [2, 0, 2], [2, 2, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        self.model_mock.return_value.sample['t'] = Mesh(vertices, indices, normals)
        self.model_mock.return_value.sample_changed.emit()

        dialog.tool.ignore_combobox.setCurrentIndex(2)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([-2.0, 0.0, -2.0], None, TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(3)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([-2.0, -2.0, 0.0], None, TransformType.Translate)

        dialog.combobox.setCurrentIndex(2)
        dialog.combobox.activated[str].emit('t')
        dialog.tool.move_combobox.setCurrentIndex(0)
        dialog.tool.ignore_combobox.setCurrentIndex(4)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([-1.5, 0.0, 0.0], 't', TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(5)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, 0.0, -1.0], 't', TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(6)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, -1.0, 0.0], 't', TransformType.Translate)

        self.model_mock.return_value.sample = {}
        self.model_mock.return_value.sample_changed.emit()
        dialog.tool.move_combobox.setCurrentIndex(1)
        self.assertFalse(dialog.tool.execute_button.isEnabled())

    @mock.patch('sscanss.ui.dialogs.tools.rotation_btw_vectors', autospec=False)
    @mock.patch('sscanss.ui.dialogs.tools.point_selection', autospec=True)
    def testPlaneAlignmentTool(self, select_mock, rot_vec_mock):
        self.view.gl_widget = mock.create_autospec(GLWidget)
        self.view.gl_widget.pick_added = TestSignal()
        self.view.gl_widget.picks = []
        self.view.gl_widget.picking = False

        dialog = TransformDialog(TransformType.Plane, self.view)
        self.presenter.transformSample = mock.Mock()
        dialog.tool.final_plane_normal = 1
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        dialog.tool.plane_combobox.setCurrentIndex(3)
        dialog.tool.final_plane_normal = 1
        dialog.tool.x_axis.value = 0
        self.assertIsNone(dialog.tool.final_plane_normal)

        dialog.tool.final_plane_normal = 1
        dialog.tool.x_axis.value = -2
        self.assertIsNone(dialog.tool.final_plane_normal)

        dialog.tool.final_plane_normal = 1
        dialog.tool.x_axis.value = 1
        self.assertIsNotNone(dialog.tool.final_plane_normal)

        dialog.tool.pick_button.click()
        self.assertTrue(self.view.gl_widget.picking)
        dialog.tool.select_button.click()
        self.assertFalse(self.view.gl_widget.picking)
        dialog.tool.pick_button.click()
        self.assertTrue(self.view.gl_widget.picking)

        self.assertEqual(dialog.tool.table_widget.rowCount(), 0)
        select_mock.return_value = np.array([])
        self.view.gl_widget.pick_added.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 0)

        select_mock.return_value = np.array([[0., 0., 0.]])
        self.view.gl_widget.pick_added.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 1)

        select_mock.return_value = np.array([[1., 0., 0.]])
        self.view.gl_widget.pick_added.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 2)

        vertices = np.array([[1, 1, 1], [2, 0, 2], [2, 2, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        self.model_mock.return_value.sample['t'] = Mesh(vertices, indices, normals)
        self.model_mock.return_value.sample_changed.emit()

        self.assertIsNone(dialog.tool.initial_plane)
        select_mock.return_value = np.array([[0., 1., 0.]])
        self.view.gl_widget.pick_added.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 3)
        select_mock.assert_called()
        np.testing.assert_array_almost_equal(select_mock.call_args[0][2], [[0, 0, 0, 1, 0, 1, 1, 1, 0],
                                                                           [1, 1, 1, 2, 0, 2, 2, 2, 0]], decimal=5)
        self.assertIsNotNone(dialog.tool.initial_plane)

        select_mock.return_value = np.array([[1., 1., 0.]])
        self.view.gl_widget.pick_added.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 4)

        self.assertEqual(self.view.gl_widget.picks, [[[0., 0., 0.], False], [[1., 0., 0.], False],
                                                     [[0., 1., 0.], False], [[1., 1., 0.], False]])

        dialog.tool.table_widget.selectRow(3)
        self.assertEqual(self.view.gl_widget.picks, [[[0., 0., 0.], False], [[1., 0., 0.], False],
                                                     [[0., 1., 0.], False], [[1., 1., 0.], True]])

        dialog.tool.removePicks()
        self.assertEqual(self.view.gl_widget.picks, [[[0., 0., 0.], False], [[1., 0., 0.], False],
                                                     [[0., 1., 0.], False]])

        matrix = np.eye(4)
        rot_vec_mock.return_value = matrix[:3, :3]
        dialog.tool.plane_combobox.setCurrentIndex(1)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        matrix[:3, :3] = np.random.rand(3)
        rot_vec_mock.return_value = matrix[:3, :3]
        dialog.tool.plane_combobox.setCurrentIndex(2)
        plane = dialog.tool.initial_plane
        dialog.tool.executeButtonClicked()
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called()
        np.testing.assert_array_almost_equal(self.presenter.transformSample.call_args[0][0], matrix, decimal=5)
        self.assertIsNone(self.presenter.transformSample.call_args[0][1])
        self.assertEqual(self.presenter.transformSample.call_args[0][2], TransformType.Custom)
        self.assertEqual(self.view.gl_widget.picks, [])

        matrix[:3, :3] = np.random.rand(3)
        rot_vec_mock.return_value = matrix[:3, :3]
        dialog.combobox.setCurrentIndex(2)
        dialog.combobox.activated[str].emit('t')
        dialog.tool.initial_plane = plane
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called()
        np.testing.assert_array_almost_equal(self.presenter.transformSample.call_args[0][0], matrix, decimal=5)
        self.assertEqual(self.presenter.transformSample.call_args[0][1], 't')
        self.assertEqual(self.presenter.transformSample.call_args[0][2], TransformType.Custom)

        self.model_mock.return_value.sample = {}
        self.model_mock.return_value.sample_changed.emit()
        self.assertFalse(dialog.tool.execute_button.isEnabled())

        dialog.close()
        self.assertFalse(self.view.gl_widget.picking)


class TestPositionerControl(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument_controlled = TestSignal()

        q1 = Link('X', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200., 200., 0)
        q2 = Link('Z', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi/2, np.pi/2, 0)
        main_positioner = SerialManipulator('p1', [q1, q2])

        q3 = Link('Z', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200., 200., 0)
        aux_positioner = SerialManipulator('p2', [q3])
        stack = PositioningStack('1', main_positioner)
        stack.addPositioner(aux_positioner)
        self.model_mock.return_value.instrument.positioners = {'p1': main_positioner, 'p2': aux_positioner}
        self.model_mock.return_value.instrument.positioning_stack = stack
        self.model_mock.return_value.instrument.positioning_stacks = {'1': None, '2': None}

        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

        self.dialog = PositionerControl(self.view)

    def testMovePositioner(self):
        self.presenter.movePositioner = mock.Mock()

        control = self.dialog.positioner_form_controls[1]
        control.value = -91  # value is in degrees
        self.assertFalse(self.dialog.move_joints_button.isEnabled())
        control.value = 0
        self.assertTrue(self.dialog.move_joints_button.isEnabled())

        control = self.dialog.positioner_form_controls[0]
        control.value = 201
        self.assertFalse(self.dialog.move_joints_button.isEnabled())
        control.value = 0
        self.assertTrue(self.dialog.move_joints_button.isEnabled())
        self.dialog.move_joints_button.click()
        self.presenter.movePositioner.assert_not_called()

        control.value = 10
        self.assertTrue(self.dialog.move_joints_button.isEnabled())
        self.dialog.move_joints_button.click()
        self.presenter.movePositioner.assert_called_with('1', [10, 0, 0])

    def testChangeStack(self):
        self.presenter.changePositioningStack = mock.Mock()

        self.dialog.stack_combobox.activated[str].emit('1')
        self.presenter.changePositioningStack.assert_not_called()

        self.dialog.stack_combobox.activated[str].emit('2')
        self.presenter.changePositioningStack.assert_called_with('2')

    def testLockJoint(self):
        self.presenter.lockPositionerJoint = mock.Mock()

        control = self.dialog.positioner_form_controls[0]
        control.extra[0].click()
        self.presenter.lockPositionerJoint.assert_called_with('1', 0, True)

        control.extra[0].click()
        self.presenter.lockPositionerJoint.assert_called_with('1', 0, False)

    def testIgnoreLimits(self):
        self.presenter.ignorePositionerJointLimits = mock.Mock()

        control = self.dialog.positioner_form_controls[0]
        control.extra[1].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with('1', 0, True)

        control.extra[1].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with('1', 0, False)

    def testChangePositionerBase(self):
        self.presenter.changePositionerBase = mock.Mock()
        self.presenter.importTransformMatrix = mock.Mock()
        self.presenter.importTransformMatrix.return_value = None

        widget = self.dialog.positioner_forms_layout.itemAt(1).widget()
        title = widget.layout().itemAt(0).widget()
        base_button = title.title_layout.itemAt(2).widget()
        reset_button = self.dialog.base_reset_buttons['p2']

        base_button.click()
        self.presenter.changePositionerBase.assert_not_called()

        matrix = np.random.rand(4, 4)
        positioner = self.dialog.instrument.positioning_stack.auxiliary[0]
        self.presenter.importTransformMatrix.return_value = matrix
        base_button.click()
        self.presenter.changePositionerBase.assert_called_with(positioner, matrix)
        reset_button.click()
        self.presenter.changePositionerBase.assert_called_with(positioner, positioner.default_base)

    def testUpdate(self):
        self.model_mock.return_value.instrument.positioning_stack.name = '2'
        self.assertEqual(self.dialog.stack_combobox.currentIndex(), 0)
        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangePositioningStack)
        self.assertEqual(self.dialog.stack_combobox.currentIndex(), 1)

        control = self.dialog.positioner_form_controls[0]
        control.value = 10
        self.assertTrue(control.valid)

        self.model_mock.return_value.instrument_controlled.emit(CommandID.MovePositioner)
        self.assertEqual(control.value, 0)

        self.dialog.instrument.positioning_stack.fixed.links[0].ignore_limits = True
        self.model_mock.return_value.instrument_controlled.emit(CommandID.IgnoreJointLimits)
        self.assertIsNone(control.range_validator.minimum)
        self.assertIsNone(control.range_validator.maximum)

        self.dialog.instrument.positioning_stack.fixed.links[0].ignore_limits = False
        link = self.dialog.instrument.positioning_stack.fixed.links[0]
        self.model_mock.return_value.instrument_controlled.emit(CommandID.IgnoreJointLimits)
        self.assertEqual(control.range_validator.minimum, link.lower_limit)
        self.assertEqual(control.range_validator.maximum, link.upper_limit)

        self.dialog.instrument.positioning_stack.fixed.links[0].locked = True
        self.model_mock.return_value.instrument_controlled.emit(CommandID.LockJoint)
        self.assertFalse(control.form_lineedit.isEnabled())

        visible_mock = mock.Mock()
        button = self.dialog.base_reset_buttons['p2']
        button.setVisible = visible_mock
        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangePositionerBase)
        button.setVisible.assert_called_with(False)
        self.dialog.instrument.positioning_stack.auxiliary[0].base = np.random.rand(4, 4)
        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangePositionerBase)
        button.setVisible.assert_called_with(True)


class TestDetectorControl(unittest.TestCase):
    app = QApplication([])

    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument_controlled = TestSignal()

        detectors = []
        for i in range(2):
            collimator = mock.Mock()
            collimator.name = i * 2
            detector = mock.Mock()
            detector.current_collimator = collimator
            q1 = Link('X', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200., 200., 0)
            q2 = Link('Z', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi/3., np.pi/3., 0)
            detector.positioner = SerialManipulator(f'd{i+1}', [q1, q2])
            detectors.append(detector)
        detectors = {'East': detectors[0], 'West': detectors[1]}
        self.model_mock.return_value.instrument.detectors = detectors

        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

        self.dialog = DetectorControl('East', self.view)

    def testMoveDetector(self):
        self.presenter.movePositioner = mock.Mock()

        control = self.dialog.position_form_group.form_controls[1]
        control.value = -70  # value is in degrees
        self.assertFalse(self.dialog.move_detector_button.isEnabled())
        control.value = 0
        self.assertTrue(self.dialog.move_detector_button.isEnabled())

        control = self.dialog.position_form_group.form_controls[0]
        control.value = 201
        self.assertFalse(self.dialog.move_detector_button.isEnabled())
        control.value = 0
        self.assertTrue(self.dialog.move_detector_button.isEnabled())
        self.dialog.move_detector_button.click()
        self.presenter.movePositioner.assert_not_called()

        control.value = 10
        self.assertTrue(self.dialog.move_detector_button.isEnabled())
        self.dialog.move_detector_button.click()
        self.presenter.movePositioner.assert_called_with('d1', [10, 0])

    def testIgnoreLimits(self):
        self.presenter.ignorePositionerJointLimits = mock.Mock()

        control = self.dialog.position_form_group.form_controls[0]
        control.extra[0].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with('d1', 0, True)

        control.extra[0].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with('d1', 0, False)

    def testUpdate(self):
        control = self.dialog.position_form_group.form_controls[0]
        control.value = 10
        self.assertTrue(control.valid)

        self.model_mock.return_value.instrument_controlled.emit(CommandID.MovePositioner)
        self.assertEqual(control.value, 0)

        self.dialog.detector.positioner.links[0].ignore_limits = True
        self.model_mock.return_value.instrument_controlled.emit(CommandID.IgnoreJointLimits)
        self.assertIsNone(control.range_validator.minimum)
        self.assertIsNone(control.range_validator.maximum)

        self.dialog.detector.positioner.links[0].ignore_limits = False
        link = self.dialog.detector.positioner.links[0]
        self.model_mock.return_value.instrument_controlled.emit(CommandID.IgnoreJointLimits)
        self.assertEqual(control.range_validator.minimum, link.lower_limit)
        self.assertEqual(control.range_validator.maximum, link.upper_limit)

        self.dialog.close()
        self.view.scenes.changeVisibility.assert_called()


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
        not_converged = IKResult([87.8], IKSolver.Status.NotConverged, (0., 0., 0.), (1., 1., 0.), True, False)
        non_fatal = IKResult([45], IKSolver.Status.Failed, (-1., -1., -1.), (-1., -1., -1.), False, False)
        self.model_mock.return_value.instrument.script = self.template_mock
        self.simulation_mock.results = [SimulationResult('1', converged, (['X'], [90]), 0, (120,), [False, True]),
                                        SimulationResult('3', non_fatal, (['X'], [45]), 0, None, None),
                                        SimulationResult('2', not_converged, (['X'], [87.8]), 0, (25,), [True, True]),
                                        ]

        self.presenter = MainWindowPresenter(self.view)
        self.view.presenter = self.presenter
        self.dialog = ScriptExportDialog(self.simulation_mock, self.view)

    def testRendering(self):
        self.assertEqual(self.dialog.preview_label.toPlainText(), dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.filename.value], dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.header.value], f'{Script.Key.mu_amps.value}\tX')
        self.assertEqual(self.template_mock.keys[Script.Key.count.value], 2)
        self.assertEqual(self.template_mock.keys[Script.Key.position.value], '')
        self.assertEqual(self.template_mock.keys[Script.Key.script.value],
                         [{'position': '90.000'}, {'position': '87.800'}])

        self.assertFalse(self.dialog.show_mu_amps)
        self.assertFalse(hasattr(self.dialog, 'micro_amp_textbox'))

        for _ in range(9):
            self.simulation_mock.results.append(self.simulation_mock.results[0])

        self.template_mock.keys[Script.Key.mu_amps.value] = ''
        self.dialog = ScriptExportDialog(self.simulation_mock, self.view)
        self.assertEqual(self.template_mock.keys[Script.Key.mu_amps.value], '0.000')
        self.dialog.micro_amp_textbox.setText('4.500')
        self.dialog.micro_amp_textbox.textEdited.emit(self.dialog.micro_amp_textbox.text())
        self.assertEqual(self.template_mock.keys[Script.Key.mu_amps.value], self.dialog.micro_amp_textbox.text())

        self.assertEqual(self.template_mock.keys[Script.Key.count.value], 10)
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
        self.view.showSaveDialog = mock.Mock()
        self.model_mock = model_mock
        self.model_mock.return_value.save_path = ''
        self.model_mock.return_value.instruments = [dummy]
        self.presenter = MainWindowPresenter(self.view)

        self.simulation_mock = mock.create_autospec(Simulation)
        self.model_mock.return_value.simulation = self.simulation_mock
        shape = (1, 1, 1)
        self.simulation_mock.shape = shape
        self.simulation_mock.args = {'align_first_order': True}
        self.simulation_mock.path_lengths = np.zeros(shape)
        self.simulation_mock.detector_names = ['East']

        self.view.presenter = self.presenter
        self.dialog = PathLengthPlotter(self.view)

    def testPlotting(self):
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1], line.get_xdata().astype(int).tolist())
        self.assertListEqual([0], line.get_ydata().astype(int).tolist())

        shape = (2, 2, 4)
        self.simulation_mock.shape = shape
        self.simulation_mock.path_lengths = np.zeros(shape)
        self.simulation_mock.path_lengths[:, :, 0] = [[1, 2], [3, 4]]
        self.simulation_mock.path_lengths[:, :, 3] = [[5, 6], [7, 8]]
        self.simulation_mock.detector_names = ['East', 'West']
        self.dialog = PathLengthPlotter(self.view)

        combo = self.dialog.detector_combobox
        self.assertListEqual(self.simulation_mock.detector_names, [combo.itemText(i) for i in range(1, combo.count())])
        combo = self.dialog.alignment_combobox
        self.assertListEqual(['1', '2', '3', '4'], [combo.itemText(i) for i in range(1, combo.count())])
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 8], line.get_xdata().astype(int).tolist())
        self.assertListEqual([1, 0, 0, 5, 3, 0, 0, 7], line.get_ydata().astype(int).tolist())
        line = self.dialog.axes.lines[1]
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 8], line.get_xdata().astype(int).tolist())
        self.assertListEqual([2, 0, 0, 6, 4, 0, 0, 8], line.get_ydata().astype(int).tolist())

        self.simulation_mock.args = {'align_first_order': False}
        self.dialog.plot()
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 8], line.get_xdata().astype(int).tolist())
        self.assertListEqual([1, 3, 0, 0, 0, 0, 5, 7], line.get_ydata().astype(int).tolist())
        line = self.dialog.axes.lines[1]
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 8], line.get_xdata().astype(int).tolist())
        self.assertListEqual([2, 4, 0, 0, 0, 0, 6, 8], line.get_ydata().astype(int).tolist())

        combo.setCurrentIndex(4)
        combo.activated.emit(4)
        self.assertEqual(len(self.dialog.axes.lines), 2)
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2], line.get_xdata().astype(int).tolist())
        self.assertListEqual([5, 7], line.get_ydata().astype(int).tolist())
        line = self.dialog.axes.lines[1]
        self.assertListEqual([1, 2], line.get_xdata().astype(int).tolist())
        self.assertListEqual([6, 8], line.get_ydata().astype(int).tolist())

        combo = self.dialog.detector_combobox
        combo.setCurrentIndex(2)
        combo.activated.emit(2)
        self.assertEqual(len(self.dialog.axes.lines), 1)
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2], line.get_xdata().astype(int).tolist())
        self.assertListEqual([6, 8], line.get_ydata().astype(int).tolist())

        combo = self.dialog.alignment_combobox
        combo.setCurrentIndex(3)
        combo.activated.emit(3)
        self.assertEqual(len(self.dialog.axes.lines), 1)
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2], line.get_xdata().astype(int).tolist())
        self.assertListEqual([0, 0], line.get_ydata().astype(int).tolist())

    @mock.patch('sscanss.ui.dialogs.misc.np.savetxt', autospec=True)
    def testExport(self, savetxt):
        self.presenter.notifyError = mock.Mock()
        self.dialog.figure.savefig = mock.Mock()
        self.view.showSaveDialog.return_value = ''
        self.dialog.export()
        savetxt.assert_not_called()
        self.dialog.figure.savefig.assert_not_called()

        self.view.showSaveDialog.return_value = 'demo.txt'
        self.dialog.export()
        savetxt.assert_called_once()
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], [[1., 0.]], decimal=5)

        self.view.showSaveDialog.return_value = 'demo.png'
        self.dialog.export()
        self.dialog.figure.savefig.assert_called_once()

        shape = (3, 2, 2)
        self.simulation_mock.shape = shape
        self.simulation_mock.path_lengths = np.zeros(shape)
        self.simulation_mock.path_lengths[:, :, 0] = [[1, 2], [3, 4], [5, 6]]
        self.simulation_mock.detector_names = ['East', 'West']
        self.dialog = PathLengthPlotter(self.view)

        self.view.showSaveDialog.return_value = 'demo.txt'
        self.dialog.export()
        expected = [[1, 1, 2], [2, 0, 0], [3, 3, 4], [4, 0, 0], [5, 5, 6], [6, 0, 0]]
        self.assertEqual(savetxt.call_count, 2)
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], expected, decimal=5)

        combo = self.dialog.alignment_combobox
        combo.setCurrentIndex(2)
        combo.activated.emit(2)
        self.dialog.export()
        self.assertEqual(savetxt.call_count, 3)
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], [[1, 0, 0], [2, 0, 0], [3, 0, 0]], decimal=5)

        combo.setCurrentIndex(1)
        combo = self.dialog.detector_combobox
        combo.setCurrentIndex(2)
        combo.activated.emit(2)
        self.dialog.export()
        self.assertEqual(savetxt.call_count, 4)
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], [[1, 2], [2, 4], [3, 6]], decimal=5)

        self.presenter.notifyError.assert_not_called()
        savetxt.side_effect = OSError
        self.dialog.export()
        self.presenter.notifyError.assert_called()


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
        filters = 'All Files (*);;Python Files (*.py);;3D Files (*.stl *.obj)'
        d = FileDialog(self.view, '', '', filters)
        self.assertListEqual(d.extractFilters(filters), ['', '.py', '.stl', '.obj'])

        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)', )
        self.assertEqual(filename, '')
        self.mock_dialog_exec.assert_called_once()
        self.assertEqual(len(self.mock_dialog_exec.call_args[0]), 0)
        self.mock_message_box.assert_not_called()

        self.mock_select_filter.return_value = '3D Files (*.stl *.obj)'
        self.mock_select_file.return_value = ['unknown_file']
        self.mock_dialog_exec.return_value = QFileDialog.Accepted
        self.mock_isfile.return_value = False
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)')
        self.assertEqual(filename, '')
        self.mock_message_box.assert_called()
        self.assertEqual(len(self.mock_message_box.call_args[0]), 5)
        self.assertEqual(self.mock_dialog_exec.call_count, 2)

        self.mock_isfile.return_value = True
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)')
        self.assertEqual(filename, 'unknown_file.stl')
        self.assertEqual(len(self.mock_select_file.call_args[0]), 0)
        self.assertEqual(len(self.mock_select_filter.call_args[0]), 0)
        self.mock_select_file.return_value = ['unknown_file.STL']
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.stl *.obj)')
        self.assertEqual(filename, 'unknown_file.STL')
        self.mock_select_file.return_value = ['unknown_file.stl']
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.STL *.obj)')
        self.assertEqual(filename, 'unknown_file.stl')
        self.mock_select_file.return_value = ['unknown_file.obj']
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '', '3D Files (*.STL *.obj)')
        self.assertEqual(filename, 'unknown_file.obj')
        self.mock_select_file.return_value = ['unknown_file.py']
        filename = FileDialog.getOpenFileName(self.view, 'Import Sample Model', '',
                                              '3D Files (*.stl *.obj);;Python Files (*.py)')
        self.assertEqual(filename, 'unknown_file.py.stl')
        self.assertEqual(self.mock_message_box.call_count, 1)
        self.assertEqual(self.mock_dialog_exec.call_count, 7)

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


class TestTableModel(unittest.TestCase):
    app = QApplication([])

    def testPointModel(self):
        data = np.rec.array([([1., 2., 3.], True), ([4., 5., 6.], False), ([7., 8., 9.], True)],
                            dtype=[('points', 'f4', 3), ('enabled', '?')])
        model = PointModel(data)
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(model.columnCount(), 4)

        self.assertFalse(model.data(model.index(4, 4)).isValid())
        self.assertEqual(model.data(model.index(1, 1), Qt.EditRole), '5.000')
        self.assertEqual(model.data(model.index(1, 3), Qt.DisplayRole), '')
        self.assertEqual(model.data(model.index(2, 3), Qt.CheckStateRole), Qt.Checked)
        self.assertEqual(model.data(model.index(1, 3), Qt.CheckStateRole), Qt.Unchecked)

        self.assertFalse(model.setData(model.index(4, 4), 10.))
        self.assertTrue(model.setData(model.index(0, 0), 10.))
        self.assertEqual(model.data(model.index(0, 0), Qt.EditRole), '10.000')
        self.assertFalse(model.setData(model.index(0, 3), Qt.Unchecked))
        self.assertFalse(model.setData(model.index(0, 2), Qt.Unchecked, Qt.CheckStateRole))
        self.assertTrue(model.setData(model.index(0, 3), Qt.Unchecked, Qt.CheckStateRole))
        self.assertEqual(model.data(model.index(0, 3), Qt.CheckStateRole), Qt.Unchecked)

        model.toggleCheckState(3)
        self.assertTrue(np.all(model._data.enabled))
        model.toggleCheckState(3)
        self.assertTrue(np.all(model._data.enabled == False))
        self.assertEqual(model.flags(model.index(4, 4)), Qt.NoItemFlags)

        data = np.rec.array([([11., 21., 31.], True), ([41., 51., 61.], False), ([71., 81., 91.], False),
                             ([17., 18., 19.], True)], dtype=[('points', 'f4', 3), ('enabled', '?')])
        view_mock = mock.Mock()
        model.dataChanged = TestSignal()
        model.dataChanged.connect(view_mock)
        model.update(data)
        np.testing.assert_equal(model._data, data)
        view_mock.assert_called_once()
        top = view_mock.call_args[0][0]
        bottom = view_mock.call_args[0][1]
        self.assertEqual((top.row(), top.column()), (0, 0))
        self.assertEqual((bottom.row(), bottom.column()), (3, 3))

    def testAlignmentErrorModel(self):
        index = [0, 1, 2, 3]
        error = [0., np.nan, 0.2, 0.]
        enabled = [True, True, True, False]
        model = AlignmentErrorModel(index, error, enabled)
        self.assertEqual(model.rowCount(), 4)
        self.assertEqual(model.columnCount(), 3)

        self.assertFalse(model.data(model.index(4, 4)).isValid())
        self.assertEqual(model.data(model.index(1, 1), Qt.EditRole), 'N/A')
        self.assertEqual(model.data(model.index(3, 1), Qt.EditRole), '0.000')
        self.assertEqual(model.data(model.index(3, 0), Qt.DisplayRole), '4')
        self.assertEqual(model.data(model.index(1, 2), Qt.DisplayRole), '')
        self.assertEqual(model.data(model.index(0, 2), Qt.CheckStateRole), Qt.Checked)
        self.assertEqual(model.data(model.index(3, 2), Qt.CheckStateRole), Qt.Unchecked)
        self.assertEqual(model.data(model.index(1, 2), Qt.TextAlignmentRole), Qt.AlignCenter)
        self.assertIsInstance(model.data(model.index(0, 1), Qt.ForegroundRole), QBrush)
        self.assertIsInstance(model.data(model.index(2, 1), Qt.ForegroundRole), QBrush)
        self.assertFalse(model.data(model.index(2, 1), Qt.BackgroundRole).isValid())

        self.assertFalse(model.setData(model.index(4, 4), 10.))
        self.assertFalse(model.setData(model.index(0, 0), 5))
        self.assertFalse(model.setData(model.index(0, 1), 5))
        self.assertFalse(model.setData(model.index(0, 2), Qt.Unchecked, Qt.EditRole))
        self.assertTrue(model.setData(model.index(0, 2), Qt.Unchecked, Qt.CheckStateRole))
        self.assertEqual(model.data(model.index(0, 2), Qt.CheckStateRole), Qt.Unchecked)

        self.assertEqual(model.flags(model.index(4, 4)), Qt.NoItemFlags)
        self.assertNotEqual(model.flags(model.index(1, 2)) & Qt.ItemIsUserCheckable, Qt.NoItemFlags)

        view_mock = mock.Mock()
        model.dataChanged = TestSignal()
        model.dataChanged.connect(view_mock)
        model.update()
        view_mock.assert_called_once()
        top = view_mock.call_args[0][0]
        bottom = view_mock.call_args[0][1]
        self.assertEqual((top.row(), top.column()), (0, 0))
        self.assertEqual((bottom.row(), bottom.column()), (3, 2))

    def testErrorDetailModel(self):
        index = [0, 1, 2, 3, 4, 5]
        detail = np.array([[1., 1., 0.], [1., 4., 0.2], [1.41421356, 1.41421356, 0.],
                           [1.41421356, 1.41421356, 0.], [1., 1., 0.],   [1., 1., 0.]])
        model = ErrorDetailModel(index, detail)
        self.assertEqual(model.rowCount(), 15)
        self.assertEqual(model.columnCount(), 4)

        self.assertFalse(model.data(model.index(4, 5)).isValid())
        self.assertEqual(model.data(model.index(1, 0), Qt.DisplayRole), '(1, 3)')
        self.assertEqual(model.data(model.index(1, 1), Qt.DisplayRole), '1.000')
        self.assertEqual(model.data(model.index(1, 2), Qt.DisplayRole), '4.000')
        self.assertEqual(model.data(model.index(1, 3), Qt.DisplayRole), '0.200')
        self.assertEqual(model.data(model.index(1, 3), Qt.TextAlignmentRole), Qt.AlignCenter)
        self.assertIsInstance(model.data(model.index(1, 3), Qt.ForegroundRole), QBrush)
        self.assertIsInstance(model.data(model.index(2, 3), Qt.ForegroundRole), QBrush)
        self.assertFalse(model.setData(model.index(1, 1), 10.))

        model = ErrorDetailModel(index[3:], detail[3:])
        np.testing.assert_array_equal(model._index_pairs, ['(4, 5)', '(4, 6)', '(5, 6)'])

        view_mock = mock.Mock()
        model.dataChanged = TestSignal()
        model.dataChanged.connect(view_mock)
        model.update()
        view_mock.assert_called_once()
        top = view_mock.call_args[0][0]
        bottom = view_mock.call_args[0][1]
        self.assertEqual((top.row(), top.column()), (0, 0))
        self.assertEqual((bottom.row(), bottom.column()), (2, 3))


if __name__ == '__main__':
    unittest.main()
