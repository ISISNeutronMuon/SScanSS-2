import unittest
import unittest.mock as mock
from urllib.error import URLError, HTTPError
from matplotlib.backend_bases import MouseEvent
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent
from PyQt6.QtGui import QColor, QMouseEvent, QBrush, QAction
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QLabel
from sscanss.themes import ThemeManager, IconEngine
from sscanss.core.util import PointType, POINT_DTYPE, CommandID, TransformType, Attributes
from sscanss.core.geometry import Mesh, Volume
from sscanss.core.instrument.simulation import SimulationResult, Simulation
from sscanss.core.instrument.robotics import IKSolver, IKResult, SerialManipulator, Link
from sscanss.core.instrument.instrument import Script, PositioningStack, Instrument
from sscanss.core.scene import OpenGLRenderer, SceneManager
from sscanss.core.math import Matrix44
from sscanss.core.util import (StatusBar, ColourPicker, FileDialog, FilePicker, Accordion, Pane, FormControl, FormGroup,
                               CompareValidator, StyledTabWidget, MessageType, Primitives, SliderTextInput,
                               CustomIntValidator)
from sscanss.app.dialogs import (SimulationDialog, ScriptExportDialog, PathLengthPlotter, PointManager, VectorManager,
                                 DetectorControl, JawControl, PositionerControl, TransformDialog, AlignmentErrorDialog,
                                 CalibrationErrorDialog, VolumeLoader, InstrumentCoordinatesDialog, CurveEditor,
                                 SampleProperties, InsertPrimitiveDialog, ProgressDialog)
from sscanss.app.widgets import PointModel, AlignmentErrorModel, ErrorDetailModel
from sscanss.app.window.presenter import MainWindowPresenter
from sscanss.app.window.view import Updater
from sscanss.__version import Version
from tests.helpers import TestView, TestSignal, APP, TestWorker, create_mock, FakeSettings

dummy = "dummy"


class TestIconEngine(unittest.TestCase):
    def setUp(self):
        self.view = TestView()
        self.settings_mock = create_mock(self, "sscanss.themes.cfg.settings", instance=FakeSettings())
        self.path_for_mock = create_mock(self, "sscanss.themes.path_for")

    def testIcon(self):
        self.settings_mock.theme = 'light'
        self.path_for_mock.return_value = 'light/file.png'

        icon = IconEngine('file.png')
        path_orig = icon.path
        theme_orig = icon.theme

        self.settings_mock.theme = 'dark'
        self.path_for_mock.return_value = 'dark/file.png'

        icon.updateIcon()
        path_new = icon.path
        theme_new = icon.theme

        self.assertEqual(path_orig, 'light/file.png')
        self.assertEqual(theme_orig, 'light')
        self.assertEqual(path_new, 'dark/file.png')
        self.assertEqual(theme_new, 'dark')
        self.assertNotEqual(path_orig, path_new)


class TestFormWidgets(unittest.TestCase):
    def setUp(self):
        self.form_group = FormGroup()

        self.name = FormControl("Name", " ", required=True)
        self.email = FormControl("Email", "")

        self.height = FormControl("Height", 0.0, required=True, desc="cm", number=True, tooltip=dummy)
        self.weight = FormControl("Weight", 0.0, required=True, desc="kg", number=True)

        self.form_group.addControl(self.name)
        self.form_group.addControl(self.email)
        self.form_group.addControl(self.height)
        self.form_group.addControl(self.weight)

    def testRequiredValidation(self):
        self.assertEqual(self.name.value, " ")
        self.assertFalse(self.name.valid)
        self.assertTrue(self.email.valid)
        self.assertTrue(self.weight.valid)
        self.assertTrue(self.height.valid)

    def testGroupValidation(self):
        self.assertFalse(self.form_group.validateGroup())
        self.name.text = "Space"
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
            self.weight.value = "."

        self.height.text = "."
        self.assertFalse(self.height.valid)
        self.assertRaises(ValueError, lambda: self.height.value)

    def testToolTip(self):
        self.assertEqual(self.height.form_lineedit.toolTip(), dummy)
        self.assertEqual(self.weight.form_lineedit.toolTip(), '')


class TestSimulationDialog(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument.positioning_stack.name = dummy
        self.model_mock.return_value.simulation = None
        self.model_mock.return_value.simulation_created = TestSignal()
        self.model_mock.return_value.instrument_controlled = TestSignal()
        self.presenter = MainWindowPresenter(self.view)

        self.simulation_mock = mock.create_autospec(Simulation)
        self.simulation_mock.stopped = TestSignal()
        self.simulation_mock.positioner_name = dummy
        self.simulation_mock.validateInstrumentParameters.return_value = True
        self.simulation_mock.isRunning.return_value = True
        self.simulation_mock.detector_names = ["East"]
        self.simulation_mock.result_updated = TestSignal()
        self.simulation_mock.render_graphics = True

        self.view.presenter = self.presenter
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.showSelectChoiceMessage = mock.Mock(return_value="Cancel")
        self.dialog = SimulationDialog(self.view)

    def testSimulationResult(self):
        converged = IKResult([90], IKSolver.Status.Converged, (0.0, 0.1, 0.0), (0.1, 0.0, 0.0), True, True)
        not_converged = IKResult([87.8], IKSolver.Status.NotConverged, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), True, False)
        non_fatal = IKResult([45], IKSolver.Status.Failed, (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0), False, False)
        limit = IKResult([87.8], IKSolver.Status.HardwareLimit, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), True, False)
        unreachable = IKResult([87.8], IKSolver.Status.Unreachable, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), True, False)
        deformed = IKResult([87.8], IKSolver.Status.DeformedVectors, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), True, False)
        pose_matrix = Matrix44([[1, 0, 0, 1], [1, 0, 1, 2], [0, 1, 0, -1], [0, 0, 0, 1]])

        self.simulation_mock.results = [
            SimulationResult("1", converged, pose_matrix, (["X"], [90]), 0, (120, ), [False, False]),
            SimulationResult("2", converged, pose_matrix, (["X"], [90]), 0, (120, ), [False, True]),
            SimulationResult("3", not_converged, pose_matrix, (["X"], [87.8]), 0, (25, ), [True, True]),
            SimulationResult("4", non_fatal, pose_matrix, (["X"], [45]), 0),
            SimulationResult("5", limit, pose_matrix, (["X"], [87.8]), 0, (25, ), [True, True]),
            SimulationResult("6", unreachable, pose_matrix, (["X"], [87.8]), 0, (25, ), [True, True]),
            SimulationResult("7", deformed, pose_matrix, (["X"], [87.8]), 0, (25, ), [True, True]),
            SimulationResult("8", skipped=True, note="something happened"),
        ]
        self.simulation_mock.count = len(self.simulation_mock.results)
        self.simulation_mock.scene_size = 2
        self.model_mock.return_value.simulation = self.simulation_mock
        self.dialog.filter_button_group.button(0).toggle()
        self.model_mock.return_value.simulation_created.emit()
        self.simulation_mock.result_updated.emit(False)
        self.dialog.filter_button_group.button(3).toggle()

        np.testing.assert_array_equal(self.simulation_mock.results[1].pose_matrix, pose_matrix)
        np.testing.assert_array_equal(self.simulation_mock.results[7].pose_matrix, Matrix44.identity())

        self.assertEqual(self.dialog.result_counts[self.dialog.ResultKey.Good], 1)
        self.assertEqual(self.dialog.result_counts[self.dialog.ResultKey.Warn], 5)
        self.assertEqual(self.dialog.result_counts[self.dialog.ResultKey.Fail], 1)
        self.assertEqual(self.dialog.result_counts[self.dialog.ResultKey.Skip], 1)
        self.assertEqual(len(self.dialog.result_list.panes), self.simulation_mock.count)
        actions = self.dialog.result_list.panes[0].context_menu.actions()
        actions[0].trigger()  # copy action
        self.assertEqual(APP.clipboard().text(), "90.000")

        self.assertTrue(self.dialog.result_list.panes[0].isEnabled())
        self.assertTrue(self.dialog.result_list.panes[1].isEnabled())
        self.assertTrue(self.dialog.result_list.panes[2].isEnabled())
        self.assertFalse(self.dialog.result_list.panes[3].isEnabled())
        self.assertTrue(self.dialog.result_list.panes[4].isEnabled())
        self.assertTrue(self.dialog.result_list.panes[5].isEnabled())
        self.assertTrue(self.dialog.result_list.panes[6].isEnabled())
        self.assertFalse(self.dialog.result_list.panes[7].isEnabled())

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
        self.simulation_mock.positioner_name = "new"
        actions[1].trigger()
        self.model_mock.return_value.moveInstrument.assert_not_called()
        self.view.scenes.renderCollision.assert_not_called()
        self.simulation_mock.positioner_name = dummy
        self.simulation_mock.validateInstrumentParameters.return_value = False
        actions[1].trigger()
        self.model_mock.return_value.moveInstrument.assert_called()
        self.view.scenes.renderCollision.assert_not_called()

        self.simulation_mock.result_updated.emit(True)
        self.simulation_mock.isRunning.return_value = True
        self.dialog.close()
        self.simulation_mock.abort.assert_not_called()
        self.view.showSelectChoiceMessage.return_value = "Stop"
        self.dialog.close()
        self.simulation_mock.abort.assert_called_once()
        self.view.scenes.changeVisibility.assert_called()
        self.view.scenes.changeVisibility.reset_mock()
        self.model_mock.return_value.simulation = None
        self.dialog.close()
        self.view.scenes.changeVisibility.assert_not_called()


class TestPointManager(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
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
        table.setSelectionMode(table.SelectionMode.MultiSelection)
        table.selectRow(1)
        table.selectRow(2)
        self.dialog2.delete_button.click()
        self.presenter.deletePoints.assert_called_with([1, 2], PointType.Measurement)

    @mock.patch("sscanss.app.dialogs.insert.PickPointDialog")
    def testMovePoints(self, mock_pick_point_dialog):
        self.presenter.movePoints = mock.Mock()
        self.dialog1.move_up_button.click()
        self.presenter.movePoints.assert_not_called()

        self.dialog1.table_view.selectRow(0)
        self.dialog1.move_up_button.click()
        self.presenter.movePoints.assert_not_called()

        self.dialog1.table_view.selectRow(2)
        self.dialog1.move_up_button.click()
        self.presenter.movePoints.assert_called_with(2, 1, PointType.Fiducial)

        mock_pick_point_dialog.return_value.highlightPoints = mock.Mock()
        self.dialog2.rows_highlighted.connect(mock_pick_point_dialog.return_value.highlightPoints)

        self.presenter.movePoints.reset_mock()
        self.dialog2.move_down_button.click()
        self.presenter.movePoints.assert_not_called()

        self.dialog2.table_view.selectRow(2)
        self.dialog2.move_down_button.click()
        self.presenter.movePoints.assert_not_called()
        mock_pick_point_dialog.return_value.highlightPoints.assert_called_with([False, False, True])

        self.dialog2.table_view.selectRow(0)
        self.dialog2.move_down_button.click()
        self.presenter.movePoints.assert_called_with(0, 1, PointType.Measurement)
        mock_pick_point_dialog.return_value.highlightPoints.assert_called_with([True, False, False])

        table = self.dialog2.table_view
        table.setSelectionMode(table.SelectionMode.MultiSelection)
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

        self.dialog1.table_model.edit_completed.emit(points)
        self.presenter.editPoints.assert_called_with(points, PointType.Fiducial)

        self.dialog2.table_model.edit_completed.emit(points)
        self.presenter.editPoints.assert_called_with(points, PointType.Measurement)

        self.dialog2.selected = self.dialog2.table_model.index(1, 0)
        self.model_mock.return_value.measurement_points = points
        self.model_mock.return_value.measurement_points_changed.emit()
        np.testing.assert_array_almost_equal(points.points, self.dialog2.table_model._data.points, decimal=5)
        np.testing.assert_array_equal(points.enabled, self.dialog2.table_model._data.enabled)
        self.assertEqual(self.dialog2.table_view.currentIndex().row(), 1)

        self.view.scenes.reset_mock()
        self.dialog1.close()
        self.view.scenes.changeSelected.assert_called_once()


class TestVectorManager(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
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
        detectors = {"East": detectors[0], "West": detectors[1]}
        self.model_mock.return_value.instrument.detectors = detectors
        self.model_mock.return_value.measurement_vectors_changed = TestSignal()

        self.model_mock.return_value.measurement_vectors = np.ones((4, 6, 2))

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
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument_controlled = TestSignal()

        jaws = mock.Mock()
        jaws.aperture = [2, 2]
        jaws.aperture_upper_limit = [10, 10]
        jaws.aperture_lower_limit = [1, 1]
        q1 = Link("X", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0)
        q2 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi / 3.0, np.pi / 3.0, 0)
        jaws.positioner = SerialManipulator(f"jaw", [q1, q2])
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
        self.presenter.movePositioner.assert_called_with("jaw", [10, 0])

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
        self.presenter.ignorePositionerJointLimits.assert_called_with("jaw", 0, True)

        control.extra[0].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with("jaw", 0, False)

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
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.sample_changed = TestSignal()

        data = np.zeros([3, 3, 3], np.uint8)
        self.volume = Volume(data, np.ones(3), np.ones(3))

        vertices = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        self.model_mock.return_value.sample = Mesh(vertices, indices, normals)
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
        self.presenter.transformSample.assert_called_with([0.0, 0.0, 4.0], TransformType.Rotate)

        self.model_mock.return_value.sample = self.volume
        self.model_mock.return_value.sample_changed.emit()
        dialog.tool.y_rotation.value = 4.0
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, 4.0, 4.0], TransformType.Rotate)

        dialog.tool.y_rotation.value = 361.0
        self.assertFalse(dialog.tool.execute_button.isEnabled())
        dialog.tool.y_rotation.value = 360.0
        self.assertTrue(dialog.tool.execute_button.isEnabled())
        self.model_mock.return_value.sample = None
        self.model_mock.return_value.sample_changed.emit()
        self.assertFalse(dialog.tool.execute_button.isEnabled())

    def testTranslateTool(self):
        dialog = TransformDialog(TransformType.Translate, self.view)
        self.presenter.transformSample = mock.Mock()
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_not_called()

        dialog.tool.x_position.value = 4.0
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([4.0, 0.0, 0.0], TransformType.Translate)

        self.model_mock.return_value.sample = self.volume
        self.model_mock.return_value.sample_changed.emit()
        dialog.tool.y_position.value = 4.0
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([4.0, 4.0, 0.0], TransformType.Translate)

        dialog.tool.y_position.text = "a"
        self.assertFalse(dialog.tool.execute_button.isEnabled())
        dialog.tool.y_position.value = 4.0
        self.assertTrue(dialog.tool.execute_button.isEnabled())
        self.model_mock.return_value.sample = None
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
        self.model_mock.return_value.sample = self.volume
        self.model_mock.return_value.sample_changed.emit()
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with(random_matrix, TransformType.Custom)

        self.model_mock.return_value.sample = None
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
        self.presenter.transformSample.assert_called_with([-1.0, -1.0, -1.0], TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(1)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, -1.0, -1.0], TransformType.Translate)

        self.model_mock.return_value.sample = self.volume
        self.model_mock.return_value.sample_changed.emit()

        dialog.tool.ignore_combobox.setCurrentIndex(2)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([-2.5, 0.0, -2.5], TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(3)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([-2.5, -2.5, 0.0], TransformType.Translate)

        dialog.tool.move_combobox.setCurrentIndex(0)
        dialog.tool.ignore_combobox.setCurrentIndex(4)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([-1.0, 0.0, 0.0], TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(5)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, 0.0, -1.0], TransformType.Translate)

        dialog.tool.ignore_combobox.setCurrentIndex(6)
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called_with([0.0, -1.0, 0.0], TransformType.Translate)

        self.model_mock.return_value.sample = None
        self.model_mock.return_value.sample_changed.emit()
        dialog.tool.move_combobox.setCurrentIndex(1)
        self.assertFalse(dialog.tool.execute_button.isEnabled())

    @mock.patch("sscanss.app.dialogs.tools.rotation_btw_vectors", autospec=False)
    @mock.patch("sscanss.app.dialogs.tools.point_selection", autospec=True)
    def testPlaneAlignmentTool(self, select_mock, rot_vec_mock):
        self.view.gl_widget = mock.create_autospec(OpenGLRenderer)
        self.view.gl_widget.interactor = mock.Mock()
        self.view.gl_widget.interactor.ray_picked = TestSignal()
        self.view.gl_widget.picks = []

        self.view.gl_widget.interactor.picking = False

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
        self.assertTrue(self.view.gl_widget.interactor.picking)
        dialog.tool.select_button.click()
        self.assertFalse(self.view.gl_widget.interactor.picking)
        dialog.tool.pick_button.click()
        self.assertTrue(self.view.gl_widget.interactor.picking)

        self.assertEqual(dialog.tool.table_widget.rowCount(), 0)
        select_mock.return_value = np.array([])
        self.view.gl_widget.interactor.ray_picked.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 0)

        select_mock.return_value = np.array([[0.0, 0.0, 0.0]])
        self.view.gl_widget.interactor.ray_picked.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 1)

        select_mock.return_value = np.array([[1.0, 0.0, 0.0]])
        self.view.gl_widget.interactor.ray_picked.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 2)

        vertices = np.array([[1, 1, 1], [2, 0, 2], [2, 2, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        self.model_mock.return_value.sample = Mesh(vertices, indices, normals)
        self.model_mock.return_value.sample_changed.emit()

        self.assertIsNone(dialog.tool.initial_plane)
        select_mock.return_value = np.array([[0.0, 1.0, 0.0]])
        self.view.gl_widget.interactor.ray_picked.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 3)
        select_mock.assert_called()
        np.testing.assert_array_almost_equal(select_mock.call_args[0][2], [[1, 1, 1, 2, 0, 2, 2, 2, 0]], decimal=5)
        self.assertIsNotNone(dialog.tool.initial_plane)

        select_mock.return_value = np.array([[1.0, 1.0, 0.0]])
        self.view.gl_widget.interactor.ray_picked.emit(None, None)
        self.assertEqual(dialog.tool.table_widget.rowCount(), 4)

        self.assertEqual(
            self.view.gl_widget.picks,
            [[[0.0, 0.0, 0.0], False], [[1.0, 0.0, 0.0], False], [[0.0, 1.0, 0.0], False], [[1.0, 1.0, 0.0], False]],
        )

        dialog.tool.table_widget.selectRow(3)
        self.assertEqual(
            self.view.gl_widget.picks,
            [[[0.0, 0.0, 0.0], False], [[1.0, 0.0, 0.0], False], [[0.0, 1.0, 0.0], False], [[1.0, 1.0, 0.0], True]],
        )

        dialog.tool.removePicks()
        self.assertEqual(self.view.gl_widget.picks,
                         [[[0.0, 0.0, 0.0], False], [[1.0, 0.0, 0.0], False], [[0.0, 1.0, 0.0], False]])

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
        self.assertEqual(self.presenter.transformSample.call_args[0][1], TransformType.Custom)
        self.assertEqual(self.view.gl_widget.picks, [])

        matrix[:3, :3] = np.random.rand(3)
        rot_vec_mock.return_value = matrix[:3, :3]
        dialog.tool.initial_plane = plane
        dialog.tool.execute_button.click()
        self.presenter.transformSample.assert_called()
        np.testing.assert_array_almost_equal(self.presenter.transformSample.call_args[0][0], matrix, decimal=5)
        self.assertEqual(self.presenter.transformSample.call_args[0][1], TransformType.Custom)

        self.model_mock.return_value.sample = self.volume
        self.model_mock.return_value.sample_changed.emit()
        self.assertTrue(dialog.tool.execute_button.isEnabled())

        self.model_mock.return_value.sample = None
        self.model_mock.return_value.sample_changed.emit()
        self.assertFalse(dialog.tool.execute_button.isEnabled())

        dialog.close()
        self.assertFalse(self.view.gl_widget.interactor.picking)


class TestPositionerControl(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument_controlled = TestSignal()

        q1 = Link("X", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0)
        q2 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi / 2, np.pi / 2, 0)
        main_positioner = SerialManipulator("p1", [q1, q2])

        q3 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0)
        aux_positioner = SerialManipulator("p2", [q3])
        stack = PositioningStack("1", main_positioner)
        stack.addPositioner(aux_positioner)
        self.model_mock.return_value.instrument.positioners = {"p1": main_positioner, "p2": aux_positioner}
        self.model_mock.return_value.instrument.positioning_stack = stack
        self.model_mock.return_value.instrument.positioning_stacks = {"1": None, "2": None}

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
        self.presenter.movePositioner.assert_called_with("1", [10, 0, 0])

    def testChangeStack(self):
        self.presenter.changePositioningStack = mock.Mock()

        self.dialog.stack_combobox.textActivated.emit("1")
        self.presenter.changePositioningStack.assert_not_called()

        self.dialog.stack_combobox.textActivated.emit("2")
        self.presenter.changePositioningStack.assert_called_with("2")

    def testLockJoint(self):
        self.presenter.lockPositionerJoint = mock.Mock()

        control = self.dialog.positioner_form_controls[0]
        control.extra[0].click()
        self.presenter.lockPositionerJoint.assert_called_with("1", 0, True)

        control.extra[0].click()
        self.presenter.lockPositionerJoint.assert_called_with("1", 0, False)

    def testIgnoreLimits(self):
        self.presenter.ignorePositionerJointLimits = mock.Mock()

        control = self.dialog.positioner_form_controls[0]
        control.extra[1].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with("1", 0, True)

        control.extra[1].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with("1", 0, False)

    def testChangePositionerBase(self):
        self.presenter.changePositionerBase = mock.Mock()
        self.presenter.importTransformMatrix = mock.Mock()
        self.presenter.computePositionerBase = mock.Mock()
        self.presenter.exportBaseMatrix = mock.Mock()
        self.presenter.importTransformMatrix.return_value = None

        widget = self.dialog.positioner_forms_layout.itemAt(1).widget()
        title = widget.layout().itemAt(0).widget()
        base_button = title.title_layout.itemAt(2).widget()
        reset_button = self.dialog.base_reset_buttons["p2"]

        base_button.click()
        self.presenter.changePositionerBase.assert_not_called()

        matrix = np.random.rand(4, 4)
        positioner = self.dialog.instrument.positioning_stack.auxiliary[0]
        self.presenter.importTransformMatrix.return_value = matrix
        base_button.click()
        self.presenter.changePositionerBase.assert_called_with(positioner, matrix)
        reset_button.click()
        self.presenter.changePositionerBase.assert_called_with(positioner, positioner.default_base)

        self.presenter.changePositionerBase.reset_mock()
        actions = base_button.actions()
        compute_positioner_base = actions[1]
        export_base_matrix = actions[2]
        self.presenter.computePositionerBase.return_value = None
        compute_positioner_base.trigger()
        self.presenter.computePositionerBase.assert_called_with(positioner)
        self.presenter.changePositionerBase.assert_not_called()
        self.presenter.computePositionerBase.reset_mock()
        self.presenter.computePositionerBase.return_value = matrix
        compute_positioner_base.trigger()
        self.presenter.computePositionerBase.assert_called_with(positioner)
        self.presenter.changePositionerBase.assert_called_with(positioner, matrix)
        reset_button.click()
        self.presenter.changePositionerBase.assert_called_with(positioner, positioner.default_base)
        export_base_matrix.trigger()
        self.presenter.exportBaseMatrix.assert_called_with(positioner.default_base)

    def testUpdate(self):
        self.model_mock.return_value.instrument.positioning_stack.name = "2"
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
        button = self.dialog.base_reset_buttons["p2"]
        button.setVisible = visible_mock
        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangePositionerBase)
        button.setVisible.assert_called_with(False)
        self.dialog.instrument.positioning_stack.auxiliary[0].base = np.random.rand(4, 4)
        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangePositionerBase)
        button.setVisible.assert_called_with(True)


class TestDetectorControl(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
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
            q1 = Link("X", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0)
            q2 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi / 3.0, np.pi / 3.0, 0)
            detector.positioner = SerialManipulator(f"d{i+1}", [q1, q2])
            detectors.append(detector)
        detectors = {"East": detectors[0], "West": detectors[1]}
        self.model_mock.return_value.instrument.detectors = detectors

        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

        self.dialog = DetectorControl("East", self.view)

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
        self.presenter.movePositioner.assert_called_with("d1", [10, 0])

    def testIgnoreLimits(self):
        self.presenter.ignorePositionerJointLimits = mock.Mock()

        control = self.dialog.position_form_group.form_controls[0]
        control.extra[0].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with("d1", 0, True)

        control.extra[0].click()
        self.presenter.ignorePositionerJointLimits.assert_called_with("d1", 0, False)

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
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.save_path = dummy

        self.template_mock = mock.create_autospec(Script)
        self.template_mock.Key = Script.Key
        self.template_mock.keys = {key.value: "" for key in Script.Key}
        del self.template_mock.keys[Script.Key.mu_amps.value]
        self.template_mock.header_order = [Script.Key.mu_amps.value, Script.Key.position.value]
        self.template_mock.render.return_value = dummy

        self.simulation_mock = mock.create_autospec(Simulation)
        converged = IKResult([90], IKSolver.Status.Converged, (0.0, 0.1, 0.0), (0.1, 0.0, 0.0), True, True)
        not_converged = IKResult([87.8], IKSolver.Status.NotConverged, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), True, False)
        non_fatal = IKResult([45], IKSolver.Status.Failed, (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0), False, False)
        self.model_mock.return_value.instrument.script = self.template_mock
        self.simulation_mock.results = [
            SimulationResult("1", converged, Matrix44.identity(), (["X"], [90]), 0, (120, ), [False, True]),
            SimulationResult("3", non_fatal, Matrix44.identity(), (["X"], [45]), 0, None, None),
            SimulationResult("2", not_converged, Matrix44.identity(), (["X"], [87.8]), 0, (25, ), [True, True]),
        ]

        self.presenter = MainWindowPresenter(self.view)
        self.view.presenter = self.presenter
        self.dialog = ScriptExportDialog(self.simulation_mock, self.view)

    def testRendering(self):
        self.assertEqual(self.dialog.preview_label.toPlainText(), dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.filename.value], dummy)
        self.assertEqual(self.template_mock.keys[Script.Key.header.value], f"{Script.Key.mu_amps.value}\tX")
        self.assertEqual(self.template_mock.keys[Script.Key.count.value], 2)
        self.assertEqual(self.template_mock.keys[Script.Key.position.value], "")
        self.assertEqual(self.template_mock.keys[Script.Key.script.value], [{
            "position": "90.000"
        }, {
            "position": "87.800"
        }])

        self.assertFalse(self.dialog.show_mu_amps)
        self.assertFalse(hasattr(self.dialog, "micro_amp_textbox"))

        for _ in range(9):
            self.simulation_mock.results.append(self.simulation_mock.results[0])

        self.template_mock.keys[Script.Key.mu_amps.value] = ""
        self.dialog = ScriptExportDialog(self.simulation_mock, self.view)
        self.assertEqual(self.template_mock.keys[Script.Key.mu_amps.value], "0.000")
        self.dialog.micro_amp_textbox.setText("4.500")
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
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.view.showSaveDialog = mock.Mock()
        self.model_mock = model_mock
        self.model_mock.return_value.save_path = ""
        self.model_mock.return_value.instruments = [dummy]
        self.presenter = MainWindowPresenter(self.view)

        self.simulation_mock = mock.create_autospec(Simulation)
        self.model_mock.return_value.simulation = self.simulation_mock
        shape = (1, 1, 1)
        self.simulation_mock.shape = shape
        self.simulation_mock.args = {"align_first_order": True}
        self.simulation_mock.path_lengths = np.zeros(shape)
        self.simulation_mock.detector_names = ["East"]

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
        self.simulation_mock.detector_names = ["East", "West"]
        self.dialog = PathLengthPlotter(self.view)

        combo = self.dialog.detector_combobox
        self.assertListEqual(self.simulation_mock.detector_names, [combo.itemText(i) for i in range(1, combo.count())])
        combo = self.dialog.alignment_combobox
        self.assertListEqual(["1", "2", "3", "4"], [combo.itemText(i) for i in range(1, combo.count())])
        line = self.dialog.axes.lines[0]
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 8], line.get_xdata().astype(int).tolist())
        self.assertListEqual([1, 0, 0, 5, 3, 0, 0, 7], line.get_ydata().astype(int).tolist())
        line = self.dialog.axes.lines[1]
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 8], line.get_xdata().astype(int).tolist())
        self.assertListEqual([2, 0, 0, 6, 4, 0, 0, 8], line.get_ydata().astype(int).tolist())

        self.simulation_mock.args = {"align_first_order": False}
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

    @mock.patch("sscanss.app.dialogs.misc.np.savetxt", autospec=True)
    def testExport(self, savetxt):
        self.presenter.notifyError = mock.Mock()
        self.dialog.figure.savefig = mock.Mock()
        self.view.showSaveDialog.return_value = ""
        self.dialog.export()
        savetxt.assert_not_called()
        self.dialog.figure.savefig.assert_not_called()

        self.view.showSaveDialog.return_value = "demo.txt"
        self.dialog.export()
        savetxt.assert_called_once()
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], [[1.0, 0.0]], decimal=5)

        self.view.showSaveDialog.return_value = "demo.png"
        self.dialog.export()
        self.dialog.figure.savefig.assert_called_once()

        shape = (3, 2, 2)
        self.simulation_mock.shape = shape
        self.simulation_mock.path_lengths = np.zeros(shape)
        self.simulation_mock.path_lengths[:, :, 0] = [[1, 2], [3, 4], [5, 6]]
        self.simulation_mock.detector_names = ["East", "West"]
        self.dialog = PathLengthPlotter(self.view)

        self.view.showSaveDialog.return_value = "demo.txt"
        self.dialog.export()
        expected = [[1, 1, 2], [2, 0, 0], [3, 3, 4], [4, 0, 0], [5, 5, 6], [6, 0, 0]]
        self.assertEqual(savetxt.call_count, 2)
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], expected, decimal=5)
        self.assertEqual(len(savetxt.call_args[1]["fmt"]), 3)

        combo = self.dialog.alignment_combobox
        combo.setCurrentIndex(2)
        combo.activated.emit(2)
        self.dialog.export()
        self.assertEqual(savetxt.call_count, 3)
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], [[1, 0, 0], [2, 0, 0], [3, 0, 0]], decimal=5)
        self.assertEqual(len(savetxt.call_args[1]["fmt"]), 3)

        combo.setCurrentIndex(1)
        combo = self.dialog.detector_combobox
        combo.setCurrentIndex(2)
        combo.activated.emit(2)
        self.dialog.export()
        self.assertEqual(savetxt.call_count, 4)
        np.testing.assert_array_almost_equal(savetxt.call_args[0][1], [[1, 2], [2, 4], [3, 6]], decimal=5)
        self.assertEqual(len(savetxt.call_args[1]["fmt"]), 2)

        self.presenter.notifyError.assert_not_called()
        savetxt.side_effect = OSError
        self.dialog.export()
        self.presenter.notifyError.assert_called()


class TestStatusBar(unittest.TestCase):
    def testWidgetManagement(self):
        widget = StatusBar()
        compound_widget_1 = FormControl("Name", dummy)
        self.assertEqual(widget.left_layout.count(), 0)
        self.assertEqual(widget.right_layout.count(), 0)
        widget.addPermanentWidget(compound_widget_1, alignment=Qt.AlignmentFlag.AlignRight)
        self.assertEqual(widget.left_layout.count(), 0)
        self.assertEqual(widget.right_layout.count(), 1)
        compound_widget_2 = FormControl("Age", dummy)
        widget.addPermanentWidget(compound_widget_2, alignment=Qt.AlignmentFlag.AlignLeft)
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
        message = "Hello, World!"
        widget.showMessage(message)
        self.assertEqual(widget.currentMessage(), message)
        widget.clearMessage()
        self.assertEqual(widget.currentMessage(), "")

        widget.showMessage(message, 2)
        self.assertEqual(widget.currentMessage(), message)
        self.assertEqual(widget.timer.singleShot.call_args[0][0], 2)
        clear_function = widget.timer.singleShot.call_args[0][1]
        clear_function()
        self.assertEqual(widget.currentMessage(), "")


class TestFileDialog(unittest.TestCase):
    def setUp(self):
        self.view = TestView()

        self.mock_select_filter = create_mock(self,
                                              "sscanss.core.util.widgets.QtWidgets.QFileDialog.selectedNameFilter",
                                              autospec=False)
        self.mock_select_file = create_mock(self,
                                            "sscanss.core.util.widgets.QtWidgets.QFileDialog.selectedFiles",
                                            autospec=False)
        self.mock_isfile = create_mock(self, "sscanss.core.util.widgets.os.path.isfile")
        self.mock_dialog_exec = create_mock(self,
                                            "sscanss.core.util.widgets.QtWidgets.QFileDialog.exec",
                                            autospec=False)
        self.mock_message_box = create_mock(self,
                                            "sscanss.core.util.widgets.QtWidgets.QMessageBox.warning",
                                            autospec=False)

    def testOpenFileDialog(self):
        filters = "All Files (*);;Python Files (*.py);;3D Files (*.stl *.obj)"
        d = FileDialog(self.view, "", "", filters)
        self.assertListEqual(d.extractFilters(filters), ["", ".py", ".stl", ".obj"])

        filename = FileDialog.getOpenFileName(
            self.view,
            "Import Sample Model",
            "",
            "3D Files (*.stl *.obj)",
        )
        self.assertEqual(filename, "")
        self.mock_dialog_exec.assert_called_once()
        self.assertEqual(len(self.mock_dialog_exec.call_args[0]), 0)
        self.mock_message_box.assert_not_called()

        self.mock_select_filter.return_value = "3D Files (*.stl *.obj)"
        self.mock_select_file.return_value = ["unknown_file"]
        self.mock_dialog_exec.return_value = QFileDialog.DialogCode.Accepted
        self.mock_isfile.return_value = False
        filename = FileDialog.getOpenFileName(self.view, "Import Sample Model", "", "3D Files (*.stl *.obj)")
        self.assertEqual(filename, "")
        self.mock_message_box.assert_called()
        self.assertEqual(len(self.mock_message_box.call_args[0]), 5)
        self.assertEqual(self.mock_dialog_exec.call_count, 2)

        self.mock_isfile.return_value = True
        filename = FileDialog.getOpenFileName(self.view, "Import Sample Model", "", "3D Files (*.stl *.obj)")
        self.assertEqual(filename, "unknown_file.stl")
        self.assertEqual(len(self.mock_select_file.call_args[0]), 0)
        self.assertEqual(len(self.mock_select_filter.call_args[0]), 0)
        self.mock_select_file.return_value = ["unknown_file.STL"]
        filename = FileDialog.getOpenFileName(self.view, "Import Sample Model", "", "3D Files (*.stl *.obj)")
        self.assertEqual(filename, "unknown_file.STL")
        self.mock_select_file.return_value = ["unknown_file.stl"]
        filename = FileDialog.getOpenFileName(self.view, "Import Sample Model", "", "3D Files (*.STL *.obj)")
        self.assertEqual(filename, "unknown_file.stl")
        self.mock_select_file.return_value = ["unknown_file.obj"]
        filename = FileDialog.getOpenFileName(self.view, "Import Sample Model", "", "3D Files (*.STL *.obj)")
        self.assertEqual(filename, "unknown_file.obj")
        self.mock_select_file.return_value = ["unknown_file.py"]
        filename = FileDialog.getOpenFileName(self.view, "Import Sample Model", "",
                                              "3D Files (*.stl *.obj);;Python Files (*.py)")
        self.assertEqual(filename, "unknown_file.py.stl")
        self.assertEqual(self.mock_message_box.call_count, 1)
        self.assertEqual(self.mock_dialog_exec.call_count, 7)

    def testSaveFileDialog(self):
        filename = FileDialog.getSaveFileName(
            self.view,
            "Import Sample Model",
            "",
            "3D Files (*.stl *.obj)",
        )
        self.assertEqual(filename, "")
        self.mock_dialog_exec.assert_called_once()
        self.assertEqual(len(self.mock_dialog_exec.call_args[0]), 0)
        self.mock_message_box.assert_not_called()

        self.mock_select_filter.return_value = "3D Files (*.stl *.obj)"
        self.mock_select_file.return_value = ["unknown_file"]
        self.mock_dialog_exec.return_value = QFileDialog.DialogCode.Accepted
        self.mock_message_box.return_value = QMessageBox.StandardButton.No
        self.mock_isfile.return_value = True
        filename = FileDialog.getSaveFileName(
            self.view,
            "Import Sample Model",
            "",
            "3D Files (*.stl *.obj)",
        )
        self.assertEqual(filename, "")
        self.mock_message_box.assert_called()
        self.assertEqual(len(self.mock_message_box.call_args[0]), 5)
        self.assertEqual(self.mock_dialog_exec.call_count, 2)

        self.mock_isfile.return_value = False
        filename = FileDialog.getSaveFileName(
            self.view,
            "Import Sample Model",
            "",
            "3D Files (*.stl *.obj)",
        )
        self.assertEqual(filename, "unknown_file.stl")
        self.assertEqual(self.mock_message_box.call_count, 1)
        self.assertEqual(self.mock_dialog_exec.call_count, 3)
        self.assertEqual(len(self.mock_select_file.call_args[0]), 0)
        self.assertEqual(len(self.mock_select_filter.call_args[0]), 0)


class TestSelectionWidgets(unittest.TestCase):
    @mock.patch("sscanss.core.util.widgets.QtWidgets.QColorDialog", autospec=True)
    def testColourPicker(self, color_dialog):
        colour = QColor(Qt.GlobalColor.black)
        widget = ColourPicker(colour)
        self.assertEqual(widget.value, colour)
        self.assertEqual(widget.colour_name.text(), colour.name())

        colour = QColor(Qt.GlobalColor.red)
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

    @mock.patch("sscanss.core.util.widgets.FileDialog", autospec=True)
    def testFilePicker(self, file_dialog):
        path = "some_file.txt"
        widget = FilePicker(path, False)
        self.assertEqual(widget.value, path)

        new_path = "some_other_file.txt"
        file_dialog.getOpenFileName.return_value = new_path
        widget.openFileDialog()
        self.assertEqual(widget.value, new_path)
        self.assertEqual(file_dialog.getOpenFileName.call_count, 1)
        self.assertEqual(file_dialog.getExistingDirectory.call_count, 0)

        new_path = "yet another_file.txt"
        file_dialog.getExistingDirectory.return_value = new_path
        widget.select_folder = True
        widget.openFileDialog()
        self.assertEqual(widget.value, new_path)
        self.assertEqual(file_dialog.getOpenFileName.call_count, 1)
        self.assertEqual(file_dialog.getExistingDirectory.call_count, 1)


class TestStyledTabWidget(unittest.TestCase):
    def testWidget(self):
        widget = StyledTabWidget()
        self.assertEqual(len(widget.tabs.buttons()), 0)
        widget.addTab("Tab 1")
        self.assertEqual(len(widget.tabs.buttons()), 1)
        self.assertTrue(widget.tabs.button(0).isChecked())
        self.assertEqual(widget.stack.currentIndex(), 0)
        widget.addTab("Tab 2", True)
        self.assertEqual(len(widget.tabs.buttons()), 2)
        self.assertFalse(widget.tabs.button(0).isChecked())
        self.assertTrue(widget.tabs.button(1).isChecked())
        self.assertEqual(widget.stack.currentIndex(), 1)


class TestAccordion(unittest.TestCase):
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
        pane_1 = Pane(QLabel(), QLabel(), MessageType.Warning)
        pane_2 = Pane(QLabel(), QLabel(), MessageType.Error)

        accordion = Accordion()
        accordion.addPane(pane_1)
        self.assertEqual(len(accordion.panes), 1)
        accordion.addPane(pane_2)
        self.assertEqual(len(accordion.panes), 2)
        self.assertRaises(TypeError, accordion.addPane, "Pane")
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
        APP.sendEvent(
            pane,
            QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier))
        self.assertTrue(self.pane_content_visible)

        a = QAction("Some Action!")
        self.assertEqual(len(pane.context_menu.actions()), 0)
        pane.addContextMenuAction(a)
        self.assertEqual(len(pane.context_menu.actions()), 1)
        pane.customContextMenuRequested.emit(QPoint(100, 250))
        self.assertTrue(self.context_menu_visible)


class TestTableModel(unittest.TestCase):
    def testPointModel(self):
        data = np.rec.array(
            [([1.0, 2.0, 3.0], True), ([4.0, 5.0, 6.0], False), ([7.0, 8.0, 9.0], True)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )
        model = PointModel(data)
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(model.columnCount(), 4)

        self.assertFalse(model.data(model.index(4, 4)).isValid())
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.EditRole), "5.000")
        self.assertEqual(model.data(model.index(1, 3), Qt.ItemDataRole.DisplayRole), "")
        self.assertEqual(model.data(model.index(2, 3), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Checked)
        self.assertEqual(model.data(model.index(1, 3), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Unchecked)

        self.assertFalse(model.setData(model.index(4, 4), 10.0))
        self.assertTrue(model.setData(model.index(0, 0), 10.0))
        self.assertEqual(model.data(model.index(0, 0), Qt.ItemDataRole.EditRole), "10.000")
        self.assertFalse(model.setData(model.index(0, 3), Qt.CheckState.Unchecked.value))
        self.assertFalse(model.setData(model.index(0, 2), Qt.CheckState.Unchecked.value,
                                       Qt.ItemDataRole.CheckStateRole))
        self.assertTrue(model.setData(model.index(0, 3), Qt.CheckState.Unchecked.value, Qt.ItemDataRole.CheckStateRole))
        self.assertEqual(model.data(model.index(0, 3), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Unchecked)

        model.toggleCheckState(3)
        self.assertTrue(np.all(model._data.enabled))
        model.toggleCheckState(3)
        self.assertTrue(np.all(model._data.enabled == False))
        self.assertEqual(model.flags(model.index(4, 4)), Qt.ItemFlag.NoItemFlags)

        data = np.rec.array(
            [
                ([11.0, 21.0, 31.0], True),
                ([41.0, 51.0, 61.0], False),
                ([71.0, 81.0, 91.0], False),
                ([17.0, 18.0, 19.0], True),
            ],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )
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
        index = np.array([0, 1, 2, 3])
        error = np.array([0.0, np.nan, 0.2, 0.0])
        enabled = np.array([True, True, True, False])
        model = AlignmentErrorModel(index, error, enabled)
        self.assertEqual(model.rowCount(), 4)
        self.assertEqual(model.columnCount(), 3)

        self.assertFalse(model.data(model.index(4, 4)).isValid())
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.EditRole), "N/A")
        self.assertEqual(model.data(model.index(3, 1), Qt.ItemDataRole.EditRole), "0.000")
        self.assertEqual(model.data(model.index(3, 0), Qt.ItemDataRole.DisplayRole), "4")
        self.assertEqual(model.data(model.index(1, 2), Qt.ItemDataRole.DisplayRole), "")
        self.assertEqual(model.data(model.index(0, 2), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Checked)
        self.assertEqual(model.data(model.index(3, 2), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Unchecked)
        self.assertEqual(model.data(model.index(1, 2), Qt.ItemDataRole.TextAlignmentRole), Qt.AlignmentFlag.AlignCenter)
        self.assertIsInstance(model.data(model.index(0, 1), Qt.ItemDataRole.ForegroundRole), QBrush)
        self.assertIsInstance(model.data(model.index(2, 1), Qt.ItemDataRole.ForegroundRole), QBrush)
        self.assertFalse(model.data(model.index(2, 1), Qt.ItemDataRole.BackgroundRole).isValid())

        self.assertFalse(model.setData(model.index(4, 4), 10.0))
        self.assertFalse(model.setData(model.index(0, 0), 5))
        self.assertFalse(model.setData(model.index(0, 1), 5))
        self.assertFalse(model.setData(model.index(0, 2), Qt.CheckState.Unchecked.value, Qt.ItemDataRole.EditRole))
        self.assertTrue(model.setData(model.index(0, 2), Qt.CheckState.Unchecked.value, Qt.ItemDataRole.CheckStateRole))
        self.assertEqual(model.data(model.index(0, 2), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Unchecked)

        self.assertEqual(model.flags(model.index(4, 4)), Qt.ItemFlag.NoItemFlags)
        self.assertNotEqual(model.flags(model.index(1, 2)) & Qt.ItemFlag.ItemIsUserCheckable, Qt.ItemFlag.NoItemFlags)

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
        detail = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 4.0, 0.2],
            [1.41421356, 1.41421356, 0.0],
            [1.41421356, 1.41421356, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        model = ErrorDetailModel(index, detail)
        self.assertEqual(model.rowCount(), 15)
        self.assertEqual(model.columnCount(), 4)

        self.assertFalse(model.data(model.index(4, 5)).isValid())
        self.assertEqual(model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole), "(1, 3)")
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.DisplayRole), "1.000")
        self.assertEqual(model.data(model.index(1, 2), Qt.ItemDataRole.DisplayRole), "4.000")
        self.assertEqual(model.data(model.index(1, 3), Qt.ItemDataRole.DisplayRole), "0.200")
        self.assertEqual(model.data(model.index(1, 3), Qt.ItemDataRole.TextAlignmentRole), Qt.AlignmentFlag.AlignCenter)
        self.assertIsInstance(model.data(model.index(1, 3), Qt.ItemDataRole.ForegroundRole), QBrush)
        self.assertIsInstance(model.data(model.index(2, 3), Qt.ItemDataRole.ForegroundRole), QBrush)
        self.assertFalse(model.setData(model.index(1, 1), 10.0))

        model = ErrorDetailModel(index[3:], detail[3:])
        np.testing.assert_array_equal(model._index_pairs, ["(4, 5)", "(4, 6)", "(5, 6)"])

        view_mock = mock.Mock()
        model.dataChanged = TestSignal()
        model.dataChanged.connect(view_mock)
        model.update()
        view_mock.assert_called_once()
        top = view_mock.call_args[0][0]
        bottom = view_mock.call_args[0][1]
        self.assertEqual((top.row(), top.column()), (0, 0))
        self.assertEqual((bottom.row(), bottom.column()), (2, 3))


class TestCalibrationErrorDialog(unittest.TestCase):
    def testWidget(self):
        pose_id = np.array([1, 2, 3])
        fiducial_id = np.array([3, 2, 1])
        error = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        widget = CalibrationErrorDialog(None, pose_id, fiducial_id, error)

        self.assertEqual(widget.error_table.item(0, 0).text(), "1")
        self.assertEqual(widget.error_table.item(1, 0).text(), "2")
        self.assertEqual(widget.error_table.item(2, 0).text(), "3")

        self.assertEqual(widget.error_table.item(0, 1).text(), "3")
        self.assertEqual(widget.error_table.item(1, 1).text(), "2")
        self.assertEqual(widget.error_table.item(2, 1).text(), "1")

        self.assertEqual(widget.error_table.item(0, 2).text(), "1.000")
        self.assertEqual(widget.error_table.item(1, 2).text(), "0.000")
        self.assertEqual(widget.error_table.item(2, 2).text(), "0.000")

        self.assertEqual(widget.error_table.item(0, 3).text(), "0.000")
        self.assertEqual(widget.error_table.item(1, 3).text(), "1.000")
        self.assertEqual(widget.error_table.item(2, 3).text(), "0.000")

        self.assertEqual(widget.error_table.item(0, 4).text(), "0.000")
        self.assertEqual(widget.error_table.item(1, 4).text(), "0.000")
        self.assertEqual(widget.error_table.item(2, 4).text(), "1.000")

        self.assertEqual(widget.error_table.item(0, 5).text(), "1.000")
        self.assertEqual(widget.error_table.item(1, 5).text(), "1.000")
        self.assertEqual(widget.error_table.item(2, 5).text(), "1.000")


class TestAlignmentErrorDialog(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.instrument.positioning_stack.name = dummy
        self.model_mock.return_value.fiducials = np.rec.array(
            [([0, 0, 0], True), ([1, 0, 0], True), ([0, 1, 0], True), ([1, 1, 0], True)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )

        self.presenter = MainWindowPresenter(self.view)
        self.presenter.alignSample = mock.Mock()
        self.presenter.movePositioner = mock.Mock()
        self.view.presenter = self.presenter
        self.view.scenes = mock.create_autospec(SceneManager)

    @mock.patch("sscanss.app.dialogs.misc.Banner.showMessage")
    def testWidgetGoodResult(self, banner_mock):
        indices = np.array([0, 1, 2, 3])
        enabled = np.array([True, True, True, True])
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        transform_result = self.presenter.rigidTransform(indices, points, enabled)
        end_q = [0.0] * 4
        order_fix = None

        widget = AlignmentErrorDialog(self.view, indices, enabled, points, transform_result, end_q, order_fix)
        self.assertEqual(widget.banner.action_button.text(), "ACTION")
        banner_mock.assert_not_called()

        model = widget.summary_table_view.model()
        self.assertEqual(model.rowCount(), 4)
        self.assertEqual(model.columnCount(), 3)

        self.assertEqual(model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole), "1")
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.DisplayRole), "0.000")
        self.assertEqual(model.data(model.index(2, 2), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Checked)
        self.assertTrue(model.setData(model.index(0, 2), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole))
        self.assertTrue(model.setData(model.index(1, 2), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole))

        model = widget.detail_table_view.model()
        self.assertEqual(model.rowCount(), 6)
        self.assertEqual(model.columnCount(), 4)

        self.assertEqual(model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole), "(1, 2)")
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.DisplayRole), "1.000")
        self.assertEqual(model.data(model.index(2, 2), Qt.ItemDataRole.DisplayRole), "1.414")
        self.assertEqual(model.data(model.index(3, 3), Qt.ItemDataRole.DisplayRole), "0.000")

        widget.recalculate_button.click()
        banner_mock.assert_called_once()
        model = widget.summary_table_view.model()
        self.assertTrue(model.setData(model.index(0, 2), Qt.CheckState.Checked.value, Qt.ItemDataRole.CheckStateRole))
        widget.recalculate_button.click()
        banner_mock.assert_called_once()

        widget.accept_button.click()
        self.presenter.alignSample.assert_called()
        self.presenter.movePositioner.assert_called()
        self.assertEqual(self.presenter.movePositioner.call_args[0][0], dummy)

    @mock.patch("sscanss.app.dialogs.misc.Banner.showMessage")
    def testWidgetFixResult(self, banner_mock):
        indices = np.array([0, 1, 2, 3])
        enabled = np.array([True, True, True, True])
        points = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        transform_result = self.presenter.rigidTransform(indices, points, enabled)
        end_q = [0.0] * 4
        order_fix = np.array([1, 0, 2, 3])

        widget = AlignmentErrorDialog(self.view, indices, enabled, points, transform_result, end_q, order_fix)
        self.assertEqual(widget.banner.action_button.text(), "FIX")
        banner_mock.assert_called()

        model = widget.summary_table_view.model()
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.DisplayRole), "1.000")
        widget.banner.action_button.click()
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.DisplayRole), "0.000")

        widget.check_box.click()
        widget.accept_button.click()
        self.presenter.alignSample.assert_called()
        self.presenter.movePositioner.assert_not_called()

    @mock.patch("sscanss.app.dialogs.misc.Banner.showMessage")
    def testWidgetBadResult(self, banner_mock):
        indices = np.array([0, 1, 2, 3])
        enabled = np.array([True, True, True, False])
        points = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        transform_result = self.presenter.rigidTransform(indices, points, enabled)
        end_q = [0.0] * 4
        order_fix = None

        widget = AlignmentErrorDialog(self.view, indices, enabled, points, transform_result, end_q, order_fix)
        self.assertEqual(widget.banner.action_button.text(), "ACTION")
        banner_mock.assert_called_once()

        model = widget.summary_table_view.model()
        self.assertEqual(model.rowCount(), 4)
        self.assertEqual(model.columnCount(), 3)
        self.assertEqual(model.data(model.index(1, 1), Qt.ItemDataRole.DisplayRole), "0.316")
        self.assertEqual(model.data(model.index(3, 2), Qt.ItemDataRole.CheckStateRole), Qt.CheckState.Unchecked)

        widget.check_box.click()
        widget.cancel_button.click()
        self.presenter.alignSample.assert_not_called()
        self.presenter.movePositioner.assert_not_called()


class TestVolumeLoader(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]

        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

        self.dialog = VolumeLoader(self.view)
        self.presenter.importVolume = mock.Mock()

    def testValidation(self):
        self.assertFalse(self.dialog.execute_button.isEnabled())
        self.dialog.filepath_picker.value = 'dummypath'
        self.assertTrue(self.dialog.execute_button.isEnabled())

        for box in self.dialog.pixel_size_group.form_controls:
            box.value = 0.00001
            self.assertFalse(self.dialog.execute_button.isEnabled())
            box.value = 100000
            self.assertFalse(self.dialog.execute_button.isEnabled())
            box.value = 2.0
            self.assertTrue(self.dialog.execute_button.isEnabled())

    def testExecution(self):
        self.dialog.filepath_picker.value = 'dummypath'
        for i, box in enumerate(self.dialog.pixel_size_group.form_controls):
            box.value = i + 1.0
        for i, box in enumerate(self.dialog.pixel_centre_group.form_controls):
            box.value = i
        self.presenter.importVolume.assert_not_called()
        self.assertTrue(self.dialog.execute_button.isEnabled())
        self.dialog.execute_button.click()
        self.presenter.importVolume.assert_called_with('dummypath', [1.0, 2.0, 3.0], [0.0, 1.0, 2.0])


class TestInstrumentCoordinatesDialog(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.mock_instrument = mock.create_autospec(Instrument)
        self.mock_instrument.positioning_stack = self.createPositioningStack()

        self.model_mock = model_mock
        self.model_mock.return_value.instruments = ['dummy']
        self.model_mock.return_value.instrument = self.mock_instrument
        self.model_mock.return_value.fiducials_changed = TestSignal()
        self.model_mock.return_value.model_changed = TestSignal()
        self.model_mock.return_value.instrument_controlled = TestSignal()

        points = np.rec.array([([0.0, 1.0, 2.0], False), ([0.0, 0.0, 0.0], True)], dtype=POINT_DTYPE)
        self.model_mock.return_value.fiducials = points
        self.presenter = MainWindowPresenter(self.view)
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter
        self.model_mock.return_value.alignment = None
        self.model_mock.return_value.project_data = {}
        self.dialog = InstrumentCoordinatesDialog(self.view)

    @staticmethod
    def createPositioningStack():
        q1 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        q2 = Link("Y", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0)
        s = SerialManipulator("", [q1, q2], custom_order=[1, 0], base=Matrix44.fromTranslation([0.0, 0.0, 50.0]))
        return PositioningStack(s.name, s)

    @mock.patch('sscanss.app.dialogs.misc.np.savetxt', autospec=True)
    @mock.patch('sscanss.app.dialogs.misc.FileDialog', autospec=True)
    def testInstrumentCoordinatesDialog(self, file_dialog_mock, save_txt):
        # Test non-aligned case that fiducials are parsed unchanged
        self.assertEqual(self.model_mock.return_value.fiducials['points'][0, 0],
                         float(self.dialog.fiducial_table_widget.item(0, 0).text()))
        self.assertEqual(self.model_mock.return_value.fiducials['points'][1, 1],
                         float(self.dialog.fiducial_table_widget.item(1, 1).text()))

        # Should not export fiducial coordinates if sample not aligned on instrument
        self.presenter.exportCurrentFiducials = mock.Mock()
        self.assertFalse(self.dialog.export_fiducials_action.isEnabled())
        self.dialog.exportFiducials()
        self.presenter.exportCurrentFiducials.assert_not_called()

        # Align sample on instrument at zero and test fiducial is where expected
        self.mock_instrument.positioning_stack.fkine([0, 0], set_point=True)
        self.model_mock.return_value.alignment = Matrix44.identity()
        self.model_mock.return_value.model_changed.emit(Attributes.Instrument, None)
        for i in range(3):
            self.assertAlmostEqual(float(i), float(self.dialog.fiducial_table_widget.item(0, i).text()), places=5)
            self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(1, i).text()), places=5)

        # Move positioning stack and test only correct fiducial coordinates moved
        self.mock_instrument.positioning_stack.fkine([0, 10], set_point=True)
        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangePositionerBase)

        self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(0, 0).text()), places=5)
        self.assertAlmostEqual(11.0, float(self.dialog.fiducial_table_widget.item(0, 1).text()), places=5)
        self.assertAlmostEqual(2.0, float(self.dialog.fiducial_table_widget.item(0, 2).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(1, 0).text()), places=5)
        self.assertAlmostEqual(10.0, float(self.dialog.fiducial_table_widget.item(1, 1).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(1, 2).text()), places=5)

        self.mock_instrument.positioning_stack.fkine([np.radians(90.0), 0], set_point=True)
        self.model_mock.return_value.instrument_controlled.emit(CommandID.ChangePositioningStack)

        self.assertAlmostEqual(-1.0, float(self.dialog.fiducial_table_widget.item(0, 0).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(0, 1).text()), places=5)
        self.assertAlmostEqual(2.0, float(self.dialog.fiducial_table_widget.item(0, 2).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(1, 0).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(1, 1).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.fiducial_table_widget.item(1, 2).text()), places=5)

        # Test matrix is at expected position
        self.mock_instrument.positioning_stack.fkine([0, 0], set_point=True)
        self.model_mock.return_value.instrument_controlled.emit(CommandID.MovePositioner)
        self.assertAlmostEqual(1.0, float(self.dialog.matrix_table_widget.item(0, 0).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(0, 1).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(0, 2).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(0, 3).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(1, 0).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(1, 3).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(2, 0).text()), places=5)
        self.assertAlmostEqual(50.0, float(self.dialog.matrix_table_widget.item(2, 3).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(3, 0).text()), places=5)
        self.assertAlmostEqual(1.0, float(self.dialog.matrix_table_widget.item(3, 3).text()), places=5)

        # Test matrix updates correctly
        self.mock_instrument.positioning_stack.fkine([np.radians(90.0), 10], set_point=True)
        self.model_mock.return_value.instrument_controlled.emit(CommandID.MovePositioner)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(0, 0).text()), places=5)
        self.assertAlmostEqual(-1.0, float(self.dialog.matrix_table_widget.item(0, 1).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(0, 2).text()), places=5)
        self.assertAlmostEqual(-10.0, float(self.dialog.matrix_table_widget.item(0, 3).text()), places=5)
        self.assertAlmostEqual(1.0, float(self.dialog.matrix_table_widget.item(1, 0).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(1, 3).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(2, 0).text()), places=5)
        self.assertAlmostEqual(50.0, float(self.dialog.matrix_table_widget.item(2, 3).text()), places=5)
        self.assertAlmostEqual(0.0, float(self.dialog.matrix_table_widget.item(3, 0).text()), places=5)
        self.assertAlmostEqual(1.0, float(self.dialog.matrix_table_widget.item(3, 3).text()), places=5)

        # Test export buttons call correct functions
        self.assertTrue(self.dialog.export_fiducials_action.isEnabled())
        self.dialog.export_fiducials_action.trigger()
        self.presenter.exportCurrentFiducials.assert_called_once()
        args = self.presenter.exportCurrentFiducials.call_args[0]
        np.testing.assert_equal(args[0], [0, 1])
        np.testing.assert_almost_equal(args[1], [[-11, 0, 2], [-10, 0, 0]], decimal=3)
        np.testing.assert_almost_equal(args[2], [[10, 90], [10, 90]], decimal=3)

        file_dialog_mock.getSaveFileName.return_value = ''
        self.dialog.export_matrix_action.trigger()
        file_dialog_mock.getSaveFileName.assert_called()
        save_txt.assert_not_called()
        file_dialog_mock.getSaveFileName.return_value = 'dummy'
        self.dialog.export_matrix_action.trigger()
        save_txt.assert_called()


class TestCurveEditor(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.view.showSaveDialog = mock.Mock()
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.themes = mock.create_autospec(ThemeManager)
        self.view.themes.theme_changed = TestSignal()
        self.view.themes.curve_face = QColor()
        self.view.themes.curve_line = QColor()
        self.view.themes.curve_label = QColor()
        self.view.themes.curve_plot = QColor()

        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.sample_changed = TestSignal()
        self.presenter = MainWindowPresenter(self.view)
        self.view.presenter = self.presenter
        data = np.zeros([3, 3, 3], np.uint8)
        data[1, :, :] = 2
        data[2, :, :] = 3
        volume = Volume(data, np.ones(3), np.ones(3))
        self.model_mock.return_value.sample = volume
        self.dialog = CurveEditor(self.view)

    def testPlotting(self):
        volume = self.dialog.parent.presenter.model.sample
        self.dialog.parent.presenter.model.sample = None
        self.dialog.parent.presenter.model.sample_changed.emit()
        self.assertFalse(self.dialog.canvas.isEnabled())
        self.assertFalse(self.dialog.group_box.isEnabled())
        self.assertFalse(self.dialog.accept_button.isEnabled())

        self.dialog.parent.presenter.model.sample = volume
        self.dialog.parent.presenter.model.sample_changed.emit()
        self.assertTrue(self.dialog.canvas.isEnabled())
        self.assertTrue(self.dialog.group_box.isEnabled())
        self.assertTrue(self.dialog.accept_button.isEnabled())

        np.testing.assert_array_almost_equal(self.dialog.inputs, [0., 255.], decimal=2)
        np.testing.assert_array_almost_equal(self.dialog.outputs, [0., 1.], decimal=2)

        dpi_scaling = getattr(self.dialog.canvas, '_dpi_ratio', 1.)
        x1, y1 = 90 * dpi_scaling, 90 * dpi_scaling
        x2, y2 = 200 * dpi_scaling, 200 * dpi_scaling

        event = MouseEvent('event', self.dialog.canvas, -1, -1, button=1)  # out of axes
        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 2)

        event = MouseEvent('event', self.dialog.canvas, x1, y1, button=2)  # wrong button
        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 2)

        event = MouseEvent('event', self.dialog.canvas, x1, y1, button=1)
        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 3)

        event = MouseEvent('event', self.dialog.canvas, x1, y1, button=1)  # No duplicate point
        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 3)
        self.assertEqual(self.dialog.last_pos, 1)

        event = MouseEvent('event', self.dialog.canvas, x2, y2, button=1)
        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 4)
        self.assertEqual(self.dialog.last_pos, 2)

        event = MouseEvent('event', self.dialog.canvas, x1, y1, button=2)  # wrong button
        self.dialog.canvasMouseMoveEvent(event)
        self.assertEqual(len(self.dialog.inputs), 4)
        self.assertEqual(self.dialog.last_pos, 2)
        self.dialog.canvasMouseReleaseEvent(event)
        self.assertIsNotNone(self.dialog.last_pos)

        event = MouseEvent('event', self.dialog.canvas, -1, 90, button=1)  # out of axes
        self.dialog.canvasMouseMoveEvent(event)
        self.assertEqual(len(self.dialog.inputs), 4)
        self.assertEqual(self.dialog.last_pos, 2)

        self.dialog.canvasMouseReleaseEvent(event)
        self.assertIsNone(self.dialog.last_pos)
        event = MouseEvent('event', self.dialog.canvas, x2, y2, button=1)
        self.dialog.canvasMousePressEvent(event)
        event = MouseEvent('event', self.dialog.canvas, x1, y1, button=1)
        self.dialog.canvasMouseMoveEvent(event)
        self.assertEqual(len(self.dialog.inputs), 3)
        self.assertEqual(self.dialog.last_pos, 1)

        self.dialog.last_pos = 0
        event = MouseEvent('event', self.dialog.canvas, x1, y1, button=1)
        self.dialog.canvasMouseMoveEvent(event)
        self.assertEqual(len(self.dialog.inputs), 2)
        self.assertEqual(self.dialog.last_pos, 0)

    def testOptions(self):
        dpi_scaling = getattr(self.dialog.canvas, '_dpi_ratio', 1.)
        x, y = 90 * dpi_scaling, 90 * dpi_scaling
        event = MouseEvent('event', self.dialog.canvas, x, y, button=1)
        self.assertEqual(len(self.dialog.inputs), 2)
        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 3)
        self.assertEqual(self.dialog.last_pos, 1)

        self.assertAlmostEqual(self.dialog.inputs[self.dialog.selected_index], 3.141, 3)
        self.assertAlmostEqual(self.dialog.outputs[self.dialog.selected_index], 0.031, 3)
        self.assertAlmostEqual(self.dialog.input_spinbox.value(), 3.141, 3)
        self.assertAlmostEqual(self.dialog.input_spinbox.minimum(), 0.0, 3)
        self.assertAlmostEqual(self.dialog.input_spinbox.maximum(), 255.0, 3)
        self.assertAlmostEqual(self.dialog.output_spinbox.value(), 0.031, 3)
        self.assertAlmostEqual(self.dialog.output_spinbox.minimum(), 0.0, 3)
        self.assertAlmostEqual(self.dialog.output_spinbox.maximum(), 1.0, 3)

        self.dialog.input_spinbox.setValue(3.5)
        self.dialog.output_spinbox.setValue(0.5)
        self.assertAlmostEqual(self.dialog.inputs[self.dialog.selected_index], 3.5, 3)
        self.assertAlmostEqual(self.dialog.outputs[self.dialog.selected_index], 0.5, 3)

        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 4)
        self.assertEqual(self.dialog.last_pos, 1)
        self.dialog.input_spinbox.setValue(3.6)
        self.dialog.output_spinbox.setValue(0.6)
        self.assertEqual(len(self.dialog.inputs), 3)
        self.assertEqual(self.dialog.last_pos, 1)

        event = MouseEvent('event', self.dialog.canvas, x, y, button=1)
        self.dialog.canvasMousePressEvent(event)
        self.assertEqual(len(self.dialog.inputs), 4)
        self.assertEqual(self.dialog.last_pos, 1)
        self.dialog.delete_button.click()
        np.testing.assert_array_almost_equal(self.dialog.inputs, [0., 3.6, 255.], decimal=2)
        np.testing.assert_array_almost_equal(self.dialog.outputs, [0., 0.6, 1.], decimal=2)
        self.dialog.reset_button.click()
        np.testing.assert_array_almost_equal(self.dialog.inputs, [0., 255.], decimal=2)
        np.testing.assert_array_almost_equal(self.dialog.outputs, [0., 1.], decimal=2)
        self.dialog.delete_button.click()
        np.testing.assert_array_almost_equal(self.dialog.inputs, [255.], decimal=2)
        np.testing.assert_array_almost_equal(self.dialog.outputs, [1.], decimal=2)
        self.dialog.delete_button.click()
        np.testing.assert_array_almost_equal(self.dialog.inputs, [255.], decimal=2)
        np.testing.assert_array_almost_equal(self.dialog.outputs, [1.], decimal=2)
        self.dialog.reset_button.click()
        np.testing.assert_array_almost_equal(self.dialog.inputs, [0., 255.], decimal=2)
        np.testing.assert_array_almost_equal(self.dialog.outputs, [0., 1.], decimal=2)

        volume = self.dialog.parent.presenter.model.sample
        self.assertIs(self.dialog.curve, volume.curve)
        self.presenter.changeVolumeCurve = mock.Mock()
        self.dialog.accept_button.click()
        self.assertIs(self.dialog.default_curve, volume.curve)
        self.presenter.changeVolumeCurve.assert_not_called()
        self.view.scenes.previewVolumeCurve.reset_mock()
        self.dialog.input_spinbox.setValue(3.5)
        self.dialog.output_spinbox.setValue(0.5)
        self.view.scenes.previewVolumeCurve.assert_called_with(self.dialog.curve)
        self.dialog.accept_button.click()
        self.presenter.changeVolumeCurve.assert_called()
        self.assertIs(self.presenter.changeVolumeCurve.call_args[0][0], self.dialog.curve)
        self.dialog.cancel_button.click()
        self.view.scenes.previewVolumeCurve.assert_called_with(self.dialog.default_curve)


class TestUpdater(unittest.TestCase):
    @mock.patch('sscanss.app.window.view.Worker', TestWorker)
    def setUp(self):
        self.view = TestView()
        self.view.themes = mock.create_autospec(ThemeManager)
        self.view.themes.anchor = QColor()

        self.settings = create_mock(self, 'sscanss.app.window.view.settings')
        self.logging = create_mock(self, 'sscanss.app.window.view.logging')
        self.dialog = Updater(self.view)
        self.dialog.show = mock.Mock()

    @mock.patch('sscanss.app.window.view.urllib.request.urlopen', autospec=True)
    def testDialog(self, urlopen_mock):
        self.settings.value.return_value = False
        self.dialog.check(True)
        urlopen_mock.assert_not_called()

        current_version = Version(2, 0, 0)
        with mock.patch('sscanss.app.window.view.__version__', current_version):
            self.settings.value.return_value = True
            urlopen_mock.return_value.__enter__.return_value.read.return_value = '{"tag_name":""}'
            self.dialog.check(True)
            self.dialog.show.assert_not_called()
            self.dialog.check()
            self.assertEqual(self.dialog.show.call_count, 1)
            urlopen_mock.return_value.__enter__.return_value.read.return_value = '{"tag_name":"v3.0.0"}'
            self.dialog.check()
            self.assertEqual(self.dialog.show.call_count, 2)
            self.dialog.check(True)
            self.assertEqual(self.dialog.show.call_count, 3)

        self.dialog.worker.side_effect = HTTPError('', 400, '', {}, None)
        self.dialog.check()
        self.assertEqual(self.logging.error.call_count, 1)
        self.assertEqual(self.dialog.show.call_count, 4)

        self.dialog.worker.side_effect = URLError('')
        self.dialog.check()
        self.assertEqual(self.logging.error.call_count, 2)
        self.assertEqual(self.dialog.show.call_count, 5)

        self.dialog.check(True)
        self.assertEqual(self.logging.error.call_count, 3)
        self.assertEqual(self.dialog.show.call_count, 5)

        self.dialog.worker.isRunning = mock.Mock(return_value=False)
        self.dialog.worker.terminate = mock.Mock()
        self.dialog.close()
        self.dialog.worker.terminate.assert_not_called()
        self.dialog.worker.isRunning.return_value = True
        self.dialog.close()
        self.dialog.worker.terminate.assert_called()

    def testValidation(self):
        current_version = Version(2, 0, 0)
        with mock.patch('sscanss.app.window.view.__version__', current_version):
            self.assertFalse(self.dialog.isNewVersions('2.0.0'))
            self.assertTrue(self.dialog.isNewVersions('2.2.0'))
            self.assertTrue(self.dialog.isNewVersions('2.0.1'))
            self.assertFalse(self.dialog.isNewVersions('3.0.0-alpha'))
            self.assertTrue(self.dialog.isNewVersions('3.0.0'))
            self.assertFalse(self.dialog.isNewVersions('1.2.3'))
            self.assertFalse(self.dialog.isNewVersions('1.2'))

        current_version = Version(2, 0, 0, 'alpha')
        with mock.patch('sscanss.app.window.view.__version__', current_version):
            self.assertTrue(self.dialog.isNewVersions('2.0.0'))


class TestSamplePropertiesDialog(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.sample_changed = TestSignal()
        self.model_mock.return_value.sample = None
        self.presenter = MainWindowPresenter(self.view)
        self.view.presenter = self.presenter
        self.dialog = SampleProperties(self.view)
        self.bytes_to_mb_factor = 1 / (1024**2)

    def testMeshSamplePropertiesDialog(self):
        mesh_test_cases = (
            {
                'case': 'mesh_test_case_1',
                'vertices': np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]),
                'indices': np.array([0, 1, 2, 3, 4, 5])
            },
            {
                'case': 'mesh_test_case_2',
                'vertices': np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                'indices': np.array([0, 1, 2])
            },
        )
        for test_case in mesh_test_cases:
            with self.subTest(test_case['case']):
                self._meshSamplePropertiesDialog(vertices=test_case['vertices'], indices=test_case['indices'])

    def _meshSamplePropertiesDialog(self, vertices, indices):
        self.mesh = Mesh(vertices, indices)
        self.model_mock.return_value.sample = self.mesh
        self.model_mock.return_value.sample_changed.emit()

        memory = self.mesh.vertices.nbytes * self.bytes_to_mb_factor
        self.assertEqual('Memory (MB)', self.dialog.sample_property_table.item(0, 0).text())
        self.assertEqual(f'{memory:.4f}', self.dialog.sample_property_table.item(0, 1).text())

        num_faces = self.mesh.indices.shape[0] // 3
        self.assertEqual('Faces', self.dialog.sample_property_table.item(1, 0).text())
        self.assertEqual(str(num_faces), self.dialog.sample_property_table.item(1, 1).text())

        num_vertices = self.mesh.vertices.shape[0]
        self.assertEqual('Vertices', self.dialog.sample_property_table.item(2, 0).text())
        self.assertEqual(str(num_vertices), self.dialog.sample_property_table.item(2, 1).text())

    def testVolumeSamplePropertiesDialog(self):
        volume_test_cases = (
            {
                'case': 'volume_test_case_1',
                'data': np.zeros([3, 3, 3], np.uint8),
                'voxel_size': np.array([1, 1, 1]),
                'centre': np.array([0, 0, 0])
            },
            {
                'case': 'volume_test_case_2',
                'data': np.ones([3, 3, 3], np.uint8),
                'voxel_size': np.array([2, 2, 2]),
                'centre': np.array([1, 1, 1])
            },
        )
        for test_case in volume_test_cases:
            with self.subTest(test_case['case']):
                self._volumeSamplePropertiesDialog(data=test_case['data'],
                                                   voxel_size=test_case['voxel_size'],
                                                   centre=test_case['centre'])

    def _volumeSamplePropertiesDialog(self, data, voxel_size, centre):
        self.volume = Volume(data, voxel_size, centre)
        self.model_mock.return_value.sample = self.volume
        self.model_mock.return_value.sample_changed.emit()

        memory = self.volume.data.nbytes * self.bytes_to_mb_factor
        self.assertEqual('Memory (MB)', self.dialog.sample_property_table.item(0, 0).text())
        self.assertEqual(f'{memory:.4f}', self.dialog.sample_property_table.item(0, 1).text())

        x, y, z = self.volume.data.shape
        dimension_str = f'{x} x {y} x {z}'
        self.assertEqual('Dimension', self.dialog.sample_property_table.item(1, 0).text())
        self.assertEqual(dimension_str, self.dialog.sample_property_table.item(1, 1).text())

        voxel_size = self.volume.voxel_size
        voxel_size_str = f'[x: {voxel_size[0]},  y: {voxel_size[1]},  z: {voxel_size[2]}]'
        self.assertEqual('Voxel size', self.dialog.sample_property_table.item(2, 0).text())
        self.assertEqual(voxel_size_str, self.dialog.sample_property_table.item(2, 1).text())

        min_intensity = np.min(self.volume.data)
        self.assertEqual('Minimum Intensity', self.dialog.sample_property_table.item(3, 0).text())
        self.assertEqual(str(min_intensity), self.dialog.sample_property_table.item(3, 1).text())

        max_intensity = np.max(self.volume.data)
        self.assertEqual('Maximum Intensity', self.dialog.sample_property_table.item(4, 0).text())
        self.assertEqual(str(max_intensity), self.dialog.sample_property_table.item(4, 1).text())

    def testNoneSamplePropertiesDialog(self):
        self.assertEqual('Memory (MB)', self.dialog.sample_property_table.item(0, 0).text())
        self.assertEqual('0', self.dialog.sample_property_table.item(0, 1).text())


class TestInsertPrimitiveDialog(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view = TestView()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = [dummy]
        self.model_mock.return_value.sample_changed = TestSignal()
        self.model_mock.return_value.sample = None
        self.presenter = MainWindowPresenter(self.view)
        self.presenter.addPrimitive = mock.Mock()
        self.view.scenes = mock.create_autospec(SceneManager)
        self.view.presenter = self.presenter

    def testInsertPrimitiveDialog(self):
        primitive_test_cases = (
            {
                'case': 'Cone',
                'primitive': Primitives.Cone,
                'parameter_count': 2,
                'mesh_args_default': {
                    'radius': 100.000,
                    'height': 200.000
                },
                'mesh_args_custom': {
                    'radius': 150.000,
                    'height': 760.000
                }
            },
            {
                'case': 'Cuboid',
                'primitive': Primitives.Cuboid,
                'parameter_count': 3,
                'mesh_args_default': {
                    'width': 50.000,
                    'height': 100.000,
                    'depth': 200.000
                },
                'mesh_args_custom': {
                    'width': 900.000,
                    'height': 330.000,
                    'depth': 700.000
                }
            },
            {
                'case': 'Cylinder',
                'primitive': Primitives.Cylinder,
                'parameter_count': 2,
                'mesh_args_default': {
                    'radius': 100.000,
                    'height': 200.000
                },
                'mesh_args_custom': {
                    'radius': 450.000,
                    'height': 50.000
                }
            },
            {
                'case': 'Sphere',
                'primitive': Primitives.Sphere,
                'parameter_count': 1,
                'mesh_args_default': {
                    'radius': 100.000
                },
                'mesh_args_custom': {
                    'radius': 600.000
                }
            },
            {
                'case': 'Tube',
                'primitive': Primitives.Tube,
                'parameter_count': 3,
                'mesh_args_default': {
                    'outer_radius': 100.000,
                    'inner_radius': 50.000,
                    'height': 200.000
                },
                'mesh_args_custom': {
                    'outer_radius': 300.000,
                    'inner_radius': 150.000,
                    'height': 1000.000
                }
            },
        )

        for test_case in primitive_test_cases:
            with self.subTest(test_case['case']):
                self.dialog = InsertPrimitiveDialog(test_case['primitive'], self.view)
                self.assertTrue(self.dialog.create_primitive_button.isEnabled())

                # Testing primitive creation with default values
                self.dialog.create_primitive_button.click()
                self.view.presenter.addPrimitive.assert_called_with(test_case['primitive'],
                                                                    test_case['mesh_args_default'])

                # Testing primitive creation with custom values
                for parameter, value in test_case['mesh_args_custom'].items():
                    self.dialog.textboxes[parameter].value = value
                self.dialog.create_primitive_button.click()
                self.view.presenter.addPrimitive.assert_called_with(test_case['primitive'],
                                                                    test_case['mesh_args_custom'])

                # Testing with zero and negative values
                for ix in range(test_case['parameter_count']):
                    control = self.dialog.form_group.form_controls[ix]
                    value = control.value
                    control.value = 0
                    self.assertFalse(self.dialog.create_primitive_button.isEnabled())
                    control.value = -50
                    self.assertFalse(self.dialog.create_primitive_button.isEnabled())
                    control.value = value

                # Testing with inner radius greater than outer radius for tube
                if test_case['primitive'] == Primitives.Tube:
                    control_outer = self.dialog.form_group.form_controls[0]
                    control_outer.value = 50
                    control_inner = self.dialog.form_group.form_controls[1]
                    control_inner.value = 100
                    self.assertFalse(self.dialog.create_primitive_button.isEnabled())


class TestProgressDialog(unittest.TestCase):
    @mock.patch("sscanss.app.dialogs.misc.ProgressReport", autospec=True)
    def setUp(self, report_mock):
        self.view = TestView()
        report_mock.return_value.progress_updated = TestSignal()
        report_mock.return_value.message = 'Determinate'
        self.report_mock = report_mock
        self.dialog = ProgressDialog(self.view)
        self.dialog.show = mock.Mock()

    def testProgress(self):
        self.dialog.show.assert_not_called()
        self.dialog.showMessage('Indeterminate')
        self.dialog.show.assert_called()
        self.assertEqual(self.dialog.message.text(), 'Indeterminate')
        self.assertFalse(self.dialog.determinate)
        self.assertEqual(self.dialog.percent_label.text(), '')
        self.report_mock.return_value.progress_updated.emit(0.5)
        self.assertTrue(self.dialog.determinate)
        self.assertEqual(self.dialog.percent_label.text(), '50%')
        self.assertEqual(self.dialog.message.text(), 'Determinate')
        self.report_mock.return_value.progress_updated.emit(0.75)
        self.assertEqual(self.dialog.percent_label.text(), '75%')
        self.assertEqual(self.dialog.progress_bar.value(), 75)
        self.dialog.setProgress(1.0)
        self.assertEqual(self.dialog.percent_label.text(), '100%')
        self.assertEqual(self.dialog.progress_bar.value(), 100)


class TestCustomIntValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = CustomIntValidator(5, 100)

    def testFixup(self):
        clamped_lower = '5'
        clamped_upper = '100'
        self.assertEqual(self.validator.fixup(15), '15')
        self.assertEqual(self.validator.fixup(1), clamped_lower)
        self.assertEqual(self.validator.fixup(150), clamped_upper)


class TestSliderTextInput(unittest.TestCase):
    def setUp(self) -> None:
        self.view = TestView()
        self.widget = SliderTextInput(self.view, 5)

    def testUpdateSlider(self):
        # Test that slider initialised correctly
        self.assertEqual(self.widget.slider.value(), 5)

        # Using an acceptable input to the line edit
        acceptable_input = '15'
        self.widget.slider_value.clear()
        self.widget.slider_value.insert(acceptable_input)
        self.widget.updateSlider()
        self.assertEqual(self.widget.slider.value(), 15)

        # Using an unacceptable numeric input to the line edit
        unacceptable_input = '2'
        self.widget.slider_value.clear()
        self.widget.slider_value.insert(unacceptable_input)
        self.widget.updateSlider()
        self.assertEqual(self.widget.slider.value(), 15)

        # Using an unacceptable non-numeric input to the line edit
        unacceptable_input = ''
        self.widget.slider_value.clear()
        self.widget.slider_value.insert(unacceptable_input)
        self.widget.updateSlider()
        self.assertEqual(self.widget.slider.value(), 15)


if __name__ == "__main__":
    unittest.main()
