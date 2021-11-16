import unittest
import unittest.mock as mock
import numpy as np
from PyQt5.QtWidgets import QUndoStack
from sscanss.app.window.presenter import MainWindowPresenter, MessageReplyType
import sscanss.app.window.view as view
from sscanss.core.instrument.instrument import PositioningStack
from sscanss.core.instrument.robotics import Link, SerialManipulator
from sscanss.core.util import PointType, TransformType, Primitives, InsertSampleOptions


class TestMainWindowPresenter(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(view.MainWindow)
        self.view_mock.undo_stack = QUndoStack()
        self.view_mock.undo_stack.resetClean()
        self.model_mock = model_mock
        self.model_mock.return_value.project_data = {}
        self.model_mock.return_value.instruments = ["dummy"]
        self.presenter = MainWindowPresenter(self.view_mock)
        self.notify = mock.Mock()
        self.presenter.notifyError = self.notify

        self.test_project_data = {"name": "Test Project", "instrument": "IMAT"}
        self.test_filename_1 = "C:/temp/file.h5"
        self.test_filename_2 = "C:/temp/file_2.h5"

    @mock.patch("sscanss.app.window.presenter.logging", autospec=True)
    @mock.patch("sscanss.app.window.presenter.toggleActionInGroup", autospec=True)
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def testCreateProject(self, model_mock, toggle_mock, log_mock):
        # model_mock is used instead of self.model_mock because a new presenter is created
        model_mock.return_value.instruments = []
        self.assertRaises(FileNotFoundError, MainWindowPresenter, self.view_mock)
        model_mock.return_value.instruments = ["Dummy"]
        presenter = MainWindowPresenter(self.view_mock)
        message, exception = "error!!!!", ValueError()
        presenter.notifyError(message, exception)
        log_mock.error.assert_called_with(message, exc_info=exception)
        self.view_mock.showMessage.assert_called_with(message)

        self.view_mock.scenes = mock.Mock()
        self.view_mock.docks = mock.Mock()
        self.model_mock.return_value.simulation = None
        self.presenter.createProject("demo", "EMAP")
        self.presenter.model.createProjectData.assert_called_with("demo", "EMAP")

        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.instrument = None
        self.view_mock.change_instrument_action_group = None

        self.presenter.projectCreationError(OSError, ("demo", "EMAP"))
        self.assertIsNone(self.presenter.model.project_data)
        toggle_mock.assert_not_called()
        self.notify.assert_called_once()

        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.instrument = mock.Mock()
        self.presenter.projectCreationError(OSError(), ("demo", "EMAP"))
        toggle_mock.assert_called_once()
        self.assertEqual(self.notify.call_count, 2)

    def testSaveProjectWithDefaults(self):
        self.view_mock.recent_projects = []
        self.view_mock.undo_stack.setClean()

        # When there are no unsaved changes save will not be called
        self.model_mock.reset_mock()
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.save_path = self.test_filename_1
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_not_called()

        # When there are unsaved changes save will be called
        self.model_mock.reset_mock()
        self.view_mock.undo_stack.resetClean()
        self.model_mock.return_value.save_path = self.test_filename_1
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_called_with(self.test_filename_1)
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_1])

        # When save_path is blank, filename is acquired from dialog
        self.model_mock.reset_mock()
        self.view_mock.undo_stack.resetClean()
        self.model_mock.return_value.save_path = ""
        self.view_mock.showSaveDialog.return_value = self.test_filename_2
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_called_with(self.test_filename_2)
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_2, self.test_filename_1])

        # if dialog return empty filename (user cancels save), save will not be called
        self.model_mock.reset_mock()
        self.view_mock.undo_stack.resetClean()
        self.model_mock.return_value.save_path = ""
        self.view_mock.showSaveDialog.return_value = ""
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_not_called()

    def testSaveProjectWithSaveAs(self):
        self.view_mock.recent_projects = []
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.save_path = self.test_filename_1

        # Save_as opens dialog even though there are no unsaved changes
        self.view_mock.showSaveDialog.return_value = self.test_filename_2
        self.presenter.saveProject(save_as=True)
        self.model_mock.return_value.saveProjectData.assert_called_with(self.test_filename_2)
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_2])

        self.model_mock.return_value.saveProjectData.side_effect = OSError
        self.presenter.saveProject(save_as=True)
        self.notify.assert_called_once()

    def testOpenProject(self):
        self.view_mock.recent_projects = []
        self.view_mock.undo_stack.setClean()
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.simulation = None
        self.presenter.updateView = mock.create_autospec(self.presenter.updateView)
        self.model_mock.return_value.save_path = ""

        # When non-empty filename is provided, dialog should not be called, file should be loaded,
        # recent list updated, and view name changed
        self.presenter.openProject(self.test_filename_2)
        self.view_mock.showOpenDialog.assert_not_called()
        self.model_mock.return_value.loadProjectData.assert_called_with(self.test_filename_2)
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_2])

        self.view_mock.docks = mock.Mock()
        self.presenter.projectOpenError(ValueError(), (self.test_filename_2, ))
        self.notify.assert_called_once()
        self.presenter.projectOpenError(KeyError(), (self.test_filename_2, ))
        self.assertEqual(self.notify.call_count, 2)
        self.presenter.projectOpenError(OSError(), (self.test_filename_2, ))
        self.assertEqual(self.notify.call_count, 3)
        self.presenter.projectOpenError(TypeError(), (self.test_filename_2, ))
        self.assertEqual(self.notify.call_count, 4)

    def testUpdateRecentProjects(self):
        self.presenter.recent_list_size = 10

        self.view_mock.recent_projects = []
        self.presenter.updateRecentProjects("Hello World")
        self.assertEqual(self.view_mock.recent_projects, ["Hello World"])

        # Check new values are always placed in front
        self.view_mock.recent_projects = [1, 2, 3]
        self.presenter.updateRecentProjects(4)
        self.assertEqual(self.view_mock.recent_projects, [4, 1, 2, 3])

        # When max size is exceeded the last entry is removed
        self.view_mock.recent_projects = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.presenter.updateRecentProjects(10)
        self.assertEqual(self.view_mock.recent_projects, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        # When a value already exist in the list, it is push to the front
        self.view_mock.recent_projects = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.presenter.updateRecentProjects(3)
        self.assertEqual(self.view_mock.recent_projects, [3, 9, 8, 7, 6, 5, 4, 2, 1, 0])

    def testConfirmSave(self):
        # confirmSave should return True when project_data is None
        self.model_mock.return_value.project_data = None
        self.assertTrue(self.presenter.confirmSave())
        self.view_mock.showSaveDiscardMessage.assert_not_called()

        # confirmSave should return True when there are no unsaved changes
        # and the save-discard message should not be called
        self.model_mock.return_value.project_data = self.test_project_data
        self.view_mock.undo_stack.setClean()
        self.assertTrue(self.presenter.confirmSave())
        self.view_mock.showSaveDiscardMessage.assert_not_called()

        # confirmSave should return False when user selects cancel on
        # the save-discard message box
        self.view_mock.undo_stack.resetClean()
        self.view_mock.showSaveDiscardMessage.return_value = MessageReplyType.Cancel
        self.assertFalse(self.presenter.confirmSave())

        # confirmSave should return True when user selects discard on
        # the save-discard message box
        self.view_mock.showSaveDiscardMessage.return_value = MessageReplyType.Discard
        self.assertTrue(self.presenter.confirmSave())

        # confirmSave should call save (if save path exist) then return True
        # when user selects save on the save-discard message box
        self.model_mock.return_value.save_path = self.test_filename_1
        self.presenter.saveProject = mock.create_autospec(self.presenter.saveProject)
        self.view_mock.showSaveDiscardMessage.return_value = MessageReplyType.Save
        self.assertTrue(self.presenter.confirmSave())
        self.presenter.saveProject.assert_called_with()

        # confirmSave should call save_as (if save_path does not exist)
        self.model_mock.return_value.save_path = ""
        self.presenter.confirmSave()
        self.presenter.saveProject.assert_called_with(save_as=True)

    def testConfirmClearStack(self):
        self.view_mock.undo_stack = mock.Mock()
        self.view_mock.undo_stack.count.return_value = 0
        self.assertTrue(self.presenter.confirmClearStack())

        self.view_mock.undo_stack.count.return_value = 1
        self.view_mock.showSelectChoiceMessage.return_value = "Proceed"
        self.assertTrue(self.presenter.confirmClearStack())

        self.view_mock.showSelectChoiceMessage.return_value = "Cancel"
        self.assertFalse(self.presenter.confirmClearStack())

    def testConfirmAddSampleOption(self):
        self.model_mock.return_value.sample = {}
        self.view_mock.showSelectChoiceMessage.return_value = "Combine"
        self.assertEqual(self.presenter.confirmInsertSampleOption(), InsertSampleOptions.Replace)

        self.model_mock.return_value.sample = {"": None}
        self.view_mock.showSelectChoiceMessage.return_value = "Combine"
        self.assertEqual(self.presenter.confirmInsertSampleOption(), InsertSampleOptions.Combine)

        self.view_mock.showSelectChoiceMessage.return_value = "Replace"
        self.assertEqual(self.presenter.confirmInsertSampleOption(), InsertSampleOptions.Replace)

        self.view_mock.showSelectChoiceMessage.return_value = "Cancel"
        self.assertIsNone(self.presenter.confirmInsertSampleOption())

    @mock.patch("sscanss.app.window.presenter.toggleActionInGroup", autospec=True)
    @mock.patch("sscanss.app.window.presenter.Worker", autospec=True)
    def testChangeInstrument(self, worker_mock, toggle_mock):
        self.presenter.worker = mock.Mock()
        self.view_mock.progress_dialog = mock.Mock()

        self.model_mock.return_value.instrument.name = "default"
        self.model_mock.checkInstrumentVersion.return_value = "1.0"
        self.presenter.changeInstrument("default")
        worker_mock.callFromWorker.assert_not_called()

        self.presenter.confirmClearStack = mock.Mock(return_value=False)
        self.view_mock.change_instrument_action_group = None
        self.presenter.changeInstrument("non_default")
        toggle_mock.assert_called()
        worker_mock.callFromWorker.assert_not_called()

        self.presenter.confirmClearStack.return_value = True
        self.presenter.changeInstrument("non_default")
        worker_mock.callFromWorker.assert_called_once()

        self.model_mock.return_value.simulation = None
        self.presenter._changeInstrumentHelper("non_default")
        self.presenter.model.changeInstrument.assert_called_once_with("non_default")

    def testPointImportAndExport(self):
        undo_stack = mock.Mock()
        self.view_mock.undo_stack.push = undo_stack

        self.model_mock.return_value.sample = {}
        self.presenter.importPoints(PointType.Fiducial)
        self.view_mock.showMessage.assert_called_once()
        undo_stack.assert_not_called()

        self.model_mock.return_value.sample = {"demo": None}
        self.view_mock.showOpenDialog.return_value = ""
        self.presenter.importPoints(PointType.Fiducial)
        undo_stack.assert_not_called()

        self.view_mock.showOpenDialog.return_value = "demo.txt"
        self.presenter.importPoints(PointType.Fiducial)
        undo_stack.assert_called_once()

        self.model_mock.return_value.measurement_points = np.array([])
        self.presenter.exportPoints(PointType.Measurement)
        self.assertEqual(self.view_mock.showMessage.call_count, 2)

        self.model_mock.return_value.measurement_points = np.array([1])
        self.view_mock.showSaveDialog.return_value = ""
        self.presenter.exportPoints(PointType.Measurement)
        self.presenter.model.savePoints.assert_not_called()

        self.view_mock.showSaveDialog.return_value = "demo.txt"
        self.presenter.exportPoints(PointType.Measurement)
        self.presenter.model.savePoints.assert_called_once()

        self.presenter.model.savePoints.side_effect = OSError
        self.presenter.exportPoints(PointType.Measurement)
        self.assertEqual(self.presenter.model.savePoints.call_count, 2)
        self.notify.assert_called_once()

    def testSampleImportAndExport(self):
        undo_stack = mock.Mock()
        self.view_mock.undo_stack.push = undo_stack

        self.view_mock.showOpenDialog.return_value = ""
        self.presenter.importSample()
        undo_stack.assert_not_called()

        self.view_mock.showOpenDialog.return_value = "demo"
        self.view_mock.showSelectChoiceMessage.return_value = "Cancel"
        self.presenter.importSample()
        undo_stack.assert_not_called()

        self.view_mock.showSelectChoiceMessage.return_value = "Combine"
        self.presenter.importSample()
        undo_stack.assert_called_once()

        self.model_mock.return_value.sample = {}
        self.presenter.exportSample()
        self.view_mock.showMessage.assert_called_once()

        self.model_mock.return_value.sample = {"": None}
        self.presenter.exportSample()
        self.presenter.model.saveSample.assert_not_called()

        self.view_mock.showSampleExport.return_value = "demo"
        self.model_mock.return_value.sample = {"1": None, "2": None}
        self.view_mock.showSaveDialog.return_value = ""
        self.presenter.exportSample()
        self.presenter.model.saveSample.assert_not_called()

        self.view_mock.showSaveDialog.return_value = "demo.stl"
        self.presenter.exportSample()
        self.presenter.model.saveSample.assert_called()

        self.presenter.model.saveSample.side_effect = OSError
        self.presenter.exportSample()
        self.assertEqual(self.presenter.model.saveSample.call_count, 2)
        self.notify.assert_called_once()

    def testVectorImportAndExport(self):
        undo_stack = mock.Mock()
        self.view_mock.undo_stack.push = undo_stack

        self.model_mock.return_value.sample = {}
        self.presenter.importVectors()
        self.view_mock.showMessage.assert_called_once()
        undo_stack.assert_not_called()

        self.model_mock.return_value.sample = {"demo": None}
        self.model_mock.return_value.measurement_points = np.array([])
        self.presenter.importVectors()
        self.assertEqual(self.view_mock.showMessage.call_count, 2)
        undo_stack.assert_not_called()

        self.model_mock.return_value.measurement_points = np.array([1])
        self.view_mock.showOpenDialog.return_value = ""
        self.presenter.importVectors()
        undo_stack.assert_not_called()

        self.view_mock.showOpenDialog.return_value = "demo.txt"
        self.presenter.importVectors()
        undo_stack.assert_called_once()

        self.model_mock.return_value.measurement_vectors = np.array([])
        self.presenter.exportVectors()
        self.assertEqual(self.view_mock.showMessage.call_count, 3)

        self.model_mock.return_value.measurement_vectors = np.array([1])
        self.view_mock.showSaveDialog.return_value = ""
        self.presenter.exportVectors()
        self.presenter.model.saveVectors.assert_not_called()

        self.view_mock.showSaveDialog.return_value = "demo.txt"
        self.presenter.exportVectors()
        self.presenter.model.saveVectors.assert_called_once()

        self.presenter.model.saveVectors.side_effect = OSError
        self.presenter.exportVectors()
        self.assertEqual(self.presenter.model.saveVectors.call_count, 2)
        self.notify.assert_called_once()

    @mock.patch("sscanss.app.window.presenter.read_trans_matrix", autospec=True)
    @mock.patch("sscanss.app.window.presenter.check_rotation", autospec=True)
    def testImportTransformMatrix(self, check_rotation, read_trans_matrix):
        self.view_mock.showOpenDialog.return_value = ""
        self.assertIsNone(self.presenter.importTransformMatrix())

        filename = "demo.txt"
        data = [1, 2]
        read_trans_matrix.return_value = data
        check_rotation.return_value = False
        self.view_mock.showOpenDialog.return_value = filename
        self.assertIsNone(self.presenter.importTransformMatrix())
        read_trans_matrix.assert_called_with(filename)

        check_rotation.return_value = True
        self.assertListEqual(self.presenter.importTransformMatrix(), data)

        read_trans_matrix.side_effect = OSError
        self.assertIsNone(self.presenter.importTransformMatrix())
        self.notify.assert_called_once()

        read_trans_matrix.side_effect = ValueError
        self.assertIsNone(self.presenter.importTransformMatrix())
        self.assertEqual(self.notify.call_count, 2)

    @mock.patch("sscanss.app.window.presenter.np.savetxt", autospec=True)
    def testExportAlignmentMatrix(self, savetxt):
        self.model_mock.return_value.alignment = None
        self.presenter.exportAlignmentMatrix()
        self.view_mock.showMessage.assert_called_once()

        self.model_mock.return_value.alignment = np.array([1])
        self.view_mock.showSaveDialog.return_value = ""
        self.presenter.exportAlignmentMatrix()
        savetxt.assert_not_called()

        self.view_mock.showSaveDialog.return_value = "demo.txt"
        self.presenter.exportAlignmentMatrix()
        savetxt.assert_called()

        savetxt.side_effect = OSError
        self.presenter.exportAlignmentMatrix()
        self.notify.assert_called_once()

    @mock.patch("sscanss.app.window.presenter.open")
    def testExportScript(self, open_func):
        script_renderer = mock.Mock(return_value="")
        self.model_mock.return_value.save_path = ""
        self.view_mock.showSaveDialog.return_value = ""
        self.assertFalse(self.presenter.exportScript(script_renderer))

        self.view_mock.showSaveDialog.return_value = "script.txt"
        self.assertTrue(self.presenter.exportScript(script_renderer))

        open_func.side_effect = OSError
        self.assertFalse(self.presenter.exportScript(script_renderer))
        self.notify.assert_called_once()

    @mock.patch("sscanss.app.window.presenter.settings", autospec=True)
    def testSimulationRunAndStop(self, setting_mock):
        self.view_mock.docks = mock.Mock()
        simulation = mock.Mock()
        self.model_mock.return_value.simulation = simulation

        self.model_mock.return_value.alignment = None
        self.presenter.runSimulation()
        self.view_mock.showMessage.assert_called_once()

        self.model_mock.return_value.alignment = np.array([1])
        self.model_mock.return_value.measurement_points = np.array([])
        self.presenter.runSimulation()
        self.assertEqual(self.view_mock.showMessage.call_count, 2)

        self.model_mock.return_value.alignment = np.array([1])
        self.model_mock.return_value.measurement_points = np.rec.array(
            [([1.0, 2.0, 3.0], False), ([4.0, 5.0, 6.0], False), ([7.0, 8.0, 9.0], False)],
            dtype=[("point", "f4", 3), ("enabled", "?")],
        )
        self.model_mock.return_value.measurement_vectors = np.zeros((3, 6, 1))
        self.presenter.runSimulation()
        self.assertEqual(self.view_mock.showMessage.call_count, 3)

        self.model_mock.return_value.measurement_points.enabled = [True, True, True]
        setting_mock.value.return_value = True
        self.presenter.runSimulation()
        self.assertEqual(self.view_mock.showMessage.call_count, 4)

        setting_mock.value.return_value = False
        simulation.isRunning.return_value = True
        self.presenter.runSimulation()

        simulation.isRunning.return_value = False
        self.view_mock.compute_path_length_action = mock.Mock()
        self.view_mock.check_collision_action = mock.Mock()
        self.view_mock.show_sim_graphics_action = mock.Mock()
        self.view_mock.check_limits_action = mock.Mock()
        self.presenter.runSimulation()
        self.presenter.model.createSimulation.assert_called_once()
        self.presenter.model.simulation.start.assert_called_once()

        simulation.isRunning.return_value = True
        self.presenter.resetSimulation()
        simulation.abort.assert_called_once()
        self.assertIsNone(self.model_mock.return_value.simulation)

        self.presenter.stopSimulation()
        simulation.abort.assert_called_once()

    @mock.patch("sscanss.app.window.presenter.read_fpos")
    def testAlignSample(self, read_fpos):
        undo_stack = mock.Mock()
        self.view_mock.undo_stack.push = undo_stack
        self.view_mock.scenes = mock.Mock()

        pose = [0.0] * 6
        self.model_mock.return_value.sample = {}
        self.presenter.alignSampleWithPose(pose)
        undo_stack.assert_not_called()
        self.assertEqual(self.view_mock.showMessage.call_count, 1)

        self.model_mock.return_value.sample = {}
        self.presenter.alignSampleWithMatrix()
        undo_stack.assert_not_called()
        self.assertEqual(self.view_mock.showMessage.call_count, 2)

        self.model_mock.return_value.sample = {}
        self.presenter.alignSampleWithFiducialPoints()
        undo_stack.assert_not_called()
        self.assertEqual(self.view_mock.showMessage.call_count, 3)

        self.model_mock.return_value.sample = {"demo": None}
        self.presenter.alignSampleWithPose(pose)
        undo_stack.assert_called_once()

        self.presenter.importTransformMatrix = mock.Mock(return_value=None)
        self.presenter.alignSampleWithMatrix()
        undo_stack.assert_called_once()

        self.presenter.importTransformMatrix.return_value = [1]
        self.presenter.alignSampleWithMatrix()
        self.assertEqual(undo_stack.call_count, 2)

        self.model_mock.return_value.fiducials = np.rec.array([([1.0, 2.0, 3.0], False), ([4.0, 5.0, 6.0], False)],
                                                              dtype=[("points", "f4", 3), ("enabled", "?")])
        self.presenter.alignSampleWithFiducialPoints()
        self.assertEqual(self.view_mock.showMessage.call_count, 4)

        self.model_mock.return_value.fiducials = np.rec.array(
            [([1.0, 2.0, 3.0], False), ([4.0, 5.0, 6.0], False), ([7.0, 8.0, 9.0], False)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )
        self.presenter.alignSampleWithFiducialPoints()
        self.assertEqual(self.view_mock.showMessage.call_count, 5)

        self.model_mock.return_value.fiducials.enabled = [True, True, True]
        self.view_mock.showOpenDialog.return_value = ""
        self.presenter.alignSampleWithFiducialPoints()

        self.view_mock.showOpenDialog.return_value = "demo.txt"
        read_fpos.side_effect = OSError
        self.presenter.alignSampleWithFiducialPoints()
        self.notify.assert_called_once()

        read_fpos.side_effect = ValueError
        self.presenter.alignSampleWithFiducialPoints()
        self.assertEqual(self.notify.call_count, 2)

        read_fpos.return_value = (np.array([1]), np.array([1]), np.array([]))
        read_fpos.side_effect = None
        self.presenter.alignSampleWithFiducialPoints()
        self.assertEqual(self.view_mock.showMessage.call_count, 6)

        read_fpos.return_value = (np.array([1, 2, 3]), np.array([1]), np.array([]))
        self.presenter.alignSampleWithFiducialPoints()
        self.assertEqual(self.view_mock.showMessage.call_count, 7)

        read_fpos.return_value = (np.array([-1, 2, 3]), np.array([1]), np.array([]))
        self.presenter.alignSampleWithFiducialPoints()
        self.assertEqual(self.view_mock.showMessage.call_count, 8)

        q1 = Link("", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        s = PositioningStack("", SerialManipulator("", [q1, q2]))
        self.model_mock.return_value.instrument.positioning_stack = s
        read_fpos.return_value = (
            np.array([0, 1, 2]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            np.array([[1], [2], [3]]),
        )
        self.presenter.alignSampleWithFiducialPoints()
        self.assertEqual(self.view_mock.showMessage.call_count, 9)

        self.view_mock.alignment_error = mock.Mock()
        read_fpos.return_value = (
            np.array([0, 1, 2]),
            np.array([[11.0, 2.0, -7.0], [17.0, 8.0, -1.0], [14.0, 5.0, -4.0]]),
            np.array([[10, -10], [10, -10], [10, -10]]),
        )
        self.presenter.alignSampleWithFiducialPoints()
        self.view_mock.showAlignmentError.assert_called_once()
        self.assertListEqual(self.view_mock.showAlignmentError.call_args[0][5].tolist(), [0, 2, 1])

    @mock.patch("sscanss.app.window.presenter.np.savetxt", autospec=True)
    def testExportBaseMatrix(self, savetxt):
        matrix = np.array([1])
        self.model_mock.return_value.save_path = ""
        self.view_mock.showSaveDialog.return_value = ""
        self.presenter.exportBaseMatrix(matrix)
        savetxt.assert_not_called()

        self.model_mock.return_value.save_path = "C:/sscanss/"
        self.view_mock.showSaveDialog.return_value = "demo.txt"
        self.presenter.exportBaseMatrix(matrix)
        savetxt.assert_called()

        savetxt.side_effect = OSError
        self.presenter.exportBaseMatrix(matrix)
        self.notify.assert_called_once()

    @mock.patch("sscanss.app.window.presenter.read_robot_world_calibration_file")
    def testComputePositionerBase(self, read_robot_world_calibration_file):
        self.view_mock.scenes = mock.Mock()
        self.model_mock.return_value.fiducials = np.array([])

        q1 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 100, 0)
        q2 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, 0, np.pi, 0)
        q3 = Link("", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Revolute, 0, np.pi, 0)
        positioner = SerialManipulator("", [q1])
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 1)

        positioner = SerialManipulator("", [q1, q2])
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 2)

        positioner = SerialManipulator("", [q2, q3])
        self.model_mock.return_value.fiducials = np.array([])
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 3)

        self.model_mock.return_value.fiducials = np.rec.array(
            [([1.0, 2.0, 3.0], False), ([4.0, 5.0, 6.0], False), ([9.0, 8.0, 7.0], False)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )

        self.view_mock.showOpenDialog.return_value = ""
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        read_robot_world_calibration_file.assert_not_called()

        self.view_mock.showOpenDialog.return_value = "demo.calib"
        read_robot_world_calibration_file.side_effect = OSError
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.notify.assert_called_once()

        read_robot_world_calibration_file.side_effect = ValueError
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.notify.call_count, 2)

        results = [np.array([1]), np.array([1]), np.array([]), np.array([])]
        read_robot_world_calibration_file.return_value = results
        read_robot_world_calibration_file.side_effect = None
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 4)

        results[0] = np.array([1, 2, 3])
        read_robot_world_calibration_file.return_value = results
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 5)

        results[0], results[1] = np.array([0, 1, 2]), np.array([-1])
        read_robot_world_calibration_file.return_value = results
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 6)

        results[1] = np.array([3])
        read_robot_world_calibration_file.return_value = results
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 7)

        results[1] = np.array([0, 1, 2])
        read_robot_world_calibration_file.return_value = results
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 8)

        self.model_mock.return_value.fiducials.enabled = [True, True, True]
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 9)

        results[0] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        results[1] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        results[2] = np.zeros((9, 3))
        results[3] = np.zeros((9, 1))
        read_robot_world_calibration_file.return_value = results
        self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.view_mock.showMessage.call_count, 10)

        results[3] = np.zeros((9, 2))
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in double_scalars")
            self.assertIsNone(self.presenter.computePositionerBase(positioner))
        self.assertEqual(self.notify.call_count, 3)

        results[2][:3, :] = np.array([[-3.66666651, -3.0, -2.33333349], [-0.66666651, 0.0, 0.66666651],
                                      [4.33333349, 3.0, 1.66666651]])
        results[2][3:6, :] = np.array([
            [-2.33333349e00, -3.66666651e00, -3.00000000e00],
            [6.66666508e-01, -6.66666508e-01, -2.96059403e-16],
            [1.66666651e00, 4.33333349e00, 3.00000000e00],
        ])
        results[2][6:, :] = np.array([
            [-2.75430621, -3.15322154, -3.21867669],
            [-0.54847643, -0.44227586, 0.6264616],
            [3.30278304, 3.59549772, 2.59221465],
        ])
        results[3][3:6, :] = np.ones((3, 2)) * 90
        results[3][6:, :] = np.ones((3, 2)) * 20

        base = self.presenter.computePositionerBase(positioner)
        np.testing.assert_array_almost_equal(base, np.identity(4), decimal=5)

    def testOtherCommands(self):
        undo_stack = mock.Mock()
        self.view_mock.undo_stack.push = undo_stack
        self.view_mock.docks = mock.Mock()

        self.presenter.removeVectorAlignment(1)
        self.assertEqual(undo_stack.call_count, 1)

        self.model_mock.return_value.measurement_vectors = np.array([[[-1.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]],
                                                                     [[0.0], [1.0], [0.0]], [[0.0], [0.0], [0.0]]])
        self.presenter.removeVectors([1, 3], 0, 0)
        self.assertEqual(undo_stack.call_count, 1)

        self.presenter.removeVectors([0, 2, 1], 0, 0)
        self.assertEqual(undo_stack.call_count, 2)

        positioner = mock.Mock()
        positioner.base = np.identity(4).tolist()
        self.presenter.changePositionerBase(positioner, np.random.rand(4, 4).tolist())
        self.assertEqual(undo_stack.call_count, 3)

        self.presenter.changeJawAperture([2.0, 2.0])
        self.assertEqual(undo_stack.call_count, 4)

        self.presenter.changePositioningStack("default")
        self.assertEqual(undo_stack.call_count, 5)

        self.presenter.movePositioner("default", [1])
        self.assertEqual(undo_stack.call_count, 6)

        self.model_mock.return_value.instrument.getPositioner.return_value = positioner
        link = mock.Mock()
        link.ignore_limits, link.locked = False, False
        positioner.links = [link]
        self.presenter.ignorePositionerJointLimits("default", 0, True)
        self.assertEqual(undo_stack.call_count, 7)

        self.presenter.lockPositionerJoint("default", 0, True)
        self.assertEqual(undo_stack.call_count, 8)

        self.view_mock.collimator_action_groups = {"detector": ""}
        self.view_mock.scenes = mock.Mock()
        self.presenter.changeCollimators("detector", "collimator")
        self.assertEqual(undo_stack.call_count, 9)

        self.presenter.deletePoints([1], PointType.Fiducial)
        self.assertEqual(undo_stack.call_count, 10)

        self.presenter.editPoints([1], PointType.Fiducial)
        self.assertEqual(undo_stack.call_count, 11)

        self.model_mock.return_value.fiducials = [3, 4]
        self.presenter.movePoints(0, 1, PointType.Fiducial)
        self.assertEqual(undo_stack.call_count, 12)

        self.presenter.deleteSample("demo")
        self.assertEqual(undo_stack.call_count, 13)

        self.presenter.changeMainSample("demo")
        self.assertEqual(undo_stack.call_count, 14)

        self.presenter.mergeSample(["demo", "next"])
        self.assertEqual(undo_stack.call_count, 15)

        self.view_mock.showSelectChoiceMessage.return_value = "Combine"
        self.presenter.addPrimitive(Primitives.Cuboid, {})
        self.assertEqual(undo_stack.call_count, 16)

        self.presenter.transformSample([1, 1, 1], None, TransformType.Translate)
        self.assertEqual(undo_stack.call_count, 17)
        self.presenter.transformSample([1, 1, 1], None, TransformType.Rotate)
        self.assertEqual(undo_stack.call_count, 18)
        self.presenter.transformSample(np.identity(4), None, TransformType.Custom)
        self.assertEqual(undo_stack.call_count, 19)
        self.presenter.transformSample(np.identity(4), None, TransformType.Origin)
        self.assertEqual(undo_stack.call_count, 20)
        self.presenter.transformSample(np.identity(4), None, TransformType.Plane)
        self.assertEqual(undo_stack.call_count, 21)

        self.model_mock.return_value.sample = {}
        self.assertEqual(self.view_mock.showMessage.call_count, 0)
        self.presenter.createVectorsWithEulerAngles()
        self.assertEqual(self.view_mock.showMessage.call_count, 1)
        self.model_mock.return_value.sample = mock.Mock()
        self.view_mock.showOpenDialog.return_value = ""
        self.presenter.createVectorsWithEulerAngles()
        self.view_mock.showOpenDialog.return_value = "demo.angles"
        self.presenter.createVectorsWithEulerAngles()
        self.assertEqual(undo_stack.call_count, 22)

    def testAddPointAndVectors(self):
        undo_stack = mock.Mock()
        self.view_mock.undo_stack.push = undo_stack
        self.view_mock.docks = mock.Mock()

        self.model_mock.return_value.sample = {}
        self.presenter.addPoints([1], PointType.Fiducial)
        self.view_mock.showMessage.assert_called_once()
        undo_stack.assert_not_called()

        self.presenter.addVectors(-1, 0, 0, 0)
        self.assertEqual(self.view_mock.showMessage.call_count, 2)
        undo_stack.assert_not_called()

        self.model_mock.return_value.sample = {"demo": None}
        self.presenter.addPoints([1], PointType.Measurement, False)
        undo_stack.assert_called_once()
        self.view_mock.docks.showPointManager.assert_not_called()

        self.presenter.addPoints([1], PointType.Fiducial, True)
        self.view_mock.docks.showPointManager.assert_called_with(PointType.Fiducial)
        self.presenter.addPoints([1], PointType.Measurement, True)
        self.view_mock.docks.showPointManager.assert_called_with(PointType.Measurement)

        undo_stack.reset_mock()
        self.model_mock.return_value.measurement_points = np.array([])
        self.presenter.addVectors(-1, 0, 0, 0)
        self.assertEqual(self.view_mock.showMessage.call_count, 3)
        undo_stack.assert_not_called()

        self.model_mock.return_value.measurement_points = np.array([1])
        self.presenter.addVectors(-1, 0, 0, 0)
        undo_stack.assert_called_once()


if __name__ == "__main__":
    unittest.main()
