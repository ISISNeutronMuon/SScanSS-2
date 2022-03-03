import unittest
import unittest.mock as mock
from PyQt5.QtWidgets import QUndoStack
import numpy as np
from sscanss.app.window.view import MainWindow
from sscanss.app.dialogs import ProgressDialog
from sscanss.app.window.dock_manager import DockManager
from sscanss.app.window.presenter import MainWindowPresenter
from sscanss.app.commands import (RotateSample, TranslateSample, InsertPrimitive, CreateVectorsWithEulerAngles,
                                  TransformSample, InsertPoints, DeletePoints, EditPoints, MovePoints,
                                  InsertAlignmentMatrix, RemoveVectors, RemoveVectorAlignment, InsertMeshFromFile,
                                  InsertPointsFromFile, InsertVectorsFromFile, InsertVectors, ChangeCollimator,
                                  ChangeJawAperture, ChangePositionerBase, LockJoint, IgnoreJointLimits,
                                  ChangePositioningStack, MovePositioner)
from sscanss.core.geometry import Mesh, Volume
from sscanss.core.util import (Primitives, PointType, POINT_DTYPE, CommandID, LoadVector, StrainComponents,
                               InsertSampleOptions)
from tests.helpers import TestSignal, create_worker


class TestTransformCommands(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(MainWindow)
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = ["dummy"]
        self.presenter = MainWindowPresenter(self.view_mock)

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        self.mesh = Mesh(vertices, indices, normals)

        self.data = np.zeros([3, 3, 3], np.uint8)
        self.data[[0, 1, 2], [0, 1, 2], [0, 1, 2]] = 1
        self.volume = Volume(self.data.copy(), np.ones(3), np.ones(3))

        self.fiducials = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True)], dtype=POINT_DTYPE)
        self.measurements = np.rec.array([([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], False)], dtype=POINT_DTYPE)
        self.vectors = np.array([[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]]])
        self.alignment = np.array([[1., 0., 0., -10.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])

    def testRotateSampleCommand(self):
        self.model_mock.return_value.sample = self.volume
        default_matrix = self.volume.transform_matrix

        # Command to rotate volume sample
        angles = [0, 90, 0]
        cmd = RotateSample(angles, self.presenter)
        cmd.redo()

        # Check that angles are converted to radians
        np.testing.assert_array_almost_equal(cmd.angles, np.radians(angles), decimal=5)

        expected_matrix = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.], [-1., 0., 0., -1.], [0., 0., 0., 1.]])
        np.testing.assert_array_equal(self.volume.data, self.data)  # data should be changed
        np.testing.assert_array_almost_equal(self.volume.transform_matrix, expected_matrix, decimal=5)
        cmd.undo()
        np.testing.assert_array_equal(self.volume.data, self.data)
        np.testing.assert_array_almost_equal(self.volume.transform_matrix, default_matrix, decimal=5)

        # Command to rotate the mesh sample with fiducials, measurement and alignment
        self.model_mock.return_value.sample = self.mesh
        self.model_mock.return_value.fiducials = self.fiducials.copy()
        self.model_mock.return_value.measurement_points = self.measurements.copy()
        self.model_mock.return_value.measurement_vectors = self.vectors.copy()
        self.model_mock.return_value.alignment = self.alignment.copy()

        angles = [90, 60, 30]
        cmd = RotateSample(angles, self.presenter)
        cmd.redo()

        expected_vertices = np.array([
            [-0.2320508, 3.6160254, 0.9330127],
            [-1.330127, 8.6650635, 0.3839746],
            [-2.4282032, 13.7141016, -0.1650635],
        ])
        expected_normals = np.array([[0.5, 0.75, 0.4330127], [-0.8660254, 0.4330127, 0.25], [0, 0.5, -0.8660254]])
        expected_fiducials = np.array([[0., 0., 0.], [0.5, 1.75, -1.299038]])
        expected_measurements = np.array([[0.5, 1.75, -1.299038], [-0.36602542, 1.1830127, 0.6830127]])
        expected_vectors = np.array([[[0.0], [0.5], [-0.866025404]], [[-0.866025404], [0.433012702], [0.25]]])
        expected_alignment = np.array([[0., 0.5, -0.866025, -10.], [0.5, 0.75, 0.433013, 0.],
                                       [-0.866025, 0.433013, 0.25, 0.], [0., 0., 0., 1.]])
        sample = self.model_mock.return_value.sample
        # Check that redo rotates vertices, normals but not the indices of mesh
        np.testing.assert_array_almost_equal(sample.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.fiducials.points,
                                             expected_fiducials,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.fiducials.enabled, self.fiducials.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_points.points,
                                             expected_measurements,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_points.enabled,
                                      self.measurements.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_vectors,
                                             expected_vectors,
                                             decimal=5)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.alignment, expected_alignment, decimal=5)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the rotation
        np.testing.assert_array_almost_equal(sample.vertices, self.mesh.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, self.mesh.normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.fiducials.points,
                                             self.fiducials.points,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.fiducials.enabled, self.fiducials.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_points.points,
                                             self.measurements.points,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_points.enabled,
                                      self.measurements.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_vectors, self.vectors, decimal=5)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.alignment, self.alignment, decimal=5)

    def testTranslateSampleCommand(self):
        self.model_mock.return_value.sample = self.volume
        default_matrix = self.volume.transform_matrix

        # Command to translate volume sample
        offset = [10, -5, 3]
        cmd = TranslateSample(offset, self.presenter)
        cmd.redo()

        expected_matrix = np.array([[1., 0., 0., 11.], [0., 1., 0., -4.], [0., 0., 1., 4.], [0., 0., 0., 1.]])
        np.testing.assert_array_equal(self.volume.data, self.data)  # data should be changed
        np.testing.assert_array_almost_equal(self.volume.transform_matrix, expected_matrix, decimal=5)
        cmd.undo()
        np.testing.assert_array_equal(self.volume.data, self.data)
        np.testing.assert_array_almost_equal(self.volume.transform_matrix, default_matrix, decimal=5)

        self.model_mock.return_value.sample = self.mesh
        self.model_mock.return_value.fiducials = self.fiducials.copy()
        self.model_mock.return_value.measurement_points = self.measurements.copy()
        self.model_mock.return_value.measurement_vectors = self.vectors.copy()
        self.model_mock.return_value.alignment = self.alignment.copy()

        # Command to translate the mesh sample with fiducials, measurement and alignment
        offset = [30, 60, 90]
        cmd = TranslateSample(offset, self.presenter)
        cmd.redo()
        self.fiducials = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True)], dtype=POINT_DTYPE)
        self.measurements = np.rec.array([([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], False)], dtype=POINT_DTYPE)

        expected_vertices = np.array([[31, 62, 93], [34, 65, 96], [37, 68, 99]])
        expected_fiducials = np.array([[30., 60., 90.], [32., 60., 91.]])
        expected_measurements = np.array([[32., 60., 91.], [30., 61., 91.]])
        expected_alignment = np.array([[1., 0., 0., -40.], [0., 0., 1., -90.], [0., 1., 0., -60.], [0., 0., 0., 1.]])
        sample = self.model_mock.return_value.sample
        # Check that redo translates vertices but not the normals and indices of sample
        np.testing.assert_array_almost_equal(sample.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, self.mesh.normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.fiducials.points,
                                             expected_fiducials,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.fiducials.enabled, self.fiducials.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_points.points,
                                             expected_measurements,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_points.enabled,
                                      self.measurements.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_vectors, self.vectors, decimal=5)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.alignment, expected_alignment, decimal=5)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the translation on the sample ad others
        np.testing.assert_array_almost_equal(sample.vertices, self.mesh.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, self.mesh.normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.fiducials.points,
                                             self.fiducials.points,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.fiducials.enabled, self.fiducials.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_points.points,
                                             self.measurements.points,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_points.enabled,
                                      self.measurements.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_vectors, self.vectors, decimal=5)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.alignment, self.alignment, decimal=5)

    def testTransformSampleCommand(self):
        self.model_mock.return_value.sample = self.volume
        default_matrix = self.volume.transform_matrix

        # Command to transform volume sample
        matrix = [[0.0, 0.0, 1.0, 10.0], [0.0, 1.0, 0.0, -5.0], [1.0, 0.0, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0]]
        cmd = TransformSample(matrix, self.presenter)
        cmd.redo()

        expected_matrix = np.array([[0., 0., 1., 11.], [0., 1., 0., -4.], [1., 0., 0., 1.4], [0., 0., 0., 1.]])
        np.testing.assert_array_equal(self.volume.data, self.data)  # data should be changed
        np.testing.assert_array_almost_equal(self.volume.transform_matrix, expected_matrix, decimal=5)
        cmd.undo()
        np.testing.assert_array_equal(self.volume.data, self.data)
        np.testing.assert_array_almost_equal(self.volume.transform_matrix, default_matrix, decimal=5)

        # Command to transform the mesh sample with fiducials, measurement and alignment
        self.model_mock.return_value.sample = self.mesh
        self.model_mock.return_value.fiducials = self.fiducials.copy()
        self.model_mock.return_value.measurement_points = self.measurements.copy()
        self.model_mock.return_value.measurement_vectors = self.vectors.copy()
        self.model_mock.return_value.alignment = self.alignment.copy()
        cmd = TransformSample(matrix, self.presenter)
        cmd.redo()

        expected_vertices = np.array([[13.0, -3.0, 1.4], [16.0, 0.0, 4.4], [19.0, 3.0, 7.4]])
        expected_normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        expected_fiducials = np.array([[10., -5., 0.4], [11., -5., 2.4]])
        expected_measurements = np.array([[11., -5., 2.4], [11., -4., 0.4]])
        expected_vectors = np.array([[[0.], [0.], [1.]], [[0.], [1.], [0.]]])
        expected_alignment = np.array([[0., 0., 1., -10.4], [1., 0., 0., -10.], [0., 1., 0., 5.], [0., 0., 0., 1.]])

        sample = self.model_mock.return_value.sample
        # Check that redo transforms vertices, normals but not the indices of sample
        np.testing.assert_array_almost_equal(sample.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(sample.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, self.mesh.normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.fiducials.points,
                                             expected_fiducials,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.fiducials.enabled, self.fiducials.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_points.points,
                                             expected_measurements,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_points.enabled,
                                      self.measurements.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_vectors,
                                             expected_vectors,
                                             decimal=5)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.alignment, expected_alignment, decimal=5)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the transformation on the sample
        np.testing.assert_array_almost_equal(sample.vertices, self.mesh.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, self.mesh.normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(sample.vertices, self.mesh.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample.normals, self.mesh.normals, decimal=5)
        np.testing.assert_array_equal(sample.indices, self.mesh.indices)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.fiducials.points,
                                             self.fiducials.points,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.fiducials.enabled, self.fiducials.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_points.points,
                                             self.measurements.points,
                                             decimal=5)
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_points.enabled,
                                      self.measurements.enabled)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.measurement_vectors, self.vectors, decimal=5)
        np.testing.assert_array_almost_equal(self.model_mock.return_value.alignment, self.alignment, decimal=5)


class TestInsertCommands(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(MainWindow)
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = ["dummy"]
        self.presenter = MainWindowPresenter(self.view_mock)

    def testInsertPrimitiveCommand(self):
        self.model_mock.return_value.sample = None

        # Command to add a cuboid to sample
        args = {"width": 50.000, "height": 100.000, "depth": 200.000}
        cmd = InsertPrimitive(Primitives.Cuboid, args, self.presenter, InsertSampleOptions.Combine)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()
        self.model_mock.return_value.sample = self.model_mock.return_value.addMeshToProject.call_args[0][0]
        self.assertIsNotNone(self.model_mock.return_value.sample)
        self.assertIs(self.model_mock.return_value.addMeshToProject.call_args[0][1], InsertSampleOptions.Combine)
        cmd.undo()
        self.assertIsNone(self.model_mock.return_value.sample)

        # Command to add a cylinder to sample
        self.model_mock.reset_mock()
        args = {"radius": 100.000, "height": 200.000}
        cmd = InsertPrimitive(Primitives.Cylinder, args, self.presenter, InsertSampleOptions.Combine)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()

        # Command to add a sphere to sample
        self.model_mock.reset_mock()
        args = {"radius": 100.000}
        cmd = InsertPrimitive(Primitives.Sphere, args, self.presenter, InsertSampleOptions.Combine)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()

        # Command to add a tube to sample
        self.model_mock.return_value.sample = None
        self.model_mock.reset_mock()
        args = {"outer_radius": 100.000, "inner_radius": 50.000, "height": 200.000}
        cmd = InsertPrimitive(Primitives.Tube, args, self.presenter, InsertSampleOptions.Replace)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()
        self.model_mock.return_value.sample = self.model_mock.return_value.addMeshToProject.call_args[0][0]
        self.assertIsNotNone(self.model_mock.return_value.sample)
        self.assertIs(self.model_mock.return_value.addMeshToProject.call_args[0][1], InsertSampleOptions.Replace)
        cmd.undo()
        self.assertIsNone(self.model_mock.return_value.sample)

    @mock.patch("sscanss.app.commands.insert.logging", autospec=True)
    @mock.patch("sscanss.app.commands.insert.read_3d_model", autospec=True)
    def testInsertMeshFromFileCommand(self, reader_mock, _):
        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        reader_mock.return_value = Mesh(vertices, indices, normals)
        self.view_mock.progress_dialog = mock.create_autospec(ProgressDialog)
        self.view_mock.undo_stack = mock.create_autospec(QUndoStack)
        with mock.patch("sscanss.app.commands.insert.Worker", create_worker()):
            self.model_mock.return_value.sample = None
            cmd = InsertMeshFromFile('random.stl', self.presenter, InsertSampleOptions.Combine)
            cmd.redo()
            self.view_mock.progress_dialog.showMessage.assert_called_once()
            self.assertIsNone(cmd.old_sample)
            self.view_mock.progress_dialog.close.assert_called_once()
            self.model_mock.return_value.addMeshToProject.assert_called_once()
            self.model_mock.return_value.sample = self.model_mock.return_value.addMeshToProject.call_args[0][0]
            self.assertIs(cmd.new_mesh, reader_mock.return_value)
            self.assertIs(self.model_mock.return_value.sample, reader_mock.return_value)
            self.assertIs(self.model_mock.return_value.addMeshToProject.call_args[0][1], InsertSampleOptions.Combine)
            cmd.undo()
            self.assertIsNone(self.model_mock.return_value.sample)
            self.model_mock.return_value.addMeshToProject.assert_called_once()
            cmd.redo()
            self.assertEqual(self.model_mock.return_value.addMeshToProject.call_count, 2)
            self.assertFalse(cmd.isObsolete())

        exception = ValueError()
        reader_mock.return_value = None
        with mock.patch("sscanss.app.commands.insert.Worker", create_worker(exception)):
            self.model_mock.return_value.sample = Mesh(vertices, indices, normals)
            cmd = InsertMeshFromFile('random.obj', self.presenter, InsertSampleOptions.Replace)
            cmd.redo()
            self.assertEqual(self.view_mock.progress_dialog.showMessage.call_count, 2)
            self.assertEqual(self.view_mock.progress_dialog.close.call_count, 2)
            self.assertEqual(self.model_mock.return_value.addMeshToProject.call_count, 3)
            self.assertIsNone(self.model_mock.return_value.addMeshToProject.call_args[0][0])
            self.assertIs(self.model_mock.return_value.addMeshToProject.call_args[0][1], InsertSampleOptions.Replace)
            self.assertTrue(cmd.isObsolete())

    @mock.patch("sscanss.app.commands.insert.logging", autospec=True)
    @mock.patch("sscanss.app.commands.insert.Worker", autospec=True)
    def testInsertPointsFromFileCommand(self, worker_mock, _):
        worker_mock.return_value.job_succeeded = TestSignal()
        worker_mock.return_value.job_failed = TestSignal()
        worker_mock.return_value.finished = TestSignal()
        filename = "random"
        self.view_mock.progress_dialog = mock.create_autospec(ProgressDialog)
        self.view_mock.docks = mock.create_autospec(DockManager)
        self.view_mock.undo_stack = mock.create_autospec(QUndoStack)

        self.model_mock.return_value.fiducials = [1, 2]
        cmd = InsertPointsFromFile(filename, PointType.Fiducial, self.presenter)
        cmd.redo()
        self.view_mock.progress_dialog.showMessage.assert_called_once()
        worker_mock.return_value.job_succeeded.emit()
        self.view_mock.docks.showPointManager.assert_called_once_with(PointType.Fiducial)
        worker_mock.return_value.finished.emit()
        self.view_mock.progress_dialog.close.assert_called_once()
        worker_mock.return_value.job_failed.emit(Exception())
        self.assertTrue(cmd.isObsolete())
        self.model_mock.return_value.addMeshToProject.assert_not_called()
        self.model_mock.return_value.fiducials = [1, 2, 3]
        cmd.undo()
        self.assertListEqual(list(cmd.new_points), [3])
        self.model_mock.return_value.removePointsFromProject.assert_called_once_with(slice(2, 3), PointType.Fiducial)
        cmd.redo()
        self.model_mock.return_value.addPointsToProject.assert_called_once_with(np.array([3]), PointType.Fiducial)

        self.model_mock.return_value.measurement_points = [1, 2]
        cmd = InsertPointsFromFile(filename, PointType.Measurement, self.presenter)
        cmd.undo()
        self.model_mock.return_value.removePointsFromProject.assert_called_with(slice(2, 2), PointType.Measurement)

    def testInsertPointsCommand(self):
        self.model_mock.return_value.fiducials = []
        self.model_mock.return_value.measurement_points = []

        # Command to add a fiducial point
        points = [([0.0, 0.0, 0.0], False)]
        args = (points, PointType.Fiducial)
        cmd = InsertPoints(*args, self.presenter)
        self.assertEqual(cmd.old_count, 0)
        cmd.redo()
        self.model_mock.return_value.addPointsToProject.assert_called_once_with(*args)
        self.model_mock.return_value.fiducials = args[0]
        cmd.undo()
        self.model_mock.return_value.removePointsFromProject.assert_called_once_with(slice(0, 1, None), args[1])

        self.model_mock.reset_mock()
        self.model_mock.return_value.fiducials = []
        self.model_mock.return_value.measurement_points = points

        # Command to add  measurement points
        points = [([1.0, 0.0, 0.0], False), ([1.0, 1.0, 0.0], True)]
        args = (points, PointType.Measurement)
        cmd = InsertPoints(*args, self.presenter)
        self.assertEqual(cmd.old_count, 1)
        cmd.redo()
        self.model_mock.return_value.addPointsToProject.assert_called_once_with(*args)
        self.model_mock.return_value.measurement_points.extend(args[0])
        cmd.undo()
        self.model_mock.return_value.removePointsFromProject.assert_called_once_with(slice(1, 3, None), args[1])

    def testDeletePointsCommand(self):
        points = np.rec.array([([0.0, 0.0, 0.0], False)], dtype=POINT_DTYPE)
        points_after_delete = np.recarray((0, ), dtype=POINT_DTYPE)
        self.model_mock.return_value.fiducials = points
        self.model_mock.return_value.measurement_points = []
        self.model_mock.return_value.measurement_vectors = []

        # Command to delete a fiducial point
        args = ([0], PointType.Fiducial)
        cmd = DeletePoints(*args, self.presenter)
        self.assertIsNone(cmd.removed_points)
        self.assertIsNone(cmd.removed_vectors)
        cmd.redo()
        self.model_mock.return_value.fiducials = points_after_delete
        self.assertEqual(cmd.removed_points, points[args[0]])
        self.assertIsNone(cmd.removed_vectors)
        self.model_mock.return_value.removePointsFromProject.assert_called_once_with(*args)
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, points)
        self.assertEqual(len(self.model_mock.return_value.measurement_points), 0)
        self.assertEqual(len(self.model_mock.return_value.measurement_vectors), 0)

        self.model_mock.reset_mock()
        points = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], True)],
                              dtype=POINT_DTYPE)
        vectors = np.array([[[0.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]])

        self.model_mock.return_value.fiducials = []
        self.model_mock.return_value.measurement_points = points
        self.model_mock.return_value.measurement_vectors = vectors

        # Command to delete some measurement points
        indices = [2, 0]
        sorted_indices = [0, 2]
        points_after_delete = np.delete(points, sorted_indices, 0)
        vectors_after_delete = np.delete(vectors, sorted_indices, 0)
        args = (indices, PointType.Measurement)
        cmd = DeletePoints(*args, self.presenter)
        self.assertIsNone(cmd.removed_points)
        self.assertIsNone(cmd.removed_vectors)
        cmd.redo()
        self.model_mock.return_value.measurement_points = points_after_delete
        self.model_mock.return_value.measurement_vectors = vectors_after_delete
        np.testing.assert_equal(cmd.removed_points, points[sorted_indices])
        np.testing.assert_equal(cmd.removed_vectors, vectors[sorted_indices])
        self.model_mock.return_value.removePointsFromProject.assert_called_once_with(sorted_indices, args[1])
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, points)
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, vectors)
        self.assertEqual(len(self.model_mock.return_value.fiducials), 0)

    def testEditPointsCommand(self):
        points = np.rec.array([([0.0, 0.0, 0.0], False)], dtype=POINT_DTYPE)
        new_points = np.rec.array([([1.0, 1.0, 1.0], True)], dtype=POINT_DTYPE)
        self.model_mock.return_value.fiducials = points
        self.model_mock.return_value.measurement_points = []

        # Command to edit fiducial points
        cmd = EditPoints(new_points, PointType.Fiducial, self.presenter)
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, new_points)
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, points)

        self.model_mock.reset_mock()
        self.model_mock.return_value.fiducials = []
        self.model_mock.return_value.measurement_points = points

        # Command to edit measurement points
        cmd_1 = EditPoints(new_points, PointType.Measurement, self.presenter)
        cmd_1.redo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, new_points)
        cmd_1.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, points)
        cmd_1.redo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, new_points)
        self.assertFalse(cmd.mergeWith(cmd_1))

        newer_points = np.rec.array([([2.0, 2.0, 2.0], True)], dtype=POINT_DTYPE)
        cmd_2 = EditPoints(newer_points, PointType.Measurement, self.presenter)
        self.assertTrue(cmd_1.mergeWith(cmd_2))
        cmd_1.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, points)
        cmd_1.redo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, newer_points)
        self.assertTrue(cmd_1.mergeWith(EditPoints(points, PointType.Measurement, self.presenter)))
        self.assertTrue(cmd_1.isObsolete())
        self.assertTrue(cmd_1.id(), CommandID.EditPoints)

    def testMovePointsCommand(self):
        points = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], True)],
                              dtype=POINT_DTYPE)
        copied_points = points.copy()
        self.model_mock.return_value.fiducials = points
        self.model_mock.return_value.measurement_points = []
        self.model_mock.return_value.measurement_vectors = []

        # Command to move fiducial points
        cmd = MovePoints(2, 0, PointType.Fiducial, self.presenter)
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, copied_points[[2, 1, 0]])
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, copied_points)
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, copied_points[[2, 1, 0]])

        cmd_1 = MovePoints(0, 1, PointType.Fiducial, self.presenter)
        cmd_1.redo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, copied_points[[1, 2, 0]])
        self.assertTrue(cmd.mergeWith(cmd_1))
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, copied_points)
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.fiducials, copied_points[[1, 2, 0]])

        self.assertTrue(cmd.mergeWith(MovePoints(0, 2, PointType.Fiducial, self.presenter)))
        self.assertTrue(cmd.mergeWith(MovePoints(1, 2, PointType.Fiducial, self.presenter)))
        self.assertTrue(cmd.isObsolete())
        self.assertEqual(cmd.id(), CommandID.MovePoints)

        self.model_mock.reset_mock()
        points = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], True)],
                              dtype=POINT_DTYPE)
        vectors = np.array([[[0.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]])
        copied_vectors = vectors.copy()
        self.model_mock.return_value.fiducials = []
        self.model_mock.return_value.measurement_points = points
        self.model_mock.return_value.measurement_vectors = vectors

        # Command to move measurement points
        cmd_2 = MovePoints(0, 1, PointType.Measurement, self.presenter)
        cmd_2.redo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, copied_points[[1, 0, 2]])
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors[[1, 0, 2]])
        cmd_2.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_points, copied_points)
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors)
        self.assertFalse(cmd_1.mergeWith(cmd_2))

    def testInsertAlignmentMatrixCommand(self):
        self.model_mock.return_value.alignment = None

        matrix = np.identity(4)
        cmd = InsertAlignmentMatrix(matrix, self.presenter)
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.alignment, matrix)
        cmd.undo()
        self.assertIsNone(self.model_mock.return_value.alignment)

        matrix = np.ones((4, 4))
        self.assertTrue(cmd.mergeWith(InsertAlignmentMatrix(matrix, self.presenter)))
        self.assertFalse(cmd.isObsolete())
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.alignment, matrix)
        cmd.undo()
        self.assertIsNone(self.model_mock.return_value.alignment)
        self.assertTrue(cmd.mergeWith(InsertAlignmentMatrix(None, self.presenter)))
        self.assertTrue(cmd.isObsolete())
        self.assertEqual(cmd.id(), CommandID.AlignSample)

    def testRemoveVectorsCommand(self):
        vectors = np.array([
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0]],
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0]],
        ])
        copied_vectors = vectors.copy()
        self.model_mock.return_value.measurement_vectors = vectors

        cmd = RemoveVectors([0], 0, 0, self.presenter)
        cmd.redo()
        temp = vectors.copy()
        temp[0, 0:3, 0] = 0
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, temp)
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors)

        cmd = RemoveVectors([0, 1], 1, 1, self.presenter)
        cmd.redo()
        temp = vectors.copy()
        temp[:, 3:6, 1] = 0
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, temp)
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors)

    def testRemoveVectorAlignmentCommand(self):
        vectors = np.array([
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],
        ])
        copied_vectors = vectors.copy()
        self.model_mock.return_value.measurement_vectors = vectors

        cmd = RemoveVectorAlignment(0, self.presenter)
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors[:, :, 1:])
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors)

        cmd = RemoveVectorAlignment(1, self.presenter)
        cmd.redo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors[:, :, :1])
        cmd.undo()
        np.testing.assert_equal(self.model_mock.return_value.measurement_vectors, copied_vectors)

    @mock.patch("sscanss.app.commands.insert.logging", autospec=True)
    @mock.patch("sscanss.app.commands.insert.Worker", autospec=True)
    def testInsertVectorsFromFileCommand(self, worker_mock, _):
        worker_mock.return_value.job_succeeded = TestSignal()
        worker_mock.return_value.job_failed = TestSignal()
        worker_mock.return_value.finished = TestSignal()
        filename = "random"
        self.view_mock.progress_dialog = mock.create_autospec(ProgressDialog)
        self.view_mock.docks = mock.create_autospec(DockManager)
        self.view_mock.undo_stack = mock.create_autospec(QUndoStack)

        vectors = np.array([[1, 2], [3, 4]])
        self.model_mock.return_value.measurement_vectors = vectors
        cmd = InsertVectorsFromFile(filename, self.presenter)
        cmd.redo()
        self.view_mock.progress_dialog.showMessage.assert_called_once()
        worker_mock.return_value.job_succeeded.emit(LoadVector.Smaller)
        self.assertEqual(self.view_mock.showMessage.call_count, 1)
        self.assertEqual(self.view_mock.docks.showVectorManager.call_count, 1)
        worker_mock.return_value.job_succeeded.emit(LoadVector.Larger)
        self.assertEqual(self.view_mock.showMessage.call_count, 2)
        self.assertEqual(self.view_mock.docks.showVectorManager.call_count, 2)
        worker_mock.return_value.job_succeeded.emit(LoadVector.Exact)
        self.assertEqual(self.view_mock.showMessage.call_count, 2)
        self.assertEqual(self.view_mock.docks.showVectorManager.call_count, 3)
        worker_mock.return_value.finished.emit()
        self.view_mock.progress_dialog.close.assert_called_once()
        worker_mock.return_value.job_failed.emit(Exception())
        self.assertTrue(cmd.isObsolete())
        self.model_mock.return_value.measurement_vectors = np.identity(2)
        cmd.undo()
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_vectors, vectors)
        cmd.redo()
        worker_mock.return_value.start.assert_called_once()
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_vectors, np.identity(2))

    @mock.patch("sscanss.app.commands.insert.logging", autospec=True)
    @mock.patch("sscanss.app.commands.insert.Worker", autospec=True)
    def testInsertVectorsCommand(self, worker_mock, _):
        worker_mock.return_value.job_succeeded = TestSignal()
        worker_mock.return_value.job_failed = TestSignal()
        worker_mock.return_value.finished = TestSignal()
        self.view_mock.progress_dialog = mock.create_autospec(ProgressDialog)
        self.view_mock.docks = mock.create_autospec(DockManager)
        self.view_mock.undo_stack = mock.create_autospec(QUndoStack)

        self.model_mock.return_value.measurement_points = np.array([[1, 2, 3]])
        self.model_mock.return_value.measurement_vectors = np.identity(3)
        cmd = InsertVectors(self.presenter, -1, StrainComponents.ParallelX, 1, 1)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        self.view_mock.progress_dialog.showMessage.assert_called_once()
        worker_mock.return_value.job_succeeded.emit()
        self.view_mock.docks.showVectorManager.assert_called_once()
        worker_mock.return_value.finished.emit()
        self.view_mock.progress_dialog.close.assert_called_once()
        worker_mock.return_value.job_failed.emit(Exception())
        self.assertTrue(cmd.isObsolete())

        expected = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0, 0.0]])
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        self.model_mock.return_value.measurement_vectors = actual
        np.testing.assert_array_equal(actual, expected)
        cmd.undo()
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_vectors, np.identity(3))
        cmd.redo()
        np.testing.assert_array_equal(self.model_mock.return_value.measurement_vectors, expected)

        cmd = InsertVectors(self.presenter, 0, StrainComponents.ParallelY, 1, 1)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        np.testing.assert_array_equal(actual, [[0.0, 1.0, 0.0]])

        cmd = InsertVectors(self.presenter, 2, StrainComponents.ParallelZ, 1, 1)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        np.testing.assert_array_equal(actual, [[0.0, 0.0, 1.0]])

        cmd = InsertVectors(self.presenter, 0, StrainComponents.Custom, 1, 1, key_in=[1.0, 1.0, 0.0], reverse=True)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        np.testing.assert_array_almost_equal(actual, [[-0.707107, -0.707107, 0.0]], decimal=5)

        vertices = np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        indices = np.array([2, 1, 0])

        points = np.rec.array([([0.0, 0.0, 0.0], False), ([0.1, 0.5, 0.0], True)], dtype=POINT_DTYPE)
        self.model_mock.return_value.measurement_points = points
        self.model_mock.return_value.sample = {"1": Mesh(vertices, indices, normals)}
        cmd = InsertVectors(self.presenter, 0, StrainComponents.SurfaceNormal, 1, 1)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        np.testing.assert_array_almost_equal(actual, [[0.0, 0.0, 1.0]], decimal=5)

        cmd = InsertVectors(self.presenter, -1, StrainComponents.OrthogonalWithoutX, 1, 1)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        np.testing.assert_array_almost_equal(actual, [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], decimal=5)

        cmd = InsertVectors(self.presenter, -1, StrainComponents.OrthogonalWithoutY, 1, 1)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        np.testing.assert_array_almost_equal(actual, [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], decimal=5)

        cmd = InsertVectors(self.presenter, 0, StrainComponents.OrthogonalWithoutZ, 1, 1)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        actual = self.model_mock.return_value.addVectorsToProject.call_args[0][0]
        np.testing.assert_array_almost_equal(actual, [[0.0, 0.0, 0.0]], decimal=5)

    @mock.patch("sscanss.app.commands.insert.logging", autospec=True)
    @mock.patch("sscanss.app.commands.insert.read_angles", autospec=True)
    @mock.patch("sscanss.app.commands.insert.Worker", autospec=True)
    def testCreateVectorWithEulerAnglesCommand(self, worker_mock, read_angles_func, _):
        worker_mock.return_value.job_succeeded = TestSignal()
        worker_mock.return_value.job_failed = TestSignal()
        worker_mock.return_value.finished = TestSignal()
        self.view_mock.progress_dialog = mock.create_autospec(ProgressDialog)
        self.view_mock.docks = mock.create_autospec(DockManager)
        self.view_mock.undo_stack = mock.create_autospec(QUndoStack)
        read_angles_func.return_value = (np.zeros((3, 3)), "xyz")
        self.model_mock.return_value.measurement_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.model_mock.return_value.measurement_vectors = np.array([])
        q_vectors = np.array([[-0.70710678, 0.70710678, 0.0], [-0.70710678, -0.70710678, 0.0]])
        self.model_mock.return_value.instrument.q_vectors = q_vectors
        self.model_mock.return_value.instrument.detectors = ["a", "b"]

        cmd = CreateVectorsWithEulerAngles("random.angles", self.presenter)
        worker_mock.return_value.start = cmd.createVectors
        cmd.redo()
        self.presenter.model.correctVectorAlignments.assert_called()
        expected_results = np.tile(q_vectors.flatten(), (3, 1))
        np.testing.assert_array_almost_equal(self.presenter.model.correctVectorAlignments.call_args[0][0],
                                             expected_results,
                                             decimal=5)
        self.model_mock.return_value.measurement_vectors = expected_results
        cmd.undo()
        self.assertEqual(self.presenter.model.measurement_vectors.size, 0)
        cmd.redo()
        np.testing.assert_array_almost_equal(self.presenter.model.measurement_vectors, expected_results, decimal=5)
        self.view_mock.progress_dialog.showMessage.assert_called_once()
        worker_mock.return_value.job_succeeded.emit(LoadVector.Smaller)
        self.assertEqual(self.view_mock.showMessage.call_count, 1)
        self.assertEqual(self.view_mock.docks.showVectorManager.call_count, 1)
        worker_mock.return_value.job_succeeded.emit(LoadVector.Larger)
        self.assertEqual(self.view_mock.showMessage.call_count, 2)
        self.assertEqual(self.view_mock.docks.showVectorManager.call_count, 2)
        worker_mock.return_value.job_succeeded.emit(LoadVector.Exact)
        self.assertEqual(self.view_mock.showMessage.call_count, 2)
        self.assertEqual(self.view_mock.docks.showVectorManager.call_count, 3)
        worker_mock.return_value.finished.emit()
        self.view_mock.progress_dialog.close.assert_called_once()
        worker_mock.return_value.job_failed.emit(Exception())
        self.assertTrue(cmd.isObsolete())

        read_angles_func.return_value = (np.array([[-30.0, 35.0, 0.0], [-30.0, 15.0, 0.0], [0.0, 0.0, 90.0]]), "xyz")
        cmd = CreateVectorsWithEulerAngles("random.angles", self.presenter)
        cmd.redo()
        expected_results = [
            [-0.579228, 0.8151623, -0.00231099, -0.579228, -0.40958256, 0.7047958],
            [-0.6830127, 0.70387876, -0.19505975, -0.6830127, -0.5208661, 0.51204705],
            [-0.70710677, -0.70710677, 0.0, 0.70710677, -0.70710677, 0.0],
        ]
        np.testing.assert_array_almost_equal(self.presenter.model.correctVectorAlignments.call_args[0][0],
                                             expected_results,
                                             decimal=5)

        read_angles_func.return_value = (np.array([[-30.0, 35.0, 0.0], [-30.0, 15.0, 0.0], [0.0, 0.0, 90.0]]), "zyx")
        cmd = CreateVectorsWithEulerAngles("random.angles", self.presenter)
        cmd.redo()
        expected_results = np.array([
            [-0.14807273, 0.90198642, 0.40557978, -0.85517955, -0.32275847, 0.40557978],
            [-0.23795296, 0.95387876, 0.18301271, -0.94505972, -0.2708661, 0.18301271],
            [-0.70710677, 0.0, 0.70710677, -0.70710677, 0.0, -0.70710677],
        ])
        np.testing.assert_array_almost_equal(self.presenter.model.correctVectorAlignments.call_args[0][0],
                                             expected_results,
                                             decimal=5)

        # single detector
        self.model_mock.return_value.instrument.detectors = ["a"]
        self.model_mock.return_value.instrument.q_vectors = q_vectors[0, None]
        cmd = CreateVectorsWithEulerAngles("random.angles", self.presenter)
        cmd.redo()
        np.testing.assert_array_almost_equal(self.presenter.model.correctVectorAlignments.call_args[0][0],
                                             expected_results[:, :3],
                                             decimal=5)


class TestControlCommands(unittest.TestCase):
    @mock.patch("sscanss.app.window.presenter.MainWindowModel", autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(MainWindow)
        self.view_mock.scenes = mock.Mock()
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = ["dummy"]
        self.model_mock.return_value.instrument_controlled = TestSignal()
        self.presenter = MainWindowPresenter(self.view_mock)

        self.positioner = mock.Mock()
        self.positioner.name = "default"
        self.instrument = self.model_mock.return_value.instrument
        self.instrument.positioning_stack = self.positioner
        self.instrument.getPositioner.return_value = self.positioner
        self.auxiliary = []
        for i in range(2):
            aux = mock.Mock()
            aux.base = np.random.rand(4, 4).tolist()
            self.auxiliary.append(aux)

        self.links = []
        for i in range(4):
            link = mock.Mock()
            link.ignore_limits = False
            link.locked = False
            self.links.append(link)
        self.positioner.links = self.links
        self.positioner.auxiliary = self.auxiliary

    def testLockJointCommand(self):
        command = LockJoint("", 0, True, self.presenter)
        command.redo()
        self.assertListEqual([link.locked for link in self.positioner.links], [True, False, False, False])
        command.undo()
        self.assertListEqual([link.locked for link in self.positioner.links], [False, False, False, False])

        command2 = LockJoint("", 3, True, self.presenter)
        command2.redo()
        self.assertListEqual([link.locked for link in self.positioner.links], [False, False, False, True])
        command2.undo()
        self.assertListEqual([link.locked for link in self.positioner.links], [False, False, False, False])

        self.assertFalse(command.mergeWith(LockJoint("Test", 0, 2, self.presenter)))
        self.assertTrue(command.mergeWith(LockJoint("", 0, False, self.presenter)))
        self.assertEqual(command.id(), CommandID.LockJoint)
        self.assertTrue(command.isObsolete())

    def testIgnoreJointLimitsCommand(self):
        command = IgnoreJointLimits("", 0, True, self.presenter)
        command.redo()
        self.assertListEqual([link.ignore_limits for link in self.positioner.links], [True, False, False, False])
        command.undo()
        self.assertListEqual([link.ignore_limits for link in self.positioner.links], [False, False, False, False])

        command2 = IgnoreJointLimits("", 3, True, self.presenter)
        command2.redo()
        self.assertListEqual([link.ignore_limits for link in self.positioner.links], [False, False, False, True])
        command2.undo()
        self.assertListEqual([link.ignore_limits for link in self.positioner.links], [False, False, False, False])

        self.assertFalse(command.mergeWith(IgnoreJointLimits("Test", 0, 2, self.presenter)))
        self.assertTrue(command.mergeWith(IgnoreJointLimits("", 0, False, self.presenter)))
        self.assertEqual(command.id(), CommandID.IgnoreJointLimits)
        self.assertTrue(command.isObsolete())

    def testMovePositionerCommand(self):
        set_points = [1.0, 2.0, 3.0, 4.0]
        new_set_points = [4.0, 3.0, 2.0, 1.0]
        self.positioner.set_points = set_points
        self.view_mock.scenes.isRunning = mock.Mock(return_value=True)

        command = MovePositioner("default", new_set_points, False, self.presenter)
        self.instrument.getPositioner.assert_called_with(self.positioner.name)
        self.assertEqual(command.move_from, set_points)
        self.assertEqual(command.move_to, new_set_points)
        command.redo()
        self.model_mock.return_value.moveInstrument.assert_called_once()
        command.undo()
        self.positioner.fkine.assert_called_once_with(set_points, ignore_locks=False)
        command.redo()
        self.model_mock.return_value.moveInstrument.assert_called_once()
        self.positioner.fkine.assert_called_with(new_set_points, ignore_locks=False)

        self.assertFalse(command.mergeWith(MovePositioner("another", new_set_points, False, self.presenter)))
        self.assertFalse(command.mergeWith(MovePositioner("default", new_set_points, True, self.presenter)))
        self.assertTrue(command.mergeWith(MovePositioner("default", set_points, False, self.presenter)))
        self.assertEqual(command.id(), CommandID.MovePositioner)
        self.assertTrue(command.isObsolete())

    def testChangePositioningStackCommand(self):
        command = ChangePositioningStack("new", self.presenter)
        self.assertEqual(command.old_stack, "default")
        self.assertEqual(command.new_stack, "new")
        command.redo()
        self.links[0].locked = True
        self.links[3].ignore_limits = True
        matrix, self.auxiliary[1].base = self.auxiliary[1].base, np.identity(4).tolist()
        self.instrument.loadPositioningStack.assert_called_with("new")
        command.undo()
        self.assertListEqual([link.locked for link in self.links], [False] * len(self.links))
        self.assertListEqual([link.ignore_limits for link in self.links], [False] * len(self.links))
        self.assertListEqual(self.auxiliary[1].base, matrix)
        self.instrument.loadPositioningStack.assert_called_with("default")
        self.assertEqual(command.id(), CommandID.ChangePositioningStack)

    def testChangeJawApertureCommand(self):
        aperture = [3.0, 1.5]
        new_aperture = [2.0, 2.0]
        self.instrument.jaws.aperture = aperture
        command = ChangeJawAperture(new_aperture, self.presenter)
        self.assertListEqual(command.old_aperture, aperture)
        self.assertListEqual(command.new_aperture, new_aperture)
        command.redo()
        self.assertListEqual(self.instrument.jaws.aperture, new_aperture)
        command.undo()
        self.assertListEqual(self.instrument.jaws.aperture, aperture)

        self.assertTrue(command.mergeWith(ChangeJawAperture([1.2, 0.5], self.presenter)))
        self.assertListEqual(command.new_aperture, [1.2, 0.5])
        self.assertTrue(command.mergeWith(ChangeJawAperture(aperture, self.presenter)))
        self.assertListEqual(command.new_aperture, aperture)
        self.assertTrue(command.isObsolete())

    def testChangePositionerBaseCommand(self):
        self.positioner.base = np.identity(4).tolist()
        matrix = np.random.rand(4, 4).tolist()

        command = ChangePositionerBase(self.positioner, matrix, self.presenter)
        command.redo()
        self.instrument.positioning_stack.changeBaseMatrix.assert_called_with(self.positioner, matrix)
        command.undo()
        self.instrument.positioning_stack.changeBaseMatrix.assert_called_with(self.positioner, self.positioner.base)

        positioner2 = mock.Mock()
        positioner2.base = np.identity(4).tolist()
        self.assertFalse(command.mergeWith(ChangePositionerBase(positioner2, matrix, self.presenter)))
        self.assertTrue(command.mergeWith(ChangePositionerBase(self.positioner, self.positioner.base, self.presenter)))
        self.assertTrue(command.isObsolete())
        self.assertEqual(command.id(), CommandID.ChangePositionerBase)

    @mock.patch("sscanss.app.commands.control.toggle_action_in_group", autospec=True)
    def testChangeCollimatorCommand(self, toggle_mock):
        detectors = []
        for i in range(2):
            collimator = mock.Mock()
            collimator.name = i * 2
            detector = mock.Mock()
            detector.current_collimator = collimator
            detectors.append(detector)
        detectors = {"East": detectors[0], "West": detectors[1]}
        self.view_mock.collimator_action_groups = detectors
        self.model_mock.return_value.instrument.detectors = detectors

        command = ChangeCollimator("East", None, self.presenter)
        self.assertEqual(command.old_collimator_name, 0)
        self.assertIsNone(command.new_collimator_name)
        toggle_mock.assert_not_called()
        command.redo()
        self.assertIsNone(detectors["East"].current_collimator)
        toggle_mock.assert_called()
        command.undo()
        self.assertEqual(detectors["East"].current_collimator, 0)
        self.assertEqual(toggle_mock.call_count, 2)
        command.redo()
        self.assertIsNone(detectors["East"].current_collimator)
        toggle_mock.assert_called()

        command2 = ChangeCollimator("West", 5, self.presenter)
        self.assertEqual(command2.old_collimator_name, 2)
        self.assertEqual(command2.new_collimator_name, 5)
        command2.redo()
        self.assertEqual(detectors["West"].current_collimator, 5)
        self.assertFalse(command2.mergeWith(command))

        collimator = mock.Mock()
        collimator.name = 5
        detectors["West"].current_collimator = collimator
        self.assertTrue(command2.mergeWith(ChangeCollimator("West", 6, self.presenter)))
        self.assertEqual(command2.old_collimator_name, 2)
        self.assertEqual(command2.new_collimator_name, 6)

        collimator = mock.Mock()
        collimator.name = 6
        detectors["West"].current_collimator = collimator
        self.assertTrue(command2.mergeWith(ChangeCollimator("West", 2, self.presenter)))
        self.assertTrue(command2.isObsolete())
        self.assertEqual(command2.id(), CommandID.ChangeCollimator)

        if __name__ == "__main__":
            unittest.main()
