import unittest
import unittest.mock as mock
import numpy as np
from sscanss.core.geometry import Mesh
from sscanss.core.util import Primitives
import sscanss.ui.window.view as view
from sscanss.ui.window.presenter import MainWindowPresenter
from sscanss.ui.commands import (RotateSample, TranslateSample, InsertPrimitive, DeleteSample,
                                 MergeSample, TransformSample)


class TestTransformCommands(unittest.TestCase):
    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(view.MainWindow)
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = ['dummy']
        self.presenter = MainWindowPresenter(self.view_mock)

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        self.mesh_1 = Mesh(vertices, indices, normals)

        vertices = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([1, 0, 2])
        self.mesh_2 = Mesh(vertices, indices, normals)
        self.sample = {'1': self.mesh_1, '2': self.mesh_2}

    def testRotateSampleCommand(self):
        self.model_mock.return_value.sample = self.sample.copy()

        # Command to rotate sample '1'
        angles = [0, 90, 0]
        cmd = RotateSample(angles, '1', self.presenter)
        cmd.redo()

        # Check that angles are converted to radians
        np.testing.assert_array_almost_equal(cmd.angles, np.radians(angles), decimal=5)

        expected_vertices = np.array([[3, 2, -1], [6, 5, -4], [9, 8, -7]])
        expected_normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        sample = self.model_mock.return_value.sample
        # Check that redo rotates vertices, normals but not the indices of sample '1'
        np.testing.assert_array_almost_equal(sample['1'].vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)

        # Check that redo does not rotate sample '2'
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the rotation on sample '1'
        np.testing.assert_array_almost_equal(sample['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)

        # Check that undo does not touch sample '2'
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        # Command to rotate all the samples
        angles = [30, 60, 90]
        cmd = RotateSample(angles, 'All', self.presenter)
        cmd.redo()

        expected_vertices_1 = np.array([[1.59807621, -0.75, 3.29903811],
                                       [2.69615242, -0.20096189, 8.34807621],
                                       [3.79422863, 0.34807621, 13.39711432]])
        expected_normals_1 = np.array([[0.866025, -0.25, 0.433013], [-0.5, -0.433013, 0.75], [0, 0.866025, 0.5]])
        expected_vertices_2 = np.array([[3.79422863, 0.34807621, 13.39711432],
                                        [2.69615242, -0.20096189, 8.34807621],
                                        [1.59807621, -0.75, 3.29903811]])
        expected_normals_2 = np.array([[-0.5, -0.433013, 0.75], [0.866025, -0.25, 0.433013], [0, 0.866025, 0.5]])
        sample = self.model_mock.return_value.sample
        # Check that redo rotates vertices, normals but not the indices of all samples'
        np.testing.assert_array_almost_equal(sample['1'].vertices, expected_vertices_1, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, expected_normals_1, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(sample['2'].vertices, expected_vertices_2, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, expected_normals_2, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the rotation on all samples
        np.testing.assert_array_almost_equal(sample['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

    def testTranslateSampleCommand(self):
        self.model_mock.return_value.sample = self.sample.copy()

        # Command to translate sample '2'
        offset = [10, -5, 3]
        cmd = TranslateSample(offset, '2', self.presenter)
        cmd.redo()

        expected_vertices = np.array([[17, 3, 12], [14, 0, 9], [11, -3, 6]])
        sample = self.model_mock.return_value.sample
        # Check that redo translates vertices but not the normals and indices of sample '2'
        np.testing.assert_array_almost_equal(sample['2'].vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        # Check that redo does not translate sample '2'
        np.testing.assert_array_almost_equal(sample['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the translation on sample '2'
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        # Check that undo does not touch sample '1'
        np.testing.assert_array_almost_equal(sample['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)

        # Command to translate all the samples
        offset = [30, 60, 90]
        cmd = TranslateSample(offset, 'All', self.presenter)
        cmd.redo()

        expected_vertices_1 = np.array([[31, 62, 93], [34, 65, 96], [37, 68, 99]])
        expected_vertices_2 = np.array([[37, 68, 99], [34, 65, 96], [31, 62, 93]])
        sample = self.model_mock.return_value.sample
        # Check that redo translates vertices, normals but not the indices of all samples'
        np.testing.assert_array_almost_equal(sample['1'].vertices, expected_vertices_1, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(sample['2'].vertices, expected_vertices_2, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the translation on all samples
        np.testing.assert_array_almost_equal(sample['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

    def testTransformSampleCommand(self):
        self.model_mock.return_value.sample = self.sample.copy()

        # Command to transform sample '1'
        matrix = [[0., 0., 1., 10.], [0., 1., 0., -5.], [1., 0., 0., 0.4], [0., 0., 0., 1.]]
        cmd = TransformSample(matrix, '1', self.presenter)

        cmd.redo()
        expected_vertices = np.array([[13., -3., 1.4], [16., 0., 4.4], [19., 3., 7.4]])
        expected_normals = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        sample = self.model_mock.return_value.sample

        # Check that redo transforms vertices, normals but not the indices of sample '1'
        np.testing.assert_array_almost_equal(sample['1'].vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)

        # Check that redo does not rotate sample '2'
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the translation on sample '2'
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        # Check that undo does not touch sample '1'
        np.testing.assert_array_almost_equal(sample['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)

        # Command to translate all the samples
        cmd = TransformSample(matrix, 'All', self.presenter)
        cmd.redo()

        expected_vertices_2 = np.array([[19.,  3.,  7.4], [16.,  0.,  4.4], [13., -3.,  1.4]])
        expected_normals_2 = np.array([[0., 1., 0.],[1., 0., 0.],[0., 0., 1.]])
        sample = self.model_mock.return_value.sample
        # Check that redo translates vertices, normals but not the indices of all samples'
        np.testing.assert_array_almost_equal(sample['1'].vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(sample['2'].vertices, expected_vertices_2, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, expected_normals_2, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)

        cmd.undo()
        sample = self.model_mock.return_value.sample
        # Check that undo reverses the translation on all samples
        np.testing.assert_array_almost_equal(sample['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(sample['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(sample['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(sample['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(sample['2'].indices, self.mesh_2.indices)


class TestInsertCommands(unittest.TestCase):
    @mock.patch('sscanss.ui.window.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(view.MainWindow)
        self.model_mock = model_mock
        self.model_mock.return_value.instruments = ['dummy']
        self.presenter = MainWindowPresenter(self.view_mock)

    def testInsertPrimitiveCommand(self):
        self.model_mock.return_value.sample = {}

        # Command to add a cuboid to sample
        args = {'width': 50.000, 'height': 100.000, 'depth': 200.000, 'name': 'Test'}
        cmd = InsertPrimitive(Primitives.Cuboid, args, self.presenter, True)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()
        cmd.undo()
        self.model_mock.return_value.removeMeshFromProject.assert_called_once()

        # Command to add a cylinder to sample
        self.model_mock.reset_mock()
        args = {'radius': 100.000, 'height': 200.000, 'name': 'Test'}
        cmd = InsertPrimitive(Primitives.Cylinder, args, self.presenter, True)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()

        # Command to add a sphere to sample
        self.model_mock.reset_mock()
        args = {'radius': 100.000, 'name': 'Test'}
        cmd = InsertPrimitive(Primitives.Sphere, args, self.presenter, True)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()

        # Command to add a tube to sample
        self.model_mock.reset_mock()
        args = {'outer_radius': 100.000, 'inner_radius': 50.000, 'height': 200.000, 'name': 'Test'}
        cmd = InsertPrimitive(Primitives.Tube, args, self.presenter, False)
        cmd.redo()
        self.model_mock.return_value.addMeshToProject.assert_called_once()
        self.assertEqual(cmd.old_sample, {})
        cmd.undo()
        self.model_mock.return_value.removeMeshFromProject.assert_not_called()

    def testDeleteSampleCommand(self):
        initial_sample = {'1': None, '2': None, '3': None}
        self.model_mock.return_value.sample = initial_sample

        # Command to delete multiple samples
        cmd = DeleteSample(['1', '3'], self.presenter)
        cmd.redo()
        self.assertEqual({'1': None, '3': None}, cmd.deleted_mesh)
        self.model_mock.return_value.removeMeshFromProject.assert_called_once()

        # Since removeMeshFromProject() is a mock object
        # we manually remove sample for the undo test
        self.model_mock.return_value.sample = {'2': None}
        cmd.undo()
        sample = self.model_mock.return_value.sample
        self.assertEqual(list(sample.keys()), list(initial_sample.keys()))
        self.assertEqual(sample, initial_sample)

        self.model_mock.reset_mock()
        cmd = DeleteSample(['2'], self.presenter)
        cmd.redo()
        self.assertEqual({'2': None}, cmd.deleted_mesh)
        self.model_mock.return_value.removeMeshFromProject.assert_called_once()

    def testMergeSampleCommand(self):
        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        mesh_1 = Mesh(vertices, indices, normals)

        vertices = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([1, 0, 2])
        mesh_2 = Mesh(vertices, indices, normals)

        initial_sample = {'1': mesh_1, '2': mesh_2, '3': None}
        self.model_mock.return_value.sample = initial_sample

        # Command to add a non-existent file to sample
        cmd = MergeSample(['1', '2'], self.presenter)
        cmd.redo()
        self.assertEqual([('1', 0), ('2', 3)], cmd.merged_mesh)
        self.assertEqual(initial_sample, {'3':None})
        self.model_mock.return_value.addMeshToProject.assert_called_once()

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([0, 1, 2, 1, 0, 2])
        initial_sample = {'1': mesh_1, '2': mesh_2, '3': None}
        merged = Mesh(vertices, indices, normals)
        cmd.new_name = 'merged'
        self.model_mock.return_value.sample = {'3':None, 'merged': merged}

        cmd.undo()
        sample = self.model_mock.return_value.sample
        self.assertEqual(list(sample.keys()), list(initial_sample.keys()))


if __name__ == '__main__':
    unittest.main()
