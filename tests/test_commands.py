import unittest
import unittest.mock as mock
import numpy as np
from sscanss.core.mesh import Mesh
import sscanss.ui.windows.main.view as view
from sscanss.ui.windows.main.presenter import MainWindowPresenter
from sscanss.ui.commands import RotateSample, TranslateSample


class TestTransformCommands(unittest.TestCase):
    @mock.patch('sscanss.ui.windows.main.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(view.MainWindow)
        self.model_mock = model_mock
        self.presenter = MainWindowPresenter(self.view_mock)

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        self.mesh_1 = Mesh(vertices, indices, normals)

        vertices = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([1, 0, 2])
        self.mesh_2 = Mesh(vertices, indices, normals)

    def testRotateSampleCommand(self):
        self.model_mock.return_value.sample = {'1': self.mesh_1, '2': self.mesh_2}

        # Command to rotate sample '1'
        angles = [0, 90, 0]
        cmd = RotateSample(angles, '1', self.presenter)
        cmd.redo()

        # Check that angles are converted to radians
        np.testing.assert_array_almost_equal(cmd.angles, np.radians(angles), decimal=5)

        expected_vertices = np.array([[3, 2, -1], [6, 5, -4], [9, 8, -7]])
        expected_normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        result = self.model_mock.return_value.sample
        # Check that redo rotates vertices, normals but not the indices of sample '1'
        np.testing.assert_array_almost_equal(result['1'].vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)

        # Check that redo does not rotate sample '2'
        np.testing.assert_array_almost_equal(result['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)

        cmd.undo()
        result = self.model_mock.return_value.sample
        # Check that undo reverses the rotation on sample '1'
        np.testing.assert_array_almost_equal(result['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)

        # Check that undo does not touch sample '2'
        np.testing.assert_array_almost_equal(result['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)

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
        result = self.model_mock.return_value.sample
        # Check that redo rotates vertices, normals but not the indices of all samples'
        np.testing.assert_array_almost_equal(result['1'].vertices, expected_vertices_1, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, expected_normals_1, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(result['2'].vertices, expected_vertices_2, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, expected_normals_2, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)

        cmd.undo()
        result = self.model_mock.return_value.sample
        # Check that undo reverses the rotation on all samples
        np.testing.assert_array_almost_equal(result['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(result['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)

    def testTranslateSampleCommand(self):
        self.model_mock.return_value.sample = {'1': self.mesh_1, '2': self.mesh_2}

        # Command to translate sample '2'
        offset = [10, -5, 3]
        cmd = TranslateSample(offset, '2', self.presenter)
        cmd.redo()

        expected_vertices = np.array([[17, 3, 12], [14, 0, 9], [11, -3, 6]])
        result = self.model_mock.return_value.sample
        # Check that redo translates vertices but not the normals and indices of sample '2'
        np.testing.assert_array_almost_equal(result['2'].vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)

        # Check that redo does not translate sample '2'
        np.testing.assert_array_almost_equal(result['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)

        cmd.undo()
        result = self.model_mock.return_value.sample
        # Check that undo reverses the translation on sample '2'
        np.testing.assert_array_almost_equal(result['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)

        # Check that undo does not touch sample '1'
        np.testing.assert_array_almost_equal(result['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)

        # Command to translate all the samples
        offset = [30, 60, 90]
        cmd = TranslateSample(offset, 'All', self.presenter)
        cmd.redo()

        expected_vertices_1 = np.array([[31, 62, 93], [34, 65, 96], [37, 68, 99]])
        expected_vertices_2 = np.array([[37, 68, 99], [34, 65, 96], [31, 62, 93]])
        result = self.model_mock.return_value.sample
        # Check that redo translates vertices, normals but not the indices of all samples'
        np.testing.assert_array_almost_equal(result['1'].vertices, expected_vertices_1, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(result['2'].vertices, expected_vertices_2, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)

        cmd.undo()
        result = self.model_mock.return_value.sample
        # Check that undo reverses the translation on all samples
        np.testing.assert_array_almost_equal(result['1'].vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['1'].normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(result['1'].indices, self.mesh_1.indices)
        np.testing.assert_array_almost_equal(result['2'].vertices, self.mesh_2.vertices, decimal=5)
        np.testing.assert_array_almost_equal(result['2'].normals, self.mesh_2.normals, decimal=5)
        np.testing.assert_array_equal(result['2'].indices, self.mesh_2.indices)


if __name__ == '__main__':
    unittest.main()
