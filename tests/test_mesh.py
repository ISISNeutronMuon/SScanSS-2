import unittest
import numpy as np
from sscanss.core.math import Vector3, matrix_from_xyz_eulers
from sscanss.core.mesh import Mesh


class TestMeshClass(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        self.mesh_1 = Mesh(vertices, indices, normals)

        vertices = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([1, 0, 2])
        self.mesh_2 = Mesh(vertices, indices, normals)

    def testComputeNormals(self):
        vertices = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        indices = np.array([1, 0, 2])
        mesh = Mesh(vertices, indices)

        expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

        # Check that correct normals are generated also vertices and indices are unchanged
        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, expected, decimal=5)
        np.testing.assert_array_equal(mesh.indices, indices)

    def testComputeBoundingBox(self):
        box = self.mesh_1.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([7, 8, 9]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([1, 2, 3]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([4., 5., 6.]), decimal=5)
        self.assertAlmostEqual(box.radius, 5.1961524, 5)

        box = self.mesh_2.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([7, 8, 9]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([1, 2, 3]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([4., 5., 6.]), decimal=5)
        self.assertAlmostEqual(box.radius, 5.1961524, 5)

    def testAppendAndSplit(self):
        self.mesh_1.append(self.mesh_2)

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([0, 1, 2, 4, 3, 5])

        np.testing.assert_array_almost_equal(self.mesh_1.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(self.mesh_1.normals, normals, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, indices)

        split_mesh = self.mesh_1.splitAt(3)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([0, 1, 2]))
        np.testing.assert_array_equal(split_mesh.indices, np.array([0, 1, 2]))
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_almost_equal(self.mesh_1.vertices, expected, decimal=5)
        expected = np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
        np.testing.assert_array_almost_equal(split_mesh.vertices, expected, decimal=5)
        expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_almost_equal(self.mesh_1.normals, expected, decimal=5)
        expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_almost_equal(split_mesh.normals, expected, decimal=5)

    def testTransform(self):
        angles = np.radians([30, 60, 90])
        matrix = matrix_from_xyz_eulers(Vector3(angles))
        self.mesh_1.rotate(matrix)

        expected_vertices = np.array([[1.59807621, -0.75, 3.29903811],
                                       [2.69615242, -0.20096189, 8.34807621],
                                       [3.79422863, 0.34807621, 13.39711432]])
        expected_normals = np.array([[0.866025, -0.25, 0.433013], [-0.5, -0.433013, 0.75], [0, 0.866025, 0.5]])

        np.testing.assert_array_almost_equal(self.mesh_1.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(self.mesh_1.normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([0, 1, 2]))

        offset = Vector3([10, -11, 12])
        self.mesh_1.translate(offset)
        expected_vertices = np.array([[11.59807621, -11.75, 15.29903811],
                                     [12.69615242, -11.20096189, 20.34807621],
                                     [13.79422863, -10.6519237, 25.39711432]])

        np.testing.assert_array_almost_equal(self.mesh_1.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(self.mesh_1.normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([0, 1, 2]))

        transform_matrix = np.eye(4, 4)
        transform_matrix[0:3, 0:3] = matrix.transpose()
        transform_matrix[0:3, 3] = -offset.dot(matrix)
        self.mesh_1.transform(transform_matrix)
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_almost_equal(self.mesh_1.vertices, expected, decimal=5)
        expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_almost_equal(self.mesh_1.normals, expected, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([0, 1, 2]))