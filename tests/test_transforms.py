import math
import unittest
import numpy as np
from sscanss.core.math import (Vector3, Matrix33, angle_axis_to_matrix,
                               xyz_eulers_from_matrix,  matrix_from_xyz_eulers)


class TestTransform(unittest.TestCase):
    def testAngleAxisToMatrix(self):
        axis = Vector3([1.0, 0.0, 0.0])
        expected = Matrix33.identity()
        result = angle_axis_to_matrix(0.0, axis)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        axis = Vector3([0.0, 1.0, 0.0])
        expected = Matrix33([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        result = angle_axis_to_matrix(math.radians(90), axis)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        axis = Vector3([0.0, 1.0, 1.0])
        expected = Matrix33([[0.7071068, -0.5000000, 0.5000000],
                            [0.5000000, 0.8535534, 0.1464466],
                            [-0.5000000, 0.1464466, 0.8535534]])
        result = angle_axis_to_matrix(math.radians(45), axis)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

    def testMatrixFromXYZEulers(self):
        eulers = Vector3([0.0, 0.0, 0.0])
        expected = Matrix33.identity()
        result = matrix_from_xyz_eulers(eulers)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        eulers = Vector3(np.radians([0.0, 0.0, 90.0]))
        expected = Matrix33([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        result = matrix_from_xyz_eulers(eulers)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        eulers = Vector3(np.radians([30.0, 10.0, 60.0]))
        expected = Matrix33([[0.4924039, -0.8528686, 0.1736482],
                            [0.7934120, 0.3578208, -0.4924039],
                            [0.3578208, 0.3802361, 0.8528686]])
        result = matrix_from_xyz_eulers(eulers)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

    def testXYZEulersFromMatrix(self):
        matrix = Matrix33.identity()
        expected = Vector3([0.0, 0.0, 0.0])
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        matrix = Matrix33([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        expected = Vector3(np.radians([0.0, 0.0, 90.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        matrix = Matrix33([[0.4924039, -0.8528686, 0.1736482],
                          [0.7934120, 0.3578208, -0.4924039],
                          [0.3578208, 0.3802361, 0.8528686]])
        expected = Vector3(np.radians([30.0, 10.0, 60.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        matrix = Matrix33([[-0.0000000, -0.0000000, -1.0000000],
                          [-0.3420202, 0.9396926, -0.0000000],
                          [0.9396926, 0.3420202, -0.0000000]])
        expected = Vector3(np.radians([20.0, -90.0, 0.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)

        matrix = Matrix33([[0.0000000, 0.0000000, 1.0000000],
                          [0.6427876, 0.7660444, -0.0000000],
                          [-0.7660444, 0.6427876, 0.0000000]])
        expected = Vector3(np.radians([40.0, 90, 0.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result.toArray(), expected.toArray(), decimal=5)


if __name__ == '__main__':
    unittest.main()
