import math
import unittest
import numpy as np
from sscanss.core.math import (Vector, Vector2, Vector3, Vector4, Matrix, Matrix33, Matrix44, Plane,
                               angle_axis_to_matrix, xyz_eulers_from_matrix,  matrix_from_xyz_eulers)


class TestMath(unittest.TestCase):
    def testAngleAxisToMatrix(self):
        axis = Vector3([1.0, 0.0, 0.0])
        expected = Matrix33.identity()
        result = angle_axis_to_matrix(0.0, axis)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        axis = Vector3([0.0, 1.0, 0.0])
        expected = Matrix33([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        result = angle_axis_to_matrix(math.radians(90), axis)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        axis = Vector3([0.0, 1.0, 1.0])
        expected = Matrix33([[0.7071068, -0.5000000, 0.5000000],
                            [0.5000000, 0.8535534, 0.1464466],
                            [-0.5000000, 0.1464466, 0.8535534]])
        result = angle_axis_to_matrix(math.radians(45), axis)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def testMatrixFromXYZEulers(self):
        eulers = Vector3([0.0, 0.0, 0.0])
        expected = Matrix33.identity()
        result = matrix_from_xyz_eulers(eulers)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        eulers = Vector3(np.radians([0.0, 0.0, 90.0]))
        expected = Matrix33([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        result = matrix_from_xyz_eulers(eulers)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        eulers = Vector3(np.radians([30.0, 10.0, 60.0]))
        expected = Matrix33([[0.4924039, -0.8528686, 0.1736482],
                            [0.7934120, 0.3578208, -0.4924039],
                            [0.3578208, 0.3802361, 0.8528686]])
        result = matrix_from_xyz_eulers(eulers)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def testXYZEulersFromMatrix(self):
        matrix = Matrix33.identity()
        expected = Vector3([0.0, 0.0, 0.0])
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        matrix = Matrix33([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        expected = Vector3(np.radians([0.0, 0.0, 90.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        matrix = Matrix33([[0.4924039, -0.8528686, 0.1736482],
                          [0.7934120, 0.3578208, -0.4924039],
                          [0.3578208, 0.3802361, 0.8528686]])
        expected = Vector3(np.radians([30.0, 10.0, 60.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        matrix = Matrix33([[-0.0000000, -0.0000000, -1.0000000],
                          [-0.3420202, 0.9396926, -0.0000000],
                          [0.9396926, 0.3420202, -0.0000000]])
        expected = Vector3(np.radians([20.0, -90.0, 0.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        matrix = Matrix33([[0.0000000, 0.0000000, 1.0000000],
                          [0.6427876, 0.7660444, -0.0000000],
                          [-0.7660444, 0.6427876, 0.0000000]])
        expected = Vector3(np.radians([40.0, 90, 0.0]))
        result = xyz_eulers_from_matrix(matrix)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def testVector(self):
        self.assertRaises(ValueError, Vector, -1)
        self.assertRaises(ValueError, Vector, 2, {'values': [2, 3, 4]})

        v = Vector(10)
        self.assertEqual(v.size, 10)
        self.assertEqual(len(v), 10)
        self.assertEqual(v[9], 0)
        v[9] = 7
        self.assertEqual(v[9], 7)

        v = Vector.create(2, [1, 2])
        self.assertEqual(v.size, 2)
        self.assertEqual(v.x, 1)
        self.assertTrue(isinstance(v, Vector2))

        v = Vector.create(3, [1, 2, 3])
        self.assertEqual(v.size, 3)
        self.assertEqual(v.z, 3)
        self.assertTrue(isinstance(v, Vector3))

        v = Vector.create(4, [1, 2, 3, 4])
        self.assertEqual(v.size, 4)
        self.assertEqual(v.w, 4)
        self.assertTrue(isinstance(v, Vector4))

        v = Vector.create(5, [1, 2, 3, 4, 5])
        self.assertTrue(isinstance(v, Vector))

        self.assertRaises(ValueError, lambda: Vector3() + Vector4())
        self.assertRaises(ValueError, lambda: Vector3() - Vector4())
        self.assertRaises(ValueError, lambda: Vector3() * Vector4())
        self.assertRaises(ValueError, lambda: Vector3() / Vector4())

    def testVector2(self):
        v = Vector2()
        self.assertEqual(v.size, 2)
        self.assertEqual(v.x, 0)
        v.y = 4
        self.assertEqual(v.y, 4)
        vec = Vector2([1, 2])
        np.testing.assert_array_equal(vec.cross([4, 5]), [0, 0, -3])

    def testVector3(self):
        v = Vector3([4., 2., 3.])

        v.x /= 4
        self.assertAlmostEqual(v.x, 1.0, 5)
        self.assertAlmostEqual(v.y, 2.0, 5)
        self.assertAlmostEqual(v.z, 3.0, 5)
        with self.assertRaises(AttributeError):
            v.w = 5

        np.testing.assert_array_almost_equal(v, [1., 2., 3.], decimal=5)
        self.assertAlmostEqual(v.length, 3.7416573867739413, 5)
        v1 = v.normalized
        self.assertAlmostEqual(v1.length, 1.0, 5)
        v.normalize()
        np.testing.assert_array_almost_equal(v.xyz, [0.26726124, 0.53452248, 0.80178373], decimal=5)
        self.assertAlmostEqual(v1.length, 1.0, 5)

        vec = Vector3([1, 2, 3])
        self.assertEqual(vec.dot([1, 2, 3]), 14)
        self.assertEqual(vec | vec, 14)

        vec = Vector3([1, 2, 0])
        np.testing.assert_array_equal(vec.cross([4, 5, 0]), [0, 0, -3])
        vec = Vector3([1, 0, 0])
        np.testing.assert_array_equal(vec.cross([0, 1, 0]), [0, 0, 1])
        np.testing.assert_array_equal((vec ^ [0, 1, 0]), [0, 0, 1])

        v1 = Vector3([1, 2, 3])
        v2 = Vector3([4, 5, 6])
        v3 = v1 + v2
        np.testing.assert_array_equal(v3, [5, 7, 9])
        v3 = 2 + v2
        np.testing.assert_array_equal(v3, [6, 7, 8])
        v3 = v1 + 2
        np.testing.assert_array_equal(v3, [3, 4, 5])
        v3 = v1 - v2
        np.testing.assert_array_equal(v3, [-3, -3, -3])
        v3 = 2 - v2
        np.testing.assert_array_equal(v3, [-2, -3, -4])
        v3 = v1 - 2
        np.testing.assert_array_equal(v3, [-1, 0, 1])
        v3 = v1 * v2
        np.testing.assert_array_equal(v3, [4, 10, 18])
        v3 = 2 * v2
        np.testing.assert_array_equal(v3, [8, 10, 12])
        v3 = v1 * 2
        np.testing.assert_array_equal(v3, [2, 4, 6])
        v3 = v1 / v2
        np.testing.assert_array_almost_equal(v3, [0.25, 0.4, 0.5], decimal=5)
        v3 = 2 / v2
        np.testing.assert_array_almost_equal(v3, [0.5, 0.4, 0.333333], decimal=5)
        v3 = v1 / 2
        np.testing.assert_array_almost_equal(v3, [0.5, 1.0, 1.5], decimal=5)

        v = Vector3()
        v += v1
        np.testing.assert_array_equal(v, [1, 2, 3])
        v -= v1
        np.testing.assert_array_equal(v, [0, 0, 0])
        v1 *= Vector3([2, 2, 2])
        np.testing.assert_array_equal(v1, [2, 4, 6])
        v1 /= Vector3([0.5, 0.5, 0.5])
        np.testing.assert_array_equal(v1, [4, 8, 12])

        v = np.ones(3) + Vector3([-1, 0, 1])
        np.testing.assert_array_almost_equal(v, [0, 1, 2], decimal=5)

        v = np.ones(3) - Vector3([-1, 0, 1])
        np.testing.assert_array_almost_equal(v, [2, 1, 0], decimal=5)

    def testVector4(self):
        v = Vector4([1., 2., 3., 10.])
        self.assertAlmostEqual(v.w, 10.0, 5)
        np.testing.assert_array_almost_equal(v.xyzw, [1., 2., 3., 10.], decimal=5)
        np.testing.assert_array_almost_equal(v / v.w, [0.1, 0.2, 0.3, 1.0], decimal=5)

    def testMatrix(self):
        self.assertRaises(ValueError, Matrix, 2, -1)
        self.assertRaises(ValueError, Matrix, 0, 2)
        self.assertRaises(ValueError, Matrix, 2, 2, {'values': [[1, 2], [3]]})

        m = Matrix(2, 2, dtype=int)
        self.assertEqual(m[0, 0], 0)
        m[1, 1] = 5
        self.assertEqual(m[1, 1], 5)

        m = Matrix(2, 2, [[1., 2.], [3., 4.]])
        np.testing.assert_array_almost_equal(m.inverse(), [[-2., 1.], [1.5, -0.5]], decimal=5)
        m1 = m.inverse() * m  # matrix multiplication
        np.testing.assert_array_almost_equal(m1, [[1, 0], [0, 1]])
        a = np.array([[1, 2], [3, 4]])
        m1 = m * a  # element wise multiplication
        np.testing.assert_array_almost_equal(m1, [[1, 4], [9, 16]], decimal=5)
        m1 = m * 2
        np.testing.assert_array_almost_equal(m1, [[2., 4.], [6., 8.]], decimal=5)
        m2 = 2 * m
        np.testing.assert_array_almost_equal(m1, m2, decimal=5)
        m1 = m + 2
        np.testing.assert_array_almost_equal(m1, [[3., 4.], [5., 6.]], decimal=5)
        m1 = m - 2
        np.testing.assert_array_almost_equal(m1, [[-1., 0.], [1., 2.]], decimal=5)

        m = Matrix.create(2, 4, [[1, 2, 3, 4], [5, 6, 7, 8]])
        v = Vector.create(4, [1, 2, 3, 4])
        result = m * v  # matrix vector multiplication
        self.assertTrue(isinstance(result, Vector))
        np.testing.assert_array_equal(result, [30, 70])

        self.assertRaises(ValueError, lambda: Matrix33() + Matrix44())
        self.assertRaises(ValueError, lambda: Matrix33() - Matrix44())
        self.assertRaises(ValueError, lambda: Matrix33() * Matrix44())
        self.assertRaises(ValueError, lambda: Matrix33() * Vector4())

    def testMatrix33(self):
        m = Matrix33([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(m, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(m.transpose(), [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertFalse(m.invertible)
        self.assertAlmostEqual(m.determinant, 0.0, 5)

        m = Matrix33.identity()
        np.testing.assert_array_almost_equal(m, np.eye(3), decimal=5)

        m = Matrix33.ones()
        np.testing.assert_array_almost_equal(m, np.ones((3, 3)), decimal=5)
        m.m11 = 5
        np.testing.assert_array_almost_equal(m.r1, [5, 1, 1], decimal=5)
        m.c2 = [8, 9, -7]
        np.testing.assert_array_almost_equal(m.r3, [1, -7, 1], decimal=5)

        m = Matrix33.fromTranslation([2, 3])
        expected = [[1, 0, 2], [0, 1, 3], [0, 0, 1]]
        np.testing.assert_array_almost_equal(m, expected, decimal=5)

        self.assertRaises(AttributeError, lambda: m.r4)
        with self.assertRaises(AttributeError):
            m.m14 = 50

    def testMatrix44(self):
        m = Matrix44()
        np.testing.assert_array_almost_equal(m, np.zeros((4, 4)), decimal=5)

        m = Matrix44.identity()
        np.testing.assert_array_almost_equal(m, np.eye(4), decimal=5)

        m = Matrix44.ones()
        np.testing.assert_array_almost_equal(m, np.ones((4, 4)), decimal=5)
        m.m13 = 5
        np.testing.assert_array_almost_equal(m.r1, [1, 1, 5, 1], decimal=5)
        m.r2 = [8, 9, -7, -3]
        np.testing.assert_array_almost_equal(m.c3, [5, -7, 1, 1], decimal=5)

        m = Matrix44.fromTranslation([2, 3, 4])
        expected = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(m, expected, decimal=5)

        m = Matrix44.ones() + Matrix44.identity()
        expected = [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]
        np.testing.assert_array_almost_equal(m, expected, decimal=5)

        m = np.ones((4, 4)) + Matrix44.identity()
        expected = [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]
        np.testing.assert_array_almost_equal(m, expected, decimal=5)

        m = Matrix44.ones() - Matrix44.identity()
        expected = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        np.testing.assert_array_almost_equal(m, expected, decimal=5)

        m = np.ones((4, 4)) - Matrix44.identity()
        expected = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        np.testing.assert_array_almost_equal(m, expected, decimal=5)

    def testPlane(self):
        normal = np.array([1., 0., 0.])
        point = np.array([0., 0., 0.])
        plane = Plane(normal, point)
        np.testing.assert_array_almost_equal(plane.normal, normal, decimal=5)
        np.testing.assert_array_almost_equal(plane.point, point, decimal=5)

        point_2 = np.array([0., 1., 0.])
        point_3 = np.array([0., 1., 1.])
        plane_2 = Plane.fromPlanarPoints(point, point_2, point_3)
        np.testing.assert_array_almost_equal(plane_2.normal, normal, decimal=5)
        np.testing.assert_array_almost_equal(plane_2.point, point, decimal=5)

        # bad normal
        self.assertRaises(ValueError, lambda: Plane(point, point))
        self.assertRaises(ValueError, lambda: Plane.fromCoefficient(0, 0, 0, 0))
        self.assertRaises(ValueError, lambda: Plane.fromPlanarPoints(point, point_2, point_2))

        self.assertAlmostEqual(plane.distanceFromOrigin(), 0.0, 5)
        plane = Plane(normal, normal)
        self.assertAlmostEqual(plane.distanceFromOrigin(), 1.0, 5)

        normal = np.array([1., 1., 0.])
        plane = Plane(normal, point)  # normal vector should be normalize
        np.testing.assert_array_almost_equal(plane.normal, Vector3(normal).normalized, decimal=5)
        np.testing.assert_array_almost_equal(plane.point, point, decimal=5)

        plane_2 = Plane.fromCoefficient(1, 1, 0, 0)
        np.testing.assert_array_almost_equal(plane.normal, plane_2.normal, decimal=5)
        np.testing.assert_array_almost_equal(plane.point, plane_2.point, decimal=5)


if __name__ == '__main__':
    unittest.main()
