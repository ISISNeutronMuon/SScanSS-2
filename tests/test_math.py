import math
import unittest
import numpy as np
from sscanss.core.math import (Vector, Vector2, Vector3, Vector4, Matrix, Matrix33, Matrix44, Plane,
                               angle_axis_to_matrix, xyz_eulers_from_matrix,  matrix_from_xyz_eulers,
                               Quaternion, QuaternionVectorPair, rigid_transform, find_3d_correspondence)


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
        self.assertTrue(isinstance(v, Vector3))

        v = np.ones(3) - Vector3([-1, 0, 1])
        np.testing.assert_array_almost_equal(v, [2, 1, 0], decimal=5)
        self.assertTrue(isinstance(v, Vector3))

        a = np.ones((4, 3)) - Vector3([1, 1, 1]) * 3
        np.testing.assert_array_almost_equal(a, np.ones((4, 3)) * -2, decimal=5)
        self.assertTrue(isinstance(a, np.ndarray))

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
        m1 = m.inverse() @ m  # matrix multiplication
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
        result = m @ v  # matrix vector multiplication

        np.testing.assert_array_almost_equal(m.transpose(), [[1, 5], [2, 6], [3, 7], [4, 8]])
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

        self.assertNotEqual(repr(m), str(m))

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

        self.assertNotEqual(repr(plane_2), str(plane_2))

    def testQuaternion(self):
        q = Quaternion.identity()
        q[1] = 1.
        self.assertAlmostEqual(q.x, 0., 5)
        self.assertAlmostEqual(q.y, 1., 5)
        self.assertAlmostEqual(q.z, 0., 5)
        self.assertAlmostEqual(q.w, 1., 5)
        q.axis = [1., 1., 1.]
        np.testing.assert_array_almost_equal(q.axis, [1., 1., 1.], decimal=5)

        q.x = 1.
        q.y = q[0]
        q.z = 1.
        q.w = 0.
        np.testing.assert_array_almost_equal(q, [1., 1., 1., 0.], decimal=5)
        q = q.normalize()
        np.testing.assert_array_almost_equal(q, [0.57735, 0.57735, 0.57735, 0.], decimal=5)

        matrix = Matrix33([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        q = Quaternion.fromMatrix(matrix)
        np.testing.assert_array_almost_equal([0.7071067, 0., 0.], q.axis, decimal=5)
        np.testing.assert_array_almost_equal(matrix, q.toMatrix(), decimal=5)

        matrix = Matrix33([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        q = Quaternion.fromMatrix(matrix)
        np.testing.assert_array_almost_equal([0., 0., 1.], q.axis, decimal=5)
        np.testing.assert_array_almost_equal(matrix, q.toMatrix(), decimal=5)

        matrix = Matrix33([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        q = Quaternion.fromMatrix(matrix)
        np.testing.assert_array_almost_equal([0., 0., -0.7071067], q.axis, decimal=5)
        np.testing.assert_array_almost_equal(matrix, q.toMatrix(), decimal=5)

        matrix = Matrix33([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        q = Quaternion.fromMatrix(matrix)
        np.testing.assert_array_almost_equal([0.5, 0.5, 0.5], q.axis, decimal=5)
        np.testing.assert_array_almost_equal(matrix, q.toMatrix(), decimal=5)

        data = [-1, -1, -1, 1] * np.array(q)
        np.testing.assert_array_almost_equal(q.conjugate(), data, decimal=5)

        axis, angle = q.toAxisAngle()
        self.assertAlmostEqual(angle, 2.0943951, 5)
        np.testing.assert_array_almost_equal(axis, [0.57735, 0.57735, 0.57735], decimal=5)

        axis, angle = Quaternion().toAxisAngle()
        self.assertAlmostEqual(angle, np.pi, 5)
        np.testing.assert_array_almost_equal(axis, [0., 0., 0.], decimal=5)
        array = np.array(Quaternion().normalize())
        np.testing.assert_array_almost_equal(array, [0., 0., 0., 0.], decimal=5)

        qu = Quaternion.fromAxisAngle(Vector3([1., 0., 0.]), 0.)
        axis, angle = qu.toAxisAngle()
        self.assertAlmostEqual(angle, 0.0, 5)
        np.testing.assert_array_almost_equal(axis, [0., 0., 0.], decimal=5)

        mm = Matrix33([[-0.2128074, 0.5013429, 0.8386706],
                       [0.9463776, -0.1077663, 0.3045583],
                       [0.2430686, 0.8585113, -0.4515262]])

        qq = Quaternion.fromMatrix(mm)
        p1 = qq.rotate([1, 2, 3])
        p2 = mm @ Vector3([1, 2, 3])
        np.testing.assert_array_almost_equal(p1, p2, decimal=5)
        np.testing.assert_array_almost_equal(mm, qq.toMatrix(), decimal=5)

        # test dot product
        self.assertAlmostEqual(q | qq, 0.9544055, 5)

        q = Quaternion(0., 1., 1., 1.)
        np.testing.assert_array_almost_equal(q.inverse(), [-0.57735, -0.57735, -0.57735, 0.], decimal=5)

        self.assertNotEqual(repr(qq), str(qq))

    def testQuaternionVectorPair(self):
        qv = QuaternionVectorPair.identity()
        matrix = Matrix44([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        np.testing.assert_array_almost_equal(qv.toMatrix(), matrix, decimal=5)

        matrix = Matrix44([[0, 0, 1, -1], [1, 0, 0, 3], [0, 1, 0, -8], [0., 0., 0., 1.]])
        qv = QuaternionVectorPair.fromMatrix(matrix)
        np.testing.assert_array_almost_equal(qv.vector, matrix[0:3, 3], decimal=5)
        np.testing.assert_array_almost_equal(qv.quaternion, [0.5, 0.5, 0.5, 0.5], decimal=5)
        np.testing.assert_array_almost_equal(qv.toMatrix(), matrix, decimal=5)

        qv_1 = qv.inverse() * qv
        np.testing.assert_array_almost_equal(qv_1.toMatrix(), np.identity(4), decimal=5)

        qv *= qv.inverse()
        np.testing.assert_array_almost_equal(qv.toMatrix(), np.identity(4), decimal=5)
        self.assertRaises(ValueError, lambda: qv * [0, 1, 2])

        self.assertNotEqual(repr(qv), str(qv))

    def testRigidTransform(self):
        pa = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        pb = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])
        expected_matrix = [[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]
        result = rigid_transform(pa, pb)
        np.testing.assert_array_almost_equal(result.matrix, expected_matrix, decimal=5)
        np.testing.assert_array_almost_equal([0.0, 0.0, 0.0, 0.0], result.error, decimal=5)
        np.testing.assert_array_almost_equal([0.0, 0.0, 0.0, 0.0], result.total, decimal=5)
        da = [[1., 1., 0.], [1., 1., 0.], [1.41421356, 1.41421356, 0.], [1.41421356, 1.41421356, 0.],
              [1., 1., 0.],   [1., 1., 0.]]
        np.testing.assert_array_almost_equal([0.0, 0.0, 0.0, 0.0], result.total, decimal=5)
        np.testing.assert_array_almost_equal(result.distance_analysis, da, decimal=5)

        pm = np.array([[155.771,  -476.153,  216.733],
                      [198.648, -466.08, 215.963],
                      [178.126, -441.525,  219.311],
                      [145.042, -479.553,  426.429],
                      [176.456, -490.731,  423.377],
                      [189.364, -466.381,  422.838],
                      [165.517, -448.352,  429.23],
                      [152.542, -507.876,  659.803],
                      [186.595, -496.875,  657.654],
                      [174.927, -470.3,  673.588],
                      [146.125, -476.536,  662.97]])

        index = [1, 2, 6, 5, 9]
        pa = np.array([[41.766, -19.863, -200.514],
                      [62.934, 4.292, -197.259],
                      [54.492, 15.807, 12.599],
                      [40.103, -10.687, 6.181],
                      [34.338, 3.036, 257.119]])
        result = rigid_transform(pm[index, :], pa)
        expected_matrix = [[0.152546669,  0.988295974,  0.000762508333, 471.929323],
                           [-0.988296268,  0.152546623, 0.000117696341, 247.612000],
                           [0., -0.000771538325, 0.999999702, -416.914455],
                           [0.,  0.,  0.,  1.]]
        np.testing.assert_array_almost_equal(result.matrix, expected_matrix, decimal=5)
        np.testing.assert_array_almost_equal([0.11071, 0.05361, 0.16207, 0.16251, 0.08931], result.error, decimal=5)

        index = [1, 2, 5, 6]
        pa = np.array([[19.248, 3.077, -207.266],
                      [43.908, 22.926, -201.364],
                      [-86.812, 151.35, -109.326],
                      [-67.374, 175.067, -108.072]])
        result = rigid_transform(pm[index, :], pa)
        expected_matrix = [[-0.389543404,  0.753641435, -0.529415266,  562.227328],
                           [-0.720206841,  0.109031616,  0.685138098,  49.0120861],
                           [0.574071461, 0.648179523, 0.500305170, -127.227984],
                           [0., 0.,  0.,  1.]]
        np.testing.assert_array_almost_equal(result.matrix, expected_matrix, decimal=5)
        np.testing.assert_array_almost_equal([0.02544, 0.13233, 0.15911, 0.06540], result.error, decimal=5)

    def testFindCorrespondence(self):
        pa = np.array([[155.771, -476.153, 216.733],
                      [198.648, -466.08, 215.963],
                      [178.126, -441.525, 219.311],
                      [145.042, -479.553, 426.429],
                      [176.456, -490.731, 423.377],
                      [189.364, -466.381, 422.838],
                      [165.517, -448.352, 429.23],
                      [152.542, -507.876, 659.803],
                      [186.595, -496.875, 657.654],
                      [174.927, -470.3, 673.588],
                      [146.125, -476.536, 662.97]])

        pb = np.array([[41.766, -19.863, -200.514],
                      [62.934, 4.292, -197.259],
                      [54.492, 15.807, 12.599],
                      [40.103, -10.687, 6.181],
                      [34.338, 3.036, 257.119]])

        corr = find_3d_correspondence(pa, pb)
        np.testing.assert_array_equal(corr, [1, 2, 6, 5, 9])

        pb = np.array([[-86.812, 151.35, -109.326],
                      [43.908, 22.926, -201.364],
                      [19.248, 3.077, -207.266],
                      [-67.374, 175.067, -108.072]])
        corr = find_3d_correspondence(pa, pb)
        np.testing.assert_array_equal(corr, [5, 2, 1, 6])

        pb = np.array([[-67.374, 175.067, -108.072],
                      [-67.374, 175.067, -108.072]])
        with self.assertRaises(ValueError):
            _ = find_3d_correspondence(pa, pb)


if __name__ == '__main__':
    unittest.main()
