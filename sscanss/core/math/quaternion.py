"""
Classes for Quaternion and Quaternion-Vector objects
"""
import math
from .constants import VECTOR_EPS
from .vector import Vector3, Vector4
from .matrix import Matrix33, Matrix44


class Quaternion:
    """Creates a Quaternion object in the form w + xi + yj + zk.

    :param w: real part
    :type w: float
    :param x: coefficient of imaginary part i
    :type x: float
    :param y: coefficient of imaginary part j
    :type y: float
    :param z: coefficient of imaginary part k
    :type z: float
    """

    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self._data = Vector4([x, y, z, w])

    def __array__(self, _dtype=None):
        return self._data[:]

    @classmethod
    def identity(cls):
        """Creates unit quaternion

        :return: unit quaternion
        :rtype: Quaternion
        """
        return cls(1.0, 0.0, 0.0, 0.0)

    @property
    def x(self):
        """Gets and sets the coefficient of imaginary part i of the quaternion

        :return: coefficient of imaginary part i
        :rtype: float
        """
        return self._data.x

    @x.setter
    def x(self, value):
        self._data.x = value

    @property
    def y(self):
        """Gets and sets the coefficient of imaginary part j of the quaternion

        :return: coefficient of imaginary part j
        :rtype: float
        """
        return self._data.y

    @y.setter
    def y(self, value):
        self._data.y = value

    @property
    def z(self):
        """Gets and sets the coefficient of imaginary part k of the quaternion

        :return: coefficient of imaginary part k
        :rtype: float
        """
        return self._data.z

    @z.setter
    def z(self, value):
        self._data.z = value

    @property
    def w(self):
        """Gets and sets the real part of the quaternion

        :return: real part
        :rtype: float
        """
        return self._data.w

    @w.setter
    def w(self, value):
        self._data.w = value

    @property
    def axis(self):
        """Gets and sets the coefficients of imaginary part [i, j, k] of the quaternion

        :rtype axis: Vector3
        """
        return Vector3([self.x, self.y, self.z])

    @axis.setter
    def axis(self, axis):
        self._data.xyz = axis

    def conjugate(self):
        """Computes quaternion conjugate

        :return: conjugate of quaternion
        :rtype: Quaternion
        """
        return self.__class__(self.w, -self.x, -self.y, -self.z)

    @property
    def length(self):
        """Computes length/magnitude of quaternion

        :return: length of quaternion
        :rtype: float
        """
        return self._data.length

    def toMatrix(self):
        """Converts quaternion into a rotation matrix

        :return: rotation matrix
        :rtype: Matrix33
        """
        twoxx = 2 * self.x * self.x
        twoyy = 2 * self.y * self.y
        twozz = 2 * self.z * self.z

        twowx = 2 * self.w * self.x
        twowy = 2 * self.w * self.y
        twowz = 2 * self.w * self.z

        twoxy = 2 * self.x * self.y
        twoxz = 2 * self.x * self.z

        twoyz = 2 * self.y * self.z

        return Matrix33(
            [
                [1 - twoyy - twozz, twoxy - twowz, twoxz + twowy],
                [twoxy + twowz, 1 - twoxx - twozz, twoyz - twowx],
                [twoxz - twowy, twoyz + twowx, 1 - twoxx - twoyy],
            ]
        )

    def toAxisAngle(self):
        """Converts quaternion into the angle axis representation

        :return: rotation axis and angle in radians
        :rtype: Union[Vector3, float]
        """
        angle = 2 * math.acos(self.w)
        s = math.sqrt(1 - self.w * self.w)
        if angle < VECTOR_EPS:
            axis = Vector3()
        else:
            axis = Vector3([self.x, self.y, self.z]) / s

        return axis, angle

    def inverse(self):
        """Computes inverse of the quaternion

        :return: inverse of quaternion
        :rtype: Quaternion
        """
        return self.conjugate().normalize()

    def normalize(self):
        """Normalizes quaternion

        :return: normalized quaternion
        :rtype: Quaternion
        """
        length = self.length
        if length > VECTOR_EPS:
            n = self._data.normalized
            return self.__class__(n.w, n.x, n.y, n.z)

        return self.__class__()

    def dot(self, q):
        """Computes quaternion dot product with another quaternion

        :param q: quaternion
        :type q: Quaternion
        :return: dot product
        :rtype: float
        """
        return self._data.dot(q[:])

    def rotate(self, point):
        """Rotates a 3D point with the quaternion

        :param point: 3D point
        :type point: Vector3
        :return: rotated point
        :rtype: Vector3
        """
        p = self.__class__(x=point[0], y=point[1], z=point[2])
        q_inv = self.inverse()

        rotated = self * p * q_inv

        return rotated.axis

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    @classmethod
    def fromAxisAngle(cls, axis, angle):
        """Converts angle axis representation to quaternion

        :param axis: axis of rotation
        :type axis: Vector3
        :param angle: angle of rotation in radians
        :type angle: float
        :return: quaternion
        :rtype: Quaternion
        """
        w = math.cos(angle / 2)
        x, y, z = axis.normalized * math.sin(angle / 2)

        return cls(w, x, y, z)

    @classmethod
    def fromMatrix(cls, matrix):
        """Converts rotation matrix to quaternion

        :param matrix: rotation matrix
        :type matrix: Matrix33
        :return: quaternion
        :rtype: Quaternion
        """
        if matrix.m33 < VECTOR_EPS:
            if matrix.m11 > matrix.m22:
                t = 1 + matrix.m11 - matrix.m22 - matrix.m33
                q = [
                    matrix.m32 - matrix.m23,
                    t,
                    matrix.m12 + matrix.m21,
                    matrix.m13 + matrix.m31,
                ]
            else:
                t = 1 - matrix.m11 + matrix.m22 - matrix.m33
                q = [
                    matrix.m13 - matrix.m31,
                    matrix.m12 + matrix.m21,
                    t,
                    matrix.m23 + matrix.m32,
                ]
        else:
            if matrix.m11 < -matrix.m22:
                t = 1 - matrix.m11 - matrix.m22 + matrix.m33
                q = [
                    matrix.m21 - matrix.m12,
                    matrix.m13 + matrix.m31,
                    matrix.m23 + matrix.m32,
                    t,
                ]
            else:
                t = 1 + matrix.m11 + matrix.m22 + matrix.m33
                q = [
                    t,
                    matrix.m32 - matrix.m23,
                    matrix.m13 - matrix.m31,
                    matrix.m21 - matrix.m12,
                ]

        q = Vector4(q) * 0.5 / math.sqrt(t)
        return cls(*q)

    def __str__(self):
        return "[{} <{} {} {}>]".format(self.w, *self.axis)

    def __mul__(self, other):
        w1 = self.w
        w2 = other.w

        v1 = self.axis
        v2 = other.axis

        w = w1 * w2 - (v1 | v2)
        v = w1 * v2 + w2 * v1 + (v1 ^ v2)

        return self.__class__(w, *v)

    def __or__(self, other):
        return self.dot(other)


class QuaternionVectorPair:
    """Creates a Quaternion-Vector object. The Quaternion-Vector pair
    is equivalent to a Homogeneous transformation matrix

    :param q: quaternion part
    :type q: Quaternion
    :param v: vector part
    :type v: Vector3
    """

    def __init__(self, q, v):
        self.quaternion = q
        self.vector = v

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("cannot multiply {} with QuaternionVectorPair".format(type(other)))

        q = self.quaternion * other.quaternion
        v = self.quaternion.rotate(other.vector) + self.vector

        return self.__class__(q, v)

    def __imul__(self, other):
        temp = self.__mul__(other)
        self.quaternion = temp.quaternion
        self.vector = temp.vector
        return self

    def inverse(self):
        """Computes inverse of Quaternion-Vector pair

        :return: inverse of Quaternion-Vector pair
        :rtype: QuaternionVectorPair
        """
        q = self.quaternion.inverse()
        v = -q.rotate(self.vector)

        return self.__class__(q, v)

    def toMatrix(self):
        """Converts Quaternion-Vector pair to homogeneous transformation matrix

        :return: homogeneous transformation matrix
        :rtype: matrix44
        """
        m = Matrix44.identity()
        m[0:3, 0:3] = self.quaternion.toMatrix()
        m[0:3, 3] = self.vector

        return m

    @classmethod
    def fromMatrix(cls, matrix):
        """Creates Quaternion-Vector pair from homogeneous transformation matrix

        :param matrix: homogeneous transformation matrix
        :type matrix: Matrix44
        :return: Quaternion-Vector pair
        :rtype: QuaternionVectorPair
        """
        q = Quaternion.fromMatrix(matrix)
        v = Vector3(matrix[0:3, 3])

        return cls(q, v)

    @classmethod
    def identity(cls):
        """Creates unit Quaternion-Vector pair which consist of a unit quaternion and a zero vector

        :return: unit Quaternion-Vector pair
        :rtype: QuaternionVectorPair
        """
        q = Quaternion.identity()
        v = Vector3()

        return cls(q, v)

    def __str__(self):
        return "Quaternion: {}, Vector: {}".format(self.quaternion, self.vector)
