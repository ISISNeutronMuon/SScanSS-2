import math
import numpy as np
from .vector import Vector3
from .matrix import Matrix33, Matrix44
from .algorithm import clamp

eps = 0.000001


def angle_axis_to_matrix(angle, axis):
    """ Converts rotation in angle/axis representation to a matrix

    :param angle: angle to rotate by in radians
    :type angle: float
    :param axis: axis to rotate around
    :type axis: Vector3
    :return: rotation matrix
    :rtype: Matrix33
    """
    _axis = axis.normalized
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    return Matrix33(
        [
            [
                t * _axis.x * _axis.x + c,
                t * _axis.x * _axis.y - _axis.z * s,
                t * _axis.x * _axis.z + _axis.y * s,
            ],
            [
                t * _axis.x * _axis.y + _axis.z * s,
                t * _axis.y * _axis.y + c,
                t * _axis.y * _axis.z - _axis.x * s,
            ],
            [
                t * _axis.x * _axis.z - _axis.y * s,
                t * _axis.y * _axis.z + _axis.x * s,
                t * _axis.z * _axis.z + c,
            ],
        ]
    )


def xyz_eulers_from_matrix(matrix):
    """
    Extracts XYZ Euler angles from a rotation matrix

    :param matrix: rotation matrix
    :type matrix: Matrix33
    :return: XYZ Euler angles in radians
    :rtype: Vector3
    """
    if 1 > matrix.m13 > -1:
        yaw = math.asin(clamp(matrix.m13, -1.0, 1.0))
        roll = math.atan2(-matrix.m12, matrix.m11)
        pitch = math.atan2(-matrix.m23, matrix.m33)
    elif matrix.m13 >= 1:
        roll = 0.0
        pitch = math.atan2(matrix.m21, matrix.m22)
        yaw = math.pi/2
    elif matrix.m13 <= -1:
        roll = 0.0
        pitch = -math.atan2(matrix.m21, matrix.m22)
        yaw = -math.pi / 2

    return Vector3([pitch, yaw, roll])


def matrix_from_xyz_eulers(angles):
    """
    Creates a rotation matrix from XYZ Euler angles

    :param angles: XYZ Euler angles in radians
    :type angles: Vector3
    :return: rotation matrix
    :rtype: Matrix33
    """
    sx = math.sin(angles[0])
    cx = math.cos(angles[0])
    sy = math.sin(angles[1])
    cy = math.cos(angles[1])
    sz = math.sin(angles[2])
    cz = math.cos(angles[2])

    return Matrix33(np.array(
        [
            # m1
            [
                cy * cz,
                -cy * sz,
                sy,
            ],
            # m2
            [
                cz * sx * sy + cx * sz,
                cx * cz - sx * sy * sz,
                -cy * sx,
            ],
            # m3
            [
                -cx * cz * sy + sx * sz,
                cz * sx + cx * sy * sz,
                cx * cy,
            ]
        ]
    ))


def rotation_btw_vectors(v1, v2):
    """Creates a rotation matrices to rotate from one vector (v1) to another (v2)
    based on Möller, Tomas, and John F. Hughes. "Efficiently building a matrix to rotate one vector to another."
    Journal of graphics tools 4.4 (1999): 1-4.

    :param v1: from vector
    :type v1: Vector3
    :param v2: to vector
    :type v2:  Vector3
    :return: rotation matrix
    :rtype: Matrix33
    """
    v = np.cross(v1, v2)
    e = np.dot(v1, v2)

    m = Matrix33()
    if abs(e) > (1.0 - eps):
        # if the vector a reflections i.e. 180 degree
        index = np.argmin(np.abs(v1))
        x = np.zeros(3, dtype=np.float32)
        x[index] = 1.0
        u = x - v1
        v = x - v2

        c1 = 2.0 / np.dot(u, u)
        c2 = 2.0 / np.dot(v, v)
        c3 = c1 * c2 * np.dot(u, v)

        m.m11 = 1 - c1 * u[0] * u[0] - c2 * v[0] * v[0] + c3 * v[0] * u[0]
        m.m12 = - c1 * u[0] * u[1] - c2 * v[0] * v[1] + c3 * v[0] * u[1]
        m.m13 = - c1 * u[0] * u[2] - c2 * v[0] * v[2] + c3 * v[0] * u[2]
        m.m21 = - c1 * u[1] * u[0] - c2 * v[1] * v[0] + c3 * v[1] * u[0]
        m.m22 = 1 - c1 * u[1] * u[1] - c2 * v[1] * v[1] + c3 * v[1] * u[1]
        m.m23 = - c1 * u[1] * u[2] - c2 * v[1] * v[2] + c3 * v[1] * u[2]
        m.m31 = - c1 * u[2] * u[0] - c2 * v[2] * v[0] + c3 * v[2] * u[0]
        m.m32 = - c1 * u[2] * u[1] - c2 * v[2] * v[1] + c3 * v[2] * u[1]
        m.m33 = 1 - c1 * u[2] * u[2] - c2 * v[2] * v[2] + c3 * v[2] * u[2]
    else:
        h = (1 - e) / np.dot(v, v)

        m.m11 = e + h * v[0] * v[0]
        m.m12 = h * v[0] * v[1] - v[2]
        m.m13 = h * v[0] * v[2] + v[1]
        m.m21 = h * v[0] * v[1] + v[2]
        m.m22 = e + h * v[1] * v[1]
        m.m23 = h * v[1] * v[2] - v[0]
        m.m31 = h * v[0] * v[2] - v[1]
        m.m32 = h * v[1] * v[2] + v[0]
        m.m33 = e + h * v[2] * v[2]

    return m


def matrix_from_pose(pose, angles_in_degrees=True):
    matrix = Matrix44.identity()

    position = pose[0:3]
    orientation = np.radians(pose[3:6]) if angles_in_degrees else pose[3:6]

    matrix[0:3, 0:3] = matrix_from_xyz_eulers(orientation)
    matrix[0:3, 3] = position

    return matrix
