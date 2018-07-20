import math
import numpy as np
from .util.vector import Vector3
from .util.matrix import Matrix33


def angle_axis_to_matrix(angle, axis):
    """ Converts rotation in angle/axis representation to a matrix

    :param angle: angle to rotate by
    :type angle: float
    :param axis: axis to rotate around
    :type axis: pyrr.Vector3
    :return: rotation matrix
    :rtype: pyrr.Matrix33
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
    :type matrix: pyrr.Matrix33
    :return: XYZ Euler angles
    :rtype: pyrr.Vector3
    """
    if 1 > matrix.m13 > -1:
        yaw = math.asin(matrix.m13)
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

    :param angles: XYZ Euler angles
    :type angles: pyrr.Vector3
    :return: rotation matrix
    :rtype: pyrr.Matrix33
    """
    sx = math.sin(angles.x)
    cx = math.cos(angles.x)
    sy = math.sin(angles.y)
    cy = math.cos(angles.y)
    sz = math.sin(angles.z)
    cz = math.cos(angles.z)

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
