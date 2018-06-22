import math
import numpy as np
from pyrr import Vector3, Matrix33


def angle_axis_to_matrix(angle, axis):
    """ Converts rotation in angle/axis representation to a matrix

    :param angle: angle to rotate by
    :type angle: float
    :param axis: axis to rotate around
    :type axis: pyrr.Vector3
    :return: rotation matrix
    :rtype: pyrr.Matrix33
    """
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    return Matrix33(
        [
            [
                t * axis.x * axis.x + c,
                t * axis.x * axis.y - axis.z * s,
                t * axis.x * axis.z + axis.y * s,
            ],
            [
                t * axis.x * axis.y + axis.z * s,
                t * axis.y * axis.y + c,
                t * axis.y * axis.z - axis.x * s,
            ],
            [
                t * axis.x * axis.z - axis.y * s,
                t * axis.y * axis.z + axis.x * s,
                t * axis.z * axis.z + c,
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
    eps = 1e-7
    yaw = math.asin(matrix.m13)

    if matrix.m33 < 0:
        yaw = math.pi - yaw if yaw >= 0 else -math.pi - yaw

    if eps > matrix.m11 > -eps:
        roll = 0.0
        pitch = math.atan2(matrix.m21, matrix.m22)
    else:
        roll = math.atan2(-matrix.m12, matrix.m11)
        pitch = math.atan2(-matrix.m23, matrix.m33)

    return Vector3([pitch, yaw, roll])


def matrix_from_xyz_eulers(angles):
    """
    Creates a rotation matrix from XYZ Euler angles

    :param angles: XYZ Euler angles
    :type angles: pyrr.Vector3
    :return: rotation matrix
    :rtype: pyrr.Matrix33
    """
    x = angles[0]
    y = angles[1]
    z = angles[2]

    sx = math.sin(x)
    cx = math.cos(x)
    sy = math.sin(y)
    cy = math.cos(y)
    sz = math.sin(z)
    cz = math.cos(z)

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
