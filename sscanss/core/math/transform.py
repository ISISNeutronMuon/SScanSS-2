import math
import numpy as np
from scipy import optimize, spatial
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
    """Converts a 6D pose into a transformation matrix. Pose contains
    3D translation (X, Y, Z) and 3D orientation (XYZ euler angles)

    :param pose: position and orientation
    :type pose: List[float]
    :param angles_in_degrees: indicates that angles are in degrees
    :type angles_in_degrees: bool
    :return: transformation matrix
    :rtype: sscanss.core.math.matrix.Matrix44
    """
    matrix = Matrix44.identity()

    position = pose[0:3]
    orientation = np.radians(pose[3:6]) if angles_in_degrees else pose[3:6]

    matrix[0:3, 0:3] = matrix_from_xyz_eulers(orientation)
    matrix[0:3, 3] = position

    return matrix


def rigid_transform(points_a, points_b):
    """ Calculate rigid transformation matrix given two sets of points

        Used Tutorial from http://nghiaho.com/?page_id=671
        Arun KS, Huang TS, Blostein SD (1987) Least-squares fitting of two 3-D
                point sets. IEEE Trans Pattern Anal Machine Intell 9:698–700

        points_a must have the same number of points as points_b. A minimum of 3
        points is required to get correct results

    :param points_a: array of 3D points.
    :type points_a: numpy.ndarray
    :param points_b: array of 3D points.
    :type points_b: numpy.ndarray
    :return: transformation matrix and residual errors
    :rtype: (Matrix44, list)
    """
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    h = np.dot((points_a - centroid_a).transpose(),
               points_b - centroid_b)

    u, s, v = np.linalg.svd(h)

    r = u @ np.diag([1, 1, np.linalg.det(v @ u)]) @ v
    t = -centroid_a @ r + centroid_b

    m = Matrix44.identity()
    m[0:3, 0:3] = r.transpose()
    m[0:3, 3] = t

    err = np.dot(points_a, r) + t - points_b
    err = np.linalg.norm(err, axis=1)

    return m, err


def find_3d_correspondence(source, query):
    """ Find Correspondence between 2 sets of 3D points by comparing pairwise distances.
    This method fails when points cannot be discriminated by their distances e.g in an equilateral triangle.
    source must not have less points than query and a minimum of 2 points is required to get correct results.

    :param source: array of 3D points,
    :type source: numpy.ndarray
    :param query: array of 3D point.
    :type query: numpy.ndarray
    :return: indices of correspondences
    :rtype: list[int]
    """
    a_size = source.shape[0]
    b_size = query.shape[0]
    da = spatial.distance.pdist(source, 'sqeuclidean')
    db = spatial.distance.pdist(query, 'sqeuclidean')
    pairs_a = np.array([(x, y) for x in range(a_size-1) for y in range(x + 1, a_size)])
    pairs_b = np.array([(x, y) for x in range(b_size-1) for y in range(x + 1, b_size)])

    dist = np.abs(np.tile(da, (db.size, 1)) - np.tile(db, (da.size, 1)).transpose())
    _, col_ind = optimize.linear_sum_assignment(dist)

    final = [set() for _ in range(b_size)]
    for aa, bb in zip(pairs_a[col_ind], pairs_b):
        i, j = bb
        if final[i]:
            final[i].intersection_update(aa)
        else:
            final[i].update(aa)

        if final[j]:
            final[j].intersection_update(aa)
        else:
            final[j].update(aa)

    for i, corr in enumerate(final):
        # Each query point should have exactly one match, multiple
        # matches or null match are not allowed
        if len(corr) == 1:
            final[i] = corr.pop()
        else:
            raise ValueError('One to one correspondence could not be found.')

    return final
