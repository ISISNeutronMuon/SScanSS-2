"""
A collection of functions for rigid transformation and rotation conversion
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from .constants import VECTOR_EPS
from .vector import Vector3
from .matrix import Matrix33, Matrix44
from .misc import clamp, is_close


def check_rotation(matrix):
    """Checks that the matrix is a valid rotation matrix i.e no scaling, shearing

    :param  matrix: from unit vector
    :type matrix: Matrix44
    :return: indicates that matrix is a valid rotation
    :rtype: bool
    """
    rot = matrix[0:3, 0:3]
    identity = rot @ np.transpose(rot)
    if not is_close(identity, np.eye(3)):
        return False
    return True


def angle_axis_btw_vectors(v1, v2):
    """Calculates the axis-angle representation rotation required to rotate
    from one vector (v1) to another (v2). Both vectors are assumed to be normalized.

    :param v1: from unit vector
    :type v1: Vector3
    :param v2: to unit vector
    :type v2:  Vector3
    :return: axis-angle representation. angle in radians
    :rtype: Tuple[float, Vector3]
    """
    axis = np.cross(v1, v2)
    ct = clamp(np.dot(v1, v2), -1., 1.)
    angle = math.acos(ct)
    st = math.sqrt(1 - ct * ct)

    if abs(ct) > (1 - VECTOR_EPS):
        index = np.argmin(np.abs(v1))
        axis = np.zeros(3, dtype=np.float32)
        axis[index] = 1.0

        return angle, Vector3(axis)
    else:
        return angle, Vector3(axis) / st


def matrix_to_angle_axis(matrix):
    """Converts a rotation matrix to the equivalent axis-angle representation

    :param matrix: rotation matrix
    :type matrix: Union[Matrix33, Matrix44]
    :return: axis-angle representation. angle in radians
    :rtype: Tuple[float, Vector3]
    """
    r = matrix[0:3, 0:3]
    b = r - np.identity(3)

    _, _, v = np.linalg.svd(b)

    axis = v[-1, :]

    two_cos_theta = np.trace(r) - 1
    two_sin_theta_v = [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]
    two_sin_theta = np.dot(axis, two_sin_theta_v)

    angle = math.atan2(two_sin_theta, two_cos_theta)

    return angle, Vector3(axis)


def angle_axis_to_matrix(angle, axis):
    """Converts an axis-angle rotation to the equivalent rotation matrix

    :param angle: angle to rotate by in radians
    :type angle: float
    :param axis: axis to rotate around
    :type axis: Vector3
    :return: rotation matrix
    :rtype: Matrix33
    """
    _axis = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    return Matrix33([
        [
            t * _axis[0] * _axis[0] + c,
            t * _axis[0] * _axis[1] - _axis[2] * s,
            t * _axis[0] * _axis[2] + _axis[1] * s,
        ],
        [
            t * _axis[0] * _axis[1] + _axis[2] * s,
            t * _axis[1] * _axis[1] + c,
            t * _axis[1] * _axis[2] - _axis[0] * s,
        ],
        [
            t * _axis[0] * _axis[2] - _axis[1] * s,
            t * _axis[1] * _axis[2] + _axis[0] * s,
            t * _axis[2] * _axis[2] + c,
        ],
    ])


def xyz_eulers_from_matrix(matrix):
    """Extracts XYZ Euler angles from a rotation matrix

    :param matrix: rotation matrix
    :type matrix: Matrix33
    :return: XYZ Euler angles in radians
    :rtype: Vector3
    """
    if 1 > matrix[0, 2] > -1:
        theta_y = math.asin(matrix[0, 2])
        theta_z = math.atan2(-matrix[0, 1], matrix[0, 0])
        theta_x = math.atan2(-matrix[1, 2], matrix[2, 2])
    elif matrix[0, 2] >= 1:
        theta_z = 0.0
        theta_x = math.atan2(matrix[1, 0], matrix[1, 1])
        theta_y = math.pi / 2
    else:
        theta_z = 0.0
        theta_x = -math.atan2(matrix[1, 0], matrix[1, 1])
        theta_y = -math.pi / 2

    return Vector3([theta_x, theta_y, theta_z])


def matrix_from_xyz_eulers(angles):
    """Creates a rotation matrix from XYZ Euler angles

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

    return Matrix33(
        np.array([
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
        ]))


def matrix_from_zyx_eulers(angles):
    """Creates a rotation matrix from ZYX Euler angles

    :param angles: ZYX Euler angles in radians
    :type angles: Vector3
    :return: rotation matrix
    :rtype: Matrix33
    """
    sx = math.sin(angles[2])
    cx = math.cos(angles[2])
    sy = math.sin(angles[1])
    cy = math.cos(angles[1])
    sz = math.sin(angles[0])
    cz = math.cos(angles[0])

    return Matrix33(
        np.array([
            # m1
            [
                cy * cz,
                cz * sx * sy - cx * sz,
                cx * cz * sy + sx * sz,
            ],
            # m2
            [
                cy * sz,
                cx * cz + sx * sy * sz,
                -cz * sx + cx * sy * sz,
            ],
            # m3
            [
                -sy,
                cy * sx,
                cx * cy,
            ]
        ]))


def rotation_btw_vectors(v1, v2):
    """Creates a rotation matrices to rotate from one vector (v1) to another (v2).
    Both vectors are assumed to be normalized. The implementation is based on Möller,
    Tomas, and John F. Hughes. "Efficiently building a matrix to rotate one vector to another."
    Journal of graphics tools 4.4 (1999): 1-4.

    :param v1: from unit vector
    :type v1: Vector3
    :param v2: to unit vector
    :type v2:  Vector3
    :return: rotation matrix
    :rtype: Matrix33
    """
    v = np.cross(v1, v2)
    e = np.dot(v1, v2)

    m = Matrix33.identity()
    if abs(e) > (1.0 - VECTOR_EPS):
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
        m.m12 = -c1 * u[0] * u[1] - c2 * v[0] * v[1] + c3 * v[0] * u[1]
        m.m13 = -c1 * u[0] * u[2] - c2 * v[0] * v[2] + c3 * v[0] * u[2]
        m.m21 = -c1 * u[1] * u[0] - c2 * v[1] * v[0] + c3 * v[1] * u[0]
        m.m22 = 1 - c1 * u[1] * u[1] - c2 * v[1] * v[1] + c3 * v[1] * u[1]
        m.m23 = -c1 * u[1] * u[2] - c2 * v[1] * v[2] + c3 * v[1] * u[2]
        m.m31 = -c1 * u[2] * u[0] - c2 * v[2] * v[0] + c3 * v[2] * u[0]
        m.m32 = -c1 * u[2] * u[1] - c2 * v[2] * v[1] + c3 * v[2] * u[1]
        m.m33 = 1 - c1 * u[2] * u[2] - c2 * v[2] * v[2] + c3 * v[2] * u[2]
    else:
        vv = np.dot(v, v)
        if vv < VECTOR_EPS:
            return m

        h = (1 - e) / vv

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


def matrix_from_pose(pose, angles_in_degrees=True, order='xyz'):
    """Converts a 6D pose into a transformation matrix. Pose contains
    3D translation (X, Y, Z) and 3D orientation (XYZ or ZYX euler angles)

    :param pose: position and orientation
    :type pose: List[float]
    :param angles_in_degrees: indicates that angles are in degrees
    :type angles_in_degrees: bool
    :param order: euler angle convention
    :type order: str
    :return: transformation matrix
    :rtype: Matrix44
    """
    matrix = Matrix44.identity()

    position = pose[0:3]
    orientation = np.radians(pose[3:6]) if angles_in_degrees else pose[3:6]
    order = order.lower()
    if order == 'xyz':
        matrix_from_euler_func = matrix_from_xyz_eulers
    elif order == 'zyx':
        matrix_from_euler_func = matrix_from_zyx_eulers
    else:
        raise ValueError(f'The given order {order} is not supported.')

    matrix[0:3, 0:3] = matrix_from_euler_func(orientation)
    matrix[0:3, 3] = position

    return matrix


class TransformResult:
    """Data class for the rigid transform result

    :param matrix: transformation matrix
    :type matrix: Matrix44
    :param error: residual error for each point
    :type error: List[float]
    :param point_a: array of 3D points
    :type point_a: numpy.ndarray
    :param point_b: array of 3D points
    :type point_b: numpy.ndarray
    """
    def __init__(self, matrix, error, point_a, point_b):
        self.matrix = matrix
        self.error = error
        self.point_a = point_a
        self.point_b = point_b

    @property
    def average(self):
        """Computes the average of residual errors

        :return: average of residual errors
        :rtype: float
        """
        return np.mean(self.error)

    @property
    def total(self):
        """Computes the sum of the residual errors

        :return: sum of residual errors
        :rtype: float
        """
        return np.sum(self.error)

    @property
    def distance_analysis(self):
        """Computes pairwise euclidean distances for both point sets (a and b)
        and the absolute difference of the computed distances

        :return: pairwise distance for a, b and absolute difference in columns
        :rtype: numpy,ndarray
        """
        da = distance.pdist(self.point_a, 'euclidean')
        db = distance.pdist(self.point_b, 'euclidean')
        return np.column_stack((da, db, np.abs(da - db)))


def rigid_transform(points_a, points_b):
    """Calculates rigid transformation matrix given two sets of 3D points

        S. Umeyama, Least-squares estimation of transformation parameters
        between two point patterns, IEEE Trans. Pattern Anal. Mach. Intell.
        13 (4) (1991) 376- 380

        points_a must have the same number of points as points_b. A minimum of 3
        points are required to get correct results

    :param points_a: array of 3D points.
    :type points_a: numpy.ndarray
    :param points_b: array of 3D points.
    :type points_b: numpy.ndarray
    :return: transformation matrix and residual errors
    :rtype: TransformResult
    """
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    h = (points_a - centroid_a).transpose() @ (points_b - centroid_b)

    u, _, v = np.linalg.svd(h)

    r = u @ np.diag([1, 1, np.linalg.det(v @ u)]) @ v
    t = -centroid_a @ r + centroid_b

    m = Matrix44.identity()
    m[0:3, 0:3] = r.transpose()
    m[0:3, 3] = t

    err = points_a @ r + t - points_b
    err = np.linalg.norm(err, axis=1)

    return TransformResult(m, err, points_a, points_b)


def find_3d_correspondence(source, query):
    """Finds Correspondence between two sets of 3D points by comparing pairwise distances.
    This method fails when points cannot be discriminated by their distances e.g in an equilateral triangle.
    source must not have less points than query and a minimum of two points are required to get correct results.

    :param source: array of 3D points,
    :type source: numpy.ndarray
    :param query: array of 3D point.
    :type query: numpy.ndarray
    :return: indices of correspondences
    :rtype: numpy.ndarray
    """
    a_size = source.shape[0]
    b_size = query.shape[0]
    da = distance.pdist(source, 'sqeuclidean')
    db = distance.pdist(query, 'sqeuclidean')
    pairs_a = np.array([(x, y) for x in range(a_size - 1) for y in range(x + 1, a_size)])
    pairs_b = np.array([(x, y) for x in range(b_size - 1) for y in range(x + 1, b_size)])

    dist = np.abs(np.tile(da, (db.size, 1)) - np.tile(db, (da.size, 1)).transpose())
    _, col_ind = linear_sum_assignment(dist)

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

    return np.array(final)
