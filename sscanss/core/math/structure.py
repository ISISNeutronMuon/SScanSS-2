"""
Classes representing geometric structures
"""
import numpy as np
from .constants import VECTOR_EPS


class Line:
    """Creates a Line object in the point axis form. An error is
    raised if normal is zero length

    :param axis: 3D axis of line
    :type axis: numpy.ndarray
    :param point: 3D point
    :type point: numpy.ndarray
    :raises: ValueError
    """
    def __init__(self, axis, point):
        self.point = point

        length = np.linalg.norm(axis)
        if length < VECTOR_EPS:
            raise ValueError('The line axis ({}, {}, {}) is invalid.'.format(*axis))

        self.axis = axis

    def __str__(self):
        return f'Line(axis: {self.axis}, point: {self.point})'


class Plane:
    """Creates a Plane object in the point normal form. The normal vector is
    normalized to ensure its length is 1. An error is raised if normal is zero length

    :param normal: 3D normal vector
    :type normal: numpy.ndarray
    :param point: 3D point
    :type point: numpy.ndarray
    :raises: ValueError
    """
    def __init__(self, normal, point):
        self.point = point

        length = np.linalg.norm(normal)
        if length < VECTOR_EPS:
            raise ValueError('The plane normal ({}, {}, {}) is invalid.'.format(*normal))

        self.normal = normal / length

    def distanceFromOrigin(self):
        """Computes the distance from the plane to the origin."""
        return np.dot(self.normal, self.point)

    @classmethod
    def fromCoefficient(cls, a, b, c, d):
        """Creates a plane using the standard plane equation, ax + by + cz = d.
        An error is raised if the equation coefficients are invalid.

        :param a: x coefficient of the plane equation
        :type a: float
        :param b: y coefficient of the plane equation
        :type b: float
        :param c: z coefficient of the plane equation
        :type c: float
        :param d: constant of the plane equation
        :type d: float
        :return: plane object
        :rtype: Plane
        """
        normal = np.array([a, b, c])
        length = np.linalg.norm(normal)
        if length < VECTOR_EPS:
            raise ValueError(f'The plane ({a}x + {b}y + {c}z = {d}) is invalid.')

        x = 0. if a == 0 else -d / a
        y = 0. if b == 0 else -d / b
        z = 0. if c == 0 else -d / c

        point = np.array([x, y, z])
        return cls(normal, point)

    @classmethod
    def fromPlanarPoints(cls, point_a, point_b, point_c):
        """Creates a Plane object from 3 planar points. An error is raised
        if the points are collinear. Based on code from https://www.songho.ca/math/plane/plane.html

        :param point_a: first planar 3D point
        :type point_a: numpy.ndarray
        :param point_b: second planar 3D point
        :type point_b: numpy.ndarray
        :param point_c: third planar 3D point
        :type point_c: numpy.ndarray
        :return: plane normal and point
        :rtype: Plane
        """
        v1 = point_a - point_b
        v2 = point_a - point_c

        normal = np.cross(v1, v2)
        length = np.linalg.norm(normal)
        if length < VECTOR_EPS:
            raise ValueError('The points should not be arranged in a line.')

        normal = normal / length

        return cls(normal, point_a)

    @classmethod
    def fromBestFit(cls, points):
        """Fits a Plane to a 3D point set (minimum of 3 points). Based on code from
        https://mathworks.com/matlabcentral/fileexchange/43305-plane-fit

        :param points: N x 3 array of point
        :type points: numpy.ndarray
        :return: plane normal and point
        :rtype: Plane
        """
        s = len(points)
        if s < 3:
            raise ValueError('A minimum of 3 points is required for plane fitting.')

        centroid = np.mean(points, axis=0)
        p = points - centroid
        w, v = np.linalg.eig(p.transpose() @ p)

        # Extract the output from the eigenvectors
        normal = -v[:, w.argmin()]

        return cls(normal, centroid)

    def intersectPlane(self, plane):
        """Computes intersection with another plane

        :param plane: plane to check for intersection
        :type plane: Plane
        :return: line if the planes intersect
        :rtype: Optional[Line]
        """
        n1 = self.normal
        n2 = plane.normal

        v = np.cross(n1, n2)
        if v[0] == 0 and v[1] == 0 and v[2] == 0:
            return None

        d1 = -self.distanceFromOrigin()
        d2 = -plane.distanceFromOrigin()
        dot = np.dot(v, v)
        u1 = d2 * n1
        u2 = -d1 * n2
        p = np.cross(u1 + u2, v) / dot

        return Line(v, p)

    def moveToDistance(self, distance):
        """Clones the plane and moves to given distance from origin

        :param distance: new distance from origin
        :type distance: float
        :return: translated plane
        :rtype: Plane
        """
        offset = self.normal * (distance - self.distanceFromOrigin())
        return Plane(self.normal, self.point + offset)

    def __str__(self):
        return f'Plane(normal: {self.normal}, point: {self.point})'


def fit_circle_2d(x, y):
    """Fits a Circle to a 2D point set, uses code from
    https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/

    :param x: x coordinates of points
    :type x: numpy.ndarray
    :param y: y coordinates of points
    :type y: numpy.ndarray
    :return: x coordinate of center, y coordinate of center, and radius
    :rtype: Tuple[float, float, float]
    """

    a = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Solve by method of least squares
    c = np.linalg.lstsq(a, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


def fit_circle_3d(points):
    """Fits a Circle to a 3D point set (minimum of 3 points).

    :param points: N x 3 array of point
    :type points: numpy.ndarray
    :return: center of rotation, axis of rotation, radius, and residuals
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, float, numpy.ndarray]
    """
    if len(points) < 3:
        raise ValueError('A minimum of 3 points is required for circle fitting.')

    # Create new coordinate frame on circle plane
    z_axis = Plane.fromBestFit(points).normal
    x_axis = [z_axis[2], 0., -z_axis[0]]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    matrix = np.eye(3)
    matrix[:, 0] = x_axis
    matrix[:, 1] = y_axis
    matrix[:, 2] = z_axis

    # Transform point into a new coordinate frame
    new_points = points @ matrix

    # Fit circle on 2D points
    xc, yc, radius = fit_circle_2d(new_points[:, 0], new_points[:, 1])

    zc = np.mean(new_points[:, 2])
    center = matrix @ [xc, yc, zc]

    axis = center - points
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    est_center = points + (axis * radius)
    residuals = center - est_center

    return center, z_axis, radius, residuals


def fit_line_3d(points):
    """Fits a Line to a 3D point set (minimum of 3 points).

    :param points: N x 3 array of point
    :type points: numpy.ndarray
    :return: centroid of line, axis of line, and residuals for each axis
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    n = len(points)
    if n < 3:
        raise ValueError('A minimum of 3 points is required for line fitting.')

    xyz = np.tile(points.mean(axis=0), (n, 1))
    centered_line = (points - xyz) / np.sqrt(n - 1)
    _, _, v = np.linalg.svd(centered_line)
    axis = v[0, :]
    center = xyz[0]

    diff = points - center
    proj = np.einsum('ij,ij->i', diff, np.expand_dims(axis, axis=0))[:, None]
    proj_points = center + proj * axis
    residuals = proj_points - points

    return center, axis, residuals
