"""
Classes representing geometric structures
"""
import numpy as np

eps = 0.000001


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
        if length < eps:
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
        if length < eps:
            raise ValueError('The plane ({}x + {}y + {}z = {}) is invalid.'.format(a, b, c, d))

        x = 0. if a == 0 else -d / a
        y = 0. if b == 0 else -d / b
        z = 0. if c == 0 else -d / c

        point = np.array([x, y, z])
        return cls(normal, point)

    @classmethod
    def fromPlanarPoints(cls, point_a, point_b, point_c):
        """Creates a Plane object from 3 planar points. An error is raised
        if the points are collinear.

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
        if length < eps:
            raise ValueError('The points should not be arranged in a line.')

        normal = normal / length

        return cls(normal, point_a)

    def __str__(self):
        return 'normal: {}, point: {}'.format(self.normal, self.point)
