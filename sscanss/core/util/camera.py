import math
import numpy as np
from pyrr import Vector3, Matrix44, Matrix33
from .misc import Directions

EPSILON = 0.00001
DEFAULT_Z_NEAR = 0.01
DEFAULT_Z_FAR = 1000.0


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


def get_arcball_vector(x, y):
    """ Compute the arcball vector for a point(x, y) on the screen
    https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball

    :param x: x coordinate of point on screen
    :type x: float
    :param y: y coordinate of point on screen
    :type y: float
    :return: arcball vector
    :rtype: pyrr.Vector3
    """
    vec = Vector3([x - 1.0, 1.0 - y, 0])
    distance = vec.x * vec.x + vec.y * vec.y
    if distance <= 1:
        vec.z = math.sqrt(1 - distance)

    return vec.normalized


def get_eulers(matrix):
    """
    Extracts XYZ Euler angles from a rotation matrix

    :param matrix: rotation matrix
    :type matrix: pyrr.Matrix44
    :return: XYZ Euler angles
    :rtype: pyrr.Vector3
    """
    yaw = math.asin(matrix.m13)

    if matrix.m33 < 0:
        yaw = math.pi - yaw if yaw >= 0 else -math.pi - yaw

    if EPSILON > matrix.m11 > -EPSILON:
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


class Camera:
    def __init__(self, aspect, fov):
        """
        Represents a camera with pan, rotate and zoom capabilities

        :param aspect: ratio of the x and y dimension ie x / y
        :type aspect: float
        :param fov: field of view for y dimension in degrees
        :type fov: float
        """
        self.z_near = DEFAULT_Z_NEAR
        self.z_far = DEFAULT_Z_FAR
        self.moving_z_plane = self.z_near
        self.aspect = aspect
        self.fov = fov

        self.position = Vector3()
        self.target = Vector3()
        self.rot_matrix = Matrix33.identity()
        self.angle = Vector3()
        self.distance = 0.0

        self.model_view = Matrix44.identity()

    def zoomToFit(self, center, radius):
        """
        Computes the model view matrix so that camera is looking at an
        object.

        :param center: center of the object to look at
        :type center: pyrr.Vector3
        :param radius: radius of object to look at
        :type radius: float
        """
        self.inital_target = center
        self.initial_radius = radius

        direction = Vector3([0.0, 0.0, -1.0])
        half_min_fov_in_radians = 0.5 * math.radians(self.fov)

        if self.aspect < 1.0:
            # fov in x is smaller
            half_min_fov_in_radians = math.atan(self.aspect * math.tan(half_min_fov_in_radians))

        distance_to_center = radius / math.sin(half_min_fov_in_radians)
        eye = center - direction * distance_to_center

        self.lookAt(eye, center, Vector3([0.0, 1.0, 0.0]))

        self.z_near = distance_to_center - radius
        self.z_far = distance_to_center + radius
        self.moving_z_plane = self.z_near

    def lookAt(self, position, target, up_dir=None):
        """
        Computes the model view matrix so that camera is looking at a target
        from a desired position and orientation.

        :param position: position of camera
        :type position: pyrr.Vector3
        :param target: point to look at
        :type target: pyrr.Vector3
        :param up_dir: up direction of camera
        :type up_dir: pyrr.Vector3
        """
        self.position = position
        self.target = target
        self.model_view = Matrix44.identity()

        if position == target:
            self.model_view.from_translation(-position)
            self.rot_matrix = Matrix44.identity()
            self.angle = Vector3()
            return

        up = up_dir

        forward = position - target
        self.distance = forward.length

        forward.normalize()

        if up is None:
            condition = math.fabs(forward.x) < EPSILON and math.fabs(forward.z) < EPSILON
            if condition:
                up = Vector3([0, 0, -1]) if forward.y > 0 else Vector3([0, 0, 1])
            else:
                up = Vector3([0, 1, 0])

        left = up ^ forward  # cross product
        left.normalize()

        up = forward ^ left

        self.rot_matrix.r1[:3] = left
        self.rot_matrix.r2[:3] = up
        self.rot_matrix.r3[:3] = forward

        self.model_view.r1[:3] = left
        self.model_view.r2[:3] = up
        self.model_view.r3[:3] = forward
        
        # trans = self.model_view.transpose() * -position
        trans = Vector3()
        trans.x = left.x * -position.x + left.y * -position.y + left.z * -position.z
        trans.y = up.x * -position.x + up.y * -position.y + up.z * -position.z
        trans.z = forward.x * -position.x + forward.y * -position.y + forward.z * -position.z
        self.model_view.c4[:3] = trans

        self.angle = np.degrees(get_eulers(self.rot_matrix))

    def pan(self, delta_x, delta_y):
        """
        Tilts the camera viewing axis vertically and/or horizontally
        while maintaining the camera position

        :param delta_x: offset by which camera is panned in screen x axis
        :type delta_x: float
        :param delta_y: offset by which camera is panned in screen y axis
        :type delta_y: float
        """

        camera_left = Vector3([self.model_view.m11, self.model_view.m12, self.model_view.m13])
        camera_up = Vector3([self.model_view.m21, self.model_view.m22, self.model_view.m23])

        # delta is scaled by distance so pan is larger when object is farther
        distance = self.distance if self.distance >= 1.0 else 1
        offset = (delta_x * camera_left - delta_y * camera_up) * distance

        new_target = self.target + offset

        self.target = new_target
        self.computeModelViewMatrix()

    def rotate(self, p1, p2):
        """
        Rotates the camera around the target using points in screen space

        :param p1: first point in screen space
        :type p1: tuple
        :param p2: second point in screen space
        :type p2: tuple
        """

        x1, y1 = p1
        x2, y2 = p2
        if x2 != x1 or y2 != y1:
            va = get_arcball_vector(x1, y1)
            vb = get_arcball_vector(x2, y2)

            angle = math.acos(min(1.0, va | vb))
            axis = (va ^ vb).normalized
            self.rot_matrix *= angle_axis_to_matrix(angle, axis)
            self.computeModelViewMatrix()

    def zoom(self, delta):
        """
        Moves the camera forward or back along the viewing axis and adjusts
        the view frustum (z near and far) to avoid clipping.

        :param delta: offset by which camera is zoomed
        :type delta: float
        """
        # delta is scaled by distance so zoom is faster when object is farther
        distance = self.distance if self.distance >= 1.0 else 1
        offset = delta * distance

        # re-calculate view frustum
        z_depth = self.z_far - self.z_near
        self.moving_z_plane -= offset
        self.z_near = DEFAULT_Z_NEAR if self.moving_z_plane < DEFAULT_Z_NEAR else self.moving_z_plane
        self.z_far = self.z_near + z_depth

        # re-calculate camera distance
        distance -= offset
        self.distance = distance
        self.computeModelViewMatrix()

    def computeModelViewMatrix(self):
        """
        Computes the model view matrix of camera
        """
        target = self.target
        dist = self.distance
        rot = self.rot_matrix

        left = Vector3([rot.m11, rot.m21, rot.m31])
        up = Vector3([rot.m12, rot.m22, rot.m32])
        forward = Vector3([rot.m13, rot.m23, rot.m33])

        trans = Vector3()
        trans.x = left.x * -target.x + up.x * -target.y + forward.x * -target.z
        trans.y = left.y * -target.x + up.y * -target.y + forward.y * -target.z
        trans.z = left.z * -target.x + up.z * -target.y + forward.z * -target.z - dist

        self.model_view = Matrix44.identity()
        self.model_view.c1[:3] = left
        self.model_view.c2[:3] = up
        self.model_view.c3[:3] = forward
        self.model_view.c4[:3] = trans

        forward = Vector3([-self.model_view.m31, -self.model_view.m32, -self.model_view.m33])
        self.position = target - (dist * forward)

    @property
    def perspective(self):
        """
        Computes the one-point perspective projection matrix of camera

        :return: 4 x 4 perspective projection matrix
        :rtype: pyrr.Matrix44
        """
        projection = Matrix44()

        y_max = self.z_near * math.tan(0.5 * math.radians(self.fov))
        x_max = y_max * self.aspect

        z_depth = self.z_far - self.z_near

        projection.m11 = self.z_near / x_max
        projection.m22 = self.z_near / y_max
        projection.m33 = (-self.z_near - self.z_far) / z_depth
        projection.m43 = -1
        projection.m34 = -2 * self.z_near * self.z_far / z_depth

        return projection

    def viewFrom(self, direction):
        """
        Changes the viewing direction of the camera

        :param direction: camera viewing direction
        :type direction: sscanss.core.util.misc.Directions
        """
        if direction == Directions.right:
            position = self.target - (Vector3([1.0, 0.0, 0.0]) * self.distance)
            self.lookAt(position, self.target, Vector3([0.0, 1.0, 0.0]))
        elif direction == Directions.left:
            position = self.target - (Vector3([-1.0, 0.0, 0.0]) * self.distance)
            self.lookAt(position, self.target, Vector3([0.0, 1.0, 0.0]))
        elif direction == Directions.up:
            position = self.target - (Vector3([0.0, -1.0, 0.0]) * self.distance)
            self.lookAt(position, self.target, Vector3([0.0, 0.0, -1.0]))
        elif direction == Directions.down:
            position = self.target - (Vector3([0.0, 1.0, 0.0]) * self.distance)
            self.lookAt(position, self.target, Vector3([0.0, 0.0, 1.0]))
        elif direction == Directions.front:
            position = self.target - (Vector3([0.0, 0.0, 1.0]) * self.distance)
            self.lookAt(position, self.target, Vector3([0.0, 1.0, 0.0]))
        else:
            position = self.target - (Vector3([0.0, 0.0, -1.0]) * self.distance)
            self.lookAt(position, self.target, Vector3([0.0, 1.0, 0.0]))

    def reset(self):
        """
        Resets the camera view
        """
        try:
            self.zoomToFit(self.inital_target, self.initial_radius)
        except AttributeError:
            self.position = Vector3()
            self.target = Vector3()

            self.rot_matrix = Matrix44.identity()
            self.angle = Vector3()
            self.distance = 0.0

            self.model_view = Matrix44.identity()

