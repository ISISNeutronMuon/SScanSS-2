"""
Class and functions for scene camera
"""
import math
from enum import unique, Enum
from ..math.misc import clamp
from ..math.matrix import Matrix44, Matrix33
from ..math.transform import angle_axis_to_matrix
from ..math.vector import Vector3, Vector4
from ..util.misc import Directions

eps = 1e-6
DEFAULT_Z_NEAR = 0.1
DEFAULT_Z_FAR = 1000.0


def screen_to_world(screen_point, view_matrix, projection_matrix, width, height):
    """Converts homogeneous point in world coordinates to a point in screen coordinates.

    :param screen_point: homogeneous point in screen coordinates
    :type screen_point: Vector3
    :param view_matrix: model-vew matrix
    :type view_matrix: Matrix44
    :param projection_matrix: projection matrix
    :type projection_matrix: Matrix44
    :param width: screen width
    :type width: float
    :param height: screen height
    :type height: float
    :return: world point and boolean indicating if point is valid
    :rtype: Tuple[Vector3, bool]
    """
    scrx = screen_point.x / width * 2.0 - 1.0
    scry = screen_point.y / height * 2.0 - 1.0
    scrz = 2.0 * screen_point.z - 1.0

    view_projection_matrix = projection_matrix @ view_matrix
    if not view_projection_matrix.invertible:
        return Vector3(), False

    point = view_projection_matrix.inverse() @ Vector4([scrx, scry, scrz, 1.0])
    if abs(point.w) < eps:
        return Vector3(), False

    point /= point.w
    return Vector3(point.xyz), True


def world_to_screen(world_point, view_matrix, projection_matrix, width, height):
    """Converts homogeneous point in world coordinates to a point in screen coordinates.

    :param world_point: homogeneous point in world coordinates
    :type world_point: Vector3
    :param view_matrix: model-vew matrix
    :type view_matrix: Matrix44
    :param projection_matrix: projection matrix
    :type projection_matrix: Matrix44
    :param width: screen width
    :type width: float
    :param height: screen height
    :type height: float
    :return: screen point and boolean indicating if point is valid
    :rtype: Tuple[Vector3, bool]
    """
    view_projection_matrix = projection_matrix @ view_matrix

    point = view_projection_matrix @ Vector4([*world_point, 1.0])
    if point.w < eps:
        return Vector3(), False

    point /= point.w

    winx = (point.x * 0.5 + 0.5) * width
    winy = (point.y * 0.5 + 0.5) * height
    winz = (point.z * 0.5 + 0.5)

    return Vector3([winx, winy, winz]), True


def get_arcball_vector(x, y):
    """Computes the arcball vector for a point(x, y) on the screen. Based on code from
    https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball

    :param x: x coordinate of point on screen
    :type x: float
    :param y: y coordinate of point on screen
    :type y: float
    :return: arcball vector
    :rtype: Vector3
    """
    vec = Vector3([x - 1.0, 1.0 - y, 0])
    distance = vec.x * vec.x + vec.y * vec.y
    if distance <= 1:
        vec.z = math.sqrt(1 - distance)

    return vec.normalized


class Camera:
    """Creates a camera object with pan, rotate and zoom capabilities

    :param aspect: ratio of the x and y dimension ie x / y
    :type aspect: float
    :param fov: field of view for y dimension in degrees
    :type fov: float
    :param direction: Initial direction vector
    :type direction: Union[List[float], None]
    :param up: Initial up vector
    :type up: Union[List[float], None]
    """
    @unique
    class Projection(Enum):
        """Camera projection types"""
        Perspective = 0
        Orthographic = 1

    def __init__(self, aspect, fov, direction=None, up=None):
        self.mode = Camera.Projection.Perspective

        self.z_near = DEFAULT_Z_NEAR
        self.z_far = DEFAULT_Z_FAR
        self.z_depth = self.z_far - self.z_near
        self.moving_z_plane = self.z_near
        self.aspect = aspect
        self.fov = fov

        self.initial_target = Vector3()
        self.initial_radius = 1.0
        self.position = Vector3()
        self.target = Vector3()
        self.rot_matrix = Matrix33.identity()
        self.model_view = Matrix44.identity()

        self.distance = 0.0
        self.direction = [0., 1., 0.] if direction is None else direction
        self.up = up
        self.setViewDirection(self.direction, self.up)

    @property
    def projection(self):
        if self.mode == Camera.Projection.Perspective:
            return self.perspective

        return self.orthographic

    def zoomToFit(self, center=None, radius=None):
        """Computes the model view matrix so that camera is looking at an
        object and the whole object is visible.

        :param center: center of the object to look at
        :type center: Union[Vector3, None]
        :param radius: radius of object to look at
        :type radius: Union[float, None]
        """
        self.initial_target = center if center is not None else self.initial_target
        self.initial_radius = radius if radius is not None else self.initial_radius

        rot = self.rot_matrix
        direction = -Vector3([rot.m31, rot.m32, rot.m33])
        up = Vector3([rot.m21, rot.m22, rot.m23])

        half_min_fov_in_radians = 0.5 * math.radians(self.fov)

        if self.aspect < 1.0:
            # fov in x is smaller
            half_min_fov_in_radians = math.atan(self.aspect * math.tan(half_min_fov_in_radians))

        distance_to_center = self.initial_radius / math.sin(half_min_fov_in_radians)
        eye = self.initial_target - direction * distance_to_center

        self.lookAt(eye, self.initial_target, up)

        self.z_near = distance_to_center - self.initial_radius
        self.z_far = distance_to_center + self.initial_radius
        self.z_depth = 2 * self.initial_radius
        self.moving_z_plane = self.z_near

    def updateView(self, center, radius):
        """Computes the model view matrix so that camera is looking at an object
         without changing the target point or distance.

        :param center: center of the object to look at
        :type center: Vector3
        :param radius: radius of object to look at
        :type radius: float
        """
        self.initial_target = center if center is not None else self.initial_target
        self.initial_radius = radius if radius is not None else self.initial_radius

        rot = self.rot_matrix
        up = Vector3([rot.m21, rot.m22, rot.m23])

        self.lookAt(self.position, self.target, up)

        z_shift = (self.target - self.initial_target).length
        temp = 2 * (self.initial_radius + z_shift)
        self.moving_z_plane += (self.z_depth - temp) / 2
        self.z_depth = temp
        self.z_near = DEFAULT_Z_NEAR if self.moving_z_plane < DEFAULT_Z_NEAR else self.moving_z_plane
        self.z_far = self.z_near + self.z_depth

    def lookAt(self, position, target, up_dir=None):
        """Computes the model view matrix so that camera is looking at a target
        from a desired position and orientation.

        :param position: position of camera
        :type position: Vector3
        :param target: point to look at
        :type target: Vector3
        :param up_dir: up direction of camera
        :type up_dir: Union[Vector3, None]
        """
        self.position = position
        self.target = target
        self.model_view = Matrix44.identity()

        if position == target:
            self.model_view = Matrix44.fromTranslation(-position)
            self.rot_matrix = Matrix33.identity()
            return

        up = up_dir

        forward = position - target
        self.distance = forward.length

        forward.normalize()

        if up is None:
            if math.fabs(forward.x) < eps and math.fabs(forward.z) < eps:
                up = Vector3([0, 0, -1]) if forward.y > 0 else Vector3([0, 0, 1])
            else:
                up = Vector3([0, 1, 0])

        left = up ^ forward  # cross product
        left.normalize()

        up = forward ^ left

        self.rot_matrix.r1 = left
        self.rot_matrix.r2 = up
        self.rot_matrix.r3 = forward

        self.model_view.r1[:3] = left
        self.model_view.r2[:3] = up
        self.model_view.r3[:3] = forward

        trans = Vector3()
        trans.x = left.x * -position.x + left.y * -position.y + left.z * -position.z
        trans.y = up.x * -position.x + up.y * -position.y + up.z * -position.z
        trans.z = forward.x * -position.x + forward.y * -position.y + forward.z * -position.z
        self.model_view.c4[:3] = trans

    def pan(self, delta_x, delta_y):
        """Tilts the camera viewing axis vertically and/or horizontally while maintaining
        the camera position the view frustum (z near and far) ia also adjusted
        to avoid clipping.

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
        z_shift = (new_target - self.initial_target).length
        temp = 2 * (self.initial_radius + z_shift)
        self.moving_z_plane += (self.z_depth - temp) / 2
        self.z_depth = temp
        self.z_near = DEFAULT_Z_NEAR if self.moving_z_plane < DEFAULT_Z_NEAR else self.moving_z_plane
        self.z_far = self.z_near + self.z_depth

        self.target = new_target
        self.computeModelViewMatrix()

    def rotate(self, p1, p2):
        """Rotates the camera around the target using points in screen space

        :param p1: first point in screen space
        :type p1: Tuple[float, float]
        :param p2: second point in screen space
        :type p2: Tuple[float, float]
        """

        x1, y1 = p1
        x2, y2 = p2
        if x2 != x1 or y2 != y1:
            va = get_arcball_vector(x1, y1)
            vb = get_arcball_vector(x2, y2)

            angle = math.acos(clamp(va | vb, -1.0, 1.0))
            axis = (va ^ vb).normalized
            self.rot_matrix = angle_axis_to_matrix(angle, axis) @ self.rot_matrix
            self.computeModelViewMatrix()

    def zoom(self, delta):
        """Moves the camera forward or back along the viewing axis and adjusts
        the view frustum (z near and far) to avoid clipping.

        :param delta: offset by which camera is zoomed
        :type delta: float
        """
        # delta is scaled by distance so zoom is faster when object is farther
        distance = self.distance if self.distance >= 1.0 else 1
        offset = delta * distance

        # re-calculate view frustum
        self.moving_z_plane -= offset
        self.z_near = DEFAULT_Z_NEAR if self.moving_z_plane < DEFAULT_Z_NEAR else self.moving_z_plane
        self.z_far = self.z_near + self.z_depth

        # re-calculate camera distance
        distance -= offset
        self.distance = distance
        self.computeModelViewMatrix()

    def computeModelViewMatrix(self):
        """Computes the model view matrix of camera"""
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
        """Computes the one-point perspective projection matrix of camera

        :return: 4 x 4 perspective projection matrix
        :rtype: Matrix44
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

    @property
    def orthographic(self):
        """Computes the orthographic projection matrix of camera

        :return: 4 x 4 perspective projection matrix
        :rtype: Matrix44
        """
        projection = Matrix44()

        y_max = self.z_near * math.tan(0.5 * math.radians(self.fov))
        x_max = y_max * self.aspect

        z_depth = self.z_far - self.z_near

        projection.m11 = 1 / x_max
        projection.m22 = 1 / y_max
        projection.m33 = -2 / z_depth
        projection.m34 = (-self.z_far - self.z_near) / z_depth
        projection.m44 = 1

        return projection

    def viewFrom(self, direction):
        """Changes the viewing direction of the camera to one of six standard directions

        :param direction: camera viewing direction
        :type direction: Directions
        """
        if direction == Directions.Right:
            self.setViewDirection([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        elif direction == Directions.Left:
            self.setViewDirection([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        elif direction == Directions.Up:
            self.setViewDirection([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        elif direction == Directions.Down:
            self.setViewDirection([0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
        elif direction == Directions.Front:
            self.setViewDirection([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        else:
            self.setViewDirection([0.0, -1.0, 0.0], [0.0, 0.0, 1.0])

    def setViewDirection(self, forward, up=None):
        """Changes the viewing direction of the camera by setting the forward and up axis

        :param forward: forward axis
        :type forward: List[float]
        :param up: up direction of camera
        :type up: Union[List[float], None]
        """
        distance = self.distance if self.distance >= 1.0 else 1
        position = self.target - (Vector3(forward) * distance)
        up = up if up is None else Vector3(up)
        self.lookAt(position, self.target, up)

    def reset(self):
        """Resets the camera view"""
        self.position = Vector3()
        self.target = Vector3()
        self.distance = 0.0
        self.rot_matrix = Matrix33.identity()
        self.model_view = Matrix44.identity()
        self.mode = Camera.Projection.Perspective

        self.setViewDirection(self.direction, self.up)
        self.zoomToFit()
