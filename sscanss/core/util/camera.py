import math
import numpy as np
from pyrr import Vector3, Quaternion, Matrix44
from OpenGL import GLU, GL

EPSILON = 0.00001
DEFAULT_CLIP_PLANE_NEAR = 0.001
DEFAULT_CLIP_PLANE_FAR = 3000.0


def get_eulers(matrix):
    
    roll = pitch = 0.0
    yaw = math.asin(matrix.m13)

    if matrix.m33 < 0:
        yaw = math.pi - yaw if yaw >= 0 else -math.pi - yaw

    if matrix.m11 > -EPSILON and matrix.m11 < EPSILON:
        roll = 0.0
        pitch = math.atan2(matrix.m21, matrix.m22)
    else:
        roll = math.atan2(-matrix.m12, matrix.m11)
        pitch = math.atan2(-matrix.m23, matrix.m33)

    return Vector3([pitch, yaw, roll])


def matrix_from_xyz_eulers(angles, dtype=None):
    x = angles[0]
    y = angles[1]
    z = angles[2]

    sx = math.sin(x)
    cx = math.cos(x)
    sy = math.sin(y)
    cy = math.cos(y)
    sz = math.sin(z)
    cz = math.cos(z)

    return Matrix44(np.array(
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
                cx * cz - sx * sy *sz,
                -cy * sx,
            ],
            # m3
            [
                -cx * cz * sy + sx * sz,
                cz * sx + cx * sy * sz,
                cx * cy,
            ]
        ],
        dtype=dtype
    ))

class Camera:
    def __init__(self, width, height, fov):
        self.clipplanenear = DEFAULT_CLIP_PLANE_NEAR
        self.clipplanefar = DEFAULT_CLIP_PLANE_FAR
        self.aspect = width / height
        self.fov = fov

        self.position = Vector3()
        self.target = Vector3()
        self.quaternion = Quaternion()
        self.matrix = Matrix44.identity()
        self.rot_matrix = Matrix44.identity()
        self.angle = Vector3()
        self.distance = 0.0

    def zoomToFit(self, center, radius):
        direction = Vector3([0.0, 0.0, -1.0])
        half_min_fov_in_radians = 0.5 * (self.fov * math.pi / 180)

        if self.aspect < 1.0:
            # fov in x is smaller
            half_min_fov_in_radians = math.atan(
                self.aspect * math.tan(half_min_fov_in_radians))

        distance_to_center = radius / math.sin(half_min_fov_in_radians)
        eye = center - direction * distance_to_center

        self.lookAt(eye, center, Vector3([0.0, 1.0, 0.0]))

    def setPerspective(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(self.fov, self.aspect, 
                          self.clipplanenear, self.clipplanefar)

    def lookAt(self, position, target, up_dir=None):
        self.position = position
        self.target = target
        self.matrix = Matrix44.identity()

        if position == target:
            self.matrix.from_translation(-position)
            self.rot_matrix = Matrix44.identity()
            self.angle = Vector3()
            self.quaternion = Quaternion()

            return

        left = Vector3()
        up = up_dir
        forward = Vector3()

        forward = position - target
        self.distance = forward.length

        forward.normalize()

        if up is None:
            condition = math.fabs(forward.x) < EPSILON and math.fabs(forward.z) < EPSILON
            if condition:
                up = Vector3([0, 0, -1]) if forward.y > 0 else Vector3([0, 0, 1])
            else:
                up = Vector3([0, 1, 0])

        left = up ^ forward # cross product
        left.normalize()

        up = forward ^ left

        self.rot_matrix.r1[:3] = left
        self.rot_matrix.r2[:3] = up
        self.rot_matrix.r3[:3] = forward

        self.matrix.r1[:3] = left
        self.matrix.r2[:3] = up
        self.matrix.r3[:3] = forward
        
        trans = self.matrix * -position
        self.matrix.c4[:3] = trans

        self.angle = get_eulers(self.rot_matrix) * 180/math.pi

    def pan(self, delta):
        # get left & up vectors of camera
        camera_left = Vector3([self.matrix.m11, self.matrix.m12, self.matrix.m13])
        camera_up = Vector3([self.matrix.m21, self.matrix.m22, self.matrix.m23])

        # compute delta movement
        delta_movement = delta.x * camera_left
        delta_movement += -delta.y * camera_up;   # reverse up direction

        # find new target position
        new_target = self.target + delta_movement

        self.target = new_target
        self.computeMatrix()

    def rotate(self, delta):
        self.angle = self.angle + delta
        self.rot_matrix = matrix_from_xyz_eulers(self.angles * math.pi / 180)
        self.computeMatrix()

    def zoom(self, distance):
        self.distance = distance
        self.computeMatrix()

    def computeMatrix(self):
        
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


        self.matrix = Matrix44.identity()
        self.matrix.c1[:3] = left
        self.matrix.c2[:3] = up
        self.matrix.c3[:3] = forward
        self.matrix.c4[:3] = trans

        forward = Vector3([-self.matrix.m31, -self.matrix.m32, -self.matrix.m33])
        self.position = target - (dist * forward)

    def reset(self):
        origin = Vector3([0.0, 0.0, 0.0])

        self.lookAt(origin, origin, Vector3([0.0, 1.0, 0.0]))
