from enum import Enum, unique
import numpy as np
from ..math.matrix import Matrix44
from ..math.transform import rotation_btw_vectors
from ..math.quaternion import Quaternion, QuaternionVectorPair
from ..math.vector import Vector3
from ..scene.node import Node, RenderMode


class SerialManipulator:
    def __init__(self, links, base=None, tool=None, base_mesh=None):
        self.links = links
        self.base = Matrix44.identity() if base is None else base
        self.tool = Matrix44.identity() if tool is None else tool
        self.base_mesh = base_mesh

    def fkine(self, q, start_index=0, end_index=None, include_base=True):
        link_count = self.numberOfLinks

        start = 0 if start_index < 0 else start_index
        end = link_count if end_index is None or end_index > link_count else end_index

        base = self.base if include_base else Matrix44.identity()
        tool = self.tool if end == link_count else Matrix44.identity()

        qs = QuaternionVectorPair.identity()
        for i in range(start, end):
            #TODO: clamp q between limit and provide option to disable this
            self.links[i].move(q[i])
            qs *= self.links[i].quaterionVectorPair

        return base * qs.toMatrix() * tool

    @property
    def numberOfLinks(self):
        return len(self.links)

    def __getitem__(self, index):
        return self.links[index]

    def __setitem__(self, index, link):
        if not isinstance(link, Link):
            raise ValueError('value must be a Link object')

        self.links[index] = link

    def model(self):

        node = Node()
        node.render_mode = RenderMode.Solid

        if self.base_mesh is not None:
            child = Node(self.base_mesh)
            child.render_mode = None
            child.transform = self.base

            node.addChild(child)

        qs = QuaternionVectorPair.identity()
        joint_pos = Vector3()
        up = Vector3([0., 0., 1.])
        for link in self.links:
            qs *= link.quaterionVectorPair
            rot = rotation_btw_vectors(up, link.joint_axis)
            m = Matrix44.identity()
            m[0:3, 0:3] = qs.quaternion.toMatrix() * rot
            m[0:3, 3] = joint_pos if link.type == Link.Type.Revolute else qs.vector

            m = self.base * m
            if link.mesh is not None:
                transformed_mesh = link.mesh.transformed(m)
                child = Node(transformed_mesh)
                child.render_mode = None

                node.addChild(child)
            joint_pos = qs.vector

        return node


class Link:
    @unique
    class Type(Enum):
        Revolute = 0
        Prismatic = 1

    def __init__(self, axis, point, joint_type, angle=0.0, upper_limit=None, lower_limit=None,
                 mesh=None, name=''):
        self.joint_axis = Vector3(axis)

        if self.joint_axis.length < 0.00001:
            raise ValueError('The joint axis cannot be a zero vector.')

        self.quaternion = Quaternion.fromAxisAngle(self.joint_axis, angle)
        self.vector = Vector3(point)
        self.home = Vector3(point)
        self.type = joint_type
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.mesh = mesh
        self.name = name

    def move(self, offset):
        if self.type == Link.Type.Revolute:
            self.quaternion = Quaternion.fromAxisAngle(self.joint_axis, offset)
            self.vector = self.quaternion.rotate(self.home)
        else:
            self.vector = self.home + self.joint_axis * offset

    @property
    def transformationMatrix(self):
        return self.quaterionVectorPair.toMatrix()

    @property
    def quaterionVectorPair(self):
        return QuaternionVectorPair(self.quaternion, self.vector)


def joint_space_trajectory(start_pose, stop_pose, time, step):
    dof = len(start_pose)
    trajectory = np.zeros((step, dof))

    for i in range(dof):
        t = cubic_polynomial_trajectory(start_pose[i], stop_pose[i], time, step=step)
        trajectory[:, i] = t

    return trajectory


def cubic_polynomial_trajectory(p0, p1, tf, t0=0.0, step=100, v0=0.0, v1=0.0, derivative=False):
    t = np.linspace(t0, tf, step)

    t0_2 = t0 * t0
    t0_3 = t0_2 * t0

    tf_2 = tf * tf
    tf_3 = tf_2 * tf

    M = [[1.0, t0, t0_2, t0_3],
         [0.0, 1.0, 2 * t0, 3 * t0_2],
         [1.0, tf, tf_2, tf_3],
         [0.0, 1.0, 2 * tf, 3 * tf_2]]

    b = [p0, v0, p1, v1]
    a = np.dot(np.linalg.inv(M), b)

    pd = np.polyval(a[::-1], t)

    if not derivative:
        return pd

    aa = a[1:] * [1, 2, 3]
    vd = np.polyval(aa[::-1], t)

    return pd, vd
