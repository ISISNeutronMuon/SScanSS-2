from enum import Enum, unique
from .matrix import Matrix44
from .transform import rotation_btw_vectors
from .quaternion import Quaternion, QuaternionVectorPair
from .vector import Vector3


class SerialManipulator:
    def __init__(self, links, base=None, tool=None):
        self.links = links
        self.base = Matrix44.identity() if base is None else base
        self.tool = Matrix44.identity() if tool is None else tool

    def fkine(self, q, start_index=0, end_index=None, include_base=True):
        link_count = self.numberOfLinks

        start = 0 if start_index < 0 else start_index
        end = link_count if end_index is None or end_index > link_count else end_index

        base = self.base if include_base else Matrix44.identity()
        tool = self.tool if end == link_count else Matrix44.identity()

        qs = QuaternionVectorPair.identity()
        for i in range(start, end):
            # clamp q between limit and provide option to disable this
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
        from sscanss.core.scene import Node, Colour, RenderMode
        node = Node()
        node.colour = Colour(0.4, 0.5, 0.2)
        node.render_mode = RenderMode.Solid

        qs = QuaternionVectorPair.identity()
        joint_pos = Vector3()
        up = Vector3([0., 0., 1.])
        for link in self.links:
            qs *= link.quaterionVectorPair
            rot = rotation_btw_vectors(up, link.joint_axis)
            m = Matrix44.identity()
            m[0:3, 0:3] = qs.quaternion.toMatrix()*rot
            m[0:3, 3] = joint_pos if link.type == Link.Type.Revolute else qs.vector

            child = Node(link.mesh)
            child.colour = None
            child.render_mode = None
            child.transform = m

            node.addChild(child)
            joint_pos = qs.vector

        return node


class Link:
    @unique
    class Type(Enum):
        Revolute = 0
        Prismatic = 1

    def __init__(self, axis, point, joint_type, angle=0.0, upper_limit=None, lower_limit=None, mesh=None):
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
