from enum import Enum, unique
import numpy as np
from ..math.matrix import Matrix44
from ..mesh.colour import Colour
from ..mesh.create import create_sphere, create_plane
from ..mesh.utility import BoundingBox


@unique
class RenderMode(Enum):
    Solid = 'Solid'
    Wireframe = 'Wireframe'
    Transparent = 'Transparent'
    Outline = 'Outline'


@unique
class RenderPrimitive(Enum):
    Lines = 'Lines'
    Triangles = 'Triangles'


class Node:
    def __init__(self, mesh=None):
        """Create Node used in the GL widget.

        :param mesh: mesh to add to node
        :type mesh: Union[None, sscanss.core.mesh.Mesh]
        """
        if mesh is None:
            self._vertices = np.array([])
            self.indices = np.array([])
            self.normals = np.array([])
            self.bounding_box = None
            self.colour = None
        else:
            self._vertices = mesh.vertices
            self.indices = mesh.indices
            self.normals = mesh.normals
            self.bounding_box = mesh.bounding_box
            self.colour = mesh.colour

        self.render_mode = RenderMode.Solid
        self.render_primitive = RenderPrimitive.Triangles
        self.transform = Matrix44.identity()
        self.children = []

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        """ update the bounding box of the node when vertices are changed

        :param value: N x 3 array of vertices
        :type value: numpy.array
        """
        self._vertices = value
        max_pos, min_pos = BoundingBox.fromPoints(self._vertices).bounds
        for node in self.children:
            max_pos = np.fmax(node.bounding_box.max, max_pos)
            min_pos = np.fmin(node.bounding_box.min, min_pos)
        self.bounding_box = BoundingBox(max_pos, min_pos)

    def isEmpty(self):
        """ checks if Node is empty

        :return: True if empty else False
        :rtype: bool
        """
        if not self.children and len(self.vertices) == 0:
            return True
        return False

    def addChild(self, child_node):
        """ adds child to the node and recomputes the bounding box to include child

        :param child_node:
        :type child_node: sscanss.core.scene.Node
        """
        if child_node.isEmpty():
            return

        self.children.append(child_node)

        max_pos, min_pos = child_node.bounding_box.bounds
        if self.bounding_box is not None:
            max_pos = np.fmax(self.bounding_box.max, max_pos)
            min_pos = np.fmin(self.bounding_box.min, min_pos)
        self.bounding_box = BoundingBox(max_pos, min_pos)

    def translate(self, offset):
        """ translates node vertices and bounding box. NOTE:- because this class
        stores a reference to vertices, it will modify underlying vertex data and
        affect other references.

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: Union[numpy.ndarray, Vector3]
        """
        if self.isEmpty():
            return

        self._vertices += offset
        for child in self.children:
            child.translate(offset)
        self.bounding_box.translate(offset)


def createSampleNode(samples):
    sample_node = Node()
    sample_node.colour = Colour(0.4, 0.4, 0.4)
    sample_node.render_mode = RenderMode.Solid

    for _, sample_mesh in samples.items():
        child = Node(sample_mesh)
        child.colour = None
        child.render_mode = None

        sample_node.addChild(child)

    return sample_node


def createFiducialNode(fiducials):
    fiducial_node = Node()
    fiducial_node.render_mode = RenderMode.Solid

    for point, enabled in fiducials:
        fiducial_mesh = create_sphere(5)
        fiducial_mesh.translate(point)

        child = Node(fiducial_mesh)
        child.colour = Colour(0.4, 0.9, 0.4) if enabled else Colour(0.9, 0.4, 0.4)
        child.render_mode = None

        fiducial_node.addChild(child)

    return fiducial_node


def createMeasurementPointNode(points):
    size = 5
    measurement_point_node = Node()
    measurement_point_node.render_mode = RenderMode.Solid

    for point, enabled in points:
        x, y, z = point

        child = Node()
        child.vertices = np.array([[x - size, y, z],
                                   [x + size, y, z],
                                   [x, y - size, z],
                                   [x, y + size, z],
                                   [x, y, z - size],
                                   [x, y, z + size]])

        child.indices = np.array([0, 1, 2, 3, 4, 5])
        child.colour = Colour(0.01, 0.44, 0.12) if enabled else Colour(0.9, 0.4, 0.4)
        child.render_mode = None
        child.render_primitive = RenderPrimitive.Lines

        measurement_point_node.addChild(child)

    return measurement_point_node


def createMeasurementVectorNode(points, vectors, alignment):
    size = 10
    measurement_vector_node = Node()
    measurement_vector_node.render_mode = RenderMode.Solid

    if alignment >= vectors.shape[2]:
        return measurement_vector_node

    for index, point in enumerate(points):
        start_point, _ = point
        vector = vectors[index, 0:3, alignment]
        if np.linalg.norm(vector) == 0.0:
            continue

        end_point = start_point + size * vector

        child = Node()
        child.vertices = np.array([start_point, end_point])
        child.indices = np.array([0, 1])
        child.colour = Colour(0.0, 0.0, 1.0)
        child.render_mode = None
        child.render_primitive = RenderPrimitive.Lines

        measurement_vector_node.addChild(child)

        vector = vectors[index, 3:6, alignment]
        if np.linalg.norm(vector) == 0.0:
            continue

        end_point = start_point + size * vector

        child = Node()
        child.vertices = np.array([start_point, end_point])
        child.indices = np.array([0, 1])
        child.colour = Colour(1.0, 0.0, 0.0)
        child.render_mode = None
        child.render_primitive = RenderPrimitive.Lines

        measurement_vector_node.addChild(child)

    return measurement_vector_node


def createPlaneNode(plane, width, height):
    plane_mesh = create_plane(plane, width, height)

    node = Node(plane_mesh)
    node.render_mode = RenderMode.Solid
    node.colour = Colour(0.93, 0.83, 0.53)

    return node
