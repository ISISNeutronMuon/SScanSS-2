from enum import Enum, unique
import numpy as np
from .colour import Colour
from ..math.matrix import Matrix44
from ..math.vector import Vector3
from ..mesh.create import create_sphere, create_plane
from ..util.misc import BoundingBox


@unique
class RenderMode(Enum):
    Solid = 'Solid'
    Wireframe = 'Wireframe'
    Transparent = 'Transparent'


@unique
class RenderPrimitive(Enum):
    Lines = 'Lines'
    Triangles = 'Triangles'


def create_bounding_box(max_pos, min_pos):
    bb_max = Vector3(max_pos)
    bb_min = Vector3(min_pos)
    center = (bb_max + bb_min) / 2
    radius = ((bb_max - bb_min) / 2).length

    return BoundingBox(bb_max, bb_min, center, radius)


class Node:
    def __init__(self, mesh=None):
        if mesh is None:
            self._vertices = np.array([])
            self.indices = np.array([])
            self.normals = np.array([])
            self.bounding_box = None
        else:
            self._vertices = mesh.vertices
            self.indices = mesh.indices
            self.normals = mesh.normals
            self.bounding_box = mesh.bounding_box

        self.render_mode = RenderMode.Solid
        self.render_primitive = RenderPrimitive.Triangles

        self.transform = Matrix44.identity()
        self.colour = None

        self.children = []

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value
        if self.bounding_box is None:
            self.bounding_box = create_bounding_box(np.max(self._vertices, axis=0),
                                                    np.min(self._vertices, axis=0))
        else:
            max_pos = np.fmax(self.bounding_box.max, np.max(self._vertices, axis=0))
            min_pos = np.fmin(self.bounding_box.min, np.min(self._vertices, axis=0))
            self.bounding_box = create_bounding_box(max_pos, min_pos)

    def isEmpty(self):
        if not self.children and len(self.vertices) == 0:
            return True

        return False

    def addChild(self, child_node):
        self.children.append(child_node)

        if self.bounding_box is None:
            self.bounding_box = child_node.bounding_box
        else:
            max_pos = np.fmax(self.bounding_box.max, child_node.bounding_box.max)
            min_pos = np.fmin(self.bounding_box.min, child_node.bounding_box.min)
            self.bounding_box = create_bounding_box(max_pos, min_pos)


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
