from enum import Enum, unique
import numpy as np
from ..math.matrix import Matrix44
from ..mesh.colour import Colour
from ..mesh.create import create_sphere, create_plane
from ..mesh.utility import BoundingBox
from ...config import settings


class Node:
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
            self._colour = Colour.black()
        else:
            self._vertices = mesh.vertices
            self.indices = mesh.indices
            self.normals = mesh.normals
            self.bounding_box = mesh.bounding_box
            self._colour = mesh.colour

        self._render_mode = Node.RenderMode.Solid
        self.render_primitive = Node.RenderPrimitive.Triangles
        self.transform = Matrix44.identity()
        self.parent = None
        self._visible = True
        self.selected = False
        self.children = []

    def copy(self, transform=None):
        node = Node()
        node._vertices = self._vertices
        node.indices = self.indices
        node.normals = self.normals
        node.bounding_box = self.bounding_box
        node._colour = self._colour
        node._render_mode = self._render_mode
        node.render_primitive = self.render_primitive
        node.transform = self.transform if transform is None else transform
        node.parent = self.parent
        node._visible = self._visible
        node.selected = self.selected
        node.children = self.children

        return node

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

    @property
    def colour(self):
        if self._colour is None and self.parent:
            return self.parent.colour

        return self._colour

    @colour.setter
    def colour(self, value):
        self._colour = value

    @property
    def visible(self):
        if self._visible is None and self.parent:
            return self.parent.visible

        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value

    @property
    def render_mode(self):
        if self._render_mode is None and self.parent:
            return self.parent.render_mode

        return self._render_mode

    @render_mode.setter
    def render_mode(self, value):
        self._render_mode = value

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

        child_node.parent = self
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


def createSampleNode(samples, render_mode=Node.RenderMode.Solid, transform=None):
    sample_node = Node()
    sample_node.colour = Colour(*settings.value(settings.Key.Sample_Colour))
    sample_node.render_mode = render_mode
    sample_node.transform = transform if transform is not None else sample_node.transform

    for sample_mesh in samples.values():
        child = Node(sample_mesh)
        child.colour = None
        child.render_mode = None

        sample_node.addChild(child)

    return sample_node


def createFiducialNode(fiducials, visible=True, transform=None):
    fiducial_node = Node()
    fiducial_node.visible = visible
    fiducial_node.render_mode = Node.RenderMode.Solid
    fiducial_node.transform = transform if transform is not None else fiducial_node.transform
    enabled_colour = Colour(*settings.value(settings.Key.Fiducial_Colour))
    disabled_colour = Colour(*settings.value(settings.Key.Fiducial_Disabled_Colour))
    size = settings.value(settings.Key.Fiducial_Size)
    for point, enabled in fiducials:
        fiducial_mesh = create_sphere(size)
        fiducial_mesh.translate(point)

        child = Node(fiducial_mesh)
        child.colour = enabled_colour if enabled else disabled_colour
        child.render_mode = None
        child.visible = None

        fiducial_node.addChild(child)

    return fiducial_node


def createMeasurementPointNode(points, visible=True, transform=None):
    measurement_point_node = Node()
    measurement_point_node.visible = visible
    measurement_point_node.render_mode = Node.RenderMode.Solid
    measurement_point_node.transform = transform if transform is not None else measurement_point_node.transform
    enabled_colour = Colour(*settings.value(settings.Key.Measurement_Colour))
    disabled_colour = Colour(*settings.value(settings.Key.Measurement_Disabled_Colour))
    size = settings.value(settings.Key.Measurement_Size)
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
        child.colour = enabled_colour if enabled else disabled_colour
        child.render_mode = None
        child.visible = None
        child.render_primitive = Node.RenderPrimitive.Lines

        measurement_point_node.addChild(child)

    return measurement_point_node


def createMeasurementVectorNode(points, vectors, alignment, visible=True, transform=None):
    measurement_vector_node = Node()
    measurement_vector_node.visible = visible
    measurement_vector_node.render_mode = Node.RenderMode.Solid
    measurement_vector_node.transform = transform if transform is not None else measurement_vector_node.transform
    alignment = 0 if alignment >= vectors.shape[2] else alignment
    size = settings.value(settings.Key.Vector_Size)
    colours = [Colour(*settings.value(settings.Key.Vector_1_Colour)),
               Colour(*settings.value(settings.Key.Vector_2_Colour))]
    if vectors.shape[0] == 0:
        return measurement_vector_node

    for k in range(vectors.shape[2]):
        start_point = points.points
        for j in range(0, vectors.shape[1]//3):
            end_point = start_point + size * vectors[:, j*3:j*3+3, k]

            vertices = np.column_stack((start_point, end_point)).reshape(-1, 3)

            child = Node()
            child.vertices = vertices
            child.indices = np.arange(vertices.shape[0])
            if j < 2:
                child.colour = colours[j]
            else:
                np.random.seed(j)
                child.colour = Colour(*np.random.random(3))
            child.render_mode = None
            child.visible = alignment == k
            child.render_primitive = Node.RenderPrimitive.Lines

            measurement_vector_node.addChild(child)

    return measurement_vector_node


def createPlaneNode(plane, width, height):
    plane_mesh = create_plane(plane, width, height)

    node = Node(plane_mesh)
    node.render_mode = Node.RenderMode.Solid
    node.colour = Colour(*settings.value(settings.Key.Cross_Sectional_Plane_Colour))

    return node
