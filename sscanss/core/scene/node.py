"""
Classes for scene node
"""
import copy
from enum import Enum, unique
import numpy as np
from .shader import VertexArray, Texture1D, Texture3D
from ..math.matrix import Matrix44
from ..geometry.colour import Colour
from ..geometry.mesh import BoundingBox
from ..geometry.primitive import create_cuboid


class Node:
    """Creates Node object.

    :param mesh: mesh to add to node
    :type mesh: Union[Mesh, None]
    """
    @unique
    class RenderMode(Enum):
        """Mode for rendering"""
        Solid = 'Solid'
        Wireframe = 'Wireframe'
        Transparent = 'Transparent'

    @unique
    class RenderPrimitive(Enum):
        """Type of primitive to render"""
        Lines = 'Lines'
        Triangles = 'Triangles'
        Volume = 'Volume'

    def __init__(self, mesh=None):
        self.parent = None
        self.children = []
        self.buffer = None

        if mesh is None:
            self._vertices = np.array([])
            self._indices = np.array([])
            self._normals = np.array([])
            self._bounding_box = None
            self._colour = Colour.black()
        else:
            self.vertices = mesh.vertices
            self.indices = mesh.indices
            self.normals = mesh.normals
            self._colour = mesh.colour

        self._render_mode = Node.RenderMode.Solid
        self.render_primitive = Node.RenderPrimitive.Triangles
        self.transform = Matrix44.identity()
        self._visible = True
        self.selected = False
        self.outlined = False

    def buildVertexBuffer(self):
        """Creates vertex buffer object for the node"""
        if self.vertices.size > 0 and self.indices.size > 0:
            self.buffer = VertexArray(self.vertices, self.indices, self.normals)

    def resetOutline(self):
        """Sets outlined property to False"""
        self.outlined = False

    def copy(self, transform=None, parent=None):
        """Creates shallow copy of node with unique transformation matrix

        :param transform: transformation matrix
        :type transform: Union[Matrix44, None]
        :param parent: parent node
        :type parent: Union[Node, None]
        :return: shallow copy of node
        :rtype: Node
        """
        node = copy.copy(self)
        if transform is not None:
            node.transform = transform

        if parent is not None:
            node.parent = parent

        children = []
        for i in range(len(node.children)):
            children.append(node.children[i].copy(parent=node))

        node.children = children

        return node

    @property
    def normals(self):
        """Gets and sets vertex normals

        :return: array of vertex normals
        :rtype: numpy.ndarray
        """
        return self._normals

    @normals.setter
    def normals(self, value):
        self._normals = value.astype(np.float32)

    @property
    def indices(self):
        """Gets and sets face indices

        :return: array of vertices
        :rtype: numpy.ndarray
        """
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = value.astype(np.uint32)

    @property
    def vertices(self):
        """Gets and sets vertices and updates the bounding box of the node when vertices are changed

        :return: array of vertices
        :rtype: numpy.ndarray
        """
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value.astype(np.float32)
        max_pos, min_pos = BoundingBox.fromPoints(self._vertices).bounds
        for node in self.children:
            max_pos = np.maximum(node.bounding_box.max, max_pos)
            min_pos = np.minimum(node.bounding_box.min, min_pos)
        self.bounding_box = BoundingBox(max_pos, min_pos)

    @property
    def colour(self):
        """Gets and sets node colour. Parent node colour is returned if
        node colour is None

        :return: node colour
        :rtype: Colour
        """
        if self._colour is None and self.parent:
            return self.parent.colour

        return self._colour

    @colour.setter
    def colour(self, value):
        self._colour = value

    @property
    def visible(self):
        """Gets and sets node visibility state. Parent node visibility is returned if
        node visibility state is None

        :return: indicate node is visible
        :rtype: bool
        """
        if self._visible is None and self.parent:
            return self.parent.visible

        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value

    @property
    def render_mode(self):
        """Gets and sets node render mode. Parent node render mode is returned if
        node render mode is None

        :return: node render mode
        :rtype: RenderMode
        """
        if self._render_mode is None and self.parent:
            return self.parent.render_mode

        return self._render_mode

    @render_mode.setter
    def render_mode(self, value):
        self._render_mode = value

    def isEmpty(self):
        """Checks if Node is empty

        :return: indicates node is empty
        :rtype: bool
        """
        if not self.children and len(self.vertices) == 0:
            return True
        return False

    def addChild(self, child_node):
        """Adds child to the node and recomputes the bounding box

        :param child_node: child node to add
        :type child_node: Node
        """
        if child_node.isEmpty():
            return

        child_node.parent = self
        self.children.append(child_node)

        max_pos, min_pos = child_node.bounding_box.bounds
        if self.bounding_box is not None:
            max_pos = np.maximum(self.bounding_box.max, max_pos)
            min_pos = np.minimum(self.bounding_box.min, min_pos)
        self.bounding_box = BoundingBox(max_pos, min_pos)

    def flatten(self):
        """Flattens the tree formed by nested nodes recursively

        :return: flattened node
        :rtype: Node
        """
        new_node = Node()
        new_node.bounding_box = self.bounding_box
        for node in self.children:
            if node.children:
                new_node.children.extend(node.flatten().children)
            elif not node.isEmpty():
                node.parent = None
                new_node.children.append(node)

        if len(self.vertices) != 0:
            parent = self.copy()
            parent.vertices = self.vertices
            new_node.children.append(parent)

        return new_node

    @property
    def bounding_box(self):
        """Gets and sets node bounding box. The bounding box is transformed using
        the node's transformation matrix so it may not be tight

        :return: node render mode
        :rtype: Union[BoundingBox, None]
        """
        return None if self._bounding_box is None else self._bounding_box.transform(self.transform)

    @bounding_box.setter
    def bounding_box(self, value):
        self._bounding_box = value


class BatchRenderNode(Node):
    """Creates Node object for batch rendering. The vertices of multiple drawable
    objects are place in the same array to optimize performance

    :param object_count: number of drawable objects
    :type object_count: int
    """
    def __init__(self, object_count):
        super().__init__()

        self.batch_offsets = [0] * object_count
        self.per_object_colour = [Colour.black()] * object_count
        self.per_object_transform = [Matrix44.identity()] * object_count
        self.selected = [False] * object_count
        self.resetOutline()

    def resetOutline(self):
        self.outlined = [False] * len(self.batch_offsets)


class InstanceRenderNode(Node):
    """Creates Node object for instance rendering. The same vertices will be redrawn
    multiple time with the different per object transform

    :param object_count: number of drawable objects
    :type object_count: int
    """
    def __init__(self, object_count):
        super().__init__()

        self.per_object_colour = [Colour.black()] * object_count
        self.per_object_transform = [Matrix44.identity()] * object_count
        self.selected = [False] * object_count
        self.resetOutline()

    def resetOutline(self):
        self.outlined = [False] * len(self.per_object_transform)


class VolumeRenderNode(Node):
    """Creates Node object for volume rendering.

    :param volume: volume object
    :type volume: Volume
    """
    def __init__(self, volume):
        super().__init__()

        self.render_primitive = Node.RenderPrimitive.Volume

        self.volume = Texture3D(volume.data)
        self.transfer_function = Texture1D(volume.curve.transfer_function)

        volume_mesh = create_cuboid(2, 2, 2)
        self.vertices = volume_mesh.vertices
        self.indices = volume_mesh.indices

        self.extent = volume.extent
        self._transform = Matrix44.identity()
        self.model_matrix = volume.transform_matrix
        self.scale_matrix = np.diag([*(0.5 * self.extent), 1])

    @property
    def top(self):
        """Returns top coordinates of volume

        :return: top coordinates
        :rtype: numpy.ndarray
        """
        return self.extent / 2

    @property
    def bottom(self):
        """Returns bottom coordinates of volume

        :return: bottom coordinates
        :rtype: numpy.ndarray
        """
        return -self.extent / 2

    @property
    def transform(self):
        """Gets and sets node transform matrix.

        :return: node transform matrix
        :rtype: Matrix44
        """
        return self._transform @ self.model_matrix

    @transform.setter
    def transform(self, value):
        self._transform = value

    @property
    def bounding_box(self):
        """Gets and sets node bounding box. The bounding box is transformed using
        the node's transformation matrix so it may not be tight

        :return: node render mode
        :rtype: Union[BoundingBox, None]
        """
        transform = self.transform @ self.scale_matrix
        return None if self._bounding_box is None else self._bounding_box.transform(transform)

    @bounding_box.setter
    def bounding_box(self, value):
        self._bounding_box = value
