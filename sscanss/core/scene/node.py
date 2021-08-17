"""
Classes for scene node
"""
import copy
from enum import Enum, unique
import numpy as np
from .shader import VertexArray
from ..math.matrix import Matrix44
from ..geometry.colour import Colour
from ..geometry.mesh import BoundingBox


class Node:
    """Creates Node object.

    :param mesh: mesh to add to node
    :type mesh: Union[Mesh, None]
    """
    @unique
    class RenderMode(Enum):
        Solid = 'Solid'
        Wireframe = 'Wireframe'
        Transparent = 'Transparent'

    @unique
    class RenderPrimitive(Enum):
        Lines = 'Lines'
        Triangles = 'Triangles'

    def __init__(self, mesh=None):
        if mesh is None:
            self._vertices = np.array([])
            self.indices = np.array([])
            self.normals = np.array([])
            self._bounding_box = None
            self._colour = Colour.black()
        else:
            self._vertices = mesh.vertices
            self.indices = mesh.indices
            self.normals = mesh.normals
            self._bounding_box = mesh.bounding_box
            self._colour = mesh.colour

        self._render_mode = Node.RenderMode.Solid
        self.render_primitive = Node.RenderPrimitive.Triangles
        self.transform = Matrix44.identity()
        self.parent = None
        self._visible = True
        self.selected = False
        self.outlined = False
        self.children = []
        self.buffer = None

    def buildVertexBuffer(self):
        """Creates vertex buffer object for the node"""
        if self.vertices.size > 0 and self.indices.size > 0:
            self.buffer = VertexArray(self.vertices, self.indices, self.normals)

    def resetOutline(self):
        """Sets outlined property to False"""
        self.outlined = False

    def copy(self, transform=None):
        """Creates shallow copy of node with unique transformation matrix

        :param transform: transformation matrix
        :type transform: Union[Matrix44, None]
        :return: shallow copy of node
        :rtype: Node
        """
        node = copy.copy(self)
        if transform is not None:
            node.transform = transform
        return node

    @property
    def vertices(self):
        """Gets and sets vertices and updates the bounding box of the node when vertices are changed

        :return: array of vertices
        :rtype: numpy.ndarray
        """
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value
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
