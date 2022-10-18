"""
Classes for scene node
"""
import copy
import ctypes
from enum import Enum, unique
import numpy as np
from OpenGL import GL
from PyQt5 import QtGui
from .shader import VertexArray, Texture1D, Texture3D, Text3D
from ..math.matrix import Matrix44
from ..geometry.colour import Colour
from ..geometry.mesh import BoundingBox
from ..geometry.primitive import create_cuboid
from ...config import settings


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

    def draw(self, renderer):
        """Recursively renders node with its children on the given renderer

        :param renderer: OpenGl renderer instance
        :type renderer: OpenGLRenderer
        """
        if not self.visible or self.buffer is None:
            return

        GL.glPushMatrix()
        GL.glPushAttrib(GL.GL_CURRENT_BIT)
        GL.glMultTransposeMatrixf(self.transform)

        mode = Node.RenderMode.Solid if self.render_mode is None else self.render_mode
        if mode == Node.RenderMode.Transparent:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_SRC_COLOR)
        elif mode == Node.RenderMode.Solid:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        elif mode == Node.RenderMode.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

        program = renderer.shader_programs['default']
        if self.vertices.size > 0 and self.indices.size > 0:
            if self.normals.size > 0:
                program = renderer.shader_programs['mesh']

            program.bind()
            self.buffer.bind()

            primitive = GL.GL_TRIANGLES if self.render_primitive == Node.RenderPrimitive.Triangles else GL.GL_LINES

            self._drawHelper(primitive)

            self.buffer.release()
            program.release()

            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)

        for child in self.children:
            child.draw(renderer)

        GL.glPopAttrib()
        GL.glPopMatrix()

    def _drawHelper(self, primitive):
        """Helper for drawing the given primitive

        :param primitive: OpenGL primitive to render
        :type primitive: OpenGL.constant.IntConstant
        """
        if self.selected:
            GL.glColor4f(*settings.value(settings.Key.Selected_Colour))
        else:
            GL.glColor4f(*self.colour.rgbaf)

        if self.outlined:
            self.drawOutline(primitive, self.buffer.count)

        GL.glDrawElements(primitive, self.buffer.count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def drawOutline(self, primitive, count, offset=0):
        """Renders the red outline of the bound vertex array

        :param primitive: OpenGL primitive to render
        :type primitive: OpenGL.constant.IntConstant
        :param count: number of elements in array to draw
        :type count: int
        :param offset: start index in vertex array
        :type offset: int
        """
        old_colour = GL.glGetDoublev(GL.GL_CURRENT_COLOR)
        old_line_width = GL.glGetInteger(GL.GL_LINE_WIDTH)
        polygon_mode = GL.glGetIntegerv(GL.GL_POLYGON_MODE)
        GL.glColor3f(1, 0, 0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glLineWidth(3)
        GL.glCullFace(GL.GL_FRONT)
        GL.glEnable(GL.GL_CULL_FACE)
        # First Pass
        GL.glDrawElements(primitive, count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(offset))

        GL.glColor4dv(old_colour)
        GL.glLineWidth(old_line_width)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, polygon_mode[0])


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

    def _drawHelper(self, primitive):
        start = 0
        for index, end in enumerate(self.batch_offsets):
            if self.selected[index]:
                GL.glColor4f(*settings.value(settings.Key.Selected_Colour))
            else:
                GL.glColor4f(*self.per_object_colour[index].rgbaf)
            GL.glPushMatrix()
            t = Matrix44.identity() if not self.per_object_transform else self.per_object_transform[index]
            GL.glMultTransposeMatrixf(t)

            count = end - start
            offset = start * self.vertices.itemsize

            if self.outlined[index]:
                self.drawOutline(primitive, count, offset)

            GL.glDrawElements(primitive, count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(offset))

            GL.glPopMatrix()
            start = end


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

    def _drawHelper(self, primitive):
        for index, transform in enumerate(self.per_object_transform):
            GL.glPushMatrix()
            GL.glMultTransposeMatrixf(transform)
            if self.selected[index]:
                GL.glColor4f(*settings.value(settings.Key.Selected_Colour))
            else:
                GL.glColor4f(*self.per_object_colour[index].rgbaf)

            if self.outlined[index]:
                self.drawOutline(primitive, self.buffer.count)

            GL.glDrawElements(primitive, self.buffer.count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            GL.glPopMatrix()


class VolumeNode(Node):
    """Creates Node object for volume rendering.

    :param volume: volume object
    :type volume: Volume
    """
    def __init__(self, volume):
        super().__init__()

        self.render_primitive = Node.RenderPrimitive.Volume

        self.volume = Texture3D(volume.render_target)
        self.transfer_function = Texture1D(volume.curve.transfer_function)

        volume_mesh = create_cuboid(2, 2, 2)
        self.vertices = volume_mesh.vertices
        self.indices = volume_mesh.indices

        self.extent = volume.extent
        self._transform = Matrix44.identity()
        self.model_matrix = volume.transform_matrix

        self.scale_matrix = np.diag([*(0.5 * self.extent), 1])

    def updateTransferFunction(self, transfer_function):
        """Updates node transfer function"""
        self.transfer_function = Texture1D(transfer_function)

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

    def draw(self, renderer):
        if not self.visible:
            return

        GL.glPushMatrix()
        GL.glPushAttrib(GL.GL_CURRENT_BIT)
        GL.glMultTransposeMatrixf(self.transform)

        mode = Node.RenderMode.Solid if self.render_mode is None else self.render_mode

        GL.glEnable(GL.GL_BLEND)
        if mode == Node.RenderMode.Transparent:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_SRC_COLOR)
        elif mode == Node.RenderMode.Solid:
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        elif mode == Node.RenderMode.Wireframe:
            GL.glBlendFunc(GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_SRC_ALPHA)

        program = renderer.shader_programs['volume']
        program.bind()
        self.buffer.bind()
        self.volume.bind(GL.GL_TEXTURE0)
        self.transfer_function.bind(GL.GL_TEXTURE1)

        GL.glPushMatrix()
        GL.glMultTransposeMatrixf(self.scale_matrix)

        align_transform = self.transform
        view_matrix = np.array(renderer.scene.camera.model_view @ align_transform, np.float32)
        focal_length = 1 / np.tan(np.pi / 180 * renderer.scene.camera.fov / 2)
        inverse_view_proj = np.linalg.inv(renderer.scene.camera.projection @ view_matrix)
        ratio = renderer.devicePixelRatioF()

        program.setUniform('view', view_matrix, transpose=True)
        program.setUniform('inverse_view_proj', inverse_view_proj, transpose=True)
        program.setUniform('aspect_ratio', renderer.scene.camera.aspect)
        program.setUniform('focal_length', focal_length)
        program.setUniform('viewport_size', [renderer.width() * ratio, renderer.height() * ratio])
        program.setUniform('top', self.top)
        program.setUniform('bottom', self.bottom)
        program.setUniform('step_length', 0.001)
        program.setUniform('gamma', 2.2)
        program.setUniform('volume', 0)
        program.setUniform('transfer_func', 1)
        program.setUniform('highlight', self.selected or self.outlined)

        self.buffer.bind()
        self._drawHelper(GL.GL_TRIANGLES)
        self.volume.release()
        self.transfer_function.release()
        self.buffer.release()
        program.release()
        GL.glPopMatrix()

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        for child in self.children:
            child.draw(renderer)

        GL.glPopAttrib()
        GL.glPopMatrix()

    def _drawHelper(self, primitive):
        if self.selected:
            GL.glColor4f(*settings.value(settings.Key.Selected_Colour))

        if self.outlined:
            GL.glColor4f(1, 0, 0, 1)

        GL.glDrawElements(primitive, self.buffer.count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def drawOutline(self, primitive, count, offset=0):
        raise NotImplementedError('drawOutline is not implemented for VolumeNode')


class TextNode(Node):
    """Creates Node object for text rendering.

    :param text: text
    :type text: str
    :param position: 3D position of text
    :type position: Tuple[float, float, float]
    :param colour: colour of text
    :type colour: QtGui.QColor
    :param font: font
    :type font: QtGui.QFont
    """
    def __init__(self, text, position, colour, font):
        super().__init__()

        size = 200
        image_font = QtGui.QFont(font)
        image_font.setPixelSize(size)
        metric = QtGui.QFontMetrics(image_font)
        rect = metric.boundingRect(text)
        image = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_RGBA8888)
        image.fill(0)

        # create texture image
        painter = QtGui.QPainter()
        painter.begin(image)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        painter.setFont(image_font)
        painter.setPen(colour)
        painter.drawText(0, metric.ascent(), text)
        painter.end()

        metric = QtGui.QFontMetrics(font)
        rect = metric.boundingRect(text)
        self.size = (rect.width(), rect.height())

        self.text = text
        self.position = position
        if text:
            ptr = image.constBits()
            ptr.setsize(image.byteCount())
            self.image_data = np.array(ptr).reshape((image.height(), image.width(), 4))
        else:
            self.image_data = np.array([])

    def isEmpty(self):
        return False if self.text else True

    def buildVertexBuffer(self):
        """Creates vertex buffer object for the node"""
        if not self.isEmpty():
            self.buffer = Text3D(self.size, self.image_data)

    def draw(self, renderer):
        if not self.visible and self.buffer is None:
            return

        GL.glPushAttrib(GL.GL_CURRENT_BIT)

        program = renderer.shader_programs['text']
        program.bind()
        self.buffer.bind()
        program.setUniform('viewport_size', [renderer.width(), renderer.height()])
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        text_pos, ok = renderer.project(*self.position)
        if not ok:
            return

        program.setUniform('screen_pos', [*text_pos])
        self._drawHelper(GL.GL_TRIANGLES)
        GL.glDisable(GL.GL_BLEND)
        self.buffer.release()
        program.release()

        for child in self.children:
            child.draw(renderer)

        GL.glPopAttrib()

    def drawOutline(self, primitive, count, offset=0):
        raise NotImplementedError('drawOutline is not implemented for TextNode')
