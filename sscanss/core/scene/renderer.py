import ctypes
import numpy as np
from OpenGL import GL, error
from PyQt5 import QtCore, QtGui, QtWidgets
from .camera import Camera, world_to_screen, screen_to_world
from .node import Node, InstanceRenderNode, BatchRenderNode
from .scene import Scene
from .shader import DefaultShader, GouraudShader
from ..geometry.colour import Colour
from ..math.matrix import Matrix44
from ..math.vector import Vector3
from ..util.misc import Attributes
from ...config import settings


class OpenGLRenderer(QtWidgets.QOpenGLWidget):
    """Provides OpenGL widget for draw 3D scene for the sample setup and instrument

    :param parent: main window instance
    :type parent: MainWindow
    """

    pick_added = QtCore.pyqtSignal(object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.scene = Scene()
        self.show_bounding_box = False
        self.show_coordinate_frame = True
        self.picks = []
        self.picking = False
        self.default_font = QtGui.QFont("Times", 10)
        self.error = False
        self.custom_error_handler = None
        self.shader_programs = {}

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def cleanup(self):
        self.makeCurrent()
        del self.scene
        for key in self.shader_programs.keys():
            self.shader_programs[key].destroy()
        self.doneCurrent()

    @property
    def picking(self):
        return self._picking

    @picking.setter
    def picking(self, value):
        """Enables/Disables point picking

        :param value: indicates if point picking is enabled
        :type value: bool
        """
        self._picking = value
        if value:
            self.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def initializeGL(self):
        try:
            GL.glClearColor(*Colour.white())
            GL.glColor4f(*Colour.black())

            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glEnable(GL.GL_MULTISAMPLE)

            number_of_lights = self.initLights()
            # Create and compile GLSL shaders program
            self.shader_programs["mesh"] = GouraudShader(number_of_lights)
            self.shader_programs["default"] = DefaultShader()

            self.context().aboutToBeDestroyed.connect(self.cleanup)

        except error.GLError:
            self.parent.showMessage(
                "An error occurred during OpenGL initialization. "
                "The minimum OpenGL requirement for this software is version 2.0.\n\n"
                "This error may be caused by:\n"
                "* A missing or faulty graphics driver installation.\n"
                "* Accessing SScanSS 2 from a remote connection with GPU rendering disabled.\n\n"
                "The software will be closed now."
            )
            raise

    def initLights(self):
        """Sets up light properties"""
        ambient = [0.0, 0.0, 0.0, 1.0]
        diffuse = [0.5, 0.5, 0.5, 1.0]
        specular = [0.2, 0.2, 0.2, 1.0]

        # set up light direction
        front = [0.0, 0.0, 1.0, 0.0]
        back = [0.0, 0.0, -1.0, 0.0]
        left = [-1.0, 0.0, 0.0, 0.0]
        right = [1.0, 0.0, 0.0, 0.0]
        top = [0.0, 1.0, 0.0, 0.0]
        bottom = [0.0, -1.0, 0.0, 0.0]
        directions = {
            GL.GL_LIGHT0: front,
            GL.GL_LIGHT1: back,
            GL.GL_LIGHT2: left,
            GL.GL_LIGHT3: right,
            GL.GL_LIGHT4: top,
            GL.GL_LIGHT5: bottom,
        }

        for light, direction in directions.items():
            GL.glLightfv(light, GL.GL_AMBIENT, ambient)
            GL.glLightfv(light, GL.GL_DIFFUSE, diffuse)
            GL.glLightfv(light, GL.GL_SPECULAR, specular)
            GL.glLightfv(light, GL.GL_POSITION, direction)

            GL.glEnable(light)

        GL.glEnable(GL.GL_LIGHTING)

        return len(directions)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        self.scene.camera.aspect = width / height
        GL.glLoadTransposeMatrixf(self.scene.camera.projection)

    def paintGL(self):
        if self.scene.invalid:
            if self.error:
                return

            self.error = True

            if self.custom_error_handler is not None:
                self.custom_error_handler()
                self.scene.camera.reset()
            return

        self.error = False

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadTransposeMatrixf(self.scene.camera.projection)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadTransposeMatrixf(self.scene.camera.model_view)

        if self.show_coordinate_frame:
            self.renderAxis()

        for node in self.scene.nodes:
            self.recursiveDraw(node)

        if self.show_bounding_box:
            self.renderBoundingBox()

        if self.picks:
            self.renderPicks()

    def recursiveDraw(self, node):
        """Recursive renders node from the scene with its children

        :param node: node
        :type: Node
        """
        if not node.visible:
            return

        GL.glPushMatrix()
        GL.glPushAttrib(GL.GL_CURRENT_BIT)
        GL.glMultTransposeMatrixf(node.transform)

        mode = Node.RenderMode.Solid if node.render_mode is None else node.render_mode

        if mode == Node.RenderMode.Solid:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        elif mode == Node.RenderMode.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_SRC_COLOR)

        self.draw(node)

        # reset OpenGL State
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        for child in node.children:
            self.recursiveDraw(child)

        GL.glPopAttrib()
        GL.glPopMatrix()

    def draw(self, node):
        """Renders a leaf node (node with no child) from the scene

        :param node: leaf node
        :type node: Node
        """
        program = self.shader_programs["default"]
        if node.vertices.size > 0 and node.indices.size > 0:
            if node.normals.size > 0:
                program = self.shader_programs["mesh"]

            program.bind()
            node.buffer.bind()

            primitive = GL.GL_TRIANGLES if node.render_primitive == Node.RenderPrimitive.Triangles else GL.GL_LINES

            if isinstance(node, InstanceRenderNode):
                self.drawInstanced(node, primitive)
            elif isinstance(node, BatchRenderNode):
                self.drawBatch(node, primitive)
            else:
                self.drawNode(node, primitive)

            node.buffer.release()
            program.release()

    def drawNode(self, node, primitive):
        """Renders a leaf node (node with no child) from the scene

        :param node: leaf node
        :type node: Node
        :param primitive: OpenGL primitive to render
        :type primitive: OpenGL.constant.IntConstant
        """
        node.buffer.bind()
        if node.selected:
            GL.glColor4f(*settings.value(settings.Key.Selected_Colour))
        else:
            GL.glColor4f(*node.colour.rgbaf)

        if node.outlined:
            self.drawOutline(primitive, node.indices)

        GL.glDrawElements(primitive, node.buffer.count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def drawInstanced(self, node, primitive):
        """Renders a instanced node from the scene

        :param node: leaf node
        :type node: InstanceRenderNode
        :param primitive: OpenGL primitive to render
        :type primitive: OpenGL.constant.IntConstant
        """
        for index, transform in enumerate(node.per_object_transform):
            GL.glPushMatrix()
            GL.glMultTransposeMatrixf(transform)
            if node.selected[index]:
                GL.glColor4f(*settings.value(settings.Key.Selected_Colour))
            else:
                GL.glColor4f(*node.per_object_colour[index].rgbaf)

            if node.outlined[index]:
                self.drawOutline(primitive, node.buffer.count)

            GL.glDrawElements(primitive, node.buffer.count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            GL.glPopMatrix()

    def drawBatch(self, node, primitive):
        """Renders a batch node from the scene

        :param node: leaf node
        :type node: BatchRenderNode
        :param primitive: OpenGL primitive to render
        :type primitive: OpenGL.constant.IntConstant
        """
        start = 0
        for index, end in enumerate(node.batch_offsets):
            if node.selected[index]:
                GL.glColor4f(*settings.value(settings.Key.Selected_Colour))
            else:
                GL.glColor4f(*node.per_object_colour[index].rgbaf)
            GL.glPushMatrix()
            t = Matrix44.identity() if not node.per_object_transform else node.per_object_transform[index]
            GL.glMultTransposeMatrixf(t)

            count = end - start
            offset = start * node.vertices.itemsize

            if node.outlined[index]:
                self.drawOutline(primitive, count, offset)

            GL.glDrawElements(primitive, count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(offset))

            GL.glPopMatrix()
            start = end

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

    def pickEvent(self, event):
        """Custom event for point picking

        :param event: mouse event
        :type event: QtGui.QMouseEvent
        """
        if event.buttons() != QtCore.Qt.LeftButton:
            return

        point = event.pos()
        v1, valid1 = self.unproject(point.x(), point.y(), 0.0)
        v2, valid2 = self.unproject(point.x(), point.y(), 1.0)
        if not valid1 or not valid2:
            return
        self.pick_added.emit(v1, v2)

    def renderPicks(self):
        """Renders picked points in the scene"""
        size = settings.value(settings.Key.Measurement_Size)

        node = InstanceRenderNode(len(self.picks))
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines

        vertices = np.array(
            [
                [-size, 0.0, 0.0],
                [size, 0.0, 0.0],
                [0.0, -size, 0.0],
                [0.0, size, 0.0],
                [0.0, 0.0, -size],
                [0.0, 0.0, size],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        node.vertices = vertices
        node.indices = indices
        for index, pick in enumerate(self.picks):
            point, selected = pick
            node.selected[index] = selected
            node.per_object_colour[index] = Colour(0.9, 0.4, 0.4)
            node.per_object_transform[index] = Matrix44.fromTranslation(point)

        node.buildVertexBuffer()
        self.draw(node)

    def mousePressEvent(self, event):
        if self.picking:
            self.pickEvent(event)
        else:
            self.scene.camera.mode = Camera.Projection.Perspective
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.picking:
            return

        translation_speed = 0.001

        if event.buttons() == QtCore.Qt.LeftButton:
            p1 = (self.last_pos.x() / self.width() * 2, self.last_pos.y() / self.height() * 2)
            p2 = (event.x() / self.width() * 2, event.y() / self.height() * 2)
            self.scene.camera.rotate(p1, p2)

        elif event.buttons() == QtCore.Qt.RightButton:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            x_offset = -dx * translation_speed
            y_offset = -dy * translation_speed
            self.scene.camera.pan(x_offset, y_offset)

        self.last_pos = event.pos()
        self.update()

    def showCoordinateFrame(self, state):
        """Sets visibility of the coordinate frame in the widget

        :param state: indicates if the coordinate frame should be visible
        :type state: bool
        """
        self.show_coordinate_frame = state
        self.update()

    def showBoundingBox(self, state):
        """Sets visibility of the sample bounding box frame in the widget

        :param state: indicates if the bounding box should be visible
        :type state: bool
        """
        self.show_bounding_box = state
        self.update()

    def wheelEvent(self, event):
        zoom_scale = 0.05
        delta = 0.0
        num_degrees = event.angleDelta() / 8
        if not num_degrees.isNull():
            delta = num_degrees.y() / 15

        self.scene.camera.zoom(delta * zoom_scale)
        self.update()

    def loadScene(self, scene, zoom_to_fit=True):
        """Loads a scene into the widget and adjust the camera

        :param scene: sample or instrument scene
        :type scene: Scene
        :param zoom_to_fit: indicates that the scene should be zoomed to fit window
        :type zoom_to_fit: bool
        """
        self.scene = scene
        self.scene.camera.aspect = self.width() / self.height()

        if not self.scene.isEmpty():
            bounding_box = self.scene.bounding_box
            if zoom_to_fit:
                self.scene.camera.zoomToFit(bounding_box.center, bounding_box.radius)
            else:
                self.scene.camera.updateView(bounding_box.center, bounding_box.radius)

        self.update()

    def unproject(self, x, y, z):
        """Converts point in screen coordinate to point in world coordinate

        :param x: x coordinate
        :type x: float
        :param y: y coordinate
        :type y: float
        :param z: z coordinate
        :type z: float
        :return: point in screen coordinates and flag indicating the new point is valid
        :rtype: Tuple[Vector3, bool]
        """
        y = self.height() - y  # invert y to match screen coordinate
        screen_point = Vector3([x, y, z])
        model_view = self.scene.camera.model_view
        projection = self.scene.camera.projection

        world_point, valid = screen_to_world(screen_point, model_view, projection, self.width(), self.height())
        return world_point, valid

    def project(self, x, y, z):
        """Converts point in world coordinate to point in screen coordinate

        :param x: x coordinate
        :type x: float
        :param y: y coordinate
        :type y: float
        :param z: z coordinate
        :type z: float
        :return: point in screen coordinates and flag indicating the new point is valid
        :rtype: Tuple[Vector3, bool]
        """
        world_point = Vector3([x, y, z])
        model_view = self.scene.camera.model_view
        projection = self.scene.camera.projection

        screen_point, valid = world_to_screen(world_point, model_view, projection, self.width(), self.height())
        screen_point.y = self.height() - screen_point.y  # invert y to match screen coordinate
        return screen_point, valid

    def renderBoundingBox(self):
        """Draws the axis aligned bounding box of the sample"""
        if Attributes.Sample not in self.scene:
            return

        bounding_box = self.scene[Attributes.Sample].bounding_box
        max_x, max_y, max_z = bounding_box.max
        min_x, min_y, min_z = bounding_box.min

        node = Node()
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines

        node.vertices = np.array(
            [
                [min_x, min_y, min_z],
                [min_x, max_y, min_z],
                [max_x, min_y, min_z],
                [max_x, max_y, min_z],
                [min_x, min_y, max_z],
                [min_x, max_y, max_z],
                [max_x, min_y, max_z],
                [max_x, max_y, max_z],
            ],
            dtype=np.float32,
        )
        node.indices = np.array(
            [0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7], dtype=np.uint32
        )
        node.colour = Colour(0.9, 0.4, 0.4)
        node.buildVertexBuffer()
        self.draw(node)

    def renderAxis(self):
        """Draws the X, Y and Z axis lines and centre point"""
        if self.scene.isEmpty():
            return

        scale = self.scene.bounding_box.radius

        node = BatchRenderNode(3)
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines
        node.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [scale, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, scale],
            ],
            dtype=np.float32,
        )

        node.indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        node.per_object_colour = [Colour(1.0, 0.0, 0.0), Colour(0.0, 1.0, 0.0), Colour(0.0, 0.0, 1.0)]
        node.batch_offsets = [2, 4, 6]
        node.buildVertexBuffer()

        GL.glEnable(GL.GL_DEPTH_CLAMP)
        GL.glDepthFunc(GL.GL_LEQUAL)
        self.draw(node)
        GL.glDisable(GL.GL_DEPTH_CLAMP)
        GL.glDepthFunc(GL.GL_LESS)

        origin, ok = self.project(0.0, 0.0, 0.0)
        if not ok:
            return

        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor.fromRgbF(0.5, 0.5, 0.5))
        painter.setFont(self.default_font)

        # draw origin
        painter.drawEllipse(QtCore.QPointF(origin.x, origin.y), 10, 10)

        axes = [(1, 0, 0, "X"), (0, 1, 0, "Y"), (0, 0, 1, "Z")]

        for x, y, z, label in axes:
            painter.setPen(QtGui.QColor.fromRgbF(x, y, z))

            x *= scale * 1.01
            y *= scale * 1.01
            z *= scale * 1.01

            text_pos, ok = self.project(x, y, z)
            if not ok:
                continue

            # Render text
            painter.drawText(QtCore.QPointF(*text_pos[:2]), label)

        painter.end()
        GL.glPopAttrib()

    def viewFrom(self, direction):
        """Changes view direction of scene camera

        :param direction: direction to view from
        :type direction: Direction
        """
        self.scene.camera.mode = Camera.Projection.Orthographic
        self.scene.camera.viewFrom(direction)
        self.update()

    def resetCamera(self):
        """Resets scene camera"""
        self.scene.camera.reset()
        self.update()
