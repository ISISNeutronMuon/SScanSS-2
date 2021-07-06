from abc import ABC, abstractmethod
import ctypes
from enum import Enum, unique
import math
import numpy as np
from OpenGL import GL, error
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.math import Vector3, clamp, Matrix44
from sscanss.core.geometry import Colour
from sscanss.core.scene import (Node, Camera, world_to_screen, screen_to_world, Scene, InstanceRenderNode,
                                BatchRenderNode, GouraudShader, DefaultShader)
from sscanss.core.util import Attributes
from sscanss.config import settings


class GLWidget(QtWidgets.QOpenGLWidget):
    """Provides OpenGL widget for draw 3D scene for the sample setup and instrument

    :param parent: Main window
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
            del self.shader_programs[key]
        self.doneCurrent()

    @property
    def picking(self):
        return self._picking

    @picking.setter
    def picking(self, value):
        """Enables/Disables point picking

        :param value: indicates point picking is enabled
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
            self.shader_programs['mesh'] = GouraudShader(number_of_lights)
            self.shader_programs['default'] = DefaultShader()

            self.context().aboutToBeDestroyed.connect(self.cleanup)

        except error.GLError:
            self.parent.showMessage('An error occurred during OpenGL initialization. '
                                    'The minimum OpenGL requirement for this software is version 2.0.\n\n'
                                    'This error may be caused by:\n'
                                    '* A missing or faulty graphics driver installation.\n'
                                    '* Accessing SScanSS 2 from a remote connection with GPU rendering disabled.\n\n'
                                    'The software will be closed now.'
                                    )
            raise

    def initLights(self):
        # set up light colour
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
        directions = {GL.GL_LIGHT0: front, GL.GL_LIGHT1: back, GL.GL_LIGHT2: left,
                      GL.GL_LIGHT3: right, GL.GL_LIGHT4: top, GL.GL_LIGHT5: bottom}

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
        program = self.shader_programs['default']
        if node.vertices.size > 0 and node.indices.size > 0:
            if node.normals.size > 0:
                program = self.shader_programs['mesh']

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
        """Renders picked points in the widget"""
        size = settings.value(settings.Key.Measurement_Size)

        node = InstanceRenderNode(len(self.picks))
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines

        vertices = np.array([[-size, 0., 0.], [size, 0., 0.],
                             [0., -size, 0.], [0., size, 0.],
                             [0., 0., -size], [0., 0., size]], dtype=np.float32)
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
        if Attributes.Sample not in self.scene:
            return

        bounding_box = self.scene[Attributes.Sample].bounding_box
        max_x, max_y, max_z = bounding_box.max
        min_x, min_y, min_z = bounding_box.min

        node = Node()
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines

        node.vertices = np.array([[min_x, min_y, min_z],
                                  [min_x, max_y, min_z],
                                  [max_x, min_y, min_z],
                                  [max_x, max_y, min_z],
                                  [min_x, min_y, max_z],
                                  [min_x, max_y, max_z],
                                  [max_x, min_y, max_z],
                                  [max_x, max_y, max_z]], dtype=np.float32)
        node.indices = np.array([0, 1, 1, 3, 3, 2, 2, 0,
                                 4, 5, 5, 7, 7, 6, 6, 4,
                                 0, 4, 1, 5, 2, 6, 3, 7], dtype=np.uint32)
        node.colour = Colour(0.9, 0.4, 0.4)
        node.buildVertexBuffer()
        self.draw(node)

    def renderAxis(self):
        if self.scene.isEmpty():
            return

        if self.scene.type == Scene.Type.Sample and Attributes.Sample in self.scene:
            scale = self.scene[Attributes.Sample].bounding_box.radius
        else:
            scale = self.scene.bounding_box.radius

        node = BatchRenderNode(3)
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines
        node.vertices = np.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0],
                                  [0.0, 0.0, 0.0], [0.0, scale, 0.0],
                                  [0.0, 0.0, 0.0], [0.0, 0.0, scale]], dtype=np.float32)

        node.indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        node.per_object_colour = [Colour(1.0, 0.0, 0.0), Colour(0.0, 1.0, 0.0), Colour(0.0, 0.0, 1.0)]
        node.batch_offsets = [2, 4, 6]
        node.buildVertexBuffer()

        GL.glEnable(GL.GL_DEPTH_CLAMP)
        GL.glDepthFunc(GL.GL_LEQUAL)
        self.draw(node)
        GL.glDisable(GL.GL_DEPTH_CLAMP)
        GL.glDepthFunc(GL.GL_LESS)

        origin, ok = self.project(0., 0., 0.)
        if not ok:
            return

        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor.fromRgbF(0.5, 0.5, 0.5))
        painter.setFont(self.default_font)

        # draw origin
        painter.drawEllipse(QtCore.QPointF(origin.x, origin.y), 10, 10)

        axes = [(1, 0, 0, 'X'), (0, 1, 0, 'Y'), (0, 0, 1, 'Z')]

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


class GraphicsView(QtWidgets.QGraphicsView):
    """Provides container for graphics scene

    :param scene: scene
    :type scene: GraphicsScene
    """
    mouse_moved = QtCore.pyqtSignal(object)

    def __init__(self, scene):
        super().__init__(scene)
        self.createActions()

        self.show_grid = False
        self.snap_to_grid = False
        self.show_help = False
        self.has_foreground = False
        self.grid = BoxGrid()
        self.zoom_factor = 1.2
        self.anchor = QtCore.QRectF()
        self.scene_transform = QtGui.QTransform()
        self.setMouseTracking(True)

        self.setViewportUpdateMode(self.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.updateViewMode()

    def updateViewMode(self):
        """Updates view behaviour to match scene mode"""
        if not self.scene():
            return

        if self.scene().mode == GraphicsScene.Mode.Select:
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        else:
            self.setCursor(QtCore.Qt.CrossCursor)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def drawForeground(self, painter, rect):
        """Draws the help information in foreground using the given painter and rect"""
        if not self.show_help:
            self.has_foreground = False
            return

        spacing = 10

        textDocument = QtGui.QTextDocument()
        textDocument.setDefaultStyleSheet("* { color: #ffffff }")
        textDocument.setHtml("<h3 align=\"center\">Shortcuts</h3>"
                             "<div>"
                             "<pre>Delete&#9;&nbsp;Deletes selected point</pre>"
                             "<pre>+ \\ -&#9;&nbsp;Zoom in \\ out </pre>"
                             "<pre>Mouse&#9;&nbsp;Zoom in \\ out<br>Wheel</pre>"
                             "<pre>Right&#9;&nbsp;Rotate view<br>Click</pre>"
                             "<pre>Ctrl + &#9;&nbsp;Pan view<br>Right Click</pre>"
                             "<pre>Middle &#9;&nbsp;Pan view<br>Click</pre>"
                             "<pre>Ctrl + R&#9;&nbsp;Reset view</pre>"
                             "</div></table>")
        textDocument.setTextWidth(textDocument.size().width())

        text_rect = QtCore.QRect(0, 0, 300, 280)
        painter.save()
        transform = QtGui.QTransform()
        painter.setWorldTransform(transform.translate(self.width()//2, self.height()//2))
        painter.translate(-text_rect.center().x() - spacing, -text_rect.center().y() - spacing)
        pen = QtGui.QPen(QtGui.QColor(180, 180, 180), 3)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 230))
        painter.drawRoundedRect(text_rect, 20, 20)

        painter.translate(spacing, spacing)
        textDocument.drawContents(painter)
        painter.restore()
        self.has_foreground = True
        self.show_help = False

    def createActions(self):
        zoom_in = QtWidgets.QAction("Zoom in", self)
        zoom_in.triggered.connect(self.zoomIn)
        zoom_in.setShortcut(QtGui.QKeySequence("+"))
        zoom_in.setShortcutContext(QtCore.Qt.WidgetShortcut)

        zoom_out = QtWidgets.QAction("Zoom out", self)
        zoom_out.triggered.connect(self.zoomOut)
        zoom_out.setShortcut(QtGui.QKeySequence("-"))
        zoom_out.setShortcutContext(QtCore.Qt.WidgetShortcut)

        reset = QtWidgets.QAction("Reset View", self)
        reset.triggered.connect(self.reset)
        reset.setShortcut(QtGui.QKeySequence('Ctrl+R'))
        reset.setShortcutContext(QtCore.Qt.WidgetShortcut)

        self.addActions([zoom_in, zoom_out, reset])

    def setGridType(self, grid_type):
        """Sets the type of grid to draw

        :param grid_type: grid type
        :type grid_type: Grid.Type
        """
        if grid_type == Grid.Type.Box:
            self.grid = BoxGrid()
        else:
            self.grid = PolarGrid()

    def setGridSize(self, size):
        """Sets size of active grid

        :param size: size of grid
        :type size: Tuple[int, int]
        """
        self.grid.size = size
        self.scene().update()

    def rotateSceneItems(self, angle, offset):
        """Rotates scene items

        :param angle: angle in radians
        :type angle: float
        :param offset: centre of rotation
        :type offset: QtCore.QPointF
        """
        if not self.scene():
            return

        gr = self.scene().createItemGroup(self.scene().items())
        transform = QtGui.QTransform().translate(offset.x(), offset.y())
        transform.rotateRadians(angle)
        transform.translate(-offset.x(), -offset.y())
        self.scene_transform *= transform
        gr.setTransform(transform)

        self.scene().destroyItemGroup(gr)
        self.scene().update()

    def translateSceneItems(self, dx, dy):
        """Translates scene items

        :param dx: x offset
        :type dx: float
        :param dy: y offset
        :type dy: float
        """
        if not self.scene():
            return

        gr = self.scene().createItemGroup(self.scene().items())
        transform = QtGui.QTransform().translate(dx, dy)
        self.scene_transform *= transform
        gr.setTransform(transform)
        self.scene().destroyItemGroup(gr)
        self.scene().update()

    def resetTransform(self):
        self.scene_transform.reset()
        super().resetTransform()

    def reset(self):
        """Resets the camera of the graphics view"""
        if not self.scene() or not self.scene().items():
            return

        gr = self.scene().createItemGroup(self.scene().items())
        gr.setTransform(self.scene_transform.inverted()[0])
        self.scene().destroyItemGroup(gr)

        rect = self.scene().itemsBoundingRect()
        # calculate new rectangle that encloses original rect with a different anchor
        rect.united(rect.translated(self.anchor.center() - rect.center()))

        self.resetTransform()
        self.setSceneRect(rect)
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def zoomIn(self):
        """Zooms in the camera of the graphics view"""
        if not self.scene():
            return

        self.scale(self.zoom_factor, self.zoom_factor)
        anchor = self.scene_transform.mapRect(self.anchor)
        self.centerOn(anchor.center())

    def zoomOut(self):
        """Zooms out the camera of the graphics view"""
        if not self.scene():
            return

        factor = 1.0/self.zoom_factor
        self.scale(factor, factor)
        anchor = self.scene_transform.mapRect(self.anchor)
        self.centerOn(anchor.center())

    def leaveEvent(self, _event):
        self.mouse_moved.emit(QtCore.QPoint(-1, -1))

    def mousePressEvent(self, event):
        is_rotating = event.button() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.NoModifier
        is_panning = ((event.button() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.ControlModifier)
                      or (event.buttons() == QtCore.Qt.MiddleButton and event.modifiers() == QtCore.Qt.NoModifier))

        if is_rotating:
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif is_panning:
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        self.last_pos = self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton or event.button() == QtCore.Qt.MiddleButton:
            if self.scene():
                self.updateViewMode()

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event.pos())
        if event.buttons() == QtCore.Qt.NoButton:
            super().mouseMoveEvent(event)
            return

        is_rotating = event.buttons() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.NoModifier
        is_panning = ((event.buttons() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.ControlModifier)
                      or (event.buttons() == QtCore.Qt.MiddleButton and event.modifiers() == QtCore.Qt.NoModifier))

        pos = self.mapToScene(event.pos())
        if is_rotating:
            anchor = self.scene_transform.mapRect(self.anchor)
            adj_pos = pos - anchor.center()
            adj_last_pos = self.last_pos - anchor.center()

            if adj_pos.manhattanLength() < 0.1 or adj_last_pos.manhattanLength() < 0.1:
                return

            va = Vector3([adj_last_pos.x(), adj_last_pos.y(), 0.]).normalized
            vb = Vector3([adj_pos.x(), adj_pos.y(), 0.]).normalized
            angle = -math.acos(clamp(va | vb, -1.0, 1.0))

            if np.dot([0., 0., 1.], va ^ vb) > 0:
                angle = -angle

            self.rotateSceneItems(angle, anchor.center())
        elif is_panning:
            dx = pos.x() - self.last_pos.x()
            dy = pos.y() - self.last_pos.y()
            self.translateSceneItems(dx, dy)

        self.last_pos = pos
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        if event.buttons() != QtCore.Qt.NoButton:
            return

        delta = 0.0
        num_degrees = event.angleDelta() / 8
        if not num_degrees.isNull():
            delta = num_degrees.y() / 15

        if delta < 0:
            self.zoomOut()
        elif delta > 0:
            self.zoomIn()

    def drawBackground(self, painter, rect):
        """Draws the grid in background using the given painter and rect"""
        if not self.show_grid:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setOpacity(0.3)
        pen = QtGui.QPen(QtCore.Qt.darkGreen, 0)
        painter.setPen(pen)

        center = self.anchor.center()
        top_left = rect.topLeft() - center
        bottom_right = rect.bottomRight() - center
        half_size = QtCore.QPointF(max(abs(top_left.x()), abs(bottom_right.x())),
                                   max(abs(top_left.y()), abs(bottom_right.y())))
        adjusted_rect = QtCore.QRectF(center - half_size, center + half_size)
        self.grid.render(painter, adjusted_rect)

        painter.restore()


class GraphicsScene(QtWidgets.QGraphicsScene):
    """Provides graphics scene for measurement point selection

    :param scale: scale of the scene
    :type scale: int
    :param parent: parent widget
    :type parent: QtCore.QObject
    """
    @unique
    class Mode(Enum):
        Select = 1
        Draw_point = 2
        Draw_line = 3
        Draw_area = 4

    def __init__(self, scale=1, parent=None):
        super().__init__(parent)

        self.scale = scale
        self.point_size = 6 * scale
        self.path_pen = QtGui.QPen(QtCore.Qt.black, 0)

        self.item_to_draw = None
        self.current_obj = None
        self.mode = GraphicsScene.Mode.Select

        self.setLineToolSize(2)
        self.setAreaToolSize(2, 2)
        self.start_pos = QtCore.QPointF()

    @property
    def view(self):
        """returns graphics view associated with scene

        :rtype: Union[QtWidgets.QGraphicsView, None]
        """
        view = self.views()
        if view:
            return view[0]
        else:
            return None

    @property
    def mode(self):
        """returns scene's mode

        :rtype: GraphicsScene.Mode
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        view = self.view
        if view is None:
            return

        if value == GraphicsScene.Mode.Select:
            self.makeItemsControllable(True)
        else:
            self.makeItemsControllable(False)
        view.updateViewMode()

    def setAreaToolSize(self, x_count, y_count):
        """Sets the x, y divisions of the area tool

        :param x_count: number of divisions in the x axis
        :type x_count: int
        :param y_count: number of divisions in the y axis
        :type y_count: int
        """
        self.area_tool_size = (x_count, y_count)
        self.area_tool_x_offsets = np.tile(np.linspace(0., 1., self.area_tool_size[0]), self.area_tool_size[1])
        self.area_tool_y_offsets = np.repeat(np.linspace(0., 1., self.area_tool_size[1]), self.area_tool_size[0])

    def setLineToolSize(self, count):
        """Sets the number of divisions of the line tool

        :param count: number of divisions on the line
        :type count: int
        """
        self.line_tool_size = count
        self.line_tool_point_offsets = np.linspace(0., 1., self.line_tool_size)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            view = self.view
            pos = event.scenePos()
            if view.snap_to_grid:
                pos = view.grid.snap(pos)

            if self.mode == GraphicsScene.Mode.Draw_point:
                self.addPoint(pos)
            elif self.mode != GraphicsScene.Mode.Select:
                self.start_pos = pos

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() != QtCore.Qt.LeftButton:
            super().mouseMoveEvent(event)
            return

        start = self.start_pos
        stop = event.scenePos()
        if self.mode == GraphicsScene.Mode.Draw_line:
            if self.item_to_draw is None:
                self.item_to_draw = QtWidgets.QGraphicsLineItem()
                self.addItem(self.item_to_draw)
                self.item_to_draw.setPen(self.path_pen)

            self.current_obj = QtCore.QLineF(start, stop)
            self.item_to_draw.setLine(self.current_obj)

        elif self.mode == GraphicsScene.Mode.Draw_area:
            if self.item_to_draw is None:
                self.item_to_draw = QtWidgets.QGraphicsRectItem()
                self.addItem(self.item_to_draw)
                self.item_to_draw.setPen(self.path_pen)

            top, bottom = (stop.y(), start.y()) if start.y() > stop.y() else (start.y(), stop.y())
            left, right = (stop.x(), start.x()) if start.x() > stop.x() else (start.x(), stop.x())
            self.current_obj = QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom))
            self.item_to_draw.setRect(self.current_obj)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.item_to_draw is None:
            super().mouseReleaseEvent(event)
            return

        view = self.view
        start = self.start_pos
        stop = event.scenePos()
        if view.snap_to_grid:
            stop = view.grid.snap(stop)

        if self.mode == GraphicsScene.Mode.Draw_line:
            self.current_obj = QtCore.QLineF(start, stop)
            self.item_to_draw.setLine(self.current_obj)
            for t in self.line_tool_point_offsets:
                point = self.current_obj.pointAt(t)
                self.addPoint(point)

        elif self.mode == GraphicsScene.Mode.Draw_area:
            top, bottom = (stop.y(), start.y()) if start.y() > stop.y() else (start.y(), stop.y())
            left, right = (stop.x(), start.x()) if start.x() > stop.x() else (start.x(), stop.x())
            self.current_obj = QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom))
            self.item_to_draw.setRect(self.current_obj)
            diag = self.current_obj.bottomRight() - self.current_obj.topLeft()
            x = self.current_obj.x() + self.area_tool_x_offsets * diag.x()
            y = self.current_obj.y() + self.area_tool_y_offsets * diag.y()
            for t1, t2 in zip(x, y):
                point = QtCore.QPointF(t1, t2)
                self.addPoint(point)

        self.removeItem(self.item_to_draw)
        self.item_to_draw = None

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            for item in self.selectedItems():
                self.removeItem(item)
        else:
            super().keyPressEvent(event)

    def makeItemsControllable(self, flag):
        """Enables/Disables the ability to select and move the graphics
        point item in the scene with the mouse

        :param flag: indicates if the item can be controlled
        :type flag: bool
        """
        for item in self.items():
            if isinstance(item, GraphicsPointItem):
                item.makeControllable(flag)

    def addPoint(self, point):
        """Adds graphics point item into the scene at specified coordinates

        :param point: point coordinates
        :type point:
        """
        p = GraphicsPointItem(point, size=self.point_size)
        p.setPen(self.path_pen)
        p.setZValue(1.0)  # Ensure point is drawn above cross section
        self.addItem(p)


class GraphicsPointItem(QtWidgets.QAbstractGraphicsShapeItem):
    def __init__(self, point, *args, size=6, **kwargs):
        """Provides shape item for points in graphics view. The point is drawn as a cross with
        equal width and height.

        :param point: point coordinates
        :type point: QtCore.QPoint
        :param size: pixel size of point
        :type size: int
        """
        super().__init__(*args, **kwargs)

        self.size = size
        self.setPos(point)
        self.fixed = False
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)

    def makeControllable(self, flag):
        """Enables/Disables the ability to select and move the graphics item with the mouse

        :param flag: indicates if the item can be controlled
        :type flag: bool
        """
        if not self.fixed:
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, flag)
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, flag)

    @property
    def x(self):
        return self.pos().x()

    @property
    def y(self):
        return self.pos().y()

    def boundingRect(self):
        """Calculates the bounding box of the graphics item

        :return: bounding rect
        :rtype: QtCore.QRect
        """
        pen_width = self.pen().widthF()
        half_pen_width = pen_width * 0.5

        top = -(self.size * 0.5) - half_pen_width
        new_size = self.size + pen_width
        return QtCore.QRectF(top, top, new_size, new_size)

    def paint(self, painter, _options, _widget):
        """Draws the graphics item

        :param painter: painter to use for drawing
        :type painter: QtGui.QPainter
        """
        pen = self.pen()
        painter.setPen(pen)
        painter.setBrush(self.brush())

        half = self.size * 0.5
        painter.drawLine(-half, -half, half, half)
        painter.drawLine(-half, half, half, -half)

        if self.isSelected():
            painter.save()
            pen = QtGui.QPen(QtCore.Qt.black, 0, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(self.boundingRect())
            painter.restore()


class Grid(ABC):
    """Base class for form graphics view grid """
    @unique
    class Type(Enum):
        Box = 'Box'
        Polar = 'Polar'

    @property
    @abstractmethod
    def type(self):
        """Return Type of Grid
        :rtype: Grid.Type
        """

    @property
    def size(self):
        """Return size of grid
        :rtype: Tuple[int, int]
        """

    @size.setter
    @abstractmethod
    def size(self, value):
        """Sets the size of the grid
        :param value: grid size
        :type value: Tuple[int, int]
        """

    @abstractmethod
    def render(self, painter, rect):
        """Draws the grid using the given painter and rect
        :param painter: painter to use when drawing the grid
        :type painter: QtGui.QPainter
        :param rect: rectangle dimension to draw the grid in
        :type rect: QtCore.QRect
        """

    @abstractmethod
    def snap(self, pos):
        """Calculate closest grid position to the given pos
        :param pos: point to snap
        :type pos: QtCore.QPointF
        :return: snapped point
        :rtype: QtCore.QPointF
        """


class BoxGrid(Grid):
    """Provides rectangular grid for the graphics view

    :param x: x size
    :type x: int
    :param y: y size
    :type y: int
    """
    def __init__(self, x=10, y=10):
        self.x = x
        self.y = y

    @property
    def type(self):
        return Grid.Type.Box

    @property
    def size(self):
        return self.x, self.y

    @size.setter
    def size(self, value):
        self.x, self.y = value

    def render(self, painter, rect):
        scene_rect = rect.toRect()
        left = scene_rect.left()
        top = scene_rect.top()
        right = scene_rect.right() + 2 * self.x
        bottom = scene_rect.bottom() + 2 * self.y

        left = left - left % self.x
        top = top - top % self.y

        x_offsets = np.array(range(left, right, self.x))
        y_offsets = np.array(range(top, bottom, self.y))

        for x in x_offsets:
            painter.drawLine(x, top, x, bottom)

        for y in y_offsets:
            painter.drawLine(left, y, right, y)

    def snap(self, pos):
        pos_x = round(pos.x() / self.x) * self.x
        pos_y = round(pos.y() / self.y) * self.y

        return QtCore.QPointF(pos_x, pos_y)


class PolarGrid(Grid):
    """Provides radial/polar grid for the graphics view

    :param radial: radial distance
    :type radial: int
    :param angular: angle
    :type angular: int
    """
    def __init__(self, radial=10, angular=45):
        self.radial = radial
        self.angular = angular
        self.center = QtCore.QPoint()

    @property
    def type(self):
        return Grid.Type.Polar

    @property
    def size(self):
        return self.radial, self.angular

    @size.setter
    def size(self, value):
        self.radial, self.angular = value

    def render(self, painter, rect):
        center = rect.center().toPoint()
        radius = (rect.topRight().toPoint() - center).manhattanLength()
        radius = radius + radius % self.radial
        point = center + QtCore.QPoint(radius, 0)

        radial_offsets = np.array(range(self.radial, radius, self.radial))
        angular_offsets = np.arange(0.0, 360.0, self.angular)

        for r in radial_offsets:
            painter.drawEllipse(center, r, r)

        for angle in angular_offsets:
            transform = QtGui.QTransform().translate(center.x(), center.y())
            transform.rotate(angle)
            transform.translate(-center.x(), -center.y())
            rotated_point = transform.map(point)
            painter.drawLine(center.x(), center.y(), rotated_point.x(), rotated_point.y())

        self.center = center

    @staticmethod
    def toPolar(x, y):
        radius = math.sqrt(x * x + y * y)
        angle = math.atan2(y, x)

        return radius, math.degrees(angle)

    @staticmethod
    def toCartesian(radius, angle):
        angle = math.radians(angle)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        return x, y

    def snap(self, pos):
        pos = pos - self.center
        radius, angle = self.toPolar(pos.x(), pos.y())
        pos_x = round(radius / self.radial) * self.radial
        pos_y = round(angle / self.angular) * self.angular

        return QtCore.QPointF(*self.toCartesian(pos_x, pos_y)) + self.center
