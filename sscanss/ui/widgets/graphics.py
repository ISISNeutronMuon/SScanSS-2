import math
from enum import Enum, unique
import numpy as np
from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.math import Vector4, Vector3, clamp
from sscanss.core.geometry import Colour
from sscanss.core.scene import Node, Camera, world_to_screen, Scene
from sscanss.core.util import Attributes
from sscanss.config import settings


class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.scene = Scene()
        self.show_bounding_box = False
        self.show_coordinate_frame = True

        self.default_font = QtGui.QFont("Times", 10)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def initializeGL(self):
        GL.glClearColor(*Colour.white())
        GL.glColor4f(*Colour.black())

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glColorMaterial(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glShadeModel(GL.GL_SMOOTH)

        self.initLights()

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

        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, front)

        GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, back)

        GL.glLightfv(GL.GL_LIGHT2, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_POSITION, left)

        GL.glLightfv(GL.GL_LIGHT3, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_POSITION, right)

        GL.glLightfv(GL.GL_LIGHT4, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT4, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT4, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT4, GL.GL_POSITION, top)

        GL.glLightfv(GL.GL_LIGHT5, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT5, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT5, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT5, GL.GL_POSITION, bottom)

        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHT1)
        GL.glEnable(GL.GL_LIGHT2)
        GL.glEnable(GL.GL_LIGHT3)
        GL.glEnable(GL.GL_LIGHT4)
        GL.glEnable(GL.GL_LIGHT5)
        GL.glEnable(GL.GL_LIGHTING)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        self.scene.camera.aspect = width / height
        GL.glLoadTransposeMatrixf(self.scene.camera.projection)

    def paintGL(self):
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

    def recursiveDraw(self, node):
        if not node.visible:
            return

        GL.glPushMatrix()
        GL.glPushAttrib(GL.GL_CURRENT_BIT)
        GL.glMultTransposeMatrixf(node.transform)

        if node.selected:
            GL.glColor4f(*settings.value(settings.Key.Selected_Colour))
        else:
            GL.glColor4f(*node.colour.rgbaf)

        mode = Node.RenderMode.Solid if node.render_mode is None else node.render_mode

        if mode == Node.RenderMode.Solid:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        elif mode == Node.RenderMode.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        elif mode == Node.RenderMode.Transparent:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_SRC_COLOR)
        else:
            current_colour = GL.glGetDoublev(GL.GL_CURRENT_COLOR)
            GL.glColor3f(0.1, 0.1, 0.1)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glCullFace(GL.GL_FRONT)
            GL.glEnable(GL.GL_CULL_FACE)
            # First Pass
            self.renderNode(node)

            GL.glColor4dv(current_colour)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        self.renderNode(node)

        # reset OpenGL State
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        for child in node.children:
            self.recursiveDraw(child)

        GL.glPopAttrib()
        GL.glPopMatrix()

    def renderNode(self, node):
        if node.vertices.size != 0 and node.indices.size != 0:
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointerf(node.vertices)
            if node.normals.size != 0:
                GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
                GL.glNormalPointerf(node.normals)

            primitive = GL.GL_TRIANGLES if node.render_primitive == Node.RenderPrimitive.Triangles else GL.GL_LINES
            GL.glDrawElementsui(primitive, node.indices)

            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)

    def mousePressEvent(self, event):
        self.scene.camera.mode = Camera.Projection.Perspective
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
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
        self.show_coordinate_frame = state
        self.update()

    def showBoundingBox(self, state):
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
        self.scene = scene
        self.scene.camera.aspect = self.width() / self.height()

        if not self.scene.isEmpty():
            bounding_box = self.scene.bounding_box
            if bounding_box is not None:
                if zoom_to_fit:
                    self.scene.camera.zoomToFit(bounding_box.center, bounding_box.radius)
                else:
                    self.scene.camera.updateView(bounding_box.center, bounding_box.radius)
            else:
                self.scene.camera.reset()

        self.update()

    def project(self, x, y, z):
        world_point = Vector4([x, y, z, 1])
        model_view = self.scene.camera.model_view
        projection = self.scene.camera.projection

        screen_point, valid = world_to_screen(world_point, model_view, projection, self.width(), self.height())
        screen_point.y = self.height() - screen_point.y  # invert y to match screen coordinate
        return screen_point, valid

    def renderBoundingBox(self):
        if Attributes.Sample not in self.scene:
            return

        bounding_box = self.scene[Attributes.Sample].bounding_box
        transform = self.scene[Attributes.Sample].transform
        max_x, max_y, max_z = bounding_box.max
        min_x, min_y, min_z = bounding_box.min

        lines = np.array([[min_x, min_y, min_z],
                          [min_x, max_y, min_z],
                          [max_x, min_y, min_z],
                          [max_x, max_y, min_z],
                          [min_x, min_y, max_z],
                          [min_x, max_y, max_z],
                          [max_x, min_y, max_z],
                          [max_x, max_y, max_z]])

        indices = np.array([0, 1, 1, 3, 3, 2, 2, 0,
                            4, 5, 5, 7, 7, 6, 6, 4,
                            0, 4, 1, 5, 2, 6, 3, 7])

        GL.glPushMatrix()
        GL.glMultTransposeMatrixf(transform)
        GL.glEnable(GL.GL_LINE_STIPPLE)
        GL.glColor3fv([0.9, 0.4, 0.4])
        GL.glLineStipple(4, 0xAAAA)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointerf(lines)
        GL.glDrawElementsui(GL.GL_LINES, indices)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

        GL.glDisable(GL.GL_LINE_STIPPLE)
        GL.glPopMatrix()

    def renderAxis(self):
        if self.scene.isEmpty():
            return
        # Render Axis in 2 passes to avoid clipping

        # First Pass
        if self.scene.type == Scene.Type.Sample and Attributes.Sample in self.scene:
            scale = self.scene[Attributes.Sample].bounding_box.radius
        else:
            scale = self.scene.bounding_box.radius

        axis_vertices = [[0.0, 0.0, 0.0],
                         [scale, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, scale, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, scale]]

        axis_index = [0, 1, 2, 3, 4, 5]
        axis_colour = [[1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0]]

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)

        GL.glEnable(GL.GL_DEPTH_CLAMP)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glVertexPointerf(axis_vertices)
        GL.glColorPointerf(axis_colour)
        GL.glDrawElementsui(GL.GL_LINES, axis_index)
        GL.glDisable(GL.GL_DEPTH_CLAMP)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_MULTISAMPLE)

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)

        # Second Pass
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
            painter.drawText(text_pos[0], text_pos[1], label)

        painter.end()
        GL.glPopAttrib()

    def renderText(self, x, y, z, text, font=QtGui.QFont(), font_colour=QtGui.QColor(0,0,0)):
        text_pos, ok = self.project(x, y, z)
        if not ok:
            return

        # Render text
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        painter = QtGui.QPainter(self)
        painter.setPen(font_colour)
        painter.setFont(font)
        painter.drawText(text_pos[0], self.height() - text_pos[1], text)
        painter.end()
        GL.glPopAttrib()

    def viewFrom(self, direction):
        self.scene.camera.mode = Camera.Projection.Orthographic
        self.scene.camera.viewFrom(direction)
        self.update()

    def resetCamera(self):
        self.scene.camera.reset()
        self.update()


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, *args):
        super().__init__(*args)
        self.createActions()

        self.show_grid = False
        self.snap_to_grid = False
        self.show_help = False
        self.has_foreground = False
        self.grid_x_size = 10
        self.grid_y_size = 10
        self.zoom_factor = 1.5
        self.scene_transform = QtGui.QTransform()

        self.setViewportUpdateMode(self.FullViewportUpdate)
        self.horizontalScrollBar().hide()
        self.horizontalScrollBar().setStyleSheet('QScrollBar {height:0px;}')
        self.verticalScrollBar().hide()
        self.verticalScrollBar().setStyleSheet('QScrollBar {width:0px;}')

        if self.scene():
            self.scene().mode_changed.connect(self.updateViewMode)
            self.updateViewMode(self.scene().mode)

    def setScene(self, new_scene):
        super().setScene(new_scene)
        new_scene.mode_changed.connect(self.updateViewMode)
        self.updateViewMode(new_scene.mode)

    def updateViewMode(self, scene_mode):
        if scene_mode == GraphicsScene.Mode.Select:
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        else:
            self.setCursor(QtCore.Qt.CrossCursor)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def drawForeground(self, painter, rect):
        if not self.show_help:
            self.has_foreground = False
            return

        spacing = 10

        textDocument = QtGui.QTextDocument()
        textDocument.setDefaultStyleSheet("* { color: #ffffff }")
        textDocument.setHtml("<h3 align=\"center\">Shortcuts</h3>"
                             "<div>"
                             "<pre>Delete&#9;&nbsp;Deletes selected point</pre>"
                             "<pre>+ \ -&#9;&nbsp;Zoom in \ out </pre>"
                             "<pre>Mouse&#9;&nbsp;Zoom in \ out<br>Wheel</pre>"
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

    def setGridSize(self, x_size, y_size):
        self.grid_x_size = x_size
        self.grid_y_size = y_size
        self.scene().update()

    def rotateSceneItems(self, angle):
        if not self.scene():
            return

        gr = self.scene().createItemGroup(self.scene().items())
        transform = QtGui.QTransform().rotateRadians(angle)
        self.scene_transform *= transform
        gr.setTransform(transform)

        self.scene().destroyItemGroup(gr)
        self.scene().update()

    def translateSceneItems(self, dx, dy):
        if not self.scene():
            return

        gr = self.scene().createItemGroup(self.scene().items())
        transform = QtGui.QTransform().translate(dx, dy)
        self.scene_transform *= transform
        gr.setTransform(transform)

        self.scene().destroyItemGroup(gr)
        self.scene().update()

    def reset(self):
        gr = self.scene().createItemGroup(self.scene().items())
        gr.setTransform(self.scene_transform.inverted()[0])
        self.scene().destroyItemGroup(gr)
        self.scene_transform.reset()
        self.resetTransform()

    def zoomIn(self):
        if not self.scene():
            return

        self.scale(self.zoom_factor, self.zoom_factor)

    def zoomOut(self):
        if not self.scene():
            return

        factor = 1.0/self.zoom_factor
        self.scale(factor, factor)

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

        self.last_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton or event.button() == QtCore.Qt.MiddleButton:
            if self.scene():
                self.updateViewMode(self.scene().mode)

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        is_rotating = event.buttons() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.NoModifier
        is_panning = ((event.buttons() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.ControlModifier)
                      or (event.buttons() == QtCore.Qt.MiddleButton and event.modifiers() == QtCore.Qt.NoModifier))
        if is_rotating:
            w = self.width()
            h = self.height()
            va = Vector3([1. - (self.last_pos.x()/w * 2), (self.last_pos.y()/h * 2) - 1., 0.]).normalized
            vb = Vector3([1. - (event.x()/w * 2), (event.y()/h * 2) - 1., 0.]).normalized

            angle = math.acos(clamp(va | vb, -1.0, 1.0))
            if np.dot([0., 0., 1.], va ^ vb) > 0:
                angle = -angle
            self.rotateSceneItems(angle)
        elif is_panning:
                dx = event.x() - self.last_pos.x()
                dy = event.y() - self.last_pos.y()
                self.translateSceneItems(dx, dy)

        self.last_pos = event.pos()
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
        if not self.show_grid:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setOpacity(0.3)
        pen = QtGui.QPen(QtCore.Qt.darkGreen)
        painter.setPen(pen)

        scene_rect = rect.toRect()
        left = scene_rect.left()
        top = scene_rect.top()
        right = scene_rect.right() + 2 * self.grid_x_size
        bottom = scene_rect.bottom() + 2 * self.grid_y_size

        left = left - left % self.grid_x_size
        top = top - top % self.grid_y_size

        x_offsets = np.array(range(left, right, self.grid_x_size))
        y_offsets = np.array(range(top, bottom, self.grid_y_size))

        for x in x_offsets:
            painter.drawLine(x, top, x, bottom)

        for y in y_offsets:
            painter.drawLine(left, y, right, y)

        painter.restore()


class GraphicsScene(QtWidgets.QGraphicsScene):
    @unique
    class Mode(Enum):
        Select = 1
        Draw_point = 2
        Draw_line = 3
        Draw_area = 4

    mode_changed = QtCore.pyqtSignal(Mode)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._view = None
        self.item_to_draw = None
        self.current_obj = None
        self.mode = GraphicsScene.Mode.Select

        self.setLineToolPointCount(2)
        self.setAreaToolPointCount(2, 2)

    @property
    def view(self):
        if self._view is not None:
            return self._view
        view = self.views()
        if view:
            self._view = view[0]

        return self._view

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

        if value == GraphicsScene.Mode.Select:
            self.makeItemsControllable(True)
        else:
            self.makeItemsControllable(False)

        self.mode_changed.emit(value)

    def setAreaToolPointCount(self, x_count, y_count):
        self.area_tool_size = (x_count, y_count)
        self.area_tool_x_offsets = np.repeat(np.linspace(0., 1., self.area_tool_size[0]), self.area_tool_size[1])
        self.area_tool_y_offsets = np.tile(np.linspace(0., 1., self.area_tool_size[1]), self.area_tool_size[0])

    def setLineToolPointCount(self, value):
        self.line_tool_point_count = value
        self.line_tool_point_offsets = np.linspace(0., 1., self.line_tool_point_count)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            view = self.view
            pos = event.scenePos()
            if view.snap_to_grid:
                pos_x = round(pos.x() / view.grid_x_size) * view.grid_x_size
                pos_y = round(pos.y() / view.grid_y_size) * view.grid_y_size
                pos = QtCore.QPoint(pos_x, pos_y)

            if self.mode == GraphicsScene.Mode.Draw_point:
                self.addPoint(pos)
            elif self.mode != GraphicsScene.Mode.Select:
                self.origin_point = pos

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() != QtCore.Qt.LeftButton:
            super().mouseMoveEvent(event)
            return

        if self.mode == GraphicsScene.Mode.Draw_line:
            if self.item_to_draw is None:
                self.item_to_draw = QtWidgets.QGraphicsLineItem()
                self.addItem(self.item_to_draw)
                self.item_to_draw.setPen(QtGui.QPen(QtCore.Qt.black, 0, QtCore.Qt.SolidLine))

            self.current_obj = QtCore.QLineF(self.origin_point, event.scenePos())
            self.item_to_draw.setLine(self.current_obj)

        elif self.mode == GraphicsScene.Mode.Draw_area:
            if self.item_to_draw is None:
                self.item_to_draw = QtWidgets.QGraphicsRectItem()
                self.addItem(self.item_to_draw)
                self.item_to_draw.setPen(QtGui.QPen(QtCore.Qt.black, 0, QtCore.Qt.SolidLine))

            self.current_obj = QtCore.QRectF(self.origin_point, event.scenePos())
            self.item_to_draw.setRect(self.current_obj)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.item_to_draw is None:
            super().mouseReleaseEvent(event)
            return
        view = self.view
        pos_x = event.scenePos().x()
        pos_y = event.scenePos().y()
        if view.snap_to_grid:
            pos_x = round(pos_x / view.grid_x_size) * view.grid_x_size
            pos_y = round(pos_y / view.grid_y_size) * view.grid_y_size

        if self.mode == GraphicsScene.Mode.Draw_line:
            self.current_obj = QtCore.QLineF(self.origin_point, QtCore.QPointF(pos_x, pos_y))
            self.item_to_draw.setLine(self.current_obj)
            for t in self.line_tool_point_offsets:
                point = self.current_obj.pointAt(t)
                self.addPoint(point)

        elif self.mode == GraphicsScene.Mode.Draw_area:
            self.current_obj = QtCore.QRectF(self.origin_point, QtCore.QPointF(pos_x, pos_y))
            self.item_to_draw.setRect(self.current_obj)
            diag = self.current_obj.bottomRight() - self.current_obj.topLeft()
            x = self.current_obj.x() + self.area_tool_x_offsets * diag.x()
            y = self.current_obj.y() + self.area_tool_y_offsets * diag.y()
            for index, t1 in enumerate(x):
                t2 = y[index]
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
        for item in self.items():
            if isinstance(item, GraphicsPointItem):
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, flag)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, flag)

    def addPoint(self, point):
        p = GraphicsPointItem(point)
        p.setZValue(1.0)  # Ensure point is drawn above cross section
        self.addItem(p)


class GraphicsPointItem(QtWidgets.QAbstractGraphicsShapeItem):
    def __init__(self, point, *args, size=6, **kwargs):
        super().__init__(*args, **kwargs)

        self.size = size
        self.setPos(point)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)

    @property
    def x(self):
        return self.pos().x()

    @property
    def y(self):
        return self.pos().y()

    def boundingRect(self):
        pen_width = self.pen().widthF()
        half_pen_width = pen_width * 0.5

        top = -(self.size * 0.5) - half_pen_width
        new_size = self.size + pen_width
        return QtCore.QRectF(top, top, new_size, new_size)

    def paint(self, painter, _options, _widget):
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
