import math
from enum import Enum, unique
import numpy as np
from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.math import Vector4, Vector3
from sscanss.core.util import Camera, Colour, RenderMode, RenderPrimitive, SceneType, world_to_screen

SAMPLE_KEY = 'sample'


class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.camera = Camera(self.width()/self.height(), 60)

        self.scene = {}
        self.scene_type = SceneType.Sample

        self.render_colour = Colour.black()
        self.render_mode = RenderMode.Solid

        self.default_font = QtGui.QFont("Times", 10)

        self.parent_model.scene_updated.connect(self.loadScene)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def initializeGL(self):
        GL.glClearColor(*Colour.white())

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
        self.camera.aspect = width / height
        GL.glLoadMatrixf(self.camera.perspective.transpose())

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(self.camera.perspective.transpose())

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(self.camera.model_view.transpose())

        for _, node in list(self.scene.items()):
            self.recursive_draw(node)

    def recursive_draw(self, node):

        GL.glPushMatrix()
        GL.glMultMatrixf(node.transform.transpose())
        if node.colour is not None:
            self.render_colour = node.colour

        if node.render_mode is not None:
            self.render_mode = node.render_mode

        GL.glColor4f(*self.render_colour.rgbaf)

        if self.render_mode == RenderMode.Solid:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        elif self.render_mode == RenderMode.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_ONE_MINUS_SRC_COLOR)
            inverted_colour = self.render_colour.invert()
            GL.glColor4f(*inverted_colour.rgbaf)

        if node.vertices.size != 0 and node.indices.size != 0:
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointerf(node.vertices)
            if node.normals.size != 0:
                GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
                GL.glNormalPointerf(node.normals)

            render_primitive = GL.GL_TRIANGLES if node.render_primitive == RenderPrimitive.Triangles else GL.GL_LINES
            GL.glDrawElementsui(render_primitive, node.indices)

            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        # reset OpenGL State
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        for child in node.children:
            self.recursive_draw(child)

        GL.glPopMatrix()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        translation_speed = 0.001

        if event.buttons() == QtCore.Qt.LeftButton:
            p1 = (self.lastPos.x() / self.width() * 2, self.lastPos.y() / self.height() * 2)
            p2 = (event.x() / self.width() * 2, event.y() / self.height() * 2)
            self.camera.rotate(p1, p2)

        elif event.buttons() == QtCore.Qt.RightButton:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            x_offset = -dx * translation_speed
            y_offset = -dy * translation_speed
            self.camera.pan(x_offset, y_offset)

        self.lastPos = event.pos()
        self.update()

    def wheelEvent(self, event):
        zoom_scale = 0.05
        delta = 0.0
        num_degrees = event.angleDelta() / 8
        if not num_degrees.isNull():
            delta = num_degrees.y() / 15

        self.camera.zoom(delta * zoom_scale)
        self.update()

    @property
    def sampleRenderMode(self):
        if SAMPLE_KEY in self.scene:
            return self.scene[SAMPLE_KEY].render_mode
        else:
            return RenderMode.Solid

    @sampleRenderMode.setter
    def sampleRenderMode(self, render_mode):
        if SAMPLE_KEY in self.scene:
            self.scene[SAMPLE_KEY].render_mode = render_mode
            self.update()

    def loadScene(self):
        if self.scene_type == SceneType.Sample:
            self.scene = self.parent_model.sample_scene

        if SAMPLE_KEY in self.scene:
            bounding_box = self.scene[SAMPLE_KEY].bounding_box
            if bounding_box:
                self.camera.zoomToFit(bounding_box.center, bounding_box.radius)
            else:
                self.camera.reset()
        self.update()

    def project(self, x, y, z):
        world_point = Vector4([x, y, z, 1])
        model_view = self.camera.model_view
        perspective = self.camera.perspective

        return world_to_screen(world_point, model_view, perspective, self.width(), self.height())

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


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.createActions()

        self.show_grid = False
        self.snap_to_grid = False
        self.grid_x_size = 10
        self.grid_y_size = 10
        self.angle = 0.0
        self.setViewportUpdateMode(self.FullViewportUpdate)
        self.horizontalScrollBar().hide()
        self.horizontalScrollBar().setStyleSheet('QScrollBar {height:0px;}')
        self.verticalScrollBar().hide()
        self.verticalScrollBar().setStyleSheet('QScrollBar {width:0px;}')

    def createActions(self):
        zoom_in = QtWidgets.QAction("Zoom in", self)
        zoom_in.triggered.connect(self.on_zoom_in)
        zoom_in.setShortcut(QtGui.QKeySequence("+"))
        zoom_in.setShortcutContext(QtCore.Qt.WidgetShortcut)

        zoom_out = QtWidgets.QAction("Zoom out", self)
        zoom_out.triggered.connect(self.on_zoom_out)
        zoom_out.setShortcut(QtGui.QKeySequence("-"))
        zoom_out.setShortcutContext(QtCore.Qt.WidgetShortcut)

        reset = QtWidgets.QAction("Reset View", self)
        reset.triggered.connect(self.reset)
        reset.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace))
        reset.setShortcutContext(QtCore.Qt.WidgetShortcut)

        self.addActions([zoom_in, zoom_out, reset])

    def on_rotate(self, angle):
        if not self.scene():
            return

        self.angle = (self.angle + angle) % (2 * math.pi)
        self.rotate(math.degrees(angle))

    def reset(self):
        self.resetTransform()
        self.angle = 0.0

    def on_zoom_in(self):
        if not self.scene():
            return

        self.scale(1.5, 1.5)

    def on_zoom_out(self):
        if not self.scene():
            return

        self.scale(1.0 / 1.5, 1.0 / 1.5)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                self.lastPos = event.pos()
        elif event.button() == QtCore.Qt.RightButton:
            self.lastPos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.setCursor(QtCore.Qt.ArrowCursor)

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                va = Vector3([self.lastPos.x(), self.lastPos.y(), 0.]).normalized()
                vb = Vector3([event.pos().x(), event.pos().y(), 0.]).normalized()

                angle = math.acos(min(1.0, va | vb))
                self.on_rotate(angle)

        elif event.buttons() == QtCore.Qt.RightButton:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)

        self.lastPos = event.pos()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        mode = self.scene().mode
        self.scene().mode = Scene.Mode.Select

        delta = 0.0
        num_degrees = event.angleDelta() / 8
        if not num_degrees.isNull():
            delta = num_degrees.y() / 15

        if delta < 0:
            self.on_zoom_out()
        elif delta > 0:
            self.on_zoom_in()

        self.scene().mode = mode

    def drawBackground(self, painter, rect):
        if not self.show_grid:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setOpacity(0.3)
        pen = QtGui.QPen(QtCore.Qt.darkGreen)
        painter.setPen(pen)

        cosA = math.cos(-self.angle)
        sinA = math.sin(-self.angle)

        scene_rect = rect.toRect()
        scene_center = scene_rect.center()
        x_center = scene_center.x()
        y_center = scene_center.y()

        width = self.width() if self.width() > scene_rect.width() else scene_rect.width()
        height = self.height() if self.height() > scene_rect.height() else scene_rect.height()

        left = x_center - width // 2
        top = y_center - height // 2
        right = left + width + 2 * self.grid_x_size
        bottom = top + height + 2 * self.grid_y_size

        left = left - left % self.grid_x_size
        top = top - top % self.grid_y_size

        x = np.array(range(left, right, self.grid_x_size))
        y = np.array(range(top, bottom, self.grid_y_size))

        # should be optimized further
        for old_x in x:
            _x = ((old_x - x_center) * cosA - (top - y_center) * sinA) + x_center
            _y = ((old_x - x_center) * sinA + (top - y_center) * cosA) + y_center
            _x_end = ((old_x - x_center) * cosA - (bottom - y_center) * sinA) + x_center
            _y_end = ((old_x - x_center) * sinA + (bottom - y_center) * cosA) + y_center
            painter.drawLine(_x, _y, _x_end, _y_end)

        for old_y in y:
            _x = ((left - x_center) * cosA - (old_y - y_center) * sinA) + x_center
            _y = ((left - x_center) * sinA + (old_y - y_center) * cosA) + y_center
            _x_end = ((right - x_center) * cosA - (old_y - y_center) * sinA) + x_center
            _y_end = ((right - x_center) * sinA + (old_y - y_center) * cosA) + y_center
            painter.drawLine(_x, _y, _x_end, _y_end)

        painter.restore()


class Scene(QtWidgets.QGraphicsScene):
    @unique
    class Mode(Enum):
        Select = 1
        Draw_point = 2
        Draw_line = 3
        Draw_area = 4

    def __init__(self, parent=None):
        super().__init__(parent)

        self.itemToDraw = None
        self.linePointCount(2)
        self.current_obj = None
        self.area = (3, 3)
        self.area_x = np.repeat(np.linspace(0., 1., self.area[0]), self.area[1])
        self.area_y = np.tile(np.linspace(0., 1., self.area[1]), self.area[0])
        self._mode = Scene.Mode.Select

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

        if value == Scene.Mode.Select:
            self.makeItemsControllable(True)
            view_mode = QtWidgets.QGraphicsView.RubberBandDrag
        else:
            self.makeItemsControllable(False)
            view_mode = QtWidgets.QGraphicsView.NoDrag

        mView = self.views()
        if mView:
            mView[0].setDragMode(view_mode)

            if value == Scene.Mode.Select:
                mView[0].setCursor(QtCore.Qt.ArrowCursor)
            else:
                mView[0].setCursor(QtCore.Qt.CrossCursor)

    def areaPointCount(self, value):
        self.area = value
        self.area_x = np.repeat(np.linspace(0., 1., self.area[0]), self.area[1])
        self.area_y = np.tile(np.linspace(0., 1., self.area[1]), self.area[0])

    def linePointCount(self, value):
        self.point_count = value
        self.point_offsets = np.linspace(0., 1., self.point_count)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            view = self.views()[0]
            pos_x = event.scenePos().x()
            pos_y = event.scenePos().y()
            if view.snap_to_grid:
                pos_x = round(pos_x / view.grid_x_size) * view.grid_x_size
                pos_y = round(pos_y / view.grid_y_size) * view.grid_y_size
                pos = QtCore.QPointF(pos_x, pos_y)
            else:
                pos = event.scenePos()

            if self.mode == Scene.Mode.Draw_point:
                p = GraphicsPointItem(pos)
                self.addItem(p)
            elif self.mode != Scene.Mode.Select:
                self.origPoint = pos

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == Scene.Mode.Draw_line:
            if self.itemToDraw is None:
                self.itemToDraw = QtWidgets.QGraphicsLineItem()
                self.addItem(self.itemToDraw)
                self.itemToDraw.setPen(QtGui.QPen(QtCore.Qt.black, 0, QtCore.Qt.SolidLine))
                self.itemToDraw.setPos(self.origPoint)

            self.itemToDraw.setLine(0, 0,
                                    event.scenePos().x() - self.origPoint.x(),
                                    event.scenePos().y() - self.origPoint.y())

            self.current_obj = QtCore.QLineF(self.origPoint, event.scenePos())

        elif self.mode == Scene.Mode.Draw_area:
            if self.itemToDraw is None:
                self.itemToDraw = QtWidgets.QGraphicsRectItem()
                self.addItem(self.itemToDraw)
                self.itemToDraw.setPen(QtGui.QPen(QtCore.Qt.black, 0, QtCore.Qt.SolidLine))
                self.itemToDraw.setPos(self.origPoint)

            self.itemToDraw.setRect(QtCore.QRectF(QtCore.QPoint(0, 0),
                                                  QtCore.QPoint(event.scenePos().x() - self.origPoint.x(),
                                                                event.scenePos().y() - self.origPoint.y())))

            self.current_obj = QtCore.QRectF(self.origPoint, event.scenePos())

        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.itemToDraw is None:
            super().mouseReleaseEvent(event)
            return
        view = self.views()[0]
        pos_x = event.scenePos().x()
        pos_y = event.scenePos().y()
        if view.snap_to_grid:
            pos_x = round(pos_x / view.grid_x_size) * view.grid_x_size
            pos_y = round(pos_y / view.grid_y_size) * view.grid_y_size

        if self.mode == Scene.Mode.Draw_line:

            self.itemToDraw.setLine(0, 0,
                                    pos_x - self.origPoint.x(),
                                    pos_y - self.origPoint.y())

            self.current_obj = QtCore.QLineF(self.origPoint, QtCore.QPointF(pos_x, pos_y))

            for t in self.point_offsets:
                point = self.current_obj.pointAt(t)
                p = GraphicsPointItem(point)
                self.addItem(p)

        elif self.mode == Scene.Mode.Draw_area:
            self.itemToDraw.setRect(QtCore.QRectF(QtCore.QPoint(0, 0),
                                                  QtCore.QPoint(pos_x - self.origPoint.x(),
                                                                pos_y - self.origPoint.y())))

            self.current_obj = QtCore.QRectF(self.origPoint, QtCore.QPointF(pos_x, pos_y))
            diag = self.current_obj.bottomRight() - self.current_obj.topLeft()
            x = self.current_obj.x() + self.area_x * diag.x()
            y = self.current_obj.y() + self.area_y * diag.y()
            for index, t1 in enumerate(x):
                t2 = y[index]
                point = QtCore.QPointF(t1, t2)
                p = GraphicsPointItem(point)
                self.addItem(p)

        self.removeItem(self.itemToDraw)
        self.itemToDraw = None

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            for item in self.selectedItems():
                self.removeItem(item)
                del item
        else:
            super().keyPressEvent(event)

    def makeItemsControllable(self, flag):
        for item in self.items():
            if isinstance(item, GraphicsPointItem):
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, flag)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, flag)


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
