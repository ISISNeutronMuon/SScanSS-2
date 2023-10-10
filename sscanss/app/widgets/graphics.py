from abc import ABC, abstractmethod
from enum import Enum, unique
import math
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from sscanss.core.math import Vector3, clamp, angle_axis_btw_vectors


class GraphicsView(QtWidgets.QGraphicsView):
    """Provides container for graphics scene

    :param scene: scene
    :type scene: GraphicsScene
    """
    @unique
    class DrawMode(Enum):
        """Draw mode for graphics scene"""
        None_ = 1
        Point = 2
        Line = 3
        Rectangle = 4

    def __init__(self, scene):
        super().__init__(scene)
        self.createActions()

        self.show_grid = False
        self.snap_to_grid = False
        self.show_help = False
        self.has_foreground = False
        self.grid = BoxGrid()
        self.interaction = GraphicsViewInteractor(self)
        self.object_snap_tool = ObjectSnap(self, self.grid)
        self.object_snap_tool.object_moved.connect(self.translateSceneItems)
        self.zoom_factor = 1.2
        self.viewport_rect = QtCore.QRectF()
        self.draw_tool = self.createDrawTool(GraphicsView.DrawMode.None_)
        self.grid_pen = QtGui.QPen(QtCore.Qt.GlobalColor.darkGreen, 0)
        self.setMouseTracking(True)

        self.setViewportUpdateMode(self.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    @property
    def viewport_anchor_in_scene(self):
        """Returns object anchor for snap object to grid operation

        :return: anchor point in scene coordinates
        :rtype: QtCore.QPointF
        """
        return self.scene().transform.map(self.viewport_rect.center())

    @property
    def snap_object_to_grid(self):
        """Returns if snap object to grid is enabled

       :return: indicates if snap object to grid is enabled
       :rtype: bool
       """
        return self.object_snap_tool.enabled

    @snap_object_to_grid.setter
    def snap_object_to_grid(self, value):
        self.object_snap_tool.enabled = value
        scene = self.scene()
        if value:
            scene.addItem(scene.anchor_item)
        else:
            scene.removeItem(scene.anchor_item)

    @property
    def object_anchor(self):
        """Returns object anchor for snap object to grid operation

        :return: anchor point
        :rtype: QtCore.QPointF
        """
        return self.object_snap_tool.anchor

    @object_anchor.setter
    def object_anchor(self, value):
        scene = self.scene()
        self.object_snap_tool.setAnchor(value, scene.transform)
        anchor = scene.transform.map(value)
        scene.anchor_item.setPos(anchor)
        rot = QtGui.QTransform(scene.transform.m11(), scene.transform.m12(), scene.transform.m21(),
                               scene.transform.m22(), 0, 0)
        scene.anchor_item.setTransform(rot)
        self.update()

    def createDrawTool(self, mode, count=()):
        """Creates a draw tool for the specified draw mode

        :param mode: draw mode
        :type mode: DrawMode
        :param count: number of points for each dimension of the tool shape
        :type count: Tuple[int]
        :return: draw tool instance
        :rtype: Optional[DrawTool]
        """
        if mode == GraphicsView.DrawMode.None_:
            self.scene().makeItemsControllable(True)
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
            return

        if mode == GraphicsView.DrawMode.Point:
            tool = PointTool(self)
        elif mode == GraphicsView.DrawMode.Line:
            tool = LineTool(self, *count)
        else:
            tool = RectangleTool(self, *count)

        tool.outline_updated.connect(self.scene().updateOutlineItem)
        tool.point_drawn.connect(self.scene().addPoint)
        self.scene().makeItemsControllable(False)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

        return tool

    def setDrawToolPointCount(self, count):
        """Changes the point count for the current tool

        :param count: number of points for each dimension of the tool shape
        :type count: Tuple[int]
        """
        if self.draw_tool is None:
            return

        self.draw_tool.setPointCount(*count)

    def drawForeground(self, painter, _rect):
        """Draws the help information in foreground using the given painter and rect"""
        if not self.show_help:
            self.has_foreground = False
            return

        spacing = 10

        text_document = QtGui.QTextDocument()
        text_document.setDefaultStyleSheet("* { color: #ffffff }")
        text_document.setHtml("<h3 align=\"center\">Shortcuts</h3>"
                              "<div>"
                              "<pre>Delete&#9;&nbsp;Deletes selected point</pre>"
                              "<pre>+ \\ -&#9;&nbsp;Zoom in \\ out </pre>"
                              "<pre>Mouse&#9;&nbsp;Zoom in \\ out<br>Wheel</pre>"
                              "<pre>Right&#9;&nbsp;Rotate view<br>Click</pre>"
                              "<pre>Ctrl + &#9;&nbsp;Pan view<br>Right Click</pre>"
                              "<pre>Middle &#9;&nbsp;Pan view<br>Click</pre>"
                              "<pre>Ctrl + R&#9;&nbsp;Reset view</pre>"
                              "</div></table>")
        text_document.setTextWidth(text_document.size().width())

        text_rect = QtCore.QRect(0, 0, 300, 320)
        painter.save()
        transform = QtGui.QTransform()
        painter.setWorldTransform(transform.translate(self.width() // 2, self.height() // 2))
        painter.translate(-text_rect.center().x() - spacing, -text_rect.center().y() - spacing)
        pen = QtGui.QPen(QtGui.QColor(180, 180, 180), 3)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 230))
        painter.drawRoundedRect(text_rect, 20, 20)

        painter.translate(spacing, spacing)
        text_document.drawContents(painter)
        painter.restore()
        self.has_foreground = True
        self.show_help = False

    def createActions(self):
        """Creates widget actions"""
        zoom_in = QtGui.QAction("Zoom in", self)
        zoom_in.triggered.connect(self.zoomIn)
        zoom_in.setShortcut(QtGui.QKeySequence("+"))
        zoom_in.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        zoom_out = QtGui.QAction("Zoom out", self)
        zoom_out.triggered.connect(self.zoomOut)
        zoom_out.setShortcut(QtGui.QKeySequence("-"))
        zoom_out.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        reset = QtGui.QAction("Reset View", self)
        reset.triggered.connect(self.reset)
        reset.setShortcut(QtGui.QKeySequence('Ctrl+R'))
        reset.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

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
        self.object_snap_tool.grid = self.grid

    def setGridSize(self, size):
        """Sets size of active grid

        :param size: size of grid
        :type size: Tuple[int, int]
        """
        self.grid.size = size
        self.scene().update()
        self.object_snap_tool.grid = self.grid

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
        transform = QtGui.QTransform.fromTranslate(offset.x(), offset.y())
        transform.rotateRadians(angle)
        transform.translate(-offset.x(), -offset.y())
        self.scene().transform *= transform
        gr.setTransform(transform)
        self.object_snap_tool.updateOffset(transform)
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
        transform = QtGui.QTransform.fromTranslate(dx, dy)
        self.scene().transform *= transform
        gr.setTransform(transform)
        self.scene().destroyItemGroup(gr)
        self.scene().update()

    def resetTransform(self):
        """Resets scene transform to identity"""
        self.scene().transform.reset()
        super().resetTransform()

    def reset(self):
        """Resets the camera of the graphics view"""
        if not self.scene() or not self.scene().items():
            return

        gr = self.scene().createItemGroup(self.scene().items())
        gr.setTransform(self.scene().transform.inverted()[0])
        self.object_snap_tool.reset()
        self.scene().destroyItemGroup(gr)

        rect = self.scene().itemsBoundingRect()
        # calculate new rectangle that encloses original rect with a different anchor
        rect.united(rect.translated(self.viewport_rect.center() - rect.center()))

        self.resetTransform()
        self.setSceneRect(rect)
        self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def zoomIn(self):
        """Zooms in the camera of the graphics view"""
        if not self.scene():
            return

        self.scale(self.zoom_factor, self.zoom_factor)
        # anchor = self.scene().transform.mapRect(self.anchor)
        self.centerOn(self.viewport_anchor_in_scene)

    def zoomOut(self):
        """Zooms out the camera of the graphics view"""
        if not self.scene():
            return

        factor = 1.0 / self.zoom_factor
        self.scale(factor, factor)
        # anchor = self.scene().transform.mapRect(self.anchor)
        self.centerOn(self.viewport_anchor_in_scene)

    def drawBackground(self, painter, rect):
        """Draws the grid in background using the given painter and rect"""
        if not self.show_grid:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(0.3)
        painter.setPen(self.grid_pen)

        center = self.viewport_rect.center()
        top_left = rect.topLeft() - center
        bottom_right = rect.bottomRight() - center
        half_size = QtCore.QPointF(max(abs(top_left.x()), abs(bottom_right.x())),
                                   max(abs(top_left.y()), abs(bottom_right.y())))
        adjusted_rect = QtCore.QRectF(center - half_size, center + half_size)
        self.grid.render(painter, adjusted_rect)

        painter.restore()


class GraphicsViewInteractor(QtCore.QObject):
    """A tool class that provides the scene manipulations like panning, zooming and rotating for
    the graphics view

    :param graphics_view: graphics view instance
    :type graphics_view: GraphicsView
    """
    mouse_moved = QtCore.pyqtSignal(object)

    @unique
    class State(Enum):
        """State of GraphicsViewInteractor"""
        None_ = 0
        Pan = 1
        Rotate = 2
        Zoom = 3

    def __init__(self, graphics_view):
        super().__init__()

        self.graphics_view = graphics_view
        graphics_view.viewport().installEventFilter(self)

        self._state = GraphicsViewInteractor.State.None_

        self.last_pos = QtCore.QPointF()
        self.last_cursor = graphics_view.cursor()
        self.last_drag_mode = graphics_view.dragMode()

    @property
    def state(self):
        """Gets the active state of the interactor"""
        return self._state

    @state.setter
    def state(self, value):
        """Sets appropriate cursor and drag mode for the state

        :param value: state
        :type value: GraphicsViewInteractor.State
        """
        if self._state == GraphicsViewInteractor.State.None_:
            self.last_cursor = self.graphics_view.cursor()
            self.last_drag_mode = self.graphics_view.dragMode()

        self._state = value
        if value == GraphicsViewInteractor.State.Pan:
            self.graphics_view.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.graphics_view.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        if value == GraphicsViewInteractor.State.Rotate:
            self.graphics_view.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.graphics_view.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        if value == GraphicsViewInteractor.State.Zoom:
            self.graphics_view.setCursor(self.last_cursor)
            self.graphics_view.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        if value == GraphicsViewInteractor.State.None_:
            self.graphics_view.setCursor(self.last_cursor)
            self.graphics_view.setDragMode(self.last_drag_mode)

    def isRotating(self, event):
        """Checks if the selected mouse button and keybind is for rotation

        :param event: mouse event
        :type event: QtGui.QMouseEvent
        :return: indicate the event is for rotation
        :rtype: bool
        """
        is_rotating = ((event.button() == QtCore.Qt.MouseButton.RightButton
                        or event.buttons() == QtCore.Qt.MouseButton.RightButton)
                       and event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier)

        if is_rotating:
            self.state = GraphicsViewInteractor.State.Rotate

        return is_rotating

    def isPanning(self, event):
        """Checks if the selected mouse button and keybind is for panning

        :param event: mouse event
        :type event: QtGui.QMouseEvent
        :return: indicate the event is for panning
        :rtype: bool
        """
        default_keybind = ((event.button() == QtCore.Qt.MouseButton.MiddleButton
                            or event.buttons() == QtCore.Qt.MouseButton.MiddleButton)
                           and event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier)
        alt_keybind = ((event.button() == QtCore.Qt.MouseButton.RightButton
                        or event.buttons() == QtCore.Qt.MouseButton.RightButton)
                       and event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier)

        if default_keybind or alt_keybind:
            self.state = GraphicsViewInteractor.State.Pan

        return default_keybind or alt_keybind

    def rotate(self, pos, center):
        """Rotates scene items based on mouse movements

        :param pos: mouse position
        :type pos: QtCore.QPointF
        :param center: center of viewport
        :type center: QtCore.QPointF
        """
        adj_pos = pos - center
        adj_last_pos = self.last_pos - center

        if adj_pos.manhattanLength() < 0.1 or adj_last_pos.manhattanLength() < 0.1:
            return

        va = Vector3([adj_last_pos.x(), adj_last_pos.y(), 0.]).normalized
        vb = Vector3([adj_pos.x(), adj_pos.y(), 0.]).normalized
        angle = -math.acos(clamp(va | vb, -1.0, 1.0))

        if np.dot([0., 0., 1.], va ^ vb) > 0:
            angle = -angle

        self.graphics_view.rotateSceneItems(angle, center)

    def pan(self, pos):
        """Pans scene items based on mouse movements

        :param pos: mouse position
        :type pos: QtCore.QPointF
        """
        dx = pos.x() - self.last_pos.x()
        dy = pos.y() - self.last_pos.y()
        self.graphics_view.translateSceneItems(dx, dy)

    def zoom(self, angle_delta):
        """Zooms view based on mouse movements

        :param angle_delta: relative amount that the wheel was rotated
        :type angle_delta: QtCore.QPointF
        """
        delta = 0.0
        num_degrees = angle_delta / 8
        if not num_degrees.isNull():
            delta = num_degrees.y() / 15

        if delta < 0:
            self.graphics_view.zoomOut()
        elif delta > 0:
            self.graphics_view.zoomIn()

    def interact(self, pos, center):
        """Rotates or pans scene items

        :param pos: mouse position
        :type pos: QtCore.QPointF
        :param center: center of viewport
        :type center: QtCore.QPointF
        """
        if self.state == GraphicsViewInteractor.State.Pan:
            self.pan(pos)
        elif self.state == GraphicsViewInteractor.State.Rotate:
            self.rotate(pos, center)

    def eventFilter(self, obj, event):
        """Intercepts the mouse events and computes anchor snapping based on mouse movements

        :param obj: widget
        :type obj: QtWidgets.QWidget
        :param event: Qt events
        :type event: QtCore.QEvent
        :return: indicates if event was handled
        :rtype: bool
        """
        if event.type() == QtCore.QEvent.Type.MouseButtonPress and (self.isRotating(event) or self.isPanning(event)):
            self.last_pos = self.graphics_view.mapToScene(event.pos())

        if event.type() == QtCore.QEvent.Type.MouseMove:
            self.mouse_moved.emit(event.pos())
            if self.isRotating(event) or self.isPanning(event):
                pos = self.graphics_view.mapToScene(event.pos())
                self.interact(pos, self.graphics_view.viewport_anchor_in_scene)
                self.last_pos = pos

        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if self.state != GraphicsViewInteractor.State.None_:
                self.state = GraphicsViewInteractor.State.None_

        if event.type() == QtCore.QEvent.Type.Wheel and event.buttons() == QtCore.Qt.MouseButton.NoButton:
            self.state = GraphicsViewInteractor.State.Zoom
            self.zoom(event.angleDelta())
            self.state = GraphicsViewInteractor.State.None_

        if event.type() == QtCore.QEvent.Type.Leave:
            self.mouse_moved.emit(QtCore.QPoint(-1, -1))

        return False


class ObjectSnap(QtCore.QObject):
    """A tool for moving an anchor point on a scene object along the intersection of a grid. The mouse
    events are acquired via event filters when the tool is enabled then the appropriate offsets (dx, dy)
    to the next grid intersection is computed and returned via a signal. Both box and polar grid are
    supported, the polar grid is solved by dividing the problem radial snap and angular snap, each
    having a different keybindings.

    :param graphics_view: graphic view widget with scene
    :type graphics_view: GraphicsView
    :param grid: grid
    :type grid: Grid
    """
    object_moved = QtCore.pyqtSignal(float, float)

    def __init__(self, graphics_view, grid):
        super().__init__()

        self.graphics_view = graphics_view
        self.graphics_view.viewport().installEventFilter(self)
        self.enabled = False

        self.last_pos = QtCore.QPointF()
        self.remaining_shift = (0, 0)

        self.anchor = QtCore.QPointF()
        self.anchor_offset = QtCore.QPointF()
        self.initialized = False
        self._grid = grid

    @property
    def grid(self):
        """Returns grid object

        :return: grid to snap anchor to
        :rtype: Grid
        """
        return self._grid

    @grid.setter
    def grid(self, value):
        """Sets the grid object to snap to

        :param value: grid to snap anchor to
        :type value: Grid
        """
        self._grid = value
        self.initialized = False

    def setAnchor(self, anchor, transform):
        """Sets the anchor point to be moved on the grid

        :param anchor: anchor point
        :type anchor: QtCore.QPointF
        :param transform: scene transformation
        :type transform: QtGui.QTransform
        """
        anchor_in_scene = transform.map(anchor)
        self.anchor_offset = anchor_in_scene - anchor

        self.anchor = anchor
        self.initialized = False

    def updateOffset(self, transform):
        """Updates the anchor offset when the scene is transformed by rotation

        :param transform: scene transform
        :type transform: QtGui.QTransform
        """
        self.anchor_offset = transform.map(self.anchor + self.anchor_offset) - self.anchor
        self.initialized = False

    def reset(self):
        """Resets anchor point"""
        self.anchor_offset = QtCore.QPointF()
        self.initialized = False

    def isPanning(self, event):
        """Checks if the selected mouse button and keybind is for panning

        :param event: mouse event
        :type event: QtGui.QMouseEvent
        :return: indicate the event is for panning
        :rtype: bool
        """
        default_keybind = (event.buttons() == QtCore.Qt.MouseButton.MiddleButton and event.modifiers()
                           in [QtCore.Qt.KeyboardModifier.NoModifier, QtCore.Qt.KeyboardModifier.ShiftModifier])
        alt_keybind = (event.buttons() == QtCore.Qt.MouseButton.RightButton and event.modifiers() in [
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier
        ])

        return default_keybind or alt_keybind

    def initialize(self, anchor):
        """Initializes the anchor by snapping to the closest grid intersection

        :param anchor: anchor point
        :type anchor: QtCore.QPointF
        :return: new anchor
        :rtype: QtCore.QPointF
        """
        if self.initialized:
            return anchor

        snapped_anchor = self.grid.snap(self.anchor + self.anchor_offset)
        diff = snapped_anchor - anchor
        anchor += diff
        self.object_moved.emit(diff.x(), diff.y())  # Updates scene transform
        self.initialized = True

        return anchor

    def snapAnchorToBoxGrid(self, start_pos, stop_pos):
        """Snaps anchor point to the next box grid intersection in the direction indicated
        by the mouse drag (start and stop pos)

        :param start_pos: starting scene coordinates
        :type start_pos: QtCore.QPointF
        :param stop_pos: stopping scene coordinates
        :type stop_pos: QtCore.QPointF
        :return: offset to next grid intersection
        :rtype: Tuple(float, float)
        """
        anchor = self.initialize(self.anchor + self.anchor_offset)
        strength = 200  # Should find a better way of selecting snap strength
        diff = QtCore.QPointF(*self.remaining_shift) + (stop_pos - start_pos)
        shift_amount = (diff.x() // strength, diff.y() // strength)
        remaining_shift = (diff.x() % strength, diff.y() % strength)
        dx = shift_amount[0] * self.grid.x
        dy = shift_amount[1] * self.grid.y
        new_anchor = QtCore.QPointF(anchor.x() + dx, anchor.y() + dy)
        self.anchor_offset = new_anchor - self.anchor
        self.remaining_shift = remaining_shift

        return dx, dy

    def snapAnchorToPolarGrid(self, start_pos, stop_pos, alt):
        """Snaps anchor point to the next polar grid intersection in the direction indicated
        by the mouse drag (start and stop pos). The anchor will snap the radial direction only
        if alt is False otherwise will snap to angular.

        :param start_pos: starting scene coordinates
        :type start_pos: QtCore.QPointF
        :param stop_pos: stopping scene coordinates
        :type stop_pos: QtCore.QPointF
        :param alt: indicates snap in angular direction
        :type alt: bool
        :return: offset to next grid intersection
        :rtype: Tuple(float, float)
        """
        anchor = self.initialize(self.anchor + self.anchor_offset)
        if alt:
            adj_stop_pos = stop_pos - self.grid.center
            adj_start_pos = start_pos - self.grid.center

            if adj_stop_pos.manhattanLength() < 0.1 or adj_start_pos.manhattanLength() < 0.1:
                return 0, 0

            va = Vector3([adj_start_pos.x(), adj_start_pos.y(), 0.]).normalized
            vb = Vector3([adj_stop_pos.x(), adj_stop_pos.y(), 0.]).normalized
            angle = -math.acos(clamp(va | vb, -1.0, 1.0))

            if np.dot([0., 0., 1.], va ^ vb) > 0:
                angle = -angle

            strength = 0.7  # Should find a better way of selecting snap strength
            shift_amount = math.trunc((self.remaining_shift[0] + angle) / strength)
            remaining_shift = math.fmod(self.remaining_shift[0] + angle, strength)

            center = self.grid.center
            adj_anchor = anchor - center
            adj_anchor = self.grid.toPolar(adj_anchor.x(), adj_anchor.y())

            new_angle = adj_anchor[1]
            if shift_amount != 0:
                div = round(new_angle / self.grid.angular)
                max_div = 360 // self.grid.angular
                new_angle = self.grid.angular * ((div + shift_amount) % (max_div + 1))

            snapped = self.grid.toCartesian(adj_anchor[0], new_angle)

            dx = snapped[0] + center.x() - anchor.x()
            dy = snapped[1] + center.y() - anchor.y()
            self.remaining_shift = (remaining_shift, 0)

            new_anchor = QtCore.QPointF(snapped[0] + center.x(), snapped[1] + center.y())
            self.anchor_offset = new_anchor - self.anchor
        else:
            drag_vector = stop_pos - start_pos
            drag_vector = Vector3([drag_vector.x(), drag_vector.y(), 0.])
            length = drag_vector.length
            drag_vector /= length

            # angle btw drag vector and x-axis
            drag_angle, drag_axis = angle_axis_btw_vectors(Vector3([1, 0, 0]), drag_vector)
            if drag_axis[2] < 0:
                drag_angle = 2 * math.pi - drag_angle

            round_angle = math.radians(self.grid.angular * round(math.degrees(drag_angle) / self.grid.angular))
            round_angle = min(round_angle, 360)

            strength = 200  # Should find a better way of selecting snap strength
            shift_amount = math.trunc((self.remaining_shift[0] + length) / strength)
            remaining_shift = math.fmod(self.remaining_shift[0] + length, strength)

            center = self.grid.center
            adj_anchor = anchor - center
            adj_anchor = self.grid.toPolar(adj_anchor.x(), adj_anchor.y())

            adj_angle = math.radians(adj_anchor[1])
            adj_vector = [math.cos(adj_angle), math.sin(adj_angle), 0]
            sign = 1 if np.dot(adj_vector, drag_vector) > 0 else -1

            if -0.01 < adj_anchor[0] < 0.01:
                new_radial = adj_anchor[0] + shift_amount * self.grid.radial
                snapped = self.grid.toCartesian(new_radial, math.degrees(round_angle))
            else:
                new_radial = adj_anchor[0] + sign * shift_amount * self.grid.radial
                snapped = self.grid.toCartesian(new_radial, adj_anchor[1])

            dx = snapped[0] + center.x() - anchor.x()
            dy = snapped[1] + center.y() - anchor.y()
            new_anchor = QtCore.QPointF(snapped[0] + center.x(), snapped[1] + center.y())

            self.anchor_offset = new_anchor - self.anchor
            self.remaining_shift = (remaining_shift, 0)

        return dx, dy

    def snapAnchor(self, start_pos, stop_pos, alt):
        """Snaps anchor to the next grid intersection

        :param start_pos: starting scene coordinates
        :type start_pos: QtCore.QPointF
        :param stop_pos: stopping scene coordinates
        :type stop_pos: QtCore.QPointF
        :param alt: indicates if alternate mode should be used
        :type alt: bool
        """
        if self.grid.type == Grid.Type.Box:
            dx, dy = self.snapAnchorToBoxGrid(start_pos, stop_pos)
        else:
            dx, dy = self.snapAnchorToPolarGrid(start_pos, stop_pos, alt)

        self.object_moved.emit(dx, dy)

    def eventFilter(self, obj, event):
        """Intercepts the mouse events and computes anchor snapping based on mouse movements

        :param obj: widget
        :type obj: QtWidgets.QWidget
        :param event: Qt events
        :type event: QtCore.QEvent
        :return: indicates if event was handled
        :rtype: bool
        """
        if not self.enabled:
            return False

        if event.type() == QtCore.QEvent.Type.MouseButtonPress and self.isPanning(event):
            self.last_pos = self.graphics_view.mapToScene(event.pos())
            self.remaining_shift = (0, 0)
            return True
        if event.type() == QtCore.QEvent.Type.MouseMove and self.isPanning(event):
            cur_pos = self.graphics_view.mapToScene(event.pos())
            self.snapAnchor(self.last_pos, cur_pos, event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
            self.last_pos = cur_pos
            return True

        return False


class DrawTool(QtCore.QObject):
    """A tool class for drawing points in the graphics view's scene

    :param graphics_view: graphics view instance
    :type graphics_view: GraphicsView
    """
    outline_updated = QtCore.pyqtSignal(object)
    point_drawn = QtCore.pyqtSignal(object)

    def __init__(self, graphics_view):
        super().__init__()

        self.graphics_view = graphics_view
        graphics_view.scene().installEventFilter(self)
        self.mouse_enabled = True
        self.start_pos = QtCore.QPointF()
        self.stop_pos = QtCore.QPointF()

    @property
    def mouse_enabled(self):
        """Returns if mouse interaction is enabled

       :return: indicates if mouse interaction is enabled
       :rtype: bool
       """
        return self._mouse_enabled

    @mouse_enabled.setter
    def mouse_enabled(self, state):
        if state:
            self.graphics_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.graphics_view.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self._mouse_enabled = state

    def isDrawing(self, event):
        """Checks if the selected mouse button and keybind is for drawing

        :param event: mouse event
        :type event: QtWidgets.QGraphicsSceneMouseEvent
        :return: indicate the event is for drawing
        :rtype: bool
        """
        return ((event.button() == QtCore.Qt.MouseButton.LeftButton
                 or event.buttons() == QtCore.Qt.MouseButton.LeftButton)
                and event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier)

    def updateOutline(self, clear=False):
        """Sends real-time updates on the outline as the mouse moves

        :param clear: indicates outline should be cleared
        :type clear: bool
        """
        self.outline_updated.emit(None if clear else self.getOutline())

    @property
    def mode(self):
        """Gets the draw mode of the tool

        :return: draw mode
        :rtype: GraphicView.DrawMode
        """

    def setPointCount(self, count):
        """Sets the number of points for each dimension of the tool

        :param count: number of points on the geometry
        :type count: Tuple[...]
        """

    def getOutline(self):
        """Gets the outline shape for the tool"""

    def drawPoints(self):
        """Generates points on the outline shape and send signal to draw them"""

    def eventFilter(self, obj, event):
        """Intercepts the mouse events and computes outline shape based on mouse movements

        :param obj: widget
        :type obj: QtWidgets.QWidget
        :param event: Qt events
        :type event: QtCore.QEvent
        :return: indicates if event was handled
        :rtype: bool
        """
        if not self.mouse_enabled:
            return False

        if event.type() == QtCore.QEvent.Type.GraphicsSceneMousePress and self.isDrawing(event):
            self.start_pos = event.scenePos()
            if self.graphics_view.snap_to_grid:
                self.start_pos = self.graphics_view.grid.snap(self.start_pos)

        if event.type() == QtCore.QEvent.Type.GraphicsSceneMouseMove and self.isDrawing(event):
            self.stop_pos = event.scenePos()
            self.updateOutline()

        if event.type() == QtCore.QEvent.Type.GraphicsSceneMouseRelease and self.isDrawing(event):
            self.stop_pos = event.scenePos()
            if self.graphics_view.snap_to_grid:
                self.stop_pos = self.graphics_view.grid.snap(self.stop_pos)
            self.updateOutline(True)
            self.drawPoints()

        return False


class PointTool(DrawTool):
    """A tool class for drawing a single point in the graphics view's scene

    :param graphics_view: graphics view instance
    :type graphics_view: GraphicsView
    """
    def __init__(self, graphics_view):
        super().__init__(graphics_view)

    @property
    def mode(self):
        return GraphicsView.DrawMode.Point

    def getOutline(self):
        return None

    def drawPoints(self):
        self.point_drawn.emit(self.stop_pos)


class LineTool(DrawTool):
    """A tool class for drawing points on a line in the graphics view's scene

    :param graphics_view: graphics view instance
    :type graphics_view: GraphicsView
    """
    def __init__(self, graphics_view, count=2):
        super().__init__(graphics_view)

        self.setPointCount(count)

    @property
    def mode(self):
        return GraphicsView.DrawMode.Line

    def setPointCount(self, count):
        """Sets the number of points on the line

        :param count: number of points on the line
        :type count: int
        """
        self.offsets = np.linspace(0., 1., count)

    def getOutline(self):
        """Gets the outline shape for the line tool

        :return: line shape
        :rtype: PyQt6.QtCore.QLineF
        """
        return QtCore.QLineF(self.start_pos, self.stop_pos)

    def drawPoints(self):
        line = self.getOutline()
        for t in self.offsets:
            self.point_drawn.emit(line.pointAt(t))


class RectangleTool(DrawTool):
    """A tool class for drawing point on a rectangle in the graphics view's scene

    :param graphics_view: graphics view instance
    :type graphics_view: GraphicsView
    :param x_count: number of points on the x-axis of the rectangle
    :type x_count: int
    :param y_count: number of points on the y-axis of the rectangle
    :type y_count: int
    """
    def __init__(self, graphics_view, x_count=2, y_count=2):
        super().__init__(graphics_view)

        self.setPointCount(x_count, y_count)

    @property
    def mode(self):
        return GraphicsView.DrawMode.Rectangle

    def setPointCount(self, x_count, y_count):
        """Sets the number of points for each dimension of the rectangle

        :param x_count: number of points on the x-axis of the rectangle
        :type x_count: int
        :param y_count: number of points on the y-axis of the rectangle
        :type y_count: int
        """
        self.x_offsets = np.tile(np.linspace(0., 1., x_count), y_count)
        self.y_offsets = np.repeat(np.linspace(0., 1., y_count), x_count)

    def getOutline(self):
        """Gets the outline shape for the rectangle tool

        :return: rectangle shape
        :rtype: PyQt6.QtCore.QRectF
        """
        start, stop = self.start_pos, self.stop_pos
        top, bottom = (stop.y(), start.y()) if start.y() > stop.y() else (start.y(), stop.y())
        left, right = (stop.x(), start.x()) if start.x() > stop.x() else (start.x(), stop.x())
        return QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom))

    def drawPoints(self):
        rect = self.getOutline()
        diagonal = rect.bottomRight() - rect.topLeft()
        x = rect.x() + self.x_offsets * diagonal.x()
        y = rect.y() + self.y_offsets * diagonal.y()
        for t1, t2 in zip(x, y):
            self.point_drawn.emit(QtCore.QPointF(t1, t2))


class GraphicsScene(QtWidgets.QGraphicsScene):
    """Provides graphics scene for measurement point selection

    :param scale: scale of the scene
    :type scale: int
    :param parent: parent widget
    :type parent: QtCore.QObject
    """
    def __init__(self, scale=1, parent=None):
        super().__init__(parent)

        self.parent = parent
        self.scale = scale
        self.point_size = 20 * scale
        self.path_pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 0)
        self.highlight_pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 0, QtCore.Qt.PenStyle.DashLine)

        size = 10 * scale
        self.anchor_item = GraphicsAnchorItem(QtCore.QPointF(), size=size)
        self.anchor_item.setZValue(2)

        self.bounds_item = GraphicsBoundsItem(QtCore.QRectF())
        self.bounds_item.setZValue(3)

        self.outline_item = None
        self.transform = QtGui.QTransform()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Delete:
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

    def clear(self):
        """Clears the scene. Clearing the scene deletes all the items
        so the anchor item should be removed before calling clear"""
        self.removeItem(self.anchor_item)
        self.removeItem(self.bounds_item)
        self.outline_item = None
        super().clear()

    def updateOutlineItem(self, geometry):
        """Updates the shape of the outline item

        :param geometry: geometry
        :type geometry: Optional[Union[QtCore.QRectF, QtCore.QLineF]]
        """
        if isinstance(geometry, QtCore.QRectF):
            if self.outline_item is None or not isinstance(self.outline_item, QtWidgets.QGraphicsRectItem):
                self.outline_item = QtWidgets.QGraphicsRectItem()
                self.outline_item.setZValue(4)
                self.addItem(self.outline_item)
                self.outline_item.setPen(self.path_pen)
            self.outline_item.setRect(geometry)
        elif isinstance(geometry, QtCore.QLineF):
            if self.outline_item is None or not isinstance(self.outline_item, QtWidgets.QGraphicsLineItem):
                self.outline_item = QtWidgets.QGraphicsLineItem()
                self.outline_item.setZValue(4)
                self.addItem(self.outline_item)
                self.outline_item.setPen(self.path_pen)
            self.outline_item.setLine(geometry)
        else:
            self.removeItem(self.outline_item)
            self.outline_item = None

    def addPoint(self, point):
        """Adds graphics point item into the scene at specified coordinates

        :param point: point coordinates
        :type point: QtCore.QPoint
        """
        p = GraphicsPointItem(point, size=self.point_size)
        p.default_pen = self.path_pen
        p.highlight_pen = self.highlight_pen
        rot = QtGui.QTransform(self.transform.m11(), self.transform.m12(), self.transform.m21(), self.transform.m22(),
                               0, 0)
        p.setTransform(rot)
        p.setZValue(1.0)  # Ensure point is drawn above cross-section
        self.addItem(p)


class GraphicsPointItem(QtWidgets.QAbstractGraphicsShapeItem):
    """Creates a shape item for points in graphics view. The point is drawn as a cross with
    equal width and height.

    :param point: point coordinates
    :type point: QtCore.QPoint
    :param size: pixel size of point
    :type size: int
    """
    def __init__(self, point, *args, size=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.setPos(point)
        self.fixed = False
        self.default_pen = QtGui.QPen(QtGui.QColor(), 0)
        self.highlight_pen = QtGui.QPen(QtGui.QColor(), 0, QtCore.Qt.PenStyle.DashLine)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.highlighted = False

    def isSelected(self):
        if self.highlighted:
            return True
        return super().isSelected()

    def makeControllable(self, flag):
        """Enables/Disables the ability to select and move the graphics item with the mouse

        :param flag: indicates if the item can be controlled
        :type flag: bool
        """
        if not self.fixed:
            self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, flag)
            self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, flag)

    def boundingRect(self):
        """Calculates the bounding box of the graphics item

        :return: bounding rect
        :rtype: QtCore.QRect
        """
        pen_width = self.default_pen.widthF()
        half_pen_width = pen_width * 0.5

        top = -(self.size * 0.5) - half_pen_width
        new_size = self.size + pen_width
        return QtCore.QRectF(top, top, new_size, new_size)

    def paint(self, painter, _options, _widget):
        painter.setPen(self.default_pen)
        painter.setBrush(self.brush())

        half = self.size * 0.5
        painter.drawLine(QtCore.QLineF(-half, -half, half, half))
        painter.drawLine(QtCore.QLineF(-half, half, half, -half))

        if self.isSelected():
            painter.setPen(self.highlight_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect())


class GraphicsBoundsItem(QtWidgets.QAbstractGraphicsShapeItem):
    """Creates a bounding box shape item. The box is drawn with dashed lines and a crosshair
    in the centre.

    :param rect: bounding box rectangle
    :type rect: QtCore.QRect
    """
    def __init__(self, rect, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rect = rect
        self.default_pen = QtGui.QPen()

    @property
    def rect(self):
        return self._rect

    @rect.setter
    def rect(self, value):
        self._rect = value
        self.update(value)

    def boundingRect(self):
        """Calculates the bounding box of the graphics item

        :return: bounding rect
        :rtype: QtCore.QRect
        """
        return self._rect

    def paint(self, painter, _options, _widget):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        radius = math.ceil(0.01 * max(self._rect.width(), self._rect.height()))
        pen_size = radius / 4
        self.default_pen.setWidthF(pen_size)
        self.default_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(self.default_pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(self._rect)
        painter.drawEllipse(self._rect.center(), radius, radius)

        cx = self._rect.center().x()
        cy = self._rect.center().y()
        radius += pen_size

        self.default_pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        painter.setPen(self.default_pen)
        painter.drawLine(QtCore.QLineF(cx - radius, cy, cx + radius, cy))
        painter.drawLine(QtCore.QLineF(cx, cy - radius, cx, cy + radius))


class GraphicsImageItem(QtWidgets.QAbstractGraphicsShapeItem):
    """Creates a shape item for an image in graphics view. The image is drawn
    inside the given rectangle.

    :param rect: image rectangle
    :type rect: QtCore.QRectF
    :param image: image data
    :type image: numpy.ndarray
    """
    def __init__(self, rect, image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rect = rect
        self.image = self.toQImage(image)

    @staticmethod
    def toQImage(array):
        """Converts numpy array to QImage

        :param array: grayscale image
        :type array: numpy.ndarray
        :return: Qt image
        :rtype: QImage
        """
        height, width = array.shape
        image = QtGui.QImage(array.data, width, height, QtGui.QImage.Format.Format_Indexed8)
        for i in range(256):
            image.setColor(i, QtGui.QColor(i, i, i, 150).rgba())

        return image

    def boundingRect(self):
        """Returns the bounding box of the graphics item

        :return: bounding rect
        :rtype: QtCore.QRect
        """
        return self.rect

    def paint(self, painter, _options, _widget):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())

        painter.drawImage(self.rect, self.image)


class GraphicsAnchorItem(QtWidgets.QAbstractGraphicsShapeItem):
    """Creates a shape item for anchor handles in the graphics view. The anchor point is draw as a rect with two
    lines in the x-axis, and y-axis respectively.

    :param point: anchor point
    :type point: QtCore.QPointF
    :param size: size of anchor item
    :type size: int
    """
    def __init__(self, point, *args, size=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.setPos(point)
        self.default_pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 0)

    def boundingRect(self):
        """Calculates the bounding box of the graphics item

        :return: bounding rect
        :rtype: QtCore.QRect
        """
        return QtCore.QRectF(-self.size / 2, -self.size / 2, self.size, self.size)

    def paint(self, painter, _options, _widget):
        painter.setPen(self.default_pen)
        painter.setBrush(self.brush())

        half = self.size * 0.5
        painter.drawLine(QtCore.QLineF(-half, 0, half, 0))
        painter.drawLine(QtCore.QLineF(0, -half, 0, half))

        rect_size = 0.8 * self.size
        rect = QtCore.QRectF(-rect_size / 2, -rect_size / 2, rect_size, rect_size)
        painter.drawRect(rect)


class Grid(ABC):
    """Base class for form graphics view grid """
    @unique
    class Type(Enum):
        """Types of grid"""
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
    :type x: float
    :param y: y size
    :type y: float
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
        left = rect.left()
        top = rect.top()
        right = rect.right() + 2 * self.x
        bottom = rect.bottom() + 2 * self.y

        left = left - left % self.x
        top = top - top % self.y

        for x in np.arange(left, right, self.x):
            painter.drawLine(QtCore.QLineF(x, top, x, bottom))

        for y in np.arange(top, bottom, self.y):
            painter.drawLine(QtCore.QLineF(left, y, right, y))

    def snap(self, pos):
        pos_x = round(pos.x() / self.x) * self.x
        pos_y = round(pos.y() / self.y) * self.y

        return QtCore.QPointF(pos_x, pos_y)


class PolarGrid(Grid):
    """Provides radial/polar grid for the graphics view

    :param radial: radial distance
    :type radial: float
    :param angular: angle in degrees
    :type angular: float
    """
    def __init__(self, radial=10, angular=45.0):
        self.radial = radial
        self.angular = angular
        self.center = QtCore.QPointF()

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
        center = rect.center()
        radius = (rect.topRight() - center).manhattanLength()
        radius = radius + radius % self.radial
        point = center + QtCore.QPointF(radius, 0)

        for r in np.arange(self.radial, radius, self.radial):
            painter.drawEllipse(center, r, r)

        for angle in np.arange(0.0, 360.0, self.angular):
            transform = QtGui.QTransform.fromTranslate(center.x(), center.y())
            transform.rotate(angle)
            transform.translate(-center.x(), -center.y())
            rotated_point = transform.map(point)
            painter.drawLine(QtCore.QLineF(center.x(), center.y(), rotated_point.x(), rotated_point.y()))

        self.center = center

    @staticmethod
    def toPolar(x, y):
        """Converts point in cartesian coordinate to polar coordinates

        :param x: x coordinate
        :type x: float
        :param y: y coordinate
        :type y: float
        :return: point in polar coordinates (angle in degrees)
        :rtype: Tuple(float, float)
        """
        radius = math.hypot(x, y)
        angle = math.atan2(y, x)
        if y < 0:
            angle += 2 * math.pi

        return radius, math.degrees(angle)

    @staticmethod
    def toCartesian(radius, angle):
        """Converts point in polar coordinate to cartesian coordinate

        :param radius: radius
        :type radius: float
        :param angle: angle in degrees
        :type angle: float
        :return: point in polar coordinates
        :rtype: Tuple(float, float)
        """
        angle = math.radians(angle)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        return x, y

    def snap(self, pos):
        pos = pos - self.center
        radius, angle = self.toPolar(pos.x(), pos.y())
        pos_x = round(radius / self.radial) * self.radial

        max_div = 360 // self.angular
        pos_y = self.angular * (round(angle / self.angular) % (max_div + 1))

        return QtCore.QPointF(*self.toCartesian(pos_x, pos_y)) + self.center
