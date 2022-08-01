from abc import ABC, abstractmethod
from enum import Enum, unique
import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.math import Vector3, clamp, angle_axis_btw_vectors


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
        self.object_snap_tool = ObjectSnap(self, self.grid)
        self.object_snap_tool.object_moved.connect(self.translateSceneItems)
        self.zoom_factor = 1.2
        self.anchor = QtCore.QRectF()
        self.scene_transform = QtGui.QTransform()
        self.setMouseTracking(True)

        self.setViewportUpdateMode(self.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.updateViewMode()

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

    @property
    def object_anchor(self):
        """Returns object anchor for snap object to grid operation

        :return: anchor point
        :rtype: QtCore.QPointF
        """
        return self.object_snap_tool.anchor

    @object_anchor.setter
    def object_anchor(self, value):
        self.object_snap_tool.setAnchor(value, self.scene_transform)

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

        text_rect = QtCore.QRect(0, 0, 300, 280)
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
        transform = QtGui.QTransform().translate(offset.x(), offset.y())
        transform.rotateRadians(angle)
        transform.translate(-offset.x(), -offset.y())
        self.scene_transform *= transform
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
        transform = QtGui.QTransform().translate(dx, dy)
        self.scene_transform *= transform
        gr.setTransform(transform)
        self.scene().destroyItemGroup(gr)
        self.scene().update()

    def resetTransform(self):
        """Resets scene transform to identity"""
        self.scene_transform.reset()
        super().resetTransform()

    def reset(self):
        """Resets the camera of the graphics view"""
        if not self.scene() or not self.scene().items():
            return

        gr = self.scene().createItemGroup(self.scene().items())
        gr.setTransform(self.scene_transform.inverted()[0])
        self.object_snap_tool.reset()
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

        factor = 1.0 / self.zoom_factor
        self.scale(factor, factor)
        anchor = self.scene_transform.mapRect(self.anchor)
        self.centerOn(anchor.center())

    def leaveEvent(self, _event):
        self.mouse_moved.emit(QtCore.QPoint(-1, -1))

    def mousePressEvent(self, event):
        is_rotating = event.button() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.NoModifier
        is_panning = ((event.button() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.ControlModifier)
                      or (event.button() == QtCore.Qt.MiddleButton and event.modifiers() == QtCore.Qt.NoModifier))

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
        default_keybind = (event.buttons() == QtCore.Qt.MiddleButton
                           and event.modifiers() in [QtCore.Qt.NoModifier, QtCore.Qt.ShiftModifier])
        alt_keybind = (event.buttons() == QtCore.Qt.RightButton and event.modifiers()
                       in [QtCore.Qt.ControlModifier, QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier])

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

            strength = 0.4  # Should find a better way of selecting snap strength
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

            strength = 50  # Should find a better way of selecting snap strength
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

        if event.type() == QtCore.QEvent.MouseButtonPress and self.isPanning(event):
            self.last_pos = self.graphics_view.mapToScene(event.pos())
            self.remaining_shift = (0, 0)
            return True
        if event.type() == QtCore.QEvent.MouseMove and self.isPanning(event):
            cur_pos = self.graphics_view.mapToScene(event.pos())
            self.snapAnchor(self.last_pos, cur_pos, event.modifiers() & QtCore.Qt.ShiftModifier)
            self.last_pos = cur_pos
            return True

        return False


class GraphicsScene(QtWidgets.QGraphicsScene):
    """Provides graphics scene for measurement point selection

    :param scale: scale of the scene
    :type scale: int
    :param parent: parent widget
    :type parent: QtCore.QObject
    """
    @unique
    class Mode(Enum):
        """Draw mode for graphics scene"""
        Select = 1
        Draw_point = 2
        Draw_line = 3
        Draw_area = 4

    def __init__(self, scale=1, parent=None):
        super().__init__(parent)

        self.scale = scale
        self.point_size = 20 * scale
        self.path_pen = QtGui.QPen(QtCore.Qt.black, 0)

        size = 10 * scale
        self.anchor_item = GraphicsAnchorItem(QtCore.QPointF(), size=size)
        self.anchor_item.setZValue(2)
        self.anchor_item.setPen(QtGui.QPen(QtGui.QColor(0, 0, 200), 0))

        self.item_to_draw = None
        self.current_obj = None
        self.mode = GraphicsScene.Mode.Select

        self.setLineToolSize(2)
        self.setAreaToolSize(2, 2)
        self.start_pos = QtCore.QPointF()

    @property
    def view(self):
        """Gets graphics view associated with scene

        :returns: graphics view widget
        :rtype: Union[QtWidgets.QGraphicsView, None]
        """
        view = self.views()
        if view:
            return view[0]
        else:
            return None

    @property
    def mode(self):
        """Gets and sets scene's mode

        :returns: scene mode
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

    def addAnchor(self, anchor):
        """Adds anchor item to the scene at specified coordinates

        :param anchor: anchor point
        :type anchor: QtCore.QPointF
        """
        if self.anchor_item not in self.items():
            anchor = self.view.scene_transform.map(anchor)
            self.anchor_item.setPos(anchor)
            self.addItem(self.anchor_item)
        else:
            anchor = self.view.scene_transform.map(anchor)
            self.anchor_item.setPos(anchor)

    def clear(self):
        """Clears the scene. Clearing the scene deletes all the items
        so the anchor item should be removed before calling clear"""
        self.removeItem(self.anchor_item)
        super().clear()

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
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)

    def makeControllable(self, flag):
        """Enables/Disables the ability to select and move the graphics item with the mouse

        :param flag: indicates if the item can be controlled
        :type flag: bool
        """
        if not self.fixed:
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, flag)
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, flag)

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
        pen = self.pen()
        painter.setPen(pen)
        painter.setBrush(self.brush())

        half = self.size * 0.5
        painter.drawLine(QtCore.QLineF(-half, -half, half, half))
        painter.drawLine(QtCore.QLineF(-half, half, half, -half))

        if self.isSelected():
            painter.save()
            pen = QtGui.QPen(QtCore.Qt.black, 0, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(self.boundingRect())
            painter.restore()


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
        image = QtGui.QImage(array.data, width, height, QtGui.QImage.Format_Indexed8)
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
        pen = self.pen()
        painter.setPen(pen)
        painter.setBrush(self.brush())

        painter.drawImage(self.rect, self.image)


class GraphicsAnchorItem(QtWidgets.QAbstractGraphicsShapeItem):
    """Creates a shape item for points in graphics view. The anchor point is draw as a rect with two
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

    def boundingRect(self):
        """Calculates the bounding box of the graphics item

        :return: bounding rect
        :rtype: QtCore.QRect
        """
        return QtCore.QRectF(-self.size / 2, -self.size / 2, self.size, self.size)

    def paint(self, painter, _options, _widget):
        pen = self.pen()
        painter.setPen(pen)
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
        left = rect.left()
        top = rect.top()
        right = rect.right() + 2 * self.x
        bottom = rect.bottom() + 2 * self.y

        left = left - left % self.x
        top = top - top % self.y

        x_offsets = np.arange(left, right, self.x)
        y_offsets = np.arange(top, bottom, self.y)

        for x in x_offsets:
            painter.drawLine(QtCore.QLineF(x, top, x, bottom))

        for y in y_offsets:
            painter.drawLine(QtCore.QLineF(left, y, right, y))

    def snap(self, pos):
        pos_x = round(pos.x() / self.x) * self.x
        pos_y = round(pos.y() / self.y) * self.y

        return QtCore.QPointF(pos_x, pos_y)


class PolarGrid(Grid):
    """Provides radial/polar grid for the graphics view

    :param radial: radial distance
    :type radial: int
    :param angular: angle
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

        radial_offsets = np.arange(self.radial, radius, self.radial)
        angular_offsets = np.arange(0.0, 360.0, self.angular)

        for r in radial_offsets:
            painter.drawEllipse(center, r, r)

        for angle in angular_offsets:
            transform = QtGui.QTransform().translate(center.x(), center.y())
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
        radius = math.sqrt(x * x + y * y)
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
