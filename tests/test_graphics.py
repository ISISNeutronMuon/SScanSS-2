import unittest
import unittest.mock as mock
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QEvent, Qt, QPoint
from PyQt6.QtGui import QTransform, QMouseEvent, QWheelEvent
from sscanss.app.widgets import (BoxGrid, PolarGrid, ObjectSnap, PointTool, RectangleTool, LineTool, GraphicsView,
                                 GraphicsViewInteractor)
from sscanss.core.scene import SceneInteractor, SceneManager, Node
from sscanss.core.util import Attributes
from tests.helpers import TestSignal


class TestGridClass(unittest.TestCase):
    def testBoxGrid(self):
        grid = BoxGrid(10, 10)
        self.assertEqual(grid.x, 10)
        self.assertEqual(grid.y, 10)
        self.assertEqual(grid.size, (10, 10))
        self.assertEqual(grid.type, grid.Type.Box)

        grid.size = (20, 11.5)
        self.assertEqual(grid.size, (20, 11.5))

        point = grid.snap(QPointF(30, 10.2))
        self.assertAlmostEqual(point.x(), 40, 5)
        self.assertAlmostEqual(point.y(), 11.5, 5)

        point = grid.snap(QPointF(-21.25, -4))
        self.assertAlmostEqual(point.x(), -20, 5)
        self.assertAlmostEqual(point.y(), 0, 5)

        rect = QRectF(0, 0, 20, 30)
        painter_mock = mock.Mock()
        painter_mock.drawLine.assert_not_called()
        grid.render(painter_mock, rect)
        painter_mock.drawLine.assert_called()

    def testPolarGrid(self):
        grid = PolarGrid(10, 30.0)
        self.assertEqual(grid.radial, 10)
        self.assertEqual(grid.angular, 30.0)
        self.assertEqual(grid.size, (10, 30.0))
        self.assertEqual(grid.type, grid.Type.Polar)
        self.assertEqual(grid.center.x(), 0)
        self.assertEqual(grid.center.y(), 0)

        grid.size = (20, 46.5)
        self.assertEqual(grid.size, (20, 46.5))
        self.assertEqual(grid.center.x(), 0)  # center should not change
        self.assertEqual(grid.center.y(), 0)

        point = grid.snap(QPointF(42, 1.2))
        self.assertAlmostEqual(point.x(), 40, 5)
        self.assertAlmostEqual(point.y(), 0, 5)

        point = grid.snap(QPointF(-21.25, -34.0))
        self.assertAlmostEqual(point.x(), -24.350457, 5)
        self.assertAlmostEqual(point.y(), -31.734133, 5)

        point = grid.snap(QPointF(35, 20))
        self.assertAlmostEqual(point.x(), 27.53418, 5)
        self.assertAlmostEqual(point.y(), 29.01497, 5)

        polar_point = grid.toPolar(0, -35)
        self.assertAlmostEqual(polar_point[0], 35, 5)
        self.assertAlmostEqual(polar_point[1], 270, 5)
        cart_point = grid.toCartesian(polar_point[0], polar_point[1])
        self.assertAlmostEqual(cart_point[0], 0, 3)
        self.assertAlmostEqual(cart_point[1], -35, 3)

        cart_point = grid.toCartesian(10, 135)
        self.assertAlmostEqual(cart_point[0], -7.071067, 5)
        self.assertAlmostEqual(cart_point[1], 7.071067, 5)
        polar_point = grid.toPolar(cart_point[0], cart_point[1])
        self.assertAlmostEqual(polar_point[0], 10, 5)
        self.assertAlmostEqual(polar_point[1], 135, 5)

        rect = QRectF(0, 0, 20, 30)
        painter_mock = mock.Mock()
        painter_mock.drawLine.assert_not_called()
        painter_mock.drawEllipse.assert_not_called()
        grid.render(painter_mock, rect)  # center should change to screen rect
        self.assertAlmostEqual(grid.center.x(), 10.0, 5)
        self.assertAlmostEqual(grid.center.y(), 15.0, 5)
        painter_mock.drawLine.assert_called()
        painter_mock.drawEllipse.assert_called()


class TestObjectSnapClass(unittest.TestCase):
    def setUp(self):
        self.graphics_view = mock.Mock()

    def testAnchor(self):
        grid = BoxGrid()
        object_snap = ObjectSnap(self.graphics_view, grid)
        object_snap.initialized = True

        anchor = QPointF(1, 1)
        transform = QTransform()
        object_snap.setAnchor(anchor, transform)
        self.assertEqual(object_snap.anchor.x(), 1.0)
        self.assertEqual(object_snap.anchor.y(), 1.0)
        self.assertEqual(object_snap.anchor_offset.x(), 0.0)
        self.assertEqual(object_snap.anchor_offset.y(), 0.0)
        self.assertFalse(object_snap.initialized)

        transform = transform.rotate(90).translate(0, -1)
        object_snap.initialized = True
        object_snap.setAnchor(anchor, transform)
        self.assertEqual(object_snap.anchor.x(), 1.0)
        self.assertEqual(object_snap.anchor.y(), 1.0)
        self.assertEqual(object_snap.anchor_offset.x(), -1.0)
        self.assertEqual(object_snap.anchor_offset.y(), 0.0)
        self.assertFalse(object_snap.initialized)

        transform = QTransform().rotate(90)
        object_snap.initialized = True
        object_snap.updateOffset(transform)
        self.assertEqual(object_snap.anchor.x(), 1.0)
        self.assertEqual(object_snap.anchor.y(), 1.0)
        self.assertEqual(object_snap.anchor_offset.x(), -2.0)
        self.assertEqual(object_snap.anchor_offset.y(), -1.0)
        self.assertFalse(object_snap.initialized)

        object_snap.initialized = True
        object_snap.reset()
        self.assertEqual(object_snap.anchor.x(), 1.0)
        self.assertEqual(object_snap.anchor.y(), 1.0)
        self.assertEqual(object_snap.anchor_offset.x(), 0.0)
        self.assertEqual(object_snap.anchor_offset.y(), 0.0)
        self.assertFalse(object_snap.initialized)

    def testBoxSnap(self):
        grid = BoxGrid()
        object_snap = ObjectSnap(self.graphics_view, grid)
        object_snap.setAnchor(QPointF(1, 1), QTransform())
        self.assertFalse(object_snap.initialized)

        dx, dy = object_snap.snapAnchorToBoxGrid(QPointF(0, 0), QPointF(0, 200))
        self.assertTrue(object_snap.initialized)
        self.assertAlmostEqual(dx, 0.0, 5)
        self.assertAlmostEqual(dy, 10.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), -1.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 9.0, 5)

        dx, dy = object_snap.snapAnchorToBoxGrid(QPointF(0, 0), QPointF(200, 0))
        self.assertTrue(object_snap.initialized)
        self.assertAlmostEqual(dx, 10.0, 5)
        self.assertAlmostEqual(dy, 0.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 9.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 9.0, 5)

        dx, dy = object_snap.snapAnchorToBoxGrid(QPointF(0, 0), QPointF(400, -200))
        self.assertTrue(object_snap.initialized)
        self.assertAlmostEqual(dx, 20.0, 5)
        self.assertAlmostEqual(dy, -10.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 29.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -1.0, 5)

        object_snap.updateOffset(QTransform().rotate(90))
        dx, dy = object_snap.snapAnchorToBoxGrid(QPointF(0, 0), QPointF(-200, -200))
        self.assertTrue(object_snap.initialized)
        self.assertAlmostEqual(dx, -10.0, 5)
        self.assertAlmostEqual(dy, -10.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), -11.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 19.0, 5)

    def testPolarSnap(self):
        grid = PolarGrid(10, 45)
        object_snap = ObjectSnap(self.graphics_view, grid)
        object_snap.setAnchor(QPointF(-11, 1), QTransform())
        self.assertFalse(object_snap.initialized)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(0, 200), False)
        self.assertTrue(object_snap.initialized)
        self.assertAlmostEqual(dx, -10.0, 5)
        self.assertAlmostEqual(dy, 0.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), -9.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -1.0, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(400, 0), False)
        self.assertAlmostEqual(dx, 20.0, 5)
        self.assertAlmostEqual(dy, 0.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 11.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -1.0, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(-282.88, -282.88), False)
        self.assertAlmostEqual(dx, -14.1421356, 5)
        self.assertAlmostEqual(dy, -14.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), -3.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -15.1421356, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(565.76, 565.76), False)
        self.assertAlmostEqual(dx, 28.284271, 5)
        self.assertAlmostEqual(dy, 28.284271, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 25.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 13.1421356, 5)

        object_snap.remaining_shift = (0, 0)
        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(0, 100), True)
        self.assertEqual(dx, 0)
        self.assertEqual(dy, 0)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 25.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 13.1421356, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(1, 0), QPointF(0.764842, -0.644218), True)
        self.assertAlmostEqual(dx, 5.857864, 5)
        self.assertAlmostEqual(dy, -14.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 31, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -1, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(1, 0), QPointF(0.070737, 0.997495), True)
        self.assertAlmostEqual(dx, -20, 5)
        self.assertAlmostEqual(dy, 20, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 11, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 19, 5)

    def testEvent(self):
        test_mock = mock.Mock()

        grid = BoxGrid()
        object_snap = ObjectSnap(self.graphics_view, grid)
        object_snap.object_moved = TestSignal()
        object_snap.object_moved.connect(test_mock)

        self.assertFalse(object_snap.enabled)
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.assertFalse(object_snap.eventFilter(self.graphics_view, event))

        object_snap.enabled = True
        self.graphics_view.mapToScene.return_value = QPointF(100, 0)
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 1), Qt.MouseButton.MiddleButton,
                            Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        test_mock.assert_not_called()

        self.graphics_view.mapToScene.return_value = QPointF(-100, 0)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(1, 1), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        self.assertAlmostEqual(test_mock.call_args[0][0], -10, 5)
        self.assertAlmostEqual(test_mock.call_args[0][1], 0, 5)

        object_snap.grid = PolarGrid()
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 1), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton,
                            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        self.assertEqual(test_mock.call_count, 2)

        self.graphics_view.mapToScene.return_value = QPointF(-100, 600)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(1, 1), Qt.MouseButton.MiddleButton,
                            Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.ShiftModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        self.assertAlmostEqual(test_mock.call_args[0][0], 10, 5)
        self.assertAlmostEqual(test_mock.call_args[0][1], 10, 5)

        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 1), Qt.MouseButton.MiddleButton,
                            Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.AltModifier)
        self.assertFalse(object_snap.eventFilter(self.graphics_view, event))


class TestGraphicsViewInteractorClass(unittest.TestCase):
    def setUp(self):
        self.graphics_view = mock.Mock()

    def testInteract(self):
        interactor = GraphicsViewInteractor(self.graphics_view)

        interactor.rotate(QPointF(1, 1), QPointF())
        self.graphics_view.rotateSceneItems.not_assert_called()
        interactor.last_pos = QPointF(1, 0)
        interactor.rotate(QPointF(1, 1), QPointF())
        self.graphics_view.rotateSceneItems.assert_called()
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][0], 0.7853981, 5)
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][1].x(), 0.0, 5)
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][1].y(), 0.0, 5)

        interactor.rotate(QPointF(0, 2), QPointF(2, -1))
        self.assertEqual(self.graphics_view.rotateSceneItems.call_count, 2)
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][0], -0.1973955, 5)
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][1].x(), 2.0, 5)
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][1].y(), -1.0, 5)

        interactor.pan(QPointF(1, 1))
        self.graphics_view.translateSceneItems.assert_called()
        self.assertAlmostEqual(self.graphics_view.translateSceneItems.call_args[0][0], 0.0, 5)
        self.assertAlmostEqual(self.graphics_view.translateSceneItems.call_args[0][1], 1.0, 5)

        interactor.last_pos = QPointF(13.1, 12.5)
        interactor.pan(QPointF(2, -1))
        self.assertEqual(self.graphics_view.translateSceneItems.call_count, 2)
        self.assertAlmostEqual(self.graphics_view.translateSceneItems.call_args[0][0], -11.1, 5)
        self.assertAlmostEqual(self.graphics_view.translateSceneItems.call_args[0][1], -13.5, 5)

        interactor.zoom(QPointF())
        self.graphics_view.zoomIn.not_assert_called()
        self.graphics_view.zoomOut.not_assert_called()

        interactor.zoom(QPointF(0, -8))
        self.graphics_view.zoomIn.not_assert_called()
        self.graphics_view.zoomOut.assert_called()
        self.graphics_view.zoomOut.reset_mock()

        interactor.zoom(QPointF(0, 8))
        self.graphics_view.zoomIn.assert_called()
        self.graphics_view.zoomOut.not_assert_called()

        self.graphics_view.cursor_state = GraphicsView.MouseAction.Rotate
        interactor.rotate(QPointF(1, 1), QPointF())
        self.assertEqual(self.graphics_view.rotateSceneItems.call_count, 3)
        self.graphics_view.cursor_state = GraphicsView.MouseAction.Pan
        interactor.pan(QPointF())
        self.assertEqual(self.graphics_view.translateSceneItems.call_count, 3)

    def testEvent(self):
        test_mock = mock.Mock()
        self.graphics_view.viewport_anchor_in_scene = QPointF()

        interactor = GraphicsViewInteractor(self.graphics_view)
        interactor.mouse_moved = TestSignal()
        interactor.mouse_moved.connect(test_mock)

        self.graphics_view.mapToScene.return_value = QPointF(1, 0)
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 0), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.assertEqual(self.graphics_view.cursor_state, GraphicsView.MouseAction.Rotate)

        self.graphics_view.mapToScene.return_value = QPointF(0, 1)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(0, 1), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.assertEqual(self.graphics_view.cursor_state, GraphicsView.MouseAction.Rotate)
        self.assertEqual(test_mock.call_count, 1)

        event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.assertEqual(self.graphics_view.cursor_state, GraphicsView.MouseAction.None_)
        self.graphics_view.rotateSceneItems.assert_called()
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][0], 1.5707963, 5)
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][1].x(), 0.0, 5)
        self.assertAlmostEqual(self.graphics_view.rotateSceneItems.call_args[0][1].y(), 0.0, 5)

        self.graphics_view.mapToScene.return_value = QPointF(1, 0)
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 0), Qt.MouseButton.MiddleButton,
                            Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.assertEqual(self.graphics_view.cursor_state, GraphicsView.MouseAction.Pan)

        self.graphics_view.mapToScene.return_value = QPointF(0, 1)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(0, 1), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.assertEqual(self.graphics_view.cursor_state, GraphicsView.MouseAction.Pan)
        self.assertEqual(test_mock.call_count, 2)

        event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.assertEqual(self.graphics_view.cursor_state, GraphicsView.MouseAction.None_)
        self.graphics_view.translateSceneItems.assert_called()
        self.assertAlmostEqual(self.graphics_view.translateSceneItems.call_args[0][0], -1.0, 5)
        self.assertAlmostEqual(self.graphics_view.translateSceneItems.call_args[0][1], 1.0, 5)

        event = QWheelEvent(QPointF(), QPointF(), QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.graphics_view.zoomIn.assert_called()
        self.graphics_view.zoomOut.not_assert_called()
        self.graphics_view.zoomOut.reset_mock()

        event = QWheelEvent(QPointF(), QPointF(), QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.graphics_view.zoomIn.not_assert_called()
        self.graphics_view.zoomOut.assert_called()

        event = QEvent(QEvent.Type.Leave)
        self.assertFalse(interactor.eventFilter(self.graphics_view, event))
        self.assertEqual(test_mock.call_count, 3)
        self.assertEqual(test_mock.call_args[0][0].x(), -1)
        self.assertEqual(test_mock.call_args[0][0].y(), -1)


class TestDrawToolClass(unittest.TestCase):
    def setUp(self):
        self.graphics_view = mock.Mock()

    def testPointTool(self):
        point_tool = PointTool(self.graphics_view)
        self.assertEqual(point_tool.mode, GraphicsView.DrawMode.Point)
        self.assertIsNone(point_tool.getOutline())

        test_mock = mock.Mock()
        point_tool.point_drawn = TestSignal()
        point_tool.point_drawn.connect(test_mock)

        self.assertEqual(test_mock.call_count, 0)
        point_tool.stop_pos = QPointF(-1, -2)
        point_tool.drawPoints()
        point_tool.stop_pos = QPointF(2, 1)
        point_tool.drawPoints()
        self.assertEqual(test_mock.call_count, 2)

        expected = [(-1, -2), (2, 1)]
        for i, arg in enumerate(test_mock.call_args_list):
            point = arg[0][0]
            self.assertAlmostEqual(point.x(), expected[i][0], 5)
            self.assertAlmostEqual(point.y(), expected[i][1], 5)

    def testLineTool(self):
        line_tool = LineTool(self.graphics_view)

        self.assertEqual(len(line_tool.offsets), 2)
        self.assertAlmostEqual(line_tool.offsets[0], 0, 5)
        self.assertAlmostEqual(line_tool.offsets[1], 1, 5)

        self.assertEqual(line_tool.mode, GraphicsView.DrawMode.Line)

        line_tool.start_pos = QPointF(-1, -2)
        line_tool.stop_pos = QPointF(2, 1)
        outline = line_tool.getOutline()
        self.assertEqual(outline.x1(), -1)
        self.assertEqual(outline.x2(), 2)
        self.assertEqual(outline.y1(), -2)
        self.assertEqual(outline.y2(), 1)

        test_mock = mock.Mock()
        line_tool.point_drawn = TestSignal()
        line_tool.point_drawn.connect(test_mock)

        self.assertEqual(test_mock.call_count, 0)
        line_tool.drawPoints()
        self.assertEqual(test_mock.call_count, 2)
        expected = [(-1, -2), (2, 1)]
        for i, arg in enumerate(test_mock.call_args_list):
            point = arg[0][0]
            self.assertAlmostEqual(point.x(), expected[i][0], 5)
            self.assertAlmostEqual(point.y(), expected[i][1], 5)

        line_tool = LineTool(self.graphics_view, 4)
        line_tool.point_drawn = TestSignal()
        line_tool.point_drawn.connect(test_mock)
        self.assertEqual(len(line_tool.offsets), 4)
        self.assertAlmostEqual(line_tool.offsets[0], 0, 5)
        self.assertAlmostEqual(line_tool.offsets[1], 0.333333, 5)
        self.assertAlmostEqual(line_tool.offsets[2], 0.666666, 5)
        self.assertAlmostEqual(line_tool.offsets[3], 1, 5)
        line_tool.start_pos = QPointF(-1, -1)
        line_tool.stop_pos = QPointF(1, 1)

        test_mock.reset_mock()
        self.assertEqual(test_mock.call_count, 0)
        line_tool.drawPoints()
        self.assertEqual(test_mock.call_count, 4)
        expected = [(-1, -1), (-0.333333, -0.333333), (0.333333, 0.333333), (1, 1)]
        for i, arg in enumerate(test_mock.call_args_list):
            point = arg[0][0]
            self.assertAlmostEqual(point.x(), expected[i][0], 5)
            self.assertAlmostEqual(point.y(), expected[i][1], 5)

    def testRectangleTool(self):
        rect_tool = RectangleTool(self.graphics_view)

        self.assertEqual(len(rect_tool.x_offsets), 4)
        self.assertEqual(len(rect_tool.y_offsets), 4)
        self.assertAlmostEqual(rect_tool.x_offsets[0], 0, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[1], 1, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[2], 0, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[3], 1, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[0], 0, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[1], 0, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[2], 1, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[3], 1, 5)
        self.assertEqual(rect_tool.mode, GraphicsView.DrawMode.Rectangle)

        rect_tool.start_pos = QPointF(-1, -2)
        rect_tool.stop_pos = QPointF(2, 1)
        outline = rect_tool.getOutline()
        self.assertEqual(outline.topLeft().x(), -1)
        self.assertEqual(outline.topLeft().y(), -2)
        self.assertEqual(outline.bottomRight().x(), 2)
        self.assertEqual(outline.bottomRight().y(), 1)

        test_mock = mock.Mock()
        rect_tool.point_drawn = TestSignal()
        rect_tool.point_drawn.connect(test_mock)

        self.assertEqual(test_mock.call_count, 0)
        rect_tool.drawPoints()
        self.assertEqual(test_mock.call_count, 4)
        expected = [(-1, -2), (2, -2), (-1, 1), (2, 1)]
        for i, arg in enumerate(test_mock.call_args_list):
            point = arg[0][0]
            self.assertAlmostEqual(point.x(), expected[i][0], 5)
            self.assertAlmostEqual(point.y(), expected[i][1], 5)

        rect_tool = RectangleTool(self.graphics_view, 3, 3)
        rect_tool.point_drawn = TestSignal()
        rect_tool.point_drawn.connect(test_mock)
        self.assertEqual(len(rect_tool.x_offsets), 9)
        self.assertEqual(len(rect_tool.y_offsets), 9)
        self.assertAlmostEqual(rect_tool.x_offsets[0], 0, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[1], 0.5, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[2], 1, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[3], 0, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[4], 0.5, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[5], 1, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[6], 0, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[7], 0.5, 5)
        self.assertAlmostEqual(rect_tool.x_offsets[8], 1, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[0], 0, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[1], 0, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[2], 0, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[3], 0.5, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[4], 0.5, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[5], 0.5, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[6], 1, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[7], 1, 5)
        self.assertAlmostEqual(rect_tool.y_offsets[8], 1, 5)
        rect_tool.start_pos = QPointF(-1, -1)
        rect_tool.stop_pos = QPointF(1, 1)

        test_mock.reset_mock()
        self.assertEqual(test_mock.call_count, 0)
        rect_tool.drawPoints()
        self.assertEqual(test_mock.call_count, 9)
        expected = [(-1, -1), (0., -1), (1, -1), (-1, 0), (0., 0), (1, 0), (-1, 1), (0., 1), (1, 1)]
        for i, arg in enumerate(test_mock.call_args_list):
            point = arg[0][0]
            self.assertAlmostEqual(point.x(), expected[i][0], 5)
            self.assertAlmostEqual(point.y(), expected[i][1], 5)


class TestSceneInteractorClass(unittest.TestCase):
    def setUp(self):
        self.renderer = mock.Mock()

    def testInteract(self):
        interactor = SceneInteractor(self.renderer)

        camera_mock = mock.Mock()
        self.renderer.scene.camera = camera_mock

        interactor.last_pos = QPointF()
        self.renderer.update.assert_not_called()
        interactor.rotate(QPointF(1, 1), (100, 100))
        camera_mock.rotate.assert_called()
        self.renderer.update.assert_called()
        result = camera_mock.rotate.call_args[0]
        self.assertAlmostEqual(result[0][0], 0.0, 5)
        self.assertAlmostEqual(result[0][1], 0.0, 5)
        self.assertAlmostEqual(result[1][0], 0.02, 5)
        self.assertAlmostEqual(result[1][1], 0.02, 5)

        interactor.last_pos = QPointF(10, 5)
        interactor.rotate(QPointF(5, 1), (10, 100))
        result = camera_mock.rotate.call_args[0]
        self.assertAlmostEqual(result[0][0], 2.0, 5)
        self.assertAlmostEqual(result[0][1], 0.1, 5)
        self.assertAlmostEqual(result[1][0], 1.0, 5)
        self.assertAlmostEqual(result[1][1], 0.02, 5)

        self.renderer.reset_mock()
        interactor.last_pos = QPointF()
        self.renderer.update.assert_not_called()
        camera_mock.pan.assert_not_called()
        interactor.pan(QPointF(1, 1))
        camera_mock.pan.assert_called()
        self.renderer.update.assert_called()
        self.assertAlmostEqual(camera_mock.pan.call_args[0][0], -0.001, 5)
        self.assertAlmostEqual(camera_mock.pan.call_args[0][1], -0.001, 5)

        interactor.last_pos = QPointF(100, 0)
        interactor.pan(QPointF(12, 1.1))
        self.assertAlmostEqual(camera_mock.pan.call_args[0][0], 0.088, 5)
        self.assertAlmostEqual(camera_mock.pan.call_args[0][1], -0.0011, 5)

        self.renderer.reset_mock()
        self.renderer.update.assert_not_called()
        camera_mock.zoom.assert_not_called()
        interactor.zoom(QPointF())
        camera_mock.zoom.assert_called()
        self.renderer.update.assert_called()
        self.assertAlmostEqual(camera_mock.zoom.call_args[0][0], 0.0, 5)

        interactor.zoom(QPointF(0, 120))
        self.assertAlmostEqual(camera_mock.zoom.call_args[0][0], 0.05, 5)

        interactor.zoom(QPointF(0, -120))
        self.assertAlmostEqual(camera_mock.zoom.call_args[0][0], -0.05, 5)

    def testPicking(self):
        interactor = SceneInteractor(self.renderer)

        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 0), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.assertFalse(interactor.isPicking(event))
        interactor.picking = True
        self.assertTrue(interactor.isPicking(event))
        self.assertEqual(self.renderer.setCursor.call_args[0][0], Qt.CursorShape.CrossCursor)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(1, 0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.ShiftModifier)
        self.assertFalse(interactor.isPicking(event))
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(1, 0), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        self.assertFalse(interactor.isPicking(event))
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(1, 0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        self.assertTrue(interactor.isPicking(event))
        interactor.picking = False
        self.assertFalse(interactor.isPicking(event))
        self.assertEqual(self.renderer.setCursor.call_args[0][0], Qt.CursorShape.ArrowCursor)

        test_mock = mock.Mock()
        interactor.ray_picked = TestSignal()
        interactor.ray_picked.connect(test_mock)

        unproject_mock = mock.Mock()
        self.renderer.unproject = unproject_mock
        unproject_mock.side_effect = [([1, 2, 3], False), ([4, 5, 6], False)]
        unproject_mock.assert_not_called()
        interactor.createPickRay(QPointF(2.1, -3.4))

        first_call = unproject_mock.call_args_list[0]
        second_call = unproject_mock.call_args_list[1]
        self.assertAlmostEqual(first_call[0][0], 2.1, 5)
        self.assertAlmostEqual(first_call[0][1], -3.4, 5)
        self.assertAlmostEqual(first_call[0][2], 0.0, 5)
        self.assertAlmostEqual(second_call[0][0], 2.1, 5)
        self.assertAlmostEqual(second_call[0][1], -3.4, 5)
        self.assertAlmostEqual(second_call[0][2], 1.0, 5)
        test_mock.assert_not_called()
        unproject_mock.side_effect = [([1, 2, 3], False), ([4, 5, 6], True)]
        interactor.createPickRay(QPointF(2.1, -3.4))
        test_mock.assert_not_called()
        unproject_mock.side_effect = [([1, 2, 3], True), ([4, 5, 6], False)]
        interactor.createPickRay(QPointF(2.1, -3.4))
        test_mock.assert_not_called()
        unproject_mock.side_effect = [([1, 2, 3], True), ([4, 5, 6], True)]
        interactor.createPickRay(QPointF(2.1, -3.4))
        test_mock.assert_called()
        self.assertListEqual(test_mock.call_args[0][0], [1, 2, 3])
        self.assertListEqual(test_mock.call_args[0][1], [4, 5, 6])

    def testEvent(self):
        interactor = SceneInteractor(self.renderer)
        camera_mock = mock.Mock()
        self.renderer.scene.camera = camera_mock
        self.renderer.width.return_value = 100
        self.renderer.height.return_value = 100

        self.assertEqual(interactor.last_pos.x(), 0)
        self.assertEqual(interactor.last_pos.y(), 0)
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 10), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.ShiftModifier)
        self.assertFalse(interactor.eventFilter(self.renderer, event))

        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 10), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        self.assertTrue(interactor.eventFilter(self.renderer, event))
        self.assertEqual(interactor.last_pos.x(), 1)
        self.assertEqual(interactor.last_pos.y(), 10)
        camera_mock.pan.assert_not_called()
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(0, 1), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.ShiftModifier)
        self.assertFalse(interactor.eventFilter(self.renderer, event))
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(0, 1), Qt.MouseButton.RightButton,
                            Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        self.assertTrue(interactor.eventFilter(self.renderer, event))
        self.assertEqual(interactor.last_pos.x(), 0)
        self.assertEqual(interactor.last_pos.y(), 1)
        camera_mock.pan.assert_called()

        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(2, 2), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.assertTrue(interactor.eventFilter(self.renderer, event))
        camera_mock.rotate.assert_not_called()
        self.assertEqual(interactor.last_pos.x(), 2)
        self.assertEqual(interactor.last_pos.y(), 2)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(3, 1), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        self.assertTrue(interactor.eventFilter(self.renderer, event))
        self.assertEqual(interactor.last_pos.x(), 3)
        self.assertEqual(interactor.last_pos.y(), 1)
        camera_mock.rotate.assert_called()
        event = QWheelEvent(QPointF(), QPointF(), QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.ScrollUpdate, False)
        self.assertFalse(interactor.eventFilter(self.renderer, event))
        camera_mock.zoom.assert_not_called()
        event = QWheelEvent(QPointF(), QPointF(), QPoint(), QPoint(0, 120), Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
        self.assertFalse(interactor.eventFilter(self.renderer, event))
        camera_mock.zoom.assert_not_called()
        event = QWheelEvent(QPointF(), QPointF(), QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
        self.assertTrue(interactor.eventFilter(self.renderer, event))
        camera_mock.zoom.assert_called()

        camera_mock.reset_mock()
        self.renderer.unproject.return_value = ([1, 2, 3], False)
        interactor.picking = True
        event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(2, 2), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.renderer.unproject.assert_not_called()
        self.assertTrue(interactor.eventFilter(self.renderer, event))
        camera_mock.rotate.assert_not_called()
        self.renderer.unproject.assert_called()


class TestSceneManagerClass(unittest.TestCase):
    def setUp(self):
        self.model = mock.Mock()
        self.renderer = mock.Mock()

    def testCreation(self):
        manager = SceneManager(self.model, self.renderer, True)
        self.assertIs(manager.sample_scene, manager.active_scene)
        manager = SceneManager(self.model, self.renderer, False)
        self.assertIs(manager.instrument_scene, manager.active_scene)

    def testChangeAlignment(self):
        test_mock = mock.Mock()
        self.model.measurement_vectors = np.empty((0, 6, 1))
        manager = SceneManager(self.model, self.renderer, True)
        manager.rendered_alignment_changed = TestSignal()
        manager.rendered_alignment_changed.connect(test_mock)

        manager.changeRenderedAlignment(0)
        self.renderer.loadScene.assert_not_called()
        test_mock.assert_not_called()
        self.assertEqual(manager._rendered_alignment, 0)

        self.model.measurement_vectors = np.empty((2, 6, 1))
        manager.changeRenderedAlignment(1)
        self.renderer.loadScene.assert_called_with(manager.active_scene, False)
        test_mock.assert_called_with(1)
        self.assertEqual(manager._rendered_alignment, 1)

        active_node = Node()
        manager.active_scene = {Attributes.Vectors: active_node}
        active_node.children.extend([Node() for _ in range(3)])
        self.model.measurement_vectors = np.empty((3, 6, 3))
        manager.changeRenderedAlignment(2)
        self.assertEqual(self.renderer.loadScene.call_count, 2)
        test_mock.assert_called_with(2)
        self.assertEqual(manager._rendered_alignment, 2)
        self.assertEqual([active_node.children[i].visible for i in range(3)], [True, False, True])

        manager.changeRenderedAlignment(1)
        self.assertEqual([active_node.children[i].visible for i in range(3)], [True, True, False])

        manager.changeRenderedAlignment(3)
        self.assertEqual([active_node.children[i].visible for i in range(3)], [True, False, False])
