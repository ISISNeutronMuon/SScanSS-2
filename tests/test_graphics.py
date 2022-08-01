import unittest
import unittest.mock as mock
from PyQt5.QtCore import QPointF, QRectF, QEvent, Qt
from PyQt5.QtGui import QTransform, QMouseEvent
from sscanss.app.widgets import BoxGrid, PolarGrid, ObjectSnap
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

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(0, 50), False)
        self.assertTrue(object_snap.initialized)
        self.assertAlmostEqual(dx, -10.0, 5)
        self.assertAlmostEqual(dy, 0.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), -9.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -1.0, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(100, 0), False)
        self.assertAlmostEqual(dx, 20.0, 5)
        self.assertAlmostEqual(dy, 0.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 11.0, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -1.0, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(-70.72, -70.72), False)
        self.assertAlmostEqual(dx, -14.1421356, 5)
        self.assertAlmostEqual(dy, -14.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), -3.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -15.1421356, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, 0), QPointF(141.44, 141.72), False)
        self.assertAlmostEqual(dx, 28.284271, 5)
        self.assertAlmostEqual(dy, 28.284271, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 25.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 13.1421356, 5)

        object_snap.remaining_shift = (0, 0)
        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(0, -0), QPointF(0, 50), True)
        self.assertEqual(dx, 0)
        self.assertEqual(dy, 0)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 25.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), 13.1421356, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(1, 0), QPointF(0.92106, -0.389421), True)
        self.assertAlmostEqual(dx, 5.857864, 5)
        self.assertAlmostEqual(dy, -14.1421356, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.x(), 31, 5)
        self.assertAlmostEqual(object_snap.anchor_offset.y(), -1, 5)

        dx, dy = object_snap.snapAnchorToPolarGrid(QPointF(1, 0), QPointF(0.62161, 0.78333), True)
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
        event = QMouseEvent(QEvent.MouseButtonPress, QPointF(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        self.assertFalse(object_snap.eventFilter(self.graphics_view, event))

        object_snap.enabled = True
        self.graphics_view.mapToScene.return_value = QPointF(100, 0)
        event = QMouseEvent(QEvent.MouseButtonPress, QPointF(1, 1), Qt.MiddleButton, Qt.MiddleButton, Qt.NoModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        test_mock.assert_not_called()

        self.graphics_view.mapToScene.return_value = QPointF(-100, 0)
        event = QMouseEvent(QEvent.MouseMove, QPointF(1, 1), Qt.RightButton, Qt.RightButton, Qt.ControlModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        self.assertAlmostEqual(test_mock.call_args[0][0], -10, 5)
        self.assertAlmostEqual(test_mock.call_args[0][1], 0, 5)

        object_snap.grid = PolarGrid()
        event = QMouseEvent(QEvent.MouseButtonPress, QPointF(1, 1), Qt.RightButton, Qt.RightButton,
                            Qt.ControlModifier | Qt.ShiftModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        self.assertEqual(test_mock.call_count, 2)

        self.graphics_view.mapToScene.return_value = QPointF(-100, 110)
        event = QMouseEvent(QEvent.MouseMove, QPointF(1, 1), Qt.MiddleButton, Qt.MiddleButton, Qt.ShiftModifier)
        self.assertTrue(object_snap.eventFilter(self.graphics_view, event))
        self.assertAlmostEqual(test_mock.call_args[0][0], 10, 5)
        self.assertAlmostEqual(test_mock.call_args[0][1], 10, 5)

        event = QMouseEvent(QEvent.MouseButtonPress, QPointF(1, 1), Qt.MiddleButton, Qt.MiddleButton, Qt.AltModifier)
        self.assertFalse(object_snap.eventFilter(self.graphics_view, event))
