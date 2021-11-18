import sys
import unittest
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint, QEvent, QCoreApplication, QEventLoop, QDeadlineTimer
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox

APP = QApplication([])


def do_nothing(*_args, **_kwargs):
    pass


class TestSignal:
    def __init__(self):
        self.call = do_nothing

    def connect(self, call):
        self.call = call

    def emit(self, *args):
        self.call(*args)


class TestView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.presenter = None
        self.scenes = None
        self.showSelectChoiceMessage = None
        self.showMessage = do_nothing
        self.showPathLength = do_nothing
        self.showScriptExport = do_nothing


def click_message_box(button_index=0):
    """Simulates clicking a button on a message box

    :param button_index: index of button to click
    :type button_index: int
    """
    for widget in APP.topLevelWidgets():
        if isinstance(widget, QMessageBox):
            QTest.mouseClick(widget.buttons()[button_index], Qt.LeftButton)
            break


def edit_line_edit_text(line_edit, text):
    """Simulates clear and edit of a line edit

    :param line_edit: widget edit
    :type line_edit: QtWidgets.QLineEdit
    :param text: widget to scroll on
    :type text: str
    """
    QTest.keyClick(line_edit, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(line_edit, text)


def click_check_box(check_box):
    """Simulates clicking on the active area of a checkbox

    :param check_box: widget to click
    :type check_box: QtWidgets.QCheckBox
    """
    pos = QPoint(2, check_box.height() // 2)
    QTest.mouseClick(check_box, Qt.LeftButton, pos=pos)


def mouse_drag(widget, start_pos=None, stop_pos=None, button=Qt.LeftButton):
    """Simulates dragging the mouse from a start position to a stop position

    :param widget: widget to drag mouse on
    :type widget: QtWidgets.QWidget
    :param start_pos: start mouse position
    :type start_pos: QtCore.QPoint
    :param stop_pos: stop mouse position
    :type stop_pos: QtCore.QPoint
    :param button: button to press while dragging
    :type button: Qt.MouseButtons
    """
    if start_pos is None:
        start_pos = widget.rect().topLeft()
    if stop_pos is None:
        stop_pos = widget.rect().bottomRight()

    QTest.mousePress(widget, button, pos=start_pos)

    event = QMouseEvent(QEvent.MouseMove, stop_pos, button, button, Qt.NoModifier)
    APP.sendEvent(widget, event)

    QTest.mouseRelease(widget, button, pos=stop_pos)


def mouse_wheel_scroll(widget, pos=None, delta=50):
    """Simulates mouse wheel scroll

    :param widget: widget to scroll on
    :type widget: QtWidgets.QWidget
    :param pos: position of the mouse cursor relative to the widget
    :type pos: QtCore.QPoint
    :param delta: relative amount that the wheel was rotated
    :type delta: int
    """
    if pos is None:
        pos = widget.rect().center()

    event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(), QPoint(0, delta), delta, Qt.Vertical, Qt.NoButton,
                        Qt.NoModifier)

    APP.sendEvent(widget, event)


def click_list_widget_item(list_widget, list_item_index, modifier=Qt.NoModifier):
    """Simulates clicking on list item in list widget

    :param list_widget: list widget with list item
    :type list_widget: QtWidgets.QListWidget
    :param list_item_index: index of list item to click
    :type list_item_index: int
    :param modifier: keyboard modifier when clicking
    :type modifier: Qt.KeyboardModifier
    """
    item = list_widget.item(list_item_index)
    rect = list_widget.visualItemRect(item)
    QTest.mouseClick(list_widget.viewport(), Qt.LeftButton, modifier, rect.center())


def wait_for(predicate, timeout=5000):
    """Waits for timeout milliseconds or until the predicate returns true (Copied from QT).

    :param predicate: function
    :type predicate: Callable[None, bool]
    :param timeout: maximum time to wait for
    :type timeout: int
    """
    if predicate():
        return True
    remaining = timeout
    deadline = QDeadlineTimer()
    deadline.setRemainingTime(remaining, Qt.PreciseTimer)
    while True:
        QCoreApplication.processEvents(QEventLoop.AllEvents)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        remaining = deadline.remainingTime()
        if remaining > 0:
            QTest.qSleep(min(10, remaining))
        if predicate():
            return True
        remaining = deadline.remainingTime()
        if remaining <= 0:
            break
    return predicate()  # Last chance


class QTestCase(unittest.TestCase):
    """Test case for QT UI tests that ensure exception that occur in slot are properly
    logged and lead to test failure. If setUp or tearDown methods are overridden ensure to
    call base method.

    :param method_name: name of base method
    :type method_name: str
    """
    def __init__(self, method_name="runTest"):

        super().__init__(method_name)
        self.no_exceptions = True

    def setUp(self):
        self.no_exceptions = True

        def test_exception_hook(exc_type, exc_value, exc_traceback):
            self.no_exceptions = False
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = test_exception_hook

    def tearDown(self):
        if not self.no_exceptions:
            raise self.failureException("An exception occured in a PyQt slot")
        sys.excepthook = sys.__excepthook__

    @staticmethod
    def waitFor(predicate, timeout=5000):
        return wait_for(predicate, timeout)


SAMPLE_IDF = """{
    "instrument":{
        "name": "GENERIC",
        "version": "1.0",
        "gauge_volume": [0.0, 0.0, 0.0],
        "incident_jaws":{
            "beam_direction": [1.0, 0.0, 0.0],
            "beam_source": [-300.0, 0.0, 0.0],
            "aperture": [1.0, 1.0],
            "aperture_lower_limit": [0.5, 0.5],
            "aperture_upper_limit": [15.0, 15.0],
            "positioner": "incident_jaws",
            "visual":{
                    "pose": [300.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
            }
        },
        "detectors":[
            {
                "name":"Detector",
                "default_collimator": "Snout 25mm",
                "positioner": "diffracted_jaws",
                "diffracted_beam": [0.0, 1.0, 0.0]
            }
        ],
        "collimators":[
            {
                "name": "Snout 25mm",
                "detector": "Detector",
                "aperture": [1.0, 1.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
            {
                "name": "Snout 50mm",
                "detector": "Detector",
                "aperture": [2.0, 2.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
            {
                "name": "Snout 100mm",
                "detector": "Detector",
                "aperture": [1.0, 1.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
            {
                "name": "Snout 150mm",
                "detector": "Detector",
                "aperture": [4.0, 4.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            }
        ],
        "positioning_stacks":[
        {
            "name": "Positioning Table Only",
            "positioners": ["Positioning Table"]
        },
        {
            "name": "Positioning Table + Huber Circle",
            "positioners": ["Positioning Table", "Huber Circle"]
        }
        ],
        "positioners":[
            {
                "name": "Positioning Table",
                "base": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "custom_order": ["X Stage", "Y Stage", "Omega Stage"],
                "joints":[
                    {
                        "name": "X Stage",
                        "type": "prismatic",
                        "axis": [1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -201.0,
                        "upper_limit": 192.0,
                        "parent": "y_stage",
                        "child": "x_stage"
                    },
                    {
                        "name": "Y Stage",
                        "type": "prismatic",
                        "axis": [0.0, 1.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -101.0,
                        "upper_limit": 93.0,
                        "parent": "omega_stage",
                        "child": "y_stage"
                    },
                    {
                        "name": "Omega Stage",
                        "type": "revolute",
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -170.0,
                        "upper_limit": 166.0,
                        "parent": "base",
                        "child": "omega_stage"
                    }],
                "links": [
                    {"name": "base"},
                    {"name": "omega_stage"},
                    {"name": "y_stage"},
                    {"name": "x_stage"}
                ]
            },
            {
                "name": "Huber Circle",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Chi",
                        "type": "revolute",
                        "axis": [0.0, 1.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": 0.0,
                        "upper_limit": 300.0,
                        "home_offset": 0.0,
                        "parent": "base",
                        "child": "chi_axis"
                    },
                    {
                        "name": "Phi",
                        "type": "revolute",
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -360.0,
                        "upper_limit": 360.0,
                        "parent": "chi_axis",
                        "child": "phi_axis"
                    }

                ],
                "links": [
                    {"name": "base"},
                    {"name": "chi_axis"},
                    {"name": "phi_axis"}
                ]
            },
            {
                "name": "incident_jaws",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Jaws X Axis",
                        "type": "prismatic",
                        "axis": [1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -800.0,
                        "upper_limit": 0.0,
                        "home_offset": 0.0,
                        "parent": "base",
                        "child": "jaw_x_axis"
                    }

                ],
                "links": [
                    {"name": "base"},
                    {
                        "name": "jaw_x_axis", 
                        "visual":
                        {
                            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "mesh": "model_path",
                            "colour": [0.78, 0.39, 0.39]
                        }
                    }
                ]
            },
            {
                "name": "diffracted_jaws",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Angular Axis",
                        "type": "revolute",
                        "axis": [0.0, 0.0, -1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -120.0,
                        "upper_limit": 120.0,
                        "home_offset": 0.0,
                        "parent": "base",
                        "child": "angular_axis"
                    },
                    {
                        "name": "Radial Axis",
                        "type": "prismatic",
                        "axis": [-1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": 0.0,
                        "upper_limit": 100.0,
                        "home_offset": 0.0,
                        "parent": "angular_axis",
                        "child": "radial_axis"
                    }
                ],
                "links": [
                    {
                        "name": "base", 
                        "visual":
                        {
                            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "mesh": "model_path",
                            "colour": [0.78, 0.39, 0.39]
                        }
                    },
                    {"name": "angular_axis"},
                    {"name": "radial_axis"}
                ]
            }			
        ],
        "fixed_hardware":[
            {
                "name":  "monochromator",
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 90.0, 90.0, 0.0],
                    "mesh": "model_path",
                    "colour": [0.16, 0.39, 0.39]
                }
            },
            {
                "name": "floor",
                "visual":{
                    "pose": [0.0, 0.0, -15.0, 0.0, 0.0, 0.0],
                    "mesh": "model_path",
                    "colour": [0.7, 0.7, 0.7]
                }
            }
        ]
    }
}"""
