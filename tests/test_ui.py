import unittest
import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QApplication, QToolBar
from OpenGL.plugins import FormatHandler
from sscanss.core.scene import Node
from sscanss.core.util import Primitives, TransformType, PointType, DockFlag
from sscanss.ui.dialogs import (InsertPrimitiveDialog, TransformDialog, SampleManager, InsertPointDialog,
                                InsertVectorDialog, VectorManager, PickPointDialog)
from sscanss.ui.windows.main.view import MainWindow


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FormatHandler('sscanss',
                      'OpenGL.arrays.numpymodule.NumpyHandler',
                      ['sscanss.core.math.matrix.Matrix44'])
        cls.app = QApplication([])
        cls.window = MainWindow()
        cls.model = cls.window.presenter.model
        cls.window.show()

    @classmethod
    def tearDownClass(cls):
        cls.model.unsaved = False
        cls.window.close()

    @staticmethod
    def getDockedWidget(dock_manager, dock_flag):
        if dock_flag == DockFlag.Bottom:
            dock = dock_manager.bottom_dock
        else:
            dock = dock_manager.upper_dock

        return dock.widget()

    def testMainView(self):
        if not QTest.qWaitForWindowActive(self.window):
            self.skipTest('Window is not ready!')

        self.window.showNewProjectDialog()
        if not QTest.qWaitForWindowActive(self.window.project_dialog):
            self.skipTest('New Project Dialog is not ready!')

        self.window.recent_projects = []
        self.window.populateRecentMenu()
        self.window.recent_projects = ['c://test.hdf', 'c://test2.hdf', 'c://test3.hdf',
                                       'c://test4.hdf', 'c://test5.hdf', 'c://test6.hdf']
        self.window.populateRecentMenu()

        # Test project dialog validation
        self.assertTrue(self.window.project_dialog.isVisible())
        self.assertEqual(self.window.project_dialog.validator_textbox.text(), '')
        QTest.mouseClick(self.window.project_dialog.create_project_button, Qt.LeftButton)
        self.assertNotEqual(self.window.project_dialog.validator_textbox.text(), '')
        # Create new project
        QTest.keyClicks(self.window.project_dialog.project_name_textbox, 'Test')
        self.window.project_dialog.instrument_combobox.setCurrentText('IMAT')
        QTest.mouseClick(self.window.project_dialog.create_project_button,  Qt.LeftButton)
        QTest.qWait(500)
        self.assertFalse(self.window.project_dialog.isVisible())

        self.assertEqual(self.model.project_data['name'], 'Test')
        self.assertEqual(self.model.project_data['instrument'], 'IMAT')

        # Add sample
        self.assertEqual(len(self.model.sample), 0)
        self.window.docks.showInsertPrimitiveDialog(Primitives.Tube)
        widget = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertTrue(isinstance(widget, InsertPrimitiveDialog))
        self.assertEqual(widget.primitive, Primitives.Tube)
        self.assertTrue(widget.isVisible())
        QTest.keyClick(widget.textboxes['inner_radius'].form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget.textboxes['inner_radius'].form_lineedit, '10')
        QTest.keyClick(widget.textboxes['outer_radius'].form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget.textboxes['outer_radius'].form_lineedit, '10')
        self.assertFalse(widget.create_primitive_button.isEnabled())
        # Adds '0' to '10' to mak ethe radius '100'
        QTest.keyClicks(widget.textboxes['outer_radius'].form_lineedit, '0')
        self.assertTrue(widget.create_primitive_button.isEnabled())
        QTest.mouseClick(widget.create_primitive_button, Qt.LeftButton)
        self.assertEqual(len(self.model.sample), 1)

        self.window.docks.showInsertPrimitiveDialog(Primitives.Tube)
        widget_2 = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertIs(widget, widget_2)  # Since a Tube dialog is already open a new widget is not created
        self.assertEqual(widget.primitive, Primitives.Tube)
        self.window.docks.showInsertPrimitiveDialog(Primitives.Cuboid)
        widget_2 = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertIsNot(widget, widget_2)

        # Transform Sample
        sample = list(self.model.sample.items())[0][1]
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 0.0, 0.0], decimal=5)
        self.window.docks.showTransformDialog(TransformType.Translate)
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        QTest.keyClick(widget.y_axis.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.y_axis.form_lineedit, Qt.Key_Delete)
        self.assertFalse(widget.execute_button.isEnabled())

        QTest.keyClicks(widget.y_axis.form_lineedit, '100')
        self.assertTrue(widget.execute_button.isEnabled())
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        sample = list(self.model.sample.items())[0][1]
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)

        self.window.docks.showSampleManager()
        widget = self.getDockedWidget(self.window.docks, SampleManager.dock_flag)
        self.assertTrue(widget.isVisible())

        # render in transparent
        toolbar = self.window.findChildren(QToolBar, 'FileToolBar')[0]
        QTest.mouseClick(toolbar.widgetForAction(self.window.blend_render_action), Qt.LeftButton)
        self.assertEqual(self.window.gl_widget.render_mode, Node.RenderMode.Transparent)

        # Add Fiducial Points
        self.window.docks.showInsertPointDialog(PointType.Fiducial)
        widget = self.getDockedWidget(self.window.docks, InsertPointDialog.dock_flag)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key_Delete)
        self.assertFalse(widget.execute_button.isEnabled())

        QTest.keyClicks(widget.z_axis.form_lineedit, '100')
        self.assertTrue(widget.execute_button.isEnabled())
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)

        QTest.keyClick(widget.x_axis.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget.x_axis.form_lineedit, '50')
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)

        # Add Measurement Points
        self.window.docks.showInsertPointDialog(PointType.Measurement)
        widget = self.getDockedWidget(self.window.docks, InsertPointDialog.dock_flag)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key_Delete)
        self.assertFalse(widget.execute_button.isEnabled())

        QTest.keyClicks(widget.z_axis.form_lineedit, '10')
        self.assertTrue(widget.execute_button.isEnabled())
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)

        QTest.keyClick(widget.x_axis.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget.x_axis.form_lineedit, '20')
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)

        # render in wireframe
        QTest.mouseClick(toolbar.widgetForAction(self.window.line_render_action), Qt.LeftButton)
        self.assertEqual(self.window.gl_widget.render_mode, Node.RenderMode.Wireframe)

        # Add Vectors
        self.window.docks.showVectorManager()
        widget = self.getDockedWidget(self.window.docks, VectorManager.dock_flag)
        self.assertTrue(widget.isVisible())

        self.window.docks.showInsertVectorDialog()
        widget = self.getDockedWidget(self.window.docks, InsertVectorDialog.dock_flag)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)

        # Add points graphically
        self.window.docks.showPickPointDialog()
        widget = self.getDockedWidget(self.window.docks, PickPointDialog.dock_flag)
        QTest.mouseClick(widget.help_button, Qt.LeftButton)
        self.assertTrue(widget.view.show_help)
        QTest.mouseClick(widget.reset_button, Qt.LeftButton)
        QTest.keyClick(widget.plane_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.plane_lineedit, Qt.Key_Delete)
        QTest.keyClicks(widget.plane_lineedit, '-10')
        QTest.keyClick(widget.plane_lineedit, Qt.Key_Enter)

        widget.tabs.setCurrentIndex(1)
        QTest.mouseClick(widget.line_selector, Qt.LeftButton)
        self.assertTrue(widget.line_tool_widget.isVisible())
        QTest.mouseClick(widget.area_selector, Qt.LeftButton)
        self.assertFalse(widget.line_tool_widget.isVisible())
        self.assertTrue(widget.area_tool_widget.isVisible())
        QTest.mouseClick(widget.point_selector, Qt.LeftButton)
        self.assertFalse(widget.line_tool_widget.isVisible())
        self.assertFalse(widget.area_tool_widget.isVisible())

        widget.tabs.setCurrentIndex(2)
        pos = QPoint(2, widget.show_grid_checkbox.height() / 2)
        QTest.mouseClick(widget.show_grid_checkbox, Qt.LeftButton, pos=pos)
        self.assertTrue(widget.view.show_grid)
        pos = QPoint(2, widget.snap_to_grid_checkbox.height() / 2)
        QTest.mouseClick(widget.snap_to_grid_checkbox, Qt.LeftButton, pos=pos)
        self.assertTrue(widget.view.snap_to_grid)
        self.assertTrue(widget.grid_size_widget.isVisible())

        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        QTest.mouseClick(widget.view.viewport(), Qt.LeftButton)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        widget.hide()

    def testOtherWindows(self):
        self.window.showUndoHistory()
        self.assertTrue(self.window.undo_view.isVisible())
        self.window.undo_view.close()

        self.window.showProgressDialog('Testing')
        self.assertTrue(self.window.progress_dialog.isVisible())
        self.window.progress_dialog.close()
