import unittest
import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtWidgets import QApplication, QToolBar, QMessageBox
from OpenGL.plugins import FormatHandler
from sscanss.core.scene import Node
from sscanss.core.util import Primitives, PointType, DockFlag
from sscanss.ui.dialogs import (InsertPrimitiveDialog, TransformDialog, SampleManager, InsertPointDialog,
                                InsertVectorDialog, VectorManager, PickPointDialog, JawControl, PositionerControl,
                                DetectorControl, PointManager)
from sscanss.ui.windows.main.view import MainWindow


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FormatHandler('sscanss',
                      'OpenGL.arrays.numpymodule.NumpyHandler',
                      ['sscanss.core.math.matrix.Matrix44'])
        cls.app = QApplication([])
        cls.window = MainWindow()
        cls.toolbar = cls.window.findChildren(QToolBar, 'FileToolBar')[0]
        cls.model = cls.window.presenter.model
        cls.window.show()

    @classmethod
    def tearDownClass(cls):
        cls.model.unsaved = False
        cls.window.close()

    @classmethod
    def clickMessageBox(cls, button_index=0):
        for widget in cls.app.topLevelWidgets():
            if isinstance(widget, QMessageBox):
                QTest.mouseClick(widget.buttons()[button_index], Qt.LeftButton)
                break

    @staticmethod
    def editFormControl(control, text, delay_before=-1):
        QTest.keyClick(control.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(control.form_lineedit, text, delay=delay_before)

    @staticmethod
    def clickCheckBox(check_box, delay_before=-1):
        pos = QPoint(2, check_box.height() / 2)
        QTest.mouseClick(check_box, Qt.LeftButton, pos=pos, delay=delay_before)

    @staticmethod
    def clickListWidgetItem(list_widget, list_item_index, modifier=Qt.NoModifier):
        item = list_widget.item(list_item_index)
        rect = list_widget.visualItemRect(item)
        QTest.mouseClick(list_widget.viewport(), Qt.LeftButton, modifier, rect.center())

    @classmethod
    def triggerUndo(cls):
        cls.window.undo_action.trigger()
        QTest.qWait(100)

    @classmethod
    def triggerRedo(cls):
        cls.window.redo_action.trigger()
        QTest.qWait(100)

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

        self.createProject()
        self.addSample()
        self.window.show_bounding_box_action.trigger()
        self.transformSample()

        # render in transparent
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.blend_render_action), Qt.LeftButton)
        self.assertEqual(self.window.selected_render_mode, Node.RenderMode.Transparent)

        self.keyinFiducials()
        self.keyinPoints()

        # render in wireframe
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.line_render_action), Qt.LeftButton)
        self.assertEqual(self.window.selected_render_mode, Node.RenderMode.Wireframe)

        self.insertVectors()
        self.jawControl()
        self.pointPicking()
        self.switchInstrument()
        self.positionerControl()
        self.detectorControl()

    def createProject(self):
        self.window.showNewProjectDialog()
        if not QTest.qWaitForWindowActive(self.window.project_dialog):
            self.skipTest('New Project Dialog is not ready!')

        # Test project dialog validation
        self.assertTrue(self.window.project_dialog.isVisible())
        self.assertEqual(self.window.project_dialog.validator_textbox.text(), '')
        QTest.mouseClick(self.window.project_dialog.create_project_button, Qt.LeftButton)
        self.assertNotEqual(self.window.project_dialog.validator_textbox.text(), '')
        # Create new project
        QTest.keyClicks(self.window.project_dialog.project_name_textbox, 'Test')
        self.window.project_dialog.instrument_combobox.setCurrentText('IMAT')
        QTest.mouseClick(self.window.project_dialog.create_project_button,  Qt.LeftButton)
        QTest.qWait(500)  # wait is necessary since instrument is created on another thread
        self.assertFalse(self.window.project_dialog.isVisible())
        self.assertEqual(self.model.project_data['name'], 'Test')
        self.assertEqual(self.model.instrument.name, 'IMAT')

    def addSample(self):
        # Add sample
        self.assertEqual(len(self.model.sample), 0)
        self.window.docks.showInsertPrimitiveDialog(Primitives.Tube)
        widget = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertTrue(isinstance(widget, InsertPrimitiveDialog))
        self.assertEqual(widget.primitive, Primitives.Tube)
        self.assertTrue(widget.isVisible())
        self.editFormControl(widget.textboxes['inner_radius'], '10')
        self.editFormControl(widget.textboxes['outer_radius'], '10')
        # equal inner radius and outer radius is an invalid tube so validation should trigger
        self.assertFalse(widget.create_primitive_button.isEnabled())
        # Adds '0' to '10' to make the radius '100'
        QTest.keyClicks(widget.textboxes['outer_radius'].form_lineedit, '0')
        self.assertTrue(widget.create_primitive_button.isEnabled())
        QTest.mouseClick(widget.create_primitive_button, Qt.LeftButton)
        self.assertEqual(len(self.model.sample), 1)

        # Add a second sample
        self.window.docks.showInsertPrimitiveDialog(Primitives.Tube)
        widget_2 = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertIs(widget, widget_2)  # Since a Tube dialog is already open a new widget is not created
        self.assertEqual(widget.primitive, Primitives.Tube)
        self.window.docks.showInsertPrimitiveDialog(Primitives.Cuboid)
        widget_2 = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertIsNot(widget, widget_2)
        QTimer.singleShot(100, lambda: self.clickMessageBox(0)) # click first button in message box
        QTest.mouseClick(widget_2.create_primitive_button, Qt.LeftButton)
        QTest.qWait(50)

        # Checks Sample Manager
        widget = self.getDockedWidget(self.window.docks, SampleManager.dock_flag)
        self.assertTrue(widget.isVisible())
        self.assertEqual(list(self.model.sample.keys())[0], 'Tube')
        QTest.mouseClick(widget.priority_button, Qt.LeftButton)
        self.clickListWidgetItem(widget.list_widget, 1)
        QTest.mouseClick(widget.priority_button, Qt.LeftButton)
        self.assertEqual(list(self.model.sample.keys())[0], 'Cuboid')
        self.triggerUndo()
        self.assertEqual(list(self.model.sample.keys())[0], 'Tube')

        self.clickListWidgetItem(widget.list_widget, 0)
        QTest.mouseClick(widget.merge_button, Qt.LeftButton)
        self.assertEqual(len(self.model.sample), 2)
        self.clickListWidgetItem(widget.list_widget, 1, Qt.ControlModifier)
        QTest.mouseClick(widget.merge_button, Qt.LeftButton)
        self.assertEqual(len(self.model.sample), 1)
        self.triggerUndo()
        self.assertEqual(len(self.model.sample), 2)

        self.clickListWidgetItem(widget.list_widget, 1)
        QTest.mouseClick(widget.delete_button, Qt.LeftButton)
        self.assertEqual(len(self.model.sample), 1)
        self.triggerUndo()
        self.assertEqual(len(self.model.sample), 2)

    def transformSample(self):
        # Transform Sample
        sample = list(self.model.sample.items())[0][1]
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 0.0, 0.0], decimal=5)
        self.window.translate_sample_action.trigger()
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        QTest.keyClick(widget.y_axis.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.y_axis.form_lineedit, Qt.Key_Delete)
        self.assertFalse(widget.execute_button.isEnabled())

        QTest.keyClicks(widget.y_axis.form_lineedit, '100')
        self.assertTrue(widget.execute_button.isEnabled())
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        sample = list(self.model.sample.items())[0][1]
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 0.0, 0.0], decimal=5)
        self.triggerRedo()
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)

    def keyinFiducials(self):
        # Add Fiducial Points
        self.window.keyin_fiducial_action.trigger()
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
        self.triggerUndo()
        self.assertEqual(self.model.fiducials.size, 1)
        self.triggerRedo()
        self.assertEqual(self.model.fiducials.size, 2)

        # Test Point Manager
        widget = self.getDockedWidget(self.window.docks, PointManager.dock_flag)
        self.assertTrue(widget.isVisible())
        self.assertEqual(widget.point_type, PointType.Fiducial)
        xPos = widget.table_view.columnViewportPosition(0) + 5
        yPos = widget.table_view.rowViewportPosition(1) + 10
        pos = QPoint(xPos, yPos)
        QTest.mouseClick(widget.table_view.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
        QTest.mouseClick(widget.move_up_button, Qt.LeftButton)
        QTest.qWait(50)
        QTest.mouseClick(widget.move_down_button, Qt.LeftButton)
        QTest.qWait(50)

        QTest.mouseDClick(widget.table_view.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
        QTest.keyClicks(widget.table_view.viewport().focusWidget(), '100')
        QTest.keyClick(widget.table_view.viewport().focusWidget(), Qt.Key_Enter)
        QTest.qWait(50)
        np.testing.assert_array_almost_equal(self.model.fiducials[1].points, [100., 0., 100.], decimal=3)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(self.model.fiducials[1].points, [50., 0., 100.], decimal=3)
        QTest.qWait(50)

        QTest.mouseClick(widget.delete_button, Qt.LeftButton)
        QTest.qWait(50)
        self.assertEqual(self.model.fiducials.size, 1)
        self.triggerUndo()
        self.assertEqual(self.model.fiducials.size, 2)

    def keyinPoints(self):
        # Add Measurement Points
        self.window.keyin_measurement_action.trigger()
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
        self.triggerUndo()
        self.assertEqual(self.model.measurement_points.size, 1)
        self.triggerRedo()
        self.assertEqual(self.model.measurement_points.size, 2)

        # Test Point Manager
        widget = self.getDockedWidget(self.window.docks, PointManager.dock_flag)
        self.assertTrue(widget.isVisible())
        self.assertEqual(widget.point_type, PointType.Measurement)

    def insertVectors(self):
        # Add Vectors via the dialog
        self.window.select_strain_component_action.trigger()
        widget = self.getDockedWidget(self.window.docks, InsertVectorDialog.dock_flag)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        QTest.keyClicks(widget.detector_combobox, '2')
        QTest.mouseClick(widget.component_combobox, Qt.LeftButton)
        QTest.keyClick(widget.component_combobox, Qt.Key_Down)
        self.clickCheckBox(widget.reverse_checkbox)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        QTest.qWait(200)  # wait is necessary since vectors are created on another thread

        mv = widget.parent_model.measurement_vectors
        self.assertEqual(mv.shape, (2, 6, 1))
        np.testing.assert_array_almost_equal(mv[0, :, 0], [1, 0, 0, 0, -1, 0], decimal=5)

        QTest.keyClicks(widget.alignment_combobox, 'a')
        QTest.mouseClick(widget.component_combobox, Qt.LeftButton, delay=100)
        QTest.keyClick(widget.component_combobox, Qt.Key_Down, Qt.NoModifier)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        QTest.qWait(200)

        QTest.keyClicks(widget.component_combobox, 'k')
        QTest.keyClicks(widget.detector_combobox, '1', delay=50)
        self.editFormControl(widget.x_axis, '1.0')
        self.editFormControl(widget.y_axis, '1.0')
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        QTest.qWait(200)

        mv = widget.parent_model.measurement_vectors
        self.assertEqual(mv.shape, (2, 6, 2))
        np.testing.assert_array_almost_equal(mv[0, :, 1], [-0.70711, -0.70711, 0, 0, 0, -1.0], decimal=5)

        # Test Vector Manager
        widget = self.getDockedWidget(self.window.docks, VectorManager.dock_flag)
        self.assertTrue(widget.isVisible())

    def pointPicking(self):
        # Add points graphically
        self.window.pick_measurement_action.trigger()
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
        self.clickCheckBox(widget.show_grid_checkbox)
        self.assertTrue(widget.view.show_grid)
        self.clickCheckBox(widget.snap_to_grid_checkbox)
        self.assertTrue(widget.view.snap_to_grid)
        self.assertTrue(widget.grid_size_widget.isVisible())

        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 2)
        QTest.mouseClick(widget.view.viewport(), Qt.LeftButton)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 3)
        widget.hide()

    def switchInstrument(self):
        # switch instruments
        self.assertNotEqual(self.window.undo_stack.count(), 0)
        QTimer.singleShot(200, lambda: self.clickMessageBox(0))  # click first button in message box
        self.window.presenter.changeInstrument('ENGIN-X')
        QTest.qWait(500)
        self.assertEqual(self.window.undo_stack.count(), 0)
        self.assertEqual(self.model.project_data['name'], 'Test')
        self.assertEqual(self.model.instrument.name, 'ENGIN-X')

        self.assertIs(self.window.scenes.active_scene, self.window.scenes.sample_scene)
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.toggle_scene_action), Qt.LeftButton)
        self.assertIs(self.window.scenes.active_scene, self.window.scenes.instrument_scene)
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.toggle_scene_action), Qt.LeftButton)
        self.assertIs(self.window.scenes.active_scene, self.window.scenes.sample_scene)

    def jawControl(self):
        # Test incident jaws Dialog and change jaw position
        self.window.docks.showJawControl()
        widget = self.getDockedWidget(self.window.docks, JawControl.dock_flag)
        jaw_form = widget.position_form_group.form_controls[0]
        jaw = self.model.instrument.jaws.positioner
        new_value = jaw.links[0].lower_limit + (jaw.links[0].offset - jaw.links[0].lower_limit) / 2
        self.editFormControl(jaw_form, f'{new_value}')
        QTest.mouseClick(widget.move_jaws_button, Qt.LeftButton)
        set_point = self.model.instrument.jaws.positioner.set_points[0]
        self.assertAlmostEqual(set_point, new_value, 3)

        self.editFormControl(jaw_form, f'{jaw.links[0].lower_limit - 1}')
        self.assertFalse(jaw_form.valid)
        self.assertFalse(widget.move_jaws_button.isEnabled())
        QTest.mouseClick(jaw_form.extra[0], Qt.LeftButton)
        self.assertTrue(jaw_form.valid)
        self.assertTrue(widget.move_jaws_button.isEnabled())
        self.triggerUndo()
        self.assertFalse(jaw_form.valid)
        self.assertFalse(widget.move_jaws_button.isEnabled())

        # Change aperture of the jaw
        aperture_form = widget.aperture_form_group.form_controls
        self.editFormControl(aperture_form[0], '5.000')
        self.editFormControl(aperture_form[1], '6.000')
        old_aperture = self.model.instrument.jaws.aperture
        QTest.mouseClick(widget.change_aperture_button, Qt.LeftButton)
        aperture = self.model.instrument.jaws.aperture
        np.testing.assert_array_almost_equal(aperture, (5.000, 6.000), decimal=3)
        self.triggerUndo()
        aperture = self.model.instrument.jaws.aperture
        np.testing.assert_array_almost_equal(aperture, old_aperture, decimal=3)

    def positionerControl(self):
        # Test Positioner Dialog
        self.window.docks.showPositionerControl()
        widget = self.getDockedWidget(self.window.docks, PositionerControl.dock_flag)
        positioner_name = self.model.instrument.positioning_stack.name
        QTest.mouseClick(widget.stack_combobox, Qt.LeftButton, delay=100)
        QTest.keyClick(widget.stack_combobox, Qt.Key_Down, Qt.NoModifier)
        self.assertNotEqual(self.model.instrument.positioning_stack.name, positioner_name)
        self.triggerUndo()
        self.assertEqual(self.model.instrument.positioning_stack.name, positioner_name)

        form = widget.positioner_form_controls[0]
        stack = self.model.instrument.positioning_stack
        new_value = stack.links[0].upper_limit - (stack.links[0].upper_limit - stack.links[0].offset) / 2
        self.editFormControl(form, f'{new_value}')

        form = widget.positioner_form_controls[1]
        QTest.mouseClick(form.extra[0], Qt.LeftButton)
        self.triggerUndo()
        QTest.mouseClick(form.extra[1], Qt.LeftButton)
        self.triggerUndo()
        old_set_point = self.model.instrument.positioning_stack.set_points[0]
        self.window.scenes.switchToSampleScene()
        QTest.mouseClick(widget.move_joints_button, Qt.LeftButton)
        set_point = self.model.instrument.positioning_stack.set_points[0]
        self.assertAlmostEqual(set_point, new_value, 3)
        self.triggerUndo()
        set_point = self.model.instrument.positioning_stack.set_points[0]
        self.assertAlmostEqual(old_set_point, set_point, 3)
        self.triggerRedo()
        set_point = self.model.instrument.positioning_stack.set_points[0]
        self.assertAlmostEqual(new_value, set_point, 3)

    def detectorControl(self):
        # Test Detector Widget
        detector_name = list(self.model.instrument.detectors.keys())[0]
        self.window.docks.showDetectorControl(detector_name)
        widget = self.getDockedWidget(self.window.docks, DetectorControl.dock_flag)
        widget.hide()

        detector = self.model.instrument.detectors[detector_name]
        old_collimator = detector.current_collimator
        self.window.presenter.changeCollimators(detector_name, None)
        self.assertIs(detector.current_collimator, None)
        self.triggerUndo()
        self.assertEqual(detector.current_collimator, old_collimator)

    def testOtherWindows(self):
        # Test the Recent project menu
        self.window.recent_projects = []
        self.window.populateRecentMenu()
        self.window.recent_projects = ['c://test.hdf', 'c://test2.hdf', 'c://test3.hdf',
                                       'c://test4.hdf', 'c://test5.hdf', 'c://test6.hdf']
        self.window.populateRecentMenu()

        self.window.showUndoHistory()
        self.assertTrue(self.window.undo_view.isVisible())
        self.window.undo_view.close()

        self.window.showProgressDialog('Testing')
        self.assertTrue(self.window.progress_dialog.isVisible())
        self.window.progress_dialog.close()

        self.window.showPreferences()
        self.assertTrue(self.window.preferences.isVisible())
        self.window.preferences.close()
