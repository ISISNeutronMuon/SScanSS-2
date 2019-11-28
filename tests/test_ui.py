import enum
import pathlib
import shutil
import tempfile
import unittest
import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint, QTimer, QSettings, QEvent
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QApplication, QToolBar, QMessageBox, QComboBox
from OpenGL.plugins import FormatHandler
import sscanss.config as config
from sscanss.core.scene import Node
from sscanss.core.util import Primitives, PointType, DockFlag
from sscanss.ui.dialogs import (InsertPrimitiveDialog, TransformDialog, SampleManager, InsertPointDialog,
                                InsertVectorDialog, VectorManager, PickPointDialog, JawControl, PositionerControl,
                                DetectorControl, PointManager)
from sscanss.ui.window.view import MainWindow


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FormatHandler('sscanss',
                      'OpenGL.arrays.numpymodule.NumpyHandler',
                      ['sscanss.core.math.matrix.Matrix44'])
        cls.app = QApplication([])
        cls.window = MainWindow()
        cls.toolbar = cls.window.findChild(QToolBar)
        cls.model = cls.window.presenter.model
        cls.data_dir = pathlib.Path(tempfile.mkdtemp())
        cls.ini_file = cls.data_dir / 'settings.ini'
        config.settings._setting = QSettings(str(cls.ini_file), QSettings.IniFormat)
        config.LOG_PATH = cls.data_dir / 'logs'
        cls.window.show()
        # if not QTest.qWaitForWindowActive(cls.window):
        #     raise unittest.SkipTest('Window is not ready!')

    @classmethod
    def tearDownClass(cls):
        cls.window.undo_stack.setClean()
        cls.window.close()
        config.logging.shutdown()
        shutil.rmtree(cls.data_dir)

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
    def mouseDrag(widget, start_pos=None, stop_pos=None, button=Qt.LeftButton):
        if start_pos is None:
            start_pos = widget.rect().topLeft()
        if stop_pos is None:
            stop_pos = widget.rect().bottomRight()

        QTest.mousePress(widget, button, pos=start_pos)

        event = QMouseEvent(QEvent.MouseMove, stop_pos, button, button, Qt.NoModifier)
        QApplication.sendEvent(widget, event)

        QTest.mouseRelease(widget, button, pos=stop_pos)

    @staticmethod
    def mouseWheelScroll(widget, pos=None, delta=50):
        if pos is None:
            pos = widget.rect().center()
        event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(), QPoint(0, delta), delta, Qt.Vertical, Qt.NoButton,
                            Qt.NoModifier)
        QApplication.sendEvent(widget, event)

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
        self.createProject()
        self.addSample()
        self.assertFalse(self.window.gl_widget.show_bounding_box)
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.show_bounding_box_action), Qt.LeftButton)
        self.assertTrue(self.window.gl_widget.show_bounding_box)
        self.assertTrue(self.window.gl_widget.show_coordinate_frame)
        self.window.show_coordinate_frame_action.trigger()
        self.assertFalse(self.window.gl_widget.show_coordinate_frame)
        camera = self.window.gl_widget.scene.camera
        self.window.view_from_menu.actions()[0].trigger()
        self.assertEqual(camera.mode, camera.Projection.Orthographic)
        self.window.reset_camera_action.trigger()
        self.assertEqual(camera.mode, camera.Projection.Perspective)

        self.mouseDrag(self.window.gl_widget)
        self.mouseDrag(self.window.gl_widget, button=Qt.RightButton)
        self.mouseWheelScroll(self.window.gl_widget, delta=20)
        self.mouseWheelScroll(self.window.gl_widget, delta=-10)

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
        self.alignSample()

    def createProject(self):
        self.window.showNewProjectDialog()

        # Test project dialog validation
        self.assertTrue(self.window.project_dialog.isVisible())
        self.assertEqual(self.window.project_dialog.validator_textbox.text(), '')
        QTest.mouseClick(self.window.project_dialog.create_project_button, Qt.LeftButton)
        self.assertNotEqual(self.window.project_dialog.validator_textbox.text(), '')
        # Create new project
        QTest.keyClicks(self.window.project_dialog.project_name_textbox, 'Test')
        self.window.project_dialog.instrument_combobox.setCurrentText('IMAT')
        QTest.mouseClick(self.window.project_dialog.create_project_button,  Qt.LeftButton)
        QTest.qWait(1000)  # wait is necessary since instrument is created on another thread
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
        QTest.keyClick(widget.tool.y_position.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.tool.y_position.form_lineedit, Qt.Key_Delete)
        self.assertFalse(widget.tool.execute_button.isEnabled())

        QTest.keyClicks(widget.tool.y_position.form_lineedit, '100')
        self.assertTrue(widget.tool.execute_button.isEnabled())
        QTest.mouseClick(widget.tool.execute_button, Qt.LeftButton)
        sample = list(self.model.sample.items())[0][1]
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 0.0, 0.0], decimal=5)
        self.triggerRedo()
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)

        self.window.rotate_sample_action.trigger()
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        QTest.keyClick(widget.tool.z_rotation.form_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.tool.z_rotation.form_lineedit, Qt.Key_Delete)
        self.assertFalse(widget.tool.execute_button.isEnabled())

        QTest.keyClicks(widget.tool.z_rotation.form_lineedit, '90')
        self.assertTrue(widget.tool.execute_button.isEnabled())
        QTest.mouseClick(widget.tool.execute_button, Qt.LeftButton)
        sample = list(self.model.sample.items())[0][1]
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [-100.0, 0.0, 0.0], decimal=5)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)
        self.triggerRedo()
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [-100.0, 0.0, 0.0], decimal=5)

        self.window.transform_sample_action.trigger()

        self.window.move_origin_action.trigger()
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        for i in range(widget.tool.move_combobox.count()):
            widget.tool.move_combobox.setCurrentIndex(i)

        for i in range(widget.tool.ignore_combobox.count()):
            widget.tool.ignore_combobox.setCurrentIndex(i)

        QTest.mouseClick(widget.tool.execute_button, Qt.LeftButton)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(sample.bounding_box.center, [-100., 0.0, 0.0], decimal=5)
        self.triggerRedo()

        self.window.plane_align_action.trigger()
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        for i in range(widget.tool.plane_combobox.count()):
            widget.tool.plane_combobox.setCurrentIndex(i)

        QTest.mouseClick(widget.tool.execute_button, Qt.LeftButton)

        QTest.mouseClick(widget.tool.pick_button, Qt.LeftButton)
        QTest.mouseClick(self.window.gl_widget, Qt.LeftButton)
        QTest.mouseClick(widget.tool.select_button, Qt.LeftButton)


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
        detector_names = list(widget.parent_model.instrument.detectors.keys())

        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        QTest.keyClicks(widget.detector_combobox, detector_names[1][0], delay=50)
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
        QTest.keyClicks(widget.detector_combobox, detector_names[0][0], delay=50)
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
        viewport = widget.view.viewport()

        for i in range(widget.plane_combobox.count()):
            widget.plane_combobox.setCurrentIndex(i)

        self.mouseDrag(widget.plane_slider, QPoint(), QPoint(10, 0))

        QTest.keyClick(widget.plane_lineedit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget.plane_lineedit, Qt.Key_Delete)
        QTest.keyClicks(widget.plane_lineedit, '-10')
        QTest.keyClick(widget.plane_lineedit, Qt.Key_Enter)

        widget.tabs.setCurrentIndex(2)
        self.clickCheckBox(widget.show_grid_checkbox)
        self.assertTrue(widget.view.show_grid)
        self.clickCheckBox(widget.snap_to_grid_checkbox)
        self.assertTrue(widget.view.snap_to_grid)
        self.assertTrue(widget.grid_widget.isVisible())
        combo = widget.grid_widget.findChild(QComboBox)
        current_index = combo.currentIndex()
        new_index = (current_index + 1) % combo.count()
        grid_type = widget.view.grid.type
        combo.setCurrentIndex(new_index)
        QTest.qWait(10)  # Delay allow the grid to render
        self.assertNotEqual(grid_type, widget.view.grid.type)
        combo.setCurrentIndex(current_index)
        QTest.qWait(10)  # Delay allow the grid to render
        self.assertEqual(grid_type, widget.view.grid.type)

        widget.tabs.setCurrentIndex(1)
        QTest.mouseClick(widget.point_selector, Qt.LeftButton)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 2)
        QTest.mouseClick(viewport, Qt.LeftButton)
        QTest.mouseClick(widget.execute_button, Qt.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 3)

        widget.tabs.setCurrentIndex(1)
        QTest.mouseClick(widget.line_selector, Qt.LeftButton)
        self.assertTrue(widget.line_tool_widget.isVisible())
        widget.line_point_count_spinbox.setValue(widget.scene.line_tool_point_count + 1)
        expected_count = len(widget.scene.items()) + widget.scene.line_tool_point_count
        self.mouseDrag(viewport)
        self.assertEqual(len(widget.scene.items()), expected_count)

        QTest.mouseClick(widget.area_selector, Qt.LeftButton)
        self.assertFalse(widget.line_tool_widget.isVisible())
        self.assertTrue(widget.area_tool_widget.isVisible())
        widget.area_x_spinbox.setValue(widget.scene.area_tool_size[0] + 1)
        widget.area_y_spinbox.setValue(widget.scene.area_tool_size[1] + 2)
        expected_count = len(widget.scene.items()) + (widget.scene.area_tool_size[0] * widget.scene.area_tool_size[1])
        self.mouseDrag(viewport)
        self.assertEqual(len(widget.scene.items()), expected_count)
        QTest.mouseClick(widget.object_selector, Qt.LeftButton)
        self.assertFalse(widget.line_tool_widget.isVisible())
        self.assertFalse(widget.area_tool_widget.isVisible())
        self.mouseDrag(viewport)
        selected_count = len(widget.scene.selectedItems())
        QTest.keyClick(viewport, Qt.Key_Delete)
        self.assertEqual(len(widget.scene.items()), expected_count - selected_count)

        self.assertFalse(widget.view.has_foreground)
        QTest.mouseClick(widget.help_button, Qt.LeftButton)
        QTest.qWait(10)  # Delay allow the grid to render
        self.assertTrue(widget.view.has_foreground and not widget.view.show_help)

        self.assertTrue(widget.view.scene_transform.isIdentity())
        self.mouseDrag(viewport, button=Qt.MiddleButton)
        self.assertTrue(widget.view.scene_transform.isTranslating())
        self.assertFalse(widget.view.scene_transform.isRotating())
        self.mouseDrag(viewport, button=Qt.RightButton)
        # QTransform type is always True for translation when rotation is True
        self.assertTrue(widget.view.scene_transform.isTranslating())
        self.assertTrue(widget.view.scene_transform.isRotating())
        widget.view.resetTransform()
        self.assertTrue(widget.view.transform().isIdentity())
        self.assertFalse(widget.view.transform().isScaling())
        self.mouseWheelScroll(viewport)
        self.assertTrue(widget.view.transform().isScaling())
        self.mouseWheelScroll(viewport, delta=-10)
        self.assertTrue(widget.view.transform().isIdentity())
        QTest.mouseClick(widget.reset_button, Qt.LeftButton)
        self.assertTrue(widget.view.transform().isIdentity() and widget.view.scene_transform.isIdentity())

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
        index = stack.order[0]
        new_value = stack.links[index].upper_limit - (stack.links[index].upper_limit - stack.links[index].offset) / 2
        self.editFormControl(form, f'{new_value}')

        form = widget.positioner_form_controls[1]
        QTest.mouseClick(form.extra[0], Qt.LeftButton)
        self.triggerUndo()
        QTest.mouseClick(form.extra[1], Qt.LeftButton)
        self.triggerUndo()
        old_set_point = stack.toUserFormat(stack.set_points)[0]
        self.window.scenes.switchToSampleScene()
        QTest.mouseClick(widget.move_joints_button, Qt.LeftButton)
        set_point = stack.toUserFormat(stack.set_points)[0]
        self.assertAlmostEqual(set_point, new_value, 3)
        self.triggerUndo()
        set_point = stack.toUserFormat(stack.set_points)[0]
        self.assertAlmostEqual(old_set_point, set_point, 3)
        self.triggerRedo()
        set_point = stack.toUserFormat(stack.set_points)[0]
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

    def alignSample(self):
        # Test Detector Widget
        self.window.docks.showAlignSample()
        widget = self.getDockedWidget(self.window.docks, DetectorControl.dock_flag)
        self.assertIsNone(self.model.alignment)
        self.editFormControl(widget.x_position, '5.000')
        self.editFormControl(widget.y_position, '6.000')
        self.editFormControl(widget.z_position, '9.000')
        QTest.mouseClick(widget.execute_button, Qt.LeftButton, delay=100)
        self.assertIsNotNone(self.model.alignment)
        self.editFormControl(widget.x_rotation, '20.000')
        self.editFormControl(widget.y_rotation, '90.000')
        self.editFormControl(widget.z_rotation, '-50.000')
        QTest.mouseClick(widget.execute_button, Qt.LeftButton, delay=100)
        self.assertIsNotNone(self.model.alignment)
        self.triggerUndo()
        self.assertIsNone(self.model.alignment)
        self.triggerRedo()
        self.assertIsNotNone(self.model.alignment)

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

        self.window.progress_dialog.show('Testing')
        self.assertTrue(self.window.progress_dialog.isVisible())
        self.window.progress_dialog.close()

        self.window.showAlignmentError()
        self.assertTrue(self.window.alignment_error.isVisible())
        self.window.alignment_error.close()

    def testSettings(self):
        log_filename = 'main.logs'
        config.setup_logging(log_filename)
        self.assertTrue((config.LOG_PATH / log_filename).exists())

        group = config.settings.Group.Simulation.value
        key = enum.Enum('key', {'bool': f'{group}/bool', 'integer': f'{group}/integer', 'float': f'{group}/float',
                                'tuple': f'{group}/tuple', 'no_default': 'no_default'})

        config.__defaults__[key.bool] = True
        config.__defaults__[key.integer] = 150
        config.__defaults__[key.float] = 2.77
        config.__defaults__[key.tuple] = (3, 5, 6)

        self.assertTrue(config.settings.value(key.bool))
        config.settings.setValue(key.bool, False)
        self.assertFalse(config.settings.value(key.bool))

        self.assertEqual(config.settings.value(key.integer), config.__defaults__[key.integer])
        self.assertEqual(config.settings.value(key.float), config.__defaults__[key.float])
        self.assertEqual(config.settings.value(key.tuple), config.__defaults__[key.tuple])
        self.assertEqual(config.settings.value(key.no_default), None)

        config.settings._setting.sync()
        self.assertTrue(self.ini_file.samefile(config.settings.filename()))

        config.settings.reset()
        self.assertTrue(config.settings.value(key.bool))

        self.window.showPreferences()
        self.assertTrue(self.window.preferences.isVisible())
        comboboxes = self.window.preferences.findChildren(QComboBox)

        combo = comboboxes[0]
        current_index = combo.currentIndex()
        new_index = (current_index + 1) % combo.count()
        combo.setCurrentIndex(new_index)
        self.assertTrue(self.window.preferences.accept_button.isEnabled())
        combo.setCurrentIndex(current_index)
        self.assertFalse(self.window.preferences.accept_button.isEnabled())
        combo.setCurrentIndex(new_index)
        stored_key, old_value = combo.property(self.window.preferences.prop_name)
        self.assertEqual(config.settings.value(stored_key), old_value)
        QTest.mouseClick(self.window.preferences.accept_button, Qt.LeftButton, delay=100)
        self.assertNotEqual(config.settings.value(stored_key), old_value)
        self.assertFalse(self.window.preferences.isVisible())
        self.window.showPreferences()
        QTest.mouseClick(self.window.preferences.default_button, Qt.LeftButton, delay=100)
        self.assertFalse(self.window.preferences.isVisible())
        self.window.showPreferences()
        QTest.mouseClick(self.window.preferences.cancel_button, Qt.LeftButton, delay=100)