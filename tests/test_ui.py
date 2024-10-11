from contextlib import suppress
import pathlib
import platform
import shutil
import tempfile
from unittest import mock
import numpy as np
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QPoint, QTimer, QSettings
from PyQt6.QtWidgets import QToolBar, QComboBox, QToolButton, QSlider, QMessageBox, QWidget
from OpenGL.plugins import FormatHandler
from sscanss.app.dialogs import (InsertPrimitiveDialog, TransformDialog, InsertPointDialog, PathLengthPlotter,
                                 InsertVectorDialog, VectorManager, PickPointDialog, JawControl, PositionerControl,
                                 DetectorControl, PointManager, SimulationDialog, ScriptExportDialog, ProjectDialog,
                                 Preferences, CalibrationErrorDialog, AlignmentErrorDialog, SampleProperties,
                                 AboutDialog)
from sscanss.app.window.view import MainWindow
import sscanss.config as config
from sscanss.core.instrument import Simulation
from sscanss.core.math import rigid_transform
from sscanss.core.scene import Node
from sscanss.app.widgets.graphics import GraphicsPointItem
from sscanss.core.util import Primitives, PointType, DockFlag, ImageHeader
from tests.helpers import (QTestCase, mouse_drag, mouse_wheel_scroll, click_check_box, edit_line_edit_text, click_table,
                           MessageBoxClicker, create_mock)

WAIT_TIME = 5000

FUNC = Simulation.execute


def wrapped(args):
    import logging

    logging.disable(level=logging.INFO)
    return FUNC(args)


class TestMainWindow(QTestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = pathlib.Path(tempfile.mkdtemp())
        cls.ini_file = cls.data_dir / "settings.ini"

        cls.sys_settings = create_mock(cls,
                                       'sscanss.config.settings.system',
                                       instance=QSettings(str(cls.ini_file), QSettings.Format.IniFormat))
        cls.log_path = create_mock(cls, 'sscanss.config.LOG_PATH', instance=cls.data_dir / "logs")
        FormatHandler("sscanss", "OpenGL.arrays.numpymodule.NumpyHandler", ["sscanss.core.math.matrix.Matrix44"])
        cls.setupWindow()

    @classmethod
    def setupWindow(cls):
        cls.window = MainWindow()
        cls.toolbar = cls.window.findChild(QToolBar)
        cls.model = cls.window.presenter.model
        cls.window.presenter.notifyError = cls.notifyError
        cls.window.show()

    @staticmethod
    def notifyError(message, exception):
        """Logs error and notifies user of them

        :param message: message to display to user and in the log
        :type message: str
        :param exception: exception to log
        :type exception: Exception
        """
        raise Exception(message) from exception

    @classmethod
    def tearDownClass(cls):
        with suppress(RuntimeError):
            cls.window.undo_stack.setClean()
            cls.window.close()
        root_logger = config.logging.getLogger()
        for _ in range(len(root_logger.handlers) - 1):
            handler = root_logger.handlers[-1]
            handler.close()
            root_logger.removeHandler(handler)
        config.logging.shutdown()
        shutil.rmtree(cls.data_dir)

    @classmethod
    def triggerUndo(cls):
        cls.window.undo_action.trigger()
        QTest.qWait(WAIT_TIME // 10)

    @classmethod
    def triggerRedo(cls):
        cls.window.redo_action.trigger()
        QTest.qWait(WAIT_TIME // 10)

    @staticmethod
    def getDockedWidget(dock_manager, dock_flag):
        if dock_flag == DockFlag.Bottom:
            dock = dock_manager.bottom_dock
        else:
            dock = dock_manager.upper_dock

        return dock.widget()

    @mock.patch("sscanss.app.window.view.FileDialog", autospec=True)
    def testMainView(self, file_dialog_mock):
        self.file_dialog_mock = file_dialog_mock
        self.changeSettings()
        self.toggleThemes()
        self.openOtherWindows()

        self.createProject()
        self.addSample()
        self.assertFalse(self.window.gl_widget.show_bounding_box)
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.show_bounding_box_action), Qt.MouseButton.LeftButton)
        self.assertTrue(self.window.gl_widget.show_bounding_box)
        self.assertTrue(self.window.gl_widget.show_coordinate_frame)
        self.window.show_coordinate_frame_action.trigger()
        self.assertFalse(self.window.gl_widget.show_coordinate_frame)
        camera = self.window.gl_widget.scene.camera
        self.window.view_from_menu.actions()[0].trigger()
        self.assertEqual(camera.mode, camera.Projection.Orthographic)
        self.window.reset_camera_action.trigger()
        self.assertEqual(camera.mode, camera.Projection.Perspective)

        self.graphicalPointSelection()

        mouse_drag(self.window.gl_widget)
        mouse_drag(self.window.gl_widget, button=Qt.MouseButton.RightButton)
        mouse_wheel_scroll(self.window.gl_widget, delta=20)
        mouse_wheel_scroll(self.window.gl_widget, delta=-10)

        self.transformSample()

        # render in transparent
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.blend_render_action), Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.scenes.sample_render_mode, Node.RenderMode.Transparent)

        self.keyinFiducials()
        self.keyinPoints()

        # render in wireframe
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.line_render_action), Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.scenes.sample_render_mode, Node.RenderMode.Wireframe)

        self.insertVectors()
        self.jawControl()
        self.pointPicking()
        self.switchInstrument()
        self.positionerControl()
        self.detectorControl()
        self.alignSample()
        self.runSimulation()
        self.checkSaveOnClose()

    def toggleThemes(self):
        # Verifies the theme is toggled by the toolbar action
        light_expected = self.window.themes.loadStylesheet('style.css')\
            if platform.system() != 'Darwin' else self.window.themes.loadStylesheet('mac_style.css')
        light_theme = self.window.styleSheet()
        self.assertEqual(light_theme, light_expected)
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.theme_action), Qt.MouseButton.LeftButton)
        dark_theme = self.window.styleSheet()
        dark_expected = self.window.themes.loadStylesheet('dark_theme.css')
        self.assertEqual(dark_theme, dark_expected)

    def createProject(self):
        self.window.showNewProjectDialog()

        # Test project dialog validation
        project_dialog = self.window.findChild(ProjectDialog)
        self.assertTrue(project_dialog.isVisible())
        self.assertEqual(project_dialog.validator_textbox.text(), "")
        QTest.mouseClick(project_dialog.create_project_button, Qt.MouseButton.LeftButton)
        self.assertNotEqual(project_dialog.validator_textbox.text(), "")
        # Create new project
        QTest.keyClicks(project_dialog.project_name_textbox, "Test")
        for _ in range(project_dialog.instrument_combobox.count()):
            if project_dialog.instrument_combobox.currentText().strip().upper() == "IMAT":
                break
            QTest.keyClick(project_dialog.instrument_combobox, Qt.Key.Key_Down)
        QTest.mouseClick(project_dialog.create_project_button, Qt.MouseButton.LeftButton)
        QTest.keyClick(project_dialog, Qt.Key.Key_Escape)  # should not close until the project is created
        self.assertTrue(project_dialog.isVisible())
        QTest.qWait(WAIT_TIME)  # wait is necessary since instrument is created on another thread
        self.assertEqual(self.model.project_data["name"], "Test")
        self.assertEqual(self.model.instrument.name, "IMAT")

    def addSample(self):
        # Add sample
        self.assertIsNone(self.model.sample)
        self.window.docks.showInsertPrimitiveDialog(Primitives.Tube)
        widget = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertTrue(isinstance(widget, InsertPrimitiveDialog))
        self.assertEqual(widget.primitive, Primitives.Tube)
        self.assertTrue(widget.isVisible())
        edit_line_edit_text(widget.textboxes["inner_radius"].form_lineedit, "10")
        edit_line_edit_text(widget.textboxes["outer_radius"].form_lineedit, "10")
        # equal inner radius and outer radius is an invalid tube so validation should trigger
        self.assertFalse(widget.create_primitive_button.isEnabled())
        # Adds '0' to '10' to make the radius '100'
        QTest.keyClicks(widget.textboxes["outer_radius"].form_lineedit, "0")
        self.assertTrue(widget.create_primitive_button.isEnabled())
        QTest.mouseClick(widget.create_primitive_button, Qt.MouseButton.LeftButton)
        self.assertIsNotNone(self.model.sample)

        self.window.sample_properties_dialog_action.trigger()
        sample_properties_widget = self.getDockedWidget(self.window.docks, SampleProperties.dock_flag)
        self.assertTrue(isinstance(sample_properties_widget, SampleProperties))
        bytes_to_mb_factor = 1 / (1024**2)
        memory = self.model.sample.vertices.nbytes * bytes_to_mb_factor
        self.assertEqual('Memory (MB)', sample_properties_widget.sample_property_table.item(0, 0).text())
        self.assertEqual(f'{memory:.4f}', sample_properties_widget.sample_property_table.item(0, 1).text())
        num_faces = self.model.sample.indices.shape[0] // 3
        self.assertEqual('Faces', sample_properties_widget.sample_property_table.item(1, 0).text())
        self.assertEqual(str(num_faces), sample_properties_widget.sample_property_table.item(1, 1).text())
        num_vertices = self.model.sample.vertices.shape[0]
        self.assertEqual('Vertices', sample_properties_widget.sample_property_table.item(2, 0).text())
        self.assertEqual(str(num_vertices), sample_properties_widget.sample_property_table.item(2, 1).text())

        # Add a second sample
        self.window.docks.showInsertPrimitiveDialog(Primitives.Tube)
        widget_2 = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertIs(widget, widget_2)  # Since a Tube dialog is already open a new widget is not created
        self.assertEqual(widget.primitive, Primitives.Tube)
        self.window.docks.showInsertPrimitiveDialog(Primitives.Cuboid)
        widget_2 = self.getDockedWidget(self.window.docks, InsertPrimitiveDialog.dock_flag)
        self.assertIsNot(widget, widget_2)
        old_vertex_count = len(self.model.sample.vertices)

        memory = self.model.sample.vertices.nbytes * bytes_to_mb_factor
        self.assertEqual('Memory (MB)', sample_properties_widget.sample_property_table.item(0, 0).text())
        self.assertEqual(f'{memory:.4f}', sample_properties_widget.sample_property_table.item(0, 1).text())
        num_faces = self.model.sample.indices.shape[0] // 3
        self.assertEqual('Faces', sample_properties_widget.sample_property_table.item(1, 0).text())
        self.assertEqual(str(num_faces), sample_properties_widget.sample_property_table.item(1, 1).text())
        num_vertices = self.model.sample.vertices.shape[0]
        self.assertEqual('Vertices', sample_properties_widget.sample_property_table.item(2, 0).text())
        self.assertEqual(str(num_vertices), sample_properties_widget.sample_property_table.item(2, 1).text())

        with MessageBoxClicker('combine', timeout=100):  # click first button in message box
            QTest.mouseClick(widget_2.create_primitive_button, Qt.MouseButton.LeftButton)
            QTest.qWait(WAIT_TIME // 10)
            self.assertGreater(len(self.model.sample.vertices), old_vertex_count)

    def transformSample(self):
        # Transform Sample
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [0.0, 0.0, 0.0], decimal=5)

        QTest.mouseClick(self.toolbar.widgetForAction(self.window.translate_sample_action), Qt.MouseButton.LeftButton)
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        QTest.keyClick(widget.tool.y_position.form_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(widget.tool.y_position.form_lineedit, Qt.Key.Key_Delete)
        self.assertFalse(widget.tool.execute_button.isEnabled())

        QTest.keyClicks(widget.tool.y_position.form_lineedit, "100")
        self.assertTrue(widget.tool.execute_button.isEnabled())
        QTest.mouseClick(widget.tool.execute_button, Qt.MouseButton.LeftButton)
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [0.0, 0.0, 0.0], decimal=5)
        self.triggerRedo()
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)

        QTest.mouseClick(self.toolbar.widgetForAction(self.window.rotate_sample_action), Qt.MouseButton.LeftButton)
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        QTest.keyClick(widget.tool.z_rotation.form_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(widget.tool.z_rotation.form_lineedit, Qt.Key.Key_Delete)
        self.assertFalse(widget.tool.execute_button.isEnabled())

        QTest.keyClicks(widget.tool.z_rotation.form_lineedit, "90")
        self.assertTrue(widget.tool.execute_button.isEnabled())
        QTest.mouseClick(widget.tool.execute_button, Qt.MouseButton.LeftButton)
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [-100.0, 0.0, 0.0], decimal=5)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [0.0, 100.0, 0.0], decimal=5)
        self.triggerRedo()
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [-100.0, 0.0, 0.0], decimal=5)

        QTest.mouseClick(self.toolbar.widgetForAction(self.window.transform_sample_action), Qt.MouseButton.LeftButton)

        QTest.mouseClick(self.toolbar.widgetForAction(self.window.move_origin_action), Qt.MouseButton.LeftButton)
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        for i in range(widget.tool.move_combobox.count()):
            QTest.keyClick(widget.tool.move_combobox, Qt.Key.Key_Down)

        for i in range(widget.tool.ignore_combobox.count()):
            QTest.keyClick(widget.tool.ignore_combobox, Qt.Key.Key_Down)

        QTest.mouseClick(widget.tool.execute_button, Qt.MouseButton.LeftButton)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(self.model.sample.bounding_box.center, [-100.0, 0.0, 0.0], decimal=5)
        self.triggerRedo()

        QTest.mouseClick(self.toolbar.widgetForAction(self.window.plane_align_action), Qt.MouseButton.LeftButton)
        widget = self.getDockedWidget(self.window.docks, TransformDialog.dock_flag)
        for i in range(widget.tool.plane_combobox.count()):
            QTest.keyClick(widget.tool.plane_combobox, Qt.Key.Key_Down)

        QTest.mouseClick(widget.tool.execute_button, Qt.MouseButton.LeftButton)

        QTest.mouseClick(widget.tool.pick_button, Qt.MouseButton.LeftButton)
        QTest.mouseClick(self.window.gl_widget, Qt.MouseButton.LeftButton)
        QTest.mouseClick(widget.tool.select_button, Qt.MouseButton.LeftButton)

    def graphicalPointSelection(self):
        self.window.pick_measurement_action.trigger()
        dialog = self.getDockedWidget(self.window.docks, PickPointDialog.dock_flag)
        manager = dialog.findChild(PointManager)

        # Add three points
        QTest.mouseClick(dialog.point_selector, Qt.MouseButton.LeftButton)
        for i in range(3):
            self.assertEqual(self.model.measurement_points.size, i)
            if i <= 3:
                QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton)
            else:
                break
            QTest.mouseClick(dialog.execute_button, Qt.MouseButton.LeftButton)
        self.assertEqual(len([item for item in dialog.scene.items() if isinstance(item, GraphicsPointItem)]), 3)

        dialog.tabs.setCurrentIndex(3)
        self.assertTrue(manager.isVisible())
        self.assertEqual(manager.point_type, PointType.Measurement)

        find_highlighted = lambda items: {
            item.id: item.highlighted
            for item in items if isinstance(item, GraphicsPointItem) and item.fixed
        }

        # Sequentially select the points from the table and check they are highlighted
        for i in range(3):
            click_table(manager.table_view, 0, i, x_offset=5, y_offset=10)
            self.assertTrue(find_highlighted(dialog.scene.items())[i])

        # Move slider down through plane of cross-section
        dialog.tabs.setCurrentIndex(0)
        edit_line_edit_text(dialog.plane_lineedit, "-20")
        QTest.mouseClick(dialog.add_ref_plane, Qt.MouseButton.LeftButton)
        self.assertEqual(len(dialog.reference_planes), 1)
        self.assertIs(dialog.reference_planes[-1], dialog.plane)
        self.assertEqual(len(self.window.scenes.plane_entity.reference_plane_offsets), 1)
        edit_line_edit_text(dialog.plane_lineedit, "-10")
        QTest.mouseClick(dialog.add_ref_plane, Qt.MouseButton.LeftButton)
        self.assertIs(dialog.reference_planes[-1], dialog.plane)
        self.assertEqual(len(dialog.reference_planes), 2)
        self.assertIsNot(dialog.reference_planes[0], dialog.reference_planes[1])
        self.assertEqual(len(self.window.scenes.plane_entity.reference_plane_offsets), 2)
        QTest.mouseClick(dialog.clear_ref_plane, Qt.MouseButton.LeftButton)
        self.assertEqual(len(dialog.reference_planes), 0)
        self.assertEqual(len(self.window.scenes.plane_entity.reference_plane_offsets), 0)
        edit_line_edit_text(dialog.plane_lineedit, "0")
        QTest.mouseClick(dialog.point_selector, Qt.MouseButton.LeftButton)

        # Add a point
        QTest.mouseClick(dialog.point_selector, Qt.MouseButton.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 3)
        QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton)
        QTest.mouseClick(dialog.execute_button, Qt.MouseButton.LeftButton)
        self.assertEqual(len([item for item in dialog.scene.items() if isinstance(item, GraphicsPointItem)]), 4)

        # Highlight the new point
        dialog.tabs.setCurrentIndex(3)
        click_table(manager.table_view, 0, 3, x_offset=5, y_offset=10)
        self.assertTrue(find_highlighted(dialog.scene.items())[3])
        self.assertFalse(self.window.gl_widget.show_annotations)
        self.window.show_measurement_labels_action.trigger()
        self.assertTrue(self.window.gl_widget.show_annotations)
        state = [item.show_label for item in dialog.scene.items() if isinstance(item, GraphicsPointItem) and item.fixed]
        self.assertEqual(state, [True] * len(state))
        self.window.show_measurement_labels_action.trigger()
        self.assertFalse(self.window.gl_widget.show_annotations)
        state = [item.show_label for item in dialog.scene.items() if isinstance(item, GraphicsPointItem) and item.fixed]
        self.assertEqual(state, [False] * len(state))

        # Move slider upwards through cross-section, adding two more points
        dialog.tabs.setCurrentIndex(0)
        QTest.mouseClick(dialog.point_selector, Qt.MouseButton.LeftButton)
        for i in range(2):
            n = 4 + i
            self.assertEqual(self.model.measurement_points.size, n)
            if i <= 2:
                QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton)
            else:
                break
            QTest.mouseClick(dialog.execute_button, Qt.MouseButton.LeftButton)
        self.assertEqual(len([item for item in dialog.scene.items() if isinstance(item, GraphicsPointItem)]), 6)

        # Check the new points are highlighted when clicking the table
        dialog.tabs.setCurrentIndex(3)
        for i in range(2):
            n = 4 + i
            click_table(manager.table_view, 0, n, x_offset=5, y_offset=10)
            self.assertTrue(find_highlighted(dialog.scene.items())[n])

        self.window.presenter.deletePoints(list(range(len(dialog.parent_model.measurement_points))),
                                           PointType.Measurement)

    def keyinFiducials(self):
        # Add Fiducial Points
        self.window.keyin_fiducial_action.trigger()
        widget = self.getDockedWidget(self.window.docks, InsertPointDialog.dock_flag)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key.Key_Delete)
        self.assertFalse(widget.execute_button.isEnabled())

        QTest.keyClicks(widget.z_axis.form_lineedit, "100")
        self.assertTrue(widget.execute_button.isEnabled())
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        QTest.keyClick(widget.x_axis.form_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(widget.x_axis.form_lineedit, "50")
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        self.triggerUndo()
        self.assertEqual(self.model.fiducials.size, 1)
        self.triggerRedo()
        self.assertEqual(self.model.fiducials.size, 2)

        # Test Point Manager
        widget = self.getDockedWidget(self.window.docks, PointManager.dock_flag)
        self.assertTrue(widget.isVisible())
        self.assertEqual(widget.point_type, PointType.Fiducial)
        x_pos = widget.table_view.columnViewportPosition(0) + 5
        y_pos = widget.table_view.rowViewportPosition(1) + 10
        pos = QPoint(x_pos, y_pos)
        QTest.mouseClick(widget.table_view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
        QTest.mouseClick(widget.move_up_button, Qt.MouseButton.LeftButton)
        QTest.qWait(WAIT_TIME // 20)
        QTest.mouseClick(widget.move_down_button, Qt.MouseButton.LeftButton)
        QTest.qWait(WAIT_TIME // 20)

        QTest.mouseDClick(widget.table_view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
        QTest.keyClicks(widget.table_view.viewport().focusWidget(), "100")
        QTest.keyClick(widget.table_view.viewport().focusWidget(), Qt.Key.Key_Enter)
        QTest.qWait(WAIT_TIME // 20)
        np.testing.assert_array_almost_equal(self.model.fiducials[1].points, [100.0, 0.0, 100.0], decimal=3)
        self.triggerUndo()
        np.testing.assert_array_almost_equal(self.model.fiducials[1].points, [50.0, 0.0, 100.0], decimal=3)
        QTest.qWait(WAIT_TIME // 20)

        QTest.mouseClick(widget.delete_button, Qt.MouseButton.LeftButton)
        QTest.qWait(WAIT_TIME // 20)
        self.assertEqual(self.model.fiducials.size, 1)
        self.triggerUndo()
        self.assertEqual(self.model.fiducials.size, 2)

    def keyinPoints(self):
        # Add Measurement Points
        self.window.keyin_measurement_action.trigger()
        widget = self.getDockedWidget(self.window.docks, InsertPointDialog.dock_flag)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(widget.z_axis.form_lineedit, Qt.Key.Key_Delete)
        self.assertFalse(widget.execute_button.isEnabled())

        QTest.keyClicks(widget.z_axis.form_lineedit, "10")
        self.assertTrue(widget.execute_button.isEnabled())
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)

        QTest.keyClick(widget.x_axis.form_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(widget.x_axis.form_lineedit, "20")
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
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

        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        QTest.keyClicks(widget.detector_combobox, detector_names[1][0], delay=50)
        QTest.keyClick(widget.component_combobox, Qt.Key.Key_Down)
        click_check_box(widget.reverse_checkbox)
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        QTest.qWait(WAIT_TIME // 5)  # wait is necessary since vectors are created on another thread

        mv = widget.parent_model.measurement_vectors
        self.assertEqual(mv.shape, (2, 6, 1))
        np.testing.assert_array_almost_equal(mv[0, :, 0], [1, 0, 0, 0, -1, 0], decimal=5)

        QTest.keyClicks(widget.alignment_combobox, "a")
        # QTest.mouseClick(widget.component_combobox, Qt.MouseButton.LeftButton, delay=100)
        QTest.keyClick(widget.component_combobox, Qt.Key.Key_Down)
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        QTest.qWait(WAIT_TIME // 5)

        QTest.keyClicks(widget.component_combobox, "k")
        QTest.keyClicks(widget.detector_combobox, detector_names[0][0], delay=50)
        edit_line_edit_text(widget.x_axis.form_lineedit, "1.0")
        edit_line_edit_text(widget.y_axis.form_lineedit, "1.0")
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        QTest.qWait(WAIT_TIME // 5)

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
            QTest.keyClick(widget.plane_combobox, Qt.Key.Key_Down)

        mouse_drag(widget.plane_slider, QPoint(), QPoint(10, 0))

        edit_line_edit_text(widget.plane_lineedit, "-200.01")
        self.assertTrue(widget.scene.disabled)
        # self.assertAlmostEqual(widget.cross_section_rect.height(), 50, 3)
        # self.assertAlmostEqual(widget.cross_section_rect.width(), 200, 3)
        QTest.keyClick(widget.plane_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(widget.plane_lineedit, Qt.Key.Key_Delete)
        QTest.keyClicks(widget.plane_lineedit, "-10")
        QTest.keyClick(widget.plane_lineedit, Qt.Key.Key_Enter)
        self.assertFalse(widget.scene.disabled)
        # self.assertAlmostEqual(widget.cross_section_rect.height(), 86.634, 3)
        # self.assertAlmostEqual(widget.cross_section_rect.width(), 200, 3)

        widget.tabs.setCurrentIndex(2)
        click_check_box(widget.show_grid_checkbox)
        self.assertTrue(widget.view.show_grid)
        click_check_box(widget.snap_select_to_grid_checkbox)
        self.assertTrue(widget.view.snap_to_grid)
        self.assertTrue(widget.grid_widget.isVisible())

        combo = widget.grid_widget.findChild(QComboBox)
        grid_type = widget.view.grid.type
        QTest.keyClick(combo, Qt.Key.Key_Down)
        QTest.qWait(WAIT_TIME // 100)  # Delay allow the grid to render
        self.assertNotEqual(grid_type, widget.view.grid.type)
        QTest.keyClick(combo, Qt.Key.Key_Up)
        QTest.qWait(WAIT_TIME // 100)  # Delay allow the grid to render
        self.assertEqual(grid_type, widget.view.grid.type)

        self.assertFalse(widget.snap_anchor_widget.isVisible())
        self.assertFalse(widget.view.object_snap_tool.enabled)
        click_check_box(widget.snap_object_to_grid_checkbox)
        self.assertTrue(widget.snap_anchor_widget.isVisible())
        self.assertTrue(widget.view.object_snap_tool.enabled)
        self.assertAlmostEqual(widget.view.object_anchor.x(), 0, 3)
        self.assertAlmostEqual(widget.view.object_anchor.y(), 2000, 3)

        expected = [[-2000.0, 1133.6551], [2000.0, 1133.6551], [-2000.0, 2866.3449], [2000.0, 2866.3449]]
        for i in range(widget.snap_anchor_combobox.count() - 1):
            QTest.keyClick(widget.snap_anchor_combobox, Qt.Key.Key_Down)
            self.assertAlmostEqual(widget.view.object_anchor.x(), expected[i][0], 3)
            self.assertAlmostEqual(widget.view.object_anchor.y(), expected[i][1], 3)
        click_check_box(widget.snap_object_to_grid_checkbox)

        widget.tabs.setCurrentIndex(1)
        QTest.mouseClick(widget.point_selector, Qt.MouseButton.LeftButton)
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 2)
        widget.scene.disabled = True
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton)
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 2)
        widget.scene.disabled = False
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton)
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.model.measurement_points.size, 3)

        widget.tabs.setCurrentIndex(1)
        QTest.mouseClick(widget.line_selector, Qt.MouseButton.LeftButton)
        self.assertTrue(widget.line_tool_widget.isVisible())
        edit_line_edit_text(widget.line_point_count_spinbox, '3')
        expected_count = len(widget.scene.items()) + 3
        mouse_drag(viewport)
        self.assertEqual(len(widget.scene.items()), expected_count)

        QTest.mouseClick(widget.area_selector, Qt.MouseButton.LeftButton)
        self.assertFalse(widget.line_tool_widget.isVisible())
        self.assertTrue(widget.area_tool_widget.isVisible())

        edit_line_edit_text(widget.area_x_spinbox, '4')
        edit_line_edit_text(widget.area_y_spinbox, '5')
        expected_count = len(widget.scene.items()) + 20
        mouse_drag(viewport)
        self.assertEqual(len(widget.scene.items()), expected_count)
        QTest.mouseClick(widget.object_selector, Qt.MouseButton.LeftButton)
        self.assertFalse(widget.line_tool_widget.isVisible())
        self.assertFalse(widget.area_tool_widget.isVisible())
        mouse_drag(viewport)
        selected_count = len(widget.scene.selectedItems())
        QTest.keyClick(viewport, Qt.Key.Key_Delete)
        self.assertEqual(len(widget.scene.items()), expected_count - selected_count)

        QTest.keyClick(widget.plane_lineedit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(widget.plane_lineedit, Qt.Key.Key_Delete)
        QTest.keyClicks(widget.plane_lineedit, "-12")
        QTest.keyClick(widget.plane_lineedit, Qt.Key.Key_Enter)
        mouse_drag(viewport, QPoint(), QPoint(20, 0), button=Qt.MouseButton.MiddleButton)
        QTest.mouseClick(widget.area_selector, Qt.MouseButton.LeftButton)
        edit_line_edit_text(widget.area_x_spinbox, '2')
        edit_line_edit_text(widget.area_y_spinbox, '2')
        expected_count = len(widget.scene.items()) + 4
        QTest.mouseClick(widget.key_in_button, Qt.MouseButton.LeftButton)
        self.assertIsNone(widget.scene.outline_item)
        edit_line_edit_text(widget.start_x_spinbox, '-2')
        edit_line_edit_text(widget.start_y_spinbox, '60')
        edit_line_edit_text(widget.stop_x_spinbox, '2')
        edit_line_edit_text(widget.stop_y_spinbox, '64')
        self.assertIn(widget.scene.outline_item, widget.scene.items())
        buttons = widget.position_widget.findChildren(QToolButton)
        accept_button, clear_button = buttons if buttons[0].toolTip().startswith('Accept') else buttons[::-1]
        QTest.mouseClick(clear_button, Qt.MouseButton.LeftButton)
        self.assertIsNone(widget.scene.outline_item)
        edit_line_edit_text(widget.stop_x_spinbox, '2')
        edit_line_edit_text(widget.stop_y_spinbox, '64')
        self.assertIn(widget.scene.outline_item, widget.scene.items())
        QTest.mouseClick(widget.key_in_button, Qt.MouseButton.LeftButton)
        self.assertIsNone(widget.scene.outline_item)
        QTest.mouseClick(widget.key_in_button, Qt.MouseButton.LeftButton)
        edit_line_edit_text(widget.stop_x_spinbox, '2')
        edit_line_edit_text(widget.stop_y_spinbox, '64')

        QTest.mouseClick(accept_button, Qt.MouseButton.LeftButton)
        points = [(2, 100), (0, 100), (2, 64), (0, 64)]
        transform = widget.scene.transform.inverted()[0]
        for point, item in zip(points, widget.scene.items()):
            pp = transform.map(item.pos()) / widget.sample_scale
            self.assertAlmostEqual(pp.x(), point[0], 3)
            self.assertAlmostEqual(pp.y(), point[1], 3)
        self.assertEqual(len(widget.scene.items()), expected_count)

        self.assertFalse(widget.view.has_foreground)
        QTest.mouseClick(widget.help_button, Qt.MouseButton.LeftButton)
        QTest.qWait(WAIT_TIME // 100)  # Delay allow the grid to render
        self.assertTrue(widget.view.has_foreground and not widget.view.show_help)

        QTest.mouseClick(widget.reset_button, Qt.MouseButton.LeftButton)
        self.assertTrue(widget.view.scene().transform.isIdentity())
        mouse_drag(viewport, button=Qt.MouseButton.MiddleButton)
        self.assertTrue(widget.view.scene().transform.isTranslating())
        self.assertFalse(widget.view.scene().transform.isRotating())
        mouse_drag(viewport, button=Qt.MouseButton.RightButton)
        # QTransform type is always True for translation when rotation is True
        self.assertTrue(widget.view.scene().transform.isTranslating())
        self.assertTrue(widget.view.scene().transform.isRotating())
        widget.view.resetTransform()
        self.assertTrue(widget.view.transform().isIdentity())
        self.assertFalse(widget.view.transform().isScaling())
        mouse_wheel_scroll(viewport)
        self.assertTrue(widget.view.transform().isScaling())
        mouse_wheel_scroll(viewport, delta=-10)
        self.assertTrue(widget.view.transform().isIdentity())
        QTest.mouseClick(widget.reset_button, Qt.MouseButton.LeftButton)
        self.assertTrue(widget.view.scene().transform.isIdentity())
        self.assertNotIn(widget.scene.bounds_item, widget.scene.items())
        QTest.mouseClick(widget.bounds_button, Qt.MouseButton.LeftButton)
        self.assertIn(widget.scene.bounds_item, widget.scene.items())

    def switchInstrument(self):
        # switch instruments
        self.assertNotEqual(self.window.undo_stack.count(), 0)
        with MessageBoxClicker('proceed', timeout=200):  # click first button in message box
            self.window.presenter.changeInstrument("ENGIN-X")
            QTest.qWait(WAIT_TIME)
            self.assertEqual(self.window.undo_stack.count(), 0)
            self.assertEqual(self.model.project_data["name"], "Test")
            self.assertEqual(self.model.instrument.name, "ENGIN-X")

        self.assertIs(self.window.scenes.active_scene, self.window.scenes.sample_scene)
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.toggle_scene_action), Qt.MouseButton.LeftButton)
        self.assertIs(self.window.scenes.active_scene, self.window.scenes.instrument_scene)
        QTest.mouseClick(self.toolbar.widgetForAction(self.window.toggle_scene_action), Qt.MouseButton.LeftButton)
        self.assertIs(self.window.scenes.active_scene, self.window.scenes.sample_scene)

    def jawControl(self):
        # Test incident jaws Dialog and change jaw position
        self.window.docks.showJawControl()
        widget = self.getDockedWidget(self.window.docks, JawControl.dock_flag)
        jaw_form = widget.position_form_group.form_controls[0]
        jaw = self.model.instrument.jaws.positioner
        new_value = jaw.links[0].lower_limit + (jaw.links[0].offset - jaw.links[0].lower_limit) / 2
        edit_line_edit_text(jaw_form.form_lineedit, f"{new_value}")
        QTest.mouseClick(widget.move_jaws_button, Qt.MouseButton.LeftButton)
        set_point = self.model.instrument.jaws.positioner.set_points[0]
        self.assertAlmostEqual(set_point, new_value, 3)

        edit_line_edit_text(jaw_form.form_lineedit, f"{jaw.links[0].lower_limit - 1}")
        self.assertFalse(jaw_form.valid)
        self.assertFalse(widget.move_jaws_button.isEnabled())
        QTest.mouseClick(jaw_form.extra[0], Qt.MouseButton.LeftButton)
        self.assertTrue(jaw_form.valid)
        self.assertTrue(widget.move_jaws_button.isEnabled())
        self.triggerUndo()
        self.assertFalse(jaw_form.valid)
        self.assertFalse(widget.move_jaws_button.isEnabled())

        # Change aperture of the jaw
        aperture_form = widget.aperture_form_group.form_controls
        edit_line_edit_text(aperture_form[0].form_lineedit, "5.000")
        edit_line_edit_text(aperture_form[1].form_lineedit, "6.000")
        old_aperture = self.model.instrument.jaws.aperture
        QTest.mouseClick(widget.change_aperture_button, Qt.MouseButton.LeftButton)
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
        QTest.keyClick(widget.stack_combobox, Qt.Key.Key_Down)
        self.assertNotEqual(self.model.instrument.positioning_stack.name, positioner_name)
        self.triggerUndo()
        self.assertEqual(self.model.instrument.positioning_stack.name, positioner_name)

        form = widget.positioner_form_controls[0]
        stack = self.model.instrument.positioning_stack
        index = stack.order[0]
        new_value = stack.links[index].upper_limit - (stack.links[index].upper_limit - stack.links[index].offset) / 2
        edit_line_edit_text(form.form_lineedit, f"{new_value}")

        form = widget.positioner_form_controls[1]
        QTest.mouseClick(form.extra[0], Qt.MouseButton.LeftButton)
        self.triggerUndo()
        QTest.mouseClick(form.extra[1], Qt.MouseButton.LeftButton)
        self.triggerUndo()
        old_set_point = stack.toUserFormat(stack.set_points)[0]
        self.window.scenes.switchToSampleScene()
        QTest.mouseClick(widget.move_joints_button, Qt.MouseButton.LeftButton)
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
        # Test Sample Alignment
        self.window.docks.showAlignSample()
        widget = self.getDockedWidget(self.window.docks, DetectorControl.dock_flag)
        self.assertIsNone(self.model.alignment)
        edit_line_edit_text(widget.x_position.form_lineedit, "5.000")
        edit_line_edit_text(widget.y_position.form_lineedit, "6.000")
        edit_line_edit_text(widget.z_position.form_lineedit, "9.000")
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton, delay=100)
        self.assertIsNotNone(self.model.alignment)
        edit_line_edit_text(widget.x_rotation.form_lineedit, "20.000")
        edit_line_edit_text(widget.y_rotation.form_lineedit, "90.000")
        edit_line_edit_text(widget.z_rotation.form_lineedit, "-50.000")
        QTest.mouseClick(widget.execute_button, Qt.MouseButton.LeftButton, delay=100)
        self.assertIsNotNone(self.model.alignment)
        self.triggerUndo()
        self.assertIsNone(self.model.alignment)
        self.triggerRedo()
        self.assertIsNotNone(self.model.alignment)

    def runSimulation(self):
        self.model.alignment = self.model.alignment.identity()
        self.window.check_collision_action.setChecked(True)
        self.window.check_limits_action.setChecked(False)
        self.window.compute_path_length_action.setChecked(True)

        Simulation.execute = wrapped
        self.window.run_simulation_action.trigger()
        self.assertIsNotNone(self.model.simulation)
        QTest.qWait(WAIT_TIME // 5)
        self.assertTrue(self.model.simulation.isRunning())

        QTest.qWait(WAIT_TIME * 5)
        self.assertFalse(self.model.simulation.isRunning())
        self.assertEqual(len(self.model.simulation.results), 6)

        widget = self.getDockedWidget(self.window.docks, SimulationDialog.dock_flag)
        self.assertEqual(len(widget.result_list.panes), 6)
        self.assertEqual(widget.result_counts[widget.ResultKey.Good], 2)
        self.assertEqual(widget.result_counts[widget.ResultKey.Warn], 4)
        self.assertEqual(widget.result_counts[widget.ResultKey.Fail], 0)
        self.assertEqual(widget.result_counts[widget.ResultKey.Skip], 0)
        QTest.mouseClick(widget.filter_button_group.button(2), Qt.MouseButton.LeftButton)
        self.assertEqual([pane.isHidden() for pane in widget.result_list.panes].count(True), 0)
        QTest.mouseClick(widget.filter_button_group.button(0), Qt.MouseButton.LeftButton)
        self.assertEqual([pane.isHidden() for pane in widget.result_list.panes].count(True), 2)
        QTest.mouseClick(widget.filter_button_group.button(1), Qt.MouseButton.LeftButton)
        self.assertEqual([pane.isHidden() for pane in widget.result_list.panes].count(True), 6)
        QTest.mouseClick(widget.filter_button_group.button(0), Qt.MouseButton.LeftButton)
        self.assertEqual([pane.isHidden() for pane in widget.result_list.panes].count(True), 4)

        QTest.mouseClick(widget.path_length_button, Qt.MouseButton.LeftButton)
        path_length_plotter = self.window.findChild(PathLengthPlotter)
        self.assertTrue(path_length_plotter.isVisible())
        path_length_plotter.close()
        self.assertFalse(path_length_plotter.isVisible())

        QTest.mouseClick(widget.export_button, Qt.MouseButton.LeftButton)
        script_exporter = self.window.findChild(ScriptExportDialog)
        self.assertTrue(script_exporter.isVisible())
        script_exporter.close()
        self.assertFalse(script_exporter.isVisible())

        self.window.fiducial_manager_action.trigger()
        widget = self.getDockedWidget(self.window.docks, PointManager.dock_flag)
        self.assertEqual(widget.point_type, PointType.Fiducial)
        widget = self.getDockedWidget(self.window.docks, SimulationDialog.dock_flag)
        self.window.simulation_dialog_action.trigger()
        self.assertFalse(widget.simulation.isRunning())
        self.assertEqual(len(widget.result_list.panes), 6)

    def openOtherWindows(self):
        self.window.show_about_action.trigger()
        self.assertIsInstance(self.window.non_modal_dialog, AboutDialog)
        about_dialog = self.window.non_modal_dialog
        self.assertTrue(about_dialog.isVisible())
        header = about_dialog.findChild(ImageHeader)
        QTest.mouseClick(header.close_button, Qt.MouseButton.LeftButton, delay=100)
        self.assertFalse(about_dialog.isVisible())
        self.assertEqual(about_dialog.result(), about_dialog.DialogCode.Rejected)

        # Test the Recent project menu
        self.window.recent_projects = []
        self.assertTrue(self.window.recent_menu.isEmpty())
        self.window.populateRecentMenu()
        self.assertEqual(len(self.window.recent_menu.actions()), 1)
        self.assertEqual(self.window.recent_menu.actions()[0].text(), "None")
        self.window.recent_projects = [
            "c://test.hdf",
            "c://test2.hdf",
            "c://test3.hdf",
            "c://test4.hdf",
            "c://test5.hdf",
            "c://test6.hdf",
            "c://test7.hdf",
            "c://test8.hdf",
        ]
        self.window.populateRecentMenu()
        self.assertEqual(len(self.window.recent_menu.actions()), 8)

        self.window.undo_stack.setClean()
        self.window.show_about_action.trigger()
        self.assertIsInstance(self.window.non_modal_dialog, AboutDialog)
        self.window.new_project_action.trigger()
        self.assertIsInstance(self.window.non_modal_dialog, ProjectDialog)
        project_dialog = self.window.non_modal_dialog
        # update recent project in the show method
        self.assertEqual(project_dialog.list_widget.count(), 6)
        QTest.keyClick(project_dialog, Qt.Key.Key_Escape)
        self.assertFalse(project_dialog.isVisible())

        self.window.undo_view_action.trigger()
        self.assertTrue(self.window.undo_view.isVisible())
        self.window.undo_view.close()
        self.assertFalse(self.window.undo_view.isVisible())

        self.window.progress_dialog.showMessage("Testing")
        self.assertTrue(self.window.progress_dialog.isVisible())
        QTest.keyClick(project_dialog, Qt.Key.Key_Escape)
        self.assertTrue(self.window.progress_dialog.isVisible())
        self.window.progress_dialog.close()
        self.assertFalse(self.window.progress_dialog.isVisible())

        indices = np.array([0, 1, 2, 3])
        enabled = np.array([True, True, True, True])
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        transform_result = rigid_transform(points, points)
        end_q = [0.0] * 4
        order_fix = [3, 2, 1, 0]

        self.window.showAlignmentError(indices, enabled, points, transform_result, end_q, order_fix)
        alignment_error = self.window.findChild(AlignmentErrorDialog)
        self.assertTrue(alignment_error.isVisible())
        alignment_error.close()
        self.assertFalse(alignment_error.isVisible())

        pose_id = np.array([1, 2, 3])
        fiducial_id = np.array([3, 2, 1])
        error = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        QTimer.singleShot(WAIT_TIME // 5, lambda: self.window.findChild(CalibrationErrorDialog).accept())
        self.assertTrue(self.window.showCalibrationError(pose_id, fiducial_id, error))

    def changeSettings(self):
        log_filename = "main.logs"
        config.setup_logging(log_filename)
        self.assertTrue((config.LOG_PATH / log_filename).exists())

        self.assertTrue(config.settings.value(config.settings.Key.Align_First))
        config.settings.setValue(config.settings.Key.Align_First, False, True)
        self.assertFalse(config.settings.value(config.settings.Key.Align_First))
        config.settings.setValue(config.settings.Key.Align_First, "true", True)
        self.assertTrue(config.settings.value(config.settings.Key.Align_First))
        config.settings.setValue(config.settings.Key.Align_First, -2, True)
        self.assertTrue(config.settings.value(config.settings.Key.Align_First))

        item = config.settings.default(config.settings.Key.Local_Max_Eval)
        self.assertEqual(config.settings.value(config.settings.Key.Local_Max_Eval), item.default)
        config.settings.setValue(config.settings.Key.Local_Max_Eval, item.limits[1] + 1, True)
        self.assertEqual(config.settings.value(config.settings.Key.Local_Max_Eval), item.default)
        config.settings.setValue(config.settings.Key.Local_Max_Eval, item.limits[0] - 1, True)
        self.assertEqual(config.settings.value(config.settings.Key.Local_Max_Eval), item.default)
        config.settings.setValue(config.settings.Key.Local_Max_Eval, item.limits[1] - 1, True)
        self.assertEqual(config.settings.value(config.settings.Key.Local_Max_Eval), item.limits[1] - 1)

        item = config.settings.default(config.settings.Key.Angular_Stop_Val)
        self.assertEqual(config.settings.value(config.settings.Key.Angular_Stop_Val), item.default)
        config.settings.setValue(config.settings.Key.Angular_Stop_Val, item.limits[1] + 1, True)
        self.assertEqual(config.settings.value(config.settings.Key.Angular_Stop_Val), item.default)
        config.settings.setValue(config.settings.Key.Angular_Stop_Val, item.limits[0] - 1, True)
        self.assertEqual(config.settings.value(config.settings.Key.Angular_Stop_Val), item.default)
        config.settings.setValue(config.settings.Key.Angular_Stop_Val, item.limits[1] - 1, True)
        self.assertEqual(config.settings.value(config.settings.Key.Angular_Stop_Val), item.limits[1] - 1)

        item = config.settings.default(config.settings.Key.Fiducial_Colour)
        self.assertEqual(config.settings.value(config.settings.Key.Fiducial_Colour), item.default)
        config.settings.setValue(config.settings.Key.Fiducial_Colour, (2, 3, 4, 5), True)
        self.assertEqual(config.settings.value(config.settings.Key.Fiducial_Colour), item.default)
        config.settings.setValue(config.settings.Key.Fiducial_Colour, (2, 3, 4), True)
        self.assertEqual(config.settings.value(config.settings.Key.Fiducial_Colour), item.default)
        config.settings.setValue(config.settings.Key.Fiducial_Colour, ("h", "1.0", "1.0", "1.0"), True)
        self.assertEqual(config.settings.value(config.settings.Key.Fiducial_Colour), item.default)
        config.settings.setValue(config.settings.Key.Fiducial_Colour, ("1.0", "1.0", "1.0", "1.0"), True)
        self.assertEqual(config.settings.value(config.settings.Key.Fiducial_Colour), (1, 1, 1, 1))
        config.settings.setValue(config.settings.Key.Fiducial_Colour, (2, 3, 4, 5), True)
        self.assertEqual(config.settings.value(config.settings.Key.Fiducial_Colour), item.default)

        item = config.settings.default(config.settings.Key.Geometry)
        self.assertEqual(config.settings.value(config.settings.Key.Geometry), item.default)
        config.settings.setValue(config.settings.Key.Geometry, "12345", True)
        self.assertEqual(config.settings.value(config.settings.Key.Geometry), item.default)
        config.settings.setValue(config.settings.Key.Geometry, bytearray(b"12345"), True)
        self.assertEqual(config.settings.value(config.settings.Key.Geometry), bytearray(b"12345"))

        item = config.settings.default(config.settings.Key.Recent_Projects)
        self.assertEqual(config.settings.value(config.settings.Key.Recent_Projects), item.default)
        config.settings.setValue(config.settings.Key.Recent_Projects, "name", True)
        self.assertEqual(config.settings.value(config.settings.Key.Recent_Projects), ["name"])
        config.settings.setValue(config.settings.Key.Recent_Projects, ["name", "other"], True)
        self.assertEqual(config.settings.value(config.settings.Key.Recent_Projects), ["name", "other"])

        config.settings.system.sync()
        self.assertTrue(self.ini_file.samefile(config.settings.filename()))

        config.settings.setValue(config.settings.Key.Align_First, False, True)
        config.settings.reset()
        self.assertFalse(config.settings.value(config.settings.Key.Align_First))
        config.settings.reset(True)
        self.assertTrue(config.settings.value(config.settings.Key.Align_First))
        config.settings.setValue(config.settings.Key.Align_First, False)
        self.assertNotEqual(config.settings.value(config.settings.Key.Align_First),
                            config.settings.system.value(config.settings.Key.Align_First.value))

        self.window.showPreferences()
        preferences = self.window.findChild(Preferences)
        self.assertTrue(preferences.isVisible())
        sliders = preferences.findChildren(QSlider)

        slider = sliders[0]
        current_value = slider.value()
        new_value = current_value + 1
        slider.setValue(new_value)
        self.assertTrue(preferences.accept_button.isEnabled())
        slider.setValue(current_value)
        self.assertFalse(preferences.accept_button.isEnabled())
        slider.setValue(new_value)
        stored_key, old_value = slider.property(preferences.prop_name)
        self.assertEqual(config.settings.value(stored_key), old_value)
        QTest.mouseClick(preferences.accept_button, Qt.MouseButton.LeftButton, delay=100)
        self.assertNotEqual(config.settings.value(stored_key), old_value)
        self.assertFalse(preferences.isVisible())
        QTest.qWait(WAIT_TIME // 50)
        self.window.showPreferences()
        preferences = self.window.findChild(Preferences)
        self.assertTrue(preferences.isVisible())
        QTest.mouseClick(preferences.reset_button, Qt.MouseButton.LeftButton, delay=100)
        self.assertFalse(preferences.isVisible())
        QTest.qWait(WAIT_TIME // 50)
        self.window.presenter.model.project_data = None
        self.window.showPreferences()
        preferences = self.window.findChild(Preferences)
        self.assertTrue(preferences.isVisible())
        QTest.mouseClick(preferences.cancel_button, Qt.MouseButton.LeftButton, delay=100)
        self.assertFalse(preferences.isVisible())
        QTest.qWait(WAIT_TIME // 50)

    def checkSaveOnClose(self):
        # QMessageBox standard buttons can have different text depending on OS so, we get text from button
        box = QMessageBox()
        box.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                               | QMessageBox.StandardButton.Cancel)
        cancel_button_text = box.button(QMessageBox.StandardButton.Cancel).text()
        discard_button_text = box.button(QMessageBox.StandardButton.Discard).text()
        save_button_text = box.button(QMessageBox.StandardButton.Save).text()
        box.deleteLater()

        self.assertTrue(self.window.isVisible())
        with MessageBoxClicker(cancel_button_text, timeout=100):
            self.window.exit_action.trigger()
            QTest.qWait(WAIT_TIME // 100)
        self.assertTrue(self.window.isVisible())

        save_path = self.data_dir / 'file.h5'
        self.file_dialog_mock.getSaveFileName.return_value = str(save_path)
        self.assertFalse(save_path.is_file())
        with MessageBoxClicker(save_button_text, timeout=100):
            self.window.exit_action.trigger()
            QTest.qWait(WAIT_TIME // 100)
        self.assertTrue((self.data_dir / 'file.h5').is_file())

        self.setupWindow()
        self.window.presenter.openProject(save_path)
        QTest.qWait(WAIT_TIME)

        self.assertTrue(self.window.undo_stack.isClean())
        self.window.presenter.alignSampleWithPose([0] * 6)
        self.assertFalse(self.window.undo_stack.isClean())
        self.assertFalse(self.window.presenter.can_discard)
        with MessageBoxClicker(discard_button_text, timeout=100):
            self.window.exit_action.trigger()
            QTest.qWait(WAIT_TIME // 100)
        self.assertTrue(self.window.presenter.can_discard)
