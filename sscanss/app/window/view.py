import json
import logging
import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError
import webbrowser
from PyQt6 import QtCore, QtGui, QtWidgets
from .presenter import MainWindowPresenter
from .dock_manager import DockManager
from sscanss.__version import __version__, Version
from sscanss.config import settings, path_for, DOCS_URL, UPDATE_URL, RELEASES_URL, load_stylesheet, Themes, Key
from sscanss.app.dialogs import (ProgressDialog, ProjectDialog, Preferences, AlignmentErrorDialog, ScriptExportDialog,
                                 PathLengthPlotter, AboutDialog, CalibrationErrorDialog, InstrumentCoordinatesDialog,
                                 CurveEditor, VolumeLoader)
from sscanss.core.scene import Node, OpenGLRenderer, SceneManager
from sscanss.core.util import (Primitives, Directions, TransformType, PointType, MessageType, Attributes,
                               toggle_action_in_group, StatusBar, FileDialog, MessageReplyType, Worker, IconEngine)

MAIN_WINDOW_TITLE = 'SScanSS 2'


class MainWindow(QtWidgets.QMainWindow):
    """Creates the main view for the sscanss app"""
    def __init__(self):
        super().__init__()

        self.recent_projects = []
        self.loadAppStyleSheet()
        self.presenter = MainWindowPresenter(self)
        window_icon = QtGui.QIcon(path_for('logo.png'))

        self.undo_stack = QtGui.QUndoStack(self)
        self.undo_view = QtWidgets.QUndoView(self.undo_stack)
        self.undo_view.setWindowTitle('Undo History')
        self.undo_view.setWindowIcon(window_icon)
        self.undo_view.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)

        self.gl_widget = OpenGLRenderer(self)
        self.gl_widget.custom_error_handler = self.sceneSizeErrorHandler
        self.setCentralWidget(self.gl_widget)

        self.docks = DockManager(self)
        self.scenes = SceneManager(self.presenter.model, self.gl_widget)
        self.presenter.model.sample_model_updated.connect(self.scenes.updateSampleScene)
        self.presenter.model.instrument_model_updated.connect(self.scenes.updateInstrumentScene)

        self.progress_dialog = ProgressDialog(self)
        self.about_dialog = AboutDialog(self)
        self.updater = Updater(self)
        self.non_modal_dialog = None

        self.createActions()
        self.createMenus()
        self.createToolBar()
        self.createStatusBar()

        self.setWindowTitle(MAIN_WINDOW_TITLE)
        self.setMinimumSize(1024, 900)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.readSettings()
        self.updateMenus()

    def loadAppStyleSheet(self):
        """loads the style sheet"""
        settings.system.setValue(Key.Theme.value, Themes.Light.value)
        if sys.platform == 'darwin':
            style = load_stylesheet("light_theme_mac.css")
        else:
            style = load_stylesheet("style.css")
        self.setStyleSheet(style)

    def createActions(self):
        """Creates the menu and toolbar actions """
        self.new_project_action = QtGui.QAction('&New Project', self)
        self.new_project_action.setStatusTip('Create a new project')
        self.new_project_action.setIcon(QtGui.QIcon(IconEngine('file.png')))
        self.new_project_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        self.new_project_action.triggered.connect(self.showNewProjectDialog)

        self.open_project_action = QtGui.QAction('&Open Project', self)
        self.open_project_action.setStatusTip('Open an existing project')
        self.open_project_action.setIcon(QtGui.QIcon(IconEngine('folder-open.png')))
        self.open_project_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_project_action.triggered.connect(lambda: self.presenter.openProject())

        self.save_project_action = QtGui.QAction('&Save Project', self)
        self.save_project_action.setStatusTip('Save project')
        self.save_project_action.setIcon(QtGui.QIcon(IconEngine('save.png')))
        self.save_project_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_project_action.triggered.connect(lambda: self.presenter.saveProject())

        self.save_as_action = QtGui.QAction('Save &As...', self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.presenter.saveProject(save_as=True))

        self.export_sample_action = QtGui.QAction('Sample', self)
        self.export_sample_action.setStatusTip('Export sample')
        self.export_sample_action.triggered.connect(self.presenter.exportSample)

        self.export_fiducials_action = QtGui.QAction('Fiducial Points', self)
        self.export_fiducials_action.setStatusTip('Export fiducial points')
        self.export_fiducials_action.triggered.connect(lambda: self.presenter.exportPoints(PointType.Fiducial))

        self.export_measurements_action = QtGui.QAction('Measurement Points', self)
        self.export_measurements_action.setStatusTip('Export measurement points')
        self.export_measurements_action.triggered.connect(lambda: self.presenter.exportPoints(PointType.Measurement))

        self.export_vectors_action = QtGui.QAction('Measurement Vectors', self)
        self.export_vectors_action.setStatusTip('Export measurement vectors')
        self.export_vectors_action.triggered.connect(self.presenter.exportVectors)

        self.export_alignment_action = QtGui.QAction('Alignment Matrix', self)
        self.export_alignment_action.setStatusTip('Export alignment matrix')
        self.export_alignment_action.triggered.connect(self.presenter.exportAlignmentMatrix)

        self.export_script_action = QtGui.QAction('Script', self)
        self.export_script_action.setStatusTip('Export script')
        self.export_script_action.triggered.connect(self.showScriptExport)

        self.exit_action = QtGui.QAction('E&xit', self)
        self.exit_action.setStatusTip(f'Quit {MAIN_WINDOW_TITLE}')
        self.exit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)

        # Edit Menu Actions
        self.undo_action = self.undo_stack.createUndoAction(self, '&Undo')
        self.undo_action.setStatusTip('Undo the last action')
        self.undo_action.setIcon(QtGui.QIcon(IconEngine('undo.png')))
        self.undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)

        self.redo_action = self.undo_stack.createRedoAction(self, '&Redo')
        self.redo_action.setStatusTip('Redo the last undone action')
        self.redo_action.setIcon(QtGui.QIcon(IconEngine('redo.png')))
        self.redo_action.setShortcut(QtGui.QKeySequence.StandardKey.Redo)

        self.undo_view_action = QtGui.QAction('Undo &History', self)
        self.undo_view_action.setStatusTip('View undo history')
        self.undo_view_action.triggered.connect(self.undo_view.show)

        self.preferences_action = QtGui.QAction('Preferences', self)
        self.preferences_action.setStatusTip('Change application preferences')
        self.preferences_action.triggered.connect(lambda: self.showPreferences(None))
        self.preferences_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+P'))

        # View Menu Actions
        self.solid_render_action = QtGui.QAction(Node.RenderMode.Solid.value, self)
        self.solid_render_action.setStatusTip('Render sample as solid object')
        self.solid_render_action.setIcon(QtGui.QIcon(IconEngine('solid.png')))
        self.solid_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Solid))
        self.solid_render_action.setCheckable(True)
        self.solid_render_action.setChecked(self.scenes.sample_render_mode is Node.RenderMode.Solid)

        self.line_render_action = QtGui.QAction(Node.RenderMode.Wireframe.value, self)
        self.line_render_action.setStatusTip('Render sample as wireframe object')
        self.line_render_action.setIcon(QtGui.QIcon(IconEngine('wireframe.png')))
        self.line_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Wireframe))
        self.line_render_action.setCheckable(True)
        self.line_render_action.setChecked(self.scenes.sample_render_mode is Node.RenderMode.Wireframe)

        self.blend_render_action = QtGui.QAction(Node.RenderMode.Transparent.value, self)
        self.blend_render_action.setStatusTip('Render sample as transparent object')
        self.blend_render_action.setIcon(QtGui.QIcon(IconEngine('blend.png')))
        self.blend_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Transparent))
        self.blend_render_action.setCheckable(True)
        self.blend_render_action.setChecked(self.scenes.sample_render_mode is Node.RenderMode.Transparent)

        self.render_action_group = QtGui.QActionGroup(self)
        self.render_action_group.addAction(self.solid_render_action)
        self.render_action_group.addAction(self.line_render_action)
        self.render_action_group.addAction(self.blend_render_action)

        self.show_bounding_box_action = QtGui.QAction('Toggle Bounding Box', self)
        self.show_bounding_box_action.setStatusTip('Toggle sample bounding box')
        self.show_bounding_box_action.setIcon(QtGui.QIcon(IconEngine('bounding-box.png')))
        self.show_bounding_box_action.setCheckable(True)
        self.show_bounding_box_action.setChecked(self.gl_widget.show_bounding_box)
        self.show_bounding_box_action.toggled.connect(self.gl_widget.showBoundingBox)

        self.show_coordinate_frame_action = QtGui.QAction('Toggle Coordinate Frame', self)
        self.show_coordinate_frame_action.setStatusTip('Toggle scene coordinate frame')
        self.show_coordinate_frame_action.setIcon(QtGui.QIcon(IconEngine('hide-coordinate-frame.png')))
        self.show_coordinate_frame_action.setCheckable(True)
        self.show_coordinate_frame_action.setChecked(self.gl_widget.show_coordinate_frame)
        self.show_coordinate_frame_action.toggled.connect(self.gl_widget.showCoordinateFrame)

        self.show_fiducials_action = QtGui.QAction('Toggle Fiducial Points', self)
        self.show_fiducials_action.setStatusTip('Show or hide fiducial points')
        self.show_fiducials_action.setIcon(QtGui.QIcon(IconEngine('hide-fiducials.png')))
        action = self.scenes.changeVisibility
        self.show_fiducials_action.toggled.connect(lambda state, a=Attributes.Fiducials: action(a, state))
        self.show_fiducials_action.setCheckable(True)
        self.show_fiducials_action.setChecked(True)

        self.show_measurement_action = QtGui.QAction('Toggle Measurement Points', self)
        self.show_measurement_action.setStatusTip('Show or hide measurement points')
        self.show_measurement_action.setIcon(QtGui.QIcon(IconEngine('hide-measurement.png')))
        self.show_measurement_action.toggled.connect(lambda state, a=Attributes.Measurements: action(a, state))
        self.show_measurement_action.setCheckable(True)
        self.show_measurement_action.setChecked(True)

        self.show_vectors_action = QtGui.QAction('Toggle Measurement Vectors', self)
        self.show_vectors_action.setStatusTip('Show or hide measurement vectors')
        self.show_vectors_action.setIcon(QtGui.QIcon(IconEngine('hide-vectors.png')))
        self.show_vectors_action.toggled.connect(lambda state, a=Attributes.Vectors: action(a, state))
        self.show_vectors_action.setCheckable(True)
        self.show_vectors_action.setChecked(True)

        self.reset_camera_action = QtGui.QAction('Reset View', self)
        self.reset_camera_action.setStatusTip('Reset camera view')
        self.reset_camera_action.triggered.connect(self.gl_widget.resetCamera)
        self.reset_camera_action.setShortcut(QtGui.QKeySequence('Ctrl+0'))

        self.fiducial_manager_action = QtGui.QAction('Fiducial Points', self)
        self.fiducial_manager_action.setStatusTip('Open fiducial point manager')
        self.fiducial_manager_action.triggered.connect(lambda: self.docks.showPointManager(PointType.Fiducial))
        self.fiducial_manager_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+F'))

        self.measurement_manager_action = QtGui.QAction('Measurements Points', self)
        self.measurement_manager_action.setStatusTip('Open measurement point manager')
        self.measurement_manager_action.triggered.connect(lambda: self.docks.showPointManager(PointType.Measurement))
        self.measurement_manager_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+M'))

        self.vector_manager_action = QtGui.QAction('Measurements Vectors', self)
        self.vector_manager_action.setStatusTip('Open measurement vector manager')
        self.vector_manager_action.triggered.connect(self.docks.showVectorManager)
        self.vector_manager_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+V'))

        self.simulation_dialog_action = QtGui.QAction('Simulation Results', self)
        self.simulation_dialog_action.setStatusTip('Open simulation dialog')
        self.simulation_dialog_action.triggered.connect(self.docks.showSimulationResults)
        self.simulation_dialog_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+L'))

        self.sample_properties_dialog_action = QtGui.QAction('Sample Properties', self)
        self.sample_properties_dialog_action.setStatusTip('Open sample properties dialog')
        self.sample_properties_dialog_action.triggered.connect(self.docks.showSampleProperties)
        self.sample_properties_dialog_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+I'))

        self.theme_action = QtGui.QAction('Toggle Theme', self)
        self.theme_action.setStatusTip('Toggle application theme')
        self.theme_action.setIcon(QtGui.QIcon(IconEngine('toggle-theme.png')))
        self.theme_action.triggered.connect(self.toggleTheme)
        self.theme_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+T'))

        # Insert Menu Actions
        self.import_sample_action = QtGui.QAction('File...', self)
        self.import_sample_action.setStatusTip('Import sample from 3D model file')
        self.import_sample_action.triggered.connect(self.presenter.importMesh)

        self.import_nexus_volume_action = QtGui.QAction('&Nexus File...', self)
        self.import_nexus_volume_action.setStatusTip('Import volume from Nexus file using NXtomoproc standard')
        self.import_nexus_volume_action.triggered.connect(self.presenter.loadVolumeFromNexus)

        self.import_tiff_volume_action = QtGui.QAction('&TIFF Files...', self)
        self.import_tiff_volume_action.setStatusTip('Import volume from stacks of TIFF files')
        self.import_tiff_volume_action.triggered.connect(self.showVolumeLoader)

        self.import_fiducial_action = QtGui.QAction('File...', self)
        self.import_fiducial_action.setStatusTip('Import fiducial points from file')
        self.import_fiducial_action.triggered.connect(lambda: self.presenter.importPoints(PointType.Fiducial))

        self.keyin_fiducial_action = QtGui.QAction('Key-in', self)
        self.keyin_fiducial_action.setStatusTip('Input X,Y,Z values of a fiducial point')
        self.keyin_fiducial_action.triggered.connect(lambda: self.docks.showInsertPointDialog(PointType.Fiducial))

        self.import_measurement_action = QtGui.QAction('File...', self)
        self.import_measurement_action.setStatusTip('Import measurement points from file')
        self.import_measurement_action.triggered.connect(lambda: self.presenter.importPoints(PointType.Measurement))

        self.keyin_measurement_action = QtGui.QAction('Key-in', self)
        self.keyin_measurement_action.setStatusTip('Input X,Y,Z values of a measurement point')
        self.keyin_measurement_action.triggered.connect(lambda: self.docks.showInsertPointDialog(PointType.Measurement))

        self.pick_measurement_action = QtGui.QAction('Graphical Selection', self)
        self.pick_measurement_action.setStatusTip('Select measurement points on a cross-section of sample')
        self.pick_measurement_action.triggered.connect(self.docks.showPickPointDialog)

        self.import_measurement_vector_action = QtGui.QAction('File...', self)
        self.import_measurement_vector_action.setStatusTip('Import measurement vectors from file')
        self.import_measurement_vector_action.triggered.connect(self.presenter.importVectors)

        self.vectors_from_angles_action = QtGui.QAction('Euler Angles...', self)
        self.vectors_from_angles_action.setStatusTip('Create measurement vectors using Euler angles')
        self.vectors_from_angles_action.triggered.connect(self.presenter.createVectorsWithEulerAngles)

        self.select_strain_component_action = QtGui.QAction('Select Strain Component', self)
        self.select_strain_component_action.setStatusTip('Specify measurement vector direction')
        self.select_strain_component_action.triggered.connect(self.docks.showInsertVectorDialog)

        self.align_via_pose_action = QtGui.QAction('6D Pose', self)
        self.align_via_pose_action.setStatusTip('Specify position and orientation of sample on instrument')
        self.align_via_pose_action.triggered.connect(self.docks.showAlignSample)

        self.align_via_matrix_action = QtGui.QAction('Transformation Matrix', self)
        self.align_via_matrix_action.triggered.connect(self.presenter.alignSampleWithMatrix)
        self.align_via_matrix_action.setStatusTip('Align sample on instrument with transformation matrix')

        self.align_via_fiducials_action = QtGui.QAction('Fiducials Points', self)
        self.align_via_fiducials_action.triggered.connect(self.presenter.alignSampleWithFiducialPoints)
        self.align_via_fiducials_action.setStatusTip('Align sample on instrument using measured fiducial points')

        self.run_simulation_action = QtGui.QAction('&Run Simulation', self)
        self.run_simulation_action.setStatusTip('Start new simulation')
        self.run_simulation_action.setShortcut('F5')
        self.run_simulation_action.setIcon(QtGui.QIcon(IconEngine('play.png')))
        self.run_simulation_action.triggered.connect(lambda: self.presenter.runSimulation(False))

        self.run_forward_simulation_action = QtGui.QAction('Run with &Offsets...', self)
        self.run_forward_simulation_action.setStatusTip('Start a simulation using a list of joint offsets')
        self.run_forward_simulation_action.setShortcut('Ctrl+F5')
        self.run_forward_simulation_action.setIcon(QtGui.QIcon(IconEngine('play-script.png')))
        self.run_forward_simulation_action.triggered.connect(lambda: self.presenter.runSimulation(True))

        self.stop_simulation_action = QtGui.QAction('&Stop Simulation', self)
        self.stop_simulation_action.setStatusTip('Stop active simulation')
        self.stop_simulation_action.setShortcut('Shift+F5')
        self.stop_simulation_action.setIcon(QtGui.QIcon(IconEngine('stop.png')))
        self.stop_simulation_action.triggered.connect(self.presenter.stopSimulation)

        self.compute_path_length_action = QtGui.QAction('Calculate Path Length', self)
        self.compute_path_length_action.setStatusTip('Enable path length calculation in simulation')
        self.compute_path_length_action.setCheckable(True)
        self.compute_path_length_action.setChecked(False)

        self.check_limits_action = QtGui.QAction('Hardware Limits Check', self)
        self.check_limits_action.setStatusTip('Enable positioning system joint limit checks in simulation')
        self.check_limits_action.setCheckable(True)
        self.check_limits_action.setChecked(True)

        self.show_sim_graphics_action = QtGui.QAction('Show Graphically', self)
        self.show_sim_graphics_action.setStatusTip('Enable graphics rendering in simulation')
        self.show_sim_graphics_action.setCheckable(True)
        self.show_sim_graphics_action.setChecked(True)

        self.check_collision_action = QtGui.QAction('Collision Detection', self)
        self.check_collision_action.setStatusTip('Enable collision detection in simulation')
        self.check_collision_action.setCheckable(True)
        self.check_collision_action.setChecked(False)

        self.show_sim_options_action = QtGui.QAction('Simulation Options', self)
        self.show_sim_options_action.setStatusTip('Change simulation settings')
        self.show_sim_options_action.triggered.connect(lambda: self.showPreferences(settings.Group.Simulation))

        # Instrument Actions
        self.positioning_system_action = QtGui.QAction('Positioning System', self)
        self.positioning_system_action.setStatusTip('Change positioning system settings')
        self.positioning_system_action.triggered.connect(self.docks.showPositionerControl)

        self.jaw_action = QtGui.QAction('Incident Jaws', self)
        self.jaw_action.setStatusTip('Change incident jaws settings')
        self.jaw_action.triggered.connect(self.docks.showJawControl)

        # Help Actions
        self.show_documentation_action = QtGui.QAction('&Documentation', self)
        self.show_documentation_action.setStatusTip('Show online documentation')
        self.show_documentation_action.setShortcut('F1')
        self.show_documentation_action.setIcon(QtGui.QIcon(IconEngine('question.png')))
        self.show_documentation_action.triggered.connect(self.showDocumentation)

        self.check_update_action = QtGui.QAction('&Check for Update', self)
        self.check_update_action.setStatusTip('Check the internet for software updates')
        self.check_update_action.triggered.connect(lambda: self.updater.check())

        self.show_about_action = QtGui.QAction(f'&About {MAIN_WINDOW_TITLE}', self)
        self.show_about_action.setStatusTip(f'About {MAIN_WINDOW_TITLE}')
        self.show_about_action.triggered.connect(self.about_dialog.show)

        # ToolBar Actions
        self.rotate_sample_action = QtGui.QAction('Rotate Sample', self)
        self.rotate_sample_action.setStatusTip('Rotate sample around fixed coordinate frame axis')
        self.rotate_sample_action.setIcon(QtGui.QIcon(IconEngine('rotate.png')))
        self.rotate_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Rotate))

        self.translate_sample_action = QtGui.QAction('Translate Sample', self)
        self.translate_sample_action.setStatusTip('Translate sample along fixed coordinate frame axis')
        self.translate_sample_action.setIcon(QtGui.QIcon(IconEngine('translate.png')))
        self.translate_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Translate))

        self.transform_sample_action = QtGui.QAction('Transform Sample with Matrix', self)
        self.transform_sample_action.setStatusTip('Transform sample with transformation matrix')
        self.transform_sample_action.setIcon(QtGui.QIcon(IconEngine('transform-matrix.png')))
        self.transform_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Custom))

        self.move_origin_action = QtGui.QAction('Move Origin to Sample', self)
        self.move_origin_action.setStatusTip('Translate sample using bounding box')
        self.move_origin_action.setIcon(QtGui.QIcon(IconEngine('origin.png')))
        self.move_origin_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Origin))

        self.plane_align_action = QtGui.QAction('Rotate Sample by Plane Alignment', self)
        self.plane_align_action.setStatusTip('Rotate sample using a selected plane')
        self.plane_align_action.setIcon(QtGui.QIcon(IconEngine('plane-align.png')))
        self.plane_align_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Plane))

        self.toggle_scene_action = QtGui.QAction('Toggle Scene', self)
        self.toggle_scene_action.setStatusTip('Toggle between sample and instrument scene')
        self.toggle_scene_action.setIcon(QtGui.QIcon(IconEngine('exchange.png')))
        self.toggle_scene_action.triggered.connect(self.scenes.toggleScene)
        self.toggle_scene_action.setShortcut(QtGui.QKeySequence('Ctrl+T'))

        self.current_coordinates_action = QtGui.QAction('Instrument Coordinates', self)
        self.current_coordinates_action.setStatusTip('Display fiducials in the instrument coordinate frame')
        self.current_coordinates_action.setIcon(QtGui.QIcon(IconEngine('current-points.png')))
        self.current_coordinates_action.triggered.connect(self.showInstrumentCoordinates)

        self.show_curve_editor_action = QtGui.QAction('Curve Editor', self)
        self.show_curve_editor_action.setStatusTip('Change alpha values for rendering a volume')
        self.show_curve_editor_action.setIcon(QtGui.QIcon(IconEngine('curve.png')))
        self.show_curve_editor_action.triggered.connect(self.showCurveEditor)

    def updateImages(self):
        """Updates the images of the actions """
        self.new_project_action.setIcon(QtGui.QIcon(IconEngine('file.png')))
        self.open_project_action.setIcon(QtGui.QIcon(IconEngine('folder-open.png')))
        self.save_project_action.setIcon(QtGui.QIcon(IconEngine('save.png')))
        self.undo_action.setIcon(QtGui.QIcon(IconEngine('undo.png')))
        self.redo_action.setIcon(QtGui.QIcon(IconEngine('redo.png')))
        self.solid_render_action.setIcon(QtGui.QIcon(IconEngine('solid.png')))
        self.line_render_action.setIcon(QtGui.QIcon(IconEngine('wireframe.png')))
        self.blend_render_action.setIcon(QtGui.QIcon(IconEngine('blend.png')))
        self.show_bounding_box_action.setIcon(QtGui.QIcon(IconEngine('bounding-box.png')))
        self.show_coordinate_frame_action.setIcon(QtGui.QIcon(IconEngine('hide-coordinate-frame.png')))
        self.show_fiducials_action.setIcon(QtGui.QIcon(IconEngine('hide-fiducials.png')))
        self.show_measurement_action.setIcon(QtGui.QIcon(IconEngine('hide-measurement.png')))
        self.show_vectors_action.setIcon(QtGui.QIcon(IconEngine('hide-vectors.png')))
        self.theme_action.setIcon(QtGui.QIcon(IconEngine('toggle-theme.png')))
        self.run_simulation_action.setIcon(QtGui.QIcon(IconEngine('play.png')))
        self.run_forward_simulation_action.setIcon(QtGui.QIcon(IconEngine('play-script.png')))
        self.stop_simulation_action.setIcon(QtGui.QIcon(IconEngine('stop.png')))
        self.show_documentation_action.setIcon(QtGui.QIcon(IconEngine('question.png')))

    def createMenus(self):
        """Creates the main menu and sub menus"""
        main_menu = self.menuBar()
        main_menu.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)

        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        self.recent_menu = file_menu.addMenu('Open &Recent')
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        self.export_menu = file_menu.addMenu('Export...')
        self.export_menu.addAction(self.export_script_action)
        self.export_menu.addSeparator()
        self.export_menu.addAction(self.export_sample_action)
        self.export_menu.addAction(self.export_fiducials_action)
        self.export_menu.addAction(self.export_measurements_action)
        self.export_menu.addAction(self.export_vectors_action)
        self.export_menu.addAction(self.export_alignment_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        file_menu.aboutToShow.connect(self.populateRecentMenu)

        edit_menu = main_menu.addMenu('&Edit')
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addAction(self.undo_view_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.preferences_action)

        view_menu = main_menu.addMenu('&View')
        view_menu.addAction(self.solid_render_action)
        view_menu.addAction(self.line_render_action)
        view_menu.addAction(self.blend_render_action)
        view_menu.addSeparator()
        self.view_from_menu = view_menu.addMenu('View From')
        for index, direction in enumerate(Directions):
            view_from_action = QtGui.QAction(direction.value, self)
            view_from_action.setStatusTip(f'View scene from the {direction.value} axis')
            view_from_action.setShortcut(QtGui.QKeySequence(f'Ctrl+{index+1}'))
            view_from_action.triggered.connect(lambda ignore, d=direction: self.gl_widget.viewFrom(d))
            self.view_from_menu.addAction(view_from_action)

        view_menu.addAction(self.reset_camera_action)
        view_menu.addSeparator()
        view_menu.addAction(self.show_bounding_box_action)
        view_menu.addAction(self.show_fiducials_action)
        view_menu.addAction(self.show_measurement_action)
        view_menu.addAction(self.show_vectors_action)
        view_menu.addAction(self.show_coordinate_frame_action)
        view_menu.addSeparator()
        view_menu.addAction(self.theme_action)
        view_menu.addSeparator()
        self.other_windows_menu = view_menu.addMenu('Other Windows')
        self.other_windows_menu.addAction(self.fiducial_manager_action)
        self.other_windows_menu.addAction(self.measurement_manager_action)
        self.other_windows_menu.addAction(self.vector_manager_action)
        self.other_windows_menu.addAction(self.simulation_dialog_action)
        self.other_windows_menu.addAction(self.sample_properties_dialog_action)

        insert_menu = main_menu.addMenu('&Insert')
        sample_menu = insert_menu.addMenu('Sample')
        sample_menu.addAction(self.import_sample_action)

        self.primitives_menu = sample_menu.addMenu('Primitives')

        for primitive in Primitives:
            add_primitive_action = QtGui.QAction(primitive.value, self)
            add_primitive_action.setStatusTip(f'Add {primitive.value} model as sample')
            add_primitive_action.triggered.connect(lambda ignore, p=primitive: self.docks.showInsertPrimitiveDialog(p))
            self.primitives_menu.addAction(add_primitive_action)

        volume_menu = sample_menu.addMenu('Volume')
        volume_menu.addAction(self.import_nexus_volume_action)
        volume_menu.addAction(self.import_tiff_volume_action)

        fiducial_points_menu = insert_menu.addMenu('Fiducial Points')
        fiducial_points_menu.addAction(self.import_fiducial_action)
        fiducial_points_menu.addAction(self.keyin_fiducial_action)

        measurement_points_menu = insert_menu.addMenu('Measurement Points')
        measurement_points_menu.addAction(self.import_measurement_action)
        measurement_points_menu.addAction(self.keyin_measurement_action)
        measurement_points_menu.addAction(self.pick_measurement_action)

        measurement_vectors_menu = insert_menu.addMenu('Measurement Vectors')
        measurement_vectors_menu.addAction(self.import_measurement_vector_action)
        measurement_vectors_menu.addAction(self.vectors_from_angles_action)
        measurement_vectors_menu.addAction(self.select_strain_component_action)

        self.instrument_menu = main_menu.addMenu('I&nstrument')
        self.change_instrument_menu = self.instrument_menu.addMenu('Change Instrument')
        self.updateChangeInstrumentMenu()

        self.align_sample_menu = self.instrument_menu.addMenu('Align Sample on Instrument')
        self.align_sample_menu.addAction(self.align_via_pose_action)
        self.align_sample_menu.addAction(self.align_via_matrix_action)
        self.align_sample_menu.addAction(self.align_via_fiducials_action)
        self.collimator_action_groups = {}

        simulation_menu = main_menu.addMenu('&Simulation')
        simulation_menu.addAction(self.run_simulation_action)
        simulation_menu.addAction(self.stop_simulation_action)
        simulation_menu.addSeparator()
        simulation_menu.addAction(self.run_forward_simulation_action)
        simulation_menu.addSeparator()
        simulation_menu.addAction(self.check_limits_action)
        simulation_menu.addAction(self.show_sim_graphics_action)
        simulation_menu.addAction(self.compute_path_length_action)
        simulation_menu.addAction(self.check_collision_action)
        simulation_menu.addSeparator()
        simulation_menu.addAction(self.show_sim_options_action)

        help_menu = main_menu.addMenu('&Help')
        help_menu.addAction(self.show_documentation_action)
        help_menu.addSeparator()
        help_menu.addAction(self.check_update_action)
        help_menu.addAction(self.show_about_action)

    def updateMenus(self):
        """Disables the menus when a project is not created and enables menus when a project is created"""
        enable = self.presenter.model.project_data is not None

        self.save_project_action.setEnabled(enable)
        self.save_as_action.setEnabled(enable)
        for action in self.export_menu.actions():
            action.setEnabled(enable)

        self.render_action_group.setEnabled(enable)
        self.show_curve_editor_action.setEnabled(enable)

        for action in self.view_from_menu.actions():
            action.setEnabled(enable)
        self.reset_camera_action.setEnabled(enable)
        self.show_bounding_box_action.setEnabled(enable)
        self.show_coordinate_frame_action.setEnabled(enable)
        self.show_fiducials_action.setEnabled(enable)
        self.show_measurement_action.setEnabled(enable)
        self.show_vectors_action.setEnabled(enable)

        for action in self.other_windows_menu.actions():
            action.setEnabled(enable)

        self.import_sample_action.setEnabled(enable)
        for action in self.primitives_menu.actions():
            action.setEnabled(enable)

        self.import_fiducial_action.setEnabled(enable)
        self.keyin_fiducial_action.setEnabled(enable)

        self.import_measurement_action.setEnabled(enable)
        self.keyin_measurement_action.setEnabled(enable)
        self.pick_measurement_action.setEnabled(enable)

        self.import_measurement_vector_action.setEnabled(enable)
        self.select_strain_component_action.setEnabled(enable)

        self.instrument_menu.setEnabled(enable)

        self.run_simulation_action.setEnabled(enable)
        self.stop_simulation_action.setEnabled(enable)
        self.check_limits_action.setEnabled(enable)
        self.show_sim_graphics_action.setEnabled(enable)
        self.compute_path_length_action.setEnabled(enable)
        self.check_collision_action.setEnabled(enable)

        self.rotate_sample_action.setEnabled(enable)
        self.translate_sample_action.setEnabled(enable)
        self.transform_sample_action.setEnabled(enable)
        self.move_origin_action.setEnabled(enable)
        self.plane_align_action.setEnabled(enable)
        self.toggle_scene_action.setEnabled(enable)
        self.current_coordinates_action.setEnabled(enable)

    def createToolBar(self):
        """Creates the tool bar"""
        # self.toolbar = QtWidgets.QToolBar("ToolBar")
        # self.addToolBar(self.toolbar)
        self.toolbar = self.addToolBar('ToolBar')
        self.toolbar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)
        self.toolbar.setMovable(False)

        self.toolbar.addAction(self.new_project_action)
        self.toolbar.addAction(self.open_project_action)
        self.toolbar.addAction(self.save_project_action)
        self.toolbar.addAction(self.undo_action)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.solid_render_action)
        self.toolbar.addAction(self.line_render_action)
        self.toolbar.addAction(self.blend_render_action)
        self.toolbar.addAction(self.show_curve_editor_action)
        self.toolbar.addAction(self.show_bounding_box_action)

        sub_button = QtWidgets.QToolButton(self)
        sub_button.setIcon(QtGui.QIcon(IconEngine('eye-slash.png')))
        sub_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        sub_button.setToolTip('Show/Hide Elements')
        sub_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        sub_button.addAction(self.show_fiducials_action)
        sub_button.addAction(self.show_measurement_action)
        sub_button.addAction(self.show_vectors_action)
        sub_button.addAction(self.show_coordinate_frame_action)
        self.toolbar.addWidget(sub_button)

        sub_button = QtWidgets.QToolButton(self)
        sub_button.setIcon(QtGui.QIcon(IconEngine('camera.png')))
        sub_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        sub_button.setToolTip('Preset Views')
        sub_button.setMenu(self.view_from_menu)
        self.toolbar.addWidget(sub_button)

        self.toolbar.addSeparator()
        self.toolbar.addAction(self.rotate_sample_action)
        self.toolbar.addAction(self.translate_sample_action)
        self.toolbar.addAction(self.transform_sample_action)
        self.toolbar.addAction(self.move_origin_action)
        self.toolbar.addAction(self.plane_align_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.toggle_scene_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.current_coordinates_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.theme_action)

    def createStatusBar(self):
        """Creates the status bar"""
        sb = StatusBar()
        self.setStatusBar(sb)
        self.instrument_label = QtWidgets.QLabel()
        self.instrument_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.instrument_label.setToolTip('Current Instrument')
        sb.addPermanentWidget(self.instrument_label, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        self.cursor_label = QtWidgets.QLabel()
        self.cursor_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sb.addPermanentWidget(self.cursor_label)

        self.size_label = QtWidgets.QLabel()
        self.size_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sb.addPermanentWidget(self.size_label)

    def closeNonModalDialog(self):
        if self.non_modal_dialog is not None:
            self.non_modal_dialog.close()

    def clearUndoStack(self):
        """Clears the undo stack and ensures stack is cleaned even when stack is empty"""
        if self.undo_stack.count() == 0:
            self.undo_stack.setClean()
        self.undo_stack.clear()

    def readSettings(self):
        """Loads the window geometry and recent projects from settings"""
        self.restoreGeometry(settings.value(settings.Key.Geometry))
        self.recent_projects = settings.value(settings.Key.Recent_Projects)

    def populateRecentMenu(self):
        """Populates the recent project sub-menu"""
        self.recent_menu.clear()
        if self.recent_projects:
            for project in self.recent_projects:
                recent_project_action = QtGui.QAction(project, self)
                recent_project_action.triggered.connect(lambda ignore, p=project: self.presenter.openProject(p))
                self.recent_menu.addAction(recent_project_action)
        else:
            recent_project_action = QtGui.QAction('None', self)
            self.recent_menu.addAction(recent_project_action)

    def closeEvent(self, event):
        if self.presenter.confirmSave():
            settings.system.setValue(settings.Key.Geometry.value, self.saveGeometry())
            if self.recent_projects:
                settings.system.setValue(settings.Key.Recent_Projects.value, self.recent_projects)
            event.accept()
        else:
            event.ignore()

    def resetInstrumentMenu(self):
        """Clears the instrument menu """
        self.instrument_menu.clear()
        self.instrument_menu.addMenu(self.change_instrument_menu)
        self.instrument_menu.addSeparator()
        self.instrument_menu.addAction(self.jaw_action)
        self.instrument_menu.addAction(self.positioning_system_action)
        self.instrument_menu.addSeparator()
        self.instrument_menu.addMenu(self.align_sample_menu)
        self.collimator_action_groups = {}

    def updateChangeInstrumentMenu(self):
        """Populates the instrument menu"""
        self.change_instrument_menu.clear()
        self.change_instrument_action_group = QtGui.QActionGroup(self)
        self.project_file_instrument_action = QtGui.QAction('', self)
        self.change_instrument_action_group.addAction(self.project_file_instrument_action)
        self.change_instrument_menu.addAction(self.project_file_instrument_action)
        self.project_file_instrument_separator = self.change_instrument_menu.addSeparator()
        self.project_file_instrument_action.setCheckable(True)
        self.project_file_instrument_action.setVisible(False)
        self.project_file_instrument_separator.setVisible(False)
        for name in sorted(self.presenter.model.instruments.keys()):
            change_instrument_action = QtGui.QAction(name, self)
            change_instrument_action.setStatusTip(f'Change instrument to {name}')
            change_instrument_action.setCheckable(True)
            change_instrument_action.triggered.connect(lambda ignore, n=name: self.presenter.changeInstrument(n))
            self.change_instrument_action_group.addAction(change_instrument_action)
            self.change_instrument_menu.addAction(change_instrument_action)
        self.toggleActiveInstrument()

    def toggleActiveInstrument(self):
        """Toggles the active instrument in the instrument menu"""
        model = self.presenter.model
        if model.project_data is None or model.instrument is None:
            return

        self.instrument_label.setText(model.instrument.name)
        if model.checkInstrumentVersion():
            toggle_action_in_group(model.instrument.name, self.change_instrument_action_group)
            self.project_file_instrument_action.setVisible(False)
            self.project_file_instrument_separator.setVisible(False)
        else:
            self.project_file_instrument_action.setText(f'{model.instrument.name} (Project)')
            self.project_file_instrument_action.setChecked(True)
            self.project_file_instrument_action.setVisible(True)
            self.project_file_instrument_separator.setVisible(True)

    def addCollimatorMenu(self, detector, collimators, active, menu='Detector', show_more_settings=False):
        """Creates and populates the detector menu with sub menus for each collimator for the detector

        :param detector: name of detector
        :type detector: str
        :param collimators: name of collimators for detector
        :type collimators: List[str]
        :param active: name of active collimator
        :type active: str
        :param menu: name of menu
        :type menu: str
        :param show_more_settings:
        :type show_more_settings: bool
        """
        collimator_menu = QtWidgets.QMenu(menu, self)
        self.instrument_menu.insertMenu(self.jaw_action, collimator_menu)
        action_group = QtGui.QActionGroup(self)
        for name in collimators:
            change_collimator_action = self.__createChangeCollimatorAction(name, active, detector)
            collimator_menu.addAction(change_collimator_action)
            action_group.addAction(change_collimator_action)

        change_collimator_action = self.__createChangeCollimatorAction('None', str(active), detector)
        collimator_menu.addAction(change_collimator_action)
        action_group.addAction(change_collimator_action)
        self.collimator_action_groups[detector] = action_group
        if show_more_settings:
            collimator_menu.addSeparator()
            other_settings_action = QtGui.QAction('Other Settings', self)
            other_settings_action.setStatusTip('Change detector position settings')
            other_settings_action.triggered.connect(lambda ignore, d=detector: self.docks.showDetectorControl(d))
            collimator_menu.addAction(other_settings_action)

    def __createChangeCollimatorAction(self, name, active, detector):
        """Creates the action for changing active collimator

        :param name: name of collimator
        :type name: str
        :param active: name of active collimator
        :type active: str
        :param detector: name of detector
        :type detector: str
        :return: action
        :rtype: QtGui.QAction
        """
        change_collimator_action = QtGui.QAction(name, self)
        change_collimator_action.setStatusTip(f'Change collimator to {name}')
        change_collimator_action.setCheckable(True)
        change_collimator_action.setChecked(active == name)
        change_collimator_action.triggered.connect(
            lambda ignore, n=detector, t=name: self.presenter.changeCollimators(n, t))

        return change_collimator_action

    def showNewProjectDialog(self):
        """Opens the new project dialog"""
        if self.presenter.confirmSave():
            self.closeNonModalDialog()
            project_dialog = ProjectDialog(self.recent_projects, parent=self)
            project_dialog.setModal(True)
            project_dialog.show()

    def showPreferences(self, group=None):
        """Opens the preferences dialog"""
        self.closeNonModalDialog()
        preferences = Preferences(self)
        preferences.setActiveGroup(group)
        preferences.setModal(True)
        preferences.show()

    def toggleTheme(self):
        """Toggles the stylesheet of the app"""
        if settings.system.value(Key.Theme.value) == Themes.Light.value:
            settings.system.setValue(Key.Theme.value, Themes.Dark.value)
            style = load_stylesheet("dark_theme.css")
        else:
            settings.system.setValue(Key.Theme.value, Themes.Light.value)
            if sys.platform == 'darwin':
                style = load_stylesheet("light_theme_mac.css")
            else:
                style = load_stylesheet("style.css")
        self.setStyleSheet(style)
        self.updateImages()

    def showCurveEditor(self):
        """Opens the volume curve editor dialog"""
        if isinstance(self.non_modal_dialog, CurveEditor):
            if self.non_modal_dialog.isHidden():
                self.non_modal_dialog.show()
            return

        self.closeNonModalDialog()
        curve_editor = CurveEditor(self)
        curve_editor.show()
        self.non_modal_dialog = curve_editor

    def showInstrumentCoordinates(self):
        """Opens the instrument coordinates dialog"""
        if isinstance(self.non_modal_dialog, InstrumentCoordinatesDialog):
            if self.non_modal_dialog.isHidden():
                self.non_modal_dialog.show()
            return

        self.closeNonModalDialog()
        instrument_coordinates = InstrumentCoordinatesDialog(self)
        instrument_coordinates.show()
        self.non_modal_dialog = instrument_coordinates

    def showVolumeLoader(self):
        """Opens the volume loader dialog"""
        self.closeNonModalDialog()
        volume_loader = VolumeLoader(self)
        volume_loader.setModal(True)
        volume_loader.show()

    def showAlignmentError(self, indices, enabled, points, transform_result, end_configuration, order_fix=None):
        """Opens the dialog for showing sample alignment errors

        :param indices: N x 1 array containing indices of each point
        :type indices: numpy.ndarray[int]
        :param enabled: N x 1 array containing enabled state of each point
        :type enabled: numpy.ndarray[bool]
        :param points: N X 3 array of measured fiducial points
        :type points: numpy.ndarray[float]
        :param transform_result: initial alignment result
        :type transform_result: TransformResult
        :param end_configuration: final configuration of the positioning system
        :type end_configuration: List[float]
        :param order_fix: suggested indices for wrong order correction
        :type order_fix: Union[numpy.ndarray[int], None]
        """
        self.closeNonModalDialog()
        alignment_error = AlignmentErrorDialog(self, indices, enabled, points, transform_result, end_configuration,
                                               order_fix)
        alignment_error.setModal(True)
        alignment_error.show()

    def showCalibrationError(self, pose_id, fiducial_id, error):
        """Opens the dialog for showing errors from the base computation

        :param pose_id: array of pose index
        :type pose_id: numpy.ndarray
        :param fiducial_id: array of fiducial point index
        :type fiducial_id: numpy.ndarray
        :param error: difference between measured point and computed point
        :type error: numpy.ndarray
        :return: indicates if the results were accepted
        :rtype: bool
        """
        self.closeNonModalDialog()
        calibration_error = CalibrationErrorDialog(self, pose_id, fiducial_id, error)
        return calibration_error.exec() == QtWidgets.QDialog.DialogCode.Accepted

    def showPathLength(self):
        """Opens the path length plotter dialog"""
        simulation = self.presenter.model.simulation
        if simulation is None:
            self.showMessage('There are no simulation results.', MessageType.Information)
            return

        if not simulation.compute_path_length:
            self.showMessage(
                'Path Length computation is not enabled for this simulation.\n'
                'Go to "Simulation > Compute Path Length" to enable it then \nrestart simulation.',
                MessageType.Information)
            return

        self.closeNonModalDialog()
        path_length_plotter = PathLengthPlotter(self)
        path_length_plotter.setModal(True)
        path_length_plotter.show()

    def showScriptExport(self):
        """Shows the dialog for exporting the resulting script from a simulation"""
        simulation = self.presenter.model.simulation
        if simulation is None:
            self.showMessage('There are no simulation results to write in script.', MessageType.Information)
            return

        if simulation.isRunning():
            self.showMessage('Finish or Stop the current simulation before attempting to write script.',
                             MessageType.Information)
            return

        if not simulation.has_valid_result:
            self.showMessage('There are no valid simulation results to write in script.', MessageType.Information)
            return

        self.closeNonModalDialog()
        script_export = ScriptExportDialog(simulation, parent=self)
        script_export.setModal(True)
        script_export.show()

    def showProjectName(self):
        """Displays the project name, save path in the window's title bar """
        project_name = self.presenter.model.project_data['name']
        save_path = self.presenter.model.save_path
        if save_path:
            title = f'{project_name} [{save_path}] - {MAIN_WINDOW_TITLE}'
        else:
            title = f'{project_name} - {MAIN_WINDOW_TITLE}'
        self.setWindowTitle(title)

    def showSaveDialog(self, filters='', current_dir='', title='', select_folder=False):
        """Shows the file dialog for selecting path to save a file to

        :param filters: file filters
        :type filters: str
        :param current_dir: initial path
        :type current_dir: str
        :param title: dialog title
        :type title: str
        :param select_folder: indicates if folder mode is enabled
        :type select_folder: bool
        :return: selected file path
        :rtype: str
        """
        directory = current_dir if current_dir else os.path.splitext(self.presenter.model.save_path)[0]
        if not select_folder:
            path = FileDialog.getSaveFileName(self, title, directory, filters)
        else:
            path = FileDialog.getExistingDirectory(
                self, title, directory,
                QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontResolveSymlinks)

        return path

    def showOpenDialog(self, filters, current_dir='', title=''):
        """Shows the file dialog for selecting files to open

        :param filters: file filters
        :type filters: str
        :param current_dir: initial path
        :type current_dir: str
        :param title: dialog title
        :type title: str
        :return: selected file path
        :rtype: str
        """
        directory = current_dir if current_dir else os.path.dirname(self.presenter.model.save_path)
        filename = FileDialog.getOpenFileName(self, title, directory, filters)
        return filename

    def showMessage(self, message, severity=MessageType.Error):
        """Shows a message with a given severity.

        :param message: user message
        :type message: str
        :param severity: severity of the message
        :type severity: MessageType
        """
        if severity == MessageType.Error:
            QtWidgets.QMessageBox.critical(self, MAIN_WINDOW_TITLE, message)
        elif severity == MessageType.Warning:
            QtWidgets.QMessageBox.warning(self, MAIN_WINDOW_TITLE, message)
        else:
            QtWidgets.QMessageBox.information(self, MAIN_WINDOW_TITLE, message)

    def showSaveDiscardMessage(self, name):
        """Shows a message to confirm if unsaved changes should be saved or discarded.

        :param name: the name of the unsaved project
        :type name: str
        """
        message = 'The document has been modified.\n\n' \
                  f'Do you want to save changes to "{name}"?\t'
        buttons = (QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
                   | QtWidgets.QMessageBox.StandardButton.Cancel)
        reply = QtWidgets.QMessageBox.warning(self, MAIN_WINDOW_TITLE, message, buttons,
                                              QtWidgets.QMessageBox.StandardButton.Cancel)

        if reply == QtWidgets.QMessageBox.StandardButton.Save:
            return MessageReplyType.Save
        elif reply == QtWidgets.QMessageBox.StandardButton.Discard:
            return MessageReplyType.Discard
        else:
            return MessageReplyType.Cancel

    def showSelectChoiceMessage(self, message, choices, default_choice=-1, cancel_choice=None):
        """Shows a message box to allow the user to select one of multiple choices

        :param message: message
        :type message: str
        :param choices: list of choices
        :type choices: List[str]
        :param default_choice: index of default choice
        :type default_choice: int
        :param cancel_choice: index of cancel choice
        :type cancel_choice: Optional[int]
        :return: selected choice
        :rtype: str
        """
        message_box = QtWidgets.QMessageBox(self)
        message_box.setWindowTitle(MAIN_WINDOW_TITLE)
        message_box.setText(message)

        buttons = []
        for index, choice in enumerate(choices):
            buttons.append(QtWidgets.QPushButton(choice))
            if index == cancel_choice:
                role = QtWidgets.QMessageBox.ButtonRole.NoRole
            else:
                role = QtWidgets.QMessageBox.ButtonRole.YesRole
            message_box.addButton(buttons[-1], role)

        message_box.setDefaultButton(buttons[default_choice])
        message_box.exec()

        for index, button in enumerate(buttons):
            if message_box.clickedButton() == button:
                return choices[index]

    def showDocumentation(self):
        """Opens the documentation in the system's default browser"""
        webbrowser.open_new(DOCS_URL)

    def sceneSizeErrorHandler(self):
        """Handles error when the active scene is unreasonably big"""
        msg = (f'The scene is too big the distance from the origin exceeds {self.gl_widget.scene.max_extent}mm.'
               ' The last operation will be undone to revert to previous scene size.')

        self.presenter.notifyError(msg, ValueError(msg))

        # Remove command that caused scene to exceed max size.
        # This hack adds an empty command to remove the bad one.
        self.undo_stack.undo()
        cmd = QtWidgets.QUndoCommand()
        self.undo_stack.push(cmd)
        cmd.setObsolete(True)
        self.undo_stack.undo()


class Updater(QtWidgets.QDialog):
    """Handles checking for software updates

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.startup = False
        self.parent = parent
        self.setFixedSize(400, 200)
        self.setWindowTitle(f'{MAIN_WINDOW_TITLE} Update')

        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        self.stack = QtWidgets.QStackedLayout()
        main_layout.addLayout(self.stack)
        self.stack.addWidget(QtWidgets.QWidget())
        self.stack.addWidget(QtWidgets.QWidget())

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setTextVisible(False)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        message = QtWidgets.QLabel('Checking the Internet for Updates')
        message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        sub_layout = QtWidgets.QVBoxLayout()
        sub_layout.addStretch(1)
        sub_layout.addWidget(progress_bar)
        sub_layout.addWidget(message)
        sub_layout.addStretch(1)
        widget = self.stack.widget(0)
        widget.setLayout(sub_layout)

        self.result = QtWidgets.QLabel('')
        self.result.setWordWrap(True)
        self.result.setOpenExternalLinks(True)
        self.result.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.result.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        checkbox = QtWidgets.QCheckBox('Check for updates on startup')
        checkbox.setChecked(settings.value(settings.Key.Check_Update))
        checkbox.stateChanged.connect(lambda state: settings.system.setValue(settings.Key.Check_Update.value, state ==
                                                                             QtCore.Qt.CheckState.Checked))

        sub_layout = QtWidgets.QVBoxLayout()
        sub_layout.addStretch(1)
        sub_layout.addWidget(self.result)
        sub_layout.addSpacing(10)
        sub_layout.addWidget(checkbox)
        sub_layout.addStretch(1)
        widget = self.stack.widget(1)
        widget.setLayout(sub_layout)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        close_button = QtWidgets.QPushButton('Close')
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        main_layout.addLayout(button_layout)

        self.worker = Worker(self.checkHelper, [])
        self.worker.job_succeeded.connect(self.onSuccess)
        self.worker.job_failed.connect(self.onFailure)

    def check(self, startup=False):
        """Asynchronously checks for new release using the Github release API and notifies the user when
        update is found, not found or an error occurred. When startup is true, the user will
        only be notified if update is found.

        :param startup: flag that indicates the check is at startup
        :type startup: bool
        """
        if startup and not settings.value(settings.Key.Check_Update):
            return

        self.startup = startup
        if not startup:
            self.stack.setCurrentIndex(0)
            self.show()
        self.worker.start()

    def checkHelper(self):
        """Checks for the latest release version on the GitHub repo

        :return: version
        :rtype: str
        """
        with urllib.request.urlopen(UPDATE_URL) as response:
            tag_name = json.loads(response.read()).get('tag_name')

        return tag_name

    def isNewVersions(self, version):
        """

        :param version:
        :type version: str
        :return:
        :rtype: bool
        """
        current_version = __version__
        try:
            new_version = Version.parse(version)
        except ValueError:
            return False

        if new_version.pre_release is not None:
            # For now, new alpha, beta versions will be ignored
            return False

        if new_version.major > current_version.major:
            return True

        if new_version.major == current_version.major and new_version.minor > current_version.minor:
            return True

        if (new_version.major == current_version.major and new_version.minor == current_version.minor
                and new_version.patch > current_version.patch):
            return True

        if (new_version.major == current_version.major and new_version.minor == current_version.minor
                and new_version.patch == current_version.patch and current_version.pre_release is not None):
            # Guarantee, alpha, beta version will receive updated
            return True

        return False

    def onSuccess(self, version):
        """Reports the version found after successful check

        :param version: version tag
        :type version: str
        """

        if self.isNewVersions(version[1:]):
            self.showMessage(f'A new version ({version}) of {MAIN_WINDOW_TITLE} is available. Download '
                             f'the installer from <a href="{RELEASES_URL}">{RELEASES_URL}</a>.<br/><br/>')
        else:
            if self.startup:
                return
            self.showMessage(f'You are running the latest version of {MAIN_WINDOW_TITLE}.<br/><br/>')

    def onFailure(self, exception):
        """Logs and reports error after failed check

        :param exception: exception when checking for update
        :type exception: Union[HTTPError, URLError]
        """
        logging.error('An error occurred while checking for updates', exc_info=exception)
        if self.startup:
            return

        if isinstance(exception, HTTPError):
            self.showMessage(f'You are running the latest version of {MAIN_WINDOW_TITLE}.<br/><br/>')
        elif isinstance(exception, URLError):
            self.showMessage('An error occurred when attempting to connect to the update server. '
                             'Check your internet connection and/or firewall and try again.')

    def showMessage(self, message):
        """Show dialog with given message

        :param message: message to display
        :type message: str
        """
        self.stack.setCurrentIndex(1)
        self.result.setText(message)
        if self.startup:
            self.show()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.terminate()
        event.accept()
