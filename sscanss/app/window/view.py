import os
import webbrowser
from PyQt5 import QtCore, QtGui, QtWidgets
from .presenter import MainWindowPresenter, MessageReplyType
from .dock_manager import DockManager
from .scene_manager import SceneManager
from sscanss.config import settings, path_for, DOCS_URL, __version__, UPDATE_URL, RELEASES_URL
from sscanss.app.dialogs import (ProgressDialog, ProjectDialog, Preferences, AlignmentErrorDialog,
                                 SampleExportDialog, ScriptExportDialog, PathLengthPlotter, AboutDialog,
                                 CalibrationErrorDialog)
from sscanss.core.scene import Node, OpenGLRenderer
from sscanss.core.util import (Primitives, Directions, TransformType, PointType, MessageSeverity, Attributes,
                               toggleActionInGroup, StatusBar, FileDialog)

MAIN_WINDOW_TITLE = 'SScanSS 2'


class MainWindow(QtWidgets.QMainWindow):
    """Creates the main view for the sscanss app"""
    def __init__(self):
        super().__init__()

        self.recent_projects = []
        self.presenter = MainWindowPresenter(self)
        window_icon = QtGui.QIcon(":/images/logo.ico")

        self.undo_stack = QtWidgets.QUndoStack(self)
        self.undo_view = QtWidgets.QUndoView(self.undo_stack)
        self.undo_view.setWindowTitle('Undo History')
        self.undo_view.setWindowIcon(window_icon)
        self.undo_view.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

        self.gl_widget = OpenGLRenderer(self)
        self.gl_widget.custom_error_handler = self.sceneSizeErrorHandler
        self.setCentralWidget(self.gl_widget)

        self.docks = DockManager(self)
        self.scenes = SceneManager(self)
        self.scenes.changeRenderMode(Node.RenderMode.Transparent)
        self.progress_dialog = ProgressDialog(self)
        self.about_dialog = AboutDialog(self)
        self.updater = Updater(self)

        self.createActions()
        self.createMenus()
        self.createToolBar()
        self.createStatusBar()

        self.setWindowTitle(MAIN_WINDOW_TITLE)
        self.setWindowIcon(window_icon)
        self.setMinimumSize(1024, 900)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.readSettings()
        self.updateMenus()

    def createActions(self):
        """Creates the menu and toolbar actions """
        self.new_project_action = QtWidgets.QAction('&New Project', self)
        self.new_project_action.setStatusTip('Create a new project')
        self.new_project_action.setIcon(QtGui.QIcon(path_for('file.png')))
        self.new_project_action.setShortcut(QtGui.QKeySequence.New)
        self.new_project_action.triggered.connect(self.showNewProjectDialog)

        self.open_project_action = QtWidgets.QAction('&Open Project', self)
        self.open_project_action.setStatusTip('Open an existing project')
        self.open_project_action.setIcon(QtGui.QIcon(path_for('folder-open.png')))
        self.open_project_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_project_action.triggered.connect(lambda: self.openProject())

        self.save_project_action = QtWidgets.QAction('&Save Project', self)
        self.save_project_action.setStatusTip('Save project')
        self.save_project_action.setIcon(QtGui.QIcon(path_for('save.png')))
        self.save_project_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_project_action.triggered.connect(lambda: self.presenter.saveProject())

        self.save_as_action = QtWidgets.QAction('Save &As...', self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.presenter.saveProject(save_as=True))

        self.export_samples_action = QtWidgets.QAction('Samples', self)
        self.export_samples_action.setStatusTip('Export samples')
        self.export_samples_action.triggered.connect(self.presenter.exportSample)

        self.export_fiducials_action = QtWidgets.QAction('Fiducial Points', self)
        self.export_fiducials_action.setStatusTip('Export fiducial points')
        self.export_fiducials_action.triggered.connect(lambda: self.presenter.exportPoints(PointType.Fiducial))

        self.export_measurements_action = QtWidgets.QAction('Measurement Points', self)
        self.export_measurements_action.setStatusTip('Export measurement points')
        self.export_measurements_action.triggered.connect(lambda: self.presenter.exportPoints(PointType.Measurement))

        self.export_vectors_action = QtWidgets.QAction('Measurement Vectors', self)
        self.export_vectors_action.setStatusTip('Export measurement vectors')
        self.export_vectors_action.triggered.connect(self.presenter.exportVectors)

        self.export_alignment_action = QtWidgets.QAction('Alignment Matrix', self)
        self.export_alignment_action.setStatusTip('Export alignment matrix')
        self.export_alignment_action.triggered.connect(self.presenter.exportAlignmentMatrix)

        self.export_script_action = QtWidgets.QAction('Script', self)
        self.export_script_action.setStatusTip('Export script')
        self.export_script_action.triggered.connect(self.showScriptExport)

        self.exit_action = QtWidgets.QAction('E&xit', self)
        self.exit_action.setStatusTip(f'Quit {MAIN_WINDOW_TITLE}')
        self.exit_action.setShortcut(QtGui.QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)

        # Edit Menu Actions
        self.undo_action = self.undo_stack.createUndoAction(self, '&Undo')
        self.undo_action.setStatusTip('Undo the last action')
        self.undo_action.setIcon(QtGui.QIcon(path_for('undo.png')))
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)

        self.redo_action = self.undo_stack.createRedoAction(self, '&Redo')
        self.redo_action.setStatusTip('Redo the last undone action')
        self.redo_action.setIcon(QtGui.QIcon(path_for('redo.png')))
        self.redo_action.setShortcut(QtGui.QKeySequence.Redo)

        self.undo_view_action = QtWidgets.QAction('Undo &History', self)
        self.undo_view_action.setStatusTip('View undo history')
        self.undo_view_action.triggered.connect(self.undo_view.show)

        self.preferences_action = QtWidgets.QAction('Preferences', self)
        self.preferences_action.setStatusTip('Change application preferences')
        self.preferences_action.triggered.connect(lambda: self.showPreferences(None))
        self.preferences_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+P'))

        # View Menu Actions
        self.solid_render_action = QtWidgets.QAction(Node.RenderMode.Solid.value, self)
        self.solid_render_action.setStatusTip('Render sample as solid object')
        self.solid_render_action.setIcon(QtGui.QIcon(path_for('solid.png')))
        self.solid_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Solid))
        self.solid_render_action.setCheckable(True)

        self.line_render_action = QtWidgets.QAction(Node.RenderMode.Wireframe.value, self)
        self.line_render_action.setStatusTip('Render sample as wireframe object')
        self.line_render_action.setIcon(QtGui.QIcon(path_for('wireframe.png')))
        self.line_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Wireframe))
        self.line_render_action.setCheckable(True)

        self.blend_render_action = QtWidgets.QAction(Node.RenderMode.Transparent.value, self)
        self.blend_render_action.setStatusTip('Render sample as transparent object')
        self.blend_render_action.setIcon(QtGui.QIcon(path_for('blend.png')))
        self.blend_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Transparent))
        self.blend_render_action.setCheckable(True)
        self.blend_render_action.setChecked(True)

        self.render_action_group = QtWidgets.QActionGroup(self)
        self.render_action_group.addAction(self.solid_render_action)
        self.render_action_group.addAction(self.line_render_action)
        self.render_action_group.addAction(self.blend_render_action)

        self.show_bounding_box_action = QtWidgets.QAction('Toggle Bounding Box', self)
        self.show_bounding_box_action.setStatusTip('Toggle sample bounding box')
        self.show_bounding_box_action.setIcon(QtGui.QIcon(path_for('boundingbox.png')))
        self.show_bounding_box_action.setCheckable(True)
        self.show_bounding_box_action.setChecked(self.gl_widget.show_bounding_box)
        self.show_bounding_box_action.toggled.connect(self.gl_widget.showBoundingBox)

        self.show_coordinate_frame_action = QtWidgets.QAction('Toggle Coordinate Frame', self)
        self.show_coordinate_frame_action.setStatusTip('Toggle scene coordinate frame')
        self.show_coordinate_frame_action.setIcon(QtGui.QIcon(path_for('hide_coordinate_frame.png')))
        self.show_coordinate_frame_action.setCheckable(True)
        self.show_coordinate_frame_action.setChecked(self.gl_widget.show_coordinate_frame)
        self.show_coordinate_frame_action.toggled.connect(self.gl_widget.showCoordinateFrame)

        self.show_fiducials_action = QtWidgets.QAction('Toggle Fiducial Points', self)
        self.show_fiducials_action.setStatusTip('Show or hide fiducial points')
        self.show_fiducials_action.setIcon(QtGui.QIcon(path_for('hide_fiducials.png')))
        self.show_fiducials_action.setCheckable(True)
        self.show_fiducials_action.setChecked(True)
        action = self.scenes.changeVisibility
        self.show_fiducials_action.toggled.connect(lambda state, a=Attributes.Fiducials: action(a, state))

        self.show_measurement_action = QtWidgets.QAction('Toggle Measurement Points', self)
        self.show_measurement_action.setStatusTip('Show or hide measurement points')
        self.show_measurement_action.setIcon(QtGui.QIcon(path_for('hide_measurement.png')))
        self.show_measurement_action.setCheckable(True)
        self.show_measurement_action.setChecked(True)
        self.show_measurement_action.toggled.connect(lambda state, a=Attributes.Measurements: action(a, state))

        self.show_vectors_action = QtWidgets.QAction('Toggle Measurement Vectors', self)
        self.show_vectors_action.setStatusTip('Show or hide measurement vectors')
        self.show_vectors_action.setIcon(QtGui.QIcon(path_for('hide_vectors.png')))
        self.show_vectors_action.setCheckable(True)
        self.show_vectors_action.setChecked(True)
        self.show_vectors_action.toggled.connect(lambda state, a=Attributes.Vectors: action(a, state))

        self.reset_camera_action = QtWidgets.QAction('Reset View', self)
        self.reset_camera_action.setStatusTip('Reset camera view')
        self.reset_camera_action.triggered.connect(self.gl_widget.resetCamera)
        self.reset_camera_action.setShortcut(QtGui.QKeySequence('Ctrl+0'))

        self.sample_manager_action = QtWidgets.QAction('Samples', self)
        self.sample_manager_action.setStatusTip('Open sample manager')
        self.sample_manager_action.triggered.connect(self.docks.showSampleManager)
        self.sample_manager_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+A'))

        self.fiducial_manager_action = QtWidgets.QAction('Fiducial Points', self)
        self.fiducial_manager_action.setStatusTip('Open fiducial point manager')
        self.fiducial_manager_action.triggered.connect(lambda: self.docks.showPointManager(PointType.Fiducial))
        self.fiducial_manager_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+F'))

        self.measurement_manager_action = QtWidgets.QAction('Measurements Points', self)
        self.measurement_manager_action.setStatusTip('Open measurement point manager')
        self.measurement_manager_action.triggered.connect(lambda: self.docks.showPointManager(PointType.Measurement))
        self.measurement_manager_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+M'))

        self.vector_manager_action = QtWidgets.QAction('Measurements Vectors', self)
        self.vector_manager_action.setStatusTip('Open measurement vector manager')
        self.vector_manager_action.triggered.connect(self.docks.showVectorManager)
        self.vector_manager_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+V'))

        self.simulation_dialog_action = QtWidgets.QAction('Simulation Results', self)
        self.simulation_dialog_action.setStatusTip('Open simulation dialog')
        self.simulation_dialog_action.triggered.connect(self.docks.showSimulationResults)
        self.simulation_dialog_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+L'))

        # Insert Menu Actions
        self.import_sample_action = QtWidgets.QAction('File...', self)
        self.import_sample_action.setStatusTip('Import sample from 3D model file')
        self.import_sample_action.triggered.connect(self.presenter.importSample)

        self.import_fiducial_action = QtWidgets.QAction('File...', self)
        self.import_fiducial_action.setStatusTip('Import fiducial points from file')
        self.import_fiducial_action.triggered.connect(lambda: self.presenter.importPoints(PointType.Fiducial))

        self.keyin_fiducial_action = QtWidgets.QAction('Key-in', self)
        self.keyin_fiducial_action.setStatusTip('Input X,Y,Z values of a fiducial point')
        self.keyin_fiducial_action.triggered.connect(lambda: self.docks.showInsertPointDialog(PointType.Fiducial))

        self.import_measurement_action = QtWidgets.QAction('File...', self)
        self.import_measurement_action.setStatusTip('Import measurement points from file')
        self.import_measurement_action.triggered.connect(lambda: self.presenter.importPoints(PointType.Measurement))

        self.keyin_measurement_action = QtWidgets.QAction('Key-in', self)
        self.keyin_measurement_action.setStatusTip('Input X,Y,Z values of a measurement point')
        self.keyin_measurement_action.triggered.connect(lambda: self.docks.showInsertPointDialog(PointType.Measurement))

        self.pick_measurement_action = QtWidgets.QAction('Graphical Selection', self)
        self.pick_measurement_action.setStatusTip('Select measurement points on a cross-section of sample')
        self.pick_measurement_action.triggered.connect(self.docks.showPickPointDialog)

        self.import_measurement_vector_action = QtWidgets.QAction('File...', self)
        self.import_measurement_vector_action.setStatusTip('Import measurement vectors from file')
        self.import_measurement_vector_action.triggered.connect(self.presenter.importVectors)

        self.select_strain_component_action = QtWidgets.QAction('Select Strain Component', self)
        self.select_strain_component_action.setStatusTip('Specify measurement vector direction')
        self.select_strain_component_action.triggered.connect(self.docks.showInsertVectorDialog)

        self.align_via_pose_action = QtWidgets.QAction('6D Pose', self)
        self.align_via_pose_action.setStatusTip('Specify position and orientation of sample on instrument')
        self.align_via_pose_action.triggered.connect(self.docks.showAlignSample)

        self.align_via_matrix_action = QtWidgets.QAction('Transformation Matrix', self)
        self.align_via_matrix_action.triggered.connect(self.presenter.alignSampleWithMatrix)
        self.align_via_matrix_action.setStatusTip('Align sample on instrument with transformation matrix')

        self.align_via_fiducials_action = QtWidgets.QAction('Fiducials Points', self)
        self.align_via_fiducials_action.triggered.connect(self.presenter.alignSampleWithFiducialPoints)
        self.align_via_fiducials_action.setStatusTip('Align sample on instrument using measured fiducial points')

        self.run_simulation_action = QtWidgets.QAction('&Run Simulation', self)
        self.run_simulation_action.setStatusTip('Start new simulation')
        self.run_simulation_action.setShortcut('F5')
        self.run_simulation_action.setIcon(QtGui.QIcon(path_for('play.png')))
        self.run_simulation_action.triggered.connect(self.presenter.runSimulation)

        self.stop_simulation_action = QtWidgets.QAction('&Stop Simulation', self)
        self.stop_simulation_action.setStatusTip('Stop active simulation')
        self.stop_simulation_action.setShortcut('Shift+F5')
        self.stop_simulation_action.setIcon(QtGui.QIcon(path_for('stop.png')))
        self.stop_simulation_action.triggered.connect(self.presenter.stopSimulation)

        self.compute_path_length_action = QtWidgets.QAction('Calculate Path Length', self)
        self.compute_path_length_action.setStatusTip('Enable path length calculation in simulation')
        self.compute_path_length_action.setCheckable(True)
        self.compute_path_length_action.setChecked(False)

        self.check_limits_action = QtWidgets.QAction('Hardware Limits Check', self)
        self.check_limits_action.setStatusTip('Enable positioning system joint limit checks in simulation')
        self.check_limits_action.setCheckable(True)
        self.check_limits_action.setChecked(True)

        self.show_sim_graphics_action = QtWidgets.QAction('Show Graphically', self)
        self.show_sim_graphics_action.setStatusTip('Enable graphics rendering in simulation')
        self.show_sim_graphics_action.setCheckable(True)
        self.show_sim_graphics_action.setChecked(True)

        self.check_collision_action = QtWidgets.QAction('Collision Detection', self)
        self.check_collision_action.setStatusTip('Enable collision detection in simulation')
        self.check_collision_action.setCheckable(True)
        self.check_collision_action.setChecked(False)

        self.show_sim_options_action = QtWidgets.QAction('Simulation Options', self)
        self.show_sim_options_action.setStatusTip('Change simulation settings')
        self.show_sim_options_action.triggered.connect(lambda: self.showPreferences(settings.Group.Simulation))

        # Instrument Actions
        self.positioning_system_action = QtWidgets.QAction('Positioning System', self)
        self.positioning_system_action.setStatusTip('Change positioning system settings')
        self.positioning_system_action.triggered.connect(self.docks.showPositionerControl)

        self.jaw_action = QtWidgets.QAction('Incident Jaws', self)
        self.jaw_action.setStatusTip('Change incident jaws settings')
        self.jaw_action.triggered.connect(self.docks.showJawControl)

        # Help Actions
        self.show_documentation_action = QtWidgets.QAction('&Documentation', self)
        self.show_documentation_action.setStatusTip('Show online documentation')
        self.show_documentation_action.setShortcut('F1')
        self.show_documentation_action.setIcon(QtGui.QIcon(path_for('question.png')))
        self.show_documentation_action.triggered.connect(self.showDocumentation)

        self.check_update_action = QtWidgets.QAction(f'&Check for Update', self)
        self.check_update_action.setStatusTip('Check the internet for software updates')
        self.check_update_action.triggered.connect(lambda: self.updater.check())

        self.show_about_action = QtWidgets.QAction(f'&About {MAIN_WINDOW_TITLE}', self)
        self.show_about_action.setStatusTip(f'About {MAIN_WINDOW_TITLE}')
        self.show_about_action.triggered.connect(self.about_dialog.show)

        # ToolBar Actions
        self.rotate_sample_action = QtWidgets.QAction('Rotate Sample', self)
        self.rotate_sample_action.setStatusTip('Rotate sample around fixed coordinate frame axis')
        self.rotate_sample_action.setIcon(QtGui.QIcon(path_for('rotate.png')))
        self.rotate_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Rotate))

        self.translate_sample_action = QtWidgets.QAction('Translate Sample', self)
        self.translate_sample_action.setStatusTip('Translate sample along fixed coordinate frame axis')
        self.translate_sample_action.setIcon(QtGui.QIcon(path_for('translate.png')))
        self.translate_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Translate))

        self.transform_sample_action = QtWidgets.QAction('Transform Sample with Matrix', self)
        self.transform_sample_action.setStatusTip('Transform sample with transformation matrix')
        self.transform_sample_action.setIcon(QtGui.QIcon(path_for('transform-matrix.png')))
        self.transform_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Custom))

        self.move_origin_action = QtWidgets.QAction('Move Origin to Sample', self)
        self.move_origin_action.setStatusTip('Translate sample using bounding box')
        self.move_origin_action.setIcon(QtGui.QIcon(path_for('origin.png')))
        self.move_origin_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Origin))

        self.plane_align_action = QtWidgets.QAction('Rotate Sample by Plane Alignment', self)
        self.plane_align_action.setStatusTip('Rotate sample using a selected plane')
        self.plane_align_action.setIcon(QtGui.QIcon(path_for('plane_align.png')))
        self.plane_align_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Plane))

        self.toggle_scene_action = QtWidgets.QAction('Toggle Scene', self)
        self.toggle_scene_action.setStatusTip('Toggle between sample and instrument scene')
        self.toggle_scene_action.setIcon(QtGui.QIcon(path_for('exchange.png')))
        self.toggle_scene_action.triggered.connect(self.scenes.toggleScene)
        self.toggle_scene_action.setShortcut(QtGui.QKeySequence('Ctrl+T'))

    def createMenus(self):
        """Creates the main menu and sub menus"""
        main_menu = self.menuBar()
        main_menu.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

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
        self.export_menu.addAction(self.export_samples_action)
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
            view_from_action = QtWidgets.QAction(direction.value, self)
            view_from_action.setStatusTip(f'View scene from the {direction.value} axis')
            view_from_action.setShortcut(QtGui.QKeySequence(f'Ctrl+{index+1}'))
            action = self.gl_widget.viewFrom
            view_from_action.triggered.connect(lambda ignore, d=direction: action(d))
            self.view_from_menu.addAction(view_from_action)

        view_menu.addAction(self.reset_camera_action)
        view_menu.addSeparator()
        view_menu.addAction(self.show_bounding_box_action)
        view_menu.addAction(self.show_fiducials_action)
        view_menu.addAction(self.show_measurement_action)
        view_menu.addAction(self.show_vectors_action)
        view_menu.addAction(self.show_coordinate_frame_action)
        view_menu.addSeparator()
        self.other_windows_menu = view_menu.addMenu('Other Windows')
        self.other_windows_menu.addAction(self.sample_manager_action)
        self.other_windows_menu.addAction(self.fiducial_manager_action)
        self.other_windows_menu.addAction(self.measurement_manager_action)
        self.other_windows_menu.addAction(self.vector_manager_action)
        self.other_windows_menu.addAction(self.simulation_dialog_action)

        insert_menu = main_menu.addMenu('&Insert')
        sample_menu = insert_menu.addMenu('Sample')
        sample_menu.addAction(self.import_sample_action)
        self.primitives_menu = sample_menu.addMenu('Primitives')

        for primitive in Primitives:
            add_primitive_action = QtWidgets.QAction(primitive.value, self)
            add_primitive_action.setStatusTip(f'Add {primitive.value} model as sample')
            add_primitive_action.triggered.connect(lambda ignore, p=primitive: self.docks.showInsertPrimitiveDialog(p))
            self.primitives_menu.addAction(add_primitive_action)

        fiducial_points_menu = insert_menu.addMenu('Fiducial Points')
        fiducial_points_menu.addAction(self.import_fiducial_action)
        fiducial_points_menu.addAction(self.keyin_fiducial_action)

        measurement_points_menu = insert_menu.addMenu('Measurement Points')
        measurement_points_menu.addAction(self.import_measurement_action)
        measurement_points_menu.addAction(self.keyin_measurement_action)
        measurement_points_menu.addAction(self.pick_measurement_action)

        measurement_vectors_menu = insert_menu.addMenu('Measurement Vectors')
        measurement_vectors_menu.addAction(self.import_measurement_vector_action)
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
        enable = False if self.presenter.model.project_data is None else True

        self.save_project_action.setEnabled(enable)
        self.save_as_action.setEnabled(enable)
        for action in self.export_menu.actions():
            action.setEnabled(enable)

        self.render_action_group.setEnabled(enable)

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

    def createToolBar(self):
        """Creates the tool bar"""
        toolbar = self.addToolBar('ToolBar')
        toolbar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        toolbar.setMovable(False)

        toolbar.addAction(self.new_project_action)
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()
        toolbar.addAction(self.solid_render_action)
        toolbar.addAction(self.line_render_action)
        toolbar.addAction(self.blend_render_action)
        toolbar.addAction(self.show_bounding_box_action)

        sub_button = QtWidgets.QToolButton(self)
        sub_button.setIcon(QtGui.QIcon(path_for('eye-slash.png')))
        sub_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        sub_button.setToolTip('Show/Hide Elements')
        sub_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        sub_button.addAction(self.show_fiducials_action)
        sub_button.addAction(self.show_measurement_action)
        sub_button.addAction(self.show_vectors_action)
        sub_button.addAction(self.show_coordinate_frame_action)
        toolbar.addWidget(sub_button)

        sub_button = QtWidgets.QToolButton(self)
        sub_button.setIcon(QtGui.QIcon(path_for('camera.png')))
        sub_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        sub_button.setToolTip('Preset Views')
        sub_button.setMenu(self.view_from_menu)
        toolbar.addWidget(sub_button)

        toolbar.addSeparator()
        toolbar.addAction(self.rotate_sample_action)
        toolbar.addAction(self.translate_sample_action)
        toolbar.addAction(self.transform_sample_action)
        toolbar.addAction(self.move_origin_action)
        toolbar.addAction(self.plane_align_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_scene_action)

    def createStatusBar(self):
        """Creates the status bar"""
        sb = StatusBar()
        self.setStatusBar(sb)
        self.instrument_label = QtWidgets.QLabel()
        self.instrument_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instrument_label.setToolTip('Current Instrument')
        sb.addPermanentWidget(self.instrument_label, alignment=QtCore.Qt.AlignLeft)

        self.cursor_label = QtWidgets.QLabel()
        self.cursor_label.setAlignment(QtCore.Qt.AlignCenter)
        sb.addPermanentWidget(self.cursor_label)

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
                recent_project_action = QtWidgets.QAction(project, self)
                recent_project_action.triggered.connect(lambda ignore, p=project: self.openProject(p))
                self.recent_menu.addAction(recent_project_action)
        else:
            recent_project_action = QtWidgets.QAction('None', self)
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
        self.change_instrument_action_group = QtWidgets.QActionGroup(self)
        self.project_file_instrument_action = QtWidgets.QAction('', self)
        self.change_instrument_action_group.addAction(self.project_file_instrument_action)
        self.change_instrument_menu.addAction(self.project_file_instrument_action)
        self.project_file_instrument_separator = self.change_instrument_menu.addSeparator()
        self.project_file_instrument_action.setCheckable(True)
        self.project_file_instrument_action.setVisible(False)
        self.project_file_instrument_separator.setVisible(False)
        for name in sorted(self.presenter.model.instruments.keys()):
            change_instrument_action = QtWidgets.QAction(name, self)
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
            toggleActionInGroup(model.instrument.name, self.change_instrument_action_group)
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
        action_group = QtWidgets.QActionGroup(self)
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
            other_settings_action = QtWidgets.QAction('Other Settings', self)
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
        :rtype: QtWidgets.QAction
        """
        change_collimator_action = QtWidgets.QAction(name, self)
        change_collimator_action.setStatusTip(f'Change collimator to {name}')
        change_collimator_action.setCheckable(True)
        change_collimator_action.setChecked(active == name)
        change_collimator_action.triggered.connect(lambda ignore,
                                                   n=detector,
                                                   t=name: self.presenter.changeCollimators(n, t))

        return change_collimator_action

    def showNewProjectDialog(self):
        """Opens the new project dialog"""
        if self.presenter.confirmSave():
            project_dialog = ProjectDialog(self.recent_projects, parent=self)
            project_dialog.setModal(True)
            project_dialog.show()

    def showPreferences(self, group=None):
        """Opens the preferences dialog"""
        preferences = Preferences(self)
        preferences.setActiveGroup(group)
        preferences.setModal(True)
        preferences.show()

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
        calibration_error = CalibrationErrorDialog(self, pose_id, fiducial_id, error)
        return calibration_error.exec() == QtWidgets.QDialog.Accepted

    def showPathLength(self):
        """Opens the path length plotter dialog"""
        simulation = self.presenter.model.simulation
        if simulation is None:
            self.showMessage('There are no simulation results.', MessageSeverity.Information)
            return

        if not simulation.compute_path_length:
            self.showMessage('Path Length computation is not enabled for this simulation.\n'
                             'Go to "Simulation > Compute Path Length" to enable it then \nrestart simulation.',
                             MessageSeverity.Information)
            return

        path_length_plotter = PathLengthPlotter(self)
        path_length_plotter.setModal(True)
        path_length_plotter.show()

    def showSampleExport(self, sample_list):
        """Shows the dialog for selecting which sample to export"""
        sample_export = SampleExportDialog(sample_list, parent=self)
        if sample_export.exec() != QtWidgets.QFileDialog.Accepted:
            return ''

        return sample_export.selected

    def showScriptExport(self):
        """Shows the dialog for exporting the resulting script from a simulation"""
        simulation = self.presenter.model.simulation
        if simulation is None:
            self.showMessage('There are no simulation results to write in script.', MessageSeverity.Information)
            return

        if simulation.isRunning():
            self.showMessage('Finish or Stop the current simulation before attempting to write script.',
                             MessageSeverity.Information)
            return

        if not simulation.has_valid_result:
            self.showMessage('There are no valid simulation results to write in script.', MessageSeverity.Information)
            return

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

    def showSaveDialog(self, filters, current_dir='', title=''):
        """Shows the file dialog for selecting path to save a file to

        :param filters: file filters
        :type filters: str
        :param current_dir: initial path
        :type current_dir: str
        :param title: dialog title
        :type title: str
        :return: selected file path
        :rtype: str
        """
        directory = current_dir if current_dir else os.path.splitext(self.presenter.model.save_path)[0]
        filename = FileDialog.getSaveFileName(self, title,  directory, filters)
        return filename

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

    def showMessage(self, message, severity=MessageSeverity.Critical):
        """Shows a message with a given severity.

        :param message: user message
        :type message: str
        :param severity: severity of the message
        :type severity: MessageSeverity
        """
        if severity == MessageSeverity.Critical:
            QtWidgets.QMessageBox.critical(self, MAIN_WINDOW_TITLE, message)
        elif severity == MessageSeverity.Warning:
            QtWidgets.QMessageBox.warning(self, MAIN_WINDOW_TITLE, message)
        else:
            QtWidgets.QMessageBox.information(self, MAIN_WINDOW_TITLE, message)

    def showSaveDiscardMessage(self, name):
        """Shows a message to confirm if unsaved changes should be saved or discarded.

        :param name: the name of the unsaved project
        :type name: str
        """
        message = 'The document has been modified.\n\n' \
                  'Do you want to save changes to "{}"?\t'.format(name)
        buttons = QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
        reply = QtWidgets.QMessageBox.warning(self,
                                              MAIN_WINDOW_TITLE,
                                              message, buttons,
                                              QtWidgets.QMessageBox.Cancel)

        if reply == QtWidgets.QMessageBox.Save:
            return MessageReplyType.Save
        if reply == QtWidgets.QMessageBox.Discard:
            return MessageReplyType.Discard
        else:
            return MessageReplyType.Cancel

    def showSelectChoiceMessage(self, message, choices, default_choice=0):
        """Shows a message box to allows the user to select one of multiple choices

        :param message: message
        :type message: str
        :param choices: list of choices
        :type choices: List[str]
        :param default_choice: index of default choice
        :type default_choice: int
        :return: selected choice
        :rtype: str
        """
        message_box = QtWidgets.QMessageBox(self)
        message_box.setWindowTitle(MAIN_WINDOW_TITLE)
        message_box.setText(message)

        buttons = []
        for choice in choices:
            buttons.append(QtWidgets.QPushButton(choice))
            message_box.addButton(buttons[-1], QtWidgets.QMessageBox.YesRole)

        message_box.setDefaultButton(buttons[default_choice])
        message_box.exec()

        for index, button in enumerate(buttons):
            if message_box.clickedButton() == button:
                return choices[index]

    def openProject(self, filename=''):
        """Loads a project with the given filename. if filename is empty,
        a file dialog will be opened

        :param filename: full path of file
        :type filename: str
        """
        if not self.presenter.confirmSave():
            return

        if not filename:
            filename = self.showOpenDialog('hdf5 File (*.h5)', title='Open Project',
                                           current_dir=self.presenter.model.save_path)
            if not filename:
                return

        self.presenter.useWorker(self.presenter.openProject, [filename], self.presenter.updateView,
                                 self.presenter.projectOpenError)

    def showDocumentation(self):
        """Opens the documentation in the system's default browser"""
        webbrowser.open_new(DOCS_URL)

    def showUpdateMessage(self, message):
        """Shows the software update message in a custom dialog with a check-box to change
        the 'check update on startup' setting

        :param message: message
        :type message: string
        """
        message_box = QtWidgets.QMessageBox(self)
        message_box.setWindowTitle(f'{MAIN_WINDOW_TITLE} Update')
        message_box.setText(message)
        message_box.setTextFormat(QtCore.Qt.RichText)
        checkbox = QtWidgets.QCheckBox('Check for updates on startup')
        checkbox.setChecked(settings.value(settings.Key.Check_Update))
        checkbox.stateChanged.connect(lambda state: settings.system.setValue(settings.Key.Check_Update.value,
                                                                             state == QtCore.Qt.Checked))
        message_box.setCheckBox(checkbox)

        cancel_button = QtWidgets.QPushButton('Close')
        message_box.addButton(cancel_button, QtWidgets.QMessageBox.NoRole)

        message_box.exec()

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


class Updater:
    """Handles checking for software updates

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        self.startup = False
        self.parent = parent

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
            self.parent.progress_dialog.showMessage('Checking the Internet for Updates')
        self.parent.presenter.useWorker(self.checkHelper, [], self.onSuccess, self.onFailure,
                                        self.parent.progress_dialog.close)

    def checkHelper(self):
        """Checks for the latest release version on the GitHub repo

        :return: version
        :rtype: str
        """
        import json
        import urllib.request

        response = urllib.request.urlopen(UPDATE_URL)
        tag_name = json.loads(response.read()).get('tag_name')

        return tag_name

    def onSuccess(self, version):
        """Reports the version found after successful check

        :param version: version tag
        :type version: str
        """
        update_found = version and not version.endswith(__version__)
        if update_found:
            self.parent.showUpdateMessage(f'A new version ({version}) of {MAIN_WINDOW_TITLE} is available. Download '
                                          f'the installer from <a href="{RELEASES_URL}">{RELEASES_URL}</a>.<br/><br/>')
        else:
            if self.startup:
                return
            self.parent.showUpdateMessage(f'You are running the latest version of {MAIN_WINDOW_TITLE}.<br/><br/>')

    def onFailure(self, exception):
        """Logs and reports error after failed check

        :param exception: exception when checking for update
        :type exception: Union[HTTPError, URLError]
        """
        from urllib.error import URLError, HTTPError
        import logging

        logging.error('An error occurred while checking for updates', exc_info=exception)
        if self.startup:
            return

        if isinstance(exception,  HTTPError):
            self.parent.showUpdateMessage(f'You are running the latest version of {MAIN_WINDOW_TITLE}.<br/><br/>')
        elif isinstance(exception, URLError):
            self.parent.showMessage('An error occurred when attempting to connect to update server. '
                                    'Check your internet connection and/or firewall and try again.')
