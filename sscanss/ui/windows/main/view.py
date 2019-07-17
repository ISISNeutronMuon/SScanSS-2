import os
from PyQt5 import QtCore, QtGui, QtWidgets
from .presenter import MainWindowPresenter, MessageReplyType
from .dock_manager import DockManager
from .scene_manager import SceneManager
from sscanss.config import settings, path_for, DOCS_PATH
from sscanss.ui.dialogs import (ProgressDialog, ProjectDialog, Preferences, AlignmentErrorDialog, FileDialog,
                                SampleExportDialog, ScriptExportDialog, PathLengthPlotter, AboutDialog)
from sscanss.ui.widgets import GLWidget
from sscanss.core.scene import Node
from sscanss.core.util import Primitives, Directions, TransformType, PointType, MessageSeverity, Attributes

MAIN_WINDOW_TITLE = 'SScanSS 2'


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.recent_projects = []
        self.presenter = MainWindowPresenter(self)

        self.undo_stack = QtWidgets.QUndoStack(self)
        self.undo_view = QtWidgets.QUndoView(self.undo_stack)
        self.undo_view.setWindowTitle('History')
        self.undo_view.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

        self.gl_widget = GLWidget(self)
        self.setCentralWidget(self.gl_widget)

        self.docks = DockManager(self)
        self.scenes = SceneManager(self)
        self.progress_dialog = ProgressDialog(self)
        self.about_dialog = AboutDialog(self)

        self.createActions()
        self.createMenus()
        self.createToolBar()

        self.setWindowTitle(MAIN_WINDOW_TITLE)
        self.setWindowIcon(QtGui.QIcon('../logo.ico'))
        self.setMinimumSize(1024, 800)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.readSettings()
        self.updateMenus()

    def createActions(self):
        self.new_project_action = QtWidgets.QAction('&New Project', self)
        self.new_project_action.setIcon(QtGui.QIcon(path_for('file.png')))
        self.new_project_action.setShortcut(QtGui.QKeySequence.New)
        self.new_project_action.triggered.connect(self.showNewProjectDialog)

        self.open_project_action = QtWidgets.QAction('&Open Project', self)
        self.open_project_action.setIcon(QtGui.QIcon(path_for('folder-open.png')))
        self.open_project_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_project_action.triggered.connect(self.openProject)

        self.save_project_action = QtWidgets.QAction('&Save Project', self)
        self.save_project_action.setIcon(QtGui.QIcon(path_for('save.png')))
        self.save_project_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_project_action.triggered.connect(lambda: self.presenter.saveProject())

        self.save_as_action = QtWidgets.QAction('Save &As...', self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.presenter.saveProject(save_as=True))

        self.export_samples_action = QtWidgets.QAction('Samples', self)
        self.export_samples_action.triggered.connect(self.presenter.exportSamples)

        self.export_fiducials_action = QtWidgets.QAction('Fiducial Points', self)
        self.export_fiducials_action.triggered.connect(lambda: self.presenter.exportPoints(PointType.Fiducial))

        self.export_measurements_action = QtWidgets.QAction('Measurement Points', self)
        self.export_measurements_action.triggered.connect(lambda: self.presenter.exportPoints(PointType.Measurement))

        self.export_vectors_action = QtWidgets.QAction('Measurement Vectors', self)
        self.export_vectors_action.triggered.connect(self.presenter.exportVectors)

        self.export_alignment_action = QtWidgets.QAction('Alignment Matrix', self)
        self.export_alignment_action.triggered.connect(self.presenter.exportAlignmentMatrix)

        self.export_script_action = QtWidgets.QAction('Script', self)
        self.export_script_action.triggered.connect(self.showScriptExport)

        self.exit_action = QtWidgets.QAction('E&xit', self)
        self.exit_action.setShortcut(QtGui.QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)

        # Edit Menu Actions
        self.undo_action = self.undo_stack.createUndoAction(self, '&Undo')
        self.undo_action.setIcon(QtGui.QIcon(path_for('undo.png')))
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)

        self.redo_action = self.undo_stack.createRedoAction(self, '&Redo')
        self.redo_action.setIcon(QtGui.QIcon(path_for('redo.png')))
        self.redo_action.setShortcut(QtGui.QKeySequence.Redo)

        self.undo_view_action = QtWidgets.QAction('Undo &History', self)
        self.undo_view_action.triggered.connect(self.showUndoHistory)

        self.preferences_action = QtWidgets.QAction('Preferences', self)
        self.preferences_action.triggered.connect(lambda: self.showPreferences(None))

        # View Menu Actions
        self.solid_render_action = QtWidgets.QAction(Node.RenderMode.Solid.value, self)
        self.solid_render_action.setIcon(QtGui.QIcon(path_for('solid.png')))
        self.solid_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Solid))
        self.solid_render_action.setCheckable(True)

        self.line_render_action = QtWidgets.QAction(Node.RenderMode.Wireframe.value, self)
        self.line_render_action.setIcon(QtGui.QIcon(path_for('wireframe.png')))
        self.line_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Wireframe))
        self.line_render_action.setCheckable(True)

        self.blend_render_action = QtWidgets.QAction(Node.RenderMode.Transparent.value, self)
        self.blend_render_action.setIcon(QtGui.QIcon(path_for('blend.png')))
        self.blend_render_action.triggered.connect(lambda: self.scenes.changeRenderMode(Node.RenderMode.Transparent))
        self.blend_render_action.setCheckable(True)
        self.blend_render_action.setChecked(True)

        self.render_action_group = QtWidgets.QActionGroup(self)
        self.render_action_group.addAction(self.solid_render_action)
        self.render_action_group.addAction(self.line_render_action)
        self.render_action_group.addAction(self.blend_render_action)

        self.show_bounding_box_action = QtWidgets.QAction('Toggle Bounding Box', self)
        self.show_bounding_box_action.setIcon(QtGui.QIcon(path_for('boundingbox.png')))
        self.show_bounding_box_action.setCheckable(True)
        self.show_bounding_box_action.setChecked(self.gl_widget.show_bounding_box)
        self.show_bounding_box_action.toggled.connect(self.gl_widget.showBoundingBox)

        self.show_coordinate_frame_action = QtWidgets.QAction('Toggle Coordinate Frame', self)
        self.show_coordinate_frame_action.setIcon(QtGui.QIcon(path_for('hide_coordinate_frame.png')))
        self.show_coordinate_frame_action.setCheckable(True)
        self.show_coordinate_frame_action.setChecked(self.gl_widget.show_coordinate_frame)
        self.show_coordinate_frame_action.toggled.connect(self.gl_widget.showCoordinateFrame)

        self.show_fiducials_action = QtWidgets.QAction('Toggle Fiducial Points', self)
        self.show_fiducials_action.setIcon(QtGui.QIcon(path_for('hide_fiducials.png')))
        self.show_fiducials_action.setCheckable(True)
        self.show_fiducials_action.setChecked(True)
        action = self.scenes.toggleVisibility
        self.show_fiducials_action.toggled.connect(lambda state, a=Attributes.Fiducials: action(a, state))

        self.show_measurement_action = QtWidgets.QAction('Toggle Measurement Points', self)
        self.show_measurement_action.setIcon(QtGui.QIcon(path_for('hide_measurement.png')))
        self.show_measurement_action.setCheckable(True)
        self.show_measurement_action.setChecked(True)
        self.show_measurement_action.toggled.connect(lambda state, a=Attributes.Measurements: action(a, state))

        self.show_vectors_action = QtWidgets.QAction('Toggle Measurement Vectors', self)
        self.show_vectors_action.setIcon(QtGui.QIcon(path_for('hide_vectors.png')))
        self.show_vectors_action.setCheckable(True)
        self.show_vectors_action.setChecked(True)
        self.show_vectors_action.toggled.connect(lambda state, a=Attributes.Vectors: action(a, state))

        self.reset_camera_action = QtWidgets.QAction('Reset View', self)
        self.reset_camera_action.triggered.connect(self.gl_widget.resetCamera)

        self.sample_manager_action = QtWidgets.QAction('Samples', self)
        self.sample_manager_action.triggered.connect(self.docks.showSampleManager)

        self.fiducial_manager_action = QtWidgets.QAction('Fiducial Points', self)
        self.fiducial_manager_action.triggered.connect(lambda: self.docks.showPointManager(PointType.Fiducial))

        self.measurement_manager_action = QtWidgets.QAction('Measurements Points', self)
        self.measurement_manager_action.triggered.connect(lambda: self.docks.showPointManager(PointType.Measurement))

        self.vector_manager_action = QtWidgets.QAction('Measurements Vectors', self)
        self.vector_manager_action.triggered.connect(self.docks.showVectorManager)

        self.simulation_dialog_action = QtWidgets.QAction('Simulation Results', self)
        self.simulation_dialog_action.triggered.connect(self.docks.showSimulationResults)

        # Insert Menu Actions
        self.import_sample_action = QtWidgets.QAction('File...', self)
        self.import_sample_action.triggered.connect(self.presenter.importSample)

        self.import_fiducial_action = QtWidgets.QAction('File...', self)
        self.import_fiducial_action.triggered.connect(lambda: self.presenter.importPoints(PointType.Fiducial))

        self.keyin_fiducial_action = QtWidgets.QAction('Key-in', self)
        self.keyin_fiducial_action.triggered.connect(lambda: self.docks.showInsertPointDialog(PointType.Fiducial))

        self.import_measurement_action = QtWidgets.QAction('File...', self)
        self.import_measurement_action.triggered.connect(lambda: self.presenter.importPoints(PointType.Measurement))

        self.keyin_measurement_action = QtWidgets.QAction('Key-in', self)
        self.keyin_measurement_action.triggered.connect(lambda: self.docks.showInsertPointDialog(PointType.Measurement))

        self.pick_measurement_action = QtWidgets.QAction('Graphical Selection', self)
        self.pick_measurement_action.triggered.connect(self.docks.showPickPointDialog)

        self.import_measurement_vector_action = QtWidgets.QAction('File...', self)
        self.import_measurement_vector_action.triggered.connect(self.presenter.importVectors)

        self.select_strain_component_action = QtWidgets.QAction('Select Strain Component', self)
        self.select_strain_component_action.triggered.connect(self.docks.showInsertVectorDialog)

        self.align_via_pose_action = QtWidgets.QAction('6D Pose', self)
        self.align_via_pose_action.triggered.connect(self.docks.showAlignSample)

        self.align_via_matrix_action = QtWidgets.QAction('Transformation Matrix', self)
        self.align_via_matrix_action.triggered.connect(self.presenter.alignSampleWithMatrix)

        self.align_via_fiducials_action = QtWidgets.QAction('Fiducials Points', self)
        self.align_via_fiducials_action.triggered.connect(self.presenter.alignSampleWithFiducialPoints)

        self.run_simulation_action = QtWidgets.QAction('&Run Simulation', self)
        self.run_simulation_action.setShortcut('F5')
        self.run_simulation_action.triggered.connect(self.presenter.runSimulation)

        self.stop_simulation_action = QtWidgets.QAction('&Stop Simulation', self)
        self.stop_simulation_action.setShortcut('Shift+F5')
        self.stop_simulation_action.triggered.connect(self.presenter.stopSimulation)

        self.compute_path_length_action = QtWidgets.QAction('Calculate Path Length', self)
        self.compute_path_length_action.setCheckable(True)
        self.compute_path_length_action.setChecked(False)

        self.check_limits_action = QtWidgets.QAction('Hardware Limits Check', self)
        self.check_limits_action.setCheckable(True)
        self.check_limits_action.setChecked(True)

        self.show_sim_graphics_action = QtWidgets.QAction('Show Graphically', self)
        self.show_sim_graphics_action.setCheckable(True)
        self.show_sim_graphics_action.setChecked(True)

        self.show_sim_options_action = QtWidgets.QAction('Simulation Options', self)
        self.show_sim_options_action.triggered.connect(lambda: self.showPreferences(settings.Group.Simulation))

        # Instrument Actions
        self.positioning_system_action = QtWidgets.QAction('Positioning System', self)
        self.positioning_system_action.triggered.connect(self.docks.showPositionerControl)

        self.jaw_action = QtWidgets.QAction('Incident Jaws', self)
        self.jaw_action.triggered.connect(self.docks.showJawControl)

        # Help Actions
        self.show_walkthrough_action = QtWidgets.QAction('&Walkthrough', self)

        self.show_documentation_action = QtWidgets.QAction('&Documentation', self)
        self.show_documentation_action.setShortcut('F1')
        self.show_documentation_action.triggered.connect(self.showDocumentation)

        self.show_about_action = QtWidgets.QAction(f'&About {MAIN_WINDOW_TITLE}', self)
        self.show_about_action.triggered.connect(self.about_dialog.show)

        # ToolBar Actions
        self.rotate_sample_action = QtWidgets.QAction('Rotate Sample', self)
        self.rotate_sample_action.setIcon(QtGui.QIcon(path_for('rotate.png')))
        self.rotate_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Rotate))

        self.translate_sample_action = QtWidgets.QAction('Translate Sample', self)
        self.translate_sample_action.setIcon(QtGui.QIcon(path_for('translate.png')))
        self.translate_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Translate))

        self.transform_sample_action = QtWidgets.QAction('Transform Sample with Matrix', self)
        self.transform_sample_action.setIcon(QtGui.QIcon(path_for('transform-matrix.png')))
        self.transform_sample_action.triggered.connect(lambda: self.docks.showTransformDialog(TransformType.Custom))

        self.toggle_scene_action = QtWidgets.QAction('Toggle Scene', self)
        self.toggle_scene_action.setIcon(QtGui.QIcon(path_for('exchange.png')))
        self.toggle_scene_action.triggered.connect(self.scenes.toggleScene)

    def createMenus(self):
        main_menu = self.menuBar()
        main_menu.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        self.recent_menu = file_menu.addMenu('Open &Recent')
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu('Export...')
        export_menu.addAction(self.export_script_action)
        export_menu.addSeparator()
        export_menu.addAction(self.export_samples_action)
        export_menu.addAction(self.export_fiducials_action)
        export_menu.addAction(self.export_measurements_action)
        export_menu.addAction(self.export_vectors_action)
        export_menu.addAction(self.export_alignment_action)
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
        for direction in Directions:
            view_from_action = QtWidgets.QAction(direction.value, self)
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
        self.change_instrument_action_group = QtWidgets.QActionGroup(self)
        for name in self.presenter.model.instruments.keys():
            change_instrument_action = QtWidgets.QAction(name, self)
            change_instrument_action.setCheckable(True)
            change_instrument_action.triggered.connect(lambda ignore, n=name: self.presenter.changeInstrument(n))
            self.change_instrument_action_group.addAction(change_instrument_action)
            self.change_instrument_menu.addAction(change_instrument_action)

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
        simulation_menu.addSeparator()
        simulation_menu.addAction(self.show_sim_options_action)

        help_menu = main_menu.addMenu('&Help')
        help_menu.addAction(self.show_walkthrough_action)
        help_menu.addAction(self.show_documentation_action)
        help_menu.addSeparator()
        help_menu.addAction(self.show_about_action)

    def updateMenus(self):
        enable = False if self.presenter.model.project_data is None else True

        self.save_project_action.setEnabled(enable)
        self.save_as_action.setEnabled(enable)

        self.render_action_group.setEnabled(enable)

        for action in self.view_from_menu.actions():
            action.setEnabled(enable)
        self.reset_camera_action.setEnabled(enable)
        self.show_bounding_box_action.setEnabled(enable)
        self.show_coordinate_frame_action.setEnabled(enable)
        self.show_fiducials_action.setEnabled(enable)
        self.show_measurement_action.setEnabled(enable)
        self.show_vectors_action.setEnabled(enable)

        self.sample_manager_action.setEnabled(enable)
        self.sample_manager_action.setEnabled(enable)
        self.fiducial_manager_action.setEnabled(enable)
        self.measurement_manager_action.setEnabled(enable)

        self.import_sample_action.setEnabled(enable)
        self.primitives_menu.setEnabled(enable)

        self.import_fiducial_action.setEnabled(enable)
        self.keyin_fiducial_action.setEnabled(enable)

        self.import_measurement_action.setEnabled(enable)
        self.keyin_measurement_action.setEnabled(enable)
        self.pick_measurement_action.setEnabled(enable)

        self.import_measurement_vector_action.setEnabled(enable)
        self.select_strain_component_action.setEnabled(enable)

        self.instrument_menu.setEnabled(enable)

        self.rotate_sample_action.setEnabled(enable)
        self.translate_sample_action.setEnabled(enable)
        self.transform_sample_action.setEnabled(enable)
        self.toggle_scene_action.setEnabled(enable)

    def createToolBar(self):
        toolbar = self.addToolBar('FileToolBar')
        toolbar.setObjectName('FileToolBar')
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
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_scene_action)

    @property
    def selected_render_mode(self):
        return Node.RenderMode(self.render_action_group.checkedAction().text())

    def readSettings(self):
        """ Loads window geometry from INI file """
        self.restoreGeometry(settings.value(settings.Key.Geometry))
        self.restoreState(settings.value(settings.Key.Window_State))
        self.recent_projects = settings.value(settings.Key.Recent_Projects)

    def populateRecentMenu(self):
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
            settings.setValue(settings.Key.Geometry, self.saveGeometry())
            settings.setValue(settings.Key.Window_State, self.saveState())
            if self.recent_projects:
                settings.setValue(settings.Key.Recent_Projects, self.recent_projects)
            event.accept()
        else:
            event.ignore()

    def resetInstrumentMenu(self):
        self.instrument_menu.clear()
        self.instrument_menu.addMenu(self.change_instrument_menu)
        self.instrument_menu.addSeparator()
        self.instrument_menu.addAction(self.jaw_action)
        self.instrument_menu.addAction(self.positioning_system_action)
        self.instrument_menu.addSeparator()
        self.instrument_menu.addMenu(self.align_sample_menu)
        self.collimator_action_groups = {}

    def addCollimatorMenu(self, detector, collimators, active, menu='Detector', show_more_settings=False):
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
            other_settings_action.triggered.connect(lambda ignore,
                                                           d=detector,: self.docks.showDetectorControl(d))
            collimator_menu.addAction(other_settings_action)

    def __createChangeCollimatorAction(self, name, active, detector):
        change_collimator_action = QtWidgets.QAction(name, self)
        change_collimator_action.setCheckable(True)
        change_collimator_action.setChecked(active == name)
        change_collimator_action.triggered.connect(lambda ignore,
                                                          n=detector,
                                                          t=name: self.presenter.changeCollimators(n, t))

        return change_collimator_action

    def showUndoHistory(self):
        """ shows Undo History"""
        self.undo_view = QtWidgets.QUndoView(self.undo_stack)
        self.undo_view.setWindowTitle('Undo History')
        self.undo_view.show()
        self.undo_view.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def showNewProjectDialog(self):
        if self.presenter.confirmSave():
            self.project_dialog = ProjectDialog(self.recent_projects, parent=self)
            self.project_dialog.setModal(True)
            self.project_dialog.show()

    def showPreferences(self, group=None):
        self.preferences = Preferences(self)
        self.preferences.setActiveGroup(group)
        self.preferences.setModal(True)
        self.preferences.show()

    def showAlignmentError(self):
        self.alignment_error = AlignmentErrorDialog(parent=self)
        self.alignment_error.setModal(True)
        self.alignment_error.show()

    def showPathLength(self):
        simulation = self.presenter.model.simulation
        if simulation is not None and not simulation.compute_path_length:
            self.showMessage('Path Length computation is not enabled for this simulation.\n'
                             'Go to "Simulation > Compute Path Length" to enable it then \nrestart simulation.',
                             MessageSeverity.Information)
            return

        self.path_length_plotter = PathLengthPlotter(self)
        self.path_length_plotter.setModal(True)
        self.path_length_plotter.show()

    def showSampleExport(self, sample_list):
        sample_export = SampleExportDialog(sample_list, parent=self)
        if sample_export.exec() != QtWidgets.QFileDialog.Accepted:
            return ''

        return sample_export.selected

    def showScriptExport(self):
        simulation = self.presenter.model.simulation
        if simulation is None:
            self.showMessage('There are no simulation results to write in script.', MessageSeverity.Information)
            return

        if simulation.isRunning():
            self.showMessage('Finish or Stop the current simulation before attempting to write script.',
                             MessageSeverity.Information)
            return

        if not simulation.results:
            self.showMessage('There are no simulation results to write in script.', MessageSeverity.Information)
            return

        script_export = ScriptExportDialog(simulation, parent=self)
        script_export.setModal(True)
        script_export.show()

    def showProjectName(self):
        project_name = self.presenter.model.project_data['name']
        save_path = self.presenter.model.save_path
        if save_path:
            title = f'{project_name} [{save_path}] - {MAIN_WINDOW_TITLE}'
        else:
            title = f'{project_name} - {MAIN_WINDOW_TITLE}'
        self.setWindowTitle(title)

    def showSaveDialog(self, filters, current_dir='', title=''):
        directory = current_dir if current_dir else os.path.splitext(self.presenter.model.save_path)[0]
        filename = FileDialog.getSaveFileName(self, title,  directory, filters)
        return filename

    def showOpenDialog(self, filters, current_dir='', title=''):
        directory = current_dir if current_dir else os.path.dirname(self.presenter.model.save_path)
        filename = FileDialog.getOpenFileName(self, title, directory, filters)
        return filename

    def showMessage(self, message, severity=MessageSeverity.Critical):
        """
        Shows an message with a given severity.

        :param message: Error message
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
        """
        Shows an message to confirm if unsaved changes should be saved.

        :param name: the name of the unsaved project
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
        """
        This function loads a project with the given filename. if filename is empty,
        a file dialog will be opened.

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
        """
        This function opens the documentation html in the system's default application
        """
        path = DOCS_PATH / 'index.html'
        if os.path.isfile(path):
            os.startfile(path)
        else:
            self.showMessage('An error occurred while opening the offline documentation.\nYou can '
                             'access the documentation from the internet.')
