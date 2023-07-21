import datetime
import json
import webbrowser
import jsbeautifier
from PyQt6 import QtCore, QtGui, QtWidgets
from sscanss.config import settings, path_for
from sscanss.core.instrument import Sequence
from sscanss.core.scene import OpenGLRenderer, SceneManager
from sscanss.core.util import Directions, Attributes, MessageReplyType, FileDialog, create_scroll_area, MessageType
from sscanss.editor.designer import Designer
from sscanss.editor.dialogs import CalibrationWidget, Controls, FindWidget, FontWidget
from sscanss.editor.editor import Editor
from sscanss.editor.presenter import EditorPresenter, MAIN_WINDOW_TITLE
from sscanss.__version import __editor_version__, __version__


class EditorWindow(QtWidgets.QMainWindow):
    """Creates the main window of the instrument editor."""
    animate_instrument = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.recent_projects = []
        self.presenter = EditorPresenter(self)

        self.controls = Controls(self)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.setCentralWidget(self.main_splitter)
        self.filename = ''
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.splitter)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 2)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumHeight(200)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.South)
        self.main_splitter.addWidget(self.tabs)

        self.issues_table = QtWidgets.QTableWidget()
        self.issues_table.setShowGrid(False)
        self.issues_table.verticalHeader().hide()
        self.issues_table.setColumnCount(2)
        self.issues_table.setHorizontalHeaderLabels(['Path', 'Description'])
        self.issues_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.issues_table.setMinimumHeight(150)
        self.issues_table.setAlternatingRowColors(True)
        self.issues_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.issues_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.issues_table.horizontalHeader().setStretchLastSection(True)
        self.issues_table.horizontalHeader().setMinimumSectionSize(150)
        self.issues_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.tabs.addTab(self.issues_table, '&Issues')

        self.gl_widget = OpenGLRenderer(self)
        self.gl_widget.custom_error_handler = self.sceneSizeErrorHandler
        self.scene = SceneManager(self.presenter.model, self.gl_widget, False)
        self.scene.changeVisibility(Attributes.Beam, True)
        self.animate_instrument.connect(self.scene.animateInstrument)

        self.readSettings()

        self.editor = Editor(self)
        self.editor.textChanged.connect(self.presenter.model.lazyInstrumentUpdate)
        self.splitter.addWidget(self.gl_widget)
        self.splitter.addWidget(self.editor)

        self.designer = Designer(self)
        self.designer.json_updated.connect(lambda d: self.editor.setText(jsbeautifier.beautify(json.dumps(d))))

        self.updateTitle()
        self.setMinimumSize(1024, 800)
        self.setWindowIcon(QtGui.QIcon(path_for('editor-logo.png')))

        self.initActions()
        self.initMenus()

    def showSearchBox(self):
        """Opens the find dialog box."""
        self.find_dialog = FindWidget(self)
        self.find_dialog.fist_search_flag = True
        self.find_dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.find_dialog.show()

    def initActions(self):
        """Creates menu actions"""
        self.exit_action = QtGui.QAction('&Exit', self)
        self.exit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        self.exit_action.setStatusTip('Exit application')
        self.exit_action.triggered.connect(self.close)

        self.new_action = QtGui.QAction('&New File', self)
        self.new_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        self.new_action.setStatusTip('New Instrument Description File')
        self.new_action.triggered.connect(self.presenter.createNewFile)

        self.open_action = QtGui.QAction('&Open File', self)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_action.setStatusTip('Open Instrument Description File')
        self.open_action.triggered.connect(self.presenter.openFile)

        self.save_action = QtGui.QAction('&Save File', self)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.setStatusTip('Save Instrument Description File')
        self.save_action.triggered.connect(self.presenter.saveFile)

        self.save_as_action = QtGui.QAction('Save &As...', self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.presenter.saveFile(save_as=True))

        self.show_instrument_controls = QtGui.QAction('&Instrument Controls', self)
        self.show_instrument_controls.setStatusTip('Show Instrument Controls')
        self.show_instrument_controls.triggered.connect(self.controls.show)

        self.generate_robot_model_action = QtGui.QAction('&Generate Robot Model', self)
        self.generate_robot_model_action.setStatusTip('Generate Robot Model from Measurements')
        self.generate_robot_model_action.triggered.connect(self.presenter.generateRobotModel)

        self.reload_action = QtGui.QAction('&Reset Instrument', self)
        self.reload_action.setStatusTip('Reset Instrument')
        self.reload_action.triggered.connect(self.presenter.resetInstrumentControls)
        self.reload_action.setShortcut(QtGui.QKeySequence('F5'))

        self.find_action = QtGui.QAction('&Find', self)
        self.find_action.setStatusTip('Find text in editor')
        self.find_action.triggered.connect(self.showSearchBox)
        self.find_action.setShortcut(QtGui.QKeySequence('Ctrl+F'))

        self.show_world_coordinate_frame_action = QtGui.QAction('Toggle &World Coordinate Frame', self)
        self.show_world_coordinate_frame_action.setStatusTip('Toggle world coordinate frame')
        self.show_world_coordinate_frame_action.setCheckable(True)
        self.show_world_coordinate_frame_action.setChecked(self.gl_widget.show_coordinate_frame)
        self.show_world_coordinate_frame_action.toggled.connect(self.gl_widget.showCoordinateFrame)

        self.reset_camera_action = QtGui.QAction('Reset &View', self)
        self.reset_camera_action.setStatusTip('Reset camera view')
        self.reset_camera_action.triggered.connect(self.gl_widget.resetCamera)
        self.reset_camera_action.setShortcut(QtGui.QKeySequence('Ctrl+0'))

        self.show_documentation_action = QtGui.QAction('&Documentation', self)
        self.show_documentation_action.setStatusTip('Show online documentation')
        self.show_documentation_action.setShortcut('F1')
        self.show_documentation_action.triggered.connect(self.showDocumentation)

        self.about_action = QtGui.QAction('&About', self)
        self.about_action.setStatusTip(f'About {MAIN_WINDOW_TITLE}')
        self.about_action.triggered.connect(self.showAboutMessage)

        self.general_designer_action = QtGui.QAction('General', self)
        self.general_designer_action.setStatusTip('Add/Updates general instrument entries')
        self.general_designer_action.triggered.connect(lambda: self.showDesigner(Designer.Component.General))

        self.jaws_designer_action = QtGui.QAction('Incident jaws', self)
        self.jaws_designer_action.setStatusTip('Add/Updates incident jaws entry')
        self.jaws_designer_action.triggered.connect(lambda: self.showDesigner(Designer.Component.Jaws))

        self.detector_designer_action = QtGui.QAction('Detector', self)
        self.detector_designer_action.setStatusTip('Add/Updates detector entry')
        self.detector_designer_action.triggered.connect(lambda: self.showDesigner(Designer.Component.Detector))

        self.collimator_designer_action = QtGui.QAction('Collimator', self)
        self.collimator_designer_action.setStatusTip('Add/Updates collimator entry')
        self.collimator_designer_action.triggered.connect(lambda: self.showDesigner(Designer.Component.Collimator))

        self.fixed_hardware_designer_action = QtGui.QAction('Fixed Hardware', self)
        self.fixed_hardware_designer_action.setStatusTip('Add/Updates fixed hardware entry')
        self.fixed_hardware_designer_action.triggered.connect(
            lambda: self.showDesigner(Designer.Component.FixedHardware))

        self.positioning_stacks_designer_action = QtGui.QAction('Positioning Stacks', self)
        self.positioning_stacks_designer_action.setStatusTip('Add/Updates positioning stack entry')
        self.positioning_stacks_designer_action.triggered.connect(
            lambda: self.showDesigner(Designer.Component.PositioningStacks))

        self.positioners_designer_action = QtGui.QAction('Positioners', self)
        self.positioners_designer_action.setStatusTip('Add/Updates positioners entry')
        self.positioners_designer_action.triggered.connect(lambda: self.showDesigner(Designer.Component.Positioners))

    def initMenus(self):
        """Creates main menu and sub menus"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu('Open &Recent')
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        self.preferences_menu = file_menu.addMenu('Preferences')

        file_menu.addAction(self.exit_action)
        file_menu.aboutToShow.connect(self.populatePreferencesMenu)
        file_menu.aboutToShow.connect(self.populateRecentMenu)

        edit_menu = menu_bar.addMenu('&Edit')
        edit_menu.addAction(self.find_action)

        view_menu = menu_bar.addMenu('&View')
        view_menu.addAction(self.reload_action)
        view_menu.addAction(self.show_instrument_controls)
        view_menu.addSeparator()
        self.view_from_menu = view_menu.addMenu('View From')
        for index, direction in enumerate(Directions):
            view_from_action = QtGui.QAction(direction.value, self)
            view_from_action.setStatusTip(f'View scene from the {direction.value} axis')
            view_from_action.setShortcut(QtGui.QKeySequence(f'Ctrl+{index+1}'))
            view_from_action.triggered.connect(lambda ignore, d=direction: self.gl_widget.viewFrom(d))
            self.view_from_menu.addAction(view_from_action)

        view_menu.addAction(self.reset_camera_action)
        view_menu.addAction(self.show_world_coordinate_frame_action)

        tool_menu = menu_bar.addMenu('&Tool')
        designer_menu = tool_menu.addMenu('&Designer')
        designer_menu.addAction(self.general_designer_action)
        designer_menu.addAction(self.jaws_designer_action)
        designer_menu.addAction(self.detector_designer_action)
        designer_menu.addAction(self.collimator_designer_action)
        designer_menu.addAction(self.fixed_hardware_designer_action)
        designer_menu.addAction(self.positioning_stacks_designer_action)
        designer_menu.addAction(self.positioners_designer_action)
        tool_menu.addAction(self.generate_robot_model_action)

        help_menu = menu_bar.addMenu('&Help')
        help_menu.addAction(self.show_documentation_action)
        help_menu.addAction(self.show_documentation_action)
        help_menu.addAction(self.about_action)

    def reset(self):
        self.editor.setText('')
        self.updateTitle()
        self.scene.reset()
        self.controls.close()
        self.designer.clear()
        self.updateErrors([])

    def askAddress(self, must_exist, caption, directory, dir_filter):
        """Creates new window allowing the user to choose location to save file

        :param must_exist: whether the address must be of an existing file
        :type must_exist: bool
        :param caption: caption in the window
        :type caption: str
        :param directory: the starting directory
        :type directory: str
        :param dir_filter: filter to sort the file types
        :type dir_filter: str
        """
        dialog = FileDialog(self, caption, directory, dir_filter)
        if must_exist:
            filename = dialog.getOpenFileName(self, caption, directory, dir_filter)
        else:
            filename = dialog.getSaveFileName(self, caption, directory, dir_filter)
        return filename

    def showDesigner(self, component_type):
        """Shows the designer with given component

        :param component_type: component type
        :type component_type: Designer.Component
        """
        self.designer.setComponent(component_type)
        self.designer.setJson(self.presenter.parser.data)
        if self.tabs.count() < 2:
            self.tabs.addTab(create_scroll_area(self.designer), '&Designer')
        self.tabs.setCurrentIndex(1)

    def showFontComboBox(self):
        """Opens the fonts dialog box."""
        self.fonts_dialog = FontWidget(self)
        self.fonts_dialog.show()
        self.fonts_dialog.accepted.connect(lambda: self.editor.updateStyle(self.fonts_dialog.preview.font().family(),self.fonts_dialog.preview.font().pointSize()))

    def updateErrors(self, errors):
        """Updates the issue table with parser errors

        :param errors: parser errors
        :type errors: List[ParserError]
        """
        self.issues_table.setRowCount(len(errors))
        for i, error in enumerate(errors):
            x = QtWidgets.QTableWidgetItem(error.path)
            y = QtWidgets.QTableWidgetItem(error.message)

            y.setData(QtCore.Qt.ItemDataRole.ForegroundRole, QtGui.QBrush(QtGui.QColor('Tomato')))

            self.issues_table.setItem(i, 0, x)
            self.issues_table.setItem(i, 1, y)
        self.tabs.setTabText(0, f'&Issues ({len(errors)})' if errors else '&Issues')

    def createCalibrationWidget(self, points, types, offsets, homes):
        """Opens the calibration dialog
        :param points: measured 3D points for each joint
        :type points: List[numpy.ndarray]
        :param types: types of each joint
        :type types: List[Link.Type]
        :param offsets: measured offsets for each measurement
        :type offsets: List[numpy.ndarray]
        :param homes: home position for each measurement
        :type homes: List[float]
        """
        widget = CalibrationWidget(self, points, types, offsets, homes)
        widget.show()

    def moveInstrument(self, func, start_var, stop_var, duration=500, step=15):
        """Animates the movement of the instrument

        :param func: forward kinematics function
        :type func: Callable[numpy.ndarray, Any]
        :param start_var: inclusive start joint configuration/offsets
        :type start_var: List[float]
        :param stop_var: inclusive stop joint configuration/offsets
        :type stop_var: List[float]
        :param duration: time duration in milliseconds
        :type duration: int
        :param step: number of steps
        :type step: int
        """
        self.animate_instrument.emit(Sequence(func, start_var, stop_var, duration, step))

    def readSettings(self):
        """Loads the recent projects from settings"""
        self.font_family = settings.value(settings.Key.Editor_Font_Family)
        self.font_size = settings.value(settings.Key.Editor_Font_Size)
        self.recent_projects = settings.value(settings.Key.Recent_Editor_Projects)

    def populateRecentMenu(self):
        """Populates the recent project sub-menu"""
        self.recent_menu.clear()
        if self.recent_projects:
            for project in self.recent_projects:
                recent_project_action = QtGui.QAction(project, self)
                recent_project_action.triggered.connect(lambda ignore, p=project: self.presenter.openFile(p))
                self.recent_menu.addAction(recent_project_action)
        else:
            recent_project_action = QtGui.QAction('None', self)
            self.recent_menu.addAction(recent_project_action)

    def populatePreferencesMenu(self):
        """Populates the preferences sub-menu"""
        self.preferences_menu.clear()
        update_font_action = QtGui.QAction('Fonts', self)
        update_font_action.triggered.connect(self.showFontComboBox)
        self.preferences_menu.addAction(update_font_action)

    def closeEvent(self, event):
        if self.presenter.askToSaveFile():
            if self.recent_projects:
                settings.system.setValue(settings.Key.Recent_Editor_Projects.value, self.recent_projects)
            event.accept()
        else:
            event.ignore()

    def showSaveDiscardMessage(self, message):
        """Shows a dialog asking if unsaved changes should be saved or discarded
        :param message: the message shown in the window
        :type message: str
        :return: the users reply, either Save, Discard or Cancel
        :rtype: MessageReplyType
        """

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

    def showAboutMessage(self):
        """Shows the About message"""
        title = f'About {MAIN_WINDOW_TITLE}'
        about_text = (f'<h3 style="text-align:center">Version {__editor_version__}</h3>'
                      '<p style="text-align:center">This is a tool for modifying instrument '
                      'description files for SScanSS 2.</p>'
                      '<p style="text-align:center">Distributed under the BSD 3-Clause License</p>'
                      f'<p style="text-align:center">Copyright &copy; 2018-{datetime.date.today().year}, '
                      'ISIS Neutron and Muon Source. All rights reserved.</p>')

        QtWidgets.QMessageBox.about(self, title, about_text)

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

    def updateTitle(self):
        """Sets new title based on currently selected file"""
        if self.presenter.model.filename:
            self.setWindowTitle(f'{self.presenter.model.filename} - {MAIN_WINDOW_TITLE}')
        else:
            self.setWindowTitle(MAIN_WINDOW_TITLE)

    def sceneSizeErrorHandler(self):
        self.showMessage(
            'The scene is too big, the distance from the origin exceeds '
            f'{self.gl_widget.scene.max_extent}mm.', )

    def showDocumentation(self):
        """Opens the documentation in the system's default browser"""
        webbrowser.open_new(f'https://isisneutronmuon.github.io/SScanSS-2/{__version__}/api.html')
