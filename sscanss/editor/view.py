# !-! Clean the imports after refactoring is done
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.config import path_for
from sscanss.core.instrument import Sequence
from sscanss.core.io import read_kinematic_calibration_file
from sscanss.core.scene import OpenGLRenderer, SceneManager
from sscanss.core.util import Directions, Attributes
from sscanss.core.util.misc import MessageReplyType
from sscanss.editor.dialogs import CalibrationWidget, Controls, FindWidget
from sscanss.editor.editor import Editor
from sscanss.editor.presenter import EditorPresenter

class EditorWindow(QtWidgets.QMainWindow):
    """Creates the main window of the instrument editor."""
    animate_instrument = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.presenter = EditorPresenter(self)

        self.controls = Controls(self)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.setCentralWidget(self.main_splitter)
        self.filename = ''
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.splitter)
        self.main_splitter.setStretchFactor(0, 2)

        self.message = QtWidgets.QTextEdit('')
        self.message.setReadOnly(True)
        self.message.setFontFamily('courier')
        self.message.setStyleSheet('QTextEdit {background-color: white; color: red; font-size: 12px; '
                                   'padding-left: 10px; background-position: top left}')
        self.message.setMinimumHeight(100)
        self.main_splitter.addWidget(self.message)

        self.gl_widget = OpenGLRenderer(self)
        self.gl_widget.custom_error_handler = self.sceneSizeErrorHandler
        self.scene = SceneManager(self, self.gl_widget, False)
        self.scene.changeVisibility(Attributes.Beam, True)
        self.animate_instrument.connect(self.scene.animateInstrument)

        self.editor = Editor(self)
        self.editor.textChanged.connect(self.presenter.updateInstrument)
        self.splitter.addWidget(self.gl_widget)
        self.splitter.addWidget(self.editor)

        self.setMinimumSize(1024, 800)
        self.setWindowIcon(QtGui.QIcon(path_for('editor-logo.png')))

        self.initActions()
        self.initMenus()

        self.presenter.parseLaunchArguments()

    def showSearchBox(self):
        """Opens the find dialog box"""
        self.find_dialog = FindWidget(self)
        self.find_dialog.fist_search_flag = True
        self.find_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.find_dialog.show()

    def setTitle(self, newTitle):
        """Sets main window title"""
        self.setWindowTitle(newTitle)

    def setMessageText(self, text):
        self.message.setText(text)

    def setInstrument(self, instrument):
        self.instrument = instrument

    def getEditorText(self):
        return self.editor.text()

    def setEditorText(self, text):
        self.editor.setText(text)

    def resetControls(self):
        self.controls.reset()

    def updateScene(self):
        self.scene.updateInstrumentScene()

    def resetScene(self):
        self.scene.reset()

    def showCoordinateFrame(self, switch):
        self.gl_widget.showCoordinateFrame(switch)

    def resetCamera(self):
        self.gl_widget.resetCamera()

    def initActions(self):
        """Creates menu actions"""
        """Creates menu actions"""
        self.exit_action = QtWidgets.QAction('&Quit', self)
        self.exit_action.setShortcut(QtGui.QKeySequence.Quit)
        self.exit_action.setStatusTip('Exit application')
        self.exit_action.triggered.connect(self.close)

        self.new_action = QtWidgets.QAction('&New File', self)
        self.new_action.setShortcut(QtGui.QKeySequence.New)
        self.new_action.setStatusTip('New Instrument Description File')
        self.new_action.triggered.connect(self.presenter.createNewFile)

        self.open_action = QtWidgets.QAction('&Open File', self)
        self.open_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_action.setStatusTip('Open Instrument Description File')
        self.open_action.triggered.connect(self.presenter.openFile)

        self.save_action = QtWidgets.QAction('&Save File', self)
        self.save_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_action.setStatusTip('Save Instrument Description File')
        self.save_action.triggered.connect(self.presenter.saveFile)

        self.save_as_action = QtWidgets.QAction('Save &As...', self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.presenter.saveFile(save_as=True))

        self.show_instrument_controls = QtWidgets.QAction('&Instrument Controls', self)
        self.show_instrument_controls.setStatusTip('Show Instrument Controls')
        self.show_instrument_controls.triggered.connect(self.presenter.showInstrumentControls)

        self.generate_robot_model_action = QtWidgets.QAction('&Generate Robot Model', self)
        self.generate_robot_model_action.setStatusTip('Generate Robot Model from Measurements')
        self.generate_robot_model_action.triggered.connect(self.presenter.generateRobotModel)

        self.reload_action = QtWidgets.QAction('&Reset Instrument', self)
        self.reload_action.setStatusTip('Reset Instrument')
        self.reload_action.triggered.connect(self.presenter.resetInstrumentControls)
        self.reload_action.setShortcut(QtGui.QKeySequence('F5'))

        self.find_action = QtWidgets.QAction('&Find', self)
        self.find_action.setStatusTip('Find text in editor')
        self.find_action.triggered.connect(self.showSearchBox)
        self.find_action.setShortcut(QtGui.QKeySequence('Ctrl+F'))

        self.show_world_coordinate_frame_action = QtWidgets.QAction('Toggle &World Coordinate Frame', self)
        self.show_world_coordinate_frame_action.setStatusTip('Toggle world coordinate frame')
        self.show_world_coordinate_frame_action.setCheckable(True)
        self.show_world_coordinate_frame_action.setChecked(self.gl_widget.show_coordinate_frame)
        self.show_world_coordinate_frame_action.toggled.connect(self.gl_widget.showCoordinateFrame)
        
        self.reset_camera_action = QtWidgets.QAction('Reset &View', self)
        self.reset_camera_action.setStatusTip('Reset camera view')
        self.reset_camera_action.triggered.connect(self.presenter.resetCamera)
        self.reset_camera_action.setShortcut(QtGui.QKeySequence('Ctrl+0'))

        self.show_documentation_action = QtWidgets.QAction('&Documentation', self)
        self.show_documentation_action.setStatusTip('Show online documentation')
        self.show_documentation_action.setShortcut('F1')
        self.show_documentation_action.triggered.connect(self.presenter.showDocumentation)

        self.about_action = QtWidgets.QAction('&About', self)
        self.about_action.setStatusTip(f'About {self.presenter.windowName}')
        self.about_action.triggered.connect(self.presenter.showAboutMessage)

    def initMenus(self):
        """Creates main menu and sub menus"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.exit_action)

        edit_menu = menu_bar.addMenu('&Edit')
        edit_menu.addAction(self.find_action)

        view_menu = menu_bar.addMenu('&View')
        view_menu.addAction(self.reload_action)
        view_menu.addAction(self.show_instrument_controls)
        view_menu.addSeparator()
        self.view_from_menu = view_menu.addMenu('View From')
        for index, direction in enumerate(Directions):
            view_from_action = QtWidgets.QAction(direction.value, self)
            view_from_action.setStatusTip(f'View scene from the {direction.value} axis')
            view_from_action.setShortcut(QtGui.QKeySequence(f'Ctrl+{index+1}'))
            view_from_action.triggered.connect(lambda ignore, d=direction: self.gl_widget.viewFrom(d))
            self.view_from_menu.addAction(view_from_action)

        view_menu.addAction(self.reset_camera_action)
        view_menu.addAction(self.show_world_coordinate_frame_action)

        tool_menu = menu_bar.addMenu('&Tool')
        tool_menu.addAction(self.generate_robot_model_action)

        help_menu = menu_bar.addMenu('&Help')
        help_menu.addAction(self.show_documentation_action)
        help_menu.addAction(self.show_documentation_action)
        help_menu.addAction(self.about_action)

    def askAddress(self, caption, directory, filter):
        """Creates new window allowing the user to choose the file location
        :param caption: caption in the window
        :type caption: str
        :param directory: the starting directory
        :type directory: str
        :param filter: filter to sort the file types
        :type filter: str
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, directory,
                                                            filter)
        return filename

    def createCalibrationWidget(self, points, types, offsets, homes):
        widget = CalibrationWidget(self, points, types, offsets, homes)
        widget.show()

    def showControls(self):
        self.controls.show()

    def createInstrumentControls(self):
        self.controls.createWidgets()

    def hideControls(self):
        self.controls.close()

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

    def closeEvent(self, event):
        """Closes window based on presentor's response"""
        if self.presenter.exitApplication():
            event.accept()
        else:
            event.ignore()

    def showAboutMessage(self, title, text):
        QtWidgets.QMessageBox.about(self, title, text)

    def showSaveDiscardMessage(self, message):
        """Shows a message to confirm if unsaved changes should be saved or discarded"""

        buttons = QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
        reply = QtWidgets.QMessageBox.warning(self, self.presenter.windowName, message, buttons, QtWidgets.QMessageBox.Cancel)

        if reply == QtWidgets.QMessageBox.Save:
            return MessageReplyType.Save
        elif reply == QtWidgets.QMessageBox.Discard:
            return MessageReplyType.Discard
        else:
            return MessageReplyType.Cancel

    def sceneSizeErrorHandler(self):
        self.message.setText('The scene is too big, the distance from the origin exceeds '
                             f'{self.gl_widget.scene.max_extent}mm.')

