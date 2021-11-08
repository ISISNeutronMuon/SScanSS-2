import datetime
import logging
import os
import pathlib
import sys
import webbrowser
from jsonschema.exceptions import ValidationError
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.config import setup_logging, __editor_version__, __version__
from sscanss.core.instrument import read_instrument_description, Sequence
from sscanss.core.io import read_kinematic_calibration_file
from sscanss.core.scene import OpenGLRenderer, SceneManager
from sscanss.core.util import Directions, Attributes
from sscanss.editor.dialogs import CalibrationWidget, Controls, FindWidget
from sscanss.editor.editor import Editor


MAIN_WINDOW_TITLE = 'Instrument Editor'


class InstrumentWorker(QtCore.QThread):
    """Creates worker thread for updating instrument from the description file.

    :param parent: main window instance
    :type parent: MainWindow
    """
    job_succeeded = QtCore.pyqtSignal(object)
    job_failed = QtCore.pyqtSignal(Exception)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.job_succeeded.connect(self.parent.setInstrumentSuccess)
        self.job_failed.connect(self.parent.setInstrumentFailed)
    
    def run(self):
        """Updates instrument from description file"""
        try:
            result = self.parent.setInstrument()
            self.job_succeeded.emit(result)
        except Exception as e:
            self.job_failed.emit(e)


class Window(QtWidgets.QMainWindow):
    """Creates the main window of the instrument editor."""
    animate_instrument = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        
        self.filename = ''
        self.saved_text = ''
        self.initialized = False
        self.file_watcher = QtCore.QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(lambda: self.lazyInstrumentUpdate())
        self.controls = Controls(self)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.setCentralWidget(self.main_splitter)

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
        self.editor.textChanged.connect(self.lazyInstrumentUpdate)
        self.splitter.addWidget(self.gl_widget)
        self.splitter.addWidget(self.editor)

        self.setMinimumSize(1024, 800)
        self.setTitle()
        self.setWindowIcon(QtGui.QIcon(":/images/editor-logo.ico"))

        self.initActions()
        self.initMenus()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.useWorker)
        self.worker = InstrumentWorker(self)

    def showSearchBox(self):
        """Opens the find dialog box"""
        self.find_dialog = FindWidget(self)
        self.find_dialog.fist_search_flag = True
        self.find_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.find_dialog.show()

    def setTitle(self):
        """Sets main window title"""
        if self.filename:
            self.setWindowTitle(f'{self.filename} - {MAIN_WINDOW_TITLE}')
        else:
            self.setWindowTitle(MAIN_WINDOW_TITLE)

    @property
    def unsaved(self):
        return self.editor.text() != self.saved_text

    def initActions(self):
        """Creates menu actions"""
        self.exit_action = QtWidgets.QAction('&Quit', self)
        self.exit_action.setShortcut(QtGui.QKeySequence.Quit)
        self.exit_action.setStatusTip('Exit application')
        self.exit_action.triggered.connect(self.close)

        self.new_action = QtWidgets.QAction('&New File', self)
        self.new_action.setShortcut(QtGui.QKeySequence.New)
        self.new_action.setStatusTip('New Instrument Description File')
        self.new_action.triggered.connect(self.newFile)
        
        self.open_action = QtWidgets.QAction('&Open File', self)
        self.open_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_action.setStatusTip('Open Instrument Description File')
        self.open_action.triggered.connect(self.openFile)

        self.save_action = QtWidgets.QAction('&Save File', self)
        self.save_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_action.setStatusTip('Save Instrument Description File')
        self.save_action.triggered.connect(self.saveFile)

        self.save_as_action = QtWidgets.QAction('Save &As...', self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.saveFile(save_as=True))

        self.show_instrument_controls = QtWidgets.QAction('&Instrument Controls', self)
        self.show_instrument_controls.setStatusTip('Show Instrument Controls')
        self.show_instrument_controls.triggered.connect(self.controls.show)

        self.generate_robot_model_action = QtWidgets.QAction('&Generate Robot Model', self)
        self.generate_robot_model_action.setStatusTip('Generate Robot Model from Measurements')
        self.generate_robot_model_action.triggered.connect(self.generateRobotModel)

        self.reload_action = QtWidgets.QAction('&Reset Instrument', self)
        self.reload_action.setStatusTip('Reset Instrument')
        self.reload_action.triggered.connect(self.resetInstrument)
        self.reload_action.setShortcut(QtGui.QKeySequence('F5'))

        self.find_action = QtWidgets.QAction('&Find', self)
        self.find_action.setStatusTip('Find text in editor')
        self.find_action.triggered.connect(self.showSearchBox())
        self.find_action.setShortcut(QtGui.QKeySequence('Ctrl+F'))


        self.show_world_coordinate_frame_action = QtWidgets.QAction('Toggle &World Coordinate Frame', self)
        self.show_world_coordinate_frame_action.setStatusTip('Toggle world coordinate frame')
        self.show_world_coordinate_frame_action.setCheckable(True)
        self.show_world_coordinate_frame_action.setChecked(self.gl_widget.show_coordinate_frame)
        self.show_world_coordinate_frame_action.toggled.connect(self.gl_widget.showCoordinateFrame)

        self.reset_camera_action = QtWidgets.QAction('Reset &View', self)
        self.reset_camera_action.setStatusTip('Reset camera view')
        self.reset_camera_action.triggered.connect(self.gl_widget.resetCamera)
        self.reset_camera_action.setShortcut(QtGui.QKeySequence('Ctrl+0'))

        self.show_documentation_action = QtWidgets.QAction('&Documentation', self)
        self.show_documentation_action.setStatusTip('Show online documentation')
        self.show_documentation_action.setShortcut('F1')
        self.show_documentation_action.triggered.connect(self.showDocumentation)

        self.about_action = QtWidgets.QAction(f'&About', self)
        self.about_action.setStatusTip(f'About {MAIN_WINDOW_TITLE}')
        self.about_action.triggered.connect(self.about)

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
            action = self.gl_widget.viewFrom
            view_from_action.triggered.connect(lambda ignore, d=direction: action(d))
            self.view_from_menu.addAction(view_from_action)

        view_menu.addAction(self.reset_camera_action)
        view_menu.addAction(self.show_world_coordinate_frame_action)

        tool_menu = menu_bar.addMenu('&Tool')
        tool_menu.addAction(self.generate_robot_model_action)

        help_menu = menu_bar.addMenu('&Help')
        help_menu.addAction(self.show_documentation_action)
        help_menu.addAction(self.show_documentation_action)
        help_menu.addAction(self.about_action)

    def generateRobotModel(self):
        """Generates kinematic model of a positioning system from measurements"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Kinematic Calibration File', '',
                                                            'Supported Files (*.csv *.txt)')

        if not filename:
            return

        try:
            points, types, offsets, homes = read_kinematic_calibration_file(filename)
            widget = CalibrationWidget(self, points, types, offsets, homes)
            widget.show()
        except OSError as e:
            self.message.setText(f'An error occurred while attempting to open this file ({filename}). \n{e}')

    def updateWatcher(self, path):
        """Adds path to the file watcher, which monitors the path for changes to
        model or template files.

        :param path: file path of the instrument description file
        :type path: str
        """
        if self.file_watcher.directories():
            self.file_watcher.removePaths(self.file_watcher.directories())
        if path:
            self.file_watcher.addPaths([path, *[f.path for f in os.scandir(path) if f.is_dir()]])

    def newFile(self):
        """Creates a new instrument description file"""
        if self.unsaved and not self.showSaveDiscardMessage():
            return

        self.saved_text = ''
        self.editor.setText(self.saved_text)
        self.filename = ''
        self.setTitle()
        self.initialized = False 
        self.updateWatcher(self.filename)
        self.scene.reset()
        self.controls.close()
        self.message.setText('')

    def openFile(self, filename=''):
        """Loads an instrument description file from a given file path. If filename
        is empty, a file dialog will be opened

        :param filename: full path of file
        :type filename: str
        """

        if self.unsaved and not self.showSaveDiscardMessage():
            return

        if not filename:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Instrument Description File', '',
                                                                'Json File (*.json)')

            if not filename:
                return

        try:
            with open(filename, 'r') as idf:
                self.filename = filename
                self.saved_text = idf.read()
                self.setTitle()  
                self.updateWatcher(os.path.dirname(filename))
                self.editor.setText(self.saved_text)
        except OSError as e:
            self.message.setText(f'An error occurred while attempting to open this file ({filename}). \n{e}')

    def saveFile(self, save_as=False):
        """Saves the instrument description file. A file dialog should be opened for the first save
        after which the function will save to the same location. If save_as is True a dialog is
        opened every time

        :param save_as: A flag denoting whether to use file dialog or not
        :type save_as: bool
        """
        if not self.unsaved and not save_as:
            return

        filename = self.filename
        if save_as or not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Instrument Description File', '',
                                                                'Json File (*.json)')

        if not filename:
            return

        try:
            with open(filename, 'w') as idf:
                self.filename = filename
                text = self.editor.text()
                idf.write(text)
                self.saved_text = text
                self.updateWatcher(os.path.dirname(filename))
                self.setTitle()
            if save_as:
                self.resetInstrument()
        except OSError as e:
            self.message.setText(f'An error occurred while attempting to save this file ({filename}). \n{e}')
    
    def resetInstrument(self):
        """Resets control dialog and updates instrument to reflect change"""
        self.controls.reset()
        self.useWorker()

    def lazyInstrumentUpdate(self, interval=300):
        """Updates instrument after the wait time elapses

        :param interval: wait time (milliseconds)
        :type interval: int
        """
        self.initialized = True
        self.timer.stop()
        self.timer.setSingleShot(True)
        self.timer.setInterval(interval)
        self.timer.start()

    def useWorker(self):
        """Uses worker thread to create instrument from description"""
        if self.worker is not None and self.worker.isRunning():
            self.lazyInstrumentUpdate(100)
            return

        self.worker.start()

    def setInstrument(self):
        """Creates an instrument from the description file."""
        return read_instrument_description(self.editor.text(), os.path.dirname(self.filename))
        
    def setInstrumentSuccess(self, result):
        """Sets the instrument created from the instrument file.

        :param result: instrument from description file
        :type result: Instrument
        """
        self.message.setText('OK') 
        self.instrument = result
        self.controls.createWidgets()
        self.scene.updateInstrumentScene()

    def setInstrumentFailed(self, e):
        """Reports errors from instrument update worker

        :param e: raised exception
        :type e: Exception
        """
        self.controls.tabs.clear()
        if self.initialized:
            if isinstance(e, ValidationError):
                path = ''
                for p in e.absolute_path:
                    if isinstance(p, int):
                        path = f'{path}[{p}]'
                    else:
                        path = f'{path}.{p}' if path else p

                path = path if path else 'instrument description file'
                m = f'{e.message} in {path}'
            else:
                m = str(e).strip("'")
            self.message.setText(m)

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
        if not self.unsaved:
            event.accept()
            return
        
        if not self.showSaveDiscardMessage():
            event.ignore()
            return
        
        event.accept()

    def about(self):
        about_text = (f'<h3 style="text-align:center">Version {__editor_version__}</h3>'
                      '<p style="text-align:center">This is a tool for modifying instrument '
                      'description files for SScanSS 2.</p>'
                      '<p style="text-align:center">Designed by Stephen Nneji</p>'
                      '<p style="text-align:center">Distributed under the BSD 3-Clause License</p>'
                      f'<p style="text-align:center">Copyright &copy; 2018-{datetime.date.today().year}, '
                      'ISIS Neutron and Muon Source. All rights reserved.</p>')
        QtWidgets.QMessageBox.about(self, f'About {MAIN_WINDOW_TITLE}', about_text)

    def showSaveDiscardMessage(self):
        """Shows a message to confirm if unsaved changes should be saved or discarded"""
        message = f'The document has been modified.\n\nDo you want to save changes to "{self.filename}"?'
        buttons = QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
        reply = QtWidgets.QMessageBox.warning(self, MAIN_WINDOW_TITLE,
                                              message, buttons, QtWidgets.QMessageBox.Cancel)

        if reply == QtWidgets.QMessageBox.Save:
            self.saveFile()
            return True
        if reply == QtWidgets.QMessageBox.Discard:
            return True
        else:
            return False

    def showDocumentation(self):
        """Opens the documentation in the system's default browser"""
        webbrowser.open_new(f'https://isisneutronmuon.github.io/SScanSS-2/{__version__}/api.html')

    def sceneSizeErrorHandler(self):
        self.message.setText('The scene is too big, the distance from the origin exceeds '
                             f'{self.gl_widget.scene.max_extent}mm.')


style = """* {
        font-family:"Helvetica Neue", Helvetica, Arial;
        font-size: 12px;
        color: #333;
    }

    QDialog{
        border: 1px solid #ddd;
    }


    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox{
        padding: 5px;
    }

    QPushButton{
        padding: 10px 15px;
    }
"""


if __name__ == '__main__':
    setup_logging('editor.log')
    app = QtWidgets.QApplication([])
    app.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)
    app.setStyleSheet(style)
    window = Window()

    if sys.argv[1:]:
        file_path = sys.argv[1]
        if pathlib.PurePath(file_path).suffix == '.json':
            window.openFile(file_path)
        else:
            window.message.setText(f'{file_path} could not be opened because it has an unknown file type')

    window.show()
    exit_code = app.exec_()
    logging.shutdown()
    sys.exit(exit_code)
