import logging
import os
import sys
from jsonschema.exceptions import ValidationError
from PyQt5 import QtCore, QtGui, QtWidgets
from editor.ui.editor import Editor
import editor.ui.resource
from editor.ui.scene_manager import SceneManager
from editor.ui.widgets import ScriptWidget, JawsWidget, PositionerWidget, DetectorWidget
from sscanss.config import setup_logging
from sscanss.core.math import clamp
from sscanss.core.instrument import read_instrument_description
from sscanss.core.scene import Node
from sscanss.core.util import Directions

from sscanss.ui.widgets import GLWidget, create_scroll_area


MAIN_WINDOW_TITLE = 'Instrument Editor'


class InstrumentWorker(QtCore.QThread):
    """Creates worker thread object

    :param _exec: function to run on ``QThread``
    :type _exec: Callable[..., Any]
    :param args: arguments of function ``_exec``
    :type args: Tuple[Any, ...]
    """
    job_succeeded = QtCore.pyqtSignal(object)
    job_failed = QtCore.pyqtSignal(Exception)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.job_succeeded.connect(self.parent.setInstrumentSuccess)
        self.job_failed.connect(self.parent.setInstrumentFailed)
    
    def run(self):
        """This function is executed on  worker thread when the ``QThread.start``
        method is called."""
        try:
            result = self.parent.setInstrument()
            self.job_succeeded.emit(result)
        except Exception as e:
            self.job_failed.emit(e)


class Controls(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.setWindowTitle('Instrument Control')

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(600)
        self.tabs.setMinimumHeight(600)
        self.tabs.tabBarClicked.connect(self.updateTabs)
        layout.addWidget(self.tabs)
        
        self.last_tab_index = 0
        self.last_stack_name = ''
        self.last_collimator_name = {}
        
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        
    def createWidgets(self):
        self.tabs.clear()
        positioner_widget = PositionerWidget(self.parent)
        if self.last_stack_name and self.last_stack_name in self.parent.instrument.positioning_stacks.keys():
            positioner_widget.changeStack(self.last_stack_name)
        positioner_widget.stack_combobox.activated[str].connect(self.setStack)

        self.tabs.addTab(create_scroll_area(positioner_widget), 'Positioner')
        self.tabs.addTab(create_scroll_area(JawsWidget(self.parent)), 'Jaws')

        collimator_names = {}
        for name in self.parent.instrument.detectors:
            pretty_name = name if name.lower() == 'detector' else f'{name} Detector' 
            detector_widget = DetectorWidget(self.parent, name)
            self.tabs.addTab(create_scroll_area(detector_widget),  pretty_name)
            collimator_name = self.last_collimator_name.get(name, '')
            if collimator_name:
                collimator_names[name] = collimator_name
                detector_widget.combobox.setCurrentText(collimator_name) 
                detector_widget.changeCollimator()
            detector_widget.collimator_changed.connect(self.setCollimator)
        self.last_collimator_name = collimator_names

        self.script_widget = ScriptWidget(self.parent)
        self.tabs.addTab(create_scroll_area(self.script_widget), 'Script')
        self.tabs.setCurrentIndex(clamp(self.last_tab_index, 0, self.tabs.count()))
    
    def reset(self):
        self.last_tab_index = 0
        self.last_stack_name = ''
        self.last_collimator_name = {}
    
    def setStack(self, name):
        self.last_stack_name = name

    def setCollimator(self, detector, name):
        self.last_collimator_name[detector] = name

    def updateTabs(self, index):
        self.last_tab_index = index
        if self.tabs.tabText(index) == 'Script':
            self.script_widget.updateScript()


class Window(QtWidgets.QMainWindow):
    animate_instrument = QtCore.pyqtSignal(object, object, object, int, int)

    def __init__(self):
        super().__init__()
        
        self.filename = ''
        self.saved_text = ''
        self.initialized = False
        self.file_watcher = QtCore.QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(lambda: self.lazyInstrumentUpdate())
        self.manager = SceneManager(self)        
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

        self.gl_widget = GLWidget(self)
        self.gl_widget.custom_error_handler = self.sceneSizeErrorHandler
        self.splitter.addWidget(self.gl_widget)

        self.editor = Editor()
        self.editor.textChanged.connect(self.lazyInstrumentUpdate)
        self.splitter.addWidget(self.editor)
        
        self.setMinimumSize(1024, 800)
        self.setTitle()
        self.setWindowIcon(QtGui.QIcon(":/images/logo.ico"))

        self.initActions()
        self.initMenus()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.useWorker)
        self.worker = InstrumentWorker(self)
    
    def setTitle(self):
        if self.filename:
            self.setWindowTitle(f'{self.filename} - {MAIN_WINDOW_TITLE}')
        else:
            self.setWindowTitle(MAIN_WINDOW_TITLE)

    @property
    def unsaved(self):
        return self.editor.text() != self.saved_text

    def initActions(self):
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

        self.reload_action = QtWidgets.QAction('&Reset Instrument', self)
        self.reload_action.setStatusTip('Reset Instrument')
        self.reload_action.triggered.connect(self.resetInstrument)
        self.reload_action.setShortcut(QtGui.QKeySequence('F5'))

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
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.exit_action)

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

        help_menu = menu_bar.addMenu('&Help')
        help_menu.addAction(self.show_documentation_action)
        help_menu.addAction(self.about_action)

    def updateWatcher(self, path=''):
        if self.file_watcher.directories():
            self.file_watcher.removePaths(self.file_watcher.directories())
        if path:
            self.file_watcher.addPaths([path, *[f.path for f in os.scandir(path) if f.is_dir()]])

    def newFile(self):
        if self.unsaved and not self.showSaveDiscardMessage():
            return

        self.saved_text = ''
        self.editor.setText(self.saved_text)
        self.filename = ''
        self.setTitle()
        self.initialized = False 
        self.updateWatcher()
        self.manager.reset()
        self.controls.close()
        self.message.setText('')

    def openFile(self):
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
        self.controls.reset()
        self.useWorker()

    def lazyInstrumentUpdate(self, interval=300):
        self.initialized = True
        text = self.editor.text()
        self.timer.stop()
        self.timer.setSingleShot(True)
        self.timer.setInterval(interval)
        self.timer.start()

    def useWorker(self):
        if self.worker is not None and self.worker.isRunning():
            self.lazyInstrumentUpdate(100)
            return

        self.worker.start()

    def setInstrument(self):
        """Change current instrument to specified"""
        return read_instrument_description(self.editor.text(), os.path.dirname(self.filename))
        
    def setInstrumentSuccess(self, result):
        self.message.setText('OK') 
        self.instrument = result
        self.controls.createWidgets()
        self.manager.updateInstrumentScene()

    def setInstrumentFailed(self, e):
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

    def closeEvent(self, event):
        if not self.unsaved:
            event.accept()
            return
        
        if not self.showSaveDiscardMessage():
            event.ignore()
            return
        
        event.accept()

    def about(self):
        import datetime
        QtWidgets.QMessageBox.about(self, f'About {MAIN_WINDOW_TITLE}',
                ('<h3 style="text-align:center">Version 1.0-alpha</h3>'
                 '<p style="text-align:center">This is a tool for modifying instrument description files for SScanSS-2.</p>'
                 '<p style="text-align:center">Designed by Stephen Nneji</p>'
                 '<p style="text-align:center">Distributed under the BSD 3-Clause License</p>'
                 f'<p style="text-align:center">Copyright &copy; 2018-{datetime.date.today().year}, '
                 'ISIS Neutron and Muon Source. All rights reserved.</p>'))

    def showSaveDiscardMessage(self):
        """Shows an message to confirm if unsaved changes should be saved."""
        
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
        """
        This function opens the documentation html in the system's default browser
        """
        import webbrowser
        webbrowser.open_new('https://isisneutronmuon.github.io/SScanSS-2/api.html')

    def sceneSizeErrorHandler(self):
        self.message.setText('The scene is too big the distance from the origin exceeds '
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
    app.setStyleSheet(style)
    w = Window()
    w.show()
    exit_code = app.exec_()
    logging.shutdown()
    sys.exit(exit_code)
