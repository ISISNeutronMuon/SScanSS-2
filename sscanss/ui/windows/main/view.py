from PyQt5 import QtCore, QtGui, QtWidgets
from .presenter import MainWindowPresenter


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.presenter = MainWindowPresenter(self)

        self.undo_stack = QtWidgets.QUndoStack(self)
        self.undo_view = QtWidgets.QUndoView(self.undo_stack)
        self.undo_view.setWindowTitle('History')
        self.undo_view.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

        self.createActions()
        self.createMenus()

        self.setWindowTitle('SScanSS 2')
        self.setMinimumSize(800, 600)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.settings = QtCore.QSettings(
            QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, 'SScanSS 2', 'SScanSS 2')
        self.readSettings()

    def createActions(self):
        self.new_project_action = QtWidgets.QAction('&New Project', self)
        self.new_project_action.setShortcut(QtGui.QKeySequence.New)

        self.open_project_action = QtWidgets.QAction('&Open Project', self)
        self.open_project_action.setShortcut(QtGui.QKeySequence.Open)

        # self.open_recent_action = QtWidgets.QAction('Open Recent', self)

        self.save_project_action = QtWidgets.QAction('&Save Project', self)
        self.save_project_action.setShortcut(QtGui.QKeySequence.Save)

        self.exit_action = QtWidgets.QAction('E&xit', self)
        self.exit_action.setShortcut(QtGui.QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)

    def createMenus(self):
        main_menu = self.menuBar()

        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)


        edit_menu = main_menu.addMenu('&Edit')
        view_menu = main_menu.addMenu('&View')
        insert_menu = main_menu.addMenu('&Insert')
        instrument_menu = main_menu.addMenu('I&nstrument')
        simulation_menu = main_menu.addMenu('Sim&ulation')
        help_menu = main_menu.addMenu('&Help')

    def readSettings(self):
        """ Loads window geometry from INI file """
        self.restoreGeometry(self.settings.value('geometry', bytearray(b'')))
        self.restoreState(self.settings.value('windowState', bytearray(b'')))

    def closeEvent(self, event):
        """Override of the QWidget Close Event"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        super().closeEvent(event)