from PyQt5 import QtCore, QtGui, QtWidgets
from .presenter import MainWindowPresenter, MessageReplyType
from sscanss.ui.dialogs.project.view import ProjectDialog
from sscanss.ui.dialogs.progress.view import ProgressDialog
from sscanss.ui.widgets.opengl.view import GLWidget
from sscanss.core.util import RenderType

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

        self.createActions()
        self.createMenus()
        self.createToolBar()

        self.gl_widget = GLWidget(self)
        self.setCentralWidget(self.gl_widget)

        self.setWindowTitle(MAIN_WINDOW_TITLE)
        self.setMinimumSize(800, 600)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.settings = QtCore.QSettings(
            QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, 'SScanSS 2', 'SScanSS 2')

        self.readSettings()

    def createActions(self):
        self.new_project_action = QtWidgets.QAction('&New Project', self)
        self.new_project_action.setIcon(QtGui.QIcon('../static/images/file.png'))
        self.new_project_action.setShortcut(QtGui.QKeySequence.New)
        self.new_project_action.triggered.connect(self.showNewProjectDialog)

        self.open_project_action = QtWidgets.QAction('&Open Project', self)
        self.open_project_action.setIcon(QtGui.QIcon('../static/images/folder-open.png'))
        self.open_project_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_project_action.triggered.connect(self.presenter.openProject)

        self.save_project_action = QtWidgets.QAction('&Save Project', self)
        self.save_project_action.setIcon(QtGui.QIcon('../static/images/save.png'))
        self.save_project_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_project_action.triggered.connect(lambda: self.presenter.saveProject())

        self.save_as_action = QtWidgets.QAction('Save &As...', self)
        self.save_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.presenter.saveProject(save_as=True))

        self.exit_action = QtWidgets.QAction('E&xit', self)
        self.exit_action.setShortcut(QtGui.QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)

        # Edit Menu Actions
        self.undo_action = self.undo_stack.createUndoAction(self, '&Undo')
        self.undo_action.setIcon(QtGui.QIcon('../static/images/undo.png'))
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)

        self.redo_action = self.undo_stack.createRedoAction(self, '&Redo')
        self.redo_action.setIcon(QtGui.QIcon('../static/images/redo.png'))
        self.redo_action.setShortcut(QtGui.QKeySequence.Redo)

        # View Menu Actions
        self.solid_render_action = QtWidgets.QAction(RenderType.Solid.value, self)
        self.solid_render_action.triggered.connect(lambda: self.presenter.toggleRenderType(RenderType.Solid))
        self.solid_render_action.setCheckable(True)
        self.solid_render_action.setChecked(True)

        self.line_render_action = QtWidgets.QAction(RenderType.Wireframe.value, self)
        self.line_render_action.triggered.connect(lambda: self.presenter.toggleRenderType(RenderType.Wireframe))
        self.line_render_action.setCheckable(True)

        self.blend_render_action = QtWidgets.QAction(RenderType.Transparent.value, self)
        self.blend_render_action.triggered.connect(lambda: self.presenter.toggleRenderType(RenderType.Transparent))
        self.blend_render_action.setCheckable(True)

        self.render_action_group = QtWidgets.QActionGroup(self)
        self.render_action_group.addAction(self.solid_render_action)
        self.render_action_group.addAction(self.line_render_action)
        self.render_action_group.addAction(self.blend_render_action)

        self.import_sample_action = QtWidgets.QAction('File...', self)
        self.import_sample_action.triggered.connect(self.presenter.importSample)

    def createMenus(self):
        main_menu = self.menuBar()

        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        self.recent_menu = file_menu.addMenu('Open &Recent')
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        file_menu.aboutToShow.connect(self.populateRecentMenu)

        edit_menu = main_menu.addMenu('&Edit')
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        view_menu = main_menu.addMenu('&View')
        view_menu.addAction(self.solid_render_action)
        view_menu.addAction(self.line_render_action)
        view_menu.addAction(self.blend_render_action)

        insert_menu = main_menu.addMenu('&Insert')
        sample_menu = insert_menu.addMenu('Sample')
        sample_menu.addAction(self.import_sample_action)
        primitives_menu = sample_menu.addMenu('Primitives')

        instrument_menu = main_menu.addMenu('I&nstrument')
        simulation_menu = main_menu.addMenu('Sim&ulation')
        help_menu = main_menu.addMenu('&Help')

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

    def readSettings(self):
        """ Loads window geometry from INI file """
        self.restoreGeometry(self.settings.value('geometry', bytearray(b'')))
        self.restoreState(self.settings.value('windowState', bytearray(b'')))
        self.recent_projects = self.settings.value('recentProjects', [])

    def populateRecentMenu(self):
        self.recent_menu.clear()
        for project in self.recent_projects:
            recent_project_action = QtWidgets.QAction(project, self)
            recent_project_action.triggered.connect(lambda ignore, p=project: self.presenter.openProject(filename=p))
            self.recent_menu.addAction(recent_project_action)

    def closeEvent(self, event):
        if self.presenter.confirmSave():
            self.settings.setValue('geometry', self.saveGeometry())
            self.settings.setValue('windowState', self.saveState())
            if self.recent_projects:
                self.settings.setValue('recentProjects', self.recent_projects)
            event.accept()
        else:
            event.ignore()

    def showProgressDialog(self, message):

        self.progress_dialog = ProgressDialog(message, parent=self)
        self.progress_dialog.setModal(True)
        self.progress_dialog.show()

    def showNewProjectDialog(self):

        if self.presenter.confirmSave():
            self.project_dialog = ProjectDialog(self.recent_projects, parent=self)
            self.project_dialog.setModal(True)
            self.project_dialog.show()

    def showProjectName(self, project_name):
        title = '{} - {}'.format(project_name, MAIN_WINDOW_TITLE)
        self.setWindowTitle(title)

    def showSaveDialog(self, filters, current_dir=''):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self,
                                                            'Save Project',
                                                            current_dir,
                                                            filters)
        return filename

    def showOpenDialog(self, filters, current_dir=''):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                            'Open Project',
                                                            current_dir,
                                                            filters)
        return filename

    def showErrorMessage(self, message):
        """
        Shows an error message.

        :param message: Error message string
        """
        QtWidgets.QMessageBox.critical(self, 'Error', message)

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
