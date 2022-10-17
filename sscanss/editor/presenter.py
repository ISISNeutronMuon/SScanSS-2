import logging
from sscanss.core.util.misc import MessageReplyType
from sscanss.editor.model import EditorModel, InstrumentWorker
from sscanss.core.io import read_kinematic_calibration_file
from sscanss.core.instrument import InstrumentParser

MAIN_WINDOW_TITLE = 'Instrument Editor'


class EditorPresenter:
    """Main presenter for the editor app

    :param view: main window instance
    :type view: MainWindow
    """
    def __init__(self, view):
        self.view = view
        self.parser = InstrumentParser()
        worker = InstrumentWorker(view, self)
        worker.job_succeeded.connect(self.setInstrumentSuccess)
        worker.job_failed.connect(self.setInstrumentFailed)

        self.recent_list_size = 10  # Maximum size of the recent project list

        self.model = EditorModel(worker)

    def notifyError(self, message, exception):
        """Logs error and notifies user of them

        :param message: message to display to user and in the log
        :type message: str
        :param exception: exception to log
        :type exception: Exception
        """
        logging.error(message, exc_info=exception)
        self.view.showMessage(message)

    def setInstrumentSuccess(self, result):
        """Sets the instrument created from the instrument file.

        :param result: instrument from description file
        :type result: Instrument
        """
        self.view.updateErrors(self.parser.errors)
        self.model.instrument = result
        self.view.controls.createWidgets()
        self.view.scene.updateInstrumentScene()
        self.view.designer.setEnabled(True)
        self.view.designer.setJson(self.parser.data)

    def setInstrumentFailed(self, e):
        """Reports errors from instrument update worker

        :param e: raised exception
        :type e: Exception
        """
        if self.model.initialized:
            if self.parser.data:
                self.view.designer.setEnabled(True)
                self.view.designer.setJson(self.parser.data)
            else:
                self.view.designer.setEnabled(False)

            self.view.updateErrors(self.parser.errors)

    def askToSaveFile(self):
        """Checks that changes have been saved, if no then asks the user to save them.
        
        :return: whether the user wants to proceed
        :rtype: bool
        """
        proceed = True
        if self.unsaved:
            message = f'The document has been modified.\n\nDo you want to save changes to "{self.model.filename}"?'
            reply = self.view.showSaveDiscardMessage(message)

            if reply == MessageReplyType.Save:
                self.saveFile()
            elif reply == MessageReplyType.Discard:
                pass
            elif reply == MessageReplyType.Cancel:
                proceed = False

        return proceed

    @property
    def unsaved(self):
        """Returns whether the last text change is saved
        :return: whether the last change was saved
        :rtype: bool
        """
        return self.view.editor.text() != self.model.saved_text

    def createNewFile(self):
        """Creates a new instrument description file"""
        if self.unsaved and not self.askToSaveFile():
            return

        self.view.reset()
        self.model.reset()

    def openFile(self, filename=''):
        """Loads an instrument description file from a given file path. If filename
        is empty, a file dialog will be opened

        :param filename: full path of file
        :type filename: str
        """
        if self.unsaved and not self.askToSaveFile():
            return

        if not filename:
            filename = self.view.askAddress(True, 'Open Instrument Description File', '', 'Json File (*.json)')

            if not filename:
                return

        try:
            new_text = self.model.openFile(filename)
            self.view.designer.folder_path = self.model.file_directory
            self.view.editor.setText(new_text)
            self.view.updateTitle()
            self.updateRecentProjects(filename)
        except OSError as e:
            self.notifyError(f'An error occurred while attempting to open this file ({filename}).', e)

    def saveFile(self, save_as=False):
        """Saves the instrument description file. A file dialog should be opened for the first save
        after which the function will save to the same location. If save_as is True a dialog is
        opened every time

        :param save_as: A flag denoting whether to use file dialog or not
        :type save_as: bool
        """
        if not self.unsaved and not save_as:
            return

        filename = self.model.filename
        if save_as or not filename:
            filename = self.view.askAddress(False, 'Save Instrument Description File', '', 'Json File (*.json)')

        if not filename:
            return

        try:
            self.model.saveFile(self.view.editor.text(), filename)
            self.view.designer.folder_path = self.model.file_directory
            self.view.updateTitle()
            self.updateRecentProjects(filename)
            if save_as:
                self.resetInstrumentControls()
        except OSError as e:
            self.notifyError(f'An error occurred while attempting to save this file ({filename}).', e)

    def updateRecentProjects(self, filename):
        """Adds a filename entry to the front of the recent projects list
        if it does not exist in the list. if the entry already exist, it is moved to the
        front but not duplicated.

        :param filename: project path to add to recent file lists
        :type filename: str
        """
        filename = os.path.normpath(filename)
        projects = self.view.recent_projects
        projects.insert(0, filename)
        projects = list(dict.fromkeys(projects))
        self.view.recent_projects = projects[:self.recent_list_size]

    def generateRobotModel(self):
        """Generates kinematic model of a positioning system from measurements"""
        filename = self.view.askAddress(True, 'Open Kinematic Calibration File', '', 'Supported Files (*.csv *.txt)')

        if not filename:
            return

        try:
            points, types, offsets, homes = read_kinematic_calibration_file(filename)
            self.view.createCalibrationWidget(points, types, offsets, homes)
        except (OSError, ValueError) as e:
            self.notifyError(f'An error occurred while attempting to open this file ({filename}).', e)

    def resetInstrumentControls(self):
        """Makes the view reset the instrument controls widget while model updates instrument"""
        self.view.controls.reset()
        self.model.useWorker()
