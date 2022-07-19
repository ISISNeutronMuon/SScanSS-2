import os
import sys
import pathlib
from sscanss.core.util.misc import MessageReplyType
from sscanss.editor.model import EditorModel, InstrumentWorker
from sscanss.core.io import read_kinematic_calibration_file
from sscanss.core.instrument import read_instrument_description
from jsonschema.exceptions import ValidationError

MAIN_WINDOW_TITLE = 'Instrument Editor'


class EditorPresenter:
    """Main presenter for the editor app

    :param view: main window instance
    :type view: MainWindow
    """
    def __init__(self, view):
        self.view = view

        worker = InstrumentWorker(view, self)
        worker.job_succeeded.connect(self.setInstrumentSuccess)
        worker.job_failed.connect(self.setInstrumentFailed)

        self.model = EditorModel(worker)

        self.updateTitle()

    def setInstrumentSuccess(self, result):
        """Sets the instrument created from the instrument file.

        :param result: instrument from description file
        :type result: Instrument
        """
        self.view.setMessageText("OK")
        self.model.instrument = result
        self.view.controls.createWidgets()
        self.view.scene.updateInstrumentScene()

    def setInstrumentFailed(self, e):
        """Reports errors from instrument update worker

        :param e: raised exception
        :type e: Exception
        """
        if self.model.initialized:
            if isinstance(e, ValidationError):
                path = ''
                for p in e.absolute_path:
                    if isinstance(p, int):
                        path = f'{path}[{p}]'
                    else:
                        path = f'{path}.{p}' if path else p

                path = path if path else 'instrument description file'
                error_message = f'{e.message} in {path}'
            else:
                error_message = str(e).strip("'")

            self.view.setMessageText(error_message)

    def parseLaunchArguments(self):
        """Parses the launch arguments and opens relevant file if required"""
        if sys.argv[1:]:
            file_path = sys.argv[1]
            if pathlib.PurePath(file_path).suffix == '.json':
                self.openFile(file_path)
            else:
                self.view.setMessageText(f'{file_path} could not be opened because it has an unknown file type')

    def askToSaveFile(self):
        """Function checks that changes have been saved, if no then asks the user to save them.
        :return: whether the user wants to proceed
        :rtype: bool
        """
        proceed = True
        if self.unsaved:
            message = f'The document has been modified.\n\nDo you want to save changes to "{self.model.current_file}"?'
            reply = self.view.showSaveDiscardMessage(message)

            if reply == MessageReplyType.Save:
                self.saveFile()
            elif reply == MessageReplyType.Discard:
                pass
            elif reply == MessageReplyType.Cancel:
                proceed = False

        return proceed

    def updateTitle(self):
        """Sets new title based on currently selected file"""
        if self.model.current_file:
            self.view.setWindowTitle(f'{self.model.current_file} - {MAIN_WINDOW_TITLE}')
        else:
            self.view.setWindowTitle(MAIN_WINDOW_TITLE)

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

        self.model.resetAddresses()
        self.updateTitle()
        self.view.scene.reset()
        self.view.controls.close()
        self.view.editor.setText("")
        self.view.setMessageText("")

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
            self.view.editor.setText(new_text)
            self.updateTitle()
        except OSError as e:
            self.view.setMessageText(f'An error occurred while attempting to open this file ({filename}). \n{e}')

    def saveFile(self, save_as=False):
        """Saves the instrument description file. A file dialog should be opened for the first save
        after which the function will save to the same location. If save_as is True a dialog is
        opened every time

        :param save_as: A flag denoting whether to use file dialog or not
        :type save_as: bool
        """
        if not self.unsaved and not save_as:
            return

        filename = self.model.current_file
        if save_as or not filename:
            filename = self.view.askAddress(False, 'Save Instrument Description File', '', 'Json File (*.json)')

        if not filename:
            return

        try:
            text = self.view.editor.text()
            self.model.saveFile(text, filename)
            self.updateTitle()
            if save_as:
                self.view.resetInstrument()
        except OSError as e:
            self.view.setMessageText(f'An error occurred while attempting to save this file ({filename}). \n{e}')

    def createInstrument(self):
        """Creates an instrument from the description file."""
        return read_instrument_description(self.view.editor.text(), os.path.dirname(self.model.current_file))

    def generateRobotModel(self):
        """Generates kinematic model of a positioning system from measurements"""
        filename = self.view.askAddress(True, 'Open Kinematic Calibration File', '', 'Supported Files (*.csv *.txt)')

        if not filename:
            return

        try:
            points, types, offsets, homes = read_kinematic_calibration_file(filename)
            self.view.createCalibrationWidget(points, types, offsets, homes)
        except (OSError, ValueError) as e:
            self.view.setMessageText(f'An error occurred while attempting to open this file ({filename}). \n{e}')

    def resetInstrumentControls(self):
        """Makes the view reset the instrument controls widget while model updates instrument"""
        self.view.controls.reset()
        self.model.useWorker()

    def updateInstrument(self):
        """Tries to lazily update the instrument"""
        self.model.lazyInstrumentUpdate()
