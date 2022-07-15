import os
import sys
import pathlib
import datetime
import webbrowser
from sscanss.__version import __editor_version__, __version__
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
        self.view.setInstrument(result)
        self.view.createInstrumentControls()
        self.view.updateScene()

    def setInstrumentFailed(self, e):
        """Reports errors from instrument update worker

        :param e: raised exception
        :type e: Exception
        """
        if self.model.isInitialised():
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

            self.logMessage(error_message)

    def parseLaunchArguments(self):
        """Parses the launch arguments and opens relevant file if required"""
        if sys.argv[1:]:
            file_path = sys.argv[1]
            if pathlib.PurePath(file_path).suffix == '.json':
                self.openFile(file_path)
            else:
                self.view.setMessageText(f'{file_path} could not be opened because it has an unknown file type')

    def exitApplication(self):
        """Is triggered when application needs to exit."""
        return self.askToSaveFile()

    def askToSaveFile(self):
        """Function checks that changes have been saved, if no then asks the user to save them.
        :return: whether the user wants to proceed
        :rtype: bool
        """
        proceed = True
        if self.unsaved:
            message = f'The document has been modified.\n\nDo you want to save changes to "{self.model.getCurrentFile()}"?'
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
        if self.model.getCurrentFile():
            self.view.setTitle(f'{self.model.getCurrentFile()} - {MAIN_WINDOW_TITLE}')
        else:
            self.view.setTitle(MAIN_WINDOW_TITLE)

    @property
    def unsaved(self):
        return self.view.getEditorText() != self.model.getSavedText()

    @property
    def window_name(self):
        return MAIN_WINDOW_TITLE

    def createNewFile(self):
        """Creates a new instrument description file"""
        if not self.askToSaveFile():
            return

        self.model.resetAddresses()
        self.updateTitle()
        self.view.resetScene()
        self.view.hideControls()
        self.view.setEditorText("")
        self.view.setMessageText("")

    def showCoordinateFrame(self, switch):
        """Makes the view show the coordinate frame on the instrument's model"""
        self.view.showCoordinateFrame(switch)

    def resetCamera(self):
        """Resets the camera in the instrument viewer"""
        self.view.resetCamera()

    def logMessage(self, error_message):
        """Updates the message in the view to the new one"""
        self.view.setMessageText(error_message)

    def showAboutMessage(self):
        """Makes the view show the about message with the set text and title"""
        title = f'About {self.window_name}'
        about_text = (f'<h3 style="text-align:center">Version {__editor_version__}</h3>'
                      '<p style="text-align:center">This is a tool for modifying instrument '
                      'description files for SScanSS 2.</p>'
                      '<p style="text-align:center">Designed by Stephen Nneji</p>'
                      '<p style="text-align:center">Distributed under the BSD 3-Clause License</p>'
                      f'<p style="text-align:center">Copyright &copy; 2018-{datetime.date.today().year}, '
                      'ISIS Neutron and Muon Source. All rights reserved.</p>')

        self.view.showAboutMessage(title, about_text)

    def openFile(self, filename=''):
        """Loads an instrument description file from a given file path. If filename
        is empty, a file dialog will be opened

        :param filename: full path of file
        :type filename: str
        """
        if not self.askToSaveFile():
            return

        if not filename:
            filename = self.askInstrumentAddress()

            if not filename:
                return

        try:
            new_text = self.model.openFile(filename)
            self.view.setEditorText(new_text)
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

        filename = self.model.getCurrentFile()
        if save_as or not filename:
            filename = self.askInstrumentAddress()

        if not filename:
            return

        try:
            text = self.view.getEditorText()
            self.model.saveFile(text, filename)
            self.updateTitle()
            if save_as:
                self.view.resetInstrument()
        except OSError as e:
            self.view.setMessageText(f'An error occurred while attempting to save this file ({filename}). \n{e}')

    def showInstrumentControls(self):
        """Makes the view show the instrument control widget"""
        self.view.showControls()

    def askCalibrationFile(self):
        """Asks for address of a calibration file"""
        return self.view.askAddress('Open Kinematic Calibration File', '', 'Supported Files (*.csv *.txt)')

    def askInstrumentAddress(self):
        """Asks for address of an instrument file"""
        return self.view.askAddress('Open Instrument Description File', '', 'Json File (*.json)')

    def createInstrument(self):
        """Creates an instrument from the description file."""
        return read_instrument_description(self.view.getEditorText(), os.path.dirname(self.model.getCurrentFile()))

    def generateRobotModel(self):
        """Generates kinematic model of a positioning system from measurements"""
        filename = self.askCalibrationFile()

        if not filename:
            return

        try:
            points, types, offsets, homes = read_kinematic_calibration_file(filename)
            self.view.createCalibrationWidget(points, types, offsets, homes)
        except OSError as e:
            self.view.setMessageText(f'An error occurred while attempting to open this file ({filename}). \n{e}')

    def resetInstrumentControls(self):
        """Makes the view reset the instrument controls widget while model updates instrument"""
        self.view.resetControls()
        self.model.useWorker()

    def updateInstrument(self):
        """Tries to lazily update the instrument"""
        self.model.lazyInstrumentUpdate()

    def showDocumentation(self):
        """Opens the documentation in the system's default browser"""
        webbrowser.open_new(f'https://isisneutronmuon.github.io/SScanSS-2/{__version__}/api.html')
