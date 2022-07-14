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


class EditorPresenter:
    """Main presenter for the editor app

    :param view: main window instance
    :type view: MainWindow
    """
    MAIN_WINDOW_TITLE = 'Instrument Editor'

    def __init__(self, view):
        self.view = view

        worker = InstrumentWorker(view, self)
        self.model = EditorModel(worker)

        worker.job_succeeded.connect(self.setInstrumentSuccess)
        worker.job_failed.connect(self.model.setInstrumentFailed)

        self.model.error_occurred.connect(self.catchModelError)

        self.updateTitle()

    def parseLaunchArguments(self): # !-! Move to model?
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
            reply = self.view.showSaveDiscardMessage()

            if reply == MessageReplyType.Save:
                self.saveFile()
            elif reply == MessageReplyType.Discard:
                pass
            elif reply == MessageReplyType.Cancel:
                proceed = False

        return proceed

    def updateTitle(self):  # !-! Make title change automatic when file location changes
        if self.model.getCurrentFile():
            self.view.setTitle(f'{self.model.getCurrentFile()} - {self.MAIN_WINDOW_TITLE}')
        else:
            self.view.setTitle(self.MAIN_WINDOW_TITLE)

    @property
    def unsaved(self):
        return self.view.getEditorText() != self.model.getSavedText()

    @property
    def windowName(self):
        return self.MAIN_WINDOW_TITLE

    def createNewFile(self):
        """Creates a new instrument description file"""
        if not self.askToSaveFile():
            return

        self.model.createNewFile()
        self.updateTitle()
        self.view.resetScene()
        self.view.hideControls()
        self.view.setEditorText("")
        self.view.setMessageText("")

    def showCoordinateFrame(self, switch):
        self.view.showCoordinateFrame(switch)

    def resetCamera(self):
        self.view.resetCamera()

    def logError(self, errorMessage):
        self.view.setMessageText(errorMessage)

    def setInstrumentSuccess(self, result):
        """Sets the instrument created from the instrument file.

        :param result: instrument from description file
        :type result: Instrument
        """
        self.view.setMessageText('OK')
        self.view.instrument = result  # !-! Store the instrument somewhere else i.e. model
        self.view.createInstrumentControls()
        self.view.updateScene()

    def showAboutMessage(self):
        title = f'About {self.windowName}'
        about_text = (f'<h3 style="text-align:center">Version {__editor_version__}</h3>'
                          '<p style="text-align:center">This is a tool for modifying instrument '
                          'description files for SScanSS 2.</p>'
                          '<p style="text-align:center">Designed by Stephen Nneji</p>'
                          '<p style="text-align:center">Distributed under the BSD 3-Clause License</p>'
                          f'<p style="text-align:center">Copyright &copy; 2018-{datetime.date.today().year}, '
                          'ISIS Neutron and Muon Source. All rights reserved.</p>')
        self.view.showAboutMessage(title, about_text)

    def catchModelError(self, e, error_message):
        self.view.controls.tabs.clear()
        self.view.setMessageText(error_message)

    def openFile(self, filename=''):
        """Loads an instrument description file from a given file path. If filename
        is empty, a file dialog will be opened

        :param filename: full path of file
        :type filename: str
        """
        # !-! More elegant way to check for filename? (to not reset it afterwards)
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
        self.view.showControls()

    def askCalibrationFile(self):
        return self.view.askAddress('Open Kinematic Calibration File', '',
                                    'Supported Files (*.csv *.txt)')

    def askInstrumentAddress(self):
        return self.view.askAddress('Open Instrument Description File', '',
                                    'Json File (*.json)')

    def createInstrument(self):
        """Creates an instrument from the description file."""
        return read_instrument_description(self.view.getEditorText(), os.path.dirname(self.model.getCurrentFile())) # Come up with a

    def generateRobotModel(self): # !-! Look into the widget and maybe extract some logic from it?
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
        self.view.resetControls()
        self.model.useWorker()

    def updateInstrument(self):
        self.model.lazyInstrumentUpdate()

    def showDocumentation(self):
        """Opens the documentation in the system's default browser"""
        webbrowser.open_new(f'https://isisneutronmuon.github.io/SScanSS-2/{__version__}/api.html')



