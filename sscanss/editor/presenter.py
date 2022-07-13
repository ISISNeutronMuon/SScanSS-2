from sscanss.core.util.misc import MessageReplyType
from sscanss.editor.model import EditorModel, InstrumentWorker
from jsonschema.exceptions import ValidationError
from sscanss.core.io import read_kinematic_calibration_file
from sscanss.core.instrument import read_instrument_description
import os

MAIN_WINDOW_TITLE = 'Instrument Editor'  # !-! Get the constant from the view instead of defining here

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

    def quitButtonPressed(self):
        """Is triggered when close action is"""  # !-! Check if function is needed maybe move elsewhere
        self.view.close()

    def exitApplication(self):
        """Is triggered when application needs to exit.""" # !-! Also combine with something else?
        return self.askToSaveFile()


    def askToSaveFile(self):  # !-! Come up with better name - it also saves file not just asks to do it
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

    def updateTitle(self):
        if self.model.getCurrentFile():
            self.view.setTitle(f'{self.model.getCurrentFile()} - {MAIN_WINDOW_TITLE}')
        else:
            self.view.setTitle(MAIN_WINDOW_TITLE)

    @property
    def unsaved(self):
        return self.view.getEditorText() != self.model.getSavedText()

    def createNewFile(self):
        """Creates a new instrument description file"""
        if not self.askToSaveFile():
            return

        self.model.createNewFile()
        self.updateTitle()
        #self.scene.reset()
        self.view.hideControls()
        self.view.setEditorText("")
        self.view.setMessageText("")

    def showCoordinateFrame(self, switch):
        self.view.showCoordinateFrame(switch)

    def resetCamera(self):
        self.view.resetCamera()

    def setInstrumentSuccess(self, result):
        """Sets the instrument created from the instrument file.

        :param result: instrument from description file
        :type result: Instrument
        """
        self.view.setMessageText('OK')
        self.view.instrument = result  # !-! Is the saved instrument needed? Remove or add more logic to handle the variable
        self.view.createInstrumentControls()
        self.view.updateScene()

    def setInstrumentFailed(self, e):
        """Reports errors from instrument update worker

        :param e: raised exception
        :type e: Exception
        """
        self.view.controls.tabs.clear()
        if self.model.initialized:  # !-! Change later (move to model?) Also split the algorithm into separate function
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

            self.view.setMessageText(m)

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
            filename = self.view.askInstrumentAddress()

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
            filename = self.view.askInstrumentAddress()

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
        self.view.showInstrumentControls()

    def createInstrument(self):
        """Creates an instrument from the description file."""
        return read_instrument_description(self.view.getEditorText(), os.path.dirname(self.model.getCurrentFile())) # Come up with a

    def generateRobotModel(self): # !-! Look into the widget and maybe extract some logic from it?
        """Generates kinematic model of a positioning system from measurements"""
        filename = self.view.askCalibrationFile()

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

    def showSearchBox(self):
        pass

    def showDocumentation(self):
        pass

    def showAboutMessage(self):
        pass



