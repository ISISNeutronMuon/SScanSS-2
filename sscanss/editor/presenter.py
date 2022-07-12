from sscanss.core.util.misc import MessageReplyType
from sscanss.editor.model import EditorModel
#from sscanss.editor.view import MAIN_WINDOW_TITLE

MAIN_WINDOW_TITLE = 'Instrument Editor'

class EditorPresenter:
    """Main presenter for the editor app

    :param view: main window instance
    :type view: MainWindow
    """
    def __init__(self, view):
        self.view = view
        self.model = EditorModel()
        self.UpdateTitle()

    def quitButtonPressed(self):
        """Is triggered when close action is"""
        self.view.close()

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

    def UpdateTitle(self):
        if self.model.getCurrentFile():
            self.view.setTitle(f'{self.filename} - {MAIN_WINDOW_TITLE}')
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
        self.UpdateTitle()
        #self.scene.reset()
        self.view.hideControls()
        self.view.setEditorText("")
        self.view.setMessageText("")


    def openFile(self, filename=''):
        """Loads an instrument description file from a given file path. If filename
        is empty, a file dialog will be opened

        :param filename: full path of file
        :type filename: str
        """
        if not self.askToSaveFile():
            return

        if not filename:
            filename = self.view.askInstrumentAddress()

            if not filename:
                return

        try:
            new_text = self.model.openFile(filename)
            self.view.setEditorText(new_text)
            self.UpdateTitle()
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

        filename = self.filename
        if save_as or not filename:
            filename = self.view.askInstrumentAddress()

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

    def ShowInstrumentControls(self):
        pass

    def generateRobotModelAction(self):
        pass

    def resetInstrument(self):
        pass

    def showSearchBox(self):
        pass

    def showCoordinateFrame(self):
        pass

    def resetCamera(self):
        pass

    def showDocumentation(self):
        pass

    def showAboutMessage(self):
        pass



