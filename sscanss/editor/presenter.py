from sscanss.core.util.misc import MessageReplyType
from model import EditorModel
from PyQt5.QtWidgets import QMessageBox

class EditorPresenter:
    """Main presenter for the editor app

    :param view: main window instance
    :type view: MainWindow
    """
    def __init__(self, view):
        self.view = view
        self.model = EditorModel()

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
        if self.view.getEditorText() != self.model.saved_text():
            reply = self.view.showSaveDiscardMessage()

            if reply == MessageReplyType.Save:
                self.saveFile()
            elif reply == MessageReplyType.Discard:
                pass
            elif reply == MessageReplyType.Cancel:
                proceed = False

        return proceed


    def createNewFile(self):
        """Creates a new instrument description file"""
        if not self.askToSaveFile():
            return

        self.model.createNewFile()
        self.editor.setText(self.saved_text)
        self.view.setTitle()
        self.scene.reset()
        self.view.hideControls()
        self.view.setMessageText("")

    def openFile(self):
        pass

    def saveFile(self, save_as = False):
        pass

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



