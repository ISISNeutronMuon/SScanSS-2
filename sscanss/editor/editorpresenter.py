from editormodel import EditorModel

class EditorPresenter:
    """Main presenter for the editor app

    :param view: main window instance
    :type view: MainWindow
    :param model: main model instance
    :type model: EditorModel
    """
    def __init__(self, view):
        self.view = view
        self.model = EditorModel()

    def exitApplication(self):
        view.

    def createNewFile(self):
        pass

    def openFile(self):
        pass

    def SaveFile(self, save_as = False):
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



