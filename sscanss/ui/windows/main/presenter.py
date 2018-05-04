from .model import MainWindowModel

class MainWindowPresenter:
    def __init__(self, view):
        self.view = view
        self.model = MainWindowModel()

    def isProjectCreated(self):
        return True if self.model.project_data else False

    def createProject(self, name, instrument):
        self.model.createProjectData(name, instrument)
        self.view.showProjectName(name)

    def saveProject(self):
        if self.model.project_data:
            filename = self.view.showSaveDialog('hdf5 File (*.h5)',
                                                name=self.model.project_data['name'])
            if filename:
                self.model.saveProjectData(filename)

    def openProject(self):
        filename = self.view.showOpenDialog('hdf5 File (*.h5)')
        if filename:
            self.model.loadProjectData(filename)
            self.view.showProjectName(self.model.project_data['name'])


