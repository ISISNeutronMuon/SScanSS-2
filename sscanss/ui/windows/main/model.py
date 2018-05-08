from collections import defaultdict
from sscanss.core.io.writer import write_project_hdf
from sscanss.core.io.reader import read_project_hdf


class MainWindowModel:
    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.unsaved = False

    def createProjectData(self, name, instrument):

        self.project_data = {'name': name, 'instrument': instrument}

    def saveProjectData(self, filename):
        write_project_hdf(self.project_data, filename)
        self.unsaved = False
        self.save_path = filename


    def loadProjectData(self, filename):
        self.project_data = read_project_hdf(filename)
        self.save_path = filename
