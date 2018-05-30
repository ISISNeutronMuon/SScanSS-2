from sscanss.core.io.writer import write_project_hdf
from sscanss.core.io.reader import read_project_hdf, read_stl
from sscanss.core.util import Node, Colour, RenderType


class MainWindowModel:
    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.unsaved = False
        self._sampleScene = None

    def createProjectData(self, name, instrument):

        self.project_data = {'name': name,
                             'instrument': instrument,
                             'sample': []}

    def saveProjectData(self, filename):
        write_project_hdf(self.project_data, filename)
        self.unsaved = False
        self.save_path = filename

    def loadProjectData(self, filename):
        self.project_data = read_project_hdf(filename)
        self.save_path = filename

    def loadSample(self, filename):
        self.project_data['sample'].append(read_stl(filename))
        self.unsaved = True

    @property
    def sampleScene(self):
        sample_node = Node()
        sample_node.colour = Colour(0.42, 0.42, 0.83)
        sample_node.render_type = RenderType.Solid

        for sample in self.project_data['sample']:
            sample_child = Node()
            sample_child.vertices = sample['vertices']
            sample_child.indices = sample['indices']
            sample_child.normals = sample['normals']
            sample_child.colour = None
            sample_child.render_type = None

            sample_node.children.append(sample_child)

        self._sampleScene = {'sample': sample_node}

        return self._sampleScene
