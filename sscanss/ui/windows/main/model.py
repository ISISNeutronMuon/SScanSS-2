import os
from contextlib import suppress
from collections import OrderedDict
from PyQt5.QtCore import pyqtSignal, QObject
from sscanss.core.io import write_project_hdf, read_project_hdf, read_stl, read_obj
from sscanss.core.util import Node, Colour, RenderType


class MainWindowModel(QObject):
    sample_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.unsaved = False
        self._sampleScene = None

    def createProjectData(self, name, instrument):

        self.project_data = {'name': name,
                             'instrument': instrument,
                             'sample': OrderedDict()}

    def saveProjectData(self, filename):
        write_project_hdf(self.project_data, filename)
        self.unsaved = False
        self.save_path = filename

    def loadProjectData(self, filename):
        self.project_data = read_project_hdf(filename)
        self.save_path = filename

    def loadSample(self, filename, combine=True):
        name, ext = os.path.splitext(os.path.basename(filename))
        ext = ext.replace('.', '').lower()
        if ext == 'stl':
            mesh = read_stl(filename)
        else:
            mesh = read_obj(filename)
        self.addMeshToProject(name, mesh, ext, combine)

    def addMeshToProject(self, name, mesh, attribute=None, combine=True):
        key = self.create_unique_key(name, attribute)

        if combine:
            self.sample[key] = mesh
            self.sample_changed.emit()
        else:
            self.sample = OrderedDict({key: mesh})
        self.unsaved = True

        return key

    def removeMeshFromProject(self, keys):
        _keys = [keys] if not isinstance(keys, list) else keys
        for key in _keys:
            with suppress(KeyError):
                del self.sample[key]
        self.unsaved = True
        self.sample_changed.emit()

    @property
    def sample(self):
        return self.project_data['sample']

    @sample.setter
    def sample(self, value):
        self.project_data['sample'] = value
        self.sample_changed.emit()

    @property
    def sampleScene(self):
        sample_node = Node()
        sample_node.colour = Colour(0.42, 0.42, 0.83)
        sample_node.render_type = RenderType.Solid

        for _, sample in self.sample.items():
            sample_child = Node()
            sample_child.vertices = sample['vertices']
            sample_child.indices = sample['indices']
            sample_child.normals = sample['normals']
            sample_child.colour = None
            sample_child.render_type = None

            sample_node.children.append(sample_child)

        self._sampleScene = {'sample': sample_node}

        return self._sampleScene

    def create_unique_key(self, name, ext=None):
        new_key = name if ext is None else '{} [{}]'.format(name, ext)

        if new_key not in self.sample.keys():
            return new_key

        similar_keys = 0
        for key in self.sample.keys():
            if key.startswith(name):
                similar_keys += 1

        if ext is None:
            return '{} {}'.format(name, similar_keys)
        else:
            return '{} {} [{}]'.format(name, similar_keys, ext)
