import os
from contextlib import suppress
from collections import OrderedDict
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from sscanss.core.io import write_project_hdf, read_project_hdf, read_stl, read_obj, read_points
from sscanss.core.util import createSampleNode


class MainWindowModel(QObject):
    sample_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.unsaved = False
        self.sample_scene = None
        self.point_dtype = [('points', 'f4', 3), ('enabled', '?')]

    def createProjectData(self, name, instrument):

        self.project_data = {'name': name,
                             'instrument': instrument,
                             'sample': OrderedDict(),
                             'fiducials': None}

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

    def loadFiducials(self, filename):
        points, enabled = read_points(filename)
        self.addPointsToProject(list(zip(points, enabled)))

    def addMeshToProject(self, name, mesh, attribute=None, combine=True):
        key = self.uniqueKey(name, attribute)

        if combine:
            self.sample[key] = mesh
            self.updateSampleScene()
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
        self.updateSampleScene()

    @property
    def sample(self):
        return self.project_data['sample']

    @sample.setter
    def sample(self, value):
        self.project_data['sample'] = value
        self.updateSampleScene()


    def updateSampleScene(self):
        self.sample_scene = {'sample': createSampleNode(self.sample)}
        self.sample_changed.emit()

    def uniqueKey(self, name, ext=None):
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

    @property
    def fiducials(self):
        return self.project_data['fiducials']

    @fiducials.setter
    def fiducials(self, value):
        self.project_data['fiducials'] = value
        self.updateSampleScene()

    def addPointsToProject(self, points):
        if self.fiducials is None:
            fiducials = np.rec.array(points, dtype=self.point_dtype)
        else:
            fiducials = np.append(self.fiducials, np.rec.array(points, dtype=self.point_dtype))
            fiducials = fiducials.view(np.recarray)

        self.fiducials = fiducials
