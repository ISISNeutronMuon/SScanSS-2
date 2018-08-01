import os
from contextlib import suppress
from collections import OrderedDict
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from sscanss.core.io import write_project_hdf, read_project_hdf, read_stl, read_obj, read_points
from sscanss.core.util import createSampleNode, createFiducialNode, PointType, createMeasurementPointNode


class MainWindowModel(QObject):
    scene_updated = pyqtSignal()
    sample_changed = pyqtSignal()
    fiducials_changed = pyqtSignal()
    measurement_points_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.unsaved = False
        self.sample_scene = {}
        self.point_dtype = [('points', 'f4', 3), ('enabled', '?')]

    def createProjectData(self, name, instrument):

        self.project_data = {'name': name,
                             'instrument': instrument,
                             'sample': OrderedDict(),
                             'fiducials': np.recarray((0, ), dtype=self.point_dtype),
                             'measurement_points': np.recarray((0,), dtype=self.point_dtype)}

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

    def loadPoints(self, filename, point_type):
        points, enabled = read_points(filename)
        self.addPointsToProject(list(zip(points, enabled)), point_type)

    def addMeshToProject(self, name, mesh, attribute=None, combine=True):
        key = self.uniqueKey(name, attribute)

        if combine:
            self.sample[key] = mesh
            self.updateSampleScene('sample')
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
        self.updateSampleScene('sample')

    @property
    def sample(self):
        return self.project_data['sample']

    @sample.setter
    def sample(self, value):
        self.project_data['sample'] = value
        self.updateSampleScene('sample')

    def updateSampleScene(self, key):
        if key == 'sample':
            self.sample_scene[key] = createSampleNode(self.sample)
            self.sample_changed.emit()
        elif key == 'fiducials':
            self.sample_scene[key] = createFiducialNode(self.fiducials)
            self.fiducials_changed.emit()
        elif key == 'measurement_points':
            self.sample_scene[key] = createMeasurementPointNode(self.measurement_points)
            self.measurement_points_changed.emit()

        self.scene_updated.emit()

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
        self.updateSampleScene('fiducials')

    @property
    def measurement_points(self):
        return self.project_data['measurement_points']

    @measurement_points.setter
    def measurement_points(self, value):
        self.project_data['measurement_points'] = value
        self.updateSampleScene('measurement_points')

    def addPointsToProject(self, points, point_type):
        if point_type == PointType.Fiducial:
            fiducials = np.append(self.fiducials, np.rec.array(points, dtype=self.point_dtype))
            self.fiducials = fiducials.view(np.recarray)
        elif point_type == PointType.Measurement:
            measurement_points = np.append(self.measurement_points, np.rec.array(points, dtype=self.point_dtype))
            self.measurement_points = measurement_points.view(np.recarray)

    def removePointsFromProject(self, indices, point_type):
        if point_type == PointType.Fiducial:
            self.fiducials = np.delete(self.fiducials, indices, 0)
        elif point_type == PointType.Measurement:
            self.measurement_points = np.delete(self.measurement_points, indices, 0)
