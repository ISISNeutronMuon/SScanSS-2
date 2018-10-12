import os
from contextlib import suppress
from collections import OrderedDict
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from sscanss.core.io import write_project_hdf, read_project_hdf, read_stl, read_obj, read_points, read_vectors
from sscanss.core.scene import (createSampleNode, createFiducialNode, createMeasurementPointNode,
                                createMeasurementVectorNode, createPlaneNode, Scene)
from sscanss.core.util import PointType, LoadVector


class MainWindowModel(QObject):
    scene_updated = pyqtSignal()
    sample_changed = pyqtSignal()
    fiducials_changed = pyqtSignal()
    measurement_points_changed = pyqtSignal()
    measurement_vectors_changed = pyqtSignal()

    # TODO: This should be removed when instrument loading is implemented
    num_of_detector = 2

    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.unsaved = False
        self.instrument_scene = Scene(Scene.Type.Instrument)
        self.sample_scene = Scene()
        self.active_scene = self.sample_scene
        self.rendered_alignment = 0
        self.point_dtype = [('points', 'f4', 3), ('enabled', '?')]

    def createProjectData(self, name, instrument):

        self.project_data = {'name': name,
                             'instrument': instrument,
                             'sample': OrderedDict(),
                             'fiducials': np.recarray((0, ), dtype=self.point_dtype),
                             'measurement_points': np.recarray((0,), dtype=self.point_dtype),
                             'measurement_vectors': np.empty((0, self.num_of_detector * 3, 1), dtype=np.float32)}

    def saveProjectData(self, filename):
        write_project_hdf(self.project_data, filename)
        self.unsaved = False
        self.save_path = filename

    def loadProjectData(self, filename):
        self.project_data = read_project_hdf(filename)
        self.save_path = filename

    def toggleScene(self):
        if self.active_scene is self.sample_scene:
            self.active_scene = self.instrument_scene
        else:
            self.active_scene = self.sample_scene

        self.scene_updated.emit()

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

    def loadVectors(self, filename):
        vectors = read_vectors(filename, self.num_of_detector)

        vectors = np.array(vectors, np.float32)
        num_of_points = self.measurement_points.size
        num_of_vectors = vectors.shape[0]
        offset = num_of_vectors % num_of_points
        if offset != 0:
            vectors = np.vstack((vectors, np.zeros((offset, vectors.shape[1]), np.float32)))

        vectors = np.dstack(np.split(vectors, vectors.shape[0] // num_of_points))
        self.measurement_vectors = vectors

        if num_of_vectors < num_of_points:
            return LoadVector.Smaller_than_points
        elif num_of_points == num_of_vectors:
            return LoadVector.Exact
        else:
            return LoadVector.Larger_than_points

    def addMeshToProject(self, name, mesh, attribute=None, combine=True):
        key = self.uniqueKey(name, attribute)

        if combine:
            self.sample[key] = mesh
            self.updateSampleScene(Scene.sample_key)
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
        self.updateSampleScene(Scene.sample_key)

    @property
    def sample(self):
        return self.project_data['sample']

    @sample.setter
    def sample(self, value):
        self.project_data['sample'] = value
        self.updateSampleScene(Scene.sample_key)

    def addPlane(self, plane=None, width=None, height=None, shift_by=None):
        key = 'plane'
        if shift_by is not None and key in self.sample_scene:
            self.sample_scene[key].translate(shift_by)
        elif plane is not None and width is not None and height is not None:
            self.sample_scene.addNode(key, createPlaneNode(plane, width, height))
        self.scene_updated.emit()

    def removePlane(self):
        self.sample_scene.removeNode('plane')
        self.scene_updated.emit()

    def updateSampleScene(self, key):
        if key == Scene.sample_key:
            self.sample_scene.addNode(key, createSampleNode(self.sample))
            self.sample_changed.emit()
        elif key == 'fiducials':
            self.sample_scene.addNode(key, createFiducialNode(self.fiducials))
            self.fiducials_changed.emit()
        elif key == 'measurement_points':
            self.sample_scene.addNode(key, createMeasurementPointNode(self.measurement_points))
            self.measurement_points_changed.emit()
        elif key == 'measurement_vectors':
            self.sample_scene.addNode(key, createMeasurementVectorNode(self.measurement_points,
                                                                       self.measurement_vectors,
                                                                       self.rendered_alignment))
            self.measurement_vectors_changed.emit()

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

    @property
    def measurement_vectors(self):
        return self.project_data['measurement_vectors']

    @measurement_vectors.setter
    def measurement_vectors(self, value):
        self.project_data['measurement_vectors'] = value
        self.updateSampleScene('measurement_vectors')

    def addPointsToProject(self, points, point_type):
        if point_type == PointType.Fiducial:
            fiducials = np.append(self.fiducials, np.rec.array(points, dtype=self.point_dtype))
            self.fiducials = fiducials.view(np.recarray)
        elif point_type == PointType.Measurement:
            size = len(points)
            measurement_points = np.append(self.measurement_points, np.rec.array(points, dtype=self.point_dtype))
            self.measurement_points = measurement_points.view(np.recarray)
            self.measurement_vectors = np.append(self.measurement_vectors,
                                                 np.zeros((size, self.num_of_detector * 3, 1)), axis=0)

    def removePointsFromProject(self, indices, point_type):
        if point_type == PointType.Fiducial:
            self.fiducials = np.delete(self.fiducials, indices, 0)
        elif point_type == PointType.Measurement:
            self.measurement_points = np.delete(self.measurement_points, indices, 0)
            self.measurement_vectors = np.delete(self.measurement_vectors, indices, 0)

    def addVectorsToProject(self, vectors, point_indices, alignment=0, detector=0):
        size = self.measurement_vectors.shape
        if alignment >= size[2]:
            self.measurement_vectors = np.dstack((self.measurement_vectors,
                                                  np.zeros((size[0], size[1], alignment - size[2] + 1))))

        detector_index = slice(detector * 3, detector * 3 + 3)
        self.measurement_vectors[point_indices, detector_index, alignment] = np.array(vectors)
        self.updateSampleScene('measurement_vectors')
