from contextlib import suppress
from collections import OrderedDict
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from sscanss.core.io import write_project_hdf, read_project_hdf, read_3d_model, read_points, read_vectors
from sscanss.core.util import PointType, LoadVector, Attributes
from sscanss.core.instrument import read_instrument_description_file, get_instrument_list, Sequence


class MainWindowModel(QObject):
    sample_scene_updated = pyqtSignal(object)
    instrument_scene_updated = pyqtSignal()
    animate_instrument = pyqtSignal(object)
    sample_changed = pyqtSignal()
    fiducials_changed = pyqtSignal()
    measurement_points_changed = pyqtSignal()
    measurement_vectors_changed = pyqtSignal()
    instrument_controlled = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.unsaved = False
        self.point_dtype = [('points', 'f4', 3), ('enabled', '?')]

        self.instruments = get_instrument_list()

    @property
    def instrument(self):
        return self.project_data['instrument']

    @instrument.setter
    def instrument(self, value):
        self.project_data['instrument'] = value
        self.notifyChange(Attributes.Instrument)

    def createProjectData(self, instrument, name):
        self.project_data = {'name': name,
                             'instrument': None,
                             'sample': OrderedDict(),
                             'fiducials': np.recarray((0, ), dtype=self.point_dtype),
                             'measurement_points': np.recarray((0,), dtype=self.point_dtype),
                             'measurement_vectors': np.empty((0, 3, 1), dtype=np.float32)}

        self.loadInstrument(instrument)

    def saveProjectData(self, filename):
        write_project_hdf(self.project_data, filename)
        self.unsaved = False
        self.save_path = filename

    def loadInstrument(self, name):
        self.instrument = read_instrument_description_file(self.instruments[name])
        vectors = self.measurement_vectors
        new_size = 3 * len(self.instrument.detectors)
        if vectors.size == 0:
            self.measurement_vectors = np.empty((0, new_size, 1), dtype=np.float32)
        elif vectors.shape[1] > new_size:
            fold = vectors.shape[1] // 3
            temp = np.zeros((vectors.shape[0], new_size, vectors.shape[2] * fold), dtype=np.float32)
            temp[:, 0:3, :] = np.dstack(np.hsplit(vectors, fold))
            index = np.where(np.sum(temp[:, :, 1:], axis=(0, 1)) != 0)[0] + 1  # get non-zero alignments
            index = np.concatenate(([0], index))
            self.measurement_vectors = temp[:, :, index]
        elif vectors.shape[1] < new_size:
            temp = np.zeros((vectors.shape[0], new_size, vectors.shape[2]), dtype=np.float32)
            temp[:, 0:vectors.shape[1], :] = vectors
            self.measurement_vectors = temp

    def loadProjectData(self, filename):
        self.project_data = read_project_hdf(filename)
        self.save_path = filename

    def loadSample(self, filename, combine=True):
        mesh, name, _type = read_3d_model(filename)
        self.addMeshToProject(name, mesh, _type, combine)

    def loadPoints(self, filename, point_type):
        points, enabled = read_points(filename)
        self.addPointsToProject(list(zip(points, enabled)), point_type)

    def loadVectors(self, filename):
        temp = read_vectors(filename)
        detector_count = len(self.instrument.detectors)
        width = 3 * detector_count
        if temp.shape[1] > width:
            raise ValueError(f'{filename} contains vectors for more than f{detector_count} detectors.')

        num_of_points = self.measurement_points.size
        num_of_vectors = temp.shape[0]

        vectors = np.zeros((num_of_vectors, width), dtype=np.float32)
        vectors[:, 0:temp.shape[1]] = temp

        offset = (num_of_points - num_of_vectors) % num_of_points
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
            self.notifyChange(Attributes.Sample)
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
        self.notifyChange(Attributes.Sample)

    @property
    def sample(self):
        return self.project_data['sample']

    @sample.setter
    def sample(self, value):
        self.project_data['sample'] = value
        self.notifyChange(Attributes.Sample)

    def notifyChange(self, key):
        if key == Attributes.Sample:
            self.sample_scene_updated.emit(Attributes.Sample)
            self.sample_changed.emit()
        elif key == Attributes.Fiducials:
            self.sample_scene_updated.emit(Attributes.Fiducials)
            self.fiducials_changed.emit()
        elif key == Attributes.Measurements:
            self.sample_scene_updated.emit(Attributes.Measurements)
            self.measurement_points_changed.emit()
        elif key == Attributes.Vectors:
            self.sample_scene_updated.emit(Attributes.Vectors)
            self.measurement_vectors_changed.emit()
        elif key == Attributes.Instrument:
            self.instrument_scene_updated.emit()

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
        self.notifyChange(Attributes.Fiducials)

    @property
    def measurement_points(self):
        return self.project_data['measurement_points']

    @measurement_points.setter
    def measurement_points(self, value):
        self.project_data['measurement_points'] = value
        self.notifyChange(Attributes.Measurements)

    @property
    def measurement_vectors(self):
        return self.project_data['measurement_vectors']

    @measurement_vectors.setter
    def measurement_vectors(self, value):
        self.project_data['measurement_vectors'] = value
        self.notifyChange(Attributes.Vectors)

    def addPointsToProject(self, points, point_type):
        if point_type == PointType.Fiducial:
            fiducials = np.append(self.fiducials, np.rec.array(points, dtype=self.point_dtype))
            self.fiducials = fiducials.view(np.recarray)
        elif point_type == PointType.Measurement:
            size = (len(points), *self.measurement_vectors.shape[1:])
            measurement_points = np.append(self.measurement_points, np.rec.array(points, dtype=self.point_dtype))
            self.measurement_points = measurement_points.view(np.recarray)
            self.measurement_vectors = np.append(self.measurement_vectors, np.zeros(size), axis=0)

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
        self.notifyChange(Attributes.Vectors)

    def moveInstrument(self, func, start_var, stop_var, duration=1000, step=10):
        self.animate_instrument.emit(Sequence(func, start_var, stop_var, duration, step))

    def alignSampleOnInstrument(self, matrix):
        aligned_sample = OrderedDict()

        for key, sample in self.sample.items():
            aligned_sample[key] = sample.transformed(matrix)

        _matrix = matrix[0:3, 0:3].transpose()
        offset = matrix[0:3, 3].transpose()

        aligned_fiducials = self.fiducials.copy()
        aligned_fiducials.points = aligned_fiducials.points @ _matrix + offset

        aligned_measurements = self.measurement_points.copy()
        aligned_measurements.points = aligned_measurements.points @ _matrix + offset
        aligned_vectors = self.measurement_vectors.copy()
        for k in range(aligned_vectors.shape[2]):
            for j in range(0, aligned_vectors.shape[1], 3):
                aligned_vectors[:, j:j+3, k] = aligned_vectors[:, j:j+3, k] @ _matrix

        self.instrument.sample = aligned_sample
        self.instrument.fiducials = aligned_fiducials
        self.instrument.measurement_points = aligned_measurements
        self.instrument.measurement_vectors = aligned_vectors
        self.notifyChange(Attributes.Instrument)
