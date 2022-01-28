from contextlib import suppress
from collections import OrderedDict, namedtuple
import json
import os
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from sscanss.config import settings, INSTRUMENTS_PATH
from sscanss.core.instrument import read_instrument_description_file, Sequence, Simulation
from sscanss.core.io import (write_project_hdf, read_project_hdf, read_3d_model, read_points, read_vectors,
                             write_binary_stl, write_points, validate_vector_length)
from sscanss.core.scene import validate_instrument_scene_size
from sscanss.core.util import PointType, LoadVector, Attributes, POINT_DTYPE, InsertSampleOptions

IDF = namedtuple('IDF', ['name', 'path', 'version'])


class MainWindowModel(QObject):
    """Manages project data and communicates to view via signals"""
    sample_model_updated = pyqtSignal(object)
    instrument_model_updated = pyqtSignal(object)
    simulation_created = pyqtSignal()
    sample_changed = pyqtSignal()
    fiducials_changed = pyqtSignal()
    measurement_points_changed = pyqtSignal()
    measurement_vectors_changed = pyqtSignal()
    instrument_controlled = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.project_data = None
        self.save_path = ''
        self.all_sample_key = 'All Samples'

        self.simulation = None
        self.instruments = {}

        self.updateInstrumentList()

    @property
    def volume(self):
        """Gets the volume sample

        :return: volume object
        :rtype: Volume
        """
        return self.project_data['volume']

    @volume.setter
    def volume(self, value):
        """Sets the volume sample

        :param value: volume object
        :type value:: Volume
        """
        self.project_data['volume'] = value
        self.notifyChange(Attributes.Sample)

    @property
    def instrument(self):
        """Gets the diffraction instrument associated with the project

        :return: diffraction instrument
        :rtype: Instrument
        """
        return self.project_data['instrument']

    @instrument.setter
    def instrument(self, value):
        """Sets the instrument

        :param value: diffraction instrument
        :type value: Instrument
        """
        self.project_data['instrument'] = value
        self.notifyChange(Attributes.Instrument)

    def updateInstrumentList(self):
        """Updates the list of instrument description files found in the instrument directories"""
        self.instruments.clear()

        custom_path = settings.value(settings.Key.Custom_Instruments_Path)
        directories = [path for path in (custom_path, INSTRUMENTS_PATH) if os.path.isdir(path)]
        if not directories:
            return

        for path in directories:
            for name in os.listdir(path):
                idf = os.path.join(path, name, 'instrument.json')
                if not os.path.isfile(idf):
                    continue

                try:
                    with open(idf) as json_file:
                        data = json.load(json_file)
                except (OSError, ValueError):
                    data = {}

                instrument_data = data.get('instrument', None)
                if instrument_data is None:
                    continue

                name = instrument_data.get('name', '').strip().upper()
                version = instrument_data.get('version', '').strip()
                if name and version:
                    self.instruments[name] = IDF(name, idf, version)

    def createProjectData(self, name, instrument=None):
        """Creates a new project

        :param name: name of project
        :type name: str
        :param instrument: name of instrument
        :type instrument: Union[str, None]
        """
        self.project_data = {
            'name': name,
            'instrument': None,
            'instrument_version': None,
            'sample': OrderedDict(),
            'volume': None,
            'fiducials': np.recarray((0, ), dtype=POINT_DTYPE),
            'measurement_points': np.recarray((0, ), dtype=POINT_DTYPE),
            'measurement_vectors': np.empty((0, 3, 1), dtype=np.float32),
            'alignment': None
        }

        if instrument is not None:
            self.changeInstrument(instrument)

    def checkInstrumentVersion(self):
        """Checks the project instrument version is the same as in the instrument list

        :return: indicates if instrument version is the same
        :rtype: bool
        """
        if self.instrument.name not in self.instruments:
            return False

        if self.project_data['instrument_version'] != self.instruments[self.instrument.name].version:
            return False

        return True

    def saveProjectData(self, filename):
        """Saves the project data to a HDF file

        :param filename: filename
        :type filename: str
        """
        write_project_hdf(self.project_data, filename)

    def changeInstrument(self, name):
        """Changes the current instrument to instrument with given name

        :param name: name of instrument
        :type name: str
        """
        instrument = read_instrument_description_file(self.instruments[name].path)

        if not validate_instrument_scene_size(instrument):
            raise ValueError('The scene is too big the distance from the origin exceeds max extent')

        self.instrument = instrument
        self.project_data['instrument_version'] = self.instruments[name].version
        self.correctVectorDetectorSize()

    def correctVectorDetectorSize(self):
        """Folds or expands the measurement vectors to match size required by current instrument"""
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

    def correctVectorAlignments(self, vectors):
        num_of_points = self.measurement_points.size
        num_of_vectors = vectors.shape[0]

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

    def loadProjectData(self, filename):
        """Loads the project data from HDF file

        :param filename: filename
        :type filename: str
        """
        data, instrument = read_project_hdf(filename)

        if not validate_instrument_scene_size(instrument):
            raise ValueError('The scene is too big the distance from the origin exceeds max extent')

        self.createProjectData(data['name'])
        self.instrument = instrument
        self.project_data['instrument_version'] = data['instrument_version']

        self.project_data['sample'] = data['sample']
        self.project_data['fiducials'] = np.rec.fromarrays(data['fiducials'], dtype=POINT_DTYPE)
        self.project_data['measurement_points'] = np.rec.fromarrays(data['measurement_points'], dtype=POINT_DTYPE)
        self.project_data['measurement_vectors'] = data['measurement_vectors']
        self.alignment = data['alignment']

        settings.reset()
        for key, value in data['settings'].items():
            settings.local[key] = value

        self.notifyChange(Attributes.Sample)
        self.notifyChange(Attributes.Fiducials)
        self.notifyChange(Attributes.Vectors)
        self.notifyChange(Attributes.Measurements)

    def loadSample(self, filename, option=InsertSampleOptions.Combine):
        """Loads a 3D model from file. The 3D model can be added to the sample
        list or completely replace the sample

        :param filename: 3D model filename
        :type filename: str
        :param option: option for inserting sample
        :type option: InsertSampleOptions
        """
        name, ext = os.path.splitext(os.path.basename(filename))
        ext = ext.replace('.', '').lower()
        mesh = read_3d_model(filename)
        self.addMeshToProject(name, mesh, ext, option=option)

    def saveSample(self, filename, key):
        """Writes the specified sample model to file

        :param filename: filename
        :type filename: str
        :param key: key of sample to save
        :type key: str
        """
        write_binary_stl(filename, self.sample[key])

    def loadPoints(self, filename, point_type):
        """Loads a set of points from file

        :param filename: filename
        :type filename: str
        :param point_type: point type
        :type point_type: PointType
        """
        points, enabled = read_points(filename)
        self.addPointsToProject(list(zip(points, enabled)), point_type)

    def savePoints(self, filename, point_type):
        """Writes a set of points to file

        :param filename: filename
        :type filename: str
        :param point_type: point type
        :type point_type: PointType
        """
        points = self.fiducials if point_type == PointType.Fiducial else self.measurement_points
        write_points(filename, points)

    def loadVectors(self, filename):
        """Loads the measurement vectors from file

        :param filename: filename
        :type filename: str
        :return: indicate if loaded vectors were smaller, larger or exact with measurement points
        :rtype: LoadVector
        """
        temp = read_vectors(filename)
        detector_count = len(self.instrument.detectors)
        width = 3 * detector_count
        if temp.shape[1] > width:
            raise ValueError(f'The file contains vectors for more than {detector_count} detectors.')

        num_of_vectors = temp.shape[0]

        vectors = np.zeros((num_of_vectors, width), dtype=np.float32)
        vectors[:, 0:temp.shape[1]] = temp

        if not validate_vector_length(vectors):
            raise ValueError('Measurement vectors must be zero vectors or have a magnitude of 1 '
                             '(accurate to 7 decimal digits), the file contains vectors that are neither.')

        return self.correctVectorAlignments(vectors)

    def saveVectors(self, filename):
        """Writes the measurement vectors to file

        :param filename: filename
        :type filename: str
        """
        vectors = self.measurement_vectors
        vectors = np.vstack(np.dsplit(vectors, vectors.shape[2]))
        np.savetxt(filename, vectors[:, :, 0], delimiter='\t', fmt='%.7f')

    def addMeshToProject(self, name, mesh, attribute=None, option=InsertSampleOptions.Combine):
        """Adds to or replaces the project sample list with the given sample.
        A unique name is generated for the sample if necessary

        :param name: name of model
        :type name: str
        :param mesh: sample model
        :type mesh: Mesh
        :param attribute: info to append to name
        :type attribute: Union[None, str]
        :param option: option for inserting sample
        :type option: InsertSampleOptions
        :return: key of added mesh
        :rtype: str
        """
        key = self.uniqueKey(name, attribute)

        if option == InsertSampleOptions.Combine:
            self.sample[key] = mesh
            self.notifyChange(Attributes.Sample)
        else:
            self.sample = OrderedDict({key: mesh})

        return key

    def removeMeshFromProject(self, keys):
        """Removes mesh with given keys from the sample list

        :param keys: keys to remove
        :type keys: Union[List[str], str]
        """
        _keys = [keys] if not isinstance(keys, list) else keys
        for key in _keys:
            with suppress(KeyError):
                del self.sample[key]
        self.notifyChange(Attributes.Sample)

    @property
    def sample(self):
        """Gets the sample added to project. Sample is a group of meshes

        :return: sample meshes
        :rtype: OrderedDict
        """
        return self.project_data['sample']

    @sample.setter
    def sample(self, value):
        """Sets the sample

        :param value: sample meshes
        :type value: OrderedDict
        """
        self.project_data['sample'] = value
        self.notifyChange(Attributes.Sample)

    def notifyChange(self, key):
        """Notifies listeners that changes were made to project data

        :param key: attribute that has changed
        :type key: Attributes
        """
        if key == Attributes.Sample:
            self.sample_model_updated.emit(Attributes.Sample)
            self.sample_changed.emit()
        elif key == Attributes.Fiducials:
            self.sample_model_updated.emit(Attributes.Fiducials)
            self.fiducials_changed.emit()
        elif key == Attributes.Measurements:
            self.sample_model_updated.emit(Attributes.Measurements)
            self.measurement_points_changed.emit()
        elif key == Attributes.Vectors:
            self.sample_model_updated.emit(Attributes.Vectors)
            self.measurement_vectors_changed.emit()
        elif key == Attributes.Instrument:
            self.instrument_model_updated.emit(None)

    def uniqueKey(self, name, ext=None):
        """Generates a unique name for sample mesh

        :param name: name of mesh
        :type name: str
        :param ext: info to attach to name
        :type ext: Union[None, str]
        :return: new name
        :rtype: str
        """
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
        """Gets the fiducial points added to project. Fiducial points are 3D coordinates on the
        sample used for re-alignment

        :return: fiducial points
        :rtype: numpy.recarray
        """
        return self.project_data['fiducials']

    @fiducials.setter
    def fiducials(self, value):
        """Sets the fiducial points

        :param: fiducial points
        :type: numpy.recarray
        """
        self.project_data['fiducials'] = value
        self.notifyChange(Attributes.Fiducials)

    @property
    def measurement_points(self):
        """Gets the  measurement points added to project. Measurement points indicate the 3D coordinates
        for diffraction measurement

        :return: measurement points
        :rtype: numpy.recarray
        """
        return self.project_data['measurement_points']

    @measurement_points.setter
    def measurement_points(self, value):
        """Sets the measurement points (always ensure initial vectors (zeros) are added for
        each point or things could be bad). Prefer to use addPointsToProject instead

        :param value: measurement points
        :type value: numpy.recarray
        """
        self.project_data['measurement_points'] = value
        self.notifyChange(Attributes.Measurements)

    @property
    def measurement_vectors(self):
        """Gets the  measurement vectors added to project. Measurement vectors indicate the
        measurement orientations for each point.
        * The first dimension (rows) of the array must be the same as the numbers of points
        * The second dimension (columns) must be equally to 3 x number of detectors
        * The third dimension (depth) allows a point to be measured at multiple orientations

        Vectors can be zeros if the orientation is unimportant

        :return: measurement vectors
        :rtype: numpy.ndarray
        """
        return self.project_data['measurement_vectors']

    @measurement_vectors.setter
    def measurement_vectors(self, value):
        """Sets measurement vectors (always ensure dimensions are correct
        beforehand or things could be bad). Prefer to use addVectorsToProject instead.

        :param value: measurement vectors
        :type value: numpy.ndarray
        """
        self.project_data['measurement_vectors'] = value
        self.notifyChange(Attributes.Vectors)

    def addPointsToProject(self, points, point_type):
        """Adds points of specified type to the project after adding measurement point
        the measurement vectors for the points are also initialized with zeros

        :param points: points to add
        :type points: List[Tuple[List[float], bool]]
        :param point_type: point type
        :type point_type: PointType
        """
        if point_type == PointType.Fiducial:
            fiducials = np.append(self.fiducials, np.rec.array(points, dtype=POINT_DTYPE))
            self.fiducials = fiducials.view(np.recarray)
        elif point_type == PointType.Measurement:
            size = (len(points), *self.measurement_vectors.shape[1:])
            measurement_points = np.append(self.measurement_points, np.rec.array(points, dtype=POINT_DTYPE))
            self.measurement_points = measurement_points.view(np.recarray)
            self.measurement_vectors = np.append(self.measurement_vectors, np.zeros(size), axis=0)

    def removePointsFromProject(self, indices, point_type):
        """Removes points from project after removing measurement point the
        corresponding measurement vectors are also removed

        :param indices: indices to remove
        :type indices: Union[List[int], slice]
        :param point_type: point type
        :type point_type: PointType
        """
        if point_type == PointType.Fiducial:
            self.fiducials = np.delete(self.fiducials, indices, 0)
        elif point_type == PointType.Measurement:
            self.measurement_points = np.delete(self.measurement_points, indices, 0)
            self.measurement_vectors = np.delete(self.measurement_vectors, indices, 0)

    def addVectorsToProject(self, vectors, point_indices, alignment=0, detector=0):
        """Sets the measurement vector values at specified indices for a specific detector
        and alignment

        :param vectors: vector values
        :type vectors: numpy.ndarray
        :param point_indices: indices of measurement point
        :type point_indices: slice
        :param alignment: alignment index
        :type alignment: int
        :param detector: detector index
        :type detector: int
        """
        size = self.measurement_vectors.shape
        if alignment >= size[2]:
            self.measurement_vectors = np.dstack(
                (self.measurement_vectors, np.zeros((size[0], size[1], alignment - size[2] + 1))))

        detector_index = slice(detector * 3, detector * 3 + 3)
        self.measurement_vectors[point_indices, detector_index, alignment] = np.array(vectors)
        self.notifyChange(Attributes.Vectors)

    def moveInstrument(self, func, start_var, stop_var, duration=500, step=15):
        """Animates the movement of the instrument

        :param func: forward kinematics function
        :type func: Callable[numpy.ndarray, Any]
        :param start_var: inclusive start joint configuration/offsets
        :type start_var: List[float]
        :param stop_var: inclusive stop joint configuration/offsets
        :type stop_var: List[float]
        :param duration: time duration in milliseconds
        :type duration: int
        :param step: number of steps
        :type step: int
        """
        self.instrument_model_updated.emit(Sequence(func, start_var, stop_var, duration, step))

    @property
    def alignment(self):
        """Gets the alignment matrix added to project. Alignment matrix indicates the sample
        pose on the instrument. 'None' value means sample is not on instrument.

        :return: alignment matrix
        :rtype: Union[None, Matrix44]
        """
        return self.project_data['alignment']

    @alignment.setter
    def alignment(self, matrix):
        """Sets alignment matrix

        :param matrix: alignment matrix
        :type matrix: Union[None, Matrix44]
        """
        self.project_data['alignment'] = matrix
        self.notifyChange(Attributes.Instrument)

    def createSimulation(self, compute_path_length, render_graphics, check_limits, check_collision):
        """Creates a new simulation

        :param compute_path_length: indicates if simulation computes path lengths
        :type compute_path_length: bool
        :param render_graphics: indicates if graphics are rendered during simulation
        :type render_graphics: bool
        :param check_limits: indicates if simulation checks hardware limits
        :type check_limits: bool
        :param check_collision: indicates if simulation checks for collision
        :type check_collision: bool
        """
        self.simulation = Simulation(self.instrument, self.sample, self.measurement_points, self.measurement_vectors,
                                     self.alignment)
        self.simulation.compute_path_length = compute_path_length
        self.simulation.render_graphics = render_graphics
        self.simulation.check_limits = check_limits
        self.simulation.check_collision = check_collision
        self.simulation_created.emit()
