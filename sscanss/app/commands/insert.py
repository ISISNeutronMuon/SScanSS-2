import logging
import os
import warnings
import numpy as np
from PyQt6 import QtGui
from sscanss.core.geometry import (create_tube, create_sphere, create_cylinder, create_cuboid, create_cone,
                                   closest_triangle_to_point, compute_face_normals, Mesh)
from sscanss.core.io import read_angles, read_3d_model, BadDataWarning, load_volume
from sscanss.core.math import matrix_from_pose, VECTOR_EPS
from sscanss.core.util import (Primitives, Worker, PointType, LoadVector, MessageType, StrainComponents, CommandID,
                               Attributes)


class InsertPrimitive(QtGui.QUndoCommand):
    """Creates command to insert specified primitive model to the project as a sample

    :param primitive: primitive type
    :type primitive: Primitives
    :param args: arguments for primitive creation
    :type args: Dict
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    :param option: option for inserting sample
    :type option: InsertSampleOptions
    """
    def __init__(self, primitive, args, presenter, option):
        super().__init__()

        self.name = args.pop('name', 'unnamed')
        self.args = args
        self.primitive = primitive
        self.presenter = presenter
        self.option = option
        self.old_sample = None

        self.setText(f'Insert {self.primitive.value}')

    def redo(self):
        self.old_sample = self.presenter.model.sample

        if self.primitive == Primitives.Tube:
            mesh = create_tube(**self.args)
        elif self.primitive == Primitives.Sphere:
            mesh = create_sphere(**self.args)
        elif self.primitive == Primitives.Cylinder:
            mesh = create_cylinder(**self.args)
        elif self.primitive == Primitives.Cuboid:
            mesh = create_cuboid(**self.args)
        else:
            mesh = create_cone(**self.args)

        self.presenter.model.addMeshToProject(mesh, self.option)

    def undo(self):
        self.presenter.model.sample = self.old_sample


class InsertMeshFromFile(QtGui.QUndoCommand):
    """Creates command to insert a sample model from a file to the project

    :param filename: path of file
    :type filename: str
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    :param option: option for inserting sample
    :type option: InsertSampleOptions
    """
    def __init__(self, filename, presenter, option):
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.option = option
        self.new_mesh = None
        self.old_sample = None

        base_name = os.path.basename(filename)
        self.setText(f'Import {base_name}')

    def redo(self):
        self.old_sample = self.presenter.model.sample
        if self.new_mesh is None:
            self.presenter.view.progress_dialog.showMessage('Loading 3D Model')
            self.worker = Worker(self.loadMesh, [])
            self.worker.finished.connect(self.presenter.view.progress_dialog.close)
            self.worker.job_failed.connect(self.onImportFailed)
            self.worker.start()
        else:
            self.presenter.model.addMeshToProject(self.new_mesh, self.option)

    def undo(self):
        self.presenter.model.sample = self.old_sample

    def loadMesh(self):
        """Loads a 3D model from file. The 3D model can be added to the sample
        list or completely replace the sample
        """
        self.new_mesh = read_3d_model(self.filename)
        self.presenter.model.addMeshToProject(self.new_mesh, self.option)

    def onImportFailed(self, exception):
        """Logs error and clean up after failed import

        :param exception: exception when importing mesh
        :type exception: Exception
        """
        msg = 'An error occurred while loading the 3D model.\n\nPlease check that the file is valid.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class InsertVolumeFromFile(QtGui.QUndoCommand):
    """Creates command to load volume data from a nexus file or stack of TIFF files to the project

    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    :param filepath: path of file or path of the folder containing TIFF files
    :type filepath: str
    :param voxel_size: size of the volume's voxels in the x, y, and z axes
    :type voxel_size: Optional[List[float, float, float]]
    :param centre: coordinates of the volume centre in the x, y, and z axes
    :type centre: Optional[List[float, float, float]]
    """
    def __init__(self, presenter, filepath, voxel_size=None, centre=None):
        super().__init__()

        self.filepath = filepath
        self.presenter = presenter
        self.voxel_size = voxel_size
        self.centre = centre
        self.old_sample = None

    def redo(self):
        """Using a worker thread to load in volume data"""
        self.presenter.view.progress_dialog.showMessage('Loading Volume Data')
        self.old_sample = self.presenter.model.sample
        self.worker = Worker(self.loadVolume, [])
        self.worker.finished.connect(self.presenter.view.progress_dialog.close)
        self.worker.job_succeeded.connect(self.onImportSuccess)
        self.worker.job_failed.connect(self.onImportFailed)
        self.worker.start()

    def loadVolume(self):
        """Loads volume TIFFs or a nexus file

        :return: warnings
        :rtype: List[Warnings]
        """
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            self.presenter.model.sample = load_volume(self.filepath, self.voxel_size, self.centre)

            return warning

    def undo(self):
        self.presenter.model.sample = self.old_sample

    def onImportSuccess(self, result):
        """Handler after successful import

        :param result: warnings
        :type result: List[Warnings]
        """
        for warning in result:
            if issubclass(warning.category, BadDataWarning):
                msg = 'The imported volume data contains non-finite values i.e. Nans or Inf. These values ' \
                      'have be replaced with the minimum grayscale value.'
                self.presenter.view.showMessage(msg, MessageType.Information)
                logging.info(f'{msg} filepath: {self.filepath}')
                break

    def onImportFailed(self, exception):
        """Logs error and clean up after failed import

        :param exception: exception when importing data
        :type exception: Exception
        """
        msg = f'Failed to load files: {exception}'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.InsertVolumeFromFile


class ChangeVolumeCurve(QtGui.QUndoCommand):
    """Changes the curve used to create the transfer function for viewing the volume

    :param curve: volume curve
    :type curve: Curve
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, curve, presenter):
        super().__init__()

        self.presenter = presenter
        self.old_curve = presenter.model.sample.curve
        self.new_curve = curve
        self.setText('Change Volume Curve')

    def redo(self):
        self.presenter.model.sample.curve = self.new_curve
        self.presenter.model.notifyChange(Attributes.Sample)

    def undo(self):
        self.presenter.model.sample.curve = self.old_curve
        self.presenter.model.notifyChange(Attributes.Sample)


class InsertPointsFromFile(QtGui.QUndoCommand):
    """Creates command to insert measurement or fiducial points from file

    :param filename: path of file
    :type filename: str
    :param point_type: point type
    :type point_type: PointType
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, filename, point_type, presenter):
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.point_type = point_type
        self.new_points = None

        if self.point_type == PointType.Fiducial:
            self.old_count = len(self.presenter.model.fiducials)
        else:
            self.old_count = len(self.presenter.model.measurement_points)

        self.setText(f'Import {self.point_type.value} Points')

    def redo(self):
        if self.new_points is None:
            load_points_args = [self.filename, self.point_type]
            self.presenter.view.progress_dialog.showMessage(f'Loading {self.point_type.value} Points')
            self.worker = Worker(self.presenter.model.loadPoints, load_points_args)
            self.worker.job_succeeded.connect(self.onImportSuccess)
            self.worker.finished.connect(self.presenter.view.progress_dialog.close)
            self.worker.job_failed.connect(self.onImportFailed)
            self.worker.start()
        else:
            self.presenter.model.addPointsToProject(self.new_points, self.point_type)

    def undo(self):
        if self.point_type == PointType.Fiducial:
            current_count = len(self.presenter.model.fiducials)
            indices = slice(self.old_count, current_count, None)
            self.new_points = np.copy(self.presenter.model.fiducials[indices])
        else:
            current_count = len(self.presenter.model.measurement_points)
            indices = slice(self.old_count, current_count, None)
            self.new_points = np.copy(self.presenter.model.measurement_points[indices])

        self.presenter.model.removePointsFromProject(indices, self.point_type)

    def onImportSuccess(self):
        """Opens point Manager after successfully import"""
        self.presenter.view.docks.showPointManager(self.point_type)

    def onImportFailed(self, exception):
        """Logs error and clean up after failed import

        :param exception: exception when importing points
        :type exception: Exception
        """
        if isinstance(exception, ValueError):
            msg = f'{self.point_type.value} points could not be read from {self.filename} because it has incorrect ' \
                  f'data: {exception}'
        elif isinstance(exception, OSError):
            msg = 'An error occurred while opening this file.\nPlease check that ' \
                  f'the file exist and also that this user has access privileges for this file.\n({self.filename})'
        else:
            msg = f'An unknown error occurred while opening {self.filename}.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class InsertPoints(QtGui.QUndoCommand):
    """Creates command to insert measurement or fiducial points into project

    :param points: array of points
    :type points: List[Tuple[List[float], bool]]
    :param point_type: point type
    :type point_type: PointType
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, points, point_type, presenter):
        super().__init__()

        self.points = points
        self.presenter = presenter
        self.point_type = point_type

        if self.point_type == PointType.Fiducial:
            self.old_count = len(self.presenter.model.fiducials)
        else:
            self.old_count = len(self.presenter.model.measurement_points)

        self.setText(f'Insert {self.point_type.value} Points')

    def redo(self):
        self.presenter.model.addPointsToProject(self.points, self.point_type)

    def undo(self):
        if self.point_type == PointType.Fiducial:
            current_count = len(self.presenter.model.fiducials)
        else:
            current_count = len(self.presenter.model.measurement_points)
        self.presenter.model.removePointsFromProject(slice(self.old_count, current_count, None), self.point_type)


class DeletePoints(QtGui.QUndoCommand):
    """Creates command to delete measurement or fiducial points with given indices

    :param indices: indices of points
    :type indices: List[int]
    :param point_type: point type
    :type point_type: PointType
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, indices, point_type, presenter):
        super().__init__()

        self.indices = sorted(indices)
        self.model = presenter.model
        self.point_type = point_type
        self.removed_points = None
        self.removed_vectors = None

        if len(self.indices) > 1:
            self.setText(f'Delete {len(self.indices)} {self.point_type.value} Points')
        else:
            self.setText(f'Delete {self.point_type.value} Point')

    def redo(self):
        if self.point_type == PointType.Fiducial:
            self.removed_points = self.model.fiducials[self.indices]
        else:
            self.removed_points = self.model.measurement_points[self.indices]
            self.removed_vectors = self.model.measurement_vectors[self.indices, :, :]

        self.model.removePointsFromProject(self.indices, self.point_type)

    def undo(self):
        if self.point_type == PointType.Fiducial:
            points = self.reinsert(self.model.fiducials, self.removed_points)
            self.model.fiducials = points.view(np.recarray)
        else:
            points = self.reinsert(self.model.measurement_points, self.removed_points)
            vectors = self.reinsert(self.model.measurement_vectors, self.removed_vectors)
            self.model.measurement_points = points.view(np.recarray)
            self.model.measurement_vectors = vectors

    def reinsert(self, array, removed_array):
        """Re-inserts removed points into an array

        :param array: array to insert into
        :type array: numpy.ndarray
        :param removed_array: array of removed points
        :type removed_array: numpy.ndarray
        :return: array with removed points inserted
        :rtype: numpy.ndarray
        """
        for index, value in enumerate(self.indices):
            if index < len(array):
                array = np.insert(array, value, removed_array[index], 0)
            else:
                array = np.append(array, removed_array[index:index + 1], 0)

        return array


class MovePoints(QtGui.QUndoCommand):
    """Creates command to swap measurement or fiducial point at start index with another at destination index

    :param move_from: start index
    :type move_from: int
    :param move_to: destination index
    :type move_to: int
    :param point_type: point type
    :type point_type: PointType
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, move_from, move_to, point_type, presenter):
        super().__init__()

        self.move_from = move_from
        self.move_to = move_to
        self.model = presenter.model
        self.point_type = point_type

        count = len(self.model.fiducials) if point_type == PointType.Fiducial else len(self.model.measurement_points)
        self.old_order = list(range(0, count))
        self.new_order = self.old_order.copy()
        self.new_order[move_from], self.new_order[move_to] = self.new_order[move_to], self.new_order[move_from]

        self.setText(f'Change {self.point_type.value} Point Index')

    def redo(self):
        if self.point_type == PointType.Fiducial:
            self.model.fiducials[self.old_order] = self.model.fiducials[self.new_order]
            self.model.notifyChange(Attributes.Fiducials)
        else:
            self.model.measurement_points[self.old_order] = self.model.measurement_points[self.new_order]
            self.model.measurement_vectors[self.old_order, :, :] = self.model.measurement_vectors[self.new_order, :, :]
            self.model.notifyChange(Attributes.Measurements)
            self.model.notifyChange(Attributes.Vectors)

    def undo(self):
        if self.point_type == PointType.Fiducial:
            self.model.fiducials[self.new_order] = self.model.fiducials[self.old_order]
            self.model.notifyChange(Attributes.Fiducials)
        else:
            self.model.measurement_points[self.new_order] = self.model.measurement_points[self.old_order]
            self.model.measurement_vectors[self.new_order, :, :] = self.model.measurement_vectors[self.old_order, :, :]
            self.model.notifyChange(Attributes.Measurements)
            self.model.notifyChange(Attributes.Vectors)

    def mergeWith(self, command):
        if self.point_type != command.point_type:
            return False

        move_to = command.move_to
        move_from = command.move_from
        self.new_order[move_from], self.new_order[move_to] = self.new_order[move_to], self.new_order[move_from]

        if self.new_order == self.old_order:
            self.setObsolete(True)

        return True

    def id(self):
        """Returns ID used when merging commands"""
        return CommandID.MovePoints


class EditPoints(QtGui.QUndoCommand):
    """Creates command to modify measurement or fiducial points

    :param value: point array after edit
    :type value: numpy.recarray
    :param point_type: point type
    :type point_type: PointType
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, value, point_type, presenter):
        super().__init__()

        self.model = presenter.model
        self.point_type = point_type

        self.new_values = value

        self.setText(f'Edit {self.point_type.value} Points')

    @property
    def points(self):
        """Gets and sets measurement or fiducial points depending on point type"""
        return self.model.fiducials if self.point_type == PointType.Fiducial else self.model.measurement_points

    @points.setter
    def points(self, values):
        if self.point_type == PointType.Fiducial:
            self.model.fiducials = values
        else:
            self.model.measurement_points = values

    def redo(self):
        self.old_values = self.points
        self.points = self.new_values

    def undo(self):
        self.points = self.old_values

    def mergeWith(self, command):
        if self.point_type != command.point_type:
            return False

        if np.all(self.old_values == command.new_values):
            self.setObsolete(True)

        self.new_values = command.new_values

        return True

    def id(self):
        """Returns ID used when merging commands"""
        return CommandID.EditPoints


class InsertVectorsFromFile(QtGui.QUndoCommand):
    """Creates command to insert measurement vectors from file

    :param filename: path of file
    :type filename: str
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, filename, presenter):
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.old_vectors = np.copy(self.presenter.model.measurement_vectors)
        self.new_vectors = None

        self.setText('Import Measurement Vectors')

    def redo(self):
        if self.new_vectors is None:
            load_vectors_args = [self.filename]
            self.presenter.view.progress_dialog.showMessage('Loading Measurement vectors')
            self.worker = Worker(self.presenter.model.loadVectors, load_vectors_args)
            self.worker.job_succeeded.connect(self.onImportSuccess)
            self.worker.finished.connect(self.presenter.view.progress_dialog.close)
            self.worker.job_failed.connect(self.onImportFailed)
            self.worker.start()
        else:
            self.presenter.model.measurement_vectors = np.copy(self.new_vectors)

    def undo(self):
        self.new_vectors = np.copy(self.presenter.model.measurement_vectors)
        self.presenter.model.measurement_vectors = np.copy(self.old_vectors)

    def onImportSuccess(self, return_code):
        """Opens vector manager after successfully import

        :param return_code: return code from 'loadVectors' function
        :type return_code: LoadVector
        """
        if return_code == LoadVector.Smaller:
            msg = 'Fewer measurements vectors than points were loaded from the file. The remaining have been ' \
                  'assigned a zero vector.'
            self.presenter.view.showMessage(msg, MessageType.Information)
        elif return_code == LoadVector.Larger:
            msg = 'More measurements vectors than points were loaded from the file. The extra vectors have been  ' \
                  'added as secondary alignments.'
            self.presenter.view.showMessage(msg, MessageType.Information)

        self.presenter.view.docks.showVectorManager()

    def onImportFailed(self, exception):
        """Logs error and clean up after failed import

        :param exception: exception when importing points
        :type exception: Exception
        """
        if isinstance(exception, ValueError):
            msg = f'Measurement vectors could not be read from {self.filename} because it has incorrect ' \
                  f'data: {exception}'
        elif isinstance(exception, OSError):
            msg = 'An error occurred while opening this file.\nPlease check that ' \
                  f'the file exist and also that this user has access privileges for this file.\n({self.filename})'
        else:
            msg = f'An unknown error occurred while opening {self.filename}.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class CreateVectorsWithEulerAngles(QtGui.QUndoCommand):
    """Creates command to insert measurement vectors from file

    :param filename: path of file
    :type filename: str
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, filename, presenter):
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.old_vectors = np.copy(self.presenter.model.measurement_vectors)
        self.new_vectors = None

        self.setText('Create Measurement Vectors with Euler Angles')

    def redo(self):
        if self.new_vectors is None:
            self.presenter.view.progress_dialog.showMessage('Loading Euler Angles')
            self.worker = Worker(self.createVectors, [])
            self.worker.job_succeeded.connect(self.onSuccess)
            self.worker.finished.connect(self.presenter.view.progress_dialog.close)
            self.worker.job_failed.connect(self.onFailed)
            self.worker.start()
        else:
            self.presenter.model.measurement_vectors = np.copy(self.new_vectors)

    def undo(self):
        self.new_vectors = np.copy(self.presenter.model.measurement_vectors)
        self.presenter.model.measurement_vectors = np.copy(self.old_vectors)

    def createVectors(self):
        angles, order = read_angles(self.filename)
        detector_count = len(self.presenter.model.instrument.detectors)

        vectors = np.zeros((angles.shape[0], 3 * detector_count), np.float32)
        q_vectors = np.array(self.presenter.model.instrument.q_vectors)
        for i, angle in enumerate(angles):
            matrix = matrix_from_pose([0., 0., 0., *angle], order=order)[:3, :3]
            m_vectors = q_vectors @ matrix.transpose()
            vectors[i, :] = m_vectors.flatten()

        return self.presenter.model.correctVectorAlignments(vectors)

    def onSuccess(self, return_code):
        """Opens vector manager after successfully import

        :param return_code: return code from 'loadVectors' function
        :type return_code: LoadVector
        """
        if return_code == LoadVector.Smaller:
            msg = 'Fewer euler angles than points were loaded from the file. The empty vectors have been ' \
                  'assigned a zero vector.'
            self.presenter.view.showMessage(msg, MessageType.Information)
        elif return_code == LoadVector.Larger:
            msg = 'More euler angles than points were loaded from the file. The extra vectors have been  ' \
                  'added as secondary alignments.'
            self.presenter.view.showMessage(msg, MessageType.Information)

        self.presenter.view.docks.showVectorManager()

    def onFailed(self, exception):
        """Logs error and clean up after failed import

        :param exception: exception when importing points
        :type exception: Exception
        """
        if isinstance(exception, ValueError):
            msg = f'Measurement vectors could not be read from {self.filename} because it has incorrect ' \
                  f'data: {exception}'
        elif isinstance(exception, OSError):
            msg = 'An error occurred while opening this file.\nPlease check that ' \
                  f'the file exist and also that this user has access privileges for this file.\n({self.filename})'
        else:
            msg = f'An unknown error occurred while opening {self.filename}.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class InsertVectors(QtGui.QUndoCommand):
    """Creates command to compute and insert measurement vectors into project

    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    :param point_index: index of measurement point, when index is -1 adds vectors for all points
    :type point_index: int
    :param strain_component: type of strain component
    :type strain_component: StrainComponents
    :param alignment: index of alignment
    :type alignment: int
    :param detector: index of detector
    :type detector: int
    :param key_in: custom vector
    :type key_in: Union[None, List[float]]
    :param reverse: flag indicating vector should be reversed
    :type reverse: bool
    """
    def __init__(self, presenter, point_index, strain_component, alignment, detector, key_in=None, reverse=False):
        super().__init__()

        self.point_index = point_index
        self.strain_component = strain_component
        self.alignment = alignment
        self.detector = detector
        self.key_in = key_in
        self.reverse = reverse
        self.presenter = presenter
        self.old_vectors = np.copy(self.presenter.model.measurement_vectors)
        self.new_vectors = None

        self.setText('Insert Measurement Vectors')

    def redo(self):
        if self.new_vectors is None:
            self.presenter.view.progress_dialog.showMessage('Creating Measurement vectors')
            self.worker = Worker(self.createVectors, [])
            self.worker.job_succeeded.connect(self.onSuccess)
            self.worker.job_failed.connect(self.onFailed)
            self.worker.finished.connect(self.presenter.view.progress_dialog.close)
            self.worker.start()
        else:
            self.presenter.model.measurement_vectors = np.copy(self.new_vectors)

    def undo(self):
        self.new_vectors = np.copy(self.presenter.model.measurement_vectors)
        self.presenter.model.measurement_vectors = np.copy(self.old_vectors)

    def createVectors(self):
        """Creates measurement vectors using the specified strain component type"""
        if self.point_index == -1:
            index = slice(None)
            num_of_points = self.presenter.model.measurement_points.size
        else:
            index = slice(self.point_index, self.point_index + 1)
            num_of_points = 1

        vectors = []
        if self.strain_component == StrainComponents.ParallelX:
            vectors = [[1.0, 0.0, 0.0]] * num_of_points
        elif self.strain_component == StrainComponents.ParallelY:
            vectors = [[0.0, 1.0, 0.0]] * num_of_points
        elif self.strain_component == StrainComponents.ParallelZ:
            vectors = [[0.0, 0.0, 1.0]] * num_of_points
        elif self.strain_component == StrainComponents.SurfaceNormal:
            vectors = self.normalMeasurementVector(index)
        elif self.strain_component == StrainComponents.OrthogonalWithoutX:
            vectors = self.orthogonalToSurfaceNormal(index, [1.0, 0.0, 0.0])
        elif self.strain_component == StrainComponents.OrthogonalWithoutY:
            vectors = self.orthogonalToSurfaceNormal(index, [0.0, 1.0, 0.0])
        elif self.strain_component == StrainComponents.OrthogonalWithoutZ:
            vectors = self.orthogonalToSurfaceNormal(index, [0.0, 0.0, 1.0])
        elif self.strain_component == StrainComponents.Custom:
            v = np.array(self.key_in) / np.linalg.norm(self.key_in)
            vectors = [v] * num_of_points

        vectors = np.array(vectors) if not self.reverse else -np.array(vectors)
        self.presenter.model.addVectorsToProject(vectors, index, self.alignment, self.detector)

    def normalMeasurementVector(self, index):
        """Computes measurement vectors for specified point indices by finding the closest face
        in the sample mesh to the points and calculating the surface normal of that face. Only
        mesh sample model is used to compute vectors

        :param index: point indices to compute vector
        :type index: slice
        :return: surface normal measurement vectors
        :rtype: numpy.ndarray
        """
        sample = self.presenter.model.sample
        if not isinstance(sample, Mesh):
            raise ValueError('Normal cannot be computed on a volume')

        points = self.presenter.model.measurement_points.points[index]
        vertices = sample.vertices[sample.indices]
        face_vertices = vertices.reshape(-1, 9)
        faces = closest_triangle_to_point(face_vertices, points)

        return compute_face_normals(faces)

    def orthogonalToSurfaceNormal(self, index, control):
        """Computes measurement vectors for specified specified point indices by finding the vector
        perpendicular to the surface normal of the closest face and the control vector.

        :param index: point indices to compute vector
        :type index: slice
        :param control: control vector
        :type control: List[float]
        :return: measurement vectors
        :rtype: numpy.ndarray
        """
        surface_normals = self.normalMeasurementVector(index)
        vectors = np.cross(surface_normals, np.tile(control, (len(surface_normals), 1)))
        row_sums = np.linalg.norm(vectors, axis=1)
        row_sums[row_sums < VECTOR_EPS] = 1
        return vectors / row_sums[:, np.newaxis]

    def onSuccess(self):
        """Opens vector manager after successfully addition"""
        self.presenter.view.docks.showVectorManager()

    def onFailed(self, exception):
        """Logs error and clean up after failed creation

        :param exception: exception when importing points
        :type exception: Exception
        """
        msg = 'An error occurred while creating the measurement vectors'
        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class RemoveVectorAlignment(QtGui.QUndoCommand):
    """Creates command to delete measurement vector alignment with given index for all detectors

    :param index: index of alignment
    :type index: int
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, index, presenter):
        super().__init__()

        self.presenter = presenter
        self.remove_index = index

        self.setText(f'Delete Measurement Vector Alignment {index + 1}')

    def redo(self):
        self.removed_vectors = self.presenter.model.measurement_vectors[:, :, self.remove_index]
        self.presenter.model.measurement_vectors = np.delete(self.presenter.model.measurement_vectors,
                                                             self.remove_index, 2)

    def undo(self):
        self.presenter.model.measurement_vectors = np.insert(self.presenter.model.measurement_vectors,
                                                             self.remove_index, self.removed_vectors, 2)


class RemoveVectors(QtGui.QUndoCommand):
    """Creates command to remove (Sets to zero) measurement vectors at specified indices for a specific detector
    and alignment

    :param indices: indices of vectors
    :type indices: List[int]
    :param detector: index of detector
    :type detector: int
    :param alignment: index of alignment
    :type alignment: int
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, indices, detector, alignment, presenter):
        super().__init__()

        self.presenter = presenter
        self.indices = indices
        self.detector = slice(detector * 3, detector * 3 + 3)
        self.alignment = alignment

        self.setText('Delete Measurement Vectors')

    def redo(self):
        self.removed_vectors = self.presenter.model.measurement_vectors[self.indices, self.detector, self.alignment]
        self.presenter.model.measurement_vectors[self.indices, self.detector, self.alignment] = [0., 0., 0.]
        self.presenter.model.notifyChange(Attributes.Vectors)

    def undo(self):
        self.presenter.model.measurement_vectors[self.indices, self.detector, self.alignment] = self.removed_vectors
        self.presenter.model.notifyChange(Attributes.Vectors)


class InsertAlignmentMatrix(QtGui.QUndoCommand):
    """Creates command to insert sample alignment matrix

    :param matrix: transformation matrix
    :type matrix: Matrix44
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, matrix, presenter):
        super().__init__()

        self.model = presenter.model
        self.old_matrix = self.model.alignment
        self.new_matrix = matrix

        self.setText('Align Sample on Instrument')

    def redo(self):
        self.model.alignment = self.new_matrix

    def undo(self):
        self.model.alignment = self.old_matrix

    def mergeWith(self, command):
        if np.array_equal(self.old_matrix, command.new_matrix):
            self.setObsolete(True)

        self.new_matrix = command.new_matrix

        return True

    def id(self):
        """Returns ID used when merging commands"""
        return CommandID.AlignSample
