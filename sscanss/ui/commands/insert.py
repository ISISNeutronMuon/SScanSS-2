from collections import OrderedDict
import logging
import os
import numpy as np
from PyQt5 import QtWidgets
from sscanss.core.util import (Primitives, Worker, PointType, LoadVector, MessageSeverity, StrainComponents,
                               CommandID, Attributes)
from sscanss.core.geometry import (create_tube, create_sphere, create_cylinder, create_cuboid,
                                   closest_triangle_to_point, compute_face_normals)


class InsertPrimitive(QtWidgets.QUndoCommand):
    def __init__(self, primitive, args, presenter, combine):
        """ Command to insert primitive to the sample list

        :param primitive: primitive to insert
        :type primitive: sscanss.core.util.misc.Primitives
        :param args: arguments for primitive creation
        :type args: Dict
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.window.presenter.MainWindowPresenter
        :param combine: when True primitive is added to current otherwise replaces it
        :type combine: bool
        """
        super().__init__()

        self.name = args.pop('name', 'unnamed')
        self.args = args
        self.primitive = primitive
        self.presenter = presenter
        self.combine = combine

        self.setText('Insert {}'.format(self.primitive.value))

    def redo(self):
        if not self.combine:
            self.old_sample = self.presenter.model.sample

        if self.primitive == Primitives.Tube:
            mesh = create_tube(**self.args)
        elif self.primitive == Primitives.Sphere:
            mesh = create_sphere(**self.args)
        elif self.primitive == Primitives.Cylinder:
            mesh = create_cylinder(**self.args)
        else:
            mesh = create_cuboid(**self.args)

        self.sample_key = self.presenter.model.addMeshToProject(self.name, mesh, combine=self.combine)

    def undo(self):
        if self.combine:
            self.presenter.model.removeMeshFromProject(self.sample_key)
        else:
            self.presenter.model.sample = self.old_sample


class InsertSampleFromFile(QtWidgets.QUndoCommand):
    def __init__(self, filename, presenter, combine):
        """ Command to insert sample model from a file to the sample list

        :param filename: path of file
        :type filename: str
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.window.presenter.MainWindowPresenter
        :param combine: when True model is added to current otherwise replaces it
        :type combine: bool
        """
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.combine = combine
        self.new_mesh = None

        base_name = os.path.basename(filename)
        name, ext = os.path.splitext(base_name)
        ext = ext.replace('.', '').lower()
        self.sample_key = self.presenter.model.uniqueKey(name, ext)
        self.setText('Import {}'.format(base_name))

    def redo(self):
        if not self.combine:
            self.old_sample = self.presenter.model.sample
        if self.new_mesh is None:
            load_sample_args = [self.filename, self.combine]
            self.presenter.view.progress_dialog.show('Loading 3D Model')
            self.worker = Worker(self.presenter.model.loadSample, load_sample_args)
            self.worker.job_succeeded.connect(self.onImportSuccess)
            self.worker.finished.connect(self.presenter.view.progress_dialog.close)
            self.worker.job_failed.connect(self.onImportFailed)
            self.worker.start()
        else:
            self.presenter.model.addMeshToProject(self.sample_key, self.new_mesh, combine=self.combine)

    def undo(self):
        self.new_mesh = self.presenter.model.sample[self.sample_key].copy()
        if self.combine:
            self.presenter.model.removeMeshFromProject(self.sample_key)
        else:
            self.presenter.model.sample = self.old_sample

    def onImportSuccess(self):
        self.presenter.view.docks.showSampleManager()

    def onImportFailed(self, exception):
        msg = 'An error occurred while loading the 3D model.\n\n' \
              'Please check that the file is valid.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class DeleteSample(QtWidgets.QUndoCommand):
    def __init__(self, sample_key, presenter):
        """ Command to delete sample model from sample list

        :param sample_key: key(s) of sample(s) to delete
        :type sample_key: List[str]
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.window.presenter.MainWindowPresenter
        """
        super().__init__()

        self.keys = sample_key
        self.model = presenter.model
        self.old_keys = list(self.model.sample.keys())

        if len(sample_key) > 1:
            self.setText('Delete {} Samples'.format(len(sample_key)))
        else:
            self.setText('Delete {}'.format(sample_key[0]))

    def redo(self):
        self.deleted_mesh = {}
        for key in self.keys:
            self.deleted_mesh[key] = self.model.sample[key]

        self.model.removeMeshFromProject(self.keys)

    def undo(self):
        new_sample = {}
        for key in self.old_keys:
            if key in self.model.sample:
                new_sample[key] = self.model.sample[key]
            elif key in self.deleted_mesh:
                new_sample[key] = self.deleted_mesh[key]

        self.model.sample = OrderedDict(new_sample)


class MergeSample(QtWidgets.QUndoCommand):
    def __init__(self, sample_key, presenter):
        """ Command to merge sample models into a single one

        :param sample_key: key(s) of sample(s) to merge
        :type sample_key: List[str]
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.window.presenter.MainWindowPresenter
        """
        super().__init__()

        self.keys = sample_key
        self.model = presenter.model
        self.new_name = self.model.uniqueKey('merged')
        self.old_keys = list(self.model.sample.keys())

        self.setText('Merge {} Samples'.format(len(sample_key)))

    def redo(self):
        self.merged_mesh = []
        samples = self.model.sample
        new_mesh = samples.pop(self.keys[0], None)
        self.merged_mesh.append((self.keys[0], 0))
        for i in range(1, len(self.keys)):
            old_mesh = samples.pop(self.keys[i], None)
            self.merged_mesh.append((self.keys[i], new_mesh.indices.size))
            new_mesh.append(old_mesh)

        self.model.addMeshToProject(self.new_name, new_mesh, combine=True)

    def undo(self):
        mesh = self.model.sample.pop(self.new_name, None)
        temp = {}
        for key, index in reversed(self.merged_mesh):
            temp[key] = mesh.remove(index) if index != 0 else mesh

        new_sample = {}
        for key in self.old_keys:
            if key in self.model.sample:
                new_sample[key] = self.model.sample[key]
            elif key in temp:
                new_sample[key] = temp[key]

        self.model.sample = OrderedDict(new_sample)


class ChangeMainSample(QtWidgets.QUndoCommand):
    def __init__(self, sample_key, presenter):
        """ Command to make a specified sample model the main one.

        :param sample_key: key of sample to make main
        :type sample_key: str
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.window.presenter.MainWindowPresenter
        """
        super().__init__()

        self.key = sample_key
        self.model = presenter.model
        self.old_keys = list(self.model.sample.keys())
        self.new_keys = list(self.model.sample.keys())
        self.new_keys.insert(0, self.key)
        self.new_keys = list(dict.fromkeys(self.new_keys))

        self.setText('Set {} as Main Sample'.format(self.key))

    def redo(self):
        self.reorderSample(self.new_keys)

    def undo(self):
        self.reorderSample(self.old_keys)

    def mergeWith(self, command):
        """ Merges consecutive change main commands

        :param command: command to merge
        :type command: QUndoCommand
        :return: True if merge was successful
        :rtype: bool
        """
        self.new_keys = command.new_keys
        self.setText('Set {} as Main Sample'.format(self.key))

        return True

    def reorderSample(self, new_keys):
        new_sample = {}
        for key in new_keys:
            if key in self.model.sample:
                new_sample[key] = self.model.sample[key]

        self.model.sample = OrderedDict(new_sample)

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.ChangeMainSample


class InsertPointsFromFile(QtWidgets.QUndoCommand):
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

        self.setText('Import {} Points'.format(self.point_type.value))

    def redo(self):
        if self.new_points is None:
            load_points_args = [self.filename, self.point_type]
            self.presenter.view.progress_dialog.show('Loading {} Points'.format(self.point_type.value))
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
        self.presenter.view.docks.showPointManager(self.point_type)

    def onImportFailed(self, exception):
        msg = 'An error occurred while loading the {} points.\n\n' \
              'Please check that the file is valid.'.format(self.point_type.value)

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class InsertPoints(QtWidgets.QUndoCommand):
    def __init__(self, points, point_type, presenter):
        super().__init__()

        self.points = points
        self.presenter = presenter
        self.point_type = point_type

        if self.point_type == PointType.Fiducial:
            self.old_count = len(self.presenter.model.fiducials)
        else:
            self.old_count = len(self.presenter.model.measurement_points)

        self.setText('Insert {} Points'.format(self.point_type.value))

    def redo(self):
        self.presenter.model.addPointsToProject(self.points, self.point_type)

    def undo(self):
        if self.point_type == PointType.Fiducial:
            current_count = len(self.presenter.model.fiducials)
        else:
            current_count = len(self.presenter.model.measurement_points)
        self.presenter.model.removePointsFromProject(slice(self.old_count, current_count, None), self.point_type)


class DeletePoints(QtWidgets.QUndoCommand):
    def __init__(self, indices, point_type, presenter):
        super().__init__()

        self.indices = sorted(indices)
        self.model = presenter.model
        self.point_type = point_type

        if len(self.indices) > 1:
            self.setText('Delete {} {} Points'.format(len(self.indices), self.point_type.value))
        else:
            self.setText('Delete {} Point'.format(self.point_type.value))

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
        for index, value in enumerate(self.indices):
            if index < len(array):
                array = np.insert(array, value, removed_array[index], 0)
            else:
                array = np.append(array, removed_array[index:index+1], 0)

        return array


class MovePoints(QtWidgets.QUndoCommand):
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

        self.setText('Change {} Point Index'.format(self.point_type.value))

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
        """ Returns ID used when merging commands"""
        return CommandID.MovePoints


class EditPoints(QtWidgets.QUndoCommand):
    def __init__(self, value, point_type, presenter):
        super().__init__()

        self.model = presenter.model
        self.point_type = point_type

        self.new_values = value

        self.setText('Edit {} Points'.format(self.point_type.value))

    @property
    def points(self):
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
        """ Returns ID used when merging commands"""
        return CommandID.EditPoints


class InsertVectorsFromFile(QtWidgets.QUndoCommand):
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
            self.presenter.view.progress_dialog.show('Loading Measurement vectors')
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
        if return_code == LoadVector.Smaller_than_points:
            msg = 'Fewer measurements vectors than points were loaded from the file. The remaining have been ' \
                  'assigned a zero vector.'
            self.presenter.view.showMessage(msg, MessageSeverity.Information)
        elif return_code == LoadVector.Larger_than_points:
            msg = 'More measurements vectors than points were loaded from the file. The extra vectors have been  ' \
                  'added as secondary alignments.'
            self.presenter.view.showMessage(msg, MessageSeverity.Information)

        self.presenter.view.docks.showVectorManager()

    def onImportFailed(self, exception):
        msg = 'An error occurred while loading the measurement vectors.\n\n' \
              'Please check that the file is valid.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class InsertVectors(QtWidgets.QUndoCommand):
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
            self.presenter.view.progress_dialog.show('Creating Measurement vectors')
            self.worker = Worker(self.createVectors, [])
            self.worker.job_succeeded.connect(self.onImportSuccess)
            self.worker.job_failed.connect(self.onImportFailed)
            self.worker.finished.connect(self.presenter.view.progress_dialog.close)
            self.worker.start()
        else:
            self.presenter.model.measurement_vectors = np.copy(self.new_vectors)

    def undo(self):
        self.new_vectors = np.copy(self.presenter.model.measurement_vectors)
        self.presenter.model.measurement_vectors = np.copy(self.old_vectors)

    def createVectors(self):
        if self.point_index == -1:
            index = slice(None)
            num_of_points = self.presenter.model.measurement_points.size
        else:
            index = self.point_index
            num_of_points = 1

        vectors = []
        if self.strain_component == StrainComponents.parallel_to_x:
            vectors = self.stackVectors([1.0, 0.0, 0.0], num_of_points)
        elif self.strain_component == StrainComponents.parallel_to_y:
            vectors = self.stackVectors([0.0, 1.0, 0.0], num_of_points)
        elif self.strain_component == StrainComponents.parallel_to_z:
            vectors = self.stackVectors([0.0, 0.0, 1.0], num_of_points)
        elif self.strain_component == StrainComponents.normal_to_surface:
            vectors = self.normalMeasurementVector(index)
        elif self.strain_component == StrainComponents.orthogonal_to_normal_no_x:
            surface_normals = self.normalMeasurementVector(index)
            vectors = np.cross(surface_normals, [[1.0, 0.0, 0.0]] * num_of_points)
        elif self.strain_component == StrainComponents.orthogonal_to_normal_no_y:
            surface_normals = self.normalMeasurementVector(index)
            vectors = np.cross(surface_normals, [[0.0, 1.0, 0.0]] * num_of_points)
        elif self.strain_component == StrainComponents.orthogonal_to_normal_no_z:
            surface_normals = self.normalMeasurementVector(index)
            vectors = np.cross(surface_normals, [[0.0, 0.0, 1.0]] * num_of_points)
        elif self.strain_component == StrainComponents.custom:
            v = np.array(self.key_in) / np.linalg.norm(self.key_in)
            vectors = self.stackVectors(v, num_of_points)

        vectors = np.array(vectors) if not self.reverse else -np.array(vectors)
        if vectors.size != 0:
            self.presenter.model.addVectorsToProject(vectors, index, self.alignment, self.detector)

    def stackVectors(self, vector, count):
        vectors = []
        if self.point_index == -1:
            vectors.extend([vector] * count)
        else:
            vectors.append(vector)

        return vectors

    def normalMeasurementVector(self, index):
        # Only first or main sample model is used to compute vector
        mesh = list(self.presenter.model.sample.items())[0][1]
        if self.point_index == -1:
            points = self.presenter.model.measurement_points.points[index]
        else:
            points = self.presenter.model.measurement_points.points[index, None]

        vertices = mesh.vertices[mesh.indices]
        face_vertices = vertices.reshape(-1, 9)
        faces = closest_triangle_to_point(face_vertices, points)

        return compute_face_normals(faces)

    def onImportSuccess(self):
        self.presenter.view.docks.showVectorManager()

    def onImportFailed(self, exception):
        msg = 'An error occurred while creating the measurement vectors'
        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class RemoveVectorAlignment(QtWidgets.QUndoCommand):
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


class RemoveVectors(QtWidgets.QUndoCommand):
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


class InsertAlignmentMatrix(QtWidgets.QUndoCommand):
    def __init__(self, matrix, presenter):
        """ Command to insert primitive to the sample list

        :param matrix: transformation matrix
        :type matrix: Matrix44
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.window.presenter.MainWindowPresenter
        """
        super().__init__()

        self.model = presenter.model
        self.old_matrix = self.model.alignment
        self.new_matrix = matrix

        self.setText('Align Sample on Instrument.')

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
        """ Returns ID used when merging commands"""
        return CommandID.AlignSample
