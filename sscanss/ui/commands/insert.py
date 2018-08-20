from collections import OrderedDict
import logging
import os
import numpy as np
from PyQt5 import QtWidgets
from sscanss.core.util import Primitives, Worker, PointType, LoadVector, MessageSeverity
from sscanss.core.mesh import create_tube, create_sphere, create_cylinder, create_cuboid


class InsertPrimitive(QtWidgets.QUndoCommand):
    def __init__(self, primitive, args, presenter, combine):
        """ Command to insert primitive to the sample list

        :param primitive: primitive to insert
        :type primitive: sscanss.core.util.misc.Primitives
        :param args: arguments for primitive creation
        :type args: Dict
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.windows.main.presenter.MainWindowPresenter
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
        :type presenter: sscanss.ui.windows.main.presenter.MainWindowPresenter
        :param combine: when True model is added to current otherwise replaces it
        :type combine: bool
        """
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.combine = combine

        base_name = os.path.basename(filename)
        name, ext = os.path.splitext(base_name)
        ext = ext.replace('.', '').lower()
        self.sample_key = self.presenter.model.uniqueKey(name, ext)
        self.setText('Insert {}'.format(base_name))

    def redo(self):
        if not self.combine:
            self.old_sample = self.presenter.model.sample
        load_sample_args = [self.filename, self.combine]
        self.presenter.view.showProgressDialog('Loading 3D Model')
        self.worker = Worker(self.presenter.model.loadSample, load_sample_args)
        self.worker.job_succeeded.connect(self.onImportSuccess)
        self.worker.finished.connect(self.presenter.view.progress_dialog.close)
        self.worker.job_failed.connect(self.onImportFailed)
        self.worker.start()

    def undo(self):
        if self.combine:
            self.presenter.model.removeMeshFromProject(self.sample_key)
        else:
            self.presenter.model.sample = self.old_sample

    def onImportSuccess(self):
        if len(self.presenter.model.sample) > 1:
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
        :type presenter: sscanss.ui.windows.main.presenter.MainWindowPresenter
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
        :type presenter: sscanss.ui.windows.main.presenter.MainWindowPresenter
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
            temp[key] = mesh.splitAt(index) if index != 0 else mesh

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
        :type presenter: sscanss.ui.windows.main.presenter.MainWindowPresenter
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
        return 1000


class InsertPointsFromFile(QtWidgets.QUndoCommand):
    def __init__(self, filename, point_type, presenter):
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.point_type = point_type

        if self.point_type == PointType.Fiducial:
            self.old_count = len(self.presenter.model.fiducials)
        else:
            self.old_count = len(self.presenter.model.measurement_points)

        self.setText('Import {} Points'.format(self.point_type.value))

    def redo(self):
        load_points_args = [self.filename, self.point_type]
        self.presenter.view.showProgressDialog('Loading {} Points'.format(self.point_type.value))
        self.worker = Worker(self.presenter.model.loadPoints, load_points_args)
        self.worker.job_succeeded.connect(self.onImportSuccess)
        self.worker.finished.connect(self.presenter.view.progress_dialog.close)
        self.worker.job_failed.connect(self.onImportFailed)
        self.worker.start()

    def undo(self):
        if self.point_type == PointType.Fiducial:
            current_count = len(self.presenter.model.fiducials)
        else:
            current_count = len(self.presenter.model.measurement_points)

        self.presenter.model.removePointsFromProject(slice(self.old_count, current_count, None), self.point_type)

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

        self.setText('Add {} Points'.format(self.point_type.value))

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
            self.old_values = self.model.fiducials[self.indices]
        else:
            self.old_values = self.model.measurement_points[self.indices]
        self.model.removePointsFromProject(self.indices, self.point_type)

    def undo(self):
        if self.point_type == PointType.Fiducial:
            points = self.model.fiducials
        else:
            points = self.model.measurement_points
        for index, value in enumerate(self.indices):
            if index < len(points):
                points = np.insert(points, value, self.old_values[index], 0)
            else:
                temp = np.rec.array(self.old_values[index], dtype=self.model.point_dtype)
                points = np.append(points, temp)

        if self.point_type == PointType.Fiducial:
            self.model.fiducials = points.view(np.recarray)
        else:
            self.model.measurement_points = points.view(np.recarray)


class MovePoints(QtWidgets.QUndoCommand):
    def __init__(self, move_from, move_to, point_type, presenter):
        super().__init__()

        self.move_from = move_from
        self.move_to = move_to
        self.model = presenter.model
        self.point_type = point_type

        if self.point_type == PointType.Fiducial:
            points = self.model.fiducials
        else:
            points = self.model.measurement_points

        self.old_order = list(range(0, len(points)))
        self.new_order = list(range(0, len(points)))
        self.new_order[move_from], self.new_order[move_to] = self.new_order[move_to], self.new_order[move_from]

        self.setText('Change {} Point Index'.format(self.point_type.value))

    def redo(self):
        if self.point_type == PointType.Fiducial:
            points = self.model.fiducials
        else:
            points = self.model.measurement_points

        points[self.old_order] = points[self.new_order]

        # This is necessary because it emits changed signal for the point
        if self.point_type == PointType.Fiducial:
            self.model.fiducials = points
        else:
            self.model.measurement_points = points

    def undo(self):
        if self.point_type == PointType.Fiducial:
            points = self.model.fiducials
        else:
            points = self.model.measurement_points

        points[self.new_order] = points[self.old_order]

        # This is necessary because it emits changed signal for the point
        if self.point_type == PointType.Fiducial:
            self.model.fiducials = points
        else:
            self.model.measurement_points = points

    def mergeWith(self, command):
        if self.point_type != command.point_type:
            return False

        move_to = command.move_to
        move_from = command.move_from
        self.new_order[move_from], self.new_order[move_to] = self.new_order[move_to], self.new_order[move_from]

        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return 1001


class EditPoints(QtWidgets.QUndoCommand):
    def __init__(self, row, value, point_type, presenter):
        super().__init__()

        self.model = presenter.model
        self.point_type = point_type

        if self.point_type == PointType.Fiducial:
            points = self.model.fiducials
        else:
            points = self.model.measurement_points

        old_values = (np.copy(points.points[row]), points.enabled[row])
        self.old_values = {row: old_values}
        self.new_values = {row: value}

        self.setText('Edit {} Points'.format(self.point_type.value))

    def redo(self):
        if self.point_type == PointType.Fiducial:
            points = self.model.fiducials
        else:
            points = self.model.measurement_points

        for key, value in self.new_values.items():
            points[key] = value

        # This is necessary because it emits changed signal for the point
        if self.point_type == PointType.Fiducial:
            self.model.fiducials = points
        else:
            self.model.measurement_points = points

    def undo(self):
        if self.point_type == PointType.Fiducial:
            points = self.model.fiducials
        else:
            points = self.model.measurement_points

        for key, value in self.old_values.items():
            points[key] = value

        # This is necessary because it emits changed signal for the point
        if self.point_type == PointType.Fiducial:
            self.model.fiducials = points
        else:
            self.model.measurement_points = points

    def mergeWith(self, command):
        if self.point_type != command.point_type:
            return False

        self.new_values.update(command.new_values)
        command.old_values.update(self.old_values)
        self.old_values = command.old_values

        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return 1002


class InsertVectorsFromFile(QtWidgets.QUndoCommand):
    def __init__(self, filename, presenter):
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.old_vectors = np.copy(self.presenter.model.measurement_vectors)

        self.setText('Import Measurement vectors')

    def redo(self):
        load_vectors_args = [self.filename]
        self.presenter.view.showProgressDialog('Loading Measurement vectors')
        self.worker = Worker(self.presenter.model.loadVectors, load_vectors_args)
        self.worker.job_succeeded.connect(self.onImportSuccess)
        self.worker.finished.connect(self.presenter.view.progress_dialog.close)
        self.worker.job_failed.connect(self.onImportFailed)
        self.worker.start()

    def undo(self):
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

    def onImportFailed(self, exception):
        msg = 'An error occurred while loading the measurement vectors.\n\n' \
              'Please check that the file is valid.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()
