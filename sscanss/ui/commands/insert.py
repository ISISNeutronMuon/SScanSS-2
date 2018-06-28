from collections import OrderedDict
import logging
import os
from PyQt5 import QtWidgets
from sscanss.core.util import Primitives, Worker
from sscanss.core.mesh import (create_tube, create_sphere,
                               create_cylinder, create_cuboid)


class InsertPrimitive(QtWidgets.QUndoCommand):
    def __init__(self, primitive, args, presenter, combine):
        super().__init__()

        self.name = args.pop('name', 'unnamed')
        self.args = args
        self.primitive = primitive
        self.presenter = presenter
        self.combine = combine
        if not self.combine:
            self.old_sample = self.presenter.model.sample

        self.setText('Insert {}'.format(self.primitive.value))

    def redo(self):
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
        super().__init__()

        self.filename = filename
        self.presenter = presenter
        self.combine = combine
        if not self.combine:
            self.old_sample = self.presenter.model.sample

        base_name = os.path.basename(filename)
        name, ext = os.path.splitext(base_name)
        ext = ext.replace('.', '').lower()
        self.sample_key = self.presenter.model.uniqueKey(name, ext)
        self.setText('Insert {}'.format(base_name))

    def redo(self):
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
            self.presenter.view.showSampleManager()

    def onImportFailed(self, exception):
        msg = 'An error occurred while loading the 3D model.\n\n' \
              'Please check that the file is valid.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showErrorMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()


class DeleteSample(QtWidgets.QUndoCommand):
    def __init__(self, sample_key, presenter):
        super().__init__()

        self.keys = sample_key
        self.deleted_mesh = {}
        self.model = presenter.model

        self.old_keys = list(self.model.sample.keys())
        for key, mesh in self.model.sample.items():
            self.deleted_mesh[key] = mesh

        if len(sample_key) > 1:
            self.setText('Delete {} Samples'.format(len(sample_key)))
        else:
            self.setText('Delete {}'.format(sample_key[0]))

    def redo(self):
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
        super().__init__()

        self.keys = sample_key
        self.merged_mesh = []
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
