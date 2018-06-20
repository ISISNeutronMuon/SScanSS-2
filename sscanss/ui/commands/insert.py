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
        self.sample_key = self.presenter.model.create_unique_key(name, ext)
        self.setText('Insert {}'.format(base_name))

    def redo(self):
        load_sample_args = [self.filename, self.combine]
        self.presenter.view.showProgressDialog('Loading 3D Model')
        self.worker = Worker(self.presenter.model.loadSample, load_sample_args)
        self.worker.job_succeeded.connect(self.presenter.view.showSampleManager)
        self.worker.finished.connect(self.presenter.view.progress_dialog.close)
        self.worker.job_failed.connect(self.onImportFailed)
        self.worker.start()

    def undo(self):
        if self.combine:
            self.presenter.model.removeMeshFromProject(self.sample_key)
        else:
            self.presenter.model.sample = self.old_sample

    def onImportFailed(self, exception):
        msg = 'An error occurred while loading the 3D model.\n\n' \
              'Please check that the file is valid.'

        logging.error(msg, exc_info=exception)
        self.presenter.view.showErrorMessage(msg)

        # Remove the failed command from the undo_stack
        self.setObsolete(True)
        self.presenter.view.undo_stack.undo()
