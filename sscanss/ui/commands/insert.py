from PyQt5 import QtWidgets
from sscanss.core.util import Primitives
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

        self.sample_key = self.presenter.model.addMeshToProject(self.name, mesh, 'auto', self.combine)
        self.presenter.setScene()

    def undo(self):
        self.presenter.model.removeMeshFromProject(self.sample_key)
        self.presenter.setScene()
