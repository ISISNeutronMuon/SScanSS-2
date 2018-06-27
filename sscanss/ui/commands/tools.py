import numpy as np
from PyQt5 import QtWidgets
from sscanss.core.transform import matrix_from_xyz_eulers


class RotateSample(QtWidgets.QUndoCommand):
    def __init__(self, angles, sample_key, presenter):
        super().__init__()
        self.angles = np.radians(angles)
        self.key = sample_key
        self.model = presenter.model

        self.setText('Rotate Sample ({})'.format(self.key))

    def redo(self):
        matrix = matrix_from_xyz_eulers(self.angles)
        self.rotate(matrix)

    def undo(self):
        matrix = matrix_from_xyz_eulers(self.angles)
        self.rotate(matrix.transpose())

    def rotate(self, matrix):
        if self.key == 'All':
            for key in self.model.sample.keys():
                mesh = self.model.sample[key]
                mesh.rotate(matrix)
        else:
            mesh = self.model.sample[self.key]
            mesh.rotate(matrix)

        self.model.updateSampleScene()


class TranslateSample(QtWidgets.QUndoCommand):
    def __init__(self, offset, sample_key, presenter):
        super().__init__()
        self.offset = np.array(offset)
        self.key = sample_key
        self.model = presenter.model

        self.setText('Translate Sample ({})'.format(self.key))

    def redo(self):
        self.translate(self.offset)

    def undo(self):
        self.translate(-self.offset)

    def translate(self, offset):
        if self.key == 'All':
            for key in self.model.sample.keys():
                mesh = self.model.sample[key]
                mesh.translate(offset)
        else:
            mesh = self.model.sample[self.key]
            mesh.translate(offset)

        self.model.updateSampleScene()

