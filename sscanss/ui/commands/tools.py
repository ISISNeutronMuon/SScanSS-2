import numpy as np
from PyQt5 import QtWidgets
from sscanss.core.math import Vector3, matrix_from_xyz_eulers


class RotateSample(QtWidgets.QUndoCommand):
    def __init__(self, angles, sample_key, presenter):
        """ Command to rotate a sample by specified XYZ euler angles

        :param angles: XYZ euler angle in degrees
        :type angles: List[float]
        :param sample_key: key of sample to rotate or 'All' to rotate all samples
        :type sample_key: str
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.windows.main.presenter.MainWindowPresenter
        """
        super().__init__()
        self.angles = Vector3(np.radians(angles))
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

        self.model.updateSampleScene('sample')


class TranslateSample(QtWidgets.QUndoCommand):
    def __init__(self, offset, sample_key, presenter):
        """ Command to translate a sample by specified offsets in the X,Y, or Z direction

        :param offset: XYZ offsets
        :type offset: List[float]
        :param sample_key: key of sample to translate or 'All' to translate all samples
        :type sample_key: str
        :param presenter: Mainwindow presenter instance
        :type presenter: sscanss.ui.windows.main.presenter.MainWindowPresenter
        """
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

        self.model.updateSampleScene('sample')


