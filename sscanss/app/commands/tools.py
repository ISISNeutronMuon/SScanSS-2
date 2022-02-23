import numpy as np
from PyQt5 import QtWidgets
from sscanss.core.math import Vector3, Matrix44, matrix_from_zyx_eulers
from sscanss.core.util import Attributes


class RotateSample(QtWidgets.QUndoCommand):
    """Creates command to rotate a sample by specified ZYX euler angles

    :param angles: ZYX euler angle in degrees
    :type angles: List[float]
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, angles, presenter):
        super().__init__()
        self.angles = Vector3(np.radians(angles))
        self.model = presenter.model

        self.setText('Rotate Sample')

    def redo(self):
        matrix = matrix_from_zyx_eulers(self.angles)
        self.rotate(matrix)

    def undo(self):
        matrix = matrix_from_zyx_eulers(self.angles)
        self.rotate(matrix.transpose())

    def rotate(self, matrix):
        """Rotates sample, fiducials, measurements, and alignment matrix if present

        :param matrix: rotation matrix
        :type matrix: Matrix33
        """
        self.model.sample.rotate(matrix)
        _matrix = matrix.transpose()
        self.model.fiducials.points = self.model.fiducials.points @ _matrix
        self.model.measurement_points.points = self.model.measurement_points.points @ _matrix
        for k in range(self.model.measurement_vectors.shape[2]):
            for j in range(0, self.model.measurement_vectors.shape[1], 3):
                self.model.measurement_vectors[:, j:j + 3, k] = self.model.measurement_vectors[:, j:j + 3, k] @ _matrix
        if self.model.alignment is not None:
            self.model.alignment[0:3, 0:3] = self.model.alignment[0:3, 0:3] @ _matrix

        self.model.notifyChange(Attributes.Sample)
        self.model.notifyChange(Attributes.Fiducials)
        self.model.notifyChange(Attributes.Measurements)
        self.model.notifyChange(Attributes.Vectors)


class TranslateSample(QtWidgets.QUndoCommand):
    """Creates command to translate a sample by specified offsets in the X,Y, or Z direction

    :param offset: XYZ offsets
    :type offset: List[float]
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, offset, presenter):
        super().__init__()
        self.offset = np.array(offset)
        self.model = presenter.model

        self.setText('Translate Sample')

    def redo(self):
        self.translate(self.offset)

    def undo(self):
        self.translate(-self.offset)

    def translate(self, offset):
        """Translates sample, fiducials, measurements and alignment matrix if present

        :param offset: X, Y, and Z axis offsets
        :type offset: List[float]
        """
        self.model.sample.translate(offset)
        self.model.fiducials.points = self.model.fiducials.points + offset
        self.model.measurement_points.points = self.model.measurement_points.points + offset
        if self.model.alignment is not None:
            self.model.alignment = self.model.alignment @ Matrix44.fromTranslation(-offset)

        self.model.notifyChange(Attributes.Sample)
        self.model.notifyChange(Attributes.Fiducials)
        self.model.notifyChange(Attributes.Measurements)
        self.model.notifyChange(Attributes.Vectors)


class TransformSample(QtWidgets.QUndoCommand):
    """Creates command to transform a sample with a specified 4 x 4 matrix

    :param matrix: 4 x 4 matrix
    :type matrix: List[List[float]]
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, matrix, presenter):

        super().__init__()
        self.matrix = Matrix44(matrix)
        self.model = presenter.model

        self.setText('Transform Sample')

    def redo(self):
        self.transform(self.matrix)

    def undo(self):
        self.transform(self.matrix.inverse())

    def transform(self, matrix):
        """Transforms sample, fiducials, measurements and alignment matrix if present

        :param matrix: transformation matrix
        :type matrix: Matrix44
        """
        self.model.sample.transform(matrix)
        _matrix = matrix[0:3, 0:3].transpose()
        _offset = matrix[0:3, 3].transpose()
        self.model.fiducials.points = self.model.fiducials.points @ _matrix + _offset
        self.model.measurement_points.points = self.model.measurement_points.points @ _matrix + _offset
        for k in range(self.model.measurement_vectors.shape[2]):
            for j in range(0, self.model.measurement_vectors.shape[1], 3):
                self.model.measurement_vectors[:, j:j + 3, k] = self.model.measurement_vectors[:, j:j + 3, k] @ _matrix

        if self.model.alignment is not None:
            self.model.alignment = self.model.alignment @ matrix.inverse()

        self.model.notifyChange(Attributes.Sample)
        self.model.notifyChange(Attributes.Fiducials)
        self.model.notifyChange(Attributes.Measurements)
        self.model.notifyChange(Attributes.Vectors)
