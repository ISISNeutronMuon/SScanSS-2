import numpy as np
from PyQt5 import QtWidgets
from sscanss.core.math import Vector3, Matrix44, matrix_from_zyx_eulers
from sscanss.core.util import Attributes


class RotateSample(QtWidgets.QUndoCommand):
    """Creates command to rotate a sample by specified ZYX euler angles

    :param angles: ZYX euler angle in degrees
    :type angles: List[float]
    :param sample_key: key of sample to rotate or None to rotate all samples
    :type sample_key: Union[str, None]
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """

    def __init__(self, angles, sample_key, presenter):
        super().__init__()
        self.angles = Vector3(np.radians(angles))
        self.key = sample_key
        self.model = presenter.model

        self.setText("Rotate Sample ({})".format(self.key))

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
        if self.key is not None:
            mesh = self.model.sample[self.key]
            mesh.rotate(matrix)
        else:
            for key in self.model.sample.keys():
                mesh = self.model.sample[key]
                mesh.rotate(matrix)

            _matrix = matrix.transpose()
            self.model.fiducials.points = self.model.fiducials.points @ _matrix
            self.model.measurement_points.points = self.model.measurement_points.points @ _matrix
            for k in range(self.model.measurement_vectors.shape[2]):
                for j in range(0, self.model.measurement_vectors.shape[1], 3):
                    self.model.measurement_vectors[:, j : j + 3, k] = (
                        self.model.measurement_vectors[:, j : j + 3, k] @ _matrix
                    )
            if self.model.alignment is not None:
                self.model.alignment[0:3, 0:3] = self.model.alignment[0:3, 0:3] @ _matrix

            self.model.notifyChange(Attributes.Fiducials)
            self.model.notifyChange(Attributes.Measurements)
            self.model.notifyChange(Attributes.Vectors)

        self.model.notifyChange(Attributes.Sample)


class TranslateSample(QtWidgets.QUndoCommand):
    """Creates command to translate a sample by specified offsets in the X,Y, or Z direction

    :param offset: XYZ offsets
    :type offset: List[float]
    :param sample_key: key of sample to translate or None to translate all samples
    :type sample_key: Union[str, None]
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """

    def __init__(self, offset, sample_key, presenter):
        super().__init__()
        self.offset = np.array(offset)
        self.key = sample_key
        self.model = presenter.model

        self.setText("Translate Sample ({})".format(self.key))

    def redo(self):
        self.translate(self.offset)

    def undo(self):
        self.translate(-self.offset)

    def translate(self, offset):
        """Translates sample, fiducials, measurements and alignment matrix if present

        :param offset: X, Y, and Z axis offsets
        :type offset: List[float]
        """
        if self.key is not None:
            mesh = self.model.sample[self.key]
            mesh.translate(offset)
        else:
            for key in self.model.sample.keys():
                mesh = self.model.sample[key]
                mesh.translate(offset)

            self.model.fiducials.points = self.model.fiducials.points + offset
            self.model.measurement_points.points = self.model.measurement_points.points + offset
            if self.model.alignment is not None:
                self.model.alignment = self.model.alignment @ Matrix44.fromTranslation(-offset)

            self.model.notifyChange(Attributes.Fiducials)
            self.model.notifyChange(Attributes.Measurements)
            self.model.notifyChange(Attributes.Vectors)

        self.model.notifyChange(Attributes.Sample)


class TransformSample(QtWidgets.QUndoCommand):
    """Creates command to transform a sample with a specified 4 x 4 matrix

    :param matrix: 4 x 4 matrix
    :type matrix: List[List[float]]
    :param sample_key: key of sample to translate or None to translate all samples
    :type sample_key: Union[str, None]
    :param presenter: main window presenter instance
    :type presenter: MainWindowPresenter
    """

    def __init__(self, matrix, sample_key, presenter):

        super().__init__()
        self.matrix = Matrix44(matrix)
        self.key = sample_key
        self.model = presenter.model

        self.setText("Transform Sample ({})".format(self.key))

    def redo(self):
        self.transform(self.matrix)

    def undo(self):
        self.transform(self.matrix.inverse())

    def transform(self, matrix):
        """Transforms sample, fiducials, measurements and alignment matrix if present

        :param matrix: transformation matrix
        :type matrix: Matrix44
        """
        if self.key is not None:
            mesh = self.model.sample[self.key]
            mesh.transform(matrix)
        else:
            for key in self.model.sample.keys():
                mesh = self.model.sample[key]
                mesh.transform(matrix)

            _matrix = matrix[0:3, 0:3].transpose()
            _offset = matrix[0:3, 3].transpose()
            self.model.fiducials.points = self.model.fiducials.points @ _matrix + _offset
            self.model.measurement_points.points = self.model.measurement_points.points @ _matrix + _offset
            for k in range(self.model.measurement_vectors.shape[2]):
                for j in range(0, self.model.measurement_vectors.shape[1], 3):
                    self.model.measurement_vectors[:, j : j + 3, k] = (
                        self.model.measurement_vectors[:, j : j + 3, k] @ _matrix
                    )

            if self.model.alignment is not None:
                self.model.alignment = self.model.alignment @ matrix.inverse()

            self.model.notifyChange(Attributes.Fiducials)
            self.model.notifyChange(Attributes.Measurements)
            self.model.notifyChange(Attributes.Vectors)

        self.model.notifyChange(Attributes.Sample)
