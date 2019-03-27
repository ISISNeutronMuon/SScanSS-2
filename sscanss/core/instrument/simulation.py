from collections import OrderedDict
import numpy as np
from PyQt5 import QtCore

POINT_DTYPE = [('points', 'f4', 3), ('enabled', '?')]


class SampleAssembly:
    def __init__(self, sample=None, fiducials=None, measurements=None, vectors=None):
        self.samples = OrderedDict() if sample is None else sample
        self.fiducials = np.recarray((0,), dtype=POINT_DTYPE) if fiducials is None else fiducials
        self.measurements = np.recarray((0,), dtype=POINT_DTYPE) if measurements is None else measurements
        self.vectors = np.empty((0, 3, 1), dtype=np.float32) if vectors is None else vectors

    def transform(self, matrix):
        assembly = self.transformed(matrix)
        self.sample = assembly.sample
        self.fiducials = assembly.fiducials
        self.measurements = assembly.measurements
        self.vectors = assembly.vectors

    def transformed(self, matrix):
        _matrix = matrix[0:3, 0:3].transpose()
        offset = matrix[0:3, 3].transpose()

        samples = OrderedDict()

        for key, sample in self.samples.items():
            samples[key] = sample.transformed(matrix)

        fiducials = self.fiducials.copy()
        fiducials.points = fiducials.points @ _matrix + offset

        measurements = self.measurements.copy()
        measurements.points = measurements.points @ _matrix + offset
        vectors = self.vectors.copy()
        for k in range(vectors.shape[2]):
            for j in range(0, vectors.shape[1], 3):
                vectors[:, j:j+3, k] = vectors[:, j:j+3, k] @ _matrix

        return SampleAssembly(samples, fiducials, measurements, vectors)


class SimulationResult:
    def __init__(self, result_id,  error, q, labels, format_flag):
        self.id = result_id
        self.q = q
        self.error = error
        self.format_flag = format_flag
        self.joint_labels = labels

    @property
    def formatted(self):
        q = self.q.copy()
        q[self.format_flag] = np.degrees(q[self.format_flag])

        return q


class Simulation(QtCore.QThread):
    point_finished = QtCore.pyqtSignal()

    def __init__(self, positioner, points, vectors, alignment):
        super().__init__()

        self._abort = False
        self.positioner = positioner
        self.joint_labels =[]
        self.format_flag = []
        for link in positioner.links:
            self.joint_labels.append(link.name)
            self.format_flag.append(True if link.type == link.Type.Revolute else False)

        self.results = []

        self.points = points.points[points.enabled]
        self.vectors = vectors[points.enabled, :, 0]
        self.count = self.points.shape[0]

        matrix = alignment.transpose()
        self.points = self.points @ matrix[0:3, 0:3] + matrix[3, 0:3]
        self.vectors[:, 0:3] = self.vectors[:, 0:3] @ matrix[0:3, 0:3]

    def run(self):
        q_vec_0 = np.array([-0.7071, -0.7071, 0.])
        q_vec_1 = np.array([-0.7071, 0.7071, 0.])

        for i in range(self.points.shape[0]):
            r = self.positioner.ikine(np.array([0., 0., 0., *q_vec_1]),
                                      np.array([*self.points[i, :], *self.vectors[i, 0:3]]))
            self.results.append(SimulationResult(f'Point {i+1}', 0.0, r, self.joint_labels, self.format_flag))
            self.point_finished.emit()
            self.positioner.fkine(r)
            self.msleep(500)
            if self._abort:
                break

    def abort(self):
        self._abort = True
        self.quit()
