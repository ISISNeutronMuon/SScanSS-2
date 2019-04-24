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
    def __init__(self, result_id,  error, q, labels, format_flag, code):
        self.id = result_id
        self.q = q
        self.error = error
        self.code = code
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
        self.vectors = vectors[points.enabled, :, :]
        self.count = self.vectors.shape[0] * self.vectors.shape[2]

        matrix = alignment.transpose()
        self.points = self.points @ matrix[0:3, 0:3] + matrix[3, 0:3]
        for k in range(self.vectors.shape[2]):
            for j in range(0, self.vectors.shape[1], 3):
                self.vectors[:, j:j+3, k] = self.vectors[:, j:j+3, k] @ matrix[0:3, 0:3]

    def run(self):
        q_vec = np.array([[-0.70710678, 0.70710678, 0.], [-0.70710678, -0.70710678, 0.]])
        try:
            for i in range(self.vectors.shape[0]):
                for j in range(self.vectors.shape[2]):
                    all_mvs = self.vectors[i, :, j].reshape(-1, 3)
                    selected = np.where(np.linalg.norm(all_mvs, axis=1) > 0.00001)[0]  # greater than epsilon
                    if selected.size == 0:
                        q_vectors = np.atleast_2d(q_vec[0])
                        measurement_vectors = np.atleast_2d(self.positioner.pose[0:3, 0:3].transpose() @ q_vec[0])
                    else:
                        q_vectors = np.atleast_2d(q_vec[selected])
                        measurement_vectors = np.atleast_2d(all_mvs[selected])

                    r, error, code = self.positioner.ikine([self.points[i, :], measurement_vectors],
                                                           [np.array([0., 0., 0.]), q_vectors])
                    if self._abort:
                        break

                    self.results.append(SimulationResult(f'Point {i+1}, Alignment {j+1}', error, r, self.joint_labels,
                                                         self.format_flag, code))
                    self.point_finished.emit()
                    self.positioner.fkine(r)
                    self.msleep(500)
                    if self._abort:
                        break
        except Exception:
            # TODO: add proper exception handling for Value error and  Memory error
            pass

    def abort(self):
        if self.receivers(self.point_finished) > 0:
            self.point_finished.disconnect()
        self._abort = True
        self.quit()
