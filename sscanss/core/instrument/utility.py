from collections import OrderedDict
import numpy as np

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
