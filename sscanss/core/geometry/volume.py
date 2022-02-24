"""
Classes for Volume objects
"""
from enum import Enum, unique
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from ..math.matrix import Matrix44


class Curve:
    """Creates a Curve object used to generate transfer function for volumes

    :param inputs: input volume intensities
    :type inputs: numpy.ndarray
    :param outputs: output colour alpha
    :type outputs: numpy.ndarray
    :param bounds: minimum and maximum intensity in volume
    :type bounds: Tuple[float, float]
    :param curve_type: Type of fir for curve
    :type curve_type: Curve.Type
    """
    @unique
    class Type(Enum):
        """Type of curve"""
        Cubic = 'Cubic'
        Linear = 'Linear'

    def __init__(self, inputs, outputs, bounds, curve_type):
        self.inputs = inputs
        self.outputs = outputs
        self.bounds = bounds
        self.type = curve_type
        self.f = None

        self.transfer_function = np.tile(np.linspace(0.0, 1.0, num=256, dtype=np.float32)[:, None], (1, 4))
        if len(inputs) > 1:
            if curve_type == self.Type.Cubic:
                self.f = CubicSpline(inputs, outputs)
            else:
                self.f = interp1d(inputs, outputs, kind='linear', bounds_error=False, assume_sorted=True)

        value = self.evaluate(np.linspace(bounds[0], bounds[-1], num=256))
        self.transfer_function[:, 3] = value
        self.transfer_function = self.transfer_function.flatten()

    def evaluate(self, inputs):
        """Computes the outputs alpha values for the input intensity

        :param inputs: input volume intensities
        :type inputs: numpy.ndarray
        :return: output colour alpha
        :rtype: numpy.ndarray
        """
        if self.f is None:
            outputs = np.clip(np.full(len(inputs), self.outputs[0]), 0.0, 1.0)
        else:
            outputs = np.clip(self.f(inputs), 0.0, 1.0)

        outputs[inputs < self.inputs[0]] = self.outputs[0]
        outputs[inputs > self.inputs[-1]] = self.outputs[-1]

        return outputs


class Volume:
    """Creates a Volume object. This is the result of loading in a tomography scan, either from a nexus file,
    or a set of TIFF files. It is the equivalent of the mesh object but for tomography data

    :param data: N x M x L array of intensities, created by stacking L TIFF images, each of dimension N x M
    :type data: numpy.ndarray
    :param x: N array of pixel coordinates
    :type x: numpy.ndarray
    :param y: M array of pixel coordinates
    :type y: numpy.ndarray
    :param z: L array of pixel coordinates
    :type z: numpy.ndarray
    """
    def __init__(self, data, x, y, z):
        self.data = data
        self.x = x
        self.y = y
        self.z = z

        self.histogram = np.histogram(data, bins=256)
        inputs = np.array([self.histogram[1][0], self.histogram[1][-1]])
        outputs = np.array([0.0, 1.0])

        if inputs[0] == inputs[1]:
            inputs = inputs[1:]
            outputs = outputs[1:]

        self.curve = Curve(inputs, outputs, inputs, Curve.Type.Cubic)

        x_spacing = (x[-1] - x[0]) / (len(x) - 1)
        y_spacing = (y[-1] - y[0]) / (len(y) - 1)
        z_spacing = (z[-1] - z[0]) / (len(z) - 1)

        x_origin = x[0] + (x[-1] - x[0]) / 2
        y_origin = y[0] + (y[-1] - y[0]) / 2
        z_origin = z[0] + (z[-1] - z[0]) / 2

        self.voxel_size = np.array([x_spacing, y_spacing, z_spacing], np.float32)
        self.transform = Matrix44.fromTranslation([x_origin, y_origin, z_origin])

    @property
    def shape(self):
        """Returns shape of volume i.e. width, height, depth

        :return: shape of volume
        :rtype: Tuple[int, int, int]
        """
        return self.data.shape

    @property
    def extent(self):
        """Returns extent or diagonal of volume

        :return: extent of volume
        :rtype: numpy.ndarray[float]
        """
        return self.voxel_size * self.shape
