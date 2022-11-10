"""
Classes for Volume objects
"""
from enum import Enum, unique
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from ..geometry.mesh import BoundingBox
from ..geometry.primitive import create_cuboid
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
    """Creates a Volume object.

    :param data: 3D image data
    :type data: numpy.ndarray[uint8]
    :param voxel_size: size of the volume's voxels in the x, y, and z axes
    :type voxel_size: numpy.ndarray[float32]
    :param centre: coordinates of the volume centre in the x, y, and z axes
    :type centre: numpy.ndarray[float32]
    :param histogram: 1D histogram of volume data
    :type histogram: Optional[Tuple[numpy.ndarray[int], numpy.ndarray[float]]]
    :param binned_data: binned volume data
    :type binned_data: Optional[numpy.ndarray[uint8]]
    """
    def __init__(self, data, voxel_size, centre, histogram=None, binned_data=None):
        self.data = data

        self.voxel_size = voxel_size
        self.transform_matrix = Matrix44.fromTranslation(centre)
        self.bounding_box = BoundingBox(self.extent / 2, -self.extent / 2)
        self.bounding_box = self.bounding_box.transform(self.transform_matrix)
        self.render_target = data if binned_data is None else binned_data
        self.histogram = histogram if histogram is not None else np.histogram(self.render_target, 256, (0, 255))
        inputs = np.array([self.histogram[1][0], self.histogram[1][-1]])
        outputs = np.array([0.0, 1.0])
        self.curve = Curve(inputs, outputs, inputs, Curve.Type.Cubic)

    @property
    def shape(self):
        """Returns shape of volume

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

    def rotate(self, matrix):
        """Performs in-place rotation of volume.

        :param matrix: 3 x 3 rotation matrix
        :type matrix: Union[numpy.ndarray, Matrix33]
        """
        rot_matrix = Matrix44.identity()
        rot_matrix[:3, :3] = matrix[:3, :3]
        self.transform(rot_matrix)

    def translate(self, offset):
        """Performs in-place translation of volume.

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: Union[numpy.ndarray, Vector3]
        """
        matrix = Matrix44.fromTranslation(offset)
        self.transform(matrix)

    def transform(self, matrix):
        """Performs in-place transformation of volume

        :param matrix: 4 x 4 transformation matrix
        :type matrix: Union[numpy.ndarray, Matrix44]
        """
        self.transform_matrix = matrix @ self.transform_matrix
        self.bounding_box = self.bounding_box.transform(matrix)

    def asMesh(self):
        """Creates a mesh from the bounds of the volume"""
        model_matrix = np.diag([*(0.5 * self.extent), 1])
        return create_cuboid(2, 2, 2).transformed(self.transform_matrix @ model_matrix)
