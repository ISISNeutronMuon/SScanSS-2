"""
Classes for Volume objects
"""
import numpy as np
from ..math.matrix import Matrix44


class Volume:
    """Creates a Volume object. This is the result of loading in a tomography scan, either from a nexus file,
    or a set of TIFF files. It is the equivalent of the mesh object but for tomography data

    :param data: N x M x L array of intensities, created by stacking L TIFF images, each of dimension N x M
    :type data: numpy.ndarray
    :param x: N array of pixel co-ordinates
    :type x: numpy.ndarray
    :param y: M array of pixel co-ordinates
    :type y: numpy.ndarray
    :param z: L array of pixel co-ordinates
    :type z: numpy.ndarray
    """
    def __init__(self, data, x, y, z):
        self.data = data
        self.x = x
        self.y = y
        self.z = z

        self.transfer_function = np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 4))
        self.transfer_function[:51, :] = 0
        self.transfer_function = self.transfer_function.flatten()

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
