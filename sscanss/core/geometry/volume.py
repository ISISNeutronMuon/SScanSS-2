"""
Classes for Volume objects
"""


class Volume:
    """Creates a Volume object. This is the result of loading in a tomography scan, either from a nexus file, or a set of TIFF files.
     It is the equivalent of the mesh object but for tomography data

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
