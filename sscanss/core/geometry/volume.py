"""
Classes for Volume objects
"""


class Volume:
    """Creates a Volume object. This is the result of loading in a tomography scan, either from a nexus file, or a set of TIFF files.
     It is the equivalent of the mesh object but for tomography data

    :param data: N x M x L array of intensities, created by stacking L TIFF images, each of dimension N x M
    :type pixel_intensities: numpy.ndarray
    :param data_x_axis: N array of pixel co-ordinates
    :type data_x_axis: numpy.array
    :param data_y_axis: M array of pixel co-ordinates
    :type data_y_axis: numpy.array
    :param data_z_axis: L array of pixel co-ordinates
    :type data_z_axis: numpy.array
    """
    def __init__(self, data, x, y, z):
        self.data = data
        self.x = x
        self.y = y
        self.z = z

