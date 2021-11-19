"""
Classes for Volume objects
"""
import numpy as np
import psutil
import h5py
import tifffile as tiff


class Volume:
    """Creates a Volume object. This is the result of loading in a tomography scan, either from a nexus file, or a set of TIFF files.
     It is the equivalent of the mesh object

    :param pixel_intensities: N x M x L array of intensities, created by stacking L TIFF images, each of dimension N x M
    :type pixel_intensities: numpy.ndarray
    :param pixel_positions: N x 3 array of co-ordinates
    :type pixel_positions: numpy.ndarray
    """

    def __init__(self, data_x_axis, data_y_axis, data_z_axis, data):
        self.data_x_axis = data_x_axis
        self.data_y_axis = data_y_axis
        self.data_z_axis = data_z_axis
        self.data = data

    def redo(self):
        pass

    def undo(self):
        pass

