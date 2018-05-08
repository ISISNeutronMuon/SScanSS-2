import numpy as np


def write_project_hdf(data, filename):
    """

    :param data: A dictionary containing the project data
    :type data: dict
    :param filename: path of the hdf file to save
    :type filename: str
    """
    import h5py
    import datetime as dt
    from sscanss.version import __version__

    with h5py.File(filename, 'w') as hdf_file:
        hdf_file.attrs['name'] = data['name']
        hdf_file.attrs['instrument'] = data['instrument']
        hdf_file.attrs['version'] = __version__

        date_created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdf_file.attrs['date_created'] = date_created
