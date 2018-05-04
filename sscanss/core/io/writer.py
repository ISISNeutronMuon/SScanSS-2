import numpy as np


def write_project_hdf(data, filename):
    """

    :param data:
    :type data:
    :param filename:
    :type filename:
    """
    import h5py
    import datetime as dt
    from sscanss.version import __version__

    with h5py.File(filename, 'w') as hdf_file:
        hdf_file.attrs['name'] = np.string_(data['name'])
        hdf_file.attrs['instrument'] = np.string_(data['instrument'])
        hdf_file.attrs['version'] = np.string_(__version__)

        date_created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdf_file.attrs['date_created'] = np.string_(date_created)
