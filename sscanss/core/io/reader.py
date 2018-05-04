

def read_project_hdf(filename):
    """

    :param filename: path of the hdf file
    :type filename: str
    :return: A dictionary containing the project data
    :rtype: dict
    """
    import h5py

    data = {}
    encoding = 'ascii'
    with h5py.File(filename, 'r') as hdf_file:

        data['name'] = hdf_file.attrs['name'].decode(encoding)
        data['instrument'] = hdf_file.attrs['instrument'].decode(encoding)

    return data
