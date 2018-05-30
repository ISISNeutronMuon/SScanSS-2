import numpy as np

def read_project_hdf(filename):
    """

    :param filename: path of the hdf file to read
    :type filename: str
    :return: A dictionary containing the project data
    :rtype: dict
    """
    import h5py

    data = {}
    with h5py.File(filename, 'r') as hdf_file:

        data['name'] = hdf_file.attrs['name']
        data['instrument'] = hdf_file.attrs['instrument']

    return data


def read_stl(filename):
    try:
        return read_ascii_stl(filename)
    except (UnicodeDecodeError, IOError):
        return read_binary_stl(filename)


def read_ascii_stl(filename):
    # This is much slower than the binary version due to the string split but will have to do for now
    with open(filename, encoding='utf-8') as f:
        offset = 21

        f.readline()
        text = f.read()
        text = text.lower().rsplit('endsolid', 1)[0]
        text = np.array(text.split())
        text_size = len(text)

        if text_size % offset != 0:
            raise IOError('stl data has incorrect size')

        face_count = int(text_size / offset)
        text = text.reshape(-1, offset)
        data_pos = [2, 3, 4, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        normals = text[:, data_pos[0:3]].astype(np.float32)
        vertices = text[:, data_pos[3:]].astype(np.float32)

        points = vertices.reshape(-1, 3)
        indices = np.arange(face_count * 3)
        normals = np.repeat(normals, 3, axis=0)

        return {'vertices': points, 'indices': indices, 'normals': normals}


def read_binary_stl(filename):
    with open(filename, 'rb') as f:
        f.seek(80)
        face_count = np.frombuffer(f.read(4), dtype=np.int32)[0]

        record_dtype = np.dtype([
            ('normals', np.float32, (3,)),
            ('vertices', np.float32, (3, 3)),
            ('attr', '<i2', (1,)),
        ])
        data = np.fromfile(f, dtype=record_dtype)

    if face_count != data.size:
        raise IOError('stl data has incorrect size')

    points = data['vertices'].reshape(-1, 3)
    indices = np.arange(face_count * 3)
    normals = np.repeat(data['normals'], 3, axis=0)

    return {'vertices': points, 'indices': indices, 'normals': normals}
