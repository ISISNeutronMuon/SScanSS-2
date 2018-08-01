import re
import numpy as np
from sscanss.core.mesh import Mesh


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
    with open(filename, encoding='utf-8') as stl_file:
        offset = 21

        stl_file.readline()
        text = stl_file.read()
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

        vertices = vertices.reshape(-1, 3)
        indices = np.arange(face_count * 3)
        normals = np.repeat(normals, 3, axis=0)

        return Mesh(vertices, indices, normals)


def read_binary_stl(filename):
    with open(filename, 'rb') as stl_file:
        stl_file.seek(80)
        face_count = np.frombuffer(stl_file.read(4), dtype=np.int32)[0]

        record_dtype = np.dtype([
            ('normals', np.float32, (3,)),
            ('vertices', np.float32, (3, 3)),
            ('attr', '<i2', (1,)),
        ])
        data = np.fromfile(stl_file, dtype=record_dtype)

    if face_count != data.size:
        raise IOError('stl data has incorrect size')

    vertices = data['vertices'].reshape(-1, 3)
    indices = np.arange(face_count * 3)
    normals = np.repeat(data['normals'], 3, axis=0)

    return Mesh(vertices, indices, normals)


def read_obj(filename):
    vertices = []
    faces = []
    with open(filename, encoding='utf-8') as obj_file:
        for line in obj_file:
            prefix = line[0:2].lower()
            if prefix == 'v ':
                vertices.append(line[1:].split())
            elif prefix == 'f ':
                temp = [val.split('/')[0] for val in line[1:].split()]
                faces.extend(temp[0:3])

    vertices = np.array(vertices, dtype=np.float32)[:, 0:3]

    face_index = np.array(faces, dtype=int) - 1
    vertices = vertices[face_index, :]
    indices = np.arange(face_index.size)

    return Mesh(vertices, indices)


def read_points(filename):
    points = []
    enabled = []
    regex = re.compile(r'(\s+|(\s*,\s+))')
    with open(filename) as csv_file:
        for line in csv_file:
            line = regex.sub(line, ' ')
            row = line.split()
            if len(row) == 3:
                points.append(row)
                enabled.append(True)
            elif len(row) == 4:
                *p, d = row
                d = True if d.lower() == 'true' else False
                points.append(p)
                enabled.append(d)
            else:
                raise ValueError('data has incorrect size')

        return points, enabled
