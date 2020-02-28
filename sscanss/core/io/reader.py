"""
A collection of functions for reading data
"""
import re
import os
from collections import OrderedDict
import h5py
import numpy as np
from ..geometry.mesh import Mesh
from ..geometry.colour import Colour
from ..instrument.instrument import Instrument, Collimator, Detector, Jaws, Script
from ..instrument.robotics import Link, SerialManipulator
from ..math.matrix import Matrix44
from ..math.vector import Vector3


def read_project_hdf(filename):
    """Reads the project data dictionary from a hdf file

    :param filename: path of the hdf file
    :type filename: str
    :return: A dictionary containing the project data
    :rtype: Dict
    :raises: ValueError
    """
    data = {}
    with h5py.File(filename, 'r') as hdf_file:

        data['name'] = hdf_file.attrs['name']
        data['version'] = hdf_file.attrs['version']
        data['instrument_version'] = hdf_file.attrs['instrument_version']
        data['instrument'] = hdf_file.attrs['instrument_name']

        data['settings'] = {}
        setting_group = hdf_file.get('settings')
        if setting_group is not None:
            for key, value in setting_group.attrs.items():
                data['settings'][key] = value

        sample_group = hdf_file['sample']
        sample = OrderedDict()
        for key in sample_group.keys():
            vertices = np.array(sample_group[key]['vertices'])
            indices = np.array(sample_group[key]['indices'])

            sample[key] = Mesh(vertices, indices)

        data['sample'] = sample

        fiducial_group = hdf_file['fiducials']
        points = np.array(fiducial_group['points'])
        enabled = np.array(fiducial_group['enabled'])
        data['fiducials'] = (points, enabled)

        measurement_group = hdf_file['measurement_points']
        points = np.array(measurement_group['points'])
        enabled = np.array(measurement_group['enabled'])
        data['measurement_points'] = (points, enabled)

        data['measurement_vectors'] = np.array(hdf_file['measurement_vectors'])
        if data['measurement_vectors'].shape[0] != data['measurement_points'][0].shape[0]:
            raise ValueError('The number of vectors are not equal to number of points')

        alignment = hdf_file.get('alignment')
        data['alignment'] = alignment if alignment is None else Matrix44(alignment)

        instrument = _read_instrument(hdf_file)

        if data['measurement_vectors'].shape[1] != 3 * len(instrument.detectors):
            raise ValueError(f'{filename} does not contain correct vector size for {data["instrument"]}.')

    return data, instrument


def _read_instrument(hdf_file):
    instrument_group = hdf_file['instrument']
    name = instrument_group.attrs['name']
    gauge_volume = list(instrument_group['gauge_volume'])
    script = Script(instrument_group.attrs['script_template'])

    positioning_stacks = {}
    for key, value in instrument_group['stacks'].attrs.items():
        positioning_stacks[key] = value.tolist()

    fixed_hardware = {}
    for key, group in instrument_group['fixed_hardware'].items():
        vertices = np.array(group['mesh_vertices'])
        indices = np.array(group['mesh_indices'])
        colour = Colour(*group['mesh_colour'])
        fixed_hardware[key] = Mesh(vertices, indices, colour=colour)

    positioners = {}
    for key, group in instrument_group['positioners'].items():
        links = []
        for link_name, sub_group in group['links'].items():
            if sub_group.get('mesh_vertices') is not None:
                mesh = Mesh(np.array(sub_group['mesh_vertices']),
                            np.array(sub_group['mesh_indices']),
                            colour=Colour(*sub_group['mesh_colour']))
            else:
                mesh = None
            links.append(Link(link_name, np.array(sub_group['axis']), np.array(sub_group['point']),
                              Link.Type(sub_group.attrs['type']), float(sub_group.attrs['lower_limit']),
                              float(sub_group.attrs['upper_limit']),  float(sub_group.attrs['default_offset']), mesh))

        if group.get('base_mesh_vertices') is not None:
            mesh = Mesh(np.array(group['base_mesh_vertices']), np.array(group['base_mesh_indices']),
                        colour=Colour(*group['base_mesh_colour']))
        else:
            mesh = None

        positioners[key] = SerialManipulator(group.attrs['name'], links, base=Matrix44(group['default_base']),
                                             tool=Matrix44(group['tool']),
                                             base_mesh=mesh, custom_order=list(group['order']))

    group = instrument_group['jaws']
    mesh = Mesh(np.array(group['mesh_vertices']), np.array(group['mesh_indices']),
                colour=Colour(*group['mesh_colour']))
    jaws = Jaws(group.attrs['name'], Vector3(group['initial_source']), Vector3(group['initial_direction']),
                list(group['aperture']), list(group['aperture_lower_limit']),
                list(group['aperture_upper_limit']), mesh, None)
    jaw_positioner_name = group.attrs.get('positioner_name')
    if jaw_positioner_name is not None:
        jaws.positioner = positioners[jaw_positioner_name]
        jaws.positioner.fkine(list(group['positioner_set_points']))
        limit_state = list(group['positioner_limit_state'])
        lock_state = list(group['positioner_lock_state'])
        for index, link in enumerate(jaws.positioner.links):
            link.ignore_limits = limit_state[index]
            link.locked = lock_state[index]

    detectors = {}
    for key, group in instrument_group['detectors'].items():
        collimators = {}
        for c_key, sub_group in group['collimators'].items():
            mesh = Mesh(np.array(sub_group['mesh_vertices']),
                        np.array(sub_group['mesh_indices']),
                        colour=Colour(*sub_group['mesh_colour']))
            collimators[c_key] = Collimator(sub_group.attrs['name'], list(sub_group['aperture']), mesh)

        detectors[key] = Detector(group.attrs['name'], Vector3(group['initial_beam']), collimators, None)
        detectors[key].current_collimator = group.attrs.get('current_collimator')
        detector_positioner_name = group.attrs.get('positioner_name')
        if detector_positioner_name is not None:
            detectors[key].positioner = positioners[detector_positioner_name]
            detectors[key].positioner.fkine(list(group['positioner_set_points']))
            limit_state = list(group['positioner_limit_state'])
            lock_state = list(group['positioner_lock_state'])
            for index, link in enumerate(detectors[key].positioner.links):
                link.ignore_limits = limit_state[index]
                link.locked = lock_state[index]

    instrument = Instrument(name, gauge_volume, detectors, jaws, positioners, positioning_stacks, script,
                            fixed_hardware)

    active_stack_group = instrument_group['stacks']['active']
    instrument.loadPositioningStack(active_stack_group.attrs['name'])
    instrument.positioning_stack.fkine(list(active_stack_group['set_points']))
    lock_state = list(active_stack_group['lock_state'])
    limit_state = list(active_stack_group['limit_state'])
    for index, link in enumerate(instrument.positioning_stack.links):
        link.ignore_limits = limit_state[index]
        link.locked = lock_state[index]

    base_group = active_stack_group.get('base')
    if base_group is not None:
        for positioner in instrument.positioning_stack.auxiliary:
            base = base_group.get(positioner.name)
            if base is None:
                continue
            instrument.positioning_stack.changeBaseMatrix(positioner, Matrix44(base))

    return instrument


def read_3d_model(filename):
    """Reads a 3D triangular mesh in Obj or STL formats

    :param filename: path of the stl file
    :type filename: str
    :return: The vertices, normals and index array of the mesh
    :rtype: Mesh
    :raises: ValueError
    """
    ext = os.path.splitext(filename)[1].replace('.', '').lower()
    if ext == 'stl':
        mesh = read_stl(filename)
    elif ext == 'obj':
        mesh = read_obj(filename)
    else:
        raise ValueError('"{}" 3D files are currently unsupported.'.format(ext))

    return mesh


def read_stl(filename):
    """Reads a 3D triangular mesh from an STL file. STL has a binary
    and ASCII format and this function attempts to read the file irrespective
    of its format.

    :param filename: path of the stl file
    :type filename: str
    :return: The vertices, normals and index array of the mesh
    :rtype: Mesh
    """
    try:
        return read_ascii_stl(filename)
    except (UnicodeDecodeError, ValueError):
        return read_binary_stl(filename)


def read_ascii_stl(filename):
    """Reads a 3D triangular mesh from an STL file (ASCII format).
    This function is much slower than the binary version due to
    the string split but will have to do for now.

    :param filename: path of the stl file
    :type filename: str
    :return: The vertices, normals and index array of the mesh
    :rtype: Mesh
    :raises: ValueError
    """
    #
    with open(filename, encoding='utf-8') as stl_file:
        offset = 21

        stl_file.readline()
        text = stl_file.read()
        text = text.lower().rsplit('endsolid', 1)[0]
        text = np.array(text.split())
        text_size = len(text)

        if text_size == 0 or text_size % offset != 0:
            raise ValueError('stl data has incorrect size')

        face_count = int(text_size / offset)
        text = text.reshape(-1, offset)
        data_pos = [2, 3, 4, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        normals = text[:, data_pos[0:3]].astype(np.float32)
        vertices = text[:, data_pos[3:]].astype(np.float32)

        vertices = vertices.reshape(-1, 3)
        indices = np.arange(face_count * 3)
        normals = np.repeat(normals, 3, axis=0)

        return Mesh(vertices, indices, normals, clean=True)


def read_binary_stl(filename):
    """Reads a 3D triangular mesh from an STL file (binary format).

    :param filename: path of the stl file
    :type filename: str
    :return: The vertices, normals and index array of the mesh
    :rtype: Mesh
    :raises: ValueError
    """
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
        raise ValueError('stl data has incorrect size')

    vertices = data['vertices'].reshape(-1, 3)
    indices = np.arange(face_count * 3)
    normals = np.repeat(data['normals'], 3, axis=0)

    return Mesh(vertices, indices, normals, clean=True)


def read_obj(filename):
    """Reads a 3D triangular mesh from an obj file.
    The obj format supports several geometric objects but
    this function reads the face index and vertices only and
    the vertex normals are computed by the Mesh object.

    :param filename: path of the obj file
    :type filename: str
    :return: The vertices, normals and index array of the mesh
    :rtype: Mesh
    """
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

    return Mesh(vertices, indices, clean=True)


def read_csv(filename):
    """Reads data from a space or comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: data from file
    :rtype: List[List[str]]
    """
    data = []
    regex = re.compile(r'(\s+|(\s*,\s*))')
    with open(filename) as csv_file:
        for line in csv_file:
            line = regex.sub(' ', line)
            row = line.split()
            if not row:
                continue
            data.append(row)

    return data


def read_points(filename):
    """Reads point data and enabled status from a space or comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: 3D points and enabled status
    :rtype: Tuple[numpy.ndarray, list[bool]]
    :raises: ValueError
    """
    points = []
    enabled = []
    data = read_csv(filename)
    for row in data:
        if len(row) == 3:
            points.append(row)
            enabled.append(True)
        elif len(row) == 4:
            *p, d = row
            d = False if d.lower() == 'false' else True
            points.append(p)
            enabled.append(d)
        else:
            raise ValueError('data has incorrect size')

    result = np.array(points, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite value present in point data')

    return result, enabled


def read_vectors(filename):
    """Reads measurement vectors from a space or comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: array of vectors
    :rtype:  numpy.ndarray
    :raises: ValueError
    """
    vectors = []
    data = read_csv(filename)
    expected_size = len(data[0])
    if expected_size % 3 != 0:
        raise ValueError('Column size of vector data must be a multiple of 3.')

    for row in data:
        if len(row) == expected_size:
            vectors.append(row)
        else:
            raise ValueError('Inconsistent column size of vector data.')

    result = np.array(vectors, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite value present in vector data')

    return result


def read_trans_matrix(filename):
    """Reads transformation matrix from a space or comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: transformation matrix
    :rtype: Matrix44
    :raises: ValueError
    """
    matrix = []
    data = read_csv(filename)
    if len(data) != 4:
        raise ValueError('data has incorrect size')

    for row in data:
        if len(row) == 4:
            matrix.append(row)
        else:
            raise ValueError('data has incorrect size')

    result = Matrix44(matrix, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite value present in matrix')

    return result


def read_fpos(filename):
    """Reads index, points, and positioner pose from a space or comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: index, points, and positioner pose
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises: ValueError
    """
    index = []
    points = []
    pose = []
    data = read_csv(filename)
    expected_size = len(data[0])
    if expected_size < 4:
        raise ValueError('data has incorrect size')

    for row in data:
        if len(row) != expected_size:
            raise ValueError('Inconsistent column size of fpos data.')
        index.append(row[0])
        points.append(row[1:4])
        pose.append(row[4:])

    result = np.array(index, int) - 1, np.array(points, np.float32), np.array(pose, np.float32)
    if not (np.isfinite(result[1]).all() and np.isfinite(result[2]).all()):
        raise ValueError('Non-finite value present in fpos data')

    return result
