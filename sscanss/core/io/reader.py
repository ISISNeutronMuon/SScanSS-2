"""
A collection of functions for reading data
"""
from contextlib import suppress
import os
import re
import warnings
import h5py
import numpy as np
import psutil
from scipy.ndimage import zoom
import tifffile as tiff
from ..geometry.mesh import Mesh
from ..geometry.volume import Volume, Curve
from ..geometry.colour import Colour
from ..instrument.instrument import Instrument, Collimator, Detector, Jaws, Script
from ..instrument.robotics import Link, SerialManipulator
from ..math.constants import VECTOR_EPS
from ..math.matrix import Matrix44
from ..math.vector import Vector3
from ..util.worker import ProgressReport

SUPPORTED_IMAGE_TYPE = ('uint8', 'uint16', 'float32')


def read_project_hdf(filename):
    """Reads the project data dictionary from a hdf file. This reader will work for files
    from previous version with OrderedDict sample and files with single main sample.

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

        data['sample'] = None
        sample_group = hdf_file.get('main_sample')
        if sample_group is None:
            sample_group = hdf_file['sample']
            for _, item in sample_group.items():
                vertices = np.array(item['vertices'])
                indices = np.array(item['indices'])
                mesh = Mesh(vertices, indices)
                if data['sample'] is None:
                    data['sample'] = mesh
                else:
                    data['sample'].append(mesh)
        else:
            if sample_group.get('vertices'):  # Mesh
                vertices = np.array(sample_group['vertices'])
                indices = np.array(sample_group['indices'])
                data['sample'] = Mesh(vertices, indices)

            elif sample_group.get('image'):  # Volume
                image = np.array(sample_group['image'], order='F')
                voxel = np.array(sample_group['voxel'])
                transform = np.array(sample_group['transform'])

                curve_group = sample_group['curves/alpha']
                curve = Curve(np.array(curve_group['inputs']), np.array(curve_group['outputs']),
                              np.array(curve_group['bounds']), Curve.Type(curve_group.attrs['type']))
                volume = Volume(image, voxel, np.zeros(3))
                volume.curve = curve
                volume.transform(transform)
                data['sample'] = volume

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
            raise ValueError(f'The file does not contain correct vector size for {data["instrument"]}.')

        if not validate_vector_length(data['measurement_vectors']):
            raise ValueError('Measurement vectors must be zero vectors or have a magnitude of 1 '
                             '(accurate to 7 decimal digits), the file contains vectors that are neither.')

    return data, instrument


def _read_instrument(hdf_file):
    instrument_group = hdf_file['instrument']
    name = instrument_group.attrs['name']
    gauge_volume = instrument_group['gauge_volume'][:].tolist()
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
            links.append(
                Link(link_name, np.array(sub_group['axis']), np.array(sub_group['point']),
                     Link.Type(sub_group.attrs['type']), float(sub_group.attrs['lower_limit']),
                     float(sub_group.attrs['upper_limit']), float(sub_group.attrs['default_offset']), mesh))

        if group.get('base_mesh_vertices') is not None:
            mesh = Mesh(np.array(group['base_mesh_vertices']),
                        np.array(group['base_mesh_indices']),
                        colour=Colour(*group['base_mesh_colour']))
        else:
            mesh = None

        positioners[key] = SerialManipulator(group.attrs['name'],
                                             links,
                                             base=Matrix44(group['default_base']),
                                             tool=Matrix44(group['tool']),
                                             base_mesh=mesh,
                                             custom_order=group['order'][:].tolist())

    group = instrument_group['jaws']
    mesh = Mesh(np.array(group['mesh_vertices']), np.array(group['mesh_indices']), colour=Colour(*group['mesh_colour']))
    jaws = Jaws(group.attrs['name'], Vector3(group['initial_source']), Vector3(group['initial_direction']),
                group['aperture'][:].tolist(), group['aperture_lower_limit'][:].tolist(),
                group['aperture_upper_limit'][:].tolist(), mesh, None)
    jaw_positioner_name = group.attrs.get('positioner_name')
    if jaw_positioner_name is not None:
        jaws.positioner = positioners[jaw_positioner_name]
        jaws.positioner.fkine(group['positioner_set_points'][:].tolist())
        limit_state = group['positioner_limit_state']
        lock_state = group['positioner_lock_state']
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
            collimators[c_key] = Collimator(sub_group.attrs['name'], sub_group['aperture'][:].tolist(), mesh)

        detectors[key] = Detector(group.attrs['name'], Vector3(group['initial_beam']), collimators, None)
        detectors[key].current_collimator = group.attrs.get('current_collimator')
        detector_positioner_name = group.attrs.get('positioner_name')
        if detector_positioner_name is not None:
            detectors[key].positioner = positioners[detector_positioner_name]
            detectors[key].positioner.fkine(group['positioner_set_points'][:].tolist())
            limit_state = group['positioner_limit_state'][:].tolist()
            lock_state = group['positioner_lock_state'][:].tolist()
            for index, link in enumerate(detectors[key].positioner.links):
                link.ignore_limits = limit_state[index]
                link.locked = lock_state[index]

    instrument = Instrument(name, gauge_volume, detectors, jaws, positioners, positioning_stacks, script,
                            fixed_hardware)

    active_stack_group = instrument_group['stacks']['active']
    instrument.loadPositioningStack(active_stack_group.attrs['name'])
    instrument.positioning_stack.fkine(active_stack_group['set_points'][:].tolist())
    lock_state = active_stack_group['lock_state'][:].tolist()
    limit_state = active_stack_group['limit_state'][:].tolist()
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
        raise ValueError(f'"{ext}" 3D files are currently unsupported.')

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
        indices = np.arange(face_count * 3).astype(np.uint32)
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
            ('normals', np.float32, (3, )),
            ('vertices', np.float32, (3, 3)),
            ('attr', '<i2', (1, )),
        ])
        data = np.fromfile(stl_file, dtype=record_dtype)

    if face_count != data.size:
        raise ValueError('stl data has incorrect size')

    vertices = data['vertices'].reshape(-1, 3)
    indices = np.arange(face_count * 3).astype(np.uint32)
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
    indices = np.arange(face_index.size).astype(np.uint32)

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
    with open(filename, encoding='utf-8-sig') as csv_file:
        for line in csv_file:
            line = regex.sub(' ', line)
            row = line.split()
            if not row:
                continue
            data.append(row)

    if not data:
        raise ValueError('The file is empty')

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
            *point, enable = row
            points.append(point)
            enabled.append(enable.lower() != 'false')
        else:
            raise ValueError('Data has incorrect size')

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
        raise ValueError('Column size of vector data must be a multiple of 3')

    for row in data:
        if len(row) == expected_size:
            vectors.append(row)
        else:
            raise ValueError('Inconsistent column size of vector data')

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
        raise ValueError('Data has incorrect size')

    for row in data:
        if len(row) == 4:
            matrix.append(row)
        else:
            raise ValueError('Data has incorrect size')

    result = Matrix44(matrix, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite value present in matrix data')

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
        raise ValueError('Data has incorrect size')

    for row in data:
        if len(row) != expected_size:
            raise ValueError('Inconsistent column size of fpos data')
        index.append(row[0])
        points.append(row[1:4])
        pose.append(row[4:])

    result = np.array(index, int) - 1, np.array(points, np.float32), np.array(pose, np.float32)
    if not (np.isfinite(result[1]).all() and np.isfinite(result[2]).all()):
        raise ValueError('Non-finite value present in fpos data')

    return result


def read_angles(filename):
    """Reads index, points, and positioner pose from a space or comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: index, points, and positioner pose
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises: ValueError
    """
    data = read_csv(filename)
    if len(data[0]) != 1:
        raise ValueError('Angle order is missing')

    order = data[0][0].lower()
    angles = data[1:]

    for row in angles:
        if len(row) != 3:
            raise ValueError('Incorrect column size of angle data (expected 3 columns)')

    result = np.array(angles, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite value present in angle data')

    return result, order


def validate_vector_length(vectors):
    """Validates that the measurement vectors have a magnitude of zero or one

    :param vectors: measurement vectors
    :type vectors: numpy.ndarray
    :return: indicates that all the vectors have a magnitude of 0 or 1
    :rtype: bool
    """
    detector_count = vectors.shape[1] // 3
    for detector in range(detector_count):
        detector_index = slice(detector * 3, detector * 3 + 3)
        norm = np.linalg.norm(vectors[:, detector_index], axis=1)

        if np.any((np.abs(norm - 1) > VECTOR_EPS) & (norm > VECTOR_EPS)):
            return False
    return True


def read_kinematic_calibration_file(filename):
    """Reads index, measured points, joint offsets, joint types and joint homes from a space or
    comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: Measured points, joint types, joint offsets, and joint homes
    :rtype: Tuple[List[numpy.ndarray], List[Link.Type], List[numpy.ndarray], List[numpy.ndarray]]
    """
    points = []
    offsets = []
    types = []
    homes = []

    data = read_csv(filename)

    size = len(data)
    inputs = {
        'ids': np.empty(size, 'i4'),
        'points': np.empty((size, 3), 'f4'),
        'types': np.empty(size, 'U9'),
        'offsets': np.empty(size, 'f4'),
        'homes': np.empty(size, 'f4')
    }
    for index, row in enumerate(data):
        if len(row) != 7:
            raise ValueError('Incorrect column size of calibration data (expected 7 columns)')

        inputs['ids'][index] = row[0]
        inputs['points'][index, :] = row[1:4]
        inputs['offsets'][index] = row[4]
        inputs['types'][index] = row[5].lower()
        inputs['homes'][index] = row[6]

    unique_ids = np.unique(inputs['ids'])
    expected_types = [Link.Type.Prismatic.value, Link.Type.Revolute.value]

    for joint_id in unique_ids:
        temp = np.where(inputs['ids'] == joint_id)[0]
        if temp.shape[0] < 3:
            raise ValueError('Each Joint must have at least 3 measured points.')

        joint_type = inputs['types'][temp]
        if np.any(joint_type != joint_type[0]):
            raise ValueError(f'Joint {joint_id} has inconsistent joint types.')

        if np.any((joint_type != expected_types[0]) & (joint_type != expected_types[1])):
            raise ValueError(f'The calibration data for Joint {joint_id} contains unsupported joint types '
                             f'(The supported joint types are {expected_types}).')

        if np.any(inputs['homes'][temp] != inputs['homes'][temp][0]):
            raise ValueError(f'Joint {joint_id} has inconsistent home positions.')

        points.append(inputs['points'][temp])
        offsets.append(inputs['offsets'][temp])
        types.append(Link.Type(joint_type[0]))
        homes.append(inputs['homes'][temp][0])

    return points, types, offsets, homes


def read_robot_world_calibration_file(filename):
    """Reads pose index, fiducial index, points, and positioner pose from a space or comma delimited file.

    :param filename: path of the file
    :type filename: str
    :return: pose index, fiducial index, points, and positioner pose
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises: ValueError
    """
    pose_index = []
    fiducial_index = []
    points = []
    pose = []

    data = read_csv(filename)
    expected_size = len(data[0])
    if expected_size < 6:
        raise ValueError('Data has incorrect size')

    for row in data:
        if len(row) != expected_size:
            raise ValueError('Inconsistent column size of calibration data')
        pose_index.append(row[0])
        fiducial_index.append(row[1])
        points.append(row[2:5])
        pose.append(row[5:])

    result = (np.array(pose_index, int) - 1, np.array(fiducial_index, int) - 1, np.array(points, np.float32),
              np.array(pose, np.float32))
    if not (np.isfinite(result[2]).all() and np.isfinite(result[3]).all()):
        raise ValueError('Non-finite value present in calibration data')

    return result


class BadDataWarning(UserWarning):
    """Creates warning for when volume contains bad data i.e. Nans or Inf"""


def read_tomoproc_hdf(filename):
    """Reads the data from a nexus standard hdf file which contains an entry conforming to the NXTomoproc standard
    https://manual.nexusformat.org/classes/applications/NXtomoproc.html

    :param filename: path of the hdf file
    :type filename: str
    :return: 3D array of intensities, size of the volume's voxels and coordinates of the volume centre
    :rtype: Tuple[np.ndarray[uint8], List[float, float, float], List[float, float, float]]
    :raises: AttributeError, ValueError, MemoryError
    """
    report = ProgressReport()
    report.beginStep('Loading Volume from Nexus File')

    with h5py.File(filename, 'r') as hdf_file:
        main_entry = None
        data_folder = None

        for _, item in hdf_file.items():
            if b'NX_class' in item.attrs.keys():
                main_entry = item
                definition = hdf_file.get(f'{main_entry.name}/definition')
                with suppress(AttributeError):
                    if definition is not None and definition[()].decode('utf-8').lower() == 'nxtomoproc':
                        data_folder = definition.parent.name
                        # Check the definition to find the correct entry, AttributeError suppressed due to ISIS files
                        # not conforming to Nexus standard (returns array of string not string(NX_char))
                break

        else:
            raise AttributeError('There is no NX_class in this file')

        if not data_folder:
            hdf_interior = hdf_file[main_entry.name]
            data_folder = main_entry.parent.name

            for _, item in hdf_interior.items():
                definition = hdf_file.get(f'{item.name}/definition')
                if definition is None:
                    continue
                with suppress(AttributeError):
                    if definition[()].decode('utf-8').lower() == 'nxtomoproc':
                        data_folder = definition.parent.name
                        break

        data = hdf_file[f'{data_folder}/data/data']

        total_iterations = data.shape[0] * 2 if data.dtype == np.float32 else data.shape[0]
        total_required_size = 2 * data.shape[0] * data.shape[1] * data.shape[2]
        if total_required_size >= psutil.virtual_memory().available:
            raise MemoryError('The volume data is larger than the available memory on your machine')

        x = np.array(hdf_file[f'{data_folder}/data/x'])
        y = np.array(hdf_file[f'{data_folder}/data/y'])
        z = np.array(hdf_file[f'{data_folder}/data/z'])

        if not (data.shape == (len(x), len(y), len(z))):
            raise ValueError('The data arrays in the file are not the same size')

        x_spacing = (x[-1] - x[0]) / (len(x) - 1)
        y_spacing = (y[-1] - y[0]) / (len(y) - 1)
        z_spacing = (z[-1] - z[0]) / (len(z) - 1)
        voxel_size = [x_spacing, y_spacing, z_spacing]

        x_origin = x[0] + (x[-1] - x[0]) / 2
        y_origin = y[0] + (y[-1] - y[0]) / 2
        z_origin = z[0] + (z[-1] - z[0]) / 2
        origin = [x_origin, y_origin, z_origin]

        rescale_values = []
        any_non_finite = False
        volume_data = np.zeros(data.shape, np.uint8, order='F')

        # Slicing in the 3rd dimension is incredibly slow, so we slice 1st dimension instead
        for i in range(data.shape[0]):
            if data.dtype == np.uint8:
                volume_data[i] = data[i]
            elif data.dtype == np.uint16:
                volume_data[i] = data[i] / 65535 * 255
            elif data.dtype == np.float32:
                image = np.array(data[i])
                non_finite_values = ~np.isfinite(image)
                if non_finite_values.any():
                    any_non_finite = True

                # Scale data between 0 and 254 the use 255 for non-finite values
                result = np.full(image.shape, 255, dtype=np.uint8)
                if non_finite_values.all():
                    rescale_values.append([np.nan, np.nan])
                else:
                    valid_values = image[~non_finite_values]
                    min_value, max_value = valid_values.min(), valid_values.max()
                    scale_factor = 1 if (max_value - min_value) == 0 else 254 / (max_value - min_value)
                    result[~non_finite_values] = (valid_values - min_value) * scale_factor
                    volume_data[i] = result
                    rescale_values.append([min_value, max_value])
            else:
                raise TypeError(f'The files have an unsupported data type: {data.dtype}. The '
                                f'supported data types are {SUPPORTED_IMAGE_TYPE}')
            report.updateProgress((i + 1) / total_iterations)

        if rescale_values:
            # Hack for uniformly rescaling float images one slice at a time to reduce memory
            rescale_values = np.array(rescale_values).transpose()
            if np.invert(np.isfinite(rescale_values)).all():
                raise ValueError(f'Volume slice is non-finite i.e. contains only Nans or Inf. ({filename})')
            new_min, new_max = np.nanmin(rescale_values[0]), np.nanmax(rescale_values[1])
            scale_factor = 1 if (new_max - new_min) == 0 else 255 / (new_max - new_min)
            for i in range(volume_data.shape[0]):
                non_finite_values = volume_data[i] == 255
                old_min, old_max = rescale_values[0, i], rescale_values[1, i]
                if np.isfinite(old_max) and np.isfinite(old_min):
                    value = volume_data[i] * (old_max - old_min) / 254
                    volume_data[i] = (value + old_min - new_min) * scale_factor
                volume_data[i][non_finite_values] = 0
                report.updateProgress((i + volume_data.shape[0] + 1) / total_iterations)

        if any_non_finite:
            warnings.warn('Volume data contains non-finite values i.e. Nans or Inf.', BadDataWarning)

    report.completeStep()
    return volume_data, voxel_size, origin


def file_walker(filepath, extension=(".tiff", ".tif")):
    """Returns a list of filenames, which satisfy the extension, in the filepath folder

    :param filepath: path of the folder containing TIFF tiles
    :type filepath: str
    :param extension: Tuple of extensions which are searched for
    :type extension: Union[str, Tuple[str]]
    :return: list of filenames and paths which have appropriate file extension
    :rtype: List[str]
    """
    list_of_files = []
    for file in os.listdir(filepath):
        if file.lower().endswith(extension):
            filename = os.path.join(filepath, file)
            list_of_files.append(filename)

    return list_of_files


def filename_sorting_key(string, regex=re.compile('(\d+)')):
    """Returns a key for sorting filenames containing numbers in a natural way.

    :param string: input string
    :type string: str
    :param regex: compiled regular expression object
    :type regex: Pattern
    :return: key for sorting files
    :rtype: List[Union[str,int]]
    """
    return [int(text) if text.isdigit() else text.lower() for text in regex.split(string)]


def create_volume_from_tiffs(file_path):
    """Creates from a volume from tiff files and creates volume

    :param file_path: path of the folder containing TIFF files
    :type file_path: str
    :return: array of images
    :rtype: np.array
    :raises: ValueError, MemoryError
    """
    report = ProgressReport()
    report.beginStep('Loading Volume from Tiff Images')

    tiff_names = file_walker(file_path)

    if not tiff_names:
        raise ValueError('There are no valid ".tiff" files in this folder')

    first_image = tiff.imread(tiff_names[0])
    image_type = first_image.dtype
    y_size, x_size = np.shape(first_image)

    total_iterations = len(tiff_names) * 2 if image_type == np.float32 else len(tiff_names)

    total_required_size = 2 * x_size * y_size * len(tiff_names)
    if total_required_size >= psutil.virtual_memory().available:
        raise MemoryError('The volume data is larger than the available memory on your machine')

    stack_of_tiffs = np.zeros((x_size, y_size, len(tiff_names)), np.uint8, order='F')

    rescale_values = []
    any_non_finite = False
    for i, filename in enumerate(sorted(tiff_names, key=filename_sorting_key)):
        loaded_tiff = tiff.imread(filename).transpose()
        if image_type == np.uint8:
            stack_of_tiffs[:, :, i] = loaded_tiff
        elif image_type == np.uint16:
            stack_of_tiffs[:, :, i] = loaded_tiff * 255.0 / 65535.0
        elif image_type == np.float32:
            non_finite_values = ~np.isfinite(loaded_tiff)
            if non_finite_values.all():
                raise ValueError(f'Volume slice is non-finite i.e. contains only Nans or Inf. ({filename})')
            elif non_finite_values.any():
                any_non_finite = True

            # Scale data between 0 and 254 the use 255 for non-finite values
            result = np.full((x_size, y_size), 255, dtype=np.uint8)
            valid_values = loaded_tiff[~non_finite_values]
            min_value, max_value = valid_values.min(), valid_values.max()
            scale_factor = 1 if (max_value - min_value) == 0 else 254 / (max_value - min_value)
            result[~non_finite_values] = (valid_values - min_value) * scale_factor
            stack_of_tiffs[:, :, i] = result
            rescale_values.append([min_value, max_value])
        else:
            raise TypeError(f'The files have an unsupported data type: {image_type}. The '
                            f'supported data types are {SUPPORTED_IMAGE_TYPE}')

        report.updateProgress((i + 1) / total_iterations)

    if any_non_finite:
        warnings.warn('Volume data contains non-finite values i.e. Nans or Inf.', BadDataWarning)

    if rescale_values:
        rescale_values = np.array(rescale_values).transpose()
        new_min, new_max = rescale_values[0].min(), rescale_values[1].max()
        scale_factor = 1 if (new_max - new_min) == 0 else 255 / (new_max - new_min)
        for i in range(len(tiff_names)):
            non_finite_values = stack_of_tiffs[:, :, i] == 255
            old_min, old_max = rescale_values[0, i], rescale_values[1, i]
            value = stack_of_tiffs[:, :, i] * (old_max - old_min) / 254
            stack_of_tiffs[:, :, i] = (value + old_min - new_min) * scale_factor
            stack_of_tiffs[:, :, i][non_finite_values] = 0
            report.updateProgress((i + len(tiff_names) + 1) / total_iterations)

    report.completeStep()
    return stack_of_tiffs


def load_volume(file_path, voxel_size=None, centre=None, max_bytes=2e9, max_dim=1024):
    """Loads volume from TIFFs or a nexus file. The data is binned if larger than the max_bytes so that its
    max dimension is max_dim

    :param file_path: file path of volume (folder for tiffs or file path for nexus)
    :type file_path: str
    :param voxel_size: size of the volume's voxels in the x, y, and z axes
    :type voxel_size: Optional(List[float, float, float])
    :param centre: coordinates of the volume centre in the x, y, and z axes
    :type centre: Optional(List[float, float, float])
    :param max_bytes: maximum number of bytes before binning
    :type max_bytes: int
    :param max_dim: maximum dimension of binned data
    :type max_dim: int
    :return: volume object
    :rtype: Volume
    """
    report = ProgressReport()
    report.start('Loading Volume from File', 3)
    if voxel_size is None:
        images, voxel_size, centre = read_tomoproc_hdf(file_path)
    else:
        images = create_volume_from_tiffs(file_path)

    report.nextStep()

    if images.nbytes > max_bytes:
        image_count = images.shape[2]
        scale = max_dim / np.max(images.shape)
        new_shape = tuple([int(round(dim * scale)) for dim in images.shape])
        binned_data = np.zeros(new_shape, dtype=np.uint8, order='F')
        tmp_target = np.zeros((*new_shape[:2], image_count), dtype=np.uint8, order='F')

        total_iterations = tmp_target.shape[1] + image_count
        for i in range(image_count):
            tmp_target[:, :, i] = zoom(images[:, :, i], scale, order=0)
            report.updateProgress((i + 1) / total_iterations)

        for i in range(tmp_target.shape[1]):
            binned_data[:, i, :] = zoom(tmp_target[:, i, :], (1, scale), order=0)
            report.updateProgress((i + image_count + 1) / total_iterations)
    else:
        binned_data = images

    report.nextStep()
    count = binned_data.shape[2]
    hist_per_image = np.zeros((count, 256), dtype=np.int32)
    for i in range(count):
        hist_per_image[i, :] = np.bincount(binned_data[:, :, i].ravel(), minlength=256)
        report.updateProgress((i + 1) / count)
    histogram = (np.sum(hist_per_image, axis=0), np.linspace(0, 255, 257))

    volume = Volume(images,
                    np.array(voxel_size, np.float32),
                    np.array(centre, np.float32),
                    histogram,
                    binned_data=binned_data)
    report.complete()
    return volume
