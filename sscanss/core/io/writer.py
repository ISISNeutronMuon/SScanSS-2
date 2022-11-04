"""
A collection of functions for writing data
"""
import csv
import datetime as dt
import os
import h5py
import numpy as np
import tifffile as tiff
from ..geometry.mesh import Mesh
from ..geometry.volume import Volume
from ..util.worker import ProgressReport
from ...config import __version__, settings


def write_project_hdf(data, filename):
    """Writes the project data dictionary to a hdf file. The sample is now saved as 'main_sample' so as
    not to break back-compatibility with older version, also a volume is replaced with a mesh equivalent in
    older versions

    :param data: A dictionary containing the project data
    :type data: dict
    :param filename: path of the hdf file
    :type filename: str
    """
    with h5py.File(filename, 'w') as hdf_file:
        hdf_file.attrs['name'] = data['name']
        hdf_file.attrs['version'] = str(__version__)
        hdf_file.attrs['instrument_version'] = data['instrument_version']

        date_created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdf_file.attrs['date_created'] = date_created

        if settings.local:
            setting_group = hdf_file.create_group('settings')
            for key, value in settings.local.items():
                setting_group.attrs[key] = value

        sample = data['sample']
        sample_group = hdf_file.create_group('main_sample')
        back_compact_group = hdf_file.create_group('sample')
        if isinstance(sample, Mesh):
            sample_group['vertices'] = sample.vertices
            sample_group['indices'] = sample.indices
            back_compact_group = back_compact_group.create_group('unnamed')
            back_compact_group['vertices'] = h5py.SoftLink(sample_group['vertices'].name)
            back_compact_group['indices'] = h5py.SoftLink(sample_group['indices'].name)
        elif isinstance(sample, Volume):
            sample_group['image'] = sample.data
            sample_group['voxel'] = sample.voxel_size
            sample_group['transform'] = sample.transform_matrix
            curve_group = sample_group.create_group('curves/alpha')
            curve_group.attrs['type'] = sample.curve.type.value
            curve_group['inputs'] = sample.curve.inputs
            curve_group['outputs'] = sample.curve.outputs
            curve_group['bounds'] = sample.curve.bounds

            temp = sample.asMesh()
            back_compact_group = back_compact_group.create_group('unnamed')
            back_compact_group['vertices'] = temp.vertices
            back_compact_group['indices'] = temp.indices

        fiducials = data['fiducials']
        fiducial_group = hdf_file.create_group('fiducials')
        fiducial_group['points'] = fiducials.points
        fiducial_group['enabled'] = fiducials.enabled

        measurements = data['measurement_points']
        measurement_group = hdf_file.create_group('measurement_points')
        measurement_group['points'] = measurements.points
        measurement_group['enabled'] = measurements.enabled

        vectors = data['measurement_vectors']
        hdf_file.create_dataset('measurement_vectors', data=vectors)

        alignment = data['alignment']
        if alignment is not None:
            hdf_file.create_dataset('alignment', data=alignment)

        instrument = data['instrument']
        hdf_file.attrs['instrument_name'] = instrument.name

        _write_instrument(hdf_file, instrument)


def _write_instrument(hdf_file, instrument):
    instrument_group = hdf_file.create_group('instrument')
    instrument_group.attrs['name'] = instrument.name
    instrument_group['gauge_volume'] = instrument.gauge_volume
    instrument_group.attrs['script_template'] = instrument.script.template

    positioners_group = instrument_group.create_group('positioners')
    for key, positioner in instrument.positioners.items():
        group = positioners_group.create_group(key)
        group.attrs['name'] = positioner.name
        group['default_base'] = positioner.default_base
        group['tool'] = positioner.tool
        if positioner.base_mesh is not None:
            group['base_mesh_vertices'] = positioner.base_mesh.vertices
            group['base_mesh_indices'] = positioner.base_mesh.indices
            group['base_mesh_colour'] = positioner.base_mesh.colour.rgbaf
        group['order'] = positioner.order
        group = group.create_group('links', track_order=True)
        for link in positioner.links:
            sub_group = group.create_group(link.name)
            sub_group['axis'] = link.joint_axis
            sub_group['point'] = link.home
            sub_group.attrs['type'] = link.type.value
            sub_group.attrs['lower_limit'] = link.lower_limit
            sub_group.attrs['upper_limit'] = link.upper_limit
            sub_group.attrs['default_offset'] = link.default_offset
            if link.mesh is not None:
                sub_group['mesh_vertices'] = link.mesh.vertices
                sub_group['mesh_indices'] = link.mesh.indices
                sub_group['mesh_colour'] = link.mesh.colour.rgbaf

    stacks_group = instrument_group.create_group('stacks')
    for key, value in instrument.positioning_stacks.items():
        stacks_group.attrs[key] = value
    active_stack_group = stacks_group.create_group('active')
    active_stack_group.attrs['name'] = instrument.positioning_stack.name
    active_stack_group['set_points'] = instrument.positioning_stack.set_points
    active_stack_group['lock_state'] = [link.locked for link in instrument.positioning_stack.links]
    active_stack_group['limit_state'] = [link.ignore_limits for link in instrument.positioning_stack.links]
    for _, positioner in enumerate(instrument.positioning_stack.auxiliary):
        if positioner.base is positioner.default_base:
            continue

        base_group = active_stack_group.get('base')
        if base_group is None:
            base_group = active_stack_group.create_group('base')

        base_group[positioner.name] = positioner.base

    group = instrument_group.create_group('jaws')
    group.attrs['name'] = instrument.jaws.name
    if instrument.jaws.positioner is not None:
        group.attrs['positioner_name'] = instrument.jaws.positioner.name
        group['positioner_set_points'] = instrument.jaws.positioner.set_points
        group['positioner_lock_state'] = [link.locked for link in instrument.jaws.positioner.links]
        group['positioner_limit_state'] = [link.ignore_limits for link in instrument.jaws.positioner.links]

    group['aperture'] = instrument.jaws.aperture
    group['initial_source'] = instrument.jaws.initial_source
    group['initial_direction'] = instrument.jaws.initial_direction
    group['aperture_lower_limit'] = instrument.jaws.aperture_lower_limit
    group['aperture_upper_limit'] = instrument.jaws.aperture_upper_limit
    group['mesh_vertices'] = instrument.jaws.mesh.vertices
    group['mesh_indices'] = instrument.jaws.mesh.indices
    group['mesh_colour'] = instrument.jaws.mesh.colour.rgbaf

    detectors_group = instrument_group.create_group('detectors')
    for key, detector in instrument.detectors.items():
        group = detectors_group.create_group(key)
        group.attrs['name'] = detector.name
        if detector.current_collimator is not None:
            group.attrs['current_collimator'] = detector.current_collimator.name
        if detector.positioner is not None:
            group.attrs['positioner_name'] = detector.positioner.name
            group['positioner_set_points'] = detector.positioner.set_points
            group['positioner_lock_state'] = [link.locked for link in detector.positioner.links]
            group['positioner_limit_state'] = [link.ignore_limits for link in detector.positioner.links]
        group['initial_beam'] = detector.initial_beam
        group = group.create_group('collimators')
        for c_key, collimator in detector.collimators.items():
            sub_group = group.create_group(c_key)
            sub_group.attrs['name'] = collimator.name
            sub_group['aperture'] = collimator.aperture
            sub_group['mesh_vertices'] = collimator.mesh.vertices
            sub_group['mesh_indices'] = collimator.mesh.indices
            sub_group['mesh_colour'] = collimator.mesh.colour.rgbaf

    fixed_hardware_group = instrument_group.create_group('fixed_hardware')
    for key, mesh in instrument.fixed_hardware.items():
        group = fixed_hardware_group.create_group(key)
        group['mesh_vertices'] = mesh.vertices
        group['mesh_indices'] = mesh.indices
        group['mesh_colour'] = mesh.colour.rgbaf


def write_binary_stl(filename, mesh):
    """Writes a 3D mesh to a binary STL file. The binary STL format only
    supports face normals while the Mesh object stores vertex normals
    therefore the first vertex normal for each face is written.

    :param filename: path of the stl file
    :type filename: str
    :param mesh: The vertices, normals and index array of the mesh
    :type mesh: Mesh
    """
    record_dtype = np.dtype([
        ('normals', np.float32, (3, )),
        ('vertices', np.float32, (3, 3)),
        ('attr', '<i2', (1, )),
    ])
    face_count = mesh.indices.size // 3
    data = np.recarray(face_count, dtype=record_dtype)

    data.normals = mesh.normals[mesh.indices, :][::3]
    data.attr = np.zeros((face_count, 1), dtype=np.uint32)
    data.vertices = mesh.vertices[mesh.indices, :].reshape((-1, 3, 3))

    with open(filename, 'wb') as stl_file:
        stl_file.seek(80)
        np.array(face_count, dtype=np.int32).tofile(stl_file)
        data.tofile(stl_file)


def write_points(filename, data):
    """Writes point data and enabled status to tab delimited file.

    :param filename: path of the file
    :type filename: str
    :param data: 3D points and enabled status
    :type data: numpy.recarray
    """
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        write_enabled = data.enabled.all()
        for i in range(data.size):
            p0, p1, p2 = data[i].points
            if write_enabled:
                writer.writerow([f'{p0:.7f}', f'{p1:.7f}', f'{p2:.7f}'])
            else:
                writer.writerow([f'{p0:.7f}', f'{p1:.7f}', f'{p2:.7f}', data[i].enabled])


def write_fpos(filename, indices, points, poses=None):
    """Writes index, points, and positioner pose to a tab delimited fpos file.

    :param filename: path of the file
    :type filename: str
    :param indices: point indices
    :type indices: numpy.ndarray
    :param points: fiducial point
    :type points: numpy.ndarray
    :param poses: positioner pose
    :type poses: numpy.ndarray
    """
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')

        for i, point in enumerate(points):
            single_row = [f'{indices[i] + 1}', f'{point[0]:.3f}', f'{point[1]:.3f}', f'{point[2]:.3f}']
            if poses is not None:
                single_row.extend(f'{positioner:.3f}' for positioner in poses[i])
            writer.writerow(single_row)


def write_volume_as_images(folder_path, volume):
    """Writes the image data to given folder. Any transformation applied to the volume
    is ignored

    :param folder_path: folder path
    :type folder_path: str
    :param volume: volume
    :type volume: Volume
    """
    report = ProgressReport()
    report.beginStep('Exporting Volume as TIFF Images')
    image_count = volume.data.shape[2]
    for i in range(image_count):
        filename = os.path.join(folder_path, f'{i + 1:0>{len(str(image_count))}}.tiff')
        image = volume.data[:, :, i].transpose()
        tiff.imsave(filename, image)
        report.updateProgress((i + 1) / image_count)
    report.completeStep()
