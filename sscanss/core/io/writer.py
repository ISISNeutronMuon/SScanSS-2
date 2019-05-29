import csv
import numpy as np


def write_project_hdf(data, filename):
    """Writes the project data dictionary to a hdf file

    :param data: A dictionary containing the project data
    :type data: dict
    :param filename: path of the hdf file
    :type filename: str
    """
    import h5py
    import datetime as dt
    from sscanss.config import __version__

    with h5py.File(filename, 'w') as hdf_file:
        hdf_file.attrs['name'] = data['name']
        hdf_file.attrs['version'] = __version__

        date_created = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdf_file.attrs['date_created'] = date_created

        hdf_file.attrs['date_created'] = date_created

        samples = data['sample']
        sample_group = hdf_file.create_group('sample')
        for key, sample in samples.items():
            sample_group.create_group(key)
            sample_group[key]['vertices'] = sample.vertices
            sample_group[key]['normals'] = sample.normals
            sample_group[key]['indices'] = sample.indices

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


def write_binary_stl(filename, mesh):
    """Writes a 3D mesh to a binary STL file. The binary STL format only
    supports face normals while the Mesh object stores vertex normals
    therefore the first vertex normal for each face is written.

    :param filename: path of the stl file
    :type filename: str
    :param mesh: The vertices, normals and index array of the mesh
    :type mesh: sscanss.core.mesh.Mesh
    """
    record_dtype = np.dtype([
        ('normals', np.float32, (3,)),
        ('vertices', np.float32, (3, 3)),
        ('attr', '<i2', (1,)),
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
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for i in range(data.size):
            p0, p1, p2 = data[i].points
            writer.writerow([f'{p0:.7f}', f'{p1:.7f}', f'{p2:.7f}', data[i].enabled])
