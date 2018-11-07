import os
import json
import math
import sscanss.instruments
from .instrument import Instrument, Collimator, Detector, Jaws
from .robotics import Link, SerialManipulator
from ..io.reader import read_3d_model
from ..math.vector import Vector3
from ..math.transform import matrix_from_pose
from ..mesh.colour import Colour


def get_instrument_list():
    instruments = {}
    instruments_path = os.path.dirname(sscanss.instruments.__file__)
    files_in_instruments_path = os.listdir(instruments_path)
    for filename in files_in_instruments_path:
        idf = os.path.join(instruments_path, filename, 'instrument.json')
        if not os.path.isfile(idf):
            continue

        try:
            with open(idf) as json_file:
                data = json.load(json_file)
        except (IOError, ValueError):
            continue

        instrument_data = data.get('instrument', None)
        if instrument_data is None:
            continue

        name = instrument_data.get('name', '').strip().upper()
        if name:
            instruments[name] = idf

    return instruments


def read_jaw_description(jaws, guide, stop, positioner, path=''):
    aperture = required(jaws, 'aperture', '')
    upper_limit = required(jaws, 'aperture_upper_limit', '')
    lower_limit = required(jaws, 'aperture_lower_limit', '')
    positioner_key = required(jaws, 'positioner', '')

    s = Jaws("Incident Jaws", aperture, upper_limit, lower_limit, positioner[positioner_key])

    mesh_1 = read_visuals(guide.get('visual', None), path)
    mesh_2 = read_visuals(stop.get('visual', None), path)

    return s, mesh_1, mesh_2


def read_detector_description(detector_data, collimator_data, path=''):
    detectors = {}
    collimators = {}

    for collimator in collimator_data:
        name = required(collimator, 'name', '')
        detector_name = required(collimator, 'detector', '')
        aperture = required(collimator, 'aperture', '')
        mesh = read_visuals(collimator.get('visual', None), path)
        if detector_name not in collimators:
            collimators[detector_name] = {}
        collimators[detector_name][name] = Collimator(name, aperture, mesh)

    for detector in detector_data:
        detector_name = required(detector, 'name', '')
        detectors[detector_name] = Detector(detector_name)
        detectors[detector_name].collimators = collimators[detector_name]
        detectors[detector_name].current_collimator = detector.get('default_collimator', None)

    return detectors


DEFAULT_POSE = [0., 0., 0., 0., 0., 0.]
DEFAULT_COLOUR = [0., 0., 0.]


def read_visuals(visuals_data, path=''):
    if visuals_data is None:
        return None

    pose = visuals_data.get('pose', DEFAULT_POSE)
    pose = matrix_from_pose(pose)
    mesh_colour = visuals_data.get('colour', DEFAULT_COLOUR)

    mesh_filename = required(visuals_data, 'mesh', '')
    mesh, *ignore = read_3d_model(os.path.join(path, mesh_filename))
    mesh.transform(pose)
    mesh.colour = Colour(*mesh_colour)

    return mesh


def required(json_data, key, error_message):
    data = json_data.get(key, None)
    if data is None:
        raise KeyError(error_message)

    return data


def read_instrument_description_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)

    directory = os.path.dirname(filename)

    instrument_data = required(data, 'instrument', '')

    instrument_name = instrument_data.get('name', 'UNKNOWN')
    positioner_data = required(instrument_data, 'positioners', '')

    positioners = {}
    for positioner in positioner_data:
        p = read_positioner_description(positioner, directory)
        positioners[p.name] = p

    fixed_positioner = None  # currently only support a single fixed positioner
    auxiliary_positioners = []
    sample_positioners = required(instrument_data, 'sample_positioners', '')
    for positioner in sample_positioners:
        if positioner['movable']:
            auxiliary_positioners.append(positioner['positioner'])
        else:
            fixed_positioner = positioner['positioner']

    detector_data = required(instrument_data, 'detectors', '')
    collimator_data = required(instrument_data, 'collimators', '')

    detectors = read_detector_description(detector_data, collimator_data, directory)

    jaw = required(instrument_data, 'incident_jaws', '')
    beam_guide = required(instrument_data, 'beam_guide', '')
    beam_stop = required(instrument_data, 'beam_stop', '')

    e = read_jaw_description(jaw, beam_guide, beam_stop, positioners, directory)

    instrument = Instrument(instrument_name, detectors, e[0], positioners,
                            fixed_positioner, auxiliary_positioners, e[1], e[2])

    return instrument


def read_positioner_description(robot_data, path=''):
    positioner_name = required(robot_data, 'name', '')
    base_pose = robot_data.get('base', DEFAULT_POSE)
    base_matrix = matrix_from_pose(base_pose)
    joints_data = required(robot_data, 'joints', '')
    links_data = required(robot_data, 'links', '')

    links = {link['name']: link for link in links_data}

    parent_child = {}
    joints = {}

    for joint in joints_data:
        parent_name = required(joint, 'parent', '')
        child_name = required(joint, 'child', '')
        parent_child[parent_name] = child_name
        joints[child_name] = joint

    parent = set(parent_child.keys())
    child = set(parent_child.values())
    base_link_name = parent.difference(child).pop()
    top_link_name = child.difference(parent).pop()

    link_order = []
    for i in range(len(parent)):
        if i == 0:
            key = base_link_name

        key = parent_child[key]
        link_order.append(key)

    qv_links = []
    for index, key in enumerate(link_order):
        joint = joints[key]
        next_joint = joint
        if index < len(link_order) - 1:
            next_key = link_order[index + 1]
            next_joint = joints[next_key]

        link = links[key]

        name = required(joint, 'name', '')
        axis = required(joint, 'axis', '')
        origin = required(joint, 'origin', '')
        next_joint_origin = required(next_joint, 'origin', '')
        vector = Vector3(next_joint_origin) - Vector3(origin)
        lower_limit = required(joint, 'lower_limit', '')
        upper_limit = required(joint, 'upper_limit', '')
        home = joint.get('home_offset', (upper_limit+lower_limit)/2)
        if home > upper_limit or home < lower_limit:
            raise ValueError()
        _type = required(joint, 'type', '')
        if _type == 'revolute':
            joint_type = Link.Type.Revolute
            home = math.radians(home)
            lower_limit = math.radians(lower_limit)
            upper_limit = math.radians(upper_limit)
        elif _type == 'prismatic':
            joint_type = Link.Type.Prismatic
        else:
            raise ValueError('Invalid joint type for {}'.format(name))

        mesh = read_visuals(link.get('visual', None), path)

        tmp = Link(axis, vector, joint_type, upper_limit=upper_limit, lower_limit=lower_limit,
                   mesh=mesh, name=name, home_offset=home)
        qv_links.append(tmp)

    base_link = links[base_link_name]
    mesh = read_visuals(base_link.get('visual', None), path)

    return SerialManipulator(qv_links, base=base_matrix, base_mesh=mesh, name=positioner_name)