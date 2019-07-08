import os
import json
import math
from contextlib import suppress
from .instrument import Instrument, Collimator, Detector, Jaws, ScriptTemplate
from .robotics import Link, SerialManipulator
from ..io.reader import read_3d_model
from ..math.vector import Vector3
from ..math.transform import matrix_from_pose
from ..geometry.colour import Colour
from ...config import INSTRUMENTS_PATH


def get_instrument_list():
    instruments = {}
    files_in_instruments_path = os.listdir(INSTRUMENTS_PATH)
    for name in files_in_instruments_path:
        idf = INSTRUMENTS_PATH / name / 'instrument.json'
        if not os.path.isfile(idf):
            continue

        data = {}
        with suppress(IOError, ValueError):
            with open(idf) as json_file:
                data = json.load(json_file)

        instrument_data = data.get('instrument', None)
        if instrument_data is None:
            continue

        name = instrument_data.get('name', '').strip().upper()
        if name:
            instruments[name] = idf

    return instruments


def read_jaw_description(jaws, guide, stop, positioners, path=''):
    error = 'incident_jaws object must have a "{}" attribute, {}.'
    aperture = required(jaws, 'aperture', error.format('aperture', jaws))
    upper_limit = required(jaws, 'aperture_upper_limit', error.format('aperture_upper_limit', jaws))
    lower_limit = required(jaws, 'aperture_lower_limit', error.format('aperture_lower_limit', jaws))
    positioner_key = required(jaws, 'positioner', error.format('positioner', jaws))

    positioner = positioners.get(positioner_key, None)
    if positioner is None:
        raise ValueError('incident jaws positioner "{}" definition was not found.'.format(positioner_key))
    s = Jaws("Incident Jaws", aperture, upper_limit, lower_limit, positioner)

    mesh_1 = read_visuals(guide.get('visual', None), path)
    mesh_2 = read_visuals(stop.get('visual', None), path)

    return s, mesh_1, mesh_2


def read_detector_description(detector_data, collimator_data, positioners, path=''):
    detectors = {}
    collimators = {}
    error = 'collimator object must have a "{}" attribute, {}.'
    for collimator in collimator_data:
        name = required(collimator, 'name', error.format('name', collimator))
        detector_name = required(collimator, 'detector', error.format('detector', collimator))
        aperture = required(collimator, 'aperture', error.format('aperture', collimator))
        mesh = read_visuals(collimator.get('visual', None), path)
        if detector_name not in collimators:
            collimators[detector_name] = {}
        collimators[detector_name][name] = Collimator(name, aperture, mesh)

    error = 'detector object must have a "{}" attribute, {}.'
    for detector in detector_data:
        detector_name = required(detector, 'name', error.format('name', detector))
        detectors[detector_name] = Detector(detector_name)
        detectors[detector_name].collimators = collimators.get(detector_name, dict())
        detectors[detector_name].current_collimator = detector.get('default_collimator', None)
        positioner_key = detector.get('positioner', None)
        if positioner_key is not None:
            positioner = positioners.get(positioner_key, None)
            if positioner is None:
                raise ValueError('Detector positioner "{}" definition was not found.'.format(positioner_key))
            detectors[detector_name].positioner = positioner

    stray = set(collimators.keys()).difference(detectors.keys())
    if len(stray) > 0:
        raise ValueError('collimator object detector "{}" definition was not found.'.format(stray.pop()))

    return detectors


DEFAULT_POSE = [0., 0., 0., 0., 0., 0.]
DEFAULT_COLOUR = [0., 0., 0.]


def read_visuals(visuals_data, path=''):
    if visuals_data is None:
        return None

    pose = visuals_data.get('pose', DEFAULT_POSE)
    pose = matrix_from_pose(pose)
    mesh_colour = visuals_data.get('colour', DEFAULT_COLOUR)

    error = 'visual object must have a "{}" attribute, {}.'
    mesh_filename = required(visuals_data, 'mesh', error.format('mesh', visuals_data))
    mesh = read_3d_model(os.path.join(path, mesh_filename))
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

    error = 'instrument object must have a "{}" attribute.'
    instrument_data = required(data, 'instrument', 'description file has no instrument object')

    instrument_name = required(instrument_data, 'name', error.format('name'))
    positioner_data = required(instrument_data, 'positioners', error.format('positioners'))

    positioners = {}
    for positioner in positioner_data:
        p = read_positioner_description(positioner, directory)
        positioners[p.name] = p

    positioning_stacks_data = required(instrument_data, 'positioning_stacks', error.format('sample_positioners'))

    positioning_stacks = {}
    defined_positioners = positioners.keys()
    for stack in positioning_stacks_data:
        stack_name = required(stack, 'name',
                              '"name" attribute is required in the positioning_stacks object.')
        positioners_in_stack = required(stack, 'positioners',
                                        '"positioners" attribute is required in the positioning_stacks object.')

        temp = set(positioners_in_stack)
        if len(temp) != len(positioners_in_stack):
            raise ValueError('In "{}" positioning stack, positioners are duplicated.'.format(stack_name))

        undefined_positioners = temp.difference(defined_positioners)
        for key in undefined_positioners:
            error = 'In "{}" positioning stack, "{}" positioner definition was not found.'
            raise ValueError(error.format(stack_name, key))

        positioning_stacks[stack_name] = positioners_in_stack

    detector_data = required(instrument_data, 'detectors', error.format('detectors'))
    collimator_data = required(instrument_data, 'collimators', error.format('collimators'))

    detectors = read_detector_description(detector_data, collimator_data, positioners, directory)

    jaw = required(instrument_data, 'incident_jaws', error.format('incident_jaws'))
    beam_guide = required(instrument_data, 'beam_guide', error.format('beam_guide'))
    beam_stop = required(instrument_data, 'beam_stop', error.format('beam_stop'))

    e = read_jaw_description(jaw, beam_guide, beam_stop, positioners, directory)

    template_name = instrument_data.get('script_template_path', '')
    if not template_name:
        search_path = str(INSTRUMENTS_PATH)
        template_name = 'generic_script_template'
    else:
        search_path = os.path.dirname(filename)

    script_template = ScriptTemplate(template_name, search_path)

    instrument = Instrument(instrument_name, detectors, e[0], positioners, positioning_stacks, script_template,
                            e[1], e[2])

    return instrument


def read_positioner_description(robot_data, path=''):
    error = 'positioner object must have a "{}" attribute, {}.'
    positioner_name = required(robot_data, 'name', error.format('name', robot_data))
    base_pose = robot_data.get('base', DEFAULT_POSE)
    base_matrix = matrix_from_pose(base_pose)
    joints_data = required(robot_data, 'joints', error.format('joints', robot_data))
    links_data = required(robot_data, 'links', error.format('links', robot_data))
    custom_order = robot_data.get('custom_order', None)

    error = 'link object must have a "{}" attribute, {}.'
    links = {required(link, 'name', error.format('name', link)): link for link in links_data}

    parent_child = {}
    joints = {}

    error = 'joint object must have a "{}" attribute, {}.'
    for joint in joints_data:
        parent_name = required(joint, 'parent', error.format('parent', joint))
        child_name = required(joint, 'child', error.format('child', joint))
        parent_child[parent_name] = child_name
        joints[child_name] = joint

    parent = set(parent_child.keys())
    if len(parent) != len(joints_data):
        raise ValueError('{} has a parent linked to multiple children.'.format(positioner_name))

    child = set(parent_child.values())
    if len(child) != len(joints_data):
        raise ValueError('{} has a child linked to multiple parents.'.format(positioner_name))

    base_link_name = parent.difference(child).pop()

    link_order = []
    key = base_link_name
    for _ in range(len(parent)):
        key = parent_child.get(key, None)
        if key is None:
            raise ValueError('floating link is detected. Check the joint parent and child attributes.')

        link_order.append(key)

    qv_links = []
    for index, key in enumerate(link_order):
        joint = joints[key]
        next_joint = joint
        if index < len(link_order) - 1:
            next_key = link_order[index + 1]
            next_joint = joints[next_key]

        link = links.get(key, None)
        if link is None:
            raise ValueError(f'"{key}" link definition not found. Did you misspell its name?')

        joint_name = required(joint, 'name', error.format('name', joint))
        axis = required(joint, 'axis', error.format('axis', joint))
        origin = required(joint, 'origin', error.format('origin', joint))
        next_joint_origin = required(next_joint, 'origin', error.format('origin', next_joint))
        vector = Vector3(next_joint_origin) - Vector3(origin)
        lower_limit = required(joint, 'lower_limit', error.format('lower_limit', joint))
        upper_limit = required(joint, 'upper_limit', error.format('upper_limit', joint))
        home = joint.get('home_offset', (upper_limit+lower_limit)/2)
        if home > upper_limit or home < lower_limit:
            err = f'default offset for {joint_name} is outside joint limits [{lower_limit}, {upper_limit}]'
            raise ValueError(err)
        _type = required(joint, 'type', error.format('type', joint))
        if _type == 'revolute':
            joint_type = Link.Type.Revolute
            home = math.radians(home)
            lower_limit = math.radians(lower_limit)
            upper_limit = math.radians(upper_limit)
        elif _type == 'prismatic':
            joint_type = Link.Type.Prismatic
        else:
            raise ValueError(f'joint type for {joint_name} is invalid ')

        mesh = read_visuals(link.get('visual', None), path)

        tmp = Link(axis, vector, joint_type, upper_limit=upper_limit, lower_limit=lower_limit,
                   mesh=mesh, name=joint_name, default_offset=home)
        qv_links.append(tmp)

    base_link = links.get(base_link_name, None)
    if base_link is None:
        raise ValueError(f'"{base_link_name}" link definition not found. Did you misspell its name?')
    mesh = read_visuals(base_link.get('visual', None), path)

    if custom_order is not None:
        joint_order = [links.name for links in qv_links]
        if len(set(custom_order)) != len(joint_order):
            raise ValueError(f'{positioner_name} "custom_order" attribute must contain all joints with no duplicates')
        diff = set(custom_order).difference(joint_order)
        if diff:
            raise ValueError(f'{positioner_name} "custom_order" attribute has incorrect joint names {diff}.')

        custom_order = [joint_order.index(x) for x in custom_order]

    return SerialManipulator(qv_links, base=base_matrix, base_mesh=mesh, name=positioner_name,
                             custom_order=custom_order)
