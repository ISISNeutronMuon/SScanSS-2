import os
import json
import math
from .instrument import Instrument, Collimator, Detector, Jaws, Script
from .robotics import Link, SerialManipulator
from .__validator import validate
from ..io.reader import read_3d_model
from ..math.vector import Vector3, Vector
from ..math.transform import matrix_from_pose
from ..geometry.colour import Colour


DEFAULT_POSE = [0., 0., 0., 0., 0., 0.]
DEFAULT_COLOUR = [0., 0., 0.]
visual_key = 'visual'
GENERIC_TEMPLATE = '{{header}}\n{{#script}}\n{{position}}    {{mu_amps}}\n{{/script}}'


def read_jaw_description(jaws, positioners, path):
    beam_axis = required(jaws, 'beam_direction', 'incident_jaws', axis=True)
    beam_source = required(jaws, 'beam_source', 'incident_jaws')
    aperture = required(jaws, 'aperture', 'incident_jaws')
    upper_limit = required(jaws, 'aperture_upper_limit', 'incident_jaws')
    lower_limit = required(jaws, 'aperture_lower_limit', 'incident_jaws')
    if aperture[0] > upper_limit[0] or aperture[0] < lower_limit[0]:
        raise ValueError(f'Horizontal aperture value {aperture[0]} is outside joint limits [{lower_limit[0]}, '
                         f'{upper_limit[0]}].')

    if aperture[1] > upper_limit[1] or aperture[1] < lower_limit[1]:
        raise ValueError(f'Vertical aperture value {aperture[1]} is outside joint limits [{lower_limit[1]}, '
                         f'{upper_limit[1]}].')

    mesh = read_visuals(required(jaws, visual_key, 'incident_jaws'), path)
    positioner = None
    positioner_key = jaws.get('positioner', None)
    if positioner_key is not None:
        positioner = positioners.get(positioner_key, None)
        if positioner is None:
            raise ValueError(f'incident jaws positioner "{positioner_key}" definition was not found.')

    s = Jaws("Incident Jaws", Vector3(beam_source), Vector3(beam_axis), aperture, lower_limit, upper_limit,
             mesh, positioner)

    return s


def read_detector_description(detector_data, collimator_data, positioners, path=''):
    detectors = {}
    collimators = {}
    for collimator in collimator_data:
        name = required(collimator, 'name', 'collimator').strip()
        detector_name = required(collimator, 'detector', 'collimator').strip()
        aperture = required(collimator, 'aperture', 'collimator')
        mesh = read_visuals(required(collimator, visual_key, 'collimator'), path)
        if detector_name not in collimators:
            collimators[detector_name] = {}
        collimators[detector_name][name] = Collimator(name, aperture, mesh)

    for detector in detector_data:
        detector_name = required(detector, 'name', 'detector').strip()
        diff_beam = required(detector, 'diffracted_beam', 'detector', axis=True)
        detectors[detector_name] = Detector(detector_name, Vector3(diff_beam))
        detectors[detector_name].collimators = collimators.get(detector_name, dict())
        detectors[detector_name].current_collimator = detector.get('default_collimator', None)
        positioner_key = detector.get('positioner', None)
        if positioner_key is not None:
            positioner = positioners.get(positioner_key, None)
            if positioner is None:
                raise ValueError(f'Detector positioner "{positioner_key}" definition was not found.')
            detectors[detector_name].positioner = positioner

    stray = set(collimators.keys()).difference(detectors.keys())
    if len(stray) > 0:
        raise ValueError(f'collimator object detector "{stray.pop()}" definition was not found.')

    return detectors


def read_visuals(visuals_data, path=''):
    if visuals_data is None:
        return None

    pose = visuals_data.get('pose', DEFAULT_POSE)
    pose = matrix_from_pose(pose)
    mesh_colour = visuals_data.get('colour', DEFAULT_COLOUR)

    mesh_filename = required(visuals_data, 'mesh', visual_key).strip()
    mesh = read_3d_model(os.path.join(path, mesh_filename))
    mesh.transform(pose)
    mesh.colour = Colour(*mesh_colour)

    return mesh


def required(json_data, key, parent_key, axis=False):
    data = json_data.get(key, None)
    if data is None:
        raise KeyError(f'{parent_key} object must have a "{key}" attribute, {json_data}.')

    if axis and abs(1 - Vector(len(data), data).length) > 1e-7:
        raise ValueError(f'{parent_key}/{key} must have a magnitude of 1 (accurate to 7 decimal digits), {json_data}')

    return data


def read_instrument_description_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        validate(data)

    directory = os.path.dirname(filename)

    instrument_key = 'instrument'
    instrument_data = required(data, instrument_key, 'description')

    instrument_name = required(instrument_data, 'name', instrument_key).strip()
    gauge_volume = required(instrument_data, 'gauge_volume', instrument_key)
    positioner_data = required(instrument_data, 'positioners', instrument_key)

    positioners = {}
    for positioner in positioner_data:
        p = read_positioner_description(positioner, directory)
        positioners[p.name] = p

    positioning_stacks_data = required(instrument_data, 'positioning_stacks', instrument_key)

    positioning_stacks = {}
    defined_positioners = positioners.keys()
    for stack in positioning_stacks_data:
        stack_name = required(stack, 'name', 'positioning_stacks').strip()
        positioners_in_stack = required(stack, 'positioners', 'positioning_stacks')

        temp = set(positioners_in_stack)
        if len(temp) != len(positioners_in_stack):
            raise ValueError(f'In "{stack_name}" positioning stack, positioners are duplicated.')

        undefined_positioners = temp.difference(defined_positioners)
        for key in undefined_positioners:
            raise ValueError(f'In "{stack_name}" positioning stack, "{key}" positioner definition was not found.')

        positioning_stacks[stack_name] = positioners_in_stack

    detector_data = required(instrument_data, 'detectors', instrument_key)
    collimator_data = required(instrument_data, 'collimators', instrument_key)

    detectors = read_detector_description(detector_data, collimator_data, positioners, directory)

    jaw_data = required(instrument_data, 'incident_jaws', instrument_key)
    incident_jaw = read_jaw_description(jaw_data, positioners, directory)

    template_name = instrument_data.get('script_template', '').strip()
    template = GENERIC_TEMPLATE
    if template_name:
        template_path = os.path.join(os.path.dirname(filename), template_name)
        with open(template_path, 'r') as template_file:
            template = template_file.read()

    script = Script(template)

    fixed_hardware = {}
    fixed_hardware_data = instrument_data.get('fixed_hardware', [])
    for data in fixed_hardware_data:
        name = required(data, 'name', 'fixed_hardware').strip()
        visuals = required(data, visual_key, 'fixed_hardware')
        mesh = read_visuals(visuals, directory)
        fixed_hardware[name] = mesh

    instrument = Instrument(instrument_name, gauge_volume, detectors, incident_jaw,
                            positioners, positioning_stacks, script, fixed_hardware)

    return instrument


def read_positioner_description(robot_data, path=''):
    joint_key = 'joint'
    positioner_key = 'positioner'

    positioner_name = required(robot_data, 'name', positioner_key).strip()
    base_pose = robot_data.get('base', DEFAULT_POSE)
    base_matrix = matrix_from_pose(base_pose)
    tool_pose = robot_data.get('tool', DEFAULT_POSE)
    tool_matrix = matrix_from_pose(tool_pose)
    joints_data = required(robot_data, 'joints', positioner_key )
    links_data = required(robot_data, 'links', positioner_key )
    custom_order = robot_data.get('custom_order', None)

    links = {required(link, 'name', 'link').strip(): link for link in links_data}

    parent_child = {}
    joints = {}

    for joint in joints_data:
        parent_name = required(joint, 'parent', joint_key).strip()
        child_name = required(joint, 'child', joint_key).strip()
        parent_child[parent_name] = child_name
        joints[child_name] = joint

    parent = set(parent_child.keys())
    if len(parent) != len(joints_data):
        raise ValueError(f'{positioner_name} has a parent linked to multiple children.')

    child = set(parent_child.values())
    if len(child) != len(joints_data):
        raise ValueError(f'{positioner_name} has a child linked to multiple parents.')

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

        joint_name = required(joint, 'name', joint_key).strip()
        axis = required(joint, 'axis', joint_key, axis=True)
        origin = required(joint, 'origin', joint_key)
        next_joint_origin = required(next_joint, 'origin', joint_key)
        vector = Vector3(next_joint_origin) - Vector3(origin)
        lower_limit = required(joint, 'lower_limit', joint_key)
        upper_limit = required(joint, 'upper_limit', joint_key)
        home = joint.get('home_offset', (upper_limit + lower_limit)/2)
        if lower_limit > upper_limit:
            raise ValueError(f'lower limit ({lower_limit}) for {joint_name} is greater than upper ({upper_limit}).')
        if home > upper_limit or home < lower_limit:
            raise ValueError(f'default offset for {joint_name} is outside joint limits [{lower_limit}, {upper_limit}].')
        _type = required(joint, 'type', joint_key)
        if _type == Link.Type.Revolute.value:
            joint_type = Link.Type.Revolute
            home = math.radians(home)
            lower_limit = math.radians(lower_limit)
            upper_limit = math.radians(upper_limit)
        elif _type == Link.Type.Prismatic.value:
            joint_type = Link.Type.Prismatic
        else:
            raise ValueError(f'joint type for {joint_name} is invalid ')

        mesh = read_visuals(link.get(visual_key, None), path)

        qv_links.append(Link(joint_name, axis, vector, joint_type, lower_limit, upper_limit,
                             default_offset=home, mesh=mesh))

    base_link = links.get(base_link_name, None)
    if base_link is None:
        raise ValueError(f'"{base_link_name}" link definition not found. Did you misspell its name?')
    mesh = read_visuals(base_link.get(visual_key, None), path)

    if custom_order is not None:
        joint_order = [links.name for links in qv_links]
        if len(set(custom_order)) != len(joint_order):
            raise ValueError(f'{positioner_name} "custom_order" attribute must contain all joints with no duplicates')
        diff = set(custom_order).difference(joint_order)
        if diff:
            raise ValueError(f'{positioner_name} "custom_order" attribute has incorrect joint names {diff}.')

        custom_order = [joint_order.index(x) for x in custom_order]

    return SerialManipulator(positioner_name, qv_links, base=base_matrix, tool=tool_matrix,  base_mesh=mesh,
                             custom_order=custom_order)
