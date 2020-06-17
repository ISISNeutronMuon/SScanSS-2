"""
Functions for creating Instrument from description file
"""
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
instrument_key = 'instrument'
GENERIC_TEMPLATE = '{{header}}\n{{#script}}\n{{position}}    {{mu_amps}}\n{{/script}}'


def read_jaw_description(instrument_data, positioners, path=''):
    """Reads incident jaws description and creates Jaws object

    :param instrument_data: instrument description
    :type instrument_data: Dict
    :param positioners: positioners in instrument description file
    :type positioners: Dict[str, SerialManipulator]
    :param path: directory of the instrument description file
    :type path: str
    :return: incident jaws
    :rtype: Jaws
    """
    jaws_key = 'incident_jaws'
    jaw_data = check(instrument_data, jaws_key, instrument_key)
    beam_axis = check(jaw_data, 'beam_direction', jaws_key, axis=True)
    beam_source = check(jaw_data, 'beam_source', jaws_key)
    aperture = check(jaw_data, 'aperture', jaws_key)
    upper_limit = check(jaw_data, 'aperture_upper_limit', jaws_key)
    lower_limit = check(jaw_data, 'aperture_lower_limit', jaws_key)

    if lower_limit[0] < 0.001 or lower_limit[1] < 0.001:
        raise ValueError(f'Aperture lower limit ({lower_limit}) must be greater than zero,'
                         ' (accurate to 3 decimal digits).')

    if lower_limit[0] > upper_limit[0] or lower_limit[1] > upper_limit[1]:
        raise ValueError(f'Aperture lower limit ({lower_limit}) is greater than upper ({upper_limit}).')

    if aperture[0] > upper_limit[0] or aperture[0] < lower_limit[0]:
        raise ValueError(f'Horizontal aperture value {aperture[0]} is outside joint limits [{lower_limit[0]}, '
                         f'{upper_limit[0]}].')

    if aperture[1] > upper_limit[1] or aperture[1] < lower_limit[1]:
        raise ValueError(f'Vertical aperture value {aperture[1]} is outside joint limits [{lower_limit[1]}, '
                         f'{upper_limit[1]}].')

    mesh = read_visuals(check(jaw_data, visual_key, jaws_key), path)
    positioner = None
    positioner_key = check(jaw_data, 'positioner', jaws_key, required=False)
    if positioner_key is not None:
        positioner = positioners.get(positioner_key, None)
        if positioner is None:
            raise ValueError(f'incident jaws positioner "{positioner_key}" definition was not found.')

    return Jaws("Incident Jaws", Vector3(beam_source), Vector3(beam_axis), aperture, lower_limit, upper_limit,
                mesh, positioner)


def read_detector_description(instrument_data, positioners, path=''):
    """Reads detector description and creates Detector object

    :param instrument_data: instrument description
    :type instrument_data: Dict
    :param positioners: positioners in instrument description file
    :type positioners: Dict[str, SerialManipulator]
    :param path: directory of the instrument description file
    :type path: str
    :return: dictionary of detectors
    :rtype: Dict[str, Detector]
    """
    detectors = {}
    all_collimators = {}
    detector_key = 'detector'
    collimator_key = 'collimator'
    detector_data = check(instrument_data, 'detectors', instrument_key)
    collimator_data = check(instrument_data, 'collimators', instrument_key, required=False)

    if collimator_data is not None:
        for collimator in collimator_data:
            name = check(collimator, 'name', collimator_key, name=True)
            detector_name = check(collimator, 'detector', collimator_key)
            aperture = check(collimator, 'aperture', collimator_key)
            mesh = read_visuals(check(collimator, visual_key, collimator_key), path)
            if detector_name not in all_collimators:
                all_collimators[detector_name] = {}
            all_collimators[detector_name][name] = Collimator(name, aperture, mesh)

    for detector in detector_data:
        detector_name = check(detector, 'name', detector_key)
        diff_beam = check(detector, 'diffracted_beam', detector_key, axis=True)
        detectors[detector_name] = Detector(detector_name, Vector3(diff_beam))

        collimators_for_detector = all_collimators.get(detector_name, dict())
        collimator_name = check(detector, 'default_collimator', detector_key, required=False)
        if collimator_name is not None and collimator_name not in collimators_for_detector:
            raise ValueError(f'Collimator "{collimator_name}" for Detector "{detector_name}" was not found.')

        detectors[detector_name].collimators = collimators_for_detector
        detectors[detector_name].current_collimator = collimator_name
        positioner_key = check(detector, 'positioner', detector_key, required=False)
        if positioner_key is not None:
            positioner = positioners.get(positioner_key, None)
            if positioner is None:
                raise ValueError(f'Detector positioner "{positioner_key}" definition was not found.')
            detectors[detector_name].positioner = positioner

    stray = set(all_collimators.keys()).difference(detectors.keys())
    if len(stray) > 0:
        raise ValueError(f'collimator object detector "{stray.pop()}" definition was not found.')

    return detectors


def read_visuals(visuals_data, path=''):
    """Reads visuals description and creates Mesh object

    :param visuals_data: visuals description
    :type visuals_data: Union[Dict, None]
    :param path: directory of the instrument description file
    :type path: str
    :return: mesh
    :rtype: Union[Mesh, None]
    """
    if visuals_data is None:
        return None

    pose = visuals_data.get('pose', DEFAULT_POSE)
    pose = matrix_from_pose(pose)
    mesh_colour = visuals_data.get('colour', DEFAULT_COLOUR)

    mesh_filename = check(visuals_data, 'mesh', visual_key)
    mesh = read_3d_model(os.path.join(path, mesh_filename))
    mesh.transform(pose)
    mesh.colour = Colour(*mesh_colour)

    return mesh


def check(json_data, key, parent_key, required=True, axis=False, name=False):
    """Returns value that belongs to given key from the description json and raise error if
    key does not exist or value is not in correct format such as an axis

    :param json_data: description json
    :type json_data: Dict[str, Any]
    :param key: key to validate
    :type key: str
    :param parent_key: parent key
    :type parent_key: str
    :param required: flag indicates data is check
    :type required: bool
    :param axis: flag indicates data must be axis
    :type axis: bool
    :param name: flag indicates data is a name
    :type name: bool
    :return: value
    :rtype: Any
    """
    data = json_data.get(key, None)
    if isinstance(data, str):
        data = data.strip()

    if required and data is None:
        raise KeyError(f'{parent_key} object must have a "{key}" attribute, {json_data}.')

    if axis and abs(1 - Vector(len(data), data).length) > 1e-7:
        raise ValueError(f'{key} must have a magnitude of 1 (accurate to 7 decimal digits), {json_data}')

    if name:
        if not data:
            raise ValueError(f'{key} can not be empty, {json_data}.')
        elif data.lower() == 'none':
            raise ValueError(f'{key} can not be "{data}" (None is a reserved keyword), {json_data}.')

    return data


def read_instrument_description_file(filename):
    """Reads instrument description and creates instrument object

    :param filename: filename of instrument json file
    :type filename: Union[pathlib.WindowsPath, str]
    :return: instrument
    :rtype: Instrument
    """
    with open(filename) as json_file:
        data = json.load(json_file)
        validate(data)

    directory = os.path.dirname(filename)

    instrument_data = check(data, instrument_key, 'description')
    instrument_name = check(instrument_data, 'name', instrument_key, name=True)
    script = read_script_template(instrument_data, directory)
    gauge_volume = check(instrument_data, 'gauge_volume', instrument_key)
    positioners = read_positioners_description(instrument_data, directory)
    positioning_stacks = read_positioning_stacks_description(instrument_data, positioners)
    detectors = read_detector_description(instrument_data, positioners, directory)
    incident_jaw = read_jaw_description(instrument_data, positioners, directory)
    fixed_hardware = read_fixed_hardware_description(instrument_data, directory)

    return Instrument(instrument_name, gauge_volume, detectors, incident_jaw, positioners,
                      positioning_stacks, script, fixed_hardware)


def read_script_template(instrument_data, path=''):
    """Reads the script template from file

    :param instrument_data: instrument description
    :type instrument_data: Dict
    :param path: directory of the instrument description file
    :type path: str
    :return script template
    :rtype Script
    """
    template_name = instrument_data.get('script_template', '').strip()
    template = GENERIC_TEMPLATE
    if template_name:
        template_path = os.path.join(path, template_name)
        with open(template_path, 'r') as template_file:
            template = template_file.read()

    return Script(template)


def read_fixed_hardware_description(instrument_data, path=''):
    """Reads fixed hardware description and creates dict of mesh objects

    :param instrument_data: instrument description
    :type instrument_data: Dict
    :param path: directory of the instrument description file
    :type path: str
    :return Mesh objects
    :rtype Dict[str, Mesh]
    """
    hardware_key = 'fixed_hardware'
    fixed_hardware = {}
    fixed_hardware_data = instrument_data.get(hardware_key, [])
    for data in fixed_hardware_data:
        name = check(data, 'name', hardware_key, name=True)
        visuals = check(data, visual_key, hardware_key)
        mesh = read_visuals(visuals, path)
        fixed_hardware[name] = mesh

    return fixed_hardware


def read_positioners_description(instrument_data, path=''):
    """Reads positioners description and creates dict of Positioner objects

    :param instrument_data: instrument description
    :type instrument_data: Dict
    :param path: directory of the instrument description file
    :type path: str
    :return Positioner objects
    :rtype Dict[str, Positioner]
    """
    positioners = {}
    positioner_data = check(instrument_data, 'positioners', instrument_key)
    for positioner in positioner_data:
        p = extract_positioner(positioner, path)
        positioners[p.name] = p

    return positioners


def extract_positioner(robot_data, path=''):
    """Reads positioner description and creates positioner object

    :param robot_data: positioner description
    :type robot_data: Dict[str, str]
    :param path: directory of the instrument description file
    :type path: str
    :return: positioner
    :rtype: SerialManipulator
    """
    joint_key = 'joint'
    positioner_key = 'positioner'

    positioner_name = check(robot_data, 'name', positioner_key, name=True)
    base_pose = robot_data.get('base', DEFAULT_POSE)
    base_matrix = matrix_from_pose(base_pose)
    tool_pose = robot_data.get('tool', DEFAULT_POSE)
    tool_matrix = matrix_from_pose(tool_pose)
    joints_data = check(robot_data, 'joints', positioner_key)
    links_data = check(robot_data, 'links', positioner_key)
    custom_order = robot_data.get('custom_order', None)

    links = {check(link, 'name', 'link', name=True): link for link in links_data}

    parent_child = {}
    joints = {}

    for joint in joints_data:
        parent_name = check(joint, 'parent', joint_key)
        child_name = check(joint, 'child', joint_key)
        parent_child[parent_name] = child_name
        joints[child_name] = joint

    parent = set(parent_child.keys())
    if len(parent) != len(joints_data):
        raise ValueError(f'"{positioner_name}" has a parent linked to multiple children.')

    child = set(parent_child.values())
    if len(child) != len(joints_data):
        raise ValueError(f'"{positioner_name}" has a child linked to multiple parents.')

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

        joint_name = check(joint, 'name', joint_key, name=True)
        axis = check(joint, 'axis', joint_key, axis=True)
        origin = check(joint, 'origin', joint_key)
        next_joint_origin = check(next_joint, 'origin', joint_key)
        vector = Vector3(next_joint_origin) - Vector3(origin)
        lower_limit = check(joint, 'lower_limit', joint_key)
        upper_limit = check(joint, 'upper_limit', joint_key)
        home = joint.get('home_offset', (upper_limit + lower_limit)/2)
        if lower_limit > upper_limit:
            raise ValueError(f'lower limit ({lower_limit}) for "{joint_name}" is greater than upper ({upper_limit}).')
        if home > upper_limit or home < lower_limit:
            raise ValueError(f'default offset for "{joint_name}" is outside joint limits [{lower_limit}, {upper_limit}].')
        _type = check(joint, 'type', joint_key)
        if _type == Link.Type.Revolute.value:
            joint_type = Link.Type.Revolute
            home = math.radians(home)
            lower_limit = math.radians(lower_limit)
            upper_limit = math.radians(upper_limit)
        elif _type == Link.Type.Prismatic.value:
            joint_type = Link.Type.Prismatic
        else:
            raise ValueError(f'joint type for "{joint_name}" is invalid in "{positioner_name}".')

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
            raise ValueError(f'"{positioner_name}" custom_order attribute must contain all joints with no duplicates')
        diff = set(custom_order).difference(joint_order)
        if diff:
            raise ValueError(f'"{positioner_name}" custom_order attribute has incorrect joint names {diff}.')

        custom_order = [joint_order.index(x) for x in custom_order]

    return SerialManipulator(positioner_name, qv_links, base=base_matrix, tool=tool_matrix,  base_mesh=mesh,
                             custom_order=custom_order)


def read_positioning_stacks_description(instrument_data, positioners):
    """Reads positioning stacks description

    :param instrument_data: instrument description
    :type instrument_data: Dict
    :param positioners: Positioner objects
    :type positioners: Dict[str, Positioner]
    :return Positioning stacks
    :rtype Dict[str, List[str]]
    """
    positioning_stacks = {}
    positioning_stacks_data = check(instrument_data, 'positioning_stacks', instrument_key)
    defined_positioners = positioners.keys()

    for stack in positioning_stacks_data:
        stack_name = check(stack, 'name', 'positioning_stacks', name=True)
        positioners_in_stack = [n.strip() for n in check(stack, 'positioners', 'positioning_stacks')]

        if stack_name in defined_positioners:
            if not (len(positioners_in_stack) == 1 and stack_name == positioners_in_stack[0]):
                raise ValueError(f'"{stack_name}" positioning stack name conflicts with a positioner of the same name.')

        temp = set(positioners_in_stack)
        if len(temp) != len(positioners_in_stack):
            raise ValueError(f'In "{stack_name}" positioning stack, positioners are duplicated.')

        undefined_positioners = temp.difference(defined_positioners)
        if undefined_positioners:
            if len(undefined_positioners) == 1:
                error = f'"{undefined_positioners.pop()}" positioner definition was not found'
            else:
                error = f'positioner definitions were not found for {undefined_positioners}'
            raise ValueError(f'In "{stack_name}" positioning stack, {error}.')

        positioning_stacks[stack_name] = positioners_in_stack

    return positioning_stacks
