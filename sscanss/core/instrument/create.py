"""
Functions for creating Instrument from description file
"""
import json
import math
import pathlib
import jsonschema
from copy import deepcopy
from .instrument import Instrument, Collimator, Detector, Jaws, Script
from .robotics import Link, SerialManipulator
from ..io.reader import read_3d_model
from ..math.constants import VECTOR_EPS, POS_EPS
from ..math.vector import Vector3, Vector
from ..math.structure import Plane
from ..math.transform import matrix_from_pose
from ..geometry.colour import Colour
from ..geometry.primitive import create_cuboid, create_plane, create_sphere
from ..util.misc import find_duplicates, VisualGeometry
from ...config import INSTRUMENT_SCHEMA

DEFAULT_POSE = [0., 0., 0., 0., 0., 0.]
DEFAULT_COLOUR = [0., 0., 0.]
visual_key = 'visual'
geometry_key = 'geometry'
instrument_key = 'instrument'
GENERIC_TEMPLATE = '{{header}}\n{{#script}}\n{{position}}    {{mu_amps}}\n{{/script}}'

__cls = jsonschema.validators.validator_for(INSTRUMENT_SCHEMA)
__cls.check_schema(INSTRUMENT_SCHEMA)
schema_validator = __cls(INSTRUMENT_SCHEMA)


class ParserError(Exception):
    """Instrument description parsing error

    :param path: json path
    :type path: str
    :param msg: error message
    :type msg: str
    """
    def __init__(self, path, msg):
        self.path = path
        self.message = msg


class InstrumentParser:
    """This class parses instrument json and store a list of errors."""
    def __init__(self):
        self.errors = []
        self.data = {}

    def parse(self, json_text, directory):
        """Parses the instrument json

        :param json_text: instrument json
        :type json_text: str
        :param directory: folder path for 3D models
        :type directory: str
        :return: instrument
        :rtype: Instrument
        """
        self.data = {}
        self.errors.clear()
        try:
            self.data = json.loads(json_text)
        except json.JSONDecodeError as e:
            self.errors.append(ParserError('$', str(e).strip("'")))
            raise self.errors[0]

        errors = sorted(schema_validator.iter_errors(self.data), key=lambda ex: ex.path)
        for e in errors:
            path = '$'
            for p in e.absolute_path:
                path = f'{path}[{p}]' if isinstance(p, int) else f'{path}.{p}'

            self.errors.append(ParserError(path, e.message))

        if not self.errors:
            try:
                instrument_data = check(self.data, instrument_key, 'description')
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
            except Exception as e:
                self.errors.append(ParserError('$', str(e).strip("'")))

        raise self.errors[0]


def read_jaw_description(instrument_data, positioners, path=''):
    """Creates Jaws object from a jaws description

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

    if lower_limit[0] < POS_EPS or lower_limit[1] < POS_EPS:
        raise ValueError(f'Aperture lower limit ({lower_limit}) must be greater than zero,'
                         ' (accurate to 3 decimal digits).')

    if lower_limit[0] > upper_limit[0] or lower_limit[1] > upper_limit[1]:
        raise ValueError(f'Aperture lower limit ({lower_limit}) is greater than upper ({upper_limit}).')

    if aperture[0] > upper_limit[0] or aperture[0] < lower_limit[0]:
        raise ValueError(f'Horizontal aperture value {aperture[0]} is outside aperture limits [{lower_limit[0]}, '
                         f'{upper_limit[0]}].')

    if aperture[1] > upper_limit[1] or aperture[1] < lower_limit[1]:
        raise ValueError(f'Vertical aperture value {aperture[1]} is outside aperture limits [{lower_limit[1]}, '
                         f'{upper_limit[1]}].')

    mesh = read_visuals(check(jaw_data, visual_key, jaws_key), path)
    positioner = None
    positioner_key = check(jaw_data, 'positioner', jaws_key, required=False)
    if positioner_key is not None:
        positioner = positioners.get(positioner_key, None)
        if positioner is None:
            raise ValueError(f'incident jaws positioner "{positioner_key}" definition was not found.')

    return Jaws("Incident Jaws", Vector3(beam_source), Vector3(beam_axis), aperture, lower_limit, upper_limit, mesh,
                positioner)


def read_detector_description(instrument_data, positioners, path=''):
    """Creates Detector object from a detector description

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

    collimator_names = {}
    if collimator_data is not None:
        for collimator in collimator_data:
            name = check(collimator, 'name', collimator_key, name=True)
            detector_name = check(collimator, 'detector', collimator_key)
            aperture = check(collimator, 'aperture', collimator_key)
            mesh = read_visuals(check(collimator, visual_key, collimator_key), path)
            if detector_name not in all_collimators:
                all_collimators[detector_name] = {}
                collimator_names[detector_name] = []
            all_collimators[detector_name][name] = Collimator(name, aperture, mesh)
            collimator_names[detector_name].append(name)

    detector_names = []
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
        detector_names.append(detector_name)

    duplicate_names = find_duplicates(detector_names)
    if duplicate_names:
        raise ValueError(f'Detectors has duplicate name(s): {duplicate_names}.')

    for key, values in collimator_names.items():
        duplicate_names = find_duplicates(values)
        if duplicate_names:
            raise ValueError(f'"{key}" detector has duplicate collimator name(s): {duplicate_names}.')

    stray = set(all_collimators.keys()).difference(detectors.keys())
    if len(stray) > 0:
        raise ValueError(f'collimator object detector "{stray.pop()}" definition was not found.')

    return detectors


def read_visuals(visuals_data, path=''):
    """Creates Mesh object from a visuals description using a file if a mesh path supplied, 
    else the specified primitive geometry is used

    :param visuals_data: visuals description
    :type visuals_data: Union[Dict, None]
    :param path: directory of the instrument description file
    :type path: str
    :return: mesh
    :rtype: Union[Mesh, None]
    """
    if visuals_data:
        pose = visuals_data.get('pose', DEFAULT_POSE)
        pose = matrix_from_pose(pose)
        mesh_colour = visuals_data.get('colour', DEFAULT_COLOUR)

        try:
            geometry = visuals_data.get('geometry')
            mesh_filename = check(visuals_data, 'mesh', visual_key) if not geometry else check(visuals_data[geometry_key], 'path', geometry_key)
            mesh = read_3d_model(pathlib.Path(path).joinpath(mesh_filename).as_posix())
        except:
            geometry = check(visuals_data[geometry_key], 'type', geometry_key)
            geometry = deepcopy(visuals_data[geometry_key])
            geom_type = geometry.pop('type')
            dimensions = list(geometry.values()).pop()

            if geom_type == VisualGeometry.Box.value:
                mesh = create_cuboid(dimensions[0], dimensions[2], dimensions[1])
            if geom_type == VisualGeometry.Sphere.value:
                mesh = create_sphere(dimensions) 
            if geom_type == VisualGeometry.Plane.value:
                mesh = create_plane(Plane.fromCoefficient(1, 1, 0, 0), dimensions[0], dimensions[1])
        else:
            mesh.transform(pose)
            mesh.colour = Colour(*mesh_colour)

            return mesh
                

def check(json_data, key, parent_key, required=True, axis=False, name=False):
    """Gets the value that belongs to the given key from a description json and raise error if
    a required key does not exist or value is not in the correct format such as an axis

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
        raise KeyError(f'{parent_key} object must have a "{key}" attribute.')

    if axis and abs(1 - Vector(len(data), data).length) > VECTOR_EPS:
        raise ValueError(f'{parent_key}.{key} must have a magnitude of 1 (accurate to 7 decimal digits).')

    if name:
        if not data:
            raise ValueError(f'{parent_key}.{key} can not be empty.')
        elif data.lower() == 'none':
            raise ValueError(f'{parent_key}.{key} can not be "{data}" (None is a reserved keyword).')

    return data


def read_instrument_description_file(filename):
    """Reads instrument description file and creates instrument object

    :param filename: filename of instrument json file
    :type filename: Union[pathlib.WindowsPath, str]
    :return: instrument
    :rtype: Instrument
    """
    with open(filename) as json_file:
        data = json_file.read()

    directory = pathlib.Path(filename).parent.as_posix()
    return read_instrument_description(data, directory)


def read_instrument_description(json_data, directory):
    """Creates Instrument object from a instrument description

    :param json_data: json data
    :type json_data: str
    :param directory: directory of the instrument description file
    :type directory: str
    :return: instrument
    :rtype: Instrument
    """
    data = json.loads(json_data)
    schema_validator.validate(data)

    instrument_data = check(data, instrument_key, 'description')
    instrument_name = check(instrument_data, 'name', instrument_key, name=True)
    script = read_script_template(instrument_data, directory)
    gauge_volume = check(instrument_data, 'gauge_volume', instrument_key)
    positioners = read_positioners_description(instrument_data, directory)
    positioning_stacks = read_positioning_stacks_description(instrument_data, positioners)
    detectors = read_detector_description(instrument_data, positioners, directory)
    incident_jaw = read_jaw_description(instrument_data, positioners, directory)
    fixed_hardware = read_fixed_hardware_description(instrument_data, directory)

    return Instrument(instrument_name, gauge_volume, detectors, incident_jaw, positioners, positioning_stacks, script,
                      fixed_hardware)


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
        template_path = pathlib.Path(path).joinpath(template_name).as_posix()
        with open(template_path, 'r') as template_file:
            template = template_file.read()

    return Script(template)


def read_fixed_hardware_description(instrument_data, path=''):
    """Creates a dict of mesh objects from a fixed hardware description

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
    hardware_names = []
    for data in fixed_hardware_data:
        name = check(data, 'name', hardware_key, name=True)
        visuals = check(data, visual_key, hardware_key)
        mesh = read_visuals(visuals, path)
        fixed_hardware[name] = mesh
        hardware_names.append(name)

    duplicate_names = find_duplicates(hardware_names)
    if duplicate_names:
        raise ValueError(f'Fixed hardware has duplicate name(s): {duplicate_names}.')

    return fixed_hardware


def read_positioners_description(instrument_data, path=''):
    """Creates dict of Positioner objects from a positioners description

    :param instrument_data: instrument description
    :type instrument_data: Dict
    :param path: directory of the instrument description file
    :type path: str
    :return Positioner objects
    :rtype Dict[str, Positioner]
    """
    positioners = {}
    positioner_data = check(instrument_data, 'positioners', instrument_key)
    positioner_names = []
    for positioner in positioner_data:
        p = extract_positioner(positioner, path)
        positioners[p.name] = p
        positioner_names.append(p.name)

    duplicate_names = find_duplicates(positioner_names)
    if duplicate_names:
        raise ValueError(f'Positioners has duplicate name(s): {duplicate_names}.')

    return positioners


def extract_positioner(robot_data, path=''):
    """Creates positioner object from a positioner description

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

    links = {}
    link_names = []
    for link in links_data:
        link_name = check(link, 'name', 'link', name=True)
        link_names.append(link_name)
        links[link_name] = link

    duplicate_names = find_duplicates(link_names)
    if duplicate_names:
        raise ValueError(f'"{positioner_name}" has duplicate link name(s): {duplicate_names}.')

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
    joint_order = []
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
        home = joint.get('home_offset', (upper_limit + lower_limit) / 2)
        description = joint.get('description', '')
        if lower_limit > upper_limit:
            raise ValueError(f'lower limit ({lower_limit}) for "{joint_name}" is greater than upper ({upper_limit}).')
        if home > upper_limit or home < lower_limit:
            raise ValueError(
                f'default offset for "{joint_name}" is outside joint limits [{lower_limit}, {upper_limit}].')
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
        qv_links.append(
            Link(joint_name,
                 axis,
                 vector,
                 joint_type,
                 lower_limit,
                 upper_limit,
                 default_offset=home,
                 mesh=mesh,
                 description=description))
        joint_order.append(joint_name)

    duplicate_names = find_duplicates(joint_order)
    if duplicate_names:
        raise ValueError(f'"{positioner_name}" has duplicate joint name(s): {duplicate_names}.')

    base_link = links.get(base_link_name, None)
    if base_link is None:
        raise ValueError(f'"{base_link_name}" link definition not found. Did you misspell its name?')
    mesh = read_visuals(base_link.get(visual_key, None), path)

    if custom_order is not None:
        if len(set(custom_order)) != len(joint_order):
            raise ValueError(f'"{positioner_name}" custom_order attribute must contain all joints with no duplicates')
        diff = list(set(custom_order).difference(joint_order))
        if diff:
            raise ValueError(f'"{positioner_name}" custom_order attribute has incorrect joint name(s): {diff}.')

        custom_order = [joint_order.index(x) for x in custom_order]

    return SerialManipulator(positioner_name,
                             qv_links,
                             base=base_matrix,
                             tool=tool_matrix,
                             base_mesh=mesh,
                             custom_order=custom_order)


def read_positioning_stacks_description(instrument_data, positioners):
    """Extracts stack names and composition from a  positioning stacks description

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
    stack_names = []

    for stack in positioning_stacks_data:
        stack_name = check(stack, 'name', 'positioning_stacks', name=True)
        positioners_in_stack = [n.strip() for n in check(stack, 'positioners', 'positioning_stacks')]

        if stack_name in defined_positioners:
            if not (len(positioners_in_stack) == 1 and stack_name == positioners_in_stack[0]):
                raise ValueError(f'"{stack_name}" positioning stack name conflicts with a positioner of the same name.')

        duplicate_names = find_duplicates(positioners_in_stack)
        if duplicate_names:
            raise ValueError(f'In "{stack_name}" positioning stack, positioner(s) are duplicated: {duplicate_names}')

        undefined_positioners = set(positioners_in_stack).difference(defined_positioners)
        if undefined_positioners:
            if len(undefined_positioners) == 1:
                error = f'"{undefined_positioners.pop()}" positioner definition was not found'
            else:
                error = f'positioner definitions were not found for {undefined_positioners}'
            raise ValueError(f'In "{stack_name}" positioning stack, {error}.')

        positioning_stacks[stack_name] = positioners_in_stack
        stack_names.append(stack_name)

    duplicate_names = find_duplicates(stack_names)
    if duplicate_names:
        raise ValueError(f'Positioning stack has duplicate name(s): {duplicate_names}.')

    return positioning_stacks
