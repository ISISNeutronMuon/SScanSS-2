import os
import json
import numpy as np
import sscanss.instruments
from .robotics import Link, SerialManipulator
from ..io.reader import read_3d_model
from ..math.vector import Vector3
from ..math.matrix import Matrix44
from ..math.transform import matrix_from_xyz_eulers
from ..mesh.colour import Colour
from ..scene import Node


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
        except:
            continue

        instrument_data = data.get('instrument', None)
        if instrument_data is None:
            continue

        name = instrument_data.get('name', '').strip().upper()
        if name:
            instruments[name] = idf

    return instruments
    # file_path = 'C:/Users/osu27944/Documents/Development/sscanss/labs/labs_ENGINX/'
    # lab_files = file_path)
    #
    #     mesh = {}
    #     try:
    #         name = os.path.splitext(filename)[0]
    #         mesh = read_model_file(file_path + filename)
    #         write_binary_stl(out_path.format(name), mesh)
    #     except:
    #         pass



def matrix_from_pose(pose, angles_in_degrees=True):
    matrix = Matrix44.identity()

    position = pose[0:3]
    orientation = np.radians(pose[3:6]) if angles_in_degrees else pose[3:6]

    matrix[0:3, 0:3] = matrix_from_xyz_eulers(orientation)
    matrix[0:3, 3] = position

    return matrix


class Instrument:
    def __init__(self, name):
        self.name = name
        self.detectors = None
        self.fixed_positioner = None
        self.auxillary_positioners = []
        self.positioners = None
        self.jaws = None

        self.beam_guide = None
        self.beam_stop = None

    def model(self):
        node = Node()

        node.addChild(self.positioners[self.fixed_positioner].model())
        node.addChild(self.detectors['South'][2].model())
        node.addChild(self.detectors['North'][2].model())
        node.addChild(self.jaws.model())
        node.addChild(Node(self.beam_guide))
        node.addChild(Node(self.beam_stop))
        return node


class Collimator:
    def __init__(self, size, aperture, mesh):
        self.size = size
        self.aperture = aperture
        self.mesh = mesh

    def model(self):
        return Node(self.mesh)


def read_jaw_description(jaws, guide, stop, positioner):
    aperture = jaws['aperture']
    s = positioner[jaws['positioner']]
    mesh_1 = read_visuals(guide.get('visual', None))
    mesh_2 = read_visuals(stop.get('visual', None))

    return s, mesh_1, mesh_2


def read_detector_description(detector_data, collimator_data):

    detectors = {detector['name']: [] for detector in detector_data}

    for collimator in collimator_data:
        size = collimator['size']
        detector_name = collimator['detector']
        aperture = [collimator['aperture_h'], collimator['aperture_v']]
        mesh = read_visuals(collimator.get('visual', None))

        detectors[detector_name].append(Collimator(size, aperture, mesh))

    return detectors


DEFAULT_POSE = [0., 0., 0., 0., 0., 0.]
DEFAULT_COLOUR = [0., 0., 0.]


def read_visuals(visuals_data):
    if visuals_data is None:
        return None

    pose = visuals_data.get('pose', DEFAULT_POSE)
    pose = matrix_from_pose(pose)
    mesh_colour = visuals_data.get('colour', DEFAULT_COLOUR)

    mesh_filename = required(visuals_data, 'mesh', '')
    mesh, *ignore = read_3d_model(mesh_filename)
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

    instrument_data = required(data, 'instrument', '')

    instrument_name = instrument_data.get('name', 'UNKNOWN')
    positioner_data = required(instrument_data, 'positioners', '')

    positioners = {}
    for positioner in positioner_data:
        p = read_positioner_description(positioner)
        positioners[p[0]] = p[1]

    fixed_positioner = None  # currently only support a single fixed positioner
    auxillary_positioners = []
    sample_positioners = required(instrument_data, 'sample_positioners', '')
    for positioner in sample_positioners:
        if positioner['movable']:
            auxillary_positioners.append(positioner['positioner'])
        else:
            fixed_positioner = positioner['positioner']

    detector_data = required(instrument_data, 'detectors', '')
    collimator_data = required(instrument_data, 'collimators', '')

    detectors = read_detector_description(detector_data, collimator_data)

    jaw = required(instrument_data, 'incident_jaws', '')
    beam_guide = required(instrument_data, 'beam_guide', '')
    beam_stop = required(instrument_data, 'beam_stop', '')

    e = read_jaw_description(jaw, beam_guide, beam_stop, positioners)

    instrument = Instrument(instrument_name)
    instrument.beam_guide = e[1]
    instrument.beam_stop = e[2]
    instrument.jaws = e[0]
    instrument.detectors = detectors
    instrument.fixed_positioner = fixed_positioner
    instrument.auxillary_positioners = auxillary_positioners
    instrument.positioners = positioners

    return instrument


def read_positioner_description(robot_data):
    default_pose = [0., 0., 0., 0., 0., 0.]
    positioner_name = required(robot_data, 'name', '')
    base_pose = robot_data.get('base', default_pose)
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
        _type = required(joint, 'type', '')
        if _type == 'revolute':
            joint_type = Link.Type.Revolute
        elif _type == 'prismatic':
            joint_type = Link.Type.Prismatic
        else:
            raise ValueError('Invalid joint type for {}'.format(name))

        mesh = read_visuals(link.get('visual', None))

        tmp = Link(axis, vector, joint_type, upper_limit=upper_limit, lower_limit=lower_limit,
                   mesh=mesh, name=name)
        qv_links.append(tmp)

    base_link = links[base_link_name]
    mesh = read_visuals(base_link.get('visual', None))

    return positioner_name, SerialManipulator(qv_links, base=base_matrix, base_mesh=mesh)