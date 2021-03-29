import math
import numpy as np
from .robotics import Link, SerialManipulator
from ..math.matrix import Matrix44
from ..math.structure import fit_circle_3d, fit_line_3d
from ..math.transform import angle_axis_to_matrix, xyz_eulers_from_matrix


def correct_circle_axis(axis, center, points, offsets):
    """Corrects axis of the circle by analysing the direction of rotation so it is
    consistent with the right hand rule.

    :param axis: estimated circle axis
    :type axis: numpy.ndarray
    :param center: estimated circle center
    :type center: numpy.ndarray
    :param points: measured point on the circle
    :type points: numpy.ndarray
    :param offsets: angular offset for each measured point
    :type offsets: numpy.ndarray
    :return: corrected axis
    :rtype: numpy.ndarray
    """
    # This could be wrong if point 0 or 2 have significant error
    increasing_order = offsets[2] > offsets[0]

    a_diff = points[0] - center
    b_diff = points[2] - center
    a = a_diff / np.linalg.norm(a_diff)
    b = b_diff / np.linalg.norm(b_diff)

    if not increasing_order:
        a, b = b, a

    if np.dot(np.cross(a, b), axis) <= 0.0:
        axis = -axis
    return axis


def correct_line(axis, center, points, offsets):
    """Corrects axis of the line so it is consistent with the offsets.

    :param axis: estimated line axis
    :type axis: numpy.ndarray
    :param center: estimated line center
    :type center: numpy.ndarray
    :param points: measured point on the line
    :type points: numpy.ndarray
    :param offsets: offset for each measurement
    :type offsets: numpy.ndarray
    :return: corrected center and axis
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    avg = np.mean(offsets)
    adj_offsets = offsets - avg
    index = 0 if adj_offsets[0] != 0 else 2
    a = (center + axis * adj_offsets[index]) - points[index]
    b = (center - axis * adj_offsets[index]) - points[index]
    if np.linalg.norm(a) > np.linalg.norm(b):
        axis = -axis

    center = center - axis * avg

    return center, axis


class CPAResult:
    """Data class for the Circle Point Analysis (CPA) result"""
    def __init__(self, joint_axes, joint_origins, base, tool, fit_errors, model_errors):
        self.joint_axes = joint_axes
        self.joint_origins = joint_origins
        self.base = base
        self.tool = tool
        self.model_errors = model_errors
        self.fit_errors = fit_errors


def circle_point_analysis(data, joint_types, joint_offsets, joint_homes):
    """Implements the circle point analysis algorithm for robot calibration.

    :param data: measured 3D points for each joint
    :type data: List[numpy.ndarray]
    :param joint_types: types of each joint
    :type joint_types: List[Link.Type]
    :param joint_offsets: measured offsets for each measurement
    :type joint_offsets: List[numpy.ndarray]
    :param joint_homes: home position for each measurement
    :type joint_homes: List[float]
    :return: calibration result
    :rtype: CPAResult
    """
    number_of_joints = len(joint_types)

    # Fitting models to measured data
    joint_axes = []
    joint_origins = []
    base = None
    fit_errors = []
    for i in range(number_of_joints):
        axis_data = data[i]
        offsets = joint_offsets[i]
        if joint_types[i] == Link.Type.Revolute:
            center, axis, _, res = fit_circle_3d(axis_data)
            axis = correct_circle_axis(axis, center, axis_data, offsets)
        else:
            center, axis, res = fit_line_3d(axis_data)
            center, axis = correct_line(axis, center, axis_data, offsets)

        base = center if base is None else base
        joint_axes.append(axis)
        joint_origins.append(center - base)
        fit_errors.append(res)

    links = []
    for i in range(number_of_joints):
        center = joint_origins[i]
        axis = joint_axes[i]
        j = i if i + 1 == number_of_joints else i + 1
        next_center = joint_origins[j]
        links.append(Link('', axis, next_center - center, joint_types[i], np.min(joint_offsets[i]),
                          np.max(joint_offsets[i]), joint_homes[i]))

    base_matrix = Matrix44.identity()
    base_matrix[:3, 3] = base
    s = SerialManipulator('', links, base=base_matrix)

    if joint_types[-1] == Link.Type.Revolute:
        angle = math.radians(joint_homes[-1] - joint_offsets[-1][0])
        point = data[-1][0]
        axis = joint_axes[-1]
        center = joint_origins[-1]
        rot_matrix = angle_axis_to_matrix(angle, axis)
        tool = (rot_matrix @ (point - center - base))
        tool_matrix = Matrix44.identity()
        tool_matrix[:3, 3] = tool
        s.tool = tool_matrix

    model_errors = []
    for i in range(number_of_joints):
        q = np.zeros(number_of_joints)
        axis_data = data[i]
        offsets = joint_offsets[i]
        error = []
        for j in range(len(axis_data)):
            q[i] = math.radians(offsets[j]) if joint_types[i] == Link.Type.Revolute else offsets[j]
            matrix = s.fkine(q)
            error.append(axis_data[j, :] - matrix[:3, 3].transpose())
        model_errors.append(np.vstack(error))

    return CPAResult(joint_axes, joint_origins, s.base, s.tool, fit_errors, model_errors)


def generate_description(robot_name, base, tool, order, joint_names, joint_types, joint_axes, joint_origins,
                         joint_homes, offsets):
    """Generates a description of the robot which can be written into the instrument description file

    :param robot_name: name of robot
    :type robot_name: str
    :param base: base matrix
    :type base: Matrix44
    :param tool: tool matrix
    :type tool: Matrix44
    :param order: custom order of joints
    :type order: List[int]
    :param joint_names: list of joint names
    :type joint_names: List[str]
    :param joint_types: list of joint types
    :type joint_types: List[Link.Type]
    :param joint_axes: list of joint axes
    :type joint_axes: List[numpy.ndarray]
    :param joint_origins: list of joint origins
    :type joint_origins: List[numpy.ndarray]
    :param joint_homes: list of joint homes
    :type joint_homes: List[float]
    :param offsets: list of joint offsets
    :type offsets: List[numpy.ndarray]
    :return: robot description
    :rtype: Dict
    """
    custom_order = [joint_names[i] for i in order]
    link_names = ['base', *[f'link_{name.replace(" ", "_").lower()}' for name in joint_names]]
    robot_json = {'name': robot_name, 'base': [*base[:3, 3], *np.degrees(xyz_eulers_from_matrix(base[:3, :3]))],
                  'tool': [*tool[:3, 3], *np.degrees(xyz_eulers_from_matrix(tool[:3, :3]))],
                  "custom_order": custom_order, 'joints': [], 'links': []}
    joints = robot_json['joints']
    links = robot_json['links']
    links.append({"name": link_names[0]})

    for index in range(len(joint_axes)):
        next_link = link_names[index + 1]
        temp = {'name': joint_names[index],
                'type': joint_types[index].value,
                'axis': joint_axes[index].tolist(),
                "home_offset": float(joint_homes[index]),
                'origin': joint_origins[index].tolist(),
                'lower_limit': float(offsets[index].min()),
                'upper_limit': float(offsets[index].max()),
                'parent': link_names[index],
                'child': next_link}

        links.append({"name": next_link})
        joints.append(temp)

    return robot_json
