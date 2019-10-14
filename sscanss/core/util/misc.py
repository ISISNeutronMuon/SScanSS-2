"""
A collection of miscellaneous functions
"""
from enum import Enum, unique, IntEnum


POINT_DTYPE = [('points', 'f4', 3), ('enabled', '?')]


@unique
class Directions(Enum):
    right = '+X'
    left = '-X'
    front = '+Y'
    back = '-Y'
    up = '+Z'
    down = '-Z'


@unique
class TransformType(Enum):
    Rotate = 'Rotate'
    Translate = 'Translate'
    Origin = 'Origin'
    Plane = 'Plane'
    Custom = 'Custom'


@unique
class Primitives(Enum):
    Cuboid = 'Cuboid'
    Cylinder = 'Cylinder'
    Sphere = 'Sphere'
    Tube = 'Tube'


@unique
class DockFlag(Enum):
    Upper = 1
    Bottom = 2
    Full = 3


@unique
class PointType(Enum):
    Fiducial = 'Fiducial'
    Measurement = 'Measurement'


@unique
class Attributes(Enum):
    Sample = 'Sample'
    Fiducials = 'Fiducials '
    Measurements = 'Measurements'
    Vectors = 'Vectors'
    Instrument = 'Instrument'
    Plane = 'Plane'
    Beam = 'Beam'
    Positioner = 'Positioner'
    Detector = 'Detector'
    Jaws = 'Jaws'
    Fixture = 'fixture'


@unique
class StrainComponents(Enum):
    parallel_to_x = 'Parallel to X Axis'
    parallel_to_y = 'Parallel to Y Axis'
    parallel_to_z = 'Parallel to Z Axis'
    normal_to_surface = 'Normal to Surface'
    orthogonal_to_normal_no_x = 'Perpendicular to Surface Normal with zero X Component'
    orthogonal_to_normal_no_y = 'Perpendicular to Surface Normal with zero Y Component'
    orthogonal_to_normal_no_z = 'Perpendicular to Surface Normal with zero Z Component'
    custom = 'Key-in Vector'


@unique
class LoadVector(Enum):
    Exact = 1
    Smaller_than_points = 2
    Larger_than_points = 3


@unique
class MessageSeverity(Enum):
    Information = 1
    Warning = 2
    Critical = 3


@unique
class CommandID(IntEnum):
    ChangeMainSample = 1000
    MovePoints = 1001
    EditPoints = 1002
    LockJoint = 1003
    IgnoreJointLimits = 1004
    MovePositioner = 1005
    ChangePositioningStack = 1006
    ChangePositionerBase = 1007
    ChangeCollimator = 1008
    ChangeJawAperture = 1009
    AlignSample = 1010


@unique
class PlaneOptions(Enum):
    XY = 'XY plane'
    XZ = 'XZ plane'
    YZ = 'YZ plane'
    Custom = 'Custom Normal'


def to_float(string):
    """ Converts a string to a float if possible otherwise returns None

    :param string: a string to convert to a float
    :type string: str
    :return: the float or None if conversion failed and a success flag
    :rtype: Union[Tuple[float, bool], Tuple[None, bool]]
    """
    try:
        return float(string), True
    except ValueError:
        return None, False


def toggleActionInGroup(action_name, action_group):
    """Checks/Toggles the action with a specified name in an action group

    :param action_name: name of Action
    :type action_name: str
    :param action_group: action group
    :type action_group: PyQt5.QtWidgets.QActionGroup
    """
    actions = action_group.actions()
    for action in actions:
        if action.text() == action_name:
            action.setChecked(True)
            break
