"""
A collection of miscellaneous functions
"""
from enum import Enum, unique, IntEnum

POINT_DTYPE = [('points', 'f4', 3), ('enabled', '?')]


@unique
class Directions(Enum):
    """Camera directions"""
    right = '+X'
    left = '-X'
    front = '+Y'
    back = '-Y'
    up = '+Z'
    down = '-Z'


@unique
class TransformType(Enum):
    """Methods for sample transformation"""
    Rotate = 'Rotate'
    Translate = 'Translate'
    Origin = 'Origin'
    Plane = 'Plane'
    Custom = 'Custom'


@unique
class Primitives(Enum):
    """Types of primitive"""
    Cuboid = 'Cuboid'
    Cylinder = 'Cylinder'
    Sphere = 'Sphere'
    Tube = 'Tube'


@unique
class DockFlag(Enum):
    """Flags for the dock widget """
    Upper = 1
    Bottom = 2
    Full = 3


@unique
class PointType(Enum):
    """Types of point data"""
    Fiducial = 'Fiducial'
    Measurement = 'Measurement'


@unique
class Attributes(Enum):
    """Objects in the project"""
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
    """Methods for computing strain components"""
    parallel_to_x = 'Parallel to X Axis'
    parallel_to_y = 'Parallel to Y Axis'
    parallel_to_z = 'Parallel to Z Axis'
    normal_to_surface = 'Normal to Surface'
    orthogonal_to_normal_no_x = 'Perpendicular to Surface Normal with zero X Component'
    orthogonal_to_normal_no_y = 'Perpendicular to Surface Normal with zero Y Component'
    orthogonal_to_normal_no_z = 'Perpendicular to Surface Normal with zero Z Component'
    custom = 'Key-in Vector'


@unique
class InsertSampleOptions(Enum):
    """Options for inserting sample to project"""
    Combine = 'Combine'
    Replace = 'Replace'


@unique
class LoadVector(Enum):
    """Flags to indicate the size of measurement vector
    when compared to points"""
    Exact = 1
    Smaller_than_points = 2
    Larger_than_points = 3


@unique
class MessageType(Enum):
    """Type of displayed message"""
    Information = 1
    Warning = 2
    Error = 3


@unique
class MessageReplyType(Enum):
    """Reply types for message box"""
    Save = 1
    Discard = 2
    Cancel = 3


@unique
class CommandID(IntEnum):
    """Unique ID for undoable commands"""
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
    """Plane options for cross section and transforms"""
    XY = 'XY plane'
    XZ = 'XZ plane'
    YZ = 'YZ plane'
    Custom = 'Custom Normal'


def to_float(string):
    """Converts a string to a float if possible otherwise returns None

    :param string: a string to convert to a float
    :type string: str
    :return: the float or None if conversion failed and a success flag
    :rtype: Union[Tuple[float, bool], Tuple[None, bool]]
    """
    try:
        return float(string), True
    except ValueError:
        return None, False


def toggle_action_in_group(action_name, action_group):
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


def compact_path(file_path, length):
    """Shortens a file path to a desired length by replacing excess letters
    in the middle with ellipsis. The new path will be invalid

    :param file_path: file path to shorten (min length: 6)
    :type file_path: str
    :param length: size of new path (min size: 5)
    :type length: int
    :return: shortened file path
    :rtype: str
    """
    if length < 5:
        raise ValueError('length must be more than 3')

    if len(file_path) <= length:
        return file_path

    length -= 3
    left = length // 2
    right = length - left

    return f'{file_path[:left]}...{file_path[-right:]}'


def find_duplicates(seq):
    """Finds duplicates in an iterable.

    :param seq: iterable to check for duplicates
    :type seq: Iterable
    :return: values in the iterable that occurs more than once
    :rtype: List[Any]
    """
    seen = set()
    seen_twice = {x: '' for x in seq if x in seen or seen.add(x)}
    return list(seen_twice)
