"""
A collection of miscellaneous functions
"""
from enum import Enum, unique, IntEnum

POINT_DTYPE = [('points', 'f4', 3), ('enabled', '?')]


@unique
class Anchor(Enum):
    """Anchor positions"""
    Center = 'Center'
    TopLeft = 'Top Left'
    TopRight = 'Top Right'
    BottomLeft = 'Bottom Left'
    BottomRight = 'Bottom Right'


@unique
class Directions(Enum):
    """Camera directions"""
    Right = '+X'
    Left = '-X'
    Front = '+Y'
    Back = '-Y'
    Up = '+Z'
    Down = '-Z'


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
    Cone = 'Cone'
    Cuboid = 'Cuboid'
    Cylinder = 'Cylinder'
    Sphere = 'Sphere'
    Tube = 'Tube'


@unique
class VisualGeometry(Enum):
    """Types of geometry for instrument visuals"""
    Box = 'Box'
    Plane = 'Plane'
    Sphere = 'Sphere'
    Mesh = 'Mesh'


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
    Volume = 'Volume'


@unique
class StrainComponents(Enum):
    """Methods for computing strain components"""
    ParallelX = 'Parallel to X Axis'
    ParallelY = 'Parallel to Y Axis'
    ParallelZ = 'Parallel to Z Axis'
    SurfaceNormal = 'Normal to Surface'
    OrthogonalWithoutX = 'Perpendicular to Surface Normal with zero X Component'
    OrthogonalWithoutY = 'Perpendicular to Surface Normal with zero Y Component'
    OrthogonalWithoutZ = 'Perpendicular to Surface Normal with zero Z Component'
    Custom = 'Key-in Vector'


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
    Smaller = 2
    Larger = 3


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
    InsertVolumeFromFile = 1000
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
    :return: the float or None if conversion failed
    :rtype: Optional[float]
    """
    try:
        return float(string)
    except ValueError:
        return None


def toggle_action_in_group(action_name, action_group):
    """Checks/Toggles the action with a specified name in an action group

    :param action_name: name of Action
    :type action_name: str
    :param action_group: action group
    :type action_group: PyQt6.QtGui.QActionGroup
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
