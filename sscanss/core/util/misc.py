from enum import Enum, unique


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
    Custom = 'Custom'


@unique
class Primitives(Enum):
    Cuboid = 'Cuboid'
    Cylinder = 'Cylinder'
    Sphere = 'Sphere'
    Tube = 'Tube'


@unique
class CompareOperator(Enum):
    Equal = 1
    Not_Equal = 2
    Greater = 3
    Less = 4


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


def to_float(string):
    """ Converts a string to a float if possible otherwise returns None

    :param string: a string to convert to a float
    :type string: str
    :return: the float or None if conversion failed and a success flag
    :rtype: Union[Tuple[float, bool], Tuple[NoneType, bool]]
    """
    try:
        return float(string), True
    except ValueError:
        return None, False
