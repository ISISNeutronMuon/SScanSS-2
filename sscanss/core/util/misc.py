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
class SceneType(Enum):
    Sample = 1
    Instrument = 2


@unique
class TransformType(Enum):
    Rotate = 'Rotate'
    Translate = 'Translate'


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


def clamp(value, min_value=0.0, max_value=1.0):
    """ returns original value if it is between a minimum and maximum value
    returns minimum value if original value is less than minimum value
    returns maximum value if original value is greater than maximum value

    :param value: number to clamp
    :type value: numbers.Number
    :param min_value: maximum value
    :type min_value: numbers.Number
    :param max_value: minimum value
    :type max_value: numbers.Number
    :return: number clamped between the specified range
    :rtype: numbers.Number
    """
    return max(min(value, max_value), min_value)
