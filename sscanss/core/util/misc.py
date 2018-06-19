from enum import Enum, unique

@unique
class Direction(Enum):
    right = '+X'
    left = '-X'
    up = '+Y'
    down = '-Y'
    front = '+Z'
    back = '-Z'


@unique
class Primitives(Enum):
    Cuboid = 'Cuboid'
    Cylinder = 'Cylinder'
    Sphere = 'Sphere'
    Tube = 'Tube'


@unique
class Compare(Enum):
    Equal = 1
    Not_Equal = 2
    Greater = 3
    Less = 4


def to_float(string):
    try:
        return float(string), True
    except ValueError:
        return None, False
