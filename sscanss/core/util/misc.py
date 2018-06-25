from enum import Enum, unique

@unique
class Directions(Enum):
    right = '+X'
    left = '-X'
    up = '+Y'
    down = '-Y'
    front = '+Z'
    back = '-Z'

@unique
class SceneType(Enum):
    Sample = 1
    Instrument = 2


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


def to_float(string):
    try:
        return float(string), True
    except ValueError:
        return None, False
