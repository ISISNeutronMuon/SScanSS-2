from enum import Enum, unique

@unique
class Direction(Enum):
    right = '+X'
    left = '-X'
    up = '+Y'
    down = '-Y'
    front = '+Z'
    back = '-Z'
