from enum import Enum, unique
import numpy as np
from pyrr import Matrix44
from .colour import Colour

@unique
class RenderType(Enum):
    Solid = 'Solid'
    Wireframe = 'Wireframe'
    Transparent = 'Transparent'

class Node:
    def __init__(self):
        self.vertices = np.array([])
        self.indices = np.array([])
        self.normals = np.array([])

        self.render_type = RenderType.Solid

        self.transform = Matrix44.identity()
        self.colour = Colour.black()

        self.children = []

