from enum import Enum, unique
from collections import namedtuple
import numpy as np
from pyrr import Matrix44, Vector3
from .colour import Colour


BoundingBox = namedtuple('BoundingBox', ['max', 'min', 'center', 'radius'])


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
        self.bounding_box = None
        self.render_type = RenderType.Solid

        self.transform = Matrix44.identity()
        self.colour = Colour.black()

        self.children = []


def createSampleNode(samples):
    sample_node = Node()
    sample_node.colour = Colour(0.42, 0.42, 0.83)
    sample_node.render_type = RenderType.Solid

    max_pos = [np.nan, np.nan, np.nan]
    min_pos = [np.nan, np.nan, np.nan]
    for _, sample in samples.items():
        child = Node()
        child.vertices = sample.vertices
        child.indices = sample.indices
        child.normals = sample.normals
        child.bounding_box = sample.bounding_box
        child.colour = None
        child.render_type = None

        sample_node.children.append(child)

        max_pos = np.fmax(max_pos, np.max(child.vertices, axis=0))
        min_pos = np.fmin(min_pos, np.min(child.vertices, axis=0))

    bb_max = Vector3(max_pos)
    bb_min = Vector3(min_pos)
    center = Vector3(bb_max + bb_min) / 2
    radius = np.linalg.norm(bb_max - bb_min) / 2

    sample_node.bounding_box = BoundingBox(bb_max, bb_min, center, radius)

    return sample_node

