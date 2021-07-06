"""
Class for Scene object
"""
from contextlib import suppress
from collections import OrderedDict
from enum import unique, Enum
import numpy as np
from .camera import Camera
from .node import Node
from .entity import InstrumentEntity
from ..util.misc import Attributes
from ..geometry.mesh import BoundingBox


def validate_instrument_scene_size(instrument):
    """Checks that the instrument scene size is within maximum extents

    :param instrument: instrument object
    :type instrument: Instrument
    :return: indicates if scene size is valid
    :rtype: bool
    """
    s = Scene()
    node = Node()
    node.vertices = InstrumentEntity(instrument).vertices
    s.addNode('', node)
    return not s.invalid


class Scene:
    """Creates Scene object

    :param scene_type: scene type
    :type scene_type: Scene.Type
    """
    @unique
    class Type(Enum):
        Sample = 1
        Instrument = 2

    max_extent = 5000000  # cap for scene radius in mm
    sample_render_mode = Node.RenderMode.Solid

    def __init__(self, scene_type=Type.Sample):
        self._data = OrderedDict()
        self.bounding_box = None
        self.type = scene_type
        if scene_type == Scene.Type.Sample:
            self.camera = Camera(1.0, 60)
        else:
            self.camera = Camera(1.0, 60,  [-0.577, -0.577, -0.577], [0.0, 0.0, 1.0])
        self.invalid = False
        self.extent = 0.0

    @property
    def nodes(self):
        nodes = self._data.values()
        if Scene.sample_render_mode == Node.RenderMode.Transparent:
            nodes = reversed(nodes)
        return list(nodes)

    def addNode(self, key, node):
        """Adds a non-empty node to the scene

        :param key: name of node
        :type key: Any
        :param node: node
        :type node: Node
        """
        if node.isEmpty():
            self.removeNode(key)
            return

        self._data[key] = node
        # Ensures that the sample is drawn last so transparency is rendered properly
        if Attributes.Sample in self._data:
            self._data.move_to_end(Attributes.Sample, False)
        self.updateBoundingBox()

    def removeNode(self, key):
        """Removes specified node from the scene

        :param key: key of the node to remove
        :type key: Any
        """
        with suppress(KeyError):
            del self._data[key]
            self.updateBoundingBox()

    def updateBoundingBox(self):
        """Recalculates the bounding box after a node is added or removed"""

        if self.isEmpty():
            self.bounding_box = None
            self.extent = 0.0
            return

        self.bounding_box = BoundingBox.merge([node.bounding_box for node in self.nodes])
        self.extent = self.bounding_box.center.length + self.bounding_box.radius
        if not np.isfinite(self.extent) or self.extent > self.max_extent:
            self.invalid = True
        else:
            self.invalid = False

    def isEmpty(self):
        """Checks if Scene is empty

        :return: indicates scene is empty
        :rtype: bool
        """
        if self._data:
            return False
        return True

    def __contains__(self, key):
        if key in self._data:
            return True
        return False

    def __getitem__(self, key):
        return self._data.get(key, Node())
