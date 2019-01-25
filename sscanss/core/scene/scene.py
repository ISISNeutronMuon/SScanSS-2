from contextlib import suppress
from collections import OrderedDict
from enum import unique, Enum
import numpy as np
from .camera import Camera
from ..util.misc import Attributes
from ..mesh.utility import BoundingBox


class Scene:
    @unique
    class Type(Enum):
        Sample = 1
        Instrument = 2

    sample_key = 'sample'

    def __init__(self, scene_type=Type.Sample):
        """Creates Scene object used in the GL_Widget

        :param scene_type: scene type
        :type scene_type: sscanss.core.scene.scene.Type
        """
        self._data = OrderedDict()
        self.bounding_box = None
        self.type = scene_type
        self.camera = Camera(1.0, 60)

    @property
    def nodes(self):
        return list(self._data.values())

    def addNode(self, key, node):
        """Adds a non-empty node to the scene

        :param key: name of node
        :type key: [str, Enum]
        :param node: node
        :type node: sscanss.core.scene.node.Node
        """
        if node.isEmpty():
            self.removeNode(key)
            return

        self._data[key] = node
        # Ensures that the sample is drawn last so transparency is rendered properly
        if Attributes.Sample in self._data:
            self._data.move_to_end(Attributes.Sample)
        self.updateBoundingBox()

    def removeNode(self, key):
        """remove specified node from the scene

        :param key: key of the node to remove
        :type key: [str, Enum]
        """
        with suppress(KeyError):
            del self._data[key]
            self.updateBoundingBox()

    def updateBoundingBox(self):
        """ recalculates the bounding box after a node is added or removed"""
        max_pos = [np.nan, np.nan, np.nan]
        min_pos = [np.nan, np.nan, np.nan]

        for node in self.nodes:
            max_pos = np.fmax(max_pos, node.bounding_box.max)
            min_pos = np.fmin(min_pos, node.bounding_box.min)

        if np.any(np.isnan([max_pos, min_pos])):
            self.bounding_box = None
        else:
            self.bounding_box = BoundingBox(max_pos, min_pos)

    def isEmpty(self):
        """empty state of scene

        :return: True if scene is empty
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
        return self._data[key]
