from contextlib import suppress
from collections import OrderedDict
from enum import unique, Enum
import numpy as np
from ..mesh.utility import BoundingBox


class Scene:
    @unique
    class Type(Enum):
        Sample = 1
        Instrument = 2

    def __init__(self, scene_type=Type.Sample):
        self._data = OrderedDict()
        self.bounding_box = None
        self.type = scene_type

    @property
    def nodes(self):
        return list(self._data.values())

    def addNode(self, key, node):
        if node.isEmpty():
            self.removeNode(key)
            return

        self._data[key] = node
        # Ensures that the sample is drawn last so transparency is rendered properly
        self._data.move_to_end('sample')
        self.updateBoundingBox()

    def removeNode(self, key):
        with suppress(KeyError):
            del self._data[key]
            self.updateBoundingBox()

    def updateBoundingBox(self):
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
        if self._data == 0:
            return True
        return False

    def __contains__(self, key):
        if key in self._data:
            return True
        return False

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        return self.nodes[key]