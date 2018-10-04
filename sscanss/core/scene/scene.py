from contextlib import suppress
from collections import OrderedDict
import numpy as np
from ..math.vector import Vector3
from ..util.misc import BoundingBox


class Scene:
    def __init__(self):
        self._data = OrderedDict()
        self.bounding_box = None

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

        if not np.any(np.isnan(max_pos)):
            bb_max = Vector3(max_pos)
            bb_min = Vector3(min_pos)
            center = (bb_max + bb_min) / 2
            radius = np.linalg.norm(bb_max - bb_min) / 2
            self.bounding_box = BoundingBox(bb_max, bb_min, center, radius)
        else:
            self.bounding_box = None

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
