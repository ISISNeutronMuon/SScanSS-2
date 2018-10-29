from ..scene.node import Node


class Instrument:
    def __init__(self, name):
        self.name = name

        self.detectors = {}
        self.fixed_positioner = None
        self.auxillary_positioners = []
        self.positioners = None
        self.jaws = None

        self.beam_guide = None
        self.beam_stop = None

    def model(self):
        node = Node()

        node.addChild(self.positioners[self.fixed_positioner].model())
        for _, detector in self.detectors.items():
            active_collimator = detector.current_collimator
            if active_collimator is None:
                continue
            node.addChild(active_collimator.model())

        node.addChild(self.jaws.model())
        node.addChild(Node(self.beam_guide))
        node.addChild(Node(self.beam_stop))
        return node


class Jaws:
    def __init__(self, name, aperture, positioner=None):
        self.name = name
        self.aperture = aperture
        self.positioner = positioner

    @property
    def axes(self):
        return self.positioner.links

    def move(self, q):
        self.positioner.fkine(q)

    def model(self):
        return self.positioner.model()


class Detector:
    def __init__(self, name):
        self.name = name
        self.__current_collimator = None
        self.collimators = {}

    @property
    def current_collimator(self):
        return self.__current_collimator

    @current_collimator.setter
    def current_collimator(self, key):
        if key in self.collimators:
            self.__current_collimator = self.collimators[key]
        else:
            self.__current_collimator = None


class Collimator:
    def __init__(self, name, aperture, mesh):
        self.name = name
        self.aperture = aperture
        self.mesh = mesh

    def model(self):
        return Node(self.mesh)
