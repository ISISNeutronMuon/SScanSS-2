from ..scene.node import Node


class Instrument:
    def __init__(self, name, detectors, jaws, positioners, fixed_positioner,
                 aux_positioner=None, beam_guide=None, beam_stop=None):
        self.name = name

        self.detectors = detectors
        self.fixed_positioner = fixed_positioner
        self.auxiliary_positioners = [] if aux_positioner is None else aux_positioner
        self.positioners = positioners
        self.jaws = jaws

        self.beam_guide = beam_guide
        self.beam_stop = beam_stop

        fixed = self.positioners[fixed_positioner]
        self.positioning_stack = PositioningStack(fixed)

    def model(self):
        node = Node()

        node.addChild(self.positioning_stack.model())
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
    def __init__(self, name, aperture, lower_limit, upper_limit, positioner=None):
        self.name = name
        self.aperture = aperture
        self.aperture_lower_limit = lower_limit
        self.aperture_upper_limit = upper_limit
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


class PositioningStack:
    def __init__(self, fixed, aux=None):
        self.fixed = fixed
        self.fixed.reset()
        if aux is None:
            self.auxiliary = []
            self.link_matrix = []
        else:
            aux.reset()
            self.auxiliary = [aux]
            self.link_matrix = [self.fixed.pose.inverse()]

    def __calculateFixedLinks(self):
        self.link_matrix = []
        q = self.configuration
        self.fixed.reset()
        self.link_matrix.append(self.fixed.pose.inverse())
        for i in range(len(self.auxiliary)-1):
            aux = self.auxiliary[i]
            aux.reset()
            self.link_matrix.append(aux.pose.inverse())

        self.fkine(q)

    def changeBaseMatrix(self, positioner, matrix=None, reset=False):
        for aux in self.auxiliary:
            if aux is not positioner:
                continue
            if reset:
                aux.base = aux.default_base
            elif matrix is not None:
                aux.base = matrix
            self.__calculateFixedLinks()
            break

    def removePositioner(self, positioner):
        positioner.base = positioner.default_base
        self.auxiliary.remove(positioner)
        self.__calculateFixedLinks()

    def addPositioner(self, positioner):
        positioner.reset()
        q = self.fixed.configuration
        self.fixed.reset()
        m = self.fixed.pose.inverse()
        self.fixed.fkine(q)
        self.auxiliary.append(positioner)
        self.link_matrix.append(m)

    @property
    def configuration(self):
        conf = []
        conf.extend(self.fixed.configuration)
        for positioner in self.auxiliary:
            conf.extend(positioner.configuration)

        return conf

    @property
    def numberOfLinks(self):
        number = self.fixed.numberOfLinks
        for positioner in self.auxiliary:
            number += positioner.numberOfLinks

        return number

    def fkine(self, q):
        start, end = 0, self.fixed.numberOfLinks
        T = self.fixed.fkine(q[start:end])
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            start, end = end, end + positioner.numberOfLinks
            T *= link * positioner.fkine(q[start:end])

        return T

    def model(self):
        node = self.fixed.model()
        matrix = self.fixed.pose
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            matrix *= link
            node.addChild(positioner.model(matrix))
            matrix *= positioner.pose

        return node