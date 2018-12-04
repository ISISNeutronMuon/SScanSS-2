from ..scene.node import Node


class Instrument:
    def __init__(self, name, detectors, jaws, positioners, positioning_stacks, beam_guide=None, beam_stop=None):
        self.name = name
        self.detectors = detectors
        self.positioners = positioners
        self.jaws = jaws
        self.beam_guide = beam_guide
        self.beam_stop = beam_stop
        self.positioning_stacks = positioning_stacks
        self.loadPositioningStack(list(self.positioning_stacks.keys())[0])

    def loadPositioningStack(self, stack_key):
        positioner_keys = self.positioning_stacks[stack_key]

        for i in range(len(positioner_keys)):
            key = positioner_keys[i]
            if i == 0:
                self.positioning_stack = PositioningStack(stack_key, self.positioners[key])
            else:
                self.positioning_stack.addPositioner(self.positioners[key])

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
    def __init__(self, name, fixed):
        self.name = name
        self.fixed = fixed
        self.fixed.reset()
        self.auxiliary = []
        self.link_matrix = []

    def __calculateFixedLink(self, positioner):
        q = positioner.set_points
        positioner.resetOffsets()
        matrix = positioner.pose.inverse()
        positioner.fkine(q, ignore_locks=True)

        return matrix

    def changeBaseMatrix(self, positioner, matrix):
        index = self.auxiliary.index(positioner)
        positioner.base = matrix

        if positioner is not self.auxiliary[-1]:
            self.link_matrix[index+1] = self.__calculateFixedLink(positioner)

    def addPositioner(self, positioner):
        positioner.reset()
        last_positioner = self.auxiliary[-1] if self.auxiliary else self.fixed
        self.auxiliary.append(positioner)
        self.link_matrix.append(self.__calculateFixedLink(last_positioner))

    @property
    def configuration(self):
        conf = []
        conf.extend(self.fixed.configuration)
        for positioner in self.auxiliary:
            conf.extend(positioner.configuration)

        return conf

    @property
    def links(self):
        links = []
        links.extend(self.fixed.links)
        for positioner in self.auxiliary:
            links.extend(positioner.links)

        return links

    @property
    def numberOfLinks(self):
        number = self.fixed.numberOfLinks
        for positioner in self.auxiliary:
            number += positioner.numberOfLinks

        return number

    def fkine(self, q, ignore_locks=False, setpoint=True):
        start, end = 0, self.fixed.numberOfLinks
        T = self.fixed.fkine(q[start:end], ignore_locks=ignore_locks, setpoint=setpoint)
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            start, end = end, end + positioner.numberOfLinks
            T *= link * positioner.fkine(q[start:end], ignore_locks=ignore_locks, setpoint=setpoint)

        return T

    def model(self):
        node = self.fixed.model()
        matrix = self.fixed.pose
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            matrix *= link
            node.addChild(positioner.model(matrix))
            matrix *= positioner.pose

        return node

    @property
    def set_points(self):
        set_points = []
        set_points.extend(self.fixed.set_points)
        for positioner in self.auxiliary:
            set_points.extend(positioner.set_points)

        return set_points

    @set_points.setter
    def set_points(self, q):
        for offset, link in zip(q, self.links):
            link.set_point = offset