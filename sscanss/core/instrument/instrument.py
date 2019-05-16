from enum import Enum, unique
import pystache
from .robotics import IKSolver
from ..scene.node import Node


class Instrument:
    def __init__(self, name, detectors, jaws, positioners, positioning_stacks, script_template,
                 beam_guide=None, beam_stop=None):
        """

        :param name: name of instrument
        :type name: str
        :param detectors: detectors
        :type detectors: Dict[str, sscanss.core.instrument.instrument.Detector]
        :param jaws: jaws
        :type jaws: sscanss.core.instrument.instrument.Jaws
        :param positioners: positioners
        :type positioners: Dict[str, sscanss.core.instrument.robotics.SerialManipulator]
        :param positioning_stacks: positioning stacks
        :type positioning_stacks: Dict[str, List[str]]
        :param beam_guide: mesh of beam guide
        :type beam_guide: sscanss.core.mesh.utility.Mesh
        :param beam_stop: mesh of beam stop
        :type beam_stop: sscanss.core.mesh.utility.Mesh
        """
        self.name = name
        self.detectors = detectors
        self.positioners = positioners
        self.jaws = jaws
        self.beam_guide = beam_guide
        self.beam_stop = beam_stop
        self.positioning_stacks = positioning_stacks
        self.script_template = script_template
        self.loadPositioningStack(list(self.positioning_stacks.keys())[0])

        self.sample = None

    def getPositioner(self, name):
        """ get positioner or positioning stack by name

        :param name: name of positioner or stack
        :type name: str
        :return: positioner or positioning stack
        :rtype: Union[sscanss.core.instrument.SerialManipulator, sscanss.core.instrument.PositioningStack]
        """
        if name == self.positioning_stack.name:
            return self.positioning_stack

        if name in self.positioners:
            return self.positioners[name]
        else:
            ValueError(f'"{name}" Positioner could not be found.')

    def loadPositioningStack(self, stack_key):
        """ load a positioning stack with the specified key

        :param stack_key: name of stack
        :type stack_key: str
        """
        positioner_keys = self.positioning_stacks[stack_key]

        for i in range(len(positioner_keys)):
            key = positioner_keys[i]
            if i == 0:
                self.positioning_stack = PositioningStack(stack_key, self.positioners[key])
            else:
                self.positioning_stack.addPositioner(self.positioners[key])

    def model(self):
        """ generates 3d model of the instrument.

        :return: 3D model of instrument
        :rtype: sscanss.core.scene.node.Node
        """
        node = Node()

        node.addChild(self.positioning_stack.model())
        for _, detector in self.detectors.items():
            node.addChild(detector.model())

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

    def model(self):
        return self.positioner.model()


class Detector:
    def __init__(self, name):
        self.name = name
        self.__current_collimator = None
        self.collimators = {}
        self.positioner = None

    @property
    def current_collimator(self):
        return self.__current_collimator

    @current_collimator.setter
    def current_collimator(self, key):
        if key in self.collimators:
            self.__current_collimator = self.collimators[key]
        else:
            self.__current_collimator = None

    def model(self):
        if self.positioner is None:
            return Node() if self.current_collimator is None else self.current_collimator.model()
        else:
            node = self.positioner.model()
            if self.current_collimator is not None:
                transformed_mesh = self.current_collimator.mesh.transformed(self.positioner.pose)
                node.addChild(Node(transformed_mesh))
            return node


class Collimator:
    def __init__(self, name, aperture, mesh):
        self.name = name
        self.aperture = aperture
        self.mesh = mesh

    def model(self):
        return Node(self.mesh)


class PositioningStack:
    def __init__(self, name, fixed):
        """ This class represents a group of serial manipulators stacked on each other.
        The stack has a fixed base manipulator and auxiliary manipulator can be append to it.
         When an auxiliary is append the fixed link btw the stack and the new is computed.
         more details - https://doi.org/10.1016/j.nima.2015.12.067

        :param name: name of stack
        :type name: str
        :param fixed: base manipulator
        :type fixed: sscanss.core.instrument.robotics.SerialManipulator
        """
        self.name = name
        self.fixed = fixed
        self.fixed.reset()
        self.tool_link = self.fixed.pose.inverse()
        self.auxiliary = []
        self.link_matrix = []
        self.__payload = None
        self.ik_solver = IKSolver(self)

    @property
    def tool_pose(self):
        return self.pose @ self.tool_link

    @property
    def pose(self):
        """ the pose of the end effector of the manipulator

        :return: transformation matrix
        :rtype: sscanss.core.math.matrix.Matrix44
        """
        T = self.fixed.pose
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            T @= link @ positioner.pose
        return T

    def __defaultPoseInverse(self, positioner):
        """ calculates the inverse of the default pose for the given positioner which
        is used to calculate the fixed link

        :param positioner: auxiliary positioner
        :type positioner: sscanss.core.instrument.robotics.SerialManipulator
        :return: transformation matrix
        :rtype: sscanss.core.math.matrix.Matrix44
        """
        q = positioner.set_points
        positioner.resetOffsets()
        matrix = positioner.pose.inverse()
        positioner.fkine(q, ignore_locks=True)

        return matrix

    def changeBaseMatrix(self, positioner, matrix):
        """ change the base matrix of a positioner in the stack

        :param positioner: auxiliary positioner
        :type positioner: sscanss.core.instrument.robotics.SerialManipulator
        :param matrix: new base matrix
        :type matrix: sscanss.core.math.matrix.Matrix44
        """
        index = self.auxiliary.index(positioner)
        positioner.base = matrix

        if positioner is not self.auxiliary[-1]:
            self.link_matrix[index+1] = self.__defaultPoseInverse(positioner)

    def addPositioner(self, positioner):
        """ append a positioner to the stack

        :param positioner: auxiliary positioner
        :type positioner: sscanss.core.instrument.robotics.SerialManipulator
        """
        positioner.reset()
        self.tool_link = positioner.pose.inverse()
        last_positioner = self.auxiliary[-1] if self.auxiliary else self.fixed
        self.auxiliary.append(positioner)
        self.link_matrix.append(self.__defaultPoseInverse(last_positioner))

    @property
    def configuration(self):
        """ current configuration (joint offsets for all links) of the stack

        :return: current configuration
        :rtype: list[float]
        """
        conf = []
        conf.extend(self.fixed.configuration)
        for positioner in self.auxiliary:
            conf.extend(positioner.configuration)

        return conf

    @property
    def links(self):
        """ links from all manipulators the stack

        :return: links in stack
        :rtype: list[sscanss.core.instrument.robotics.Link]
        """
        links = []
        links.extend(self.fixed.links)
        for positioner in self.auxiliary:
            links.extend(positioner.links)

        return links

    @property
    def numberOfLinks(self):
        """ number of links in stack

        :return: number of links
        :rtype: int
        """
        number = self.fixed.numberOfLinks
        for positioner in self.auxiliary:
            number += positioner.numberOfLinks

        return number

    @property
    def bounds(self):
        return [(link.lower_limit, link.upper_limit) for link in self.links]

    def fkine(self, q, ignore_locks=False, setpoint=True):
        """ Moves the stack to specified configuration and returns the forward kinematics
        transformation matrix of the stack.

        :param q: list of joint offsets to move to. The length must be equal to number of links
        :type q: List[float]
        :param ignore_locks: indicates that joint locks should be ignored
        :type ignore_locks: bool
        :param setpoint: indicates that given configuration, q is a setpoint
        :type setpoint: bool
        :return: Forward kinematic transformation matrix
        :rtype: sscanss.core.math.matrix.Matrix44
        """
        start, end = 0, self.fixed.numberOfLinks
        T = self.fixed.fkine(q[start:end], ignore_locks=ignore_locks, setpoint=setpoint)
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            start, end = end, end + positioner.numberOfLinks
            T @= link @ positioner.fkine(q[start:end], ignore_locks=ignore_locks, setpoint=setpoint)

        return T

    def ikine(self, current_pose, target_pose,  bounded=True, tol=0.02):
        q = self.ik_solver.solve(current_pose, target_pose, tol=tol, bounded=bounded)
        return q, self.ik_solver.residual_error, self.ik_solver.status

    def model(self):
        """ generates 3d model of the stack.

        :return: 3D model of manipulator
        :rtype: sscanss.core.scene.node.Node
        """
        node = self.fixed.model()
        matrix = self.fixed.pose
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            matrix @= link
            node.addChild(positioner.model(matrix))
            matrix @= positioner.pose

        return node

    @property
    def set_points(self):
        """ expected configuration (set-point for all links) of the manipulator

        :return: expected configuration
        :rtype: list[float]
        """
        set_points = []
        set_points.extend(self.fixed.set_points)
        for positioner in self.auxiliary:
            set_points.extend(positioner.set_points)

        return set_points

    @set_points.setter
    def set_points(self, q):
        """ setter for set_points

        :param q: expected configuration
        :type q: list[float]
        """
        for offset, link in zip(q, self.links):
            link.set_point = offset


class ScriptTemplate:
    @unique
    class Key(Enum):
        script = 'script'
        position = 'position'
        count = 'count'
        header = 'header'
        mu_amps = 'mu_amps'
        filename = 'filename'

    def __init__(self, filename, search_path):
        self.renderer = pystache.Renderer(search_dirs=search_path, file_extension='')
        try:
            template = self.renderer.load_template(filename)
            self.parsed = pystache.parse(template)
        except pystache.common.TemplateNotFoundError:
            raise FileNotFoundError(f'Script Template file "{filename}" not found in {search_path}')
        except UnicodeDecodeError:
            raise ValueError('Could not decode the template file')
        except pystache.parser.ParsingError:
            raise ValueError('Template Parsing Failed')

        script_tag = ''
        self.header_order = []
        self.keys = {}
        for parse in self.parsed._parse_tree:
            if not (isinstance(parse, pystache.parser._SectionNode) or
                    isinstance(parse, pystache.parser._EscapeNode)):
                continue

            key = ScriptTemplate.Key(parse.key)  # throws ValueError if parse.key is not found
            self.keys[key.value] = ''

            if parse.key == ScriptTemplate.Key.script.value:
                script_tag = parse

        if not script_tag:
            raise ValueError('No Script Tag!')

        for node in script_tag.parsed._parse_tree:
            if isinstance(node, pystache.parser._EscapeNode):
                key = ScriptTemplate.Key(node.key)  # throws ValueError if parse.key is not found
                self.header_order.append(key.value)
                self.keys[key.value] = ''

    def render(self):
        return self.renderer.render(self.parsed, self.keys)
