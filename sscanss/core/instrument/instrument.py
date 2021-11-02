"""
A collection of classes to represent instrument and its components
"""
from enum import Enum, unique
import pystache
from .robotics import IKSolver
from ..geometry.mesh import MeshGroup


class Instrument:
    """This class represents a diffractometer instrument

    :param name: name of instrument
    :type name: str
    :param gauge_volume: gauge volume of the instrument
    :type gauge_volume: List[float]
    :param detectors: detectors
    :type detectors: Dict[str, Detector]
    :param jaws: jaws
    :type jaws: Jaws
    :param positioners: positioners
    :type positioners: Dict[str, SerialManipulator]
    :param positioning_stacks: positioning stacks
    :type positioning_stacks: Dict[str, List[str]]
    :param script: template for instrument script
    :type script: Script
    :param fixed_hardware: mesh for fixed hardware
    :type fixed_hardware: Dict[str, Mesh]
    """

    def __init__(self, name, gauge_volume, detectors, jaws, positioners, positioning_stacks, script, fixed_hardware):
        self.name = name
        self.gauge_volume = gauge_volume
        self.detectors = detectors
        self.positioners = positioners
        self.jaws = jaws
        self.fixed_hardware = fixed_hardware
        self.positioning_stacks = positioning_stacks
        self.script = script
        self.loadPositioningStack(list(self.positioning_stacks.keys())[0])

    @property
    def q_vectors(self):
        q_vectors = []
        beam_axis = -self.jaws.beam_direction
        for detector in self.detectors.values():
            vector = beam_axis + detector.diffracted_beam
            if vector.length > 0.0001:
                q_vectors.append(vector.normalized)
            else:
                q_vectors.append(vector)
        return q_vectors

    @property
    def beam_in_gauge_volume(self):
        # Check beam hits the gauge volume
        actual_axis = self.gauge_volume - self.jaws.beam_source
        axis = self.jaws.beam_direction ^ actual_axis
        if axis.length > 0.0001:
            return False

        return True

    def getPositioner(self, name):
        """get positioner or positioning stack by name

        :param name: name of positioner or stack
        :type name: str
        :return: positioner or positioning stack
        :rtype: Union[SerialManipulator, PositioningStack]
        """
        if name == self.positioning_stack.name:
            return self.positioning_stack

        if name in self.positioners:
            return self.positioners[name]
        else:
            raise ValueError(f'"{name}" Positioner could not be found.')

    def loadPositioningStack(self, stack_key):
        """load a positioning stack with the specified key

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


class Jaws:
    """This class represents an instrument jaws with adjustable aperture. The position of
    jaws could be changed by attaching a positioner

    :param name: name of the Jaws
    :type name: str
    :param beam_source: source position of the beam
    :type beam_source: Vector3
    :param beam_direction: axis of beam
    :type beam_direction: Vector3
    :param aperture: horizontal and vertical size of jaws’ aperture
    :type aperture: List[float]
    :param lower_limit: horizontal and vertical lower limit of jaws’ aperture
    :type lower_limit: List[float]
    :param upper_limit: horizontal and vertical upper limit of jaws’ aperture
    :type upper_limit: List[float]
    :param mesh: mesh object for the jaw
    :type mesh: Mesh
    :param positioner: positioner that controls jaws position
    :type positioner: Union[SerialManipulator, None]
    """

    def __init__(self, name, beam_source, beam_direction, aperture, lower_limit, upper_limit, mesh, positioner=None):
        self.name = name
        self.aperture = aperture
        self.initial_source = beam_source
        self.beam_source = beam_source
        self.initial_direction = beam_direction
        self.beam_direction = beam_direction
        self.aperture_lower_limit = lower_limit
        self.aperture_upper_limit = upper_limit
        self.positioner = positioner
        self.mesh = mesh

    def updateBeam(self):
        """Update beam source and direction"""
        pose = self.positioner.pose
        self.beam_direction = pose[0:3, 0:3] @ self.initial_direction
        self.beam_source = pose[0:3, 0:3] @ self.initial_source + pose[0:3, 3]

    @property
    def positioner(self):
        """jaws's positioner

        :return: positioner that controls jaws position
        :rtype positioner: Union[SerialManipulator, None]
        """
        return self._positioner

    @positioner.setter
    def positioner(self, value):
        """setter for jaws's positioner. The positioner forward kinematics function
        is modified so that the beam direction and source is updated when the positioner
        is moved

        :param value: positioner that controls jaws position
        :type value: Union[SerialManipulator, None]
        """
        self._positioner = value
        if value is not None:
            self._positioner.fkine = self.__wrapper(self._positioner.fkine)
            self.updateBeam()

    def __wrapper(self, func):
        """wrapper for positioner forward kinematics function to ensure beam is updated after
        movement

        :param func: function to wrap
        :type func: Callable[..., Any]
        :return: wrapped function
        :rtype: Callable[..., Any]
        """

        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            self.updateBeam()
            return result

        return wrapped

    def model(self):
        """generates 3d model of the jaw and its positioner

        :return: 3D model of jaws and positioner
        :rtype: MeshGroup
        """
        model = MeshGroup()
        if self.positioner is not None:
            model.merge(self.positioner.model())
            model.addMesh(self.mesh, self.positioner.pose)
        else:
            model.addMesh(self.mesh)

        return model


class Detector:
    """This class represents an instrument detector with swappable collimators. The position of
    detector could be changed by attaching a positioner

    :param name: name of detector
    :type name: str
    :param diffracted_beam: axis of beam coming into detector from gauge volume
    :type diffracted_beam: Vector3
    :param collimators: dictionary of collimator used by the detector
    :type collimators: Union[Dict[str, Collimator], None]
    :param positioner: positioner that controls detector position
    :type positioner: Union[SerialManipulator, None]
    """

    def __init__(self, name, diffracted_beam, collimators=None, positioner=None):
        self.name = name
        self.__current_collimator = None
        self.initial_beam = diffracted_beam
        self.diffracted_beam = diffracted_beam
        self.collimators = {} if collimators is None else collimators
        self.positioner = positioner

    def updateBeam(self):
        """Update diffracted beam direction"""
        self.diffracted_beam = self.positioner.pose[0:3, 0:3] @ self.initial_beam

    @property
    def positioner(self):
        """detector's positioner

        :return: positioner that controls detector position
        :rtype positioner: Union[SerialManipulator, None]
        """
        return self._positioner

    @positioner.setter
    def positioner(self, value):
        """setter for detector's positioner. The positioner forward kinematics function
        is modified so that the diffracted beam direction is updated when the positioner
        is moved

        :param value: positioner that controls detector position
        :type value: Union[SerialManipulator, None]
        """
        self._positioner = value
        if value is not None:
            self._positioner.fkine = self.__wrapper(self._positioner.fkine)
            self.updateBeam()

    def __wrapper(self, func):
        """wrapper for positioner forward kinematics function to ensure beam is updated after
        movement

        :param func: function to wrap
        :type func: Callable[..., Any]
        :return: wrapped function
        :rtype: Callable[..., Any]
        """

        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            self.updateBeam()
            return result

        return wrapped

    @property
    def current_collimator(self):
        """gets active collimator

        :return: active collimator
        :rtype: Union[Collimator, None]
        """
        return self.__current_collimator

    @current_collimator.setter
    def current_collimator(self, key):
        """setter for active collimator

        :param key: key of collimator to set as active
        :type key: Union[str, None]
        """
        if key in self.collimators:
            self.__current_collimator = self.collimators[key]
        else:
            self.__current_collimator = None

    def model(self):
        """generates 3d model of the detector and its positioner

        :return: 3D model of detector and positioner
        :rtype: MeshGroup
        """
        model = MeshGroup()
        transform = None

        if self.positioner is not None:
            model.merge(self.positioner.model())
            transform = self.positioner.pose

        if self.current_collimator is not None:
            model.addMesh(self.current_collimator.mesh, transform)

        return model


class Collimator:
    """This class represents an instrument collimator with fixed aperture size.

    :param name: name of collimator
    :type name: str
    :param aperture: horizontal and vertical size of collimator’s aperture
    :type aperture: List[float]
    :param mesh: mesh object for the collimator
    :type mesh: Mesh
    """

    def __init__(self, name, aperture, mesh):
        self.name = name
        self.aperture = aperture
        self.mesh = mesh


class PositioningStack:
    """This class represents a group of serial manipulators stacked on each other.
    The stack has a fixed base manipulator and auxiliary manipulator can be appended to it.
     When an auxiliary is appended the fixed link btw the stack and the new is computed.
     more details - https://doi.org/10.1016/j.nima.2015.12.067

    :param name: name of stack
    :type name: str
    :param fixed: base manipulator
    :type fixed: SerialManipulator
    """

    def __init__(self, name, fixed):

        self.name = name
        self.fixed = fixed
        self.fixed.reset()
        self.tool_link = self.fixed.pose.inverse()
        self.auxiliary = []
        self.link_matrix = []
        self.ik_solver = IKSolver(self)

    @property
    def tool_pose(self):
        return self.pose @ self.tool_link

    @property
    def pose(self):
        """the pose of the end effector of the manipulator

        :return: transformation matrix
        :rtype: Matrix44
        """
        pose = self.fixed.pose
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            pose @= link @ positioner.pose
        return pose

    def __defaultPoseInverse(self, positioner):
        """calculates the inverse of the default pose for the given positioner which
        is used to calculate the fixed link

        :param positioner: auxiliary positioner
        :type positioner: SerialManipulator
        :return: transformation matrix
        :rtype: Matrix44
        """
        q = positioner.set_points
        positioner.resetOffsets()
        matrix = positioner.pose.inverse()
        positioner.fkine(q, ignore_locks=True)

        return matrix

    def changeBaseMatrix(self, positioner, matrix):
        """change the base matrix of a positioner in the stack

        :param positioner: auxiliary positioner
        :type positioner: SerialManipulator
        :param matrix: new base matrix
        :type matrix: Matrix44
        """
        index = self.auxiliary.index(positioner)
        positioner.base = matrix

        if positioner is not self.auxiliary[-1]:
            self.link_matrix[index + 1] = self.__defaultPoseInverse(positioner)
        else:
            self.tool_link = self.__defaultPoseInverse(positioner)

    def addPositioner(self, positioner):
        """append a positioner to the stack

        :param positioner: auxiliary positioner
        :type positioner: SerialManipulator
        """
        positioner.reset()
        self.tool_link = positioner.pose.inverse()
        last_positioner = self.auxiliary[-1] if self.auxiliary else self.fixed
        self.auxiliary.append(positioner)
        self.link_matrix.append(self.__defaultPoseInverse(last_positioner))

    @property
    def configuration(self):
        """current configuration (joint offsets for all links) of the stack

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
        """links from all manipulators the stack

        :return: links in stack
        :rtype: list[Link]
        """
        links = []
        links.extend(self.fixed.links)
        for positioner in self.auxiliary:
            links.extend(positioner.links)

        return links

    def fromUserFormat(self, q):
        """converts joint offset from user defined format to kinematic order

        :param q: list of joint offsets in user format. The length must be equal to number of links
        :type q: List[float]
        :return: list of joint offsets in kinematic order.
        :rtype: List[float]
        """
        start, end = 0, self.fixed.numberOfLinks
        conf = self.fixed.fromUserFormat(q[start:end])
        for positioner in self.auxiliary:
            start, end = end, end + positioner.numberOfLinks
            conf.extend(positioner.fromUserFormat(q[start:end]))

        return conf

    def toUserFormat(self, q):
        """converts joint offset from kinematic order to user defined format

        :param q: list of joint offsets in kinematic order. The length must be equal to number of links
        :type q: List[float]
        :return: list of joint offsets in user format.
        :rtype: List[float]
        """
        start, end = 0, self.fixed.numberOfLinks
        conf = self.fixed.toUserFormat(q[start:end])
        for positioner in self.auxiliary:
            start, end = end, end + positioner.numberOfLinks
            conf.extend(positioner.toUserFormat(q[start:end]))

        return conf

    @property
    def order(self):
        """user defined order of joints

        :return: joint indices in custom order
        :rtype: List[int]
        """
        end = self.fixed.numberOfLinks
        order = self.fixed.order.copy()
        for positioner in self.auxiliary:
            order.extend([end + order for order in positioner.order])
            end = end + positioner.numberOfLinks

        return order

    @property
    def numberOfLinks(self):
        """number of links in stack

        :return: number of links
        :rtype: int
        """
        number = self.fixed.numberOfLinks
        for positioner in self.auxiliary:
            number += positioner.numberOfLinks

        return number

    @property
    def bounds(self):
        """lower and upper bounds of the positioning stack for each joint

        :return: lower and upper joint limits
        :rtype: List[Tuple[float, float]]
        """
        return [(link.lower_limit, link.upper_limit) for link in self.links]

    def fkine(self, q, ignore_locks=False, setpoint=True):
        """Moves the stack to specified configuration and returns the forward kinematics
        transformation matrix of the stack.

        :param q: list of joint offsets to move to. The length must be equal to number of links
        :type q: List[float]
        :param ignore_locks: indicates that joint locks should be ignored
        :type ignore_locks: bool
        :param setpoint: indicates that given configuration, q is a setpoint
        :type setpoint: bool
        :return: Forward kinematic transformation matrix
        :rtype: Matrix44
        """
        start, end = 0, self.fixed.numberOfLinks
        T = self.fixed.fkine(q[start:end], ignore_locks=ignore_locks, setpoint=setpoint)
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            start, end = end, end + positioner.numberOfLinks
            T @= link @ positioner.fkine(q[start:end], ignore_locks=ignore_locks, setpoint=setpoint)

        return T

    def ikine(self, current_pose, target_pose, bounded=True, tol=(1e-2, 1.0), local_max_eval=1000, global_max_eval=100):
        """
        :param current_pose: current position and vector orientation
        :type current_pose: Tuple[numpy.ndarray, numpy.ndarray]
        :param target_pose: target position and vector orientation
        :type target_pose: Tuple[numpy.ndarray, numpy.ndarray]
        :param bounded: indicates if joint bounds should be used
        :type bounded: bool
        :param tol: position and orientation convergence tolerance
        :type tol: Tuple[float, float]
        :param local_max_eval: number of evaluations for local optimization
        :type local_max_eval: int
        :param global_max_eval: number of evaluations for global optimization
        :type global_max_eval: int
        :return: result from the inverse kinematics optimization
        :rtype: IKResult
        """
        return self.ik_solver.solve(
            current_pose,
            target_pose,
            tol=tol,
            bounded=bounded,
            local_max_eval=local_max_eval,
            global_max_eval=global_max_eval,
        )

    def model(self):
        """generates 3d model of the stack.

        :return: 3D model of manipulator
        :rtype: MeshGroup
        """
        model = self.fixed.model()
        matrix = self.fixed.pose
        for link, positioner in zip(self.link_matrix, self.auxiliary):
            matrix @= link
            aux_model = positioner.model(matrix)
            model.merge(aux_model)
            matrix @= positioner.pose

        return model

    @property
    def set_points(self):
        """expected configuration (set-point for all links) of the manipulator

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
        """setter for set_points

        :param q: expected configuration
        :type q: list[float]
        """
        for offset, link in zip(q, self.links):
            link.set_point = offset


class Script:
    """This class generates instrument script from a given template.

    :param template: pystache template
    :type template: str
    """

    @unique
    class Key(Enum):
        script = "script"
        position = "position"
        count = "count"
        header = "header"
        mu_amps = "mu_amps"
        filename = "filename"

    def __init__(self, template):
        self.renderer = pystache.Renderer()
        try:
            self.template = template
            self.parsed = pystache.parse(template)
        except pystache.parser.ParsingError as e:
            raise ValueError("Template Parsing Failed") from e

        script_tag = ""
        self.header_order = []
        self.keys = {}
        key_list = [key.value for key in Script.Key]

        for parse in self.parsed._parse_tree:
            if not (isinstance(parse, pystache.parser._SectionNode) or isinstance(parse, pystache.parser._EscapeNode)):
                continue

            if parse.key not in key_list:
                raise ValueError(f'"{parse.key}" is not a valid script template key.')

            key = Script.Key(parse.key)
            self.keys[key.value] = ""

            if parse.key == Script.Key.script.value:
                script_tag = parse

        if not script_tag:
            raise ValueError('Script template must contain opening and closing "script" tag.')

        for node in script_tag.parsed._parse_tree:
            if isinstance(node, pystache.parser._EscapeNode):
                if node.key not in key_list:
                    raise ValueError(f'"{node.key}" is not a valid script template key.')

                key = Script.Key(node.key)
                self.header_order.append(key.value)
                self.keys[key.value] = ""

        if Script.Key.position.value not in self.keys:
            raise ValueError('Script template must contain "position" tag inside the "script" tag.')

    def render(self):
        """render the script from the template and key values

        :return: instrument script
        :rtype: str
        """
        return self.renderer.render(self.parsed, self.keys)
