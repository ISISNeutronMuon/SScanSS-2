from enum import Enum, unique
import logging
import math
import nlopt
import numpy as np
from PyQt5 import QtCore
from ..math.matrix import Matrix44
from ..math.misc import trunc
from ..math.transform import (rotation_btw_vectors, angle_axis_btw_vectors, rigid_transform, xyz_eulers_from_matrix,
                              matrix_to_angle_axis)
from ..math.quaternion import Quaternion, QuaternionVectorPair
from ..math.vector import Vector3
from ..scene.node import Node


class SerialManipulator:
    """This class defines a open loop kinematic chain.

    :param name: name of the manipulator
    :type name: str
    :param links: list of link objects
    :type links: List[Link]
    :param base: base matrix. None sets base to an identity matrix
    :type base: Union[Matrix44, None]
    :param tool: tool matrix. None sets tool to an identity matrix
    :type tool: Union[Matrix44, None]
    :param base_mesh: mesh object for the base of the manipulator
    :type base_mesh: Union[Mesh, None]
    :param custom_order: order of joint if order is different from kinematics
    :type custom_order: List[int]
    """
    def __init__(self, name, links, base=None, tool=None, base_mesh=None, custom_order=None):
        self.name = name
        self.links = links
        self.base = Matrix44.identity() if base is None else base
        self.default_base = self.base
        self.tool = Matrix44.identity() if tool is None else tool
        self.base_mesh = base_mesh
        self.order = custom_order if custom_order is not None else list(range(len(links)))
        self.revolute_index = [True if l.type == l.Type.Revolute else False for l in links]

    def fkine(self, q, start_index=0, end_index=None, include_base=True, ignore_locks=False, setpoint=True):
        """Moves the manipulator to specified configuration and returns the forward kinematics
        transformation matrix of the manipulator. The transformation matrix can be computed for a subset
        of links i.e a start index to end index

        :param q: list of joint offsets to move to. The length must be equal to number of links
        :type q: List[float]
        :param start_index: index to start
        :type start_index: int
        :param end_index: index to end. None sets end_index to index of last link
        :type end_index: Union[int, None]
        :param include_base: indicates that base matrix should be included
        :type include_base: bool
        :param ignore_locks: indicates that joint locks should be ignored
        :type ignore_locks: bool
        :param setpoint: indicates that given configuration, q is a setpoint
        :type setpoint: bool
        :return: Forward kinematic transformation matrix
        :rtype: Matrix44
        """
        link_count = self.numberOfLinks

        start = 0 if start_index < 0 else start_index
        end = link_count if end_index is None or end_index > link_count else end_index

        base = self.base if include_base and start == 0 else Matrix44.identity()
        tool = self.tool if end == link_count else Matrix44.identity()

        qs = QuaternionVectorPair.identity()
        for i in range(start, end):
            self.links[i].move(q[i], ignore_locks, setpoint)
            qs *= self.links[i].quaterionVectorPair

        return base @ qs.toMatrix() @ tool

    def fromUserFormat(self, q):
        """converts joint offset from user defined format to kinematic order

        :param q: list of joint offsets in user format. The length must be equal to number of links
        :type q: List[float]
        :return: list of joint offsets in kinematic order.
        :rtype: List[float]
        """
        conf = np.zeros(self.numberOfLinks)
        conf[self.order] = q
        conf[self.revolute_index] = np.radians(conf[self.revolute_index])

        return conf.tolist()

    def toUserFormat(self, q):
        """converts joint offset from kinematic order to user defined format

        :param q: list of joint offsets in kinematic order. The length must be equal to number of links
        :type q: List[float]
        :return: list of joint offsets in user format.
        :rtype: List[float]
        """
        conf = np.copy(q)
        conf[self.revolute_index] = np.degrees(conf[self.revolute_index])
        conf = conf[self.order]

        return conf.tolist()

    def resetOffsets(self):
        """
        resets link offsets to the defaults
        """
        for link in self.links:
            link.reset()

    def reset(self):
        """
         resets  base matrix, link offsets, locks, and limits to the defaults
        """
        self.base = self.default_base
        for link in self.links:
            link.reset()
            link.locked = False
            link.ignore_limits = False

    @property
    def numberOfLinks(self):
        """number of links in manipulator

        :return: number of links
        :rtype: int
        """
        return len(self.links)

    @property
    def set_points(self):
        """expected configuration (set-point for all links) of the manipulator.
        This is useful when the animating the manipulator in that case the actual configuration
        differs from the set-point or final configuration.

        :return: expected configuration
        :rtype: list[float]
        """
        return [link.set_point for link in self.links]

    @set_points.setter
    def set_points(self, q):
        """setter for set_points

        :param q: expected configuration
        :type q: list[float]
        """
        for offset, link in zip(q, self.links):
            link.set_point = offset

    @property
    def configuration(self):
        """current configuration (joint offsets for all links) of the manipulators

        :return: current configuration
        :rtype: list[float]
        """
        return [link.offset for link in self.links]

    @property
    def pose(self):
        """the pose of the end effector of the manipulator

        :return: transformation matrix
        :rtype: Matrix44
        """
        qs = QuaternionVectorPair.identity()
        for link in self.links:
            qs *= link.quaterionVectorPair

        return self.base @ qs.toMatrix() @ self.tool

    def model(self, matrix=None):
        """Generates 3d model of the manipulator and transforms it with specified matrix.

        :param matrix: transformation matrix
        :type matrix: Union[Matrix44, None]
        :return: 3D model of manipulator
        :rtype: Node
        """
        node = Node()
        node.render_mode = Node.RenderMode.Solid

        if matrix is None:
            base = self.base
        else:
            base = matrix @ self.base

        if self.base_mesh is not None:
            child = Node(self.base_mesh)
            child.transform = base
            child.render_mode = None

            node.addChild(child)

        qs = QuaternionVectorPair.identity()
        joint_pos = Vector3()
        up = Vector3([0., 0., 1.])
        for link in self.links:
            qs *= link.quaterionVectorPair
            rot = rotation_btw_vectors(up, link.joint_axis)
            m = Matrix44.identity()
            m[0:3, 0:3] = qs.quaternion.toMatrix() @ rot
            m[0:3, 3] = joint_pos if link.type == Link.Type.Revolute else qs.vector

            m = base @ m
            if link.mesh is not None:
                child = Node(link.mesh)
                child.transform = m
                child.render_mode = None

                node.addChild(child)
            joint_pos = qs.vector

        return node


class Link:
    """This class represents a link/joint that belongs to a serial manipulator.
    The joint could be revolute or prismatic. The link is represented using the Quaternion-vector
    kinematic notation.

    :param name: name of the link
    :type name: str
    :param axis: axis of rotation or translation
    :type axis: List[float]
    :param point: centre of joint
    :type point: List[float]
    :param joint_type: joint type
    :type joint_type: Link.Type
    :param lower_limit: lower limit of joint
    :type lower_limit: float
    :param upper_limit: upper limit of joint
    :type upper_limit: float
    :param default_offset: default joint offset
    :type default_offset: float
    :param mesh: mesh object for the base
    :type mesh: Mesh
    """

    @unique
    class Type(Enum):
        Revolute = 'revolute'
        Prismatic = 'prismatic'

    def __init__(self, name, axis, point, joint_type, lower_limit, upper_limit, default_offset, mesh=None):
        self.joint_axis = Vector3(axis)

        if self.joint_axis.length < 0.00001:
            raise ValueError('The joint axis cannot be a zero vector.')

        self.quaternion = Quaternion.fromAxisAngle(self.joint_axis, 0.0)
        self.vector = Vector3(point)
        self.home = Vector3(point)
        self.type = joint_type
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.default_offset = default_offset
        self.set_point = default_offset
        self.mesh = mesh
        self.name = name
        self.locked = False
        self.ignore_limits = False  # This just stores state it does not affect behaviour
        self.reset()

    def move(self, offset, ignore_locks=False, setpoint=True):
        """moves link by the specified offset

        :param offset: joint offset
        :type offset: float
        :param ignore_locks: indicates that joint locks should be ignored
        :type ignore_locks: bool
        :param setpoint: indicates that given offset is a setpoint
        :type setpoint: bool
        """
        if self.locked and not ignore_locks:
            return

        self.offset = offset
        self.set_point = offset if setpoint else self.set_point
        if self.type == Link.Type.Revolute:
            self.quaternion = Quaternion.fromAxisAngle(self.joint_axis, offset)
            self.vector = self.quaternion.rotate(self.home)
        else:
            self.vector = self.home + self.joint_axis * offset

    def reset(self):
        """
        moves link to it default offset
        """
        self.move(self.default_offset, True)

    @property
    def transformationMatrix(self):
        """pose of the link

        :return: pose of the link
        :rtype: Matrix44
        """
        return self.quaterionVectorPair.toMatrix()

    @property
    def quaterionVectorPair(self):
        """pose of the link

        :return: pose of the link
        :rtype: QuaternionVectorPair
        """
        return QuaternionVectorPair(self.quaternion, self.vector)


def joint_space_trajectory(start_pose, stop_pose, step):
    """Generates a trajectory from a start to end configuration.

    :param start_pose: inclusive start joint configuration/offsets
    :type start_pose: List[float]
    :param stop_pose: inclusive stop joint configuration/offsets
    :type stop_pose: List[float]
    :param step: number of steps
    :type step: int
    :return: array of configurations defining the trajectory
    :rtype: numpy.ndarray
    """
    dof = len(start_pose)
    trajectory = np.zeros((step, dof))

    for i in range(dof):
        t = cubic_polynomial_trajectory(start_pose[i], stop_pose[i], step=step)
        trajectory[:, i] = t

    return trajectory


def cubic_polynomial_trajectory(p0, p1, step=100):
    """Generates a trajectory from p0 to p1 using a cubic polynomial.

    :param p0: inclusive start value
    :type p0: float
    :param p1: inclusive stop value
    :type p1: float
    :param step: number of steps
    :type step: int
    :return: offsets in the trajectory
    :rtype: numpy.ndarray
    """

    t0 = 0.0
    tf = step
    t = np.linspace(t0, tf, step)

    t0_2 = t0 * t0
    t0_3 = t0_2 * t0

    tf_2 = tf * tf
    tf_3 = tf_2 * tf

    m = [[1.0, t0, t0_2, t0_3],
         [0.0, 1.0, 2 * t0, 3 * t0_2],
         [1.0, tf, tf_2, tf_3],
         [0.0, 1.0, 2 * tf, 3 * tf_2]]

    v0 = v1 = 0.0
    b = [p0, v0, p1, v1]
    a = np.dot(np.linalg.inv(m), b)

    pd = np.polyval(a[::-1], t)

    return pd


class Sequence(QtCore.QObject):
    """This class creates an animation from start to end configuration

    :param frames: function to generate frame at each way point
    :type frames: method
    :param start: inclusive start joint configuration/offsets
    :type start: List[float]
    :param stop: inclusive stop joint configuration/offsets
    :type stop: List[float]
    :param duration: time duration in milliseconds
    :type duration: int
    :param step: number of steps
    :type step: int
    """
    frame_changed = QtCore.pyqtSignal()

    def __init__(self, frames, start, stop, duration, step):
        super().__init__()

        self.timeline = QtCore.QTimeLine(duration, self)
        self.timeline.setFrameRange(0, step - 1)

        self.trajectory = joint_space_trajectory(start, stop, step)

        self.timeline.setCurrentTime(self.timeline.duration())
        self.timeline.frameChanged.connect(self.animate)
        self.frames = frames
        self.step = step

    def start(self):
        """
        starts the animation
        """
        self.timeline.start()

    def stop(self):
        """
        stops the animation
        """
        if self.timeline.currentTime() < self.timeline.duration():
            self.timeline.setCurrentTime(self.timeline.duration())

        self.timeline.stop()

    def isRunning(self):
        """indicates if the animation is running

        :return: indicates if the animation is running
        :rtype: bool
        """
        if self.timeline.state() == QtCore.QTimeLine.Running:
            return True

        return False

    def animate(self, index):
        """Calls the frame function and emits signal to notify frame change

        :param index: current step/frame in the animation
        :type index: int
        """
        self.frames(self.trajectory[index, :])
        self.frame_changed.emit()


class IKResult:
    """Data class for the inverse kinematics result

    :param q: final configuration
    :type q: Union[List[float], numpy.ndarray]
    :param status: solver status
    :type status: IKSolver.Status
    :param pos_err: 3D position error
    :type pos_err: Union[List[float], numpy.ndarray]
    :param orient_err: 3D orientation error
    :type orient_err: Union[List[float], numpy.ndarray]
    :param pos_err_ok: flag indicates if position error is within tolerance
    :type pos_err_ok: bool
    :param orient_err_ok: flag indicates if orientation error is within tolerance
    :type orient_err_ok: bool
    """
    def __init__(self, q, status, pos_err, orient_err, pos_err_ok, orient_err_ok):
        self.q = q
        self.status = status
        self.position_error = pos_err
        self.position_converged = pos_err_ok
        self.orientation_error = orient_err
        self.orientation_converged = orient_err_ok


class IKSolver:
    """General inverse kinematics solver for serial robots. Inverse kinematics is framed as an optimization
    problem and solved using randomized global optimizer with local optimization step to refine result.

    :param robot: robot used in the solver
    :type robot: PositioningStack
    """

    @unique
    class Status(Enum):
        Converged = 0
        NotConverged = 1
        Unreachable = 2
        Failed = 3

    def __init__(self, robot):
        self.robot = robot
        self.status = IKSolver.Status.NotConverged
        self.residual_error = ([-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], False, False)

    def __create_optimizer(self, n, tolerance, lower_bounds, upper_bounds, local_max_eval, global_max_eval):
        """creates optimizer to find joint configuration that achieves specified tolerance, the number of joints could
        be less than the number of joints in robot if some joints are locked. Since its not possible to change the
        optimizer size after creation, re-creating the optimizer was the simplest way to accommodate locked joints

        :param n: number of joints configuration
        :type n: int
        :param tolerance: stopping criterion for optimizer
        :type tolerance: float
        :param lower_bounds: lower joint bounds
        :type lower_bounds: numpy.ndarray
        :param upper_bounds: upper joint bounds
        :type upper_bounds: numpy.ndarray
        :param local_max_eval: number of evaluations for local optimization
        :type local_max_eval: int
        :param global_max_eval: number of evaluations for global optimization
        :type global_max_eval: int
        """
        nlopt.srand(10)
        self.optimizer = nlopt.opt(nlopt.G_MLSL, n)
        self.optimizer.set_lower_bounds(lower_bounds)
        self.optimizer.set_upper_bounds(upper_bounds)
        self.optimizer.set_min_objective(self.objective)
        self.optimizer.set_stopval(tolerance)
        self.optimizer.set_maxeval(global_max_eval)
        self.optimizer.set_ftol_abs(1e-8)

        opt = nlopt.opt(nlopt.LD_SLSQP, n)
        opt.set_maxeval(local_max_eval)
        opt.set_ftol_abs(1e-8)
        self.optimizer.set_local_optimizer(opt)

    def __gradient(self, q, epsilon, f0):
        """computes gradient of objective function at configuration q using finite difference

        :param q: joint configuration candidate
        :type q: numpy.ndarray
        :param epsilon: increment used to determine gradient
        :type epsilon: float
        :param f0: objective error
        :type f0: float
        :return: approximate gradient
        :rtype: numpy.ndarray
        """
        grad = np.zeros((len(q),))
        ei = np.zeros((len(q),))
        for k in range(len(q)):
            ei[k] = 1.0
            d = epsilon * ei
            grad[k] = (self.objective(q + d, np.array([])) - f0) / d[k]
            ei[k] = 0.0
        return grad

    def objective(self, q, gradient):
        """optimization objective

        :param q: joint configuration candidate
        :type q: numpy.ndarray
        :param gradient: gradient
        :type gradient: numpy.ndarray
        :return: objective error
        :rtype: float
        """
        conf = self.start.copy()
        conf[self.active_joints] = q
        H = self.robot.fkine(conf) @ self.robot.tool_link

        residuals = np.zeros(6)
        residuals[0:3] = self.target_position - (H[0:3, 0:3] @ self.current_position + H[0:3, 3])

        if self.current_orientation.shape[0] == 1:
            v1 = H[0:3, 0:3] @ self.current_orientation[0]
            v2 = self.target_orientation[0]
            angle, axis = angle_axis_btw_vectors(v1, v2)
        else:
            v1 = np.append(self.current_orientation @ H[0:3, 0:3].transpose(), [0., 0., 0.]).reshape(-1, 3)
            v2 = np.append(self.target_orientation, [0., 0., 0.]).reshape(-1, 3)
            result = rigid_transform(v1, v2)
            angle, axis = matrix_to_angle_axis(result.matrix)

        residuals[3:6] = math.degrees(angle) * axis
        error = np.dot(residuals, residuals)

        if error < self.best_result and np.logical_and(
                 self.lower_bounds <= q, self.upper_bounds >= q).all():
            self.best_result = error
            self.best_conf = conf

        if gradient.size > 0:
            gradient[:] = self.__gradient(q, 1e-8, error)

        return error

    def solve(self, current_pose, target_pose, start=None, tol=(1e-2, 1.0), bounded=True, local_max_eval=1000,
              global_max_eval=100):
        """finds the configuration that moves current pose to target pose within specified tolerance.

        :param current_pose: current position and vector orientation
        :type current_pose: Tuple[numpy.ndarray, numpy.ndarray]
        :param target_pose: target position and vector orientation
        :type target_pose: Tuple[numpy.ndarray, numpy.ndarray]
        :param start: starting joint configuration if None current configuration is used
        :type start: Union[None, numpy.ndarray]
        :param tol: position and orientation convergence tolerance
        :type tol: Tuple[float, float]
        :param bounded: indicates if joint bounds should be used
        :type bounded: bool
        :param local_max_eval: number of evaluations for local optimization
        :type local_max_eval: int
        :param global_max_eval: number of evaluations for global optimization
        :type global_max_eval: int
        :return: result from the inverse kinematics optimization
        :rtype: IKResult
        """
        self.tolerance = tol
        stop_eval_tol = min(tol) ** 2
        self.target_position, self.target_orientation = target_pose
        self.current_position, self.current_orientation = current_pose

        self.best_conf = np.array(self.robot.set_points, dtype=float)
        self.best_result = np.inf

        self.start = self.best_conf if start is None else start
        self.active_joints = [not link.locked for link in self.robot.links]
        q0 = self.start[self.active_joints]

        # Using very large value to simulate unbounded joints
        bounds = np.array([(-100000, 100000) if link.type == link.Type.Prismatic else (-2 * np.pi, 2 * np.pi)
                           for link in self.robot.links])

        if bounded:
            active_limits = [not l.ignore_limits for l in self.robot.links]
            real_bounds = np.array([(link.lower_limit, link.upper_limit) for link in self.robot.links])
            bounds[active_limits] = real_bounds[active_limits]

        lower_bounds, upper_bounds = list(zip(*bounds[self.active_joints]))

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        q0 = np.clip(q0, lower_bounds, upper_bounds)  # ensure starting config is bounded avoids crash

        try:
            self.__create_optimizer(q0.size, stop_eval_tol, lower_bounds, upper_bounds, local_max_eval, global_max_eval)
            self.optimizer.optimize(q0)
            self.residual_error = self.computeResidualError()
            if self.residual_error[2] and self.residual_error[3]:
                self.status = IKSolver.Status.Converged
            else:
                self.status = IKSolver.Status.NotConverged
        except nlopt.RoundoffLimited:
            self.status = IKSolver.Status.NotConverged
            logging.exception("Roundoff Error occurred during inverse kinematics")
        except RuntimeError:
            self.status = IKSolver.Status.Failed
            logging.exception("Unknown runtime error occurred during inverse kinematics")

        return IKResult(self.best_conf, self.status, *self.residual_error)

    def computeResidualError(self):
        """computes residual error and checks converges, the result is a tuple in the format
        [position_error, orientation_error, position_error_flag, orient_error_flag]

        :return: 3D position and orientation error and flags indicating convergence
        :rtype: Tuple[numpy.ndarray, numpy.ndarray, bool, bool]
        """
        H = self.robot.fkine(self.best_conf) @ self.robot.tool_link
        position_error = self.target_position - (H[0:3, 0:3] @ self.current_position + H[0:3, 3])
        position_error_good = False if trunc(np.linalg.norm(position_error), 3) > self.tolerance[0] else True

        if self.current_orientation.shape[0] == 1:
            v1 = H[0:3, 0:3] @ self.current_orientation[0]
            v2 = self.target_orientation[0]
            matrix = rotation_btw_vectors(v1, v2)
        else:
            v1 = np.append(self.current_orientation @ H[0:3, 0:3].transpose(), [0., 0., 0.]).reshape(-1, 3)
            v2 = np.append(self.target_orientation, [0., 0., 0.]).reshape(-1, 3)
            matrix = rigid_transform(v1, v2).matrix

        orientation_error = np.degrees(xyz_eulers_from_matrix(matrix))
        orient_error_good = False if trunc(np.linalg.norm(orientation_error), 3) > self.tolerance[1] else True

        return position_error, orientation_error, position_error_good, orient_error_good
