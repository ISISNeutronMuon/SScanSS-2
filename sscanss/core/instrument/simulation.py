import json
import logging
import numpy as np
from multiprocessing import Event, Process, Queue, sharedctypes
from PyQt5 import QtCore
from .collision import CollisionManager
from .instrument import PositioningStack
from ..geometry.mesh import Mesh
from ..geometry.intersection import path_length_calculation
from ..math.constants import VECTOR_EPS
from ..math.matrix import Matrix44
from .robotics import IKSolver, Link, SerialManipulator
from ..scene.entity import InstrumentEntity
from ..scene.node import Node
from ..util.misc import Attributes
from ...config import settings, setup_logging


class SharedArray:
    """Data structure for managing RawArray for multiprocessing

    :param data: raw array containing data
    :type data: RawArray
    :param shape: array shape
    :type shape: Tuple
    :param data_type: numpy data type
    :type data_type: numpy.dtype
    """
    def __init__(self, data, shape, data_type):
        self.data = data
        self.shape = shape
        self.data_type = data_type

    @staticmethod
    def fromNumpyArray(array):
        """Copies data from numpy array into a shared array.

        :param array: numpy array
        :type array: numpy.ndarray
        :return: shared array object
        :rtype: SharedArray
        """
        shape = array.shape
        if array.dtype == float:
            data = sharedctypes.RawArray('d', int(np.prod(shape)))
            temp = np.frombuffer(data).reshape(shape)
        elif array.dtype == np.float32:
            data = sharedctypes.RawArray('f', int(np.prod(shape)))
            temp = np.frombuffer(data, dtype=np.float32).reshape(shape)
        elif array.dtype == np.int32 or array.dtype == np.uint32:
            data = sharedctypes.RawArray('i', int(np.prod(shape)))
            temp = np.frombuffer(data, dtype=np.int32).reshape(shape)
        elif array.dtype == bool:
            data = sharedctypes.RawArray('b', int(np.prod(shape)))
            temp = np.frombuffer(data, dtype=np.int8).reshape(shape)
        else:
            raise ValueError(f'{array.dtype} data type is unsupported')
        # Copy data to our shared array.
        np.copyto(temp, array)

        return SharedArray(data, shape, array.dtype)

    @staticmethod
    def toNumpyArray(array):
        """Casts shared array data to numpy array (data is not copied).

        :param array: shared array object
        :type array: Array
        :return: numpy array
        :rtype: numpy.ndarray
        """
        if array.data_type == bool:
            data = np.frombuffer(array.data, dtype=np.int8).reshape(array.shape).astype(bool)
        else:
            data = np.frombuffer(array.data, dtype=array.data_type).reshape(array.shape)
        return data


class SharedInstrument:
    """Data class for the instrument model required for collision

    :param instrument: instrument object
    :type instrument: Instrument
    """
    def __init__(self, instrument):
        entity = InstrumentEntity(instrument)
        self.size = len(entity.transforms)
        self.vertices = SharedArray.fromNumpyArray(entity.vertices)
        self.indices = SharedArray.fromNumpyArray(entity.indices)
        self.transforms = SharedArray.fromNumpyArray(np.row_stack(entity.transforms))
        self.offsets = entity.offsets
        self.keys = entity.keys


def create_collision_node(instrument):
    """Creates collision node for a given instrument.

    :param instrument: instrument data for creating nodes
    :type instrument: SharedInstrument
    :return: node containing model of instrument
    :rtype: Dict[str, List[Node]]
    """
    vertices = SharedArray.toNumpyArray(instrument.vertices)
    indices = SharedArray.toNumpyArray(instrument.indices)
    transforms = np.split(SharedArray.toNumpyArray(instrument.transforms), instrument.size)

    nodes = []
    start = 0
    for index, end in enumerate(instrument.offsets):
        node = Node()
        node.vertices = vertices[indices[start:end]]
        node.indices = np.arange(0, len(node.vertices))
        node.transform = transforms[index]
        nodes.append(node)
        start = end

    output = {}
    last_count = 0
    for key, count in instrument.keys.items():
        output[key] = nodes[last_count:count]
        last_count = count

    return output


def update_colliders(manager, sample_pose, sample_ids, positioner_poses, positioner_ids):
    """Updates the sample and positioner colliders

    :param manager: collision manager
    :type manager: CollisionManager
    :param sample_pose: sample transformation matrix
    :type sample_pose: Matrix44
    :param sample_ids: list of sample collider ids
    :type sample_ids: List[int]
    :param positioner_poses: list of positioner poses
    :type positioner_poses: List[Matrix44]
    :param positioner_ids: list of positioner ids
    :type positioner_ids: List[int]
    """
    for i in sample_ids:
        manager.colliders[i].geometry.transform(sample_pose)

    for i, pose in zip(positioner_ids, positioner_poses):
        manager.colliders[i].geometry.transform(pose)

    manager.createAABBSets()


def populate_collision_manager(manager, sample, instrument_node):
    """Adds sample and instrument scene colliders to the collision manager and builds
    scene bounding boxes

    :param manager: collision manager
    :type manager: CollisionManager
    :param sample: list of sample mesh
    :type sample: List[Mesh]
    :param instrument_node: instrument node and ids
    :type instrument_node: Dict[str, List[Node]]
    :return: sample and positioner collider ids
    :rtype: Tuple[List[int], List[int]]
    """
    manager.clear()
    transform = [np.identity(4) for _ in range(len(sample))]
    manager.addColliders(sample, transform, manager.Exclude.All, True)
    sample_ids = list(range(len(sample)))
    positioner_ids = []

    for name, attribute_node in instrument_node.items():
        transform = [n.transform for n in attribute_node]
        if name == Attributes.Positioner.value:
            start_id = manager.colliders[-1].id + 1
            manager.addColliders(attribute_node, transform, exclude=manager.Exclude.Consecutive, movable=True)
            last_link_collider = manager.colliders[-1]
            for index, obj in enumerate(manager.colliders[0:len(sample)]):
                obj.excludes[last_link_collider.id] = True
                last_link_collider.excludes[index] = True
            positioner_ids.extend(range(start_id, last_link_collider.id + 1))
        else:
            exclude = manager.Exclude.Nothing if name == Attributes.Fixture.value else manager.Exclude.Consecutive
            manager.addColliders(attribute_node, transform, exclude=exclude, movable=False)

    manager.createAABBSets()
    return sample_ids, positioner_ids


class SimulationResult:
    """Data class for the simulation result

    :param result_id: result identifier
    :type result_id: str
    :param ik: inverse kinematics result
    :type ik: Union[IKResult, None]
    :param q_formatted: formatted positioner offsets
    :type q_formatted: Tuple
    :param alignment: alignment index
    :type alignment: int
    :param path_length: path length result
    :type path_length: Union[Tuple[float], None]
    :param collision_mask: mask showing which objects collided
    :type collision_mask: Union[List[bool], None]
    :param skipped: indicates if the result is skipped
    :type skipped: bool
    :param note: note about result such as reason for skipping
    :type note: str
    """
    def __init__(self,
                 result_id,
                 ik=None,
                 q_formatted=(None, None),
                 alignment=0,
                 path_length=None,
                 collision_mask=None,
                 skipped=False,
                 note=''):

        self.id = result_id
        self.ik = ik
        self.alignment = alignment
        self.joint_labels, self.formatted = q_formatted
        self.path_length = path_length
        self.collision_mask = collision_mask
        self.skipped = skipped
        self.note = note


def stack_from_string(stack):
    """Creates a PositioningStack object from the given string representation

    :param stack: string representation of stack
    :type stack: str
    :return: positioning
    :rtype: PositioningStack
    """
    robots = []
    fake_mesh = Mesh(np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]), np.array([0, 1, 2]))
    set_points = []
    limit_state = []
    lock_state = []

    descriptions = json.loads(stack)
    for desc in descriptions['stack']:
        links = []
        set_points.extend(desc['joint_set_points'])
        limit_state.extend(desc['joint_limit_state'])
        lock_state.extend(desc['joint_lock_state'])
        for i in range(len(desc['joint_names'])):
            links.append(
                Link(desc['joint_names'][i], desc['joint_axes'][i], desc['joint_vectors'][i],
                     Link.Type(desc['joint_types'][i]), desc['joint_lower_limits'][i], desc['joint_upper_limits'][i],
                     desc['joint_offsets'][i], fake_mesh if desc['joint_meshes'][i] else None))
        robots.append(
            SerialManipulator(desc['name'],
                              links,
                              base=Matrix44(np.reshape(desc['base'], (4, 4))),
                              tool=Matrix44(np.reshape(desc['tool'], (4, 4))),
                              base_mesh=fake_mesh if desc['mesh'] else None,
                              custom_order=desc['order']))

    stack = PositioningStack('no_name', robots[0])
    for robot in robots[1:]:
        stack.addPositioner(robot)
    stack.fkine(set_points)

    for index, link in enumerate(stack.links):
        link.ignore_limits = limit_state[index]
        link.locked = lock_state[index]

    return stack


def stack_to_string(stack):
    """Creates a string representation of the given PositioningStack object.

    :param stack: positioning stack
    :type stack: PositioningStack
    :return: string representation of stack
    :rtype: str
    """
    output = []
    robots = [stack.fixed, *stack.auxiliary]

    for robot in robots:
        desc = {
            'name': robot.name,
            'base': np.ravel(robot.base).tolist(),
            'tool': np.ravel(robot.tool).tolist(),
            'order': robot.order,
            'mesh': robot.base_mesh is not None,
            'joint_names': [],
            'joint_types': [],
            'joint_axes': [],
            'joint_vectors': [],
            'joint_lower_limits': [],
            'joint_upper_limits': [],
            'joint_meshes': [],
            'joint_set_points': [],
            'joint_lock_state': [],
            'joint_limit_state': [],
            'joint_offsets': []
        }

        for link in robot.links:
            desc['joint_names'].append(link.name)
            desc['joint_types'].append(link.type.value)
            desc['joint_axes'].append(np.array(link.joint_axis).tolist())
            desc['joint_vectors'].append(np.array(link.home).tolist())
            desc['joint_set_points'].append(link.set_point)
            desc['joint_offsets'].append(link.default_offset)
            desc['joint_lower_limits'].append(link.lower_limit)
            desc['joint_upper_limits'].append(link.upper_limit)
            desc['joint_lock_state'].append(link.locked)
            desc['joint_limit_state'].append(link.ignore_limits)
            desc['joint_meshes'].append(link.mesh is not None)

        output.append(desc)

    return json.dumps({'stack': output})


class Simulation(QtCore.QObject):
    """Simulates the experiment by computing inverse kinematics of positioning system to place measurement
    points in the gauge volume with the appropriate orientation. The simulation is performed on a different
    process to avoid freezing the main thread and a signal is sent when new results are available.

    :param instrument: instrument object
    :type instrument: Instrument
    :param sample: sample meshes
    :type sample: Dict[Mesh]
    :param points: measurement points
    :type points: numpy.recarray
    :param vectors: measurement vectors
    :type vectors: numpy.ndarray
    :param alignment: alignment matrix
    :type alignment: Matrix44
    """
    result_updated = QtCore.pyqtSignal(bool)
    stopped = QtCore.pyqtSignal()

    def __init__(self, instrument, sample, points, vectors, alignment):
        super().__init__()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.checkResult)

        self.args = {
            'ikine_kwargs': {
                'local_max_eval': settings.value(settings.Key.Local_Max_Eval),
                'global_max_eval': settings.value(settings.Key.Global_Max_Eval),
                'tol': (settings.value(settings.Key.Position_Stop_Val), settings.value(settings.Key.Angular_Stop_Val)),
                'bounded': True
            },
            'skip_zero_vectors': settings.value(settings.Key.Skip_Zero_Vectors),
            'align_first_order': settings.value(settings.Key.Align_First)
        }
        self.results = []
        self.process = None
        self.compute_path_length = False
        self.render_graphics = False
        self.check_limits = True
        self.check_collision = False
        self.has_valid_result = False

        self.positioner_name = instrument.positioning_stack.name
        desc = stack_to_string(instrument.positioning_stack)
        self.args['positioner'] = desc.encode()

        self.shape = (vectors.shape[0], vectors.shape[1] // 3, vectors.shape[2])
        self.count = self.shape[0] * self.shape[2]
        self.args['results'] = Queue(self.count + 1)
        self.args['exit_event'] = Event()

        matrix = alignment.transpose()
        self.args['points'] = SharedArray.fromNumpyArray(points.points @ matrix[0:3, 0:3] + matrix[3, 0:3])
        self.args['enabled'] = SharedArray.fromNumpyArray(points.enabled)
        self.args['vectors'] = np.zeros(vectors.shape, vectors.dtype)
        for k in range(self.args['vectors'].shape[2]):
            for j in range(0, self.args['vectors'].shape[1], 3):
                self.args['vectors'][:, j:j + 3, k] = vectors[:, j:j + 3, k] @ matrix[0:3, 0:3]
        self.args['vectors'] = SharedArray.fromNumpyArray(self.args['vectors'])
        self.args['sample'] = []
        for mesh in sample.values():
            temp = mesh.transformed(alignment)
            self.args['sample'].append(SharedArray.fromNumpyArray(temp.vertices[temp.indices]))

        self.args['beam_axis'] = SharedArray.fromNumpyArray(np.array(instrument.jaws.beam_direction))
        self.args['gauge_volume'] = SharedArray.fromNumpyArray(np.array(instrument.gauge_volume))
        self.args['q_vectors'] = SharedArray.fromNumpyArray(np.array(instrument.q_vectors))
        self.args['diff_axis'] = SharedArray.fromNumpyArray(
            np.array([d.diffracted_beam for d in instrument.detectors.values()]))
        self.args['beam_in_gauge'] = instrument.beam_in_gauge_volume
        self.detector_names = list(instrument.detectors.keys())
        self.params = self.extractInstrumentParameters(instrument)

        self.args['instrument_scene'] = SharedInstrument(instrument)

    def extractInstrumentParameters(self, instrument):
        """Extracts detector and jaws state

        :param instrument: instrument object
        :type instrument: Instrument
        :return: dict containing indicates if the instrument state has not changed
        :rtype: Dict
        """
        params = {}
        for key, detector in instrument.detectors.items():
            if detector.positioner is not None:
                params[f'{Attributes.Detector.value}_{key}'] = detector.positioner.configuration
            params[f'{Attributes.Detector.value}_{key}_collimator'] = ''
            if detector.current_collimator is not None:
                params[f'{Attributes.Detector.value}_{key}_collimator'] = detector.current_collimator.name
        if instrument.jaws.positioner is not None:
            params[Attributes.Jaws.value] = instrument.jaws.positioner.configuration

        return params

    def validateInstrumentParameters(self, instrument):
        """Validates if the instrument state have been changed since the simulation was last run

        :param instrument: instrument object
        :type instrument: Instrument
        :return: indicates if the instrument state has not changed
        :rtype: bool
        """
        params = self.extractInstrumentParameters(instrument)
        for key, value in self.params.items():
            if isinstance(value, str):
                if value != params.get(key):
                    return False
            else:
                if not np.allclose(value, params.get(key, []), 0, 0.001):
                    return False

        return True

    @property
    def scene_size(self):
        """Gets scene size for collision detection"""
        return self.args['instrument_scene'].size + len(self.args['sample'])

    @property
    def compute_path_length(self):
        """Gets and sets flag that indicates if path lengths should be computed

        :return: indicates if path length should be computed
        :rtype: bool
        """
        return self.args['compute_path_length']

    @compute_path_length.setter
    def compute_path_length(self, value):
        self.args['compute_path_length'] = value
        if value:
            self.args['path_lengths'] = SharedArray.fromNumpyArray(np.zeros(self.shape, np.float32))

    @property
    def check_collision(self):
        """Gets and sets flag that indicates if collisions should be detected

        :return: indicates if collisions should be detected
        :rtype: bool
        """
        return self.args['check_collision']

    @check_collision.setter
    def check_collision(self, value):
        self.args['check_collision'] = value

    @property
    def render_graphics(self):
        """Gets and sets flag that indicates if graphics rendering is enabled

        :return: indicates if graphics rendering is enabled
        :rtype: bool
        """
        return self.args['render_graphics']

    @render_graphics.setter
    def render_graphics(self, value):
        self.args['render_graphics'] = value

    @property
    def check_limits(self):
        """Gets and sets flag that indicates if hardware limits should be checked

        :return: indicates if hardware limits should be checked
        :rtype: bool
        """
        return self.args['ikine_kwargs']['bounded']

    @check_limits.setter
    def check_limits(self, value):
        self.args['ikine_kwargs']['bounded'] = value

    def start(self):
        """Starts the simulation"""
        self.process = Process(target=Simulation.execute, args=(self.args, ))
        self.process.daemon = True
        self.process.start()
        self.timer.start()

    def checkResult(self):
        """Checks for and notifies when result are availablebnbn dswawd dfgsfgs"""
        queue = self.args['results']
        print('init size: ', queue.qsize())
        if not self.process.is_alive():
            print('process: ', self.process)
            self.timer.stop()

        if queue.empty():
            print('is empty: ', queue.qsize(), self.args['results'].empty())
            return

        queue.put(None)
        error = False
        for result in iter(queue.get, None):
            print('next size: ', queue.qsize())
            if isinstance(result, SimulationResult):
                self.results.append(result)
                if not result.skipped and result.ik.status != IKSolver.Status.Failed:
                    self.has_valid_result = True
            else:
                error = True
                print('error: ', queue.qsize())
        print('finish size: ', queue.qsize())
        self.result_updated.emit(error)

    @staticmethod
    def execute(args):
        """Computes inverse kinematics, path length, and collisions for each measurement in the
        simulation.

        :param args: argument required for the simulation
        :type args: Dict
        """
        setup_logging('simulation.log')
        logger = logging.getLogger(__name__)
        logger.info('Initializing new simulation...')

        q_vec = SharedArray.toNumpyArray(args['q_vectors'])
        beam_axis = SharedArray.toNumpyArray(args['beam_axis'])
        gauge_volume = SharedArray.toNumpyArray(args['gauge_volume'])
        diff_axis = SharedArray.toNumpyArray(args['diff_axis'])
        beam_in_gauge = args['beam_in_gauge']

        results = args['results']
        exit_event = args['exit_event']
        ikine_kwargs = args['ikine_kwargs']

        positioner = stack_from_string(args['positioner'])
        joint_labels = [positioner.links[order].name for order in positioner.order]
        vectors = SharedArray.toNumpyArray(args['vectors'])
        shape = (vectors.shape[0], vectors.shape[1] // 3, vectors.shape[2])
        points = SharedArray.toNumpyArray(args['points'])
        enabled = SharedArray.toNumpyArray(args['enabled'])
        sample = []
        for vertices in args['sample']:
            node = Node()
            node.vertices = SharedArray.toNumpyArray(vertices)
            node.indices = np.arange(vertices.shape[0]).astype(np.uint32)
            sample.append(node)

        compute_path_length = args['compute_path_length']
        render_graphics = args['render_graphics']
        check_collision = args['check_collision']
        if compute_path_length and beam_in_gauge:
            path_lengths = SharedArray.toNumpyArray(args['path_lengths'])

        if check_collision:
            instrument_scene = create_collision_node(args['instrument_scene'])
            manager = CollisionManager(args['instrument_scene'].size + len(args['sample']))
            sample_ids, positioner_ids = populate_collision_manager(manager, sample, instrument_scene)

        skip_zero_vectors = args['skip_zero_vectors']
        if args['align_first_order']:
            order = [(i, j) for i in range(shape[0]) for j in range(shape[2])]
        else:
            order = [(i, j) for j in range(shape[2]) for i in range(shape[0])]

        logger.info(f'Simulation ({shape[0]} points, {shape[2]} alignments) initialized with '
                    f'render graphics: {render_graphics}, check_collision: {check_collision}, compute_path_length: '
                    f'{compute_path_length}, check_limits: {args["ikine_kwargs"]["bounded"]}')
        try:
            for index, ij in enumerate(order):
                i, j = ij
                label = f'# {index + 1} - Point {i + 1}, Alignment {j + 1}' if shape[2] > 1 else f'Point {i + 1}'

                if not enabled[i]:
                    results.put(SimulationResult(label, skipped=True, note='The measurement point is disabled'))
                    logger.info(f'Skipped Point {i+1}, Alignment {j+1} (Point Disabled)')
                    continue

                all_mvs = vectors[i, :, j].reshape(-1, 3)
                selected = np.where(np.linalg.norm(all_mvs, axis=1) > VECTOR_EPS)[0]
                if selected.size == 0:
                    if skip_zero_vectors:
                        results.put(SimulationResult(label, skipped=True, note='The measurement vector is unset'))
                        logger.info(f'Skipped Point {i+1}, Alignment {j+1} (Vector Unset)')
                        continue
                    q_vectors = np.atleast_2d(q_vec[0])
                    measurement_vectors = np.atleast_2d(positioner.pose[0:3, 0:3].transpose() @ q_vec[0])
                else:
                    q_vectors = np.atleast_2d(q_vec[selected])
                    measurement_vectors = np.atleast_2d(all_mvs[selected])

                logger.info(f'Started Point {i+1}, Alignment {j+1}')

                r = positioner.ikine((points[i, :], measurement_vectors), (gauge_volume, q_vectors), **ikine_kwargs)

                if exit_event.is_set():
                    break

                result = SimulationResult(label, r, (joint_labels, positioner.toUserFormat(r.q)), j)
                if r.status != IKSolver.Status.Failed:
                    pose = positioner.fkine(r.q) @ positioner.tool_link

                    if compute_path_length and beam_in_gauge:
                        transformed_sample = Node(sample[0])
                        matrix = pose.transpose()
                        transformed_sample.vertices = sample[0].vertices @ matrix[0:3, 0:3] + matrix[3, 0:3]
                        result.path_length = path_length_calculation(transformed_sample, gauge_volume, beam_axis,
                                                                     diff_axis)
                        path_lengths[i, :, j] = result.path_length

                    if exit_event.is_set():
                        break

                    if check_collision:
                        update_colliders(manager, pose, sample_ids, positioner.model().transforms, positioner_ids)
                        result.collision_mask = manager.collide()

                if exit_event.is_set():
                    break

                results.put(result)
                logger.info(f'Finished Point {i+1}, Alignment {j+1}')

                if exit_event.is_set():
                    break

            logger.info('Simulation Finished')
        except Exception:
            results.put('Error')
            logging.exception('An error occurred while running the simulation.')

        logging.shutdown()

    @property
    def path_lengths(self):
        """Gets computed path lengths when compute_path_length is enabled

        :return: computed path length
        :rtype: Union[numpy.ndarray, None]
        """
        if self.compute_path_length:
            return SharedArray.toNumpyArray(self.args['path_lengths'])

        return None

    def isRunning(self):
        """Indicates if the simulation is running.

        :return: indicates if the simulation is running
        :rtype: bool
        """
        if self.process is None:
            return False

        return self.process.is_alive() and not self.args['exit_event'].is_set()

    def abort(self):
        """Aborts the simulation, but not guaranteed to be instantaneous."""
        self.args['exit_event'].set()
        self.timer.stop()
        self.stopped.emit()
