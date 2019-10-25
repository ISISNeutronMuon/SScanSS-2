import time
import numpy as np
from multiprocessing import Event, Process, Queue, sharedctypes
from PyQt5 import QtCore
from .collision import CollisionManager
from ..geometry.intersection import path_length_calculation
from ..scene.node import createInstrumentNode
from ..util.misc import Attributes
from ...config import settings


def update_colliders(manager, sample_pose, sample_ids, positioner_nodes, positioner_ids):
    """ Updates the sample and positioner colliders

    :param manager: collision manager
    :type manager: CollisionManager
    :param sample_pose: sample transformation matrix
    :type sample_pose: Matrix44
    :param sample_ids: list of sample collider ids
    :type sample_ids: List[int]
    :param positioner_nodes: list of positioner nodes
    :type positioner_nodes: List[Node]
    :param positioner_ids: list of positioner ids
    :type positioner_ids: List[int]
    """
    for i in sample_ids:
        manager.colliders[i].geometry.transform(sample_pose)

    for i, node in zip(positioner_ids, positioner_nodes):
        manager.colliders[i].geometry.transform(node.transform)

    manager.createAABBSets()


def populate_collision_manager(manager, sample, sample_pose, instrument_node):
    """Adds sample and instrument scene colliders to the collision manager and builds
    scene bounding boxes

    :param manager: collision manager
    :type manager: CollisionManager
    :param sample: list of sample mesh
    :type sample: List[Mesh]
    :param sample_pose: sample transformation matrix
    :type sample_pose: matrix44
    :param instrument_node: instrument node and ids
    :type instrument_node: Tuple[List[Node], Dict[str, int]]
    :return: sample and positioner collider ids
    :rtype: Tuple[List[int], List[int]]
    """
    manager.clear()
    transform = [sample_pose for _ in range(len(sample))]
    manager.addColliders(sample, transform, manager.Exclude.All, True)
    sample_ids = list(range(len(sample)))
    positioner_ids = []

    node, indices = instrument_node

    start_index = 0
    for name, end_index in indices.items():
        attribute_node = node.children[start_index:end_index]
        if name == Attributes.Positioner.value:
            start_id = manager.colliders[-1].id + 1
            transform = [n.transform for n in attribute_node]
            manager.addColliders(attribute_node, transform, exclude=manager.Exclude.Consecutive, movable=True)
            last_link_collider = manager.colliders[-1]
            for index, obj in enumerate(manager.colliders[0:len(sample)]):
                obj.excludes[last_link_collider.id] = True
                last_link_collider.excludes[index] = True

            positioner_ids.extend(range(start_id, last_link_collider.id + 1))
        else:
            manager.addColliders(attribute_node, exclude=manager.Exclude.Consecutive, movable=False)

        start_index = end_index
    manager.createAABBSets()

    return sample_ids, positioner_ids


class SimulationResult:
    def __init__(self, result_id,  error, q,  q_formatted, alignment, path_length, collision_mask):
        self.id = result_id
        self.q = q
        self.error, self.code = error
        self.alignment = alignment
        self.joint_labels, self.formatted = q_formatted
        self.path_length = path_length
        self.collision_mask = collision_mask


class Simulation(QtCore.QObject):
    result_updated = QtCore.pyqtSignal()

    def __init__(self, instrument, sample, points, vectors, alignment):
        super().__init__()

        # Setup Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.CheckResult)

        self.args = {'ikine_kwargs': {'local_max_eval': settings.value(settings.Key.Local_Max_Eval),
                                      'global_max_eval': settings.value(settings.Key.Global_Max_Eval),
                                      'tol': (settings.value(settings.Key.Position_Stop_Val),
                                              settings.value(settings.Key.Angular_Stop_Val)),
                                      'bounded': True},
                     'align_first_order': settings.value(settings.Key.Align_First)}
        self.results = []
        self.process = None
        self.compute_path_length = False
        self.render_graphics = False
        self.check_limits = True
        self.check_collision = False
        self.args['positioner'] = instrument.positioning_stack
        self.args['points'] = points.points[points.enabled]
        self.args['vectors'] = vectors[points.enabled, :, :]
        shape = self.args['vectors'].shape
        self.shape = (shape[0], shape[1]//3, shape[2])
        self.count = shape[0] * shape[2]  # count is the number of expected results
        self.args['results'] = Queue(self.count + 1)
        self.args['exit_event'] = Event()

        matrix = alignment.transpose()
        self.args['points'] = self.args['points'] @ matrix[0:3, 0:3] + matrix[3, 0:3]
        for k in range(self.args['vectors'].shape[2]):
            for j in range(0, self.args['vectors'].shape[1], 3):
                self.args['vectors'][:, j:j+3, k] = self.args['vectors'][:, j:j+3, k] @ matrix[0:3, 0:3]

        self.args['sample'] = []
        for key, mesh in sample.items():
            self.args['sample'].append(mesh.transformed(alignment))

        self.args['beam_axis'] = np.array(instrument.jaws.beam_direction)
        self.args['gauge_volume'] = np.array(instrument.gauge_volume)
        self.args['q_vectors'] = np.array(instrument.q_vectors)
        self.args['diff_axis'] = np.array([d.diffracted_beam for d in instrument.detectors.values()])
        self.args['beam_in_gauge'] = instrument.beam_in_gauge_volume
        self.detector_names = list(instrument.detectors.keys())

        self.args['instrument_scene'] = createInstrumentNode(instrument, True)

    @property
    def compute_path_length(self):
        return self.args['compute_path_length']

    @compute_path_length.setter
    def compute_path_length(self, value):
        self.args['compute_path_length'] = value
        if value:
            self.args['path_lengths'] = sharedctypes.RawArray('f', [0.] * np.prod(self.shape))

    @property
    def check_collision(self):
        return self.args['check_collision']

    @check_collision.setter
    def check_collision(self, value):
        self.args['check_collision'] = value

    @property
    def render_graphics(self):
        return self.args['render_graphics']

    @render_graphics.setter
    def render_graphics(self, value):
        self.args['render_graphics'] = value

    @property
    def check_limits(self):
        return self.args['ikine_kwargs']['bounded']

    @check_limits.setter
    def check_limits(self, value):
        self.args['ikine_kwargs']['bounded'] = value

    def start(self):
        self.process = Process(target=self.execute, args=(self.args,))
        self.process.daemon = True
        self.process.start()
        self.timer.start()

    def CheckResult(self):
        queue = self.args['results']
        if self.args['results'].empty():
            return

        if not self.process.is_alive():
            self.timer.stop()

        queue.put(None)
        for result in iter(queue.get, None):
            self.results.append(result)

        self.result_updated.emit()

    @staticmethod
    def execute(args):
        q_vec = args['q_vectors']
        beam_axis = args['beam_axis']
        gauge_volume = args['gauge_volume']
        diff_axis = args['diff_axis']
        beam_in_gauge = args['beam_in_gauge']

        results = args['results']
        exit_event = args['exit_event']
        ikine_kwargs = args['ikine_kwargs']

        positioner = args['positioner']
        joint_labels = [positioner.links[order].name for order in positioner.order]
        vectors = args['vectors']
        shape = (vectors.shape[0], vectors.shape[1]//3,  vectors.shape[2])
        points = args['points']
        sample = args['sample']
        compute_path_length = args['compute_path_length']
        render_graphics = args['render_graphics']
        check_collision = args['check_collision']
        if compute_path_length and beam_in_gauge:
            path_lengths = np.frombuffer(args['path_lengths'], dtype=np.float32, count=np.prod(shape)).reshape(shape)

        if check_collision:
            instrument_scene = args['instrument_scene']
            manager = CollisionManager(len(instrument_scene[0].children) + len(sample))
            sample_ids, positioner_ids = populate_collision_manager(manager, sample, np.identity(4), instrument_scene)

        if args['align_first_order']:
            order = [(i, j) for i in range(shape[0]) for j in range(shape[2])]
        else:
            order = [(i, j) for j in range(shape[2]) for i in range(shape[0])]
        try:
            for index, ij in enumerate(order):
                i, j = ij
                all_mvs = vectors[i, :, j].reshape(-1, 3)
                selected = np.where(np.linalg.norm(all_mvs, axis=1) > 0.0001)[0]  # greater than epsilon
                if selected.size == 0:
                    q_vectors = np.atleast_2d(q_vec[0])
                    measurement_vectors = np.atleast_2d(positioner.pose[0:3, 0:3].transpose() @ q_vec[0])
                else:
                    q_vectors = np.atleast_2d(q_vec[selected])
                    measurement_vectors = np.atleast_2d(all_mvs[selected])

                r, error, code = positioner.ikine([points[i, :], measurement_vectors], [gauge_volume, q_vectors],
                                                  **ikine_kwargs)

                if exit_event.is_set():
                    break

                pose = positioner.fkine(r) @ positioner.tool_link

                length = None
                if compute_path_length and beam_in_gauge:
                    transformed_sample = sample[0].transformed(pose)
                    length = path_length_calculation(transformed_sample, gauge_volume, beam_axis, diff_axis)

                    path_lengths[i, :, j] = length

                if exit_event.is_set():
                    break

                collision_mask = None
                if check_collision:
                    update_colliders(manager, pose, sample_ids, positioner.model().flatten().children, positioner_ids)
                    collision_mask = manager.collide()

                if exit_event.is_set():
                    break

                label = f'# {index+1} - Point {i+1}, Alignment {j+1}' if shape[2] > 1 else f'Point {i+1}'
                results.put(SimulationResult(label, (error, code), r, (joint_labels, positioner.toUserFormat(r)),
                                             j, length, collision_mask))
                if render_graphics:
                    # Sleep to allow graphics render
                    time.sleep(0.2)

        except Exception:
            import traceback
            results.put(traceback.format_exc())

    @property
    def path_lengths(self):
        if self.compute_path_length:
            return np.frombuffer(self.args['path_lengths'], dtype=np.float32,
                                 count=np.prod(self.shape)).reshape(self.shape)

        return None

    def isRunning(self):
        if self.process is None:
            return False

        return self.process.is_alive() and not self.args['exit_event'].is_set()

    def abort(self):
        self.args['exit_event'].set()
        self.timer.stop()
