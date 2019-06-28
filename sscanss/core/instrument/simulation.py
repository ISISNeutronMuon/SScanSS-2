import ctypes
import time
import numpy as np
import multiprocessing
from multiprocessing import sharedctypes
from PyQt5 import QtCore
from ..mesh.geometry import path_length_calculation
from ...config import settings

POINT_DTYPE = [('points', 'f4', 3), ('enabled', '?')]


class SimulationResult:
    def __init__(self, result_id,  error, q,  q_formatted, code, path_length):
        self.id = result_id
        self.q = q
        self.error = error
        self.code = code
        self.joint_labels, self.formatted = q_formatted
        self.path_length = path_length


class Simulation(QtCore.QObject):
    result_updated = QtCore.pyqtSignal()

    def __init__(self, positioner, mesh, points, vectors, alignment, ):
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
        self.args['results'] = multiprocessing.SimpleQueue()
        self.args['exit_event'] = multiprocessing.Event()
        self.args['positioner'] = positioner
        self.args['joint_labels'] = [positioner.links[order].name for order in positioner.order]
        self.args['points'] = points.points[points.enabled]
        self.args['vectors'] = vectors[points.enabled, :, :]
        shape = self.args['vectors'].shape
        self.shape = (shape[0], shape[1]//3, shape[2])
        self.count = shape[0] * shape[2]  # count is the number of expected results

        matrix = alignment.transpose()
        self.args['points'] = self.args['points'] @ matrix[0:3, 0:3] + matrix[3, 0:3]
        for k in range(self.args['vectors'].shape[2]):
            for j in range(0, self.args['vectors'].shape[1], 3):
                self.args['vectors'][:, j:j+3, k] = self.args['vectors'][:, j:j+3, k] @ matrix[0:3, 0:3]

        self.args['sample'] = mesh.transformed(alignment)

    @property
    def compute_path_length(self):
        return self.args['compute_path_length']

    @compute_path_length.setter
    def compute_path_length(self, value):
        self.args['compute_path_length'] = value
        if value:
            self.detector_names = ['South', 'North']
            self.args['path_lengths'] = sharedctypes.RawArray(ctypes.c_float, [0.] * np.prod(self.shape))

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
        self.process = multiprocessing.Process(target=self.execute, args=(self.args,))
        self.process.daemon = True
        self.process.start()
        self.timer.start()

    def CheckResult(self):
        is_running = self.isRunning()
        queue = self.args['results']
        if self.args['results'].empty():
            return

        queue.put(None)
        for result in iter(queue.get, None):
            self.results.append(result)

        self.result_updated.emit()
        if not is_running:
            self.timer.stop()

    @staticmethod
    def execute(args):
        q_vec = np.array([[-0.70710678, -0.70710678, 0.], [-0.70710678, 0.70710678, 0.]])
        beam_axis = np.array([1.0, 0.0, 0.0])
        beam_origin = np.array([-1000.0, 0.0, 0.0])
        beam_length = 1000
        diff_axis = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        diff_origin = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        diff_length = [1000, 1000]

        results = args['results']
        exit_event = args['exit_event']
        ikine_kwargs = args['ikine_kwargs']

        positioner = args['positioner']
        vectors = args['vectors']
        shape = (vectors.shape[0], vectors.shape[1]//3,  vectors.shape[2])
        points = args['points']
        compute_path_length = args['compute_path_length']
        render_graphics = args['render_graphics']
        if compute_path_length:
            path_lengths = np.frombuffer(args['path_lengths'],
                                          dtype=np.float32, count=np.prod(shape)).reshape(shape)
            sample = args['sample']

        if args['align_first_order']:
            order = [(i, j) for i in range(shape[0]) for j in range(shape[2])]
        else:
            order = [(i, j) for j in range(shape[2]) for i in range(shape[0])]
        #try:
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

            r, error, code = positioner.ikine([points[i, :], measurement_vectors],
                                              [np.array([0., 0., 0.]), q_vectors], **ikine_kwargs)
            if exit_event.is_set():
                break

            pose = positioner.fkine(r) @ positioner.tool_link

            length = None
            if compute_path_length:
                transformed_sample = sample.transformed(pose)
                length = path_length_calculation(transformed_sample, beam_axis, beam_origin, beam_length,
                                                 diff_axis, diff_origin, diff_length)

                path_lengths[i, :, j] = length

            if exit_event.is_set():
                break
            label = f'# {index+1} - Point {i+1}, Alignment {j+1}' if shape[2] > 1 else f'Point {i+1}'
            results.put(SimulationResult(label, error, r, (args['joint_labels'], positioner.toUserFormat(r)),
                                         code, length))
            if render_graphics:
                # Sleep to allow graphics render
                time.sleep(0.2)


        # except Exception:
        #     # TODO: add proper exception handling for Value error and  Memory error
        #     print('error')


    @property
    def path_lengths(self):
        if self.compute_path_length:
            return np.frombuffer(self.args['path_lengths'], dtype=np.float32,
                                 count=np.prod(self.shape)).reshape(self.shape)

        return None

    def isRunning(self):
        if self.process is None:
            return False

        return self.process.is_alive()

    def abort(self):
        self.args['exit_event'].set()