from collections import namedtuple
import unittest
import unittest.mock as mock
import numpy as np
from PyQt5.QtWidgets import QApplication
from sscanss.core.geometry import create_cuboid, create_cylinder
from sscanss.core.instrument import Simulation, Instrument
from sscanss.core.instrument.collision import CollisionManager
from sscanss.core.instrument.instrument import PositioningStack
from sscanss.core.instrument.robotics import SerialManipulator, Link
from sscanss.core.scene import Node
from sscanss.core.math import Matrix44
from sscanss.core.util import POINT_DTYPE


class TestCollisionClass(unittest.TestCase):

    def testManager(self):
        manager = CollisionManager(5)
        self.assertEqual(manager.max_size, 5)

        geometry = [create_cuboid(), create_cuboid()]
        transform = [Matrix44.identity(), Matrix44.fromTranslation([0, 0, 0.5])]
        manager.addColliders(geometry, transform, movable=True)

        self.assertEqual(len(manager.queries), 2)
        self.assertEqual(len(manager.colliders), 2)

        manager.addColliders([create_cuboid()], [Matrix44.fromTranslation([0, 0, 2.])])

        self.assertEqual(len(manager.queries), 2)
        self.assertEqual(len(manager.colliders), 3)

        manager.createAABBSets()
        self.assertListEqual(manager.collide(), [True, True, False])

        manager.clear()
        self.assertEqual(len(manager.queries), 0)
        self.assertEqual(len(manager.colliders), 0)

        geometry = [create_cuboid(), create_cuboid(), create_cuboid()]
        transform = [Matrix44.identity(), Matrix44.fromTranslation([0, 0, 0.5]), Matrix44.fromTranslation([0, 0, 1.5])]
        manager.addColliders(geometry, transform, CollisionManager.Exclude.Consecutive, movable=True)
        manager.createAABBSets()
        self.assertListEqual(manager.collide(), [False, False, False])

        manager.clear()
        transform = [Matrix44.identity(), Matrix44.identity(), Matrix44.identity()]
        manager.addColliders(geometry, transform, CollisionManager.Exclude.All, movable=True)
        manager.createAABBSets()
        self.assertListEqual(manager.collide(), [False, False, False])


class TestSimulation(unittest.TestCase):
    app = QApplication([])

    def setUp(self):
        mock_fn_create_instrument_node = self.createMock('sscanss.core.instrument.simulation.createInstrumentNode')
        self.mock_process = self.createMock('sscanss.core.instrument.simulation.Process')
        self.mock_logging = self.createMock('sscanss.core.instrument.simulation.logging')
        self.mock_time = self.createMock('sscanss.core.instrument.simulation.time')

        self.mock_process.is_alive.return_value = False

        Jaws = namedtuple('Jaws', ['beam_direction'])
        Detector = namedtuple('Detector', ['diffracted_beam'])
        self.mock_instrument = mock.create_autospec(Instrument)
        self.mock_instrument.positioning_stack = self.createPositioningStack()
        self.mock_instrument.jaws = Jaws([1.0, 0.0, 0.0])
        self.mock_instrument.gauge_volume = [0.0, 0.0, 0.0]
        self.mock_instrument.q_vectors = [[-0.70710678, 0.70710678, 0.], [-0.70710678, -0.70710678, 0.]]
        self.mock_instrument.detectors = {"North": Detector([0., 1., 0.]), "South": Detector([0., -1., 0.])}
        self.mock_instrument.beam_in_gauge_volume = True

        node = self.mock_instrument.positioning_stack.model()
        beam_stop = create_cuboid(100, 100, 100)
        beam_stop.translate([0., 100., 0.])
        node.addChild(Node(beam_stop))
        mock_fn_create_instrument_node.return_value = (node, {'Positioner': 2, 'Beam_stop': 3})

        self.sample = {'sample': create_cuboid(50.0, 100.000, 200.000)}
        self.points = np.rec.array([([0., -90., 0.], True), ([0., 0., 0.], True), ([0., 90., 0.], True)],
                                   dtype=POINT_DTYPE)
        self.vectors = np.zeros((3, 6, 1), dtype=np.float32)
        self.alignment = Matrix44.identity()

    def createMock(self, module):
        patcher = mock.patch(module, autospec=True)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def createPositioningStack():
        y_axis = create_cuboid(200, 10, 200)
        z_axis = create_cylinder(25, 50)
        q1 = Link('Z', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0, z_axis)
        q2 = Link('Y', [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200., 200., 0, y_axis)
        s = SerialManipulator('', [q1, q2], custom_order=[1, 0], base=Matrix44.fromTranslation([0., 0., 50.]))
        return PositioningStack(s.name, s)

    def testSimulation(self):
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, self.alignment)
        self.assertFalse(simulation.isRunning())

        simulation.execute(simulation.args)

        result_q = simulation.args['results']
        self.assertEqual(result_q.qsize(), 3)
        self.assertEqual(len(simulation.results), 0)

        simulation.process = self.mock_process
        simulation.checkResult()

        self.assertEqual(result_q.qsize(), 0)
        self.assertEqual(len(simulation.results), 3)
        simulation.checkResult()
        self.assertEqual(result_q.qsize(), 0)

        results = [[0., 90.], [0., 0.], [0., -90.]]
        for exp, result in zip(results, simulation.results):
            self.assertTrue(result.ik.position_converged)
            self.assertTrue(result.ik.orientation_converged)
            np.testing.assert_array_almost_equal(exp, result.ik.q, decimal=2)
            self.assertIsNone(result.path_length)
            self.assertIsNone(result.collision_mask)

        self.assertIsNone(simulation.path_lengths)
        self.mock_process.is_alive.return_value = True
        simulation.start()
        self.mock_process.return_value.start.assert_called_once()
        self.assertTrue(simulation.timer.isActive())
        self.assertTrue(simulation.isRunning())

        simulation.abort()
        self.assertFalse(simulation.isRunning())
        self.assertTrue(simulation.args['exit_event'].is_set())
        self.assertFalse(simulation.timer.isActive())
        simulation.execute(simulation.args)
        self.assertEqual(result_q.qsize(), 0)

        self.mock_instrument.positioning_stack.fkine = mock.Mock(side_effect=Exception)
        simulation.execute(simulation.args)
        self.mock_logging.exception.assert_called_once()
        self.assertEqual(simulation.args['results'].get(), "Error")

    def testSimulationWithCollision(self):
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, self.alignment)
        self.assertEqual(simulation.scene_size, 4)
        self.assertTrue(simulation.check_limits)
        self.assertFalse(simulation.check_collision)
        simulation.check_limits = False
        simulation.check_collision = True
        simulation.execute(simulation.args)

        simulation.process = self.mock_process
        simulation.checkResult()

        results = [[True, True, True, True], [True, True, False, True], [True, True, False, False]]
        for exp, result in zip(results, simulation.results):
            self.assertListEqual(result.collision_mask, exp)

    def testSimulationWithPathLength(self):
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, self.alignment)
        self.assertFalse(simulation.compute_path_length)
        self.assertFalse(simulation.render_graphics)
        simulation.compute_path_length = True
        simulation.render_graphics = True
        simulation.execute(simulation.args)

        simulation.process = self.mock_process
        simulation.checkResult()

        results = [[215., 35.], [125., 125.], [35., 215.]]
        for exp, result in zip(results, simulation.results):
            np.testing.assert_array_almost_equal(exp, result.path_length, decimal=2)

        np.testing.assert_almost_equal(simulation.path_lengths[:, :, 0], results, decimal=2)
        self.assertEqual(self.mock_time.sleep.call_count, self.points.size)

    def testSimulationWithVectorAlignment(self):
        self.points = np.rec.array([([0., -90., 0.], True), ([0., 90., 0.], True)], dtype=POINT_DTYPE)
        self.vectors = np.zeros((2, 6, 2), dtype=np.float32)
        self.vectors[:, 0:3, 1] = -np.array(self.mock_instrument.q_vectors[0])
        self.vectors[:, 3:6, 1] = -np.array(self.mock_instrument.q_vectors[1])
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, self.alignment)
        simulation.args['align_first_order'] = True
        simulation.execute(simulation.args)
        simulation.process = self.mock_process
        simulation.checkResult()

        results = [(0., 90.), (3.14, 90.), (3.14, -90.), (3.14, -90.)]
        for exp, result in zip(results, simulation.results):
            np.testing.assert_array_almost_equal(exp, result.ik.q, decimal=2)

        simulation.results.clear()
        self.mock_instrument.positioning_stack.fixed.resetOffsets()
        simulation.args['align_first_order'] = False
        simulation.execute(simulation.args)
        simulation.process = self.mock_process
        simulation.checkResult()

        results = [(0., 90.), (0., -90.), (3.14, 90.), (3.14, -90.)]
        for exp, result in zip(results, simulation.results):
            np.testing.assert_array_almost_equal(exp, result.ik.q, decimal=2)
