from collections import namedtuple
import unittest
import unittest.mock as mock
import numpy as np
from sscanss.core.geometry import create_cuboid, create_cylinder
from sscanss.core.instrument.simulation import Simulation, stack_to_string, stack_from_string, SharedArray
from sscanss.core.instrument.collision import CollisionManager
from sscanss.core.instrument.instrument import PositioningStack, Instrument
from sscanss.core.instrument.robotics import SerialManipulator, Link, IKSolver
from sscanss.core.math import Matrix44
from sscanss.core.util import POINT_DTYPE
from tests.helpers import APP


class TestCollisionClass(unittest.TestCase):
    def testManager(self):
        manager = CollisionManager(5)
        self.assertEqual(manager.max_size, 5)

        geometry = [create_cuboid(), create_cuboid()]
        transform = [Matrix44.identity(), Matrix44.fromTranslation([0, 0, 0.5])]
        manager.addColliders(geometry, transform, movable=True)

        self.assertEqual(len(manager.queries), 2)
        self.assertEqual(len(manager.colliders), 2)

        manager.addColliders([create_cuboid()], [Matrix44.fromTranslation([0, 0, 2.0])])

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
    def setUp(self):
        mock_instrument_entity = self.createMock("sscanss.core.instrument.simulation.InstrumentEntity")
        self.mock_process = self.createMock("sscanss.core.instrument.simulation.Process")
        self.mock_logging = self.createMock("sscanss.core.instrument.simulation.logging")

        self.mock_process.is_alive.return_value = False

        Collimator = namedtuple("Collimator", ["name"])
        Jaws = namedtuple("Jaws", ["beam_direction", "positioner"])
        Detector = namedtuple("Detector", ["diffracted_beam", "positioner", "current_collimator"])
        self.mock_instrument = mock.create_autospec(Instrument)
        self.mock_instrument.positioning_stack = self.createPositioningStack()
        self.mock_instrument.jaws = Jaws([1.0, 0.0, 0.0], self.createPositioner())
        self.mock_instrument.gauge_volume = [0.0, 0.0, 0.0]
        self.mock_instrument.q_vectors = [[-0.70710678, 0.70710678, 0.0], [-0.70710678, -0.70710678, 0.0]]
        self.mock_instrument.detectors = {
            "North": Detector([0.0, 1.0, 0.0], self.createPositioner(), Collimator("4mm")),
            "South": Detector([0.0, -1.0, 0.0], None, Collimator("2mm")),
        }
        self.mock_instrument.beam_in_gauge_volume = True

        meshes = None
        offsets = []
        transforms = []
        for mesh, transform in self.mock_instrument.positioning_stack.model():
            if meshes is None:
                meshes = mesh
            else:
                meshes.append(meshes)
            transforms.append(transform)
            offsets.append(len(meshes.indices))
        beam_stop = create_cuboid(100, 100, 100)
        beam_stop.translate([0.0, 100.0, 0.0])
        transforms.append(Matrix44.identity())
        meshes.append(beam_stop)
        offsets.append(len(meshes.indices))

        mock_instrument_entity.return_value.vertices = meshes.vertices
        mock_instrument_entity.return_value.indices = meshes.indices
        mock_instrument_entity.return_value.transforms = transforms
        mock_instrument_entity.return_value.offsets = offsets
        mock_instrument_entity.return_value.keys = {"Positioner": 2, "Beam_stop": 3}

        self.sample = {"sample": create_cuboid(50.0, 100.000, 200.000)}
        self.points = np.rec.array(
            [([0.0, -90.0, 0.0], True), ([0.0, 0.0, 0.0], True), ([0.0, 90.0, 0.0], True), ([0.0, 0.0, 10.0], False)],
            dtype=POINT_DTYPE,
        )
        self.vectors = np.zeros((4, 6, 1), dtype=np.float32)
        self.alignment = Matrix44.identity()

    def createMock(self, module):
        patcher = mock.patch(module, autospec=True)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def createPositioner():
        q1 = Link("X", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0)
        return SerialManipulator("", [q1])

    @staticmethod
    def createPositioningStack():
        y_axis = create_cuboid(200, 10, 200)
        z_axis = create_cylinder(25, 50)
        q1 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0, z_axis)
        q2 = Link("Y", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0, y_axis)
        s = SerialManipulator("", [q1, q2], custom_order=[1, 0], base=Matrix44.fromTranslation([0.0, 0.0, 50.0]))
        return PositioningStack(s.name, s)

    def testSimulation(self):
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, self.alignment)
        self.assertFalse(simulation.isRunning())

        self.assertTrue(simulation.validateInstrumentParameters(self.mock_instrument))
        detectors = self.mock_instrument.detectors
        detectors["North"], detectors["South"] = detectors["South"], detectors["North"]
        self.assertFalse(simulation.validateInstrumentParameters(self.mock_instrument))
        detectors["South"], detectors["North"] = detectors["North"], detectors["South"]
        self.assertTrue(simulation.validateInstrumentParameters(self.mock_instrument))
        self.mock_instrument.jaws.positioner.fkine([-10.0])
        self.assertFalse(simulation.validateInstrumentParameters(self.mock_instrument))

        self.assertIs(simulation.positioner_name, self.mock_instrument.positioning_stack.name)
        self.assertEqual(simulation.scene_size, 4)

        simulation.execute(simulation.args)

        result_q = simulation.args["results"]
        self.assertEqual(result_q.qsize(), 4)
        self.assertEqual(len(simulation.results), 0)

        simulation.process = self.mock_process
        self.assertFalse(simulation.has_valid_result)
        simulation.checkResult()
        print('Fail point')
        self.assertEqual(result_q.qsize(), 0)
        self.assertEqual(len(simulation.results), 4)
        self.assertTrue(simulation.has_valid_result)
        simulation.checkResult()
        self.assertEqual(result_q.qsize(), 0)

        results = [[0.0, 90.0], [0.0, 0.0], [0.0, -90.0]]
        for exp, result in zip(results, simulation.results[:3]):
            self.assertFalse(result.skipped)
            self.assertEqual(result.note, "")
            self.assertTrue(result.ik.position_converged)
            self.assertTrue(result.ik.orientation_converged)
            np.testing.assert_array_almost_equal(exp, result.ik.q, decimal=2)
            self.assertIsNone(result.path_length)
            self.assertIsNone(result.collision_mask)

        skipped_result = simulation.results[3]
        self.assertTrue(skipped_result.skipped)
        self.assertEqual(skipped_result.note, "The measurement point is disabled")

        self.assertIsNone(simulation.path_lengths)
        self.mock_process.is_alive.return_value = True
        simulation.start()
        self.mock_process.return_value.start.assert_called_once()
        self.assertTrue(simulation.timer.isActive())
        self.assertTrue(simulation.isRunning())

        simulation.abort()
        self.assertFalse(simulation.isRunning())
        self.assertTrue(simulation.args["exit_event"].is_set())
        self.assertFalse(simulation.timer.isActive())
        simulation.execute(simulation.args)
        self.assertEqual(result_q.qsize(), 0)

        mock_stack = self.createMock("sscanss.core.instrument.simulation.PositioningStack")
        mock_stack.return_value.ikine.side_effect = Exception()
        simulation.execute(simulation.args)
        self.mock_logging.exception.assert_called_once()
        self.assertEqual(result_q.get(), "Error")
        result_q.put(-1)
        count = len(simulation.results)
        simulation.checkResult()
        self.assertEqual(len(simulation.results), count)

    def testSimulationWithCollision(self):
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, self.alignment)
        self.assertTrue(simulation.check_limits)
        self.assertFalse(simulation.check_collision)
        simulation.check_limits = False
        simulation.check_collision = True
        simulation.execute(simulation.args)

        simulation.process = self.mock_process
        simulation.checkResult()

        results = [[True, True, True, True], [True, True, False, True], [True, True, False, False]]
        for exp, result in zip(results, simulation.results[:3]):
            self.assertListEqual(result.collision_mask, exp)

        skipped_result = simulation.results[3]
        self.assertTrue(skipped_result.skipped)
        self.assertIsNone(skipped_result.collision_mask)

    def testSimulationWithPathLength(self):
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, self.alignment)
        self.assertFalse(simulation.compute_path_length)
        self.assertFalse(simulation.render_graphics)
        simulation.compute_path_length = True
        simulation.render_graphics = True
        simulation.execute(simulation.args)

        simulation.process = self.mock_process
        simulation.checkResult()

        results = [[215.0, 35.0], [125.0, 125.0], [35.0, 215.0], [0.0, 0.0]]
        for exp, result in zip(results[:3], simulation.results[:3]):
            np.testing.assert_array_almost_equal(exp, result.path_length, decimal=2)

        skipped_result = simulation.results[3]
        self.assertTrue(skipped_result.skipped)
        self.assertIsNone(skipped_result.path_length)

        np.testing.assert_almost_equal(simulation.path_lengths[:, :, 0], results, decimal=2)

    def testSimulationWithVectorAlignment(self):
        self.points = np.rec.array([([0.0, -100.0, 0.0], True), ([0.0, 100.0, 0.0], True)], dtype=POINT_DTYPE)
        self.vectors = np.zeros((2, 6, 2), dtype=np.float32)
        alignment = Matrix44([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.vectors[:, 0:3, 1] = alignment[:3, :3].transpose() @ np.array(self.mock_instrument.q_vectors[0])
        self.vectors[:, 3:6, 1] = alignment[:3, :3].transpose() @ np.array(self.mock_instrument.q_vectors[1])
        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, alignment)
        simulation.args["align_first_order"] = True
        simulation.execute(simulation.args)
        simulation.process = self.mock_process
        simulation.checkResult()

        results = [(0.0, 100.0), (0, 100.0), (0, -100.0), (0, -100.0)]
        for exp, result in zip(results, simulation.results):
            self.assertTrue(result.ik.orientation_converged)
            self.assertTrue(result.ik.position_converged)
            np.testing.assert_array_almost_equal(exp, result.ik.q, decimal=2)

        simulation.results.clear()
        self.mock_instrument.positioning_stack.fixed.resetOffsets()
        simulation.args["align_first_order"] = False
        simulation.execute(simulation.args)
        simulation.checkResult()

        results = [(0.0, 100.0), (0.0, -100.0), (0, 100.0), (0, -100.0)]
        for exp, result in zip(results, simulation.results):
            self.assertTrue(result.ik.orientation_converged)
            self.assertTrue(result.ik.position_converged)
            np.testing.assert_array_almost_equal(exp, result.ik.q, decimal=2)

        simulation = Simulation(self.mock_instrument, self.sample, self.points, self.vectors, alignment)
        self.mock_instrument.positioning_stack.fixed.resetOffsets()
        simulation.args["align_first_order"] = False
        simulation.args["skip_zero_vectors"] = True
        simulation.execute(simulation.args)
        simulation.process = self.mock_process
        simulation.checkResult()
        for result in simulation.results[:2]:
            self.assertTrue(result.skipped)
            self.assertEqual(result.note, "The measurement vector is unset")

        results = [(0, 100.0), (0, -100.0)]
        for exp, result in zip(results, simulation.results[2:]):
            self.assertTrue(result.ik.orientation_converged)
            self.assertTrue(result.ik.position_converged)
            np.testing.assert_array_almost_equal(exp, result.ik.q, decimal=2)

    def testSimulationEdgeCases(self):
        points = np.rec.array(
            [
                ([0.0, -100.0, 0.0], True),
                ([0.0, -100.0, 0.0], True),
                ([0.0, 100.0, 0.0], True),
                ([0.0, 210.0, 0.0], True),
                ([-10.0, 10.0, 0.0], True),
                ([0.0, 100.0, 0.0], True),
            ],
            dtype=POINT_DTYPE,
        )
        vectors = np.zeros((6, 6, 1), dtype=np.float32)
        vectors[:, 0:3, 0] = np.array(self.mock_instrument.q_vectors[0])
        vectors[:, 3:6, 0] = np.array(self.mock_instrument.q_vectors[1])
        vectors[1, 3:6, 0] = -vectors[1, 3:6, 0]
        vectors[2, 3:6, 0] = np.array([-1.0, 0.0, 0.0])
        vectors[5, 0:3, 0] = np.array([0.0, 0.0, 0.0])
        vectors[5, 3:6, 0] = np.array([0.0, 0.0, 1.0])

        simulation = Simulation(self.mock_instrument, self.sample, points, vectors, self.alignment)
        simulation.execute(simulation.args)
        simulation.process = self.mock_process
        simulation.checkResult()

        self.assertEqual(simulation.results[0].ik.status, IKSolver.Status.Converged)
        self.assertEqual(simulation.results[1].ik.status, IKSolver.Status.Unreachable)
        self.assertEqual(simulation.results[2].ik.status, IKSolver.Status.DeformedVectors)
        self.assertEqual(simulation.results[3].ik.status, IKSolver.Status.HardwareLimit)
        self.assertEqual(simulation.results[4].ik.status, IKSolver.Status.NotConverged)
        self.assertEqual(simulation.results[5].ik.status, IKSolver.Status.Unreachable)

        points = np.rec.array([([0.0, -100.0, 0.0], True), ([0.0, -210.0, 0.0], True)], dtype=POINT_DTYPE)
        vectors = np.zeros((2, 6, 1), dtype=np.float32)
        vectors[0, 0:3, 0] = np.array(self.mock_instrument.q_vectors[0])
        vectors[0, 3:6, 0] = -np.array(self.mock_instrument.q_vectors[1])

        self.mock_instrument.positioning_stack.links[0].locked = True
        simulation = Simulation(self.mock_instrument, self.sample, points, vectors, self.alignment)
        simulation.execute(simulation.args)
        simulation.process = self.mock_process
        simulation.checkResult()
        self.assertEqual(simulation.results[0].ik.status, IKSolver.Status.Unreachable)
        self.assertEqual(simulation.results[1].ik.status, IKSolver.Status.HardwareLimit)

        q1 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        q2 = Link("Z2", [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        q3 = Link("Y", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0)
        s = SerialManipulator("", [q1, q2, q3], custom_order=[2, 1, 0], base=Matrix44.fromTranslation([0.0, 0.0, 50.0]))
        simulation.args["positioner"] = stack_to_string(PositioningStack(s.name, s))
        simulation.results = []
        simulation.execute(simulation.args)
        simulation.checkResult()
        self.assertEqual(simulation.results[0].ik.status, IKSolver.Status.Unreachable)

        q2 = Link("X", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        s = SerialManipulator("", [q1, q2, q3], custom_order=[2, 1, 0], base=Matrix44.fromTranslation([0.0, 0.0, 50.0]))
        simulation.args["positioner"] = stack_to_string(PositioningStack(s.name, s))
        simulation.results = []
        simulation.execute(simulation.args)
        simulation.checkResult()
        self.assertEqual(simulation.results[0].ik.status, IKSolver.Status.NotConverged)


class TestSimulationHelpers(unittest.TestCase):
    def testSharedArray(self):
        # Tests for SharedArray
        data = [[1, 2], [3, 4], [5, 6]]
        shared = SharedArray.fromNumpyArray(np.array(data, np.float32))
        array = SharedArray.toNumpyArray(shared)
        self.assertEqual(array.shape, (3, 2))
        self.assertEqual(array.dtype, np.float32)
        np.testing.assert_array_equal(data, array)

        shared = SharedArray.fromNumpyArray(np.array(data, np.float64))
        array = SharedArray.toNumpyArray(shared)
        self.assertEqual(array.shape, (3, 2))
        self.assertEqual(array.dtype, np.float64)
        np.testing.assert_array_equal(data, array)

        shared = SharedArray.fromNumpyArray(np.array(data, np.uint32))
        array = SharedArray.toNumpyArray(shared)
        self.assertEqual(array.shape, (3, 2))
        self.assertEqual(array.dtype, np.uint32)
        np.testing.assert_array_equal(data, array)

        shared = SharedArray.fromNumpyArray(np.array(data, np.int32))
        array = SharedArray.toNumpyArray(shared)
        self.assertEqual(array.shape, (3, 2))
        self.assertEqual(array.dtype, np.int32)
        np.testing.assert_array_equal(data, array)

        shared = SharedArray.fromNumpyArray(np.array([[True], [False], [True]], bool))
        array = SharedArray.toNumpyArray(shared)
        self.assertEqual(array.shape, (3, 1))
        self.assertEqual(array.dtype, bool)
        np.testing.assert_array_equal([[True], [False], [True]], array)

        self.assertRaises(ValueError, SharedArray.fromNumpyArray, np.array(data, np.int64))

    def testStackToString(self):
        q1 = Link("X", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -180.0, 120.0, -10)
        first = SerialManipulator("first", [q1])

        y_axis = create_cuboid(200, 10, 200)
        z_axis = create_cylinder(25, 50)
        q1 = Link("Z", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 1.57, z_axis)
        q2 = Link("Y", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -200.0, 200.0, 0, y_axis)
        second = SerialManipulator("second", [q1, q2],
                                   custom_order=[1, 0],
                                   base=Matrix44.fromTranslation([0.0, 0.0, 50.0]))

        stack = PositioningStack("New Stack", first)
        stack.addPositioner(second)
        stack.links[0].ignore_limits = True
        stack.links[1].locked = True

        stack_string = stack_to_string(stack)
        new_stack = stack_from_string(stack_string)

        for l1, l2 in zip(stack.links, new_stack.links):
            self.assertEqual(l1.ignore_limits, l2.ignore_limits)
            self.assertEqual(l1.locked, l2.locked)
            self.assertEqual(l1.type, l2.type)
            self.assertEqual(l1.upper_limit, l2.upper_limit)
            self.assertEqual(l1.lower_limit, l2.lower_limit)
            np.testing.assert_array_almost_equal(l1.home, l2.home)
            np.testing.assert_array_almost_equal(l1.joint_axis, l2.joint_axis)

        np.testing.assert_array_almost_equal(new_stack.set_points, stack.set_points)
        np.testing.assert_array_almost_equal(new_stack.fixed.base, stack.fixed.base)
        np.testing.assert_array_almost_equal(new_stack.fixed.tool, stack.fixed.tool)
        np.testing.assert_array_almost_equal(new_stack.fixed.order, stack.fixed.order)
        np.testing.assert_array_almost_equal(new_stack.auxiliary[0].base, stack.auxiliary[0].base)
        np.testing.assert_array_almost_equal(new_stack.auxiliary[0].tool, stack.auxiliary[0].tool)
        np.testing.assert_array_almost_equal(new_stack.auxiliary[0].order, stack.auxiliary[0].order)


if __name__ == "__main__":
    unittest.main()
