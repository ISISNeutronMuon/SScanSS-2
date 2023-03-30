import copy
import json
import unittest
import unittest.mock as mock
from jsonschema.exceptions import ValidationError
import numpy as np
from PyQt6.QtTest import QTest
from sscanss.core.math import Matrix44
from sscanss.core.geometry import Mesh
from sscanss.core.instrument.instrument import PositioningStack, Script
from sscanss.core.instrument.robotics import joint_space_trajectory, Link, SerialManipulator, Sequence
from sscanss.core.instrument.create import (read_instrument_description_file, read_jaw_description, check,
                                            read_positioners_description, read_detector_description,
                                            read_positioning_stacks_description, read_fixed_hardware_description,
                                            read_script_template)
from tests.helpers import SAMPLE_IDF, QTest


class TestInstrument(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        self.mesh = Mesh(vertices, indices, normals)

    @mock.patch("sscanss.core.instrument.create.read_3d_model", autospec=True)
    def testReadIDF(self, read_model_fn):
        read_model_fn.return_value = self.mesh
        idf = json.loads(SAMPLE_IDF)
        instrument = idf["instrument"]
        with mock.patch("sscanss.core.instrument.create.open",
                        mock.mock_open(read_data='{"instrument":{"name": "FAKE"}}')):
            self.assertRaises(ValidationError, read_instrument_description_file, "")

        with mock.patch("sscanss.core.instrument.create.open", mock.mock_open(read_data=SAMPLE_IDF)):
            read_instrument_description_file("")

        instrument["name"] = "None"
        self.assertRaises(ValueError, check, instrument, "name", "instrument", name=True)
        instrument["name"] = ""
        self.assertRaises(ValueError, check, instrument, "name", "instrument", name=True)
        instrument.pop("name")
        self.assertRaises(KeyError, check, instrument, "name", "instrument", name=True)

        positioners = read_positioners_description(instrument)
        instrument["positioners"][0]["name"] = instrument["positioners"][1]["name"]
        self.assertRaises(ValueError, read_positioners_description, instrument)
        self.assertEqual(len(positioners), 4)
        instrument["positioners"][0]["custom_order"] = ["None", "None2", "None3"]
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["custom_order"] = ["None"]
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][2]["parent"] = "None"
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][1]["name"] = "Stage"
        instrument["positioners"][0]["joints"][2]["name"] = "Stage"
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][2]["type"] = "rotating"
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][2]["home_offset"] = 180
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][2]["upper_limit"] = -180
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["links"][1]["name"] = "Link"
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][2]["child"] = "None "
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][1]["child"] = "None "
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["joints"][1]["parent"] = "None "
        self.assertRaises(ValueError, read_positioners_description, instrument)
        instrument["positioners"][0]["links"][1]["name"] = "base"
        self.assertRaises(ValueError, read_positioners_description, instrument)

        instrument["incident_jaws"]["positioner"] = "None"
        self.assertRaises(ValueError, read_jaw_description, instrument, positioners)
        instrument["incident_jaws"]["aperture"] = (1, 20)
        self.assertRaises(ValueError, read_jaw_description, instrument, positioners)
        instrument["incident_jaws"]["aperture"] = (20, 1)
        self.assertRaises(ValueError, read_jaw_description, instrument, positioners)
        instrument["incident_jaws"]["aperture_lower_limit"] = (20, 1)
        self.assertRaises(ValueError, read_jaw_description, instrument, positioners)
        instrument["incident_jaws"]["aperture_lower_limit"] = (-5, 1)
        self.assertRaises(ValueError, read_jaw_description, instrument, positioners)

        instrument["collimators"][2]["detector"] = "None"
        self.assertRaises(ValueError, read_detector_description, instrument, positioners)
        instrument["collimators"][1]["name"] = instrument["collimators"][0]["name"]
        self.assertRaises(ValueError, read_detector_description, instrument, positioners)
        instrument["detectors"].append(instrument["detectors"][0])
        self.assertRaises(ValueError, read_detector_description, instrument, positioners)
        instrument["detectors"][0]["positioner"] = "None"
        self.assertRaises(ValueError, read_detector_description, instrument, positioners)
        instrument["detectors"][0]["default_collimator"] = "None"
        self.assertRaises(ValueError, read_detector_description, instrument, positioners)
        instrument["detectors"][0]["diffracted_beam"] = [1.0, 1.0, 0.0]
        self.assertRaises(ValueError, read_detector_description, instrument, positioners)

        instrument["positioning_stacks"][0]["name"] = instrument["positioning_stacks"][1]["name"]
        self.assertRaises(ValueError, read_positioning_stacks_description, instrument, positioners)
        tmp = instrument["positioning_stacks"][0]["positioners"]
        instrument["positioning_stacks"][0]["name"] = tmp[0]
        instrument["positioning_stacks"][0]["positioners"] = [tmp[0]]
        self.assertEqual(len(read_positioning_stacks_description(instrument, positioners)), 2)
        instrument["positioning_stacks"][0]["name"] = "New"
        instrument["positioning_stacks"][0]["positioners"] = ["None"]
        self.assertRaises(ValueError, read_positioning_stacks_description, instrument, positioners)
        instrument["positioning_stacks"][0]["positioners"] = ["None", "Fake"]
        self.assertRaises(ValueError, read_positioning_stacks_description, instrument, positioners)
        instrument["positioning_stacks"][0]["positioners"] = tmp * 2
        self.assertRaises(ValueError, read_positioning_stacks_description, instrument, positioners)
        instrument["positioning_stacks"][0]["name"] = tmp[0]
        instrument["positioning_stacks"][0]["positioners"] = [tmp[0], "None"]
        self.assertRaises(ValueError, read_positioning_stacks_description, instrument, positioners)
        instrument["positioning_stacks"][0]["positioners"] = ["None"]
        self.assertRaises(ValueError, read_positioning_stacks_description, instrument, positioners)

        hardware = read_fixed_hardware_description(instrument)
        instrument["fixed_hardware"][0]["name"] = instrument["fixed_hardware"][1]["name"]
        self.assertRaises(ValueError, read_fixed_hardware_description, instrument)

        self.assertEqual(len(hardware), 2)
        instrument["script_template"] = "script_template"
        with mock.patch("sscanss.core.instrument.create.open", mock.mock_open(read_data="")):
            self.assertRaises(ValueError, read_script_template, instrument)

    @mock.patch("sscanss.core.instrument.create.read_3d_model", autospec=True)
    def testInstrumentObject(self, read_model_fn):
        read_model_fn.return_value = self.mesh
        with mock.patch("sscanss.core.instrument.create.open", mock.mock_open(read_data=SAMPLE_IDF)):
            instrument = read_instrument_description_file("")

        self.assertEqual(len(instrument.positioning_stacks), 2)
        stacks = list(instrument.positioning_stacks.keys())
        instrument.loadPositioningStack(stacks[0])
        self.assertEqual(instrument.positioning_stack.name, stacks[0])
        instrument.loadPositioningStack(stacks[1])
        self.assertEqual(instrument.positioning_stack.name, stacks[1])

        self.assertEqual("incident_jaws", instrument.getPositioner("incident_jaws").name)
        self.assertEqual(instrument.positioning_stack.name, instrument.getPositioner(stacks[1]).name)
        self.assertRaises(ValueError, instrument.getPositioner, stacks[0])
        self.assertRaises(ValueError, instrument.getPositioner, "None")

        instrument.jaws.positioner.fkine([100])
        self.assertTrue(instrument.beam_in_gauge_volume)
        instrument.gauge_volume[2] = 1.0
        self.assertFalse(instrument.beam_in_gauge_volume)
        np.testing.assert_array_almost_equal(*instrument.q_vectors, [-0.70710678, 0.70710678, 0.0], decimal=5)
        instrument.getPositioner("diffracted_jaws").fkine([np.pi / 2, 20.0])
        np.testing.assert_array_almost_equal(*instrument.q_vectors, [0, 0, 0], decimal=5)

        detector = list(instrument.detectors.values())[0]
        self.assertEqual(len(detector.model().meshes), 2)
        self.assertEqual(len(detector.model().transforms), 2)
        detector.positioner = None
        self.assertEqual(len(detector.model().meshes), 1)
        self.assertEqual(len(detector.model().transforms), 1)

        self.assertEqual(len(instrument.jaws.model().meshes), 2)
        self.assertEqual(len(instrument.jaws.model().transforms), 2)
        instrument.jaws.positioner = None
        self.assertEqual(len(instrument.jaws.model().meshes), 1)
        self.assertEqual(len(instrument.jaws.model().transforms), 1)

    def testSerialLink(self):
        with self.assertRaises(ValueError):
            # zero vector as Axis
            Link("", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 0, 0)

        link_1 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0.0, 600.0, 0.0)
        np.testing.assert_array_almost_equal(np.identity(4), link_1.transformation_matrix, decimal=5)
        link_1.move(200)
        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 200], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, link_1.transformation_matrix, decimal=5)

        link_2 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi, np.pi, np.pi / 2)
        expected_result = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, link_2.transformation_matrix, decimal=5)
        link_2.move(0)
        np.testing.assert_array_almost_equal(np.identity(4), link_2.transformation_matrix, decimal=5)
        link_2.reset()
        np.testing.assert_array_almost_equal(expected_result, link_2.transformation_matrix, decimal=5)
        qv = link_2.quaternion_vector_pair
        np.testing.assert_array_almost_equal(qv.quaternion, [0.0, 0.0, 0.70711, 0.70711], decimal=5)
        np.testing.assert_array_almost_equal(qv.vector, [0.0, 0.0, 0.0], decimal=5)

        q1 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 600, 0)
        q2 = Link("", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        q3 = Link("", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -250, 250, 0)
        q4 = Link("", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -250, 250, 0)
        s = SerialManipulator("", [q1, q2, q3, q4])

        pose_0 = s.fkine([250, 1.57, 20, 30])
        np.testing.assert_array_almost_equal(s.configuration, [250, 1.57, 20, 30], decimal=5)
        s.reset()
        np.testing.assert_array_almost_equal(s.configuration, [0, 0, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(np.identity(4), s.pose, decimal=5)
        self.assertEqual(s.link_count, 4)
        self.assertEqual(len(s.links), 4)

        model = s.model()  # should be empty since no mesh is provided
        self.assertEqual(model.meshes, [])
        self.assertEqual(model.transforms, [])

        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 250], [0, 0, 0, 1]]
        pose = s.fkine([250, 1.57, 20, 30], end_index=1)
        np.testing.assert_array_almost_equal(expected_result, pose, decimal=5)

        expected_result = [[1, 0, 0, 30], [0, 1, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1]]
        pose = s.fkine([250, 1.57, 20, 30], start_index=2)
        np.testing.assert_array_almost_equal(expected_result, pose, decimal=5)
        base = Matrix44([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -500], [0, 0, 0, 1]])
        s.base = base
        pose = s.fkine([250, 1.57, 20, 30], start_index=2)
        np.testing.assert_array_almost_equal(expected_result, pose, decimal=5)

        pose = s.fkine([250, 1.57, 20, 30], end_index=1)
        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -250], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, pose, decimal=5)
        pose = s.fkine([250, 1.57, 20, 30], end_index=1, include_base=False)
        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 250], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, pose, decimal=5)

        s.tool = base
        pose = s.fkine([250, 1.57, 20, 30])
        np.testing.assert_array_almost_equal(pose, base @ pose_0 @ base, decimal=5)

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        mesh = Mesh(vertices, indices, normals)
        q1 = Link("", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0, mesh=mesh)
        q2 = Link("", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0, mesh=mesh)
        s = SerialManipulator("", [q1, q2], base_mesh=mesh)
        self.assertEqual(len(s.model(base).meshes), 3)
        self.assertEqual(len(s.model(base).transforms), 3)
        pose = s.fkine([np.pi / 2, -np.pi / 2])
        expected_result = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, pose, decimal=5)

        s.set_points = [-np.pi / 2, np.pi / 4]
        self.assertAlmostEqual(s.links[0].set_point, -np.pi / 2, 5)
        self.assertAlmostEqual(s.links[1].set_point, np.pi / 4, 5)

        s.links[0].locked = True
        s.links[1].locked = True
        s.fkine([-np.pi / 2, np.pi / 2])
        expected_result = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, s.pose, decimal=5)
        pose = s.fkine([-np.pi / 2, np.pi / 2], ignore_locks=True)
        expected_result = [[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, pose, decimal=5)

    def testTrajectoryGeneration(self):
        poses = joint_space_trajectory([0], [1], 10)
        self.assertEqual(poses.shape, (10, 1))
        self.assertAlmostEqual(poses.max(), 1, 5)
        self.assertAlmostEqual(poses.min(), 0, 5)

        poses = joint_space_trajectory([0, 1, -1], [1, 0, 1], 100)
        self.assertEqual(poses.shape, (100, 3))
        np.testing.assert_array_almost_equal(poses[0], [0, 1, -1], decimal=5)
        np.testing.assert_array_almost_equal(poses[-1], [1, 0, 1], decimal=5)

    def testSequence(self):
        frames = mock.Mock()
        sequence = Sequence(frames, [0], [1], 400, 10)
        frames.assert_not_called()
        self.assertFalse(sequence.isRunning())
        sequence.setFrame(2)
        self.assertEqual(sequence.current_frame, 2)
        sequence.setFrame(-1)
        self.assertEqual(sequence.current_frame, 9)
        sequence.start()
        self.assertTrue(sequence.isRunning())
        sequence.stop()
        self.assertEqual(sequence.current_frame, 9)
        frames.assert_called()
        frames.reset_mock()
        sequence.start()
        self.assertTrue(sequence.isRunning())
        QTest.qWait(1000)
        self.assertFalse(sequence.isRunning())

    def testPositioningStack(self):
        q1 = Link("", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -3.14, 3.14, 0)
        q2 = Link("", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -3.14, 3.14, 0)
        q3 = Link("", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        q4 = Link("", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)

        s1 = SerialManipulator("", [q1, q2], custom_order=[1, 0])
        s2 = SerialManipulator("", [q3, q4])

        ps = PositioningStack(s1.name, s1)
        ps.addPositioner(s2)
        self.assertListEqual(ps.order, [1, 0, 2, 3])
        np.testing.assert_array_almost_equal(ps.toUserFormat([0, 1, np.pi / 2, -np.pi / 2]), [1, 0, 90, -90], decimal=5)
        np.testing.assert_array_almost_equal(ps.fromUserFormat([1, 0, 90, -90]), [0, 1, np.pi / 2, -np.pi / 2],
                                             decimal=5)
        np.testing.assert_array_almost_equal(list(zip(*ps.bounds))[0], [-3.14] * 4, decimal=5)
        np.testing.assert_array_almost_equal(list(zip(*ps.bounds))[1], [3.14] * 4, decimal=5)
        self.assertEqual(ps.link_count, 4)
        np.testing.assert_array_almost_equal(ps.configuration, [0.0, 0.0, 0.0, 0.0], decimal=5)
        ps.fkine([100, -50, np.pi / 2, np.pi / 2])
        np.testing.assert_array_almost_equal(ps.configuration, [100, -50, np.pi / 2, np.pi / 2], decimal=5)
        expected_result = [[-1, 0, 0, -51], [0, -1, 0, 101], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(ps.pose, expected_result, decimal=5)
        self.assertEqual(ps.model().meshes, [])
        self.assertEqual(ps.model().transforms, [])

        ps = PositioningStack(s1.name, s1)
        ps.addPositioner(copy.deepcopy(s1))
        ps.addPositioner(copy.deepcopy(s1))
        self.assertEqual(ps.link_count, 6)
        ps.fkine([100, -50, 20, 30, 45, 32])
        expected_result = [[1, 0, 0, 12], [0, 1, 0, 165], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(ps.pose, expected_result, decimal=5)
        ps.changeBaseMatrix(ps.auxiliary[0], Matrix44.fromTranslation([0, 0, 5.4]))
        ps.fkine([100, -50, 20, 30, 45, 32])
        expected_result = [[1, 0, 0, 12], [0, 1, 0, 165], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(ps.pose, expected_result, decimal=5)

    def testScriptTemplate(self):
        template = "{{filename}}\nCount = {{count}}\n{{#script}}\n{{position}} {{mu_amps}}\n{{/script}}\n{{header}}"
        script = Script(template)
        self.assertEqual(script.render().strip(), "Count =")

        temp = {
            Script.Key.script.value: [{
                Script.Key.position.value: "1 2"
            }, {
                Script.Key.position.value: "3 4"
            }],
            Script.Key.header.value: "a b",
            Script.Key.filename.value: "a_filename",
            Script.Key.mu_amps.value: "20.0",
            Script.Key.count.value: 2,
        }
        expected = f"a_filename\nCount = 2\n1 2 20.0\n3 4 20.0\na b"
        script.keys = temp
        self.assertEqual(script.render().strip(), expected)

        self.assertRaises(ValueError, Script, "{{/script}}{{position}} {{#script}}")  # section tag is reversed

        # script section tag is missing
        template = "{{header}}\nCount = {{count}}\n{{header}}"
        self.assertRaises(ValueError, Script, template)

        # unknown tag
        template = "{{random}}\n{{#script}}{{position}}{{/script}}"
        self.assertRaises(ValueError, Script, template)
        template = "{{#script}}{{position}}{{random}}{{/script}}"
        self.assertRaises(ValueError, Script, template)

        # position tag is missing
        template = "{{header}}\nCount = {{count}}\n{{#script}}\n{{mu_amps}}\n{{/script}}\n{{header}}"
        self.assertRaises(ValueError, Script, template)


if __name__ == "__main__":
    unittest.main()
