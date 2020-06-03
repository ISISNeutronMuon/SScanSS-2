import os
import copy
import unittest
import shutil
import tempfile
from fastjsonschema.exceptions import JsonSchemaException
import numpy as np
from sscanss.core.math import Matrix44
from sscanss.core.geometry import Mesh
from sscanss.core.instrument.instrument import PositioningStack, Script
from sscanss.core.instrument.robotics import joint_space_trajectory, Link, SerialManipulator
from sscanss.core.instrument import read_instrument_description_file
from sscanss.ui.window.model import MainWindowModel


class TestInstrument(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testReadIDF(self):
        instruments = MainWindowModel().instruments
        for name, idf in instruments.items():
            instrument = read_instrument_description_file(idf.path)
            self.assertEqual(name.lower(), instrument.name.lower())

        data = '{"instrument":{"name": "FAKE"}}'
        path = self.writeTestFile('test.json', data)

        self.assertRaises(JsonSchemaException, lambda: read_instrument_description_file(path))

    def writeTestFile(self, filename, text):
        full_path = os.path.join(self.test_dir, filename)
        with open(full_path, 'w') as text_file:
            text_file.write(text)
        return full_path

    def testSerialLink(self):
        with self.assertRaises(ValueError):
            # zero vector as Axis
            Link('', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 0, 0)

        link_1 = Link('', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0.0, 600.0, 0.0)
        np.testing.assert_array_almost_equal(np.identity(4), link_1.transformationMatrix, decimal=5)
        link_1.move(200)
        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 200], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, link_1.transformationMatrix, decimal=5)

        link_2 = Link('', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -np.pi, np.pi, np.pi/2)
        expected_result = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, link_2.transformationMatrix, decimal=5)
        link_2.move(0)
        np.testing.assert_array_almost_equal(np.identity(4), link_2.transformationMatrix, decimal=5)
        link_2.reset()
        np.testing.assert_array_almost_equal(expected_result, link_2.transformationMatrix, decimal=5)
        qv = link_2.quaterionVectorPair
        np.testing.assert_array_almost_equal(qv.quaternion, [0., 0., 0.70711, 0.70711], decimal=5)
        np.testing.assert_array_almost_equal(qv.vector, [0., 0., 0.], decimal=5)

        q1 = Link('', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, 0, 600, 0)
        q2 = Link('', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        q3 = Link('', [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -250, 250, 0)
        q4 = Link('', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -250, 250, 0)
        s = SerialManipulator('', [q1, q2, q3, q4])

        T_0 = s.fkine([250, 1.57, 20, 30])
        np.testing.assert_array_almost_equal(s.configuration, [250, 1.57, 20, 30], decimal=5)
        s.reset()
        np.testing.assert_array_almost_equal(s.configuration, [0, 0, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(np.identity(4), s.pose, decimal=5)
        self.assertEqual(s.numberOfLinks, 4)
        self.assertEqual(len(s.links), 4)

        node = s.model()  # should be empty since no mesh is provided
        self.assertTrue(node.isEmpty())

        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 250], [0, 0, 0, 1]]
        T = s.fkine([250, 1.57, 20, 30], end_index=1)
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)

        expected_result = [[1, 0, 0, 30], [0, 1, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1]]
        T = s.fkine([250, 1.57, 20, 30], start_index=2)
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)
        base = Matrix44([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -500], [0, 0, 0, 1]])
        s.base = base
        T = s.fkine([250, 1.57, 20, 30], start_index=2)
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)

        T = s.fkine([250, 1.57, 20, 30], end_index=1)
        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -250], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)
        T = s.fkine([250, 1.57, 20, 30], end_index=1, include_base=False)
        expected_result = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 250], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)

        s.tool = base
        T = s.fkine([250, 1.57, 20, 30])
        np.testing.assert_array_almost_equal(T, base @ T_0 @ base, decimal=5)

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([0, 1, 2])
        mesh = Mesh(vertices, indices, normals)
        q1 = Link('', [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0, mesh=mesh)
        q2 = Link('', [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0, mesh=mesh)
        s = SerialManipulator('', [q1, q2], base_mesh=mesh)
        self.assertFalse(s.model(base).isEmpty())
        T = s.fkine([np.pi/2, -np.pi/2])
        expected_result = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)

        s.set_points = [-np.pi/2, np.pi/4]
        self.assertAlmostEqual(s.links[0].set_point, -np.pi/2, 5)
        self.assertAlmostEqual(s.links[1].set_point, np.pi/4, 5)

        s.links[0].locked = True
        s.links[1].locked = True
        T = s.fkine([-np.pi/2, np.pi/2])
        expected_result = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)
        T = s.fkine([-np.pi/2, np.pi/2], ignore_locks=True)
        expected_result = [[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(expected_result, T, decimal=5)

    def testTrajectoryGeneration(self):
        poses = joint_space_trajectory([0], [1], 10)
        self.assertEqual(poses.shape, (10, 1))
        self.assertAlmostEqual(poses.max(), 1, 5)
        self.assertAlmostEqual(poses.min(), 0, 5)

        poses = joint_space_trajectory([0, 1, -1], [1, 0, 1], 100)
        self.assertEqual(poses.shape, (100, 3))
        np.testing.assert_array_almost_equal(poses[0], [0, 1, -1], decimal=5)
        np.testing.assert_array_almost_equal(poses[-1], [1, 0, 1], decimal=5)

    def testPositioningStack(self):
        q1 = Link('', [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -3.14, 3.14, 0)
        q2 = Link('', [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], Link.Type.Prismatic, -3.14, 3.14, 0)
        q3 = Link('', [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)
        q4 = Link('', [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], Link.Type.Revolute, -3.14, 3.14, 0)

        s1 = SerialManipulator('', [q1, q2])
        s2 = SerialManipulator('', [q3, q4])

        ps = PositioningStack(s1.name, s1)
        ps.addPositioner(s2)
        self.assertEqual(ps.numberOfLinks, 4)
        np.testing.assert_array_almost_equal(ps.configuration, [0., 0., 0., 0.], decimal=5)
        T = ps.fkine([100, -50, np.pi/2, np.pi/2])
        np.testing.assert_array_almost_equal(ps.configuration, [100, -50, np.pi/2, np.pi/2], decimal=5)
        expected_result = [[-1, 0, 0, -51], [0, -1, 0, 101], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(T, expected_result, decimal=5)
        self.assertTrue(ps.model().isEmpty())

        ps = PositioningStack(s1.name, s1)
        ps.addPositioner(copy.deepcopy(s1))
        ps.addPositioner(copy.deepcopy(s1))
        self.assertEqual(ps.numberOfLinks, 6)
        T = ps.fkine([100, -50, 20, 30, 45, 32])
        expected_result = [[1, 0, 0, 12], [0, 1, 0, 165], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_array_almost_equal(T, expected_result, decimal=5)

    def testScriptTemplate(self):
        template = '{{filename}}\nCount = {{count}}\n{{#script}}\n{{position}} {{mu_amps}}\n{{/script}}\n{{header}}'
        script = Script(template)
        self.assertEqual(script.render().strip(), 'Count =')

        temp = {Script.Key.script.value: [{Script.Key.position.value: '1 2'}, {Script.Key.position.value: '3 4'}],
                Script.Key.header.value: 'a b',
                Script.Key.filename.value: 'a_filename',
                Script.Key.mu_amps.value: '20.0',
                Script.Key.count.value: 2}
        expected = f'a_filename\nCount = 2\n1 2 20.0\n3 4 20.0\na b'
        script.keys = temp
        self.assertEqual(script.render().strip(), expected)

        self.assertRaises(ValueError, Script, '{{/script}}{{position}} {{#script}}')  # section tag is reversed

        # script section tag is missing
        template = '{{header}}\nCount = {{count}}\n{{header}}'
        self.assertRaises(ValueError,  Script, template)

        # position tag is missing
        template = '{{header}}\nCount = {{count}}\n{{#script}}\n{{mu_amps}}\n{{/script}}\n{{header}}'
        self.assertRaises(ValueError, Script, template)


if __name__ == '__main__':
    unittest.main()
