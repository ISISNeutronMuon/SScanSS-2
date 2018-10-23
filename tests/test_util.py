import unittest
import numpy as np
from sscanss.core.math import Vector3, Plane, clamp
from sscanss.core.mesh import create_plane, Colour
from sscanss.core.scene import createSampleNode, Camera, Scene, Node
from sscanss.core.util import to_float, Directions


class TestUtil(unittest.TestCase):
    def testColourClass(self):
        colour = Colour.black()
        np.testing.assert_array_almost_equal(colour[:], [0.0, 0.0, 0.0, 1.0], decimal=5)

        colour = Colour.normalize(255, 0.0, 0.0)
        np.testing.assert_array_almost_equal(colour[:], [1.0, 0.0, 0.0, 1.0], decimal=5)
        np.testing.assert_array_equal(colour.rgba, [255, 0, 0, 255])

        colour = Colour.white()
        np.testing.assert_array_almost_equal(colour.rgbaf, [1.0, 1.0, 1.0, 1.0], decimal=5)

        colour = Colour(127, 0.5, 0.3, -1)
        self.assertAlmostEqual(colour.r, 1.0, 5)
        self.assertAlmostEqual(colour.g, 0.5, 5)
        self.assertAlmostEqual(colour.b, 0.3, 5)
        self.assertAlmostEqual(colour.a, 0.0, 5)
        np.testing.assert_array_almost_equal(colour.invert()[:], [0.0, 0.5, 0.7, 0.0], decimal=5)

        self.assertEqual(str(colour.white()), 'rgba(1.0, 1.0, 1.0, 1.0)')
        self.assertEqual(repr(colour.white()), 'Colour(1.0, 1.0, 1.0, 1.0)')

    def testToFloat(self):
        value, ok = to_float('21')
        self.assertAlmostEqual(value, 21, 5)
        self.assertTrue(ok)

        value, ok = to_float('2.1e3')
        self.assertAlmostEqual(value, 2100, 5)
        self.assertTrue(ok)

        value, ok = to_float('-1001.67845')
        self.assertAlmostEqual(value, -1001.67845, 5)
        self.assertTrue(ok)

        value, ok = to_float('Hello')
        self.assertEqual(value, None)
        self.assertFalse(ok)

    def testClamp(self):
        value = clamp(-1, 0, 1)
        self.assertEqual(value, 0)

        value = clamp(100, 0, 50)
        self.assertEqual(value, 50)

        value = clamp(20, 0, 50)
        self.assertEqual(value, 20)

    def testNodeCreation(self):
        mesh_1 = create_plane(Plane(np.array([1., 0., 0.]), np.array([0., 0., 0.])))
        mesh_2 = create_plane(Plane(np.array([0., 1., 0.]), np.array([0., 0., 0.])))
        mesh_3 = create_plane(Plane(np.array([0., 0., 1.]), np.array([0., 0., 0.])))
        sample = {'1': mesh_1, '2': mesh_2, '3': mesh_3}

        node = createSampleNode(sample)

        self.assertEqual(len(node.children), 3)

        box = node.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0., 0., 0.]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

        node = Node()
        self.assertTrue(node.isEmpty())
        node.translate([1., 1., 1.])
        child_node = Node(create_plane(Plane(np.array([0., 1., 0.]), np.array([0., 0., 0.]))))
        node.addChild(child_node)
        mesh = create_plane(Plane(np.array([0., 0., 1.]), np.array([0., 0., 0.])))
        node.vertices = mesh.vertices
        box = node.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0., 0., 0.]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)
        node.translate([1., 1., 1.])

        box = node.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([1.5, 1.5, 1.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([1., 1., 1.]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

        box = node.children[0].bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([1.5, 1.0, 1.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([0.5, 1.0, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([1., 1., 1.]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.707106, 5)

    def testCameraClass(self):
        # create a camera with aspect ratio of 1 and 60 deg field of view
        camera = Camera(1, 60)

        position = Vector3([0, 0, 0, ])
        target = Vector3([5.0, 0.0, 0.0])
        up = Vector3([0.0, 1.0, 0.0])

        camera.lookAt(position, target, up)
        model_view = np.array([[0, -0, 1, 0.],
                              [0, 1, 0, 0.],
                              [-1, 0, 0, 0.],
                              [0, 0, 0, 1.]])
        np.testing.assert_array_almost_equal(model_view, camera.model_view, decimal=5)
        camera.lookAt(position, target)
        np.testing.assert_array_almost_equal(model_view, camera.model_view, decimal=5)

        perspective = np.array([[1.73205081, 0,  0, 0],
                               [0, 1.73205081, 0, 0],
                               [0, 0, - 1.0002, -0.20002],
                               [0, 0, -1, 0]])
        np.testing.assert_array_almost_equal(perspective, camera.perspective, decimal=5)

        camera.reset()
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, -2], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)
        camera.lookAt(target, target)
        np.testing.assert_array_almost_equal(np.eye(4, 4), camera.model_view, decimal=5)

        camera.viewFrom(Directions.up)
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.down)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.left)
        expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.right)
        expected = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.front)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.back)
        expected = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)

        position = Vector3([0, 0, 0, ])
        target = Vector3([0.0, 5.0, 0.0])
        camera.lookAt(position, target)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

        camera.zoomToFit(target, 1.0)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 3], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

        camera.pan(-1, 1)
        expected = np.array([[1, 0, 0, 2], [0, 0, 1, 2], [0, -1, 0, 3], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

        camera.zoom(1)
        expected = np.array([[1, 0, 0, 2], [0, 0, 1, 2], [0, -1, 0, 5], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

        camera.rotate((0, 0), (0, 0))
        expected = np.array([[1, 0, 0, 2], [0, 0, 1, 2], [0, -1, 0, 5], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)
        camera.rotate((0, 0), (0.5, 0.5))
        expected = np.array([[0.8535533, -0.5, 0.1464466, 4.5],
                             [0.1464466, 0.5, 0.8535533, -0.5],
                             [-0.5, -0.707106, 0.5, 3.535533],
                             [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

        camera = Camera(0.5, 45)
        camera.zoomToFit(target, 1)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0.0691067], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

    def testScene(self):

        s = Scene()
        self.assertTrue(s.isEmpty())

        empty_node = Node()
        s.addNode('new', empty_node)
        self.assertTrue(s.isEmpty())
        self.assertRaises(KeyError, lambda: s['new'])

        mesh_1 = create_plane(Plane(np.array([0., 0., 1.]), np.array([0., 0., 0.])))
        sample_1 = {'1': mesh_1}
        node_1 = createSampleNode(sample_1)
        s.addNode('1', node_1)
        self.assertIs(node_1, s['1'])

        mesh_2 = create_plane(Plane(np.array([0., 1., 0.]), np.array([0., 0., 0.])))
        sample_2 = {'2': mesh_2}
        node_2 = createSampleNode(sample_2)
        s.addNode('2', node_2)
        self.assertIs(node_2, s['2'])
        self.assertTrue('2' in s)
        self.assertEqual(len(s.nodes), 2)

        box = s.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0., 0., 0.]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

        s.removeNode('2')
        self.assertFalse('2' in s)
        self.assertEqual(len(s.nodes), 1)
        s.removeNode('1')
        self.assertEqual(len(s.nodes), 0)

        s.addNode(Scene.sample_key, node_1)
        s.addNode('other', node_2)
        self.assertIs(s.nodes[1], node_1)  # sample node must always be last


if __name__ == '__main__':
    unittest.main()
