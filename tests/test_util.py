import unittest
import unittest.mock as mock
import numpy as np
from PyQt5.QtGui import QColor, QFont
from sscanss.__version import Version
from sscanss.core.math import Vector3, Plane, clamp, trunc, map_range, is_close
from sscanss.core.geometry import create_plane, Colour, Mesh, Volume
from sscanss.core.scene import (SampleEntity, PlaneEntity, MeasurementPointEntity, MeasurementVectorEntity, Camera,
                                Scene, Node, validate_instrument_scene_size, TextRenderNode)
from sscanss.core.util import to_float, Directions, Attributes, compact_path, find_duplicates
from tests.helpers import APP


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node_mock = self.createMock("sscanss.core.scene.entity.Node.buildVertexBuffer")

    def createMock(self, module):
        patcher = mock.patch(module, autospec=True)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def testNodeCreation(self):
        node = Node()
        self.assertEqual(node.vertices.size, 0)
        self.assertEqual(node.indices.size, 0)
        self.assertEqual(node.normals.size, 0)
        self.assertTrue(node.isEmpty())

        mesh = create_plane(Plane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))
        node = Node(mesh)
        np.testing.assert_array_almost_equal(node.vertices, mesh.vertices)
        np.testing.assert_array_equal(node.indices, mesh.indices)
        np.testing.assert_array_almost_equal(node.normals, mesh.normals)

        node = SampleEntity({}).node()
        self.assertTrue(node.isEmpty())

        sample_mesh = Mesh(
            np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]),
            np.array([0, 1, 2]),
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        node = SampleEntity(sample_mesh).node()
        np.testing.assert_array_almost_equal(node.vertices, sample_mesh.vertices)
        np.testing.assert_array_equal(node.indices, sample_mesh.indices)
        np.testing.assert_array_almost_equal(node.normals, sample_mesh.normals)
        self.assertEqual(node.render_primitive, Node.RenderPrimitive.Triangles)

        points = np.rec.array(
            [([11.0, 12.0, 13.0], True), ([14.0, 15.0, 16.0], False), ([17.0, 18.0, 19.0], True)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )

        volume = Volume(np.zeros([3, 4, 5], np.float32), np.ones(3), np.array([1., 1.5, 2.]))
        with mock.patch('sscanss.core.scene.node.Texture3D'), mock.patch('sscanss.core.scene.node.Texture1D'):
            node = SampleEntity(volume).node()
            np.testing.assert_array_almost_equal(node.top, [1.5, 2., 2.5])
            np.testing.assert_array_almost_equal(node.bottom, [-1.5, -2., -2.5])
            box = node.bounding_box
            np.testing.assert_array_almost_equal(box.max, [2.5, 3.5, 4.5], decimal=5)
            np.testing.assert_array_almost_equal(box.min, [-0.5, -0.5, -0.5], decimal=5)
            np.testing.assert_array_almost_equal(box.center, [1., 1.5, 2.], decimal=5)
            self.assertAlmostEqual(box.radius, 3.535533, 5)

        node = MeasurementPointEntity(np.array([])).node()
        self.assertTrue(node.isEmpty())
        node = MeasurementPointEntity(points).node()
        self.assertEqual(len(node.per_object_transform), points.size)

        node = MeasurementVectorEntity(np.array([]), np.array([]), 0).node()
        self.assertTrue(node.isEmpty())

        vectors = np.ones((3, 3, 2))
        node = MeasurementVectorEntity(points, vectors, 0).node()
        self.assertTrue(node.children[0].visible)
        self.assertFalse(node.children[1].visible)
        self.assertEqual(len(node.children), vectors.shape[2])

        node = PlaneEntity(Plane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])), 1.0, 1.0).node()
        np.testing.assert_array_almost_equal(node.vertices, mesh.vertices)
        np.testing.assert_array_equal(node.indices, mesh.indices)
        np.testing.assert_array_almost_equal(node.normals, mesh.normals)

        text_node = TextRenderNode('', QColor.fromRgbF(1, 1, 0), QFont())
        self.assertTrue(text_node.isEmpty())
        self.assertEqual(text_node.size, (0, 0))
        self.assertIsNone(text_node.buffer)
        text_node = TextRenderNode('Test', QColor.fromRgbF(1, 1, 0), QFont())
        self.assertFalse(text_node.isEmpty())
        text_node.buildVertexBuffer()
        self.assertEqual(text_node.buffer.count, 6)  # buffer with 6 vertices

    def testNodeChildren(self):
        node = Node()
        self.assertTrue(node.isEmpty())
        node.addChild(Node())
        self.assertTrue(node.isEmpty())
        self.assertEqual(len(node.children), 0)

        mesh = create_plane(Plane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))
        mesh_1 = create_plane(Plane(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])))
        mesh_2 = create_plane(Plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])))
        node = Node()
        self.assertTrue(node.isEmpty())
        node.addChild(Node(mesh))
        node.addChild(Node(mesh_1))
        node.addChild(Node(mesh_2))
        self.assertFalse(node.isEmpty())
        self.assertEqual(len(node.children), 3)

        box = node.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0.0, 0.0, 0.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

        # Nested Nodes
        node = Node()
        self.assertTrue(node.isEmpty())
        child_node = Node(mesh)
        child_node.addChild(Node(mesh_1))
        child_node.addChild(Node(mesh_2))
        node.addChild(child_node)

        # Nested Copy
        copied_node = node.copy(transform=np.identity(4) * 2)
        self.assertIsNot(node, copied_node)
        self.assertIsNot(node.transform, copied_node.transform)
        self.assertIs(node.vertices, copied_node.vertices)
        self.assertEqual(len(copied_node.children), 1)
        self.assertIs(child_node.vertices, copied_node.children[0].vertices)
        self.assertIsNot(child_node, copied_node.children[0])
        self.assertEqual(len(copied_node.children[0].children), 2)
        self.assertIsNot(child_node.children[0], copied_node.children[0].children[0])
        self.assertIs(copied_node.children[0].children[0].parent, copied_node.children[0])
        self.assertIs(child_node.children[0].parent, child_node)

        self.assertFalse(node.isEmpty())
        self.assertEqual(len(node.children), 1)
        self.assertEqual(len(node.children[0].children), 2)
        box = node.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0.0, 0.0, 0.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

        # Flatten Node
        node = node.flatten()
        self.assertFalse(node.isEmpty())
        self.assertEqual(len(node.children), 3)
        box = node.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0.0, 0.0, 0.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

    def testNodeBounds(self):
        mesh = create_plane(Plane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))
        mesh_1 = create_plane(Plane(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])))
        mesh_2 = create_plane(Plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])))

        node = Node()
        self.assertTrue(node.isEmpty())
        node.addChild(Node(mesh))
        node.addChild(Node(mesh_1))
        node.addChild(Node(mesh_2))

        box = node.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0.0, 0.0, 0.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

        box = node.children[0].bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.0, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([0.0, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0.0, 0.0, 0.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.707106, 5)

        box = node.children[1].bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.0, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, 0.0, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0.0, 0.0, 0.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.707106, 5)

    def testNodeProperties(self):
        mesh = create_plane(Plane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))

        node = Node()
        node.render_mode = Node.RenderMode.Solid
        node.selected = False
        node.visible = True
        node.colour = Colour.black()

        child_node = Node(mesh)
        child_node.render_mode = Node.RenderMode.Transparent
        child_node.selected = True
        child_node.visible = False
        child_node.colour = Colour.white()
        node.addChild(child_node)

        self.assertIs(node, child_node.parent)
        self.assertEqual(child_node.render_mode, Node.RenderMode.Transparent)
        child_node.render_mode = None
        self.assertEqual(child_node.render_mode, Node.RenderMode.Solid)

        self.assertTrue(child_node.selected)
        child_node.selected = None
        self.assertFalse(child_node.selected)

        self.assertFalse(child_node.visible)
        child_node.visible = None
        self.assertTrue(child_node.visible)

        np.testing.assert_array_almost_equal(child_node.colour.rgbaf, [1.0, 1.0, 1.0, 1.0], decimal=5)
        child_node.colour = None
        np.testing.assert_array_almost_equal(child_node.colour.rgbaf, [0.0, 0.0, 0.0, 1.0], decimal=5)


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

        self.assertEqual(str(colour.white()), "rgba(1.0, 1.0, 1.0, 1.0)")
        self.assertEqual(repr(colour.white()), "Colour(1.0, 1.0, 1.0, 1.0)")

    def testToFloat(self):
        value = to_float("21")
        self.assertAlmostEqual(value, 21, 5)

        value = to_float("2.1e3")
        self.assertAlmostEqual(value, 2100, 5)

        value = to_float("-1001.67845")
        self.assertAlmostEqual(value, -1001.67845, 5)

        value = to_float("Hello")
        self.assertIsNone(value, None)

    def testIsClose(self):
        self.assertTrue(is_close(2.5, 2.5))
        self.assertFalse(is_close(0.998, 0.999))
        self.assertTrue(is_close(0.998, 0.999, 1e-2))

    def testTrunc(self):
        init = 2.59
        value = trunc(init, decimals=1)
        self.assertAlmostEqual(value, 2.5, 2)

        init = -1.9999
        value = trunc(init, decimals=3)
        self.assertAlmostEqual(value, -1.999, 4)

    def testMap(self):
        value = map_range(-1, 0, 0, 100, -0.5)
        self.assertAlmostEqual(value, 50, 5)

        value = map_range(-1, 0, 0, 100, -0.8)
        self.assertAlmostEqual(value, 20, 5)

    def testCompactPath(self):
        self.assertEqual(compact_path("", 10), "")
        self.assertEqual(compact_path("abcdef", 5), "a...f")
        self.assertEqual(compact_path("C:/test/file.py", 20), "C:/test/file.py")
        self.assertEqual(compact_path("C:/test/some/file.py", 15), "C:/tes...ile.py")
        self.assertRaises(ValueError, compact_path, "C:/test/some/file.py", 0)

    def testFindDuplicate(self):
        self.assertListEqual(find_duplicates((2, 3, 4, 4, 6)), [4])
        self.assertListEqual(find_duplicates(("2", "2", "3", "4", "4", "6")), ["2", "4"])
        self.assertListEqual(find_duplicates((1, 2, 3, 4, 5, 6)), [])

    def testClamp(self):
        value = clamp(-1, 0, 1)
        self.assertEqual(value, 0)

        value = clamp(100, 0, 50)
        self.assertEqual(value, 50)

        value = clamp(20, 0, 50)
        self.assertEqual(value, 20)

    def testCameraClass(self):
        # create a camera with aspect ratio of 1 and 60 deg field of view
        camera = Camera(1, 60)

        position = Vector3([0, 0, 0])
        target = Vector3([5.0, 0.0, 0.0])
        up = Vector3([0.0, 1.0, 0.0])

        camera.lookAt(position, target, up)
        model_view = np.array([[0, -0, 1, 0.0], [0, 1, 0, 0.0], [-1, 0, 0, 0.0], [0, 0, 0, 1.0]])
        np.testing.assert_array_almost_equal(model_view, camera.model_view, decimal=5)
        camera.lookAt(position, target)
        np.testing.assert_array_almost_equal(model_view, camera.model_view, decimal=5)

        perspective = np.array([[1.73205081, 0, 0, 0], [0, 1.73205081, 0, 0], [0, 0, -1.0002, -0.20002], [0, 0, -1, 0]])
        np.testing.assert_array_almost_equal(perspective, camera.perspective, decimal=5)
        np.testing.assert_array_almost_equal(perspective, camera.projection, decimal=5)

        ortho = np.array([[17.3205081, 0, 0, 0], [0, 17.3205081, 0, 0], [0, 0, -0.0020002, -1.0002], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(ortho, camera.orthographic, decimal=5)
        camera.mode = Camera.Projection.Orthographic
        np.testing.assert_array_almost_equal(ortho, camera.projection, decimal=5)
        camera.mode = Camera.Projection.Perspective

        camera.reset()
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, -2], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)
        camera.lookAt(target, target)
        expected = np.array([[1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

        camera.viewFrom(Directions.Up)
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.Down)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.Left)
        expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.Right)
        expected = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.Front)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)
        camera.viewFrom(Directions.Back)
        expected = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_almost_equal(expected, camera.rot_matrix, decimal=5)

        position = Vector3([0, 0, 0])
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
        expected = np.array([
            [0.8535533, -0.5, 0.1464466, 4.5],
            [0.1464466, 0.5, 0.8535533, -0.5],
            [-0.5, -0.707106, 0.5, 3.535533],
            [0, 0, 0, 1],
        ])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

        camera = Camera(0.5, 45)
        camera.zoomToFit(target, 1)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0.0691067], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(expected, camera.model_view, decimal=5)

    @mock.patch("sscanss.core.scene.entity.Node.buildVertexBuffer", autospec=True)
    @mock.patch("sscanss.core.scene.scene.InstrumentEntity", autospec=True)
    def testScene(self, mock_instrument_entity, _):
        s = Scene()
        self.assertTrue(s.isEmpty())

        empty_node = Node()
        self.assertTrue(s.isEmpty())
        s.addNode("new", empty_node)
        # empty key will not be added
        self.assertNotIn("new", s)
        # bad key should return empty node
        self.assertTrue(s["random"].isEmpty())

        mesh_1 = create_plane(Plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])))
        sample_1 = mesh_1
        node_1 = SampleEntity(sample_1).node()
        s.addNode("1", node_1)
        self.assertIs(node_1, s["1"])

        mesh_2 = create_plane(Plane(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])))
        sample_2 = mesh_2
        node_2 = SampleEntity(sample_2).node()
        s.addNode("2", node_2)
        self.assertIs(node_2, s["2"])
        self.assertTrue("2" in s)
        self.assertEqual(len(s.nodes), 2)

        box = s.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([0.5, 0.5, 0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([-0.5, -0.5, -0.5]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([0.0, 0.0, 0.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 0.8660254, 5)

        s.removeNode("2")
        self.assertFalse("2" in s)
        self.assertEqual(len(s.nodes), 1)
        s.removeNode("1")
        self.assertEqual(len(s.nodes), 0)

        s.addNode(Attributes.Sample, node_1)
        s.addNode("other", node_2)
        Scene.sample_render_mode = Node.RenderMode.Solid
        self.assertIs(s.nodes[0], node_1)  # sample node is first if render mode is not transparent
        node_1.render_mode = Node.RenderMode.Wireframe
        self.assertIs(s.nodes[0], node_1)
        node_1.render_mode = Node.RenderMode.Transparent
        self.assertIs(s.nodes[-1], node_1)

        max_extent = Scene.max_extent
        mock_instrument_entity.return_value.vertices = np.array([[1, 1, 0], [-1, 0, 0]])
        Scene.max_extent = 2.0
        self.assertTrue(validate_instrument_scene_size(None))
        Scene.max_extent = 0.5
        self.assertFalse(validate_instrument_scene_size(None))
        Scene.max_extent = max_extent

    def testVersion(self):
        version = Version(1, 2, 3, 'beta', '1045')
        self.assertEqual(str(version), '1.2.3-beta+1045')
        new_version = Version.parse(str(version))
        self.assertEqual(new_version.major, 1)
        self.assertEqual(new_version.minor, 2)
        self.assertEqual(new_version.patch, 3)
        self.assertEqual(new_version.pre_release, 'beta')
        self.assertEqual(new_version.build, '1045')
        self.assertEqual(version, new_version)
        new_version.pre_release = 'gamma'
        self.assertNotEqual(version, new_version)
        self.assertEqual(Version.parse(' 1. 2.3 -beta +1045'), version)
        self.assertEqual(str(Version.parse('2.2.1')), '2.2.1')
        new_version = Version.parse('2.3.1+1102')
        self.assertEqual(new_version.major, 2)
        self.assertEqual(new_version.minor, 3)
        self.assertEqual(new_version.patch, 1)
        self.assertIsNone(new_version.pre_release)
        self.assertEqual(new_version.build, '1102')
        self.assertRaises(ValueError, Version.parse, '1.1')
        self.assertRaises(ValueError, Version.parse, 'a.a.a')


if __name__ == "__main__":
    unittest.main()
