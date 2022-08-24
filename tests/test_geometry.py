import unittest
import numpy as np
from sscanss.core.math import Vector3, matrix_from_xyz_eulers, Plane
from sscanss.core.geometry import (Mesh, MeshGroup, closest_triangle_to_point, mesh_plane_intersection, create_tube,
                                   segment_plane_intersection, BoundingBox, create_cuboid, path_length_calculation,
                                   compute_face_normals, segment_triangle_intersection, point_selection, Volume, Curve,
                                   volume_plane_intersection, volume_ray_intersection)


class TestMeshClass(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        indices = np.array([2, 1, 0])
        self.mesh_1 = Mesh(vertices, indices, normals)

        vertices = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([1, 0, 2, 0, 1, 2])
        self.mesh_2 = Mesh(vertices, indices, normals)

    def testCreation(self):
        vertices = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]])
        indices = np.array([1, 0, 2])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        mesh = Mesh(vertices, indices, normals)

        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, [1, 0, 2])

        mesh = Mesh(vertices, indices, normals, clean=True)
        expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

        np.testing.assert_array_almost_equal(mesh.vertices, vertices[[2, 1, 0]], decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, expected, decimal=5)
        np.testing.assert_array_equal(mesh.indices, [1, 2, 0])

        v = np.array([[np.nan, 1, 0], [1, 0, 0], [0, 1, 0]])
        self.assertRaises(ValueError, Mesh, v, indices, normals, clean=True)

        n = np.array([[0, 1, 0], [1, -np.inf, 0], [0, 1, 0]])
        self.assertRaises(ValueError, Mesh, vertices, indices, n)

    def testComputeNormals(self):
        vertices = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]])
        indices = np.array([1, 0, 2])
        mesh = Mesh(vertices, indices)

        expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

        # Check that correct normals are generated also vertices and indices are unchanged
        np.testing.assert_array_almost_equal(mesh.vertices, vertices[[2, 1, 0]], decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, expected, decimal=5)
        np.testing.assert_array_equal(mesh.indices, [1, 2, 0])

    def testComputeBoundingBox(self):
        box = self.mesh_1.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([7, 8, 9]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([1, 2, 3]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([4.0, 5.0, 6.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 5.1961524, 5)

        box = self.mesh_2.bounding_box
        np.testing.assert_array_almost_equal(box.max, np.array([7, 8, 9]), decimal=5)
        np.testing.assert_array_almost_equal(box.min, np.array([1, 2, 3]), decimal=5)
        np.testing.assert_array_almost_equal(box.center, np.array([4.0, 5.0, 6.0]), decimal=5)
        self.assertAlmostEqual(box.radius, 5.1961524, 5)

    def testAppendAndSplit(self):
        self.mesh_1.append(self.mesh_2)

        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9], [4, 5, 6], [1, 2, 3]])
        normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        indices = np.array([2, 1, 0, 4, 3, 5, 3, 4, 5])

        np.testing.assert_array_almost_equal(self.mesh_1.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(self.mesh_1.normals, normals, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, indices)

        split_mesh = self.mesh_1.remove(3)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([2, 1, 0]))
        np.testing.assert_array_equal(split_mesh.indices, np.array([1, 0, 2, 0, 1, 2]))
        np.testing.assert_array_almost_equal(self.mesh_1.vertices, vertices[0:3, :], decimal=5)
        np.testing.assert_array_almost_equal(split_mesh.vertices, vertices[3:, :], decimal=5)
        np.testing.assert_array_almost_equal(self.mesh_1.normals, normals[0:3, :], decimal=5)
        np.testing.assert_array_almost_equal(split_mesh.normals, normals[3:, :], decimal=5)

    def testTransform(self):
        angles = np.radians([30, 60, 90])
        matrix = matrix_from_xyz_eulers(Vector3(angles))
        self.mesh_1.rotate(matrix)

        expected_vertices = np.array([
            [1.59807621, -0.75, 3.29903811],
            [2.69615242, -0.20096189, 8.34807621],
            [3.79422863, 0.34807621, 13.39711432],
        ])
        expected_normals = np.array([[0.866025, -0.25, 0.433013], [-0.5, -0.433013, 0.75], [0, 0.866025, 0.5]])

        np.testing.assert_array_almost_equal(self.mesh_1.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(self.mesh_1.normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([2, 1, 0]))

        offset = Vector3([10, -11, 12])
        self.mesh_1.translate(offset)
        expected_vertices = np.array([
            [11.59807621, -11.75, 15.29903811],
            [12.69615242, -11.20096189, 20.34807621],
            [13.79422863, -10.6519237, 25.39711432],
        ])

        np.testing.assert_array_almost_equal(self.mesh_1.vertices, expected_vertices, decimal=5)
        np.testing.assert_array_almost_equal(self.mesh_1.normals, expected_normals, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([2, 1, 0]))

        transform_matrix = np.eye(4, 4)
        transform_matrix[0:3, 0:3] = matrix.transpose()
        transform_matrix[0:3, 3] = -offset.dot(matrix)
        self.mesh_1.transform(transform_matrix)
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_almost_equal(self.mesh_1.vertices, expected, decimal=5)
        expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_almost_equal(self.mesh_1.normals, expected, decimal=5)
        np.testing.assert_array_equal(self.mesh_1.indices, np.array([2, 1, 0]))

    def testCopy(self):
        mesh = self.mesh_1.copy()
        np.testing.assert_array_almost_equal(mesh.vertices, self.mesh_1.vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, self.mesh_1.normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, self.mesh_1.indices)
        self.assertIsNot(mesh.vertices, self.mesh_1.vertices)
        self.assertIsNot(mesh.normals, self.mesh_1.normals)
        self.assertIsNot(mesh.indices, self.mesh_1.indices)

    def testMeshGroup(self):
        group = MeshGroup()
        self.assertEqual(group.meshes, [])
        self.assertEqual(group.transforms, [])

        group.addMesh(self.mesh_1)
        self.assertEqual(group.meshes, [self.mesh_1])
        self.assertEqual(len(group.transforms), 1)
        np.testing.assert_array_almost_equal(group.transforms[0], np.identity(4))

        matrix = np.ones((4, 4))
        group.addMesh(self.mesh_2, matrix)
        self.assertEqual(group.meshes, [self.mesh_1, self.mesh_2])
        self.assertEqual(len(group.transforms), 2)
        np.testing.assert_array_almost_equal(group.transforms[0], np.identity(4))
        np.testing.assert_array_almost_equal(group.transforms[1], matrix)

        group.merge(group)
        self.assertEqual(group.meshes, [self.mesh_1, self.mesh_2, self.mesh_1, self.mesh_2])
        self.assertEqual(len(group.transforms), 4)
        np.testing.assert_array_almost_equal(group.transforms[0], np.identity(4))
        np.testing.assert_array_almost_equal(group.transforms[1], matrix)
        np.testing.assert_array_almost_equal(group.transforms[2], np.identity(4))
        np.testing.assert_array_almost_equal(group.transforms[3], matrix)

        self.assertEqual(group[3][0], self.mesh_2)
        np.testing.assert_array_almost_equal(group[3][1], matrix)


class TestBoundingBoxClass(unittest.TestCase):
    def testConstruction(self):
        max_position = np.array([1.0, 1.0, 1.0])
        min_position = np.array([-1.0, -1.0, -1.0])

        box = BoundingBox(max_position, min_position)
        max_pos, min_pos = box.bounds
        np.testing.assert_array_almost_equal(max_pos, max_position, decimal=5)
        np.testing.assert_array_almost_equal(min_pos, min_position, decimal=5)
        np.testing.assert_array_almost_equal(box.center, [0.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(box.radius, 1.73205, decimal=5)

        max_position = Vector3([1.0, 2.0, 3.0])
        min_position = Vector3([-1.0, -2.0, -3.0])
        box = BoundingBox(max_position, min_position)
        np.testing.assert_array_almost_equal(box.max, max_position, decimal=5)
        self.assertIsNot(max_position, box.max)  # make sure this are not the same object
        np.testing.assert_array_almost_equal(box.min, min_position, decimal=5)
        self.assertIsNot(min_position, box.min)  # make sure this are not the same object
        np.testing.assert_array_almost_equal(box.center, [0.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(box.radius, 3.74166, decimal=5)

        points = [[1.0, 1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 1.0]]
        box = BoundingBox.fromPoints(points)
        np.testing.assert_array_almost_equal(box.max, [1.0, 1.0, 1.0], decimal=5)
        np.testing.assert_array_almost_equal(box.min, [-1.0, -1.0, -1.0], decimal=5)
        np.testing.assert_array_almost_equal(box.center, [0.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(box.radius, 1.73205, decimal=5)

    def testTranslation(self):
        box = BoundingBox([1, 1, 1], [-1, -1, -1])
        box.translate(-2)
        np.testing.assert_array_almost_equal(box.max, [-1.0, -1.0, -1.0], decimal=5)
        np.testing.assert_array_almost_equal(box.min, [-3.0, -3.0, -3.0], decimal=5)
        np.testing.assert_array_almost_equal(box.center, [-2.0, -2.0, -2.0], decimal=5)
        np.testing.assert_array_almost_equal(box.radius, 1.73205, decimal=5)

        box.translate([1, 2, 3])
        np.testing.assert_array_almost_equal(box.max, [0.0, 1.0, 2.0], decimal=5)
        np.testing.assert_array_almost_equal(box.min, [-2.0, -1.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(box.center, [-1.0, 0.0, 1.0], decimal=5)
        np.testing.assert_array_almost_equal(box.radius, 1.73205, decimal=5)

    def testMerge(self):
        self.assertRaises(ValueError, BoundingBox.merge, [])
        boxes = [BoundingBox([1, 1, 1], [-1, -1, -1]), BoundingBox([1, 1, 2], [-1, -1, 1.5])]
        box = BoundingBox.merge(boxes)
        np.testing.assert_array_almost_equal(box.max, [1.0, 1.0, 2.0], decimal=5)
        np.testing.assert_array_almost_equal(box.min, [-1.0, -1.0, -1.0], decimal=5)
        np.testing.assert_array_almost_equal(box.center, [-0.0, 0.0, 0.5], decimal=5)
        np.testing.assert_array_almost_equal(box.radius, 2.06155, decimal=5)


class TestGeometryFunctions(unittest.TestCase):
    def testComputeFaceNormals(self):
        vertices = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]])
        indices = np.array([0, 0, 0, 1, 0, 2])  # first face has zero area
        normals = compute_face_normals(vertices[indices, :])
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        np.testing.assert_array_almost_equal(normals, expected, decimal=5)

        good_vertices, normals = compute_face_normals(vertices[indices, :], remove_degenerate=True)
        np.testing.assert_array_almost_equal(good_vertices, vertices[indices[3:], :], decimal=5)
        np.testing.assert_array_almost_equal(normals, expected[3:, :], decimal=5)

        vertices = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]])
        indices = np.array([0, 0, 0, 1, 0, 2])  # first face has zero area
        vertices = vertices[indices, :].reshape(-1, 9)  # arrange faces in a row
        normals = compute_face_normals(vertices)
        expected = np.array([[0, 0, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(normals, expected, decimal=5)

        good_vertices, normals = compute_face_normals(vertices, remove_degenerate=True)
        np.testing.assert_array_almost_equal(good_vertices, vertices[1:, :], decimal=5)
        np.testing.assert_array_almost_equal(normals, expected[1:, :], decimal=5)

    def testPathLengthCalculation(self):
        cube = create_cuboid(2, 4, 6)
        beam_axis = Vector3([1.0, 0.0, 0.0])
        gauge_volume = Vector3([0.0, 0.0, 0.0])
        diff_axis = [Vector3([0.0, 1.0, 0.0]), Vector3([0.0, 0.0, 1.0])]

        lengths = path_length_calculation(cube, gauge_volume, beam_axis, diff_axis)
        np.testing.assert_array_almost_equal(lengths, [4.0, 3.0], decimal=5)

        beam_axis = Vector3([0.0, -1.0, 0.0])
        lengths = path_length_calculation(cube, gauge_volume, beam_axis, diff_axis)
        np.testing.assert_array_almost_equal(lengths, [6.0, 5.0], decimal=5)

        # No hit
        beam_axis = Vector3([0.0, -1.0, 0.0])
        cube.vertices = cube.vertices - [0.0, 10.0, 0.0]
        lengths = path_length_calculation(cube, gauge_volume, beam_axis, diff_axis)
        np.testing.assert_array_almost_equal(lengths, [0.0, 0.0], decimal=5)

        # single detector
        diff_axis = [Vector3([0.0, 0.0, 1.0])]
        cube.vertices = cube.vertices + [0.0, 10.0, 0.0]
        length = path_length_calculation(cube, gauge_volume, beam_axis, diff_axis)
        self.assertAlmostEqual(*length, 5.0, 5)

        # beam outside at gauge volume
        cylinder = create_tube(2, 4, 6)
        beam_axis = Vector3([0.0, -1.0, 0.0])
        diff_axis = [Vector3([0.0, -1.0, 0.0])]
        length = path_length_calculation(cylinder, gauge_volume, beam_axis, diff_axis)
        self.assertAlmostEqual(*length, 0.0, 5)

        # beam cross more than a 2 faces
        cylinder.vertices = cylinder.vertices - [0.0, 3.0, 0.0]
        length = path_length_calculation(cylinder, gauge_volume, beam_axis, diff_axis)
        self.assertAlmostEqual(*length, 4.0, 5)

        # diff beam does not hit
        cube = create_cuboid(depth=0.0)
        cube.vertices = cube.vertices - [0.0, 1.0, 0.0]
        length = path_length_calculation(cube, gauge_volume, beam_axis, diff_axis)
        self.assertAlmostEqual(*length, 0.0, 5)

    def testClosestTriangleToPoint(self):
        cube = create_cuboid(2, 2, 2)
        faces = cube.vertices[cube.indices].reshape(-1, 9)
        points = np.array([[0.0, 1.0, 0.0], [2.0, 0.5, -0.1]])
        face = closest_triangle_to_point(faces, points)

        np.testing.assert_array_almost_equal(face[0], faces[2], decimal=5)
        np.testing.assert_array_almost_equal(face[1], faces[9], decimal=5)

    def testSegmentPlaneIntersection(self):
        point_a, point_b = np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
        plane = Plane.fromCoefficient(1.0, 0.0, 0.0, 0.0)
        intersection = segment_plane_intersection(point_a, point_b, plane)
        np.testing.assert_array_almost_equal(intersection, [0.0, 0.0, 0.0], decimal=5)

        # segment lies on plane
        # This is currently expected to return None
        point_a, point_b = np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
        intersection = segment_plane_intersection(point_a, point_b, plane)
        self.assertIsNone(intersection)

        # segment end is on plane
        point_a, point_b = np.array([0.5, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
        intersection = segment_plane_intersection(point_a, point_b, plane)
        np.testing.assert_array_almost_equal(intersection, [0.0, -1.0, 0.0], decimal=5)

        # segment that above plane
        point_a, point_b = np.array([0.5, 1.0, 0.0]), np.array([1.0, -1.0, 0.0])
        intersection = segment_plane_intersection(point_a, point_b, plane)
        self.assertIsNone(intersection)

    def testMeshPlaneIntersection(self):
        np.array([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        vertices = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        indices = np.array([0, 1, 2, 0, 2, 3])

        mesh = Mesh(vertices, indices)

        # plane is above geometry
        plane = Plane.fromCoefficient(1.0, 0.0, 0.0, 2.0)
        segments = mesh_plane_intersection(mesh, plane)
        self.assertEqual(len(segments), 0)

        # plane is intersects edge
        plane = Plane.fromCoefficient(1.0, 0.0, 0.0, -1.0)
        segments = mesh_plane_intersection(mesh, plane)
        self.assertEqual(len(segments), 2)

        # plane is intersects face
        plane = Plane.fromCoefficient(1.0, 0.0, 0.0, -0.5)
        segments = mesh_plane_intersection(mesh, plane)
        self.assertEqual(len(segments), 4)

        # plane is flush with face
        # This is currently expected to return nothing
        plane = Plane.fromCoefficient(0.0, 0.0, 1.0, 0.0)
        segments = mesh_plane_intersection(mesh, plane)
        self.assertEqual(len(segments), 0)

    def testSegmentTriangleIntersection(self):
        axis = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])
        length = 10.0
        faces = np.array([[-1.0, -0.5, 5.0, 0.5, 0.5, 5.0, 1.0, -0.5, 5.0],
                          [-1.0, -0.5, 7.0, 0.5, 0.5, 7.0, 1.0, -0.5, 7.0]])
        d = segment_triangle_intersection(origin, axis, length, faces)
        np.testing.assert_array_almost_equal(d, [5.0, 7.0], decimal=5)

        origin = np.array([0.0, 0.0, 6.0])
        d = segment_triangle_intersection(origin, axis, length, faces)
        np.testing.assert_array_almost_equal(d, [1.0], decimal=5)

        origin = np.array([-1.1, 0.0, 0.0])
        d = segment_triangle_intersection(origin, axis, length, faces)
        self.assertEqual(d, [])

        origin = np.array([0.0, -0.0, -6.0])
        d = segment_triangle_intersection(origin, axis, length, faces)
        self.assertEqual(d, [])

        origin = np.array([0.0, -0.6, 0.0])
        d = segment_triangle_intersection(origin, axis, length, faces)
        self.assertEqual(d, [])

        axis = np.array([1.0, 0.0, 0.0])
        d = segment_triangle_intersection(origin, axis, length, faces)
        self.assertEqual(d, [])

    def testPointSelection(self):
        start = Vector3([0.0, 0.0, 0.0])
        end = Vector3([0.0, 0.0, 10.0])
        faces = np.array([[-1.0, -0.5, 5.0, 0.5, 0.5, 5.0, 1.0, -0.5, 5.0],
                          [-1.0, -0.5, 7.0, 0.5, 0.5, 7.0, 1.0, -0.5, 7.0]])

        points = point_selection(start, end, faces, None)
        np.testing.assert_array_almost_equal(points, [[0.0, 0.0, 5.0], [0.0, 0.0, 7.0]], decimal=5)

        start = np.array([0.0, 0.0, 6.0])
        points = point_selection(start, end, faces, None)
        np.testing.assert_array_almost_equal(points, [[0.0, 0.0, 7.0]], decimal=5)

        start = np.array([-1.1, 0.0, 0.0])
        points = point_selection(start, end, faces, None)
        self.assertEqual(points.size, 0)

    def testVolumePlaneIntersection(self):
        data = np.ones([9, 9, 9], np.float32)
        data[3:6, 3:6, 3:6] = 0
        volume = Volume(data, np.ones(3), np.zeros(3))

        # plane is above geometry
        plane = Plane.fromCoefficient(1.0, 0.0, 0.0, 5.0)
        self.assertIsNone(volume_plane_intersection(volume, plane))

        # plane is above geometry
        plane = Plane.fromCoefficient(1.0, 0.0, 0.0, 0.0)
        volume_slice = volume_plane_intersection(volume, plane)
        expected = [volume_slice.image[0, 0], volume_slice.image[511, 511], volume_slice.image[1023, 1023]]
        np.testing.assert_array_almost_equal([1.0, 0.0, 1.0], expected, decimal=3)
        np.testing.assert_array_almost_equal([-4.5, -4.5, 9.0, 9.0], volume_slice.rect, decimal=3)

        plane = Plane.fromCoefficient(1.0, 0.0, 0.0, 3.0)
        volume_slice = volume_plane_intersection(volume, plane)
        expected = [volume_slice.image[0, 0], volume_slice.image[511, 511], volume_slice.image[1023, 1023]]
        np.testing.assert_array_almost_equal([1.0, 1.0, 1.0], expected, decimal=3)
        np.testing.assert_array_almost_equal([-4.5, -4.5, 9.0, 9.0], volume_slice.rect, decimal=3)

        matrix = np.identity(4)
        matrix[0, 3] = -3
        volume.transform(matrix)
        volume_slice = volume_plane_intersection(volume, plane)
        expected = [volume_slice.image[0, 0], volume_slice.image[511, 511], volume_slice.image[1023, 1023]]
        np.testing.assert_array_almost_equal([1.0, 0.0, 1.0], expected, decimal=3)
        np.testing.assert_array_almost_equal([-4.5, -4.5, 9.0, 9.0], volume_slice.rect, decimal=3)
        self.assertEqual((1024, 1024), volume_slice.image.shape)

        volume = Volume(data, np.array([.5, 1., 2.]), np.zeros(3))
        matrix = np.identity(4)
        matrix[0:2, 0:2] = [[0.7071068, -0.7071068], [0.7071068, 0.7071068]]
        volume.rotate(matrix)
        plane = Plane.fromCoefficient(0.0, 1.0, 0.0, 1.0)
        volume_slice = volume_plane_intersection(volume, plane, resolution=512)
        expected = [volume_slice.image[1, 1], volume_slice.image[255, 255], volume_slice.image[510, 510]]
        np.testing.assert_array_almost_equal([1.0, 0.253, 1.0], expected, decimal=3)
        np.testing.assert_array_almost_equal([-2.182, -9., 6.364, 18], volume_slice.rect, decimal=3)
        self.assertEqual((512, 512), volume_slice.image.shape)

    def testVolumeRayIntersection(self):
        data = np.array(
            [[[100, 100, 100], [100, 100, 100], [100, 100, 100]], [[100, 100, 100], [100, 100, 100], [100, 100, 100]],
             [[100, 100, 100], [100, 100, 100], [100, 100, 100]]],
            dtype=np.uint8)
        voxel_size = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        volume = Volume(data, voxel_size, np.zeros(3))

        start = Vector3([5.0, 0, 0])
        end = Vector3([-5.0, 0, 0])
        intersection_dist = volume_ray_intersection(start, end, volume)[0]
        self.assertAlmostEqual(2, intersection_dist, 3)

        start = Vector3([10.0, 10.0, 0])
        end = Vector3([-10.0, 10.0, 0])
        empty_intersection = volume_ray_intersection(start, end, volume)
        self.assertIsNone(empty_intersection)


class TestVolumeClass(unittest.TestCase):
    def testCreation(self):
        transform = np.identity(4)
        transform[:3, 3] = [1, 1, 1]
        data = np.zeros([3, 3, 3], np.uint8)
        volume = Volume(data, np.ones(3), np.ones(3))
        self.assertEqual(volume.shape, (3, 3, 3))
        self.assertNotEqual(volume.curve.inputs[0], volume.curve.inputs[1])
        np.testing.assert_array_equal(volume.data, data)
        np.testing.assert_array_almost_equal(volume.extent, [3, 3, 3], decimal=5)
        np.testing.assert_array_almost_equal(volume.transform_matrix, transform, decimal=5)
        self.assertIs(volume.data, volume.render_target)

        transform[:3, 3] = [1, 0.25, 0]
        data = np.full((5, 4, 3), [0, 127, 255], np.float32).transpose()
        volume = Volume(data, np.array([1, 0.5, 2]), np.array([1, 0.25, 0]), max_bytes=10, max_dim=4)
        self.assertEqual(volume.shape, (3, 4, 5))
        self.assertNotEqual(volume.curve.inputs[0], volume.curve.inputs[1])
        np.testing.assert_array_almost_equal(volume.data, data, decimal=5)
        np.testing.assert_array_almost_equal(volume.extent, (3, 2, 10), decimal=5)
        np.testing.assert_array_almost_equal(volume.transform_matrix, transform, decimal=5)
        self.assertEqual(volume.render_target.shape, (2, 3, 4))

    def testCurve(self):
        x = np.array([30])
        y = np.array([0.5])
        tf = np.tile(np.linspace(0.0, 1.0, num=256, dtype=np.float32)[:, None], (1, 4))
        tf[:, 3] = 0.5
        curve = Curve(x, y, x, Curve.Type.Cubic)
        np.testing.assert_array_equal(curve.transfer_function, tf.flatten())
        self.assertAlmostEqual(curve.evaluate([40]), 0.5)
        curve = Curve(x, y, x, Curve.Type.Linear)
        np.testing.assert_array_almost_equal(curve.transfer_function, tf.flatten(), decimal=5)
        self.assertAlmostEqual(curve.evaluate([20]), 0.5)

        x = np.array([20, 200])
        y = np.array([0.078, 0.784])
        curve = Curve(x, y, x, Curve.Type.Cubic)
        np.testing.assert_array_almost_equal(curve.evaluate([0, 90, 210]), [0.078, 0.35256, 0.784], decimal=5)
        curve = Curve(x, y, x, Curve.Type.Linear)
        np.testing.assert_array_almost_equal(curve.evaluate([0, 90, 210]), [0.078, 0.35256, 0.784], decimal=5)

        x = np.array([0, 30, 255])
        y = np.array([0.0, 0.5, 1.0])
        curve = Curve(x, y, x, Curve.Type.Cubic)
        np.testing.assert_array_almost_equal(curve.evaluate([20, 50, 120]), [0.34466231, 0.77668845, 1.0], decimal=5)
        curve = Curve(x, y, x, Curve.Type.Linear)
        np.testing.assert_array_almost_equal(curve.evaluate([20, 50, 120]), [0.333333, 0.544444, 0.7], decimal=5)
