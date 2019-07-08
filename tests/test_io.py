import unittest
import shutil
import tempfile
import os
from collections import namedtuple
import numpy as np
from sscanss.core.io import reader, writer
from sscanss.core.geometry import Mesh


class TestIO(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testHDFReadWrite(self):
        Instrument = namedtuple('Instrument', ['name'])
        data = {'name': 'Test Project',
                'instrument': Instrument('IMAT'),
                'sample': {},
                'fiducials': np.recarray((0, ), dtype=[('points', 'f4', 3), ('enabled', '?')]),
                'measurement_points': np.recarray((0,), dtype=[('points', 'f4', 3), ('enabled', '?')]),
                'measurement_vectors': np.empty((0, 3, 1), dtype=np.float32),
                'alignment': None}

        filename = os.path.join(self.test_dir, 'test.h5')

        writer.write_project_hdf(data, filename)
        result = reader.read_project_hdf(filename)

        self.assertEqual(data['name'], result['name'], 'Save and Load data are not Equal')
        self.assertEqual(data['instrument'].name, result['instrument'], 'Save and Load data are not Equal')
        self.assertDictEqual(result['sample'], {})
        self.assertTrue(result['fiducials'][0].size == 0 and result['fiducials'][1].size == 0)
        self.assertTrue(result['measurement_points'][0].size == 0 and result['measurement_points'][1].size == 0)
        self.assertTrue(result['measurement_vectors'].size == 0)
        self.assertIsNone(result['alignment'])

        sample_key = 'a mesh'
        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        mesh_to_write = Mesh(vertices, indices, normals)
        fiducials = np.rec.array([([11., 12., 13.], False), ([14., 15., 16.], True), ([17., 18., 19.], False)],
                                dtype=[('points', 'f4', 3), ('enabled', '?')])
        points = np.rec.array([([1., 2., 3.], True), ([4., 5., 6.], False), ([7., 8., 9.], True)],
                            dtype=[('points', 'f4', 3), ('enabled', '?')])
        vectors = np.ones((3, 3, 2))

        data = {'name': 'demo', 'instrument': Instrument('ENGIN-X'), 'sample': {sample_key: mesh_to_write},
                'fiducials': fiducials, 'measurement_points': points, 'measurement_vectors': vectors,
                'alignment': np.identity(4)}

        writer.write_project_hdf(data, filename)
        result = reader.read_project_hdf(filename)
        self.assertEqual(data['name'], result['name'], 'Save and Load data are not Equal')
        self.assertEqual(data['instrument'].name, result['instrument'], 'Save and Load data are not Equal')
        self.assertTrue(sample_key in result['sample'])
        np.testing.assert_array_almost_equal(fiducials.points, result['fiducials'][0])
        np.testing.assert_array_almost_equal(points.points,  result['measurement_points'][0])
        np.testing.assert_array_almost_equal(fiducials.points, result['fiducials'][0])
        np.testing.assert_array_almost_equal(points.points,  result['measurement_points'][0])
        np.testing.assert_array_equal(fiducials.enabled, result['fiducials'][1])
        np.testing.assert_array_equal(points.enabled, result['measurement_points'][1])
        np.testing.assert_array_almost_equal(vectors, result['measurement_vectors'], decimal=5)
        np.testing.assert_array_almost_equal(result['alignment'], np.identity(4), decimal=5)



    def testReadObj(self):
        # Write Obj file
        obj = ('# Demo\n'
               'v 0.5 0.5 0.0\n'
               'vn 0.0 0.0 1.0\n'
               'v -0.5 0.0 0.0\n'
               'vn 0.0 0.0 1.0\n'
               'v 0.0 0.0 0.0\n'
               'vn 0.0 0.0 1.0\n'
               '\n'
               'usemtl material_0\n'
               'f 1//1 2//2 3//3\n'
               '\n'
               '# End of file')

        filename = self.writeTestFile('test.obj', obj)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_3d_model(filename)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, np.array([0, 1, 2]))

    def testReadAsciiStl(self):
        # Write STL file
        stl = ('solid STL generated for demo\n'
               'facet normal 0.0 0.0 1.0\n'
               '  outer loop\n'
               '    vertex  0.5 0.5 0.0\n'
               '    vertex  -0.5 0.0 0.0\n'
               '    vertex  0.0 0.0 0.0\n'
               '  endloop\n'
               'endfacet\n'
               'endsolid demo\n')

        filename = self.writeTestFile('test.stl', stl)
        with open(filename, 'w') as stl_file:
            stl_file.write(stl)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_3d_model(filename)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, np.array([0, 1, 2]))

    def testReadAndWriteBinaryStl(self):
        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        mesh_to_write = Mesh(vertices, indices, normals)
        full_path = os.path.join(self.test_dir, 'test.stl')
        writer.write_binary_stl(full_path, mesh_to_write)

        mesh_read_from_file = reader.read_3d_model(full_path)
        np.testing.assert_array_almost_equal( mesh_to_write.vertices, mesh_read_from_file.vertices, decimal=5)
        np.testing.assert_array_almost_equal( mesh_to_write.normals, mesh_read_from_file.normals, decimal=5)
        np.testing.assert_array_equal(mesh_to_write.indices, mesh_read_from_file.indices)

    def testReadCsv(self):
        csvs = ['1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n',
                '1.0\t 2.0,3.0\n4.0, 5.0\t 6.0\n7.0, 8.0, 9.0\n',
                '1.0\t 2.0\t 3.0\n4.0\t 5.0\t 6.0\n7.0\t 8.0\t 9.0\n\n']

        for csv in csvs:
            filename = self.writeTestFile('test.csv', csv)

            data = reader.read_csv(filename)
            expected = [['1.0', '2.0', '3.0'], ['4.0', '5.0', '6.0'], ['7.0', '8.0', '9.0']]

            np.testing.assert_array_equal(data, expected)

    def testReadPoints(self):
        csv = '1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n'
        filename = self.writeTestFile('test.csv', csv)
        data = reader.read_points(filename)
        expected = ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [True, True, True])
        np.testing.assert_array_equal(data[1], expected[1])
        np.testing.assert_array_almost_equal(data[0], expected[0], decimal=5)

        csv = '1.0, 2.0, 3.0, false\n4.0, 5.0, 6.0, True\n7.0, 8.0, 9.0\n'
        filename = self.writeTestFile('test.csv', csv)
        data = reader.read_points(filename)
        expected = ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [False, True, True])
        np.testing.assert_array_equal(data[1], expected[1])
        np.testing.assert_array_almost_equal(data[0], expected[0], decimal=5)

        csv = '1.0, 3.9, 2.0, 3.0, false\n4.0, 5.0, 6.0, True\n7.0, 8.0, 9.0\n'  # point with 4 values
        filename = self.writeTestFile('test.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_points(filename)

        points = np.rec.array([([11., 12., 13.], True), ([14., 15., 16.], False), ([17., 18., 19.], True)],
                            dtype=[('points', 'f4', 3), ('enabled', '?')])
        filename = os.path.join(self.test_dir, 'test.csv')
        writer.write_points(filename, points)
        data, state = reader.read_points(filename)
        np.testing.assert_array_equal(state, points.enabled)
        np.testing.assert_array_almost_equal(data, points.points)

    def testReadVectors(self):
        csv = '1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'
        filename = self.writeTestFile('test.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_vectors(filename)

        csv = '1.0, 2.0, 3.0,4.0, 5.0, 6.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'
        filename = self.writeTestFile('test.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_vectors(filename)

        csv = '1.0,2.0,3.0,4.0,5.0,6.0\n,1.0,2.0,3.0,4.0,5.0,6.0\n1.0,2.0,3.0,4.0,5.0,6.0\n\n'
        filename = self.writeTestFile('test.csv', csv)
        data = reader.read_vectors(filename)
        expected = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

        csv = '1.0,2.0,3.0\n,1.0,2.0,3.0\n1.0,2.0,3.0\n'
        filename = self.writeTestFile('test.csv', csv)
        data = reader.read_vectors(filename)
        expected = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

    def testReadTransMatrix(self):
        csv = '1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'
        filename = self.writeTestFile('test.csv', csv)
        data = reader.read_trans_matrix(filename)
        expected = [[1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0]]
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

        csv = '1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'  # missing last row
        filename = self.writeTestFile('test.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_trans_matrix(filename)

        csv = '1.0, 2.0, 3.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'  # incorrect col size
        filename = self.writeTestFile('test.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_trans_matrix(filename)

    def testReadFpos(self):
        csv = ('1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n, 2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n'
               '3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n')
        filename = self.writeTestFile('test.csv', csv)
        index, points, pose = reader.read_fpos(filename)
        expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        np.testing.assert_equal(index, [0, 1, 2, 3])
        np.testing.assert_array_almost_equal(points, expected[:, 0:3], decimal=5)
        np.testing.assert_array_almost_equal(pose, expected[:, 3:], decimal=5)

        csv = '9, 1.0, 2.0, 3.0\n, 1, 1.0, 2.0, 3.0\n, 3, 1.0, 2.0, 3.0\n, 6, 1.0, 2.0, 3.0\n'
        filename = self.writeTestFile('test.csv', csv)
        index, points, pose = reader.read_fpos(filename)
        np.testing.assert_equal(index, [8, 0, 2, 5])
        np.testing.assert_array_almost_equal(points, expected[:, 0:3], decimal=5)
        self.assertEqual(pose.size, 0)

        csv = '1.0, 2.0, 3.0\n, 1.0, 2.0, 3.0\n1.0, 2.0, 3.0\n'  # missing index column
        filename = self.writeTestFile('test.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_fpos(filename)

        csv = ('9, 1.0, 2.0, 3.0, 5.0\n, 1, 1.0, 2.0, 3.0\n, '
               '3, 1.0, 2.0, 3.0\n, 6, 1.0, 2.0, 3.0\n')  # incorrect col size
        filename = self.writeTestFile('test.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_fpos(filename)

    def writeTestFile(self, filename, text):
        full_path = os.path.join(self.test_dir, filename)
        with open(full_path, 'w') as text_file:
            text_file.write(text)
        return full_path


if __name__ == '__main__':
    unittest.main()
