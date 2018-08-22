import unittest
import shutil
import tempfile
import os
import numpy as np
from sscanss.core.io import reader, writer


class TestIO(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testHDFReadWrite(self):
        data = {'name': 'Test Project', 'instrument': 'IMAT'}

        filename = os.path.join(self.test_dir, 'test.h5')

        writer.write_project_hdf(data, filename)
        self.assertTrue(os.path.isfile(filename), 'write_project_hdf failed to write file')

        result = reader.read_project_hdf(filename)

        self.assertEqual(data['name'], result['name'], 'Save and Load data are not Equal')
        self.assertEqual(data['instrument'], result['instrument'], 'Save and Load data are not Equal')

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

        filename = self.write_test_file('demo.obj', obj)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_obj(filename)
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

        filename = self.write_test_file('demo.stl', stl)
        with open(filename, 'w') as stl_file:
            stl_file.write(stl)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_stl(filename)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, np.array([0, 1, 2]))

    def testReadCSV(self):
        csvs = ['1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n',
                '1.0\t 2.0,3.0\n4.0, 5.0\t 6.0\n7.0, 8.0, 9.0\n',
                '1.0\t 2.0\t 3.0\n4.0\t 5.0\t 6.0\n7.0\t 8.0\t 9.0\n\n']

        for csv in csvs:
            filename = self.write_test_file('demo.csv', csv)

            data = reader.read_csv(filename)
            expected = [['1.0', '2.0', '3.0'], ['4.0', '5.0', '6.0'], ['7.0', '8.0', '9.0']]

            np.testing.assert_array_equal(data, expected)

    def testReadPoints(self):
        csv = '1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n'
        filename = self.write_test_file('demo.csv', csv)
        data = reader.read_points(filename)
        expected = ([['1.0', '2.0', '3.0'], ['4.0', '5.0', '6.0'], ['7.0', '8.0', '9.0']], [True, True, True])
        np.testing.assert_array_equal(data, expected)

        csv = '1.0, 2.0, 3.0, false\n4.0, 5.0, 6.0, True\n7.0, 8.0, 9.0\n'
        filename = self.write_test_file('demo.csv', csv)
        data = reader.read_points(filename)
        expected = ([['1.0', '2.0', '3.0'], ['4.0', '5.0', '6.0'], ['7.0', '8.0', '9.0']], [False, True, True])
        np.testing.assert_array_equal(data, expected)

        csv = '1.0, 3.9, 2.0, 3.0, false\n4.0, 5.0, 6.0, True\n7.0, 8.0, 9.0\n'  # point with 4 values
        filename = self.write_test_file('demo.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_points(filename)

    def testReadTransMatrix(self):
        csv = '1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'
        filename = self.write_test_file('demo.csv', csv)
        data = reader.read_trans_matrix(filename)
        expected = [['1.0', '2.0', '3.0', '4.0'],
                    ['1.0', '2.0', '3.0', '4.0'],
                    ['1.0', '2.0', '3.0', '4.0'],
                    ['1.0', '2.0', '3.0', '4.0']]
        np.testing.assert_array_equal(data, expected)

        csv = '1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'  # missing last row
        filename = self.write_test_file('demo.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_trans_matrix(filename)

        csv = '1.0, 2.0, 3.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n'  # incorrect col size
        filename = self.write_test_file('demo.csv', csv)
        with self.assertRaises(ValueError):
            reader.read_trans_matrix(filename)

    def write_test_file(self, filename, text):
        full_path = os.path.join(self.test_dir, filename)
        with open(full_path, 'w') as text_file:
            text_file.write(text)
        return full_path


if __name__ == '__main__':
    unittest.main()
