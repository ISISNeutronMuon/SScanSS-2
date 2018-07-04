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

        filename = os.path.join(self.test_dir, 'demo.obj')
        with open(filename, 'w') as obj_file:
            obj_file.write(obj)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_obj(filename)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, np.array([0, 1, 2]))

    def testReadAsciiStl(self):
        # Write STL file
        obj = ('solid STL generated for demo\n'
               'facet normal 0.0 0.0 1.0\n'
               '  outer loop\n'
               '    vertex  0.5 0.5 0.0\n'
               '    vertex  -0.5 0.0 0.0\n'
               '    vertex  0.0 0.0 0.0\n'
               '  endloop\n'
               'endfacet\n'
               'endsolid demo\n')

        filename = os.path.join(self.test_dir, 'demo.stl')
        with open(filename, 'w') as obj_file:
            obj_file.write(obj)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_stl(filename)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, np.array([0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
