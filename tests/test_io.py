import unittest
import unittest.mock as mock
import shutil
import tempfile
import os
import numpy as np
from sscanss.core.io import reader, writer
from sscanss.core.geometry import Mesh
from sscanss.core.instrument import read_instrument_description_file
from sscanss.core.math import Matrix44
from sscanss.config import __version__


idf = '''{
    "instrument":{
        "name": "GENERIC",
        "version": "1.0",
        "gauge_volume": [0.0, 0.0, 0.0],
        "incident_jaws":{
            "beam_direction": [1.0, 0.0, 0.0],
            "beam_source": [-300.0, 0.0, 0.0],
            "aperture": [1.0, 1.0],
            "aperture_upper_limit": [0.5, 0.5],
            "aperture_lower_limit": [15.0, 15.0],
            "positioner": "incident_jaws",
            "visual":{
                    "pose": [300.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
            }
        },
        "detectors":[
            {
                "name":"Detector",
                "default_collimator": "Snout 25mm",
                "positioner": "diffracted_jaws",
                "diffracted_beam": [0.0, 1.0, 0.0]
            }
        ],
        "collimators":[
            {
                "name": "Snout 25mm",
                "detector": "Detector",
                "aperture": [1.0, 1.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
			{
                "name": "Snout 50mm",
                "detector": "Detector",
                "aperture": [2.0, 2.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
			{
                "name": "Snout 100mm",
                "detector": "Detector",
                "aperture": [1.0, 1.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
            {
                "name": "Snout 150mm",
                "detector": "Detector",
                "aperture": [4.0, 4.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            }
        ],
			"positioning_stacks":[
            {
                "name": "Positioning Table Only",
                "positioners": ["Positioning Table"]
            },
            {
                "name": "Positioning Table + Huber Circle",
                "positioners": ["Positioning Table", "Huber Circle"]
            }
		],
        "positioners":[
            {
                "name": "Positioning Table",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints":[
                    {
                        "name": "X Stage",
                        "type": "prismatic",
                        "axis": [1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -201.0,
                        "upper_limit": 192.0,
                        "parent": "y_stage",
                        "child": "x_stage"
                    },
                    {
                        "name": "Y Stage",
                        "type": "prismatic",
                        "axis": [0.0, 1.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -101.0,
                        "upper_limit": 93.0,
                        "parent": "omega_stage",
                        "child": "y_stage"
                    },
                    {
                        "name": "Omega Stage",
                        "type": "revolute",
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -170.0,
                        "upper_limit": 166.0,
                        "parent": "base",
                        "child": "omega_stage"
                    }],
                "links": [
                    {"name": "base"},
                    {"name": "omega_stage"},
                    {"name": "y_stage"},
                    {"name": "x_stage"}
                ]
            },
            {
                "name": "Huber Circle",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Chi",
                        "type": "revolute",
                        "axis": [0.0, 1.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": 0.0,
                        "upper_limit": 300.0,
						"home_offset": 0.0,
                        "parent": "base",
                        "child": "chi_axis"
                    },
                    {
                        "name": "Phi",
                        "type": "revolute",
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -360.0,
                        "upper_limit": 360.0,
                        "parent": "chi_axis",
                        "child": "phi_axis"
                    }

                ],
                "links": [
                    {"name": "base"},
                    {"name": "chi_axis"},
                    {"name": "phi_axis"}
                ]
            },
            {
                "name": "incident_jaws",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Jaws X Axis",
                        "type": "prismatic",
                        "axis": [1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -800.0,
                        "upper_limit": 0.0,
						"home_offset": 0.0,
                        "parent": "base",
                        "child": "jaw_x_axis"
                    }

                ],
                "links": [
                    {"name": "base"},
                    {"name": "jaw_x_axis"}
                ]
            },
			{
                "name": "diffracted_jaws",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Angular Axis",
                        "type": "revolute",
                        "axis": [0.0, 0.0, -1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -120.0,
                        "upper_limit": 120.0,
						"home_offset": 0.0,
                        "parent": "base",
                        "child": "angular_axis"
                    }, 
					{
                        "name": "Radial Axis",
                        "type": "prismatic",
                        "axis": [-1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": 0.0,
                        "upper_limit": 100.0,
						"home_offset": 0.0,
                        "parent": "angular_axis",
                        "child": "radial_axis"
                    }
                ],
                "links": [
                    {"name": "base"},
                    {"name": "angular_axis"}, 
					{"name": "radial_axis"}
                ]
            }			
        ],
        "fixed_hardware":[]
    }
}'''


class TestIO(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    @mock.patch('sscanss.core.instrument.create.read_visuals', autospec=True)
    def testHDFReadWrite(self, mocked_function):

        mocked_function.return_value = Mesh(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]), np.array([0, 1, 2]),
                                            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
        filename = self.writeTestFile('instrument.json', idf)
        instrument = read_instrument_description_file(filename)
        data = {'name': 'Test Project',
                'instrument': instrument,
                'instrument_version': "1.0",
                'sample': {},
                'fiducials': np.recarray((0, ), dtype=[('points', 'f4', 3), ('enabled', '?')]),
                'measurement_points': np.recarray((0,), dtype=[('points', 'f4', 3), ('enabled', '?')]),
                'measurement_vectors': np.empty((0, 3, 1), dtype=np.float32),
                'alignment': None}

        filename = os.path.join(self.test_dir, 'test.h5')

        writer.write_project_hdf(data, filename)
        result, instrument = reader.read_project_hdf(filename)

        self.assertEqual(__version__, result['version'])
        self.assertEqual(data['instrument_version'], result['instrument_version'])
        self.assertEqual(data['name'], result['name'], 'Save and Load data are not Equal')
        self.assertEqual(data['instrument'].name, result['instrument'], 'Save and Load data are not Equal')
        self.assertDictEqual(result['sample'], {})
        self.assertTrue(result['fiducials'][0].size == 0 and result['fiducials'][1].size == 0)
        self.assertTrue(result['measurement_points'][0].size == 0 and result['measurement_points'][1].size == 0)
        self.assertTrue(result['measurement_vectors'].size == 0)
        self.assertIsNone(result['alignment'])

        sample_key = 'a mesh'
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        mesh_to_write = Mesh(vertices, indices, normals)
        fiducials = np.rec.array([([11., 12., 13.], False), ([14., 15., 16.], True), ([17., 18., 19.], False)],
                                dtype=[('points', 'f4', 3), ('enabled', '?')])
        points = np.rec.array([([1., 2., 3.], True), ([4., 5., 6.], False), ([7., 8., 9.], True)],
                            dtype=[('points', 'f4', 3), ('enabled', '?')])
        vectors = np.ones((3, 3, 2))
        base = Matrix44(np.random.random((4, 4)))
        stack_name = 'Positioning Table + Huber Circle'
        new_collimator = 'Snout 100mm'
        jaw_aperture = [7., 5.]

        data = {'name': 'demo', 'instrument': instrument, 'instrument_version': "1.1",
                'sample': {sample_key: mesh_to_write}, 'fiducials': fiducials, 'measurement_points': points,
                'measurement_vectors': vectors, 'alignment': np.identity(4)}

        instrument.loadPositioningStack(stack_name)
        instrument.positioning_stack.fkine([200., 0., 0., np.pi, 0.])
        instrument.positioning_stack.links[0].ignore_limits = True
        instrument.positioning_stack.links[4].locked = True
        aux = instrument.positioning_stack.auxiliary[0]
        instrument.positioning_stack.changeBaseMatrix(aux, base)

        instrument.jaws.aperture = jaw_aperture
        instrument.jaws.positioner.fkine([-600.0])
        instrument.jaws.positioner.links[0].ignore_limits = True
        instrument.jaws.positioner.links[0].locked = True

        instrument.detectors['Detector'].current_collimator = new_collimator
        instrument.detectors['Detector'].positioner.fkine([np.pi/2, 100.0])
        instrument.detectors['Detector'].positioner.links[0].ignore_limits = True
        instrument.detectors['Detector'].positioner.links[1].locked = True

        writer.write_project_hdf(data, filename)
        result, instrument2 = reader.read_project_hdf(filename)
        self.assertEqual(__version__, result['version'])
        self.assertEqual(data['name'], result['name'], 'Save and Load data are not Equal')
        self.assertEqual(data['instrument_version'], result['instrument_version'])
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

        self.assertEqual(instrument.positioning_stack.name, instrument2.positioning_stack.name)
        np.testing.assert_array_almost_equal(instrument.positioning_stack.configuration,
                                             instrument2.positioning_stack.configuration, decimal=5)
        for link1, link2 in zip(instrument.positioning_stack.links, instrument2.positioning_stack.links):
            self.assertEqual(link1.ignore_limits, link2.ignore_limits)
            self.assertEqual(link1.locked, link2.locked)
        for aux1, aux2 in zip(instrument.positioning_stack.auxiliary, instrument2.positioning_stack.auxiliary):
            np.testing.assert_array_almost_equal(aux1.base, aux2.base, decimal=5)

        np.testing.assert_array_almost_equal(instrument.jaws.aperture, instrument2.jaws.aperture, decimal=5)
        np.testing.assert_array_almost_equal(instrument.jaws.aperture_lower_limit,
                                             instrument2.jaws.aperture_lower_limit, decimal=5)
        np.testing.assert_array_almost_equal(instrument.jaws.aperture_upper_limit,
                                             instrument2.jaws.aperture_upper_limit, decimal=5)
        np.testing.assert_array_almost_equal(instrument.jaws.positioner.configuration,
                                             instrument2.jaws.positioner.configuration, decimal=5)
        for link1, link2 in zip(instrument.jaws.positioner.links, instrument2.jaws.positioner.links):
            self.assertEqual(link1.ignore_limits, link2.ignore_limits)
            self.assertEqual(link1.locked, link2.locked)

        detector1 = instrument.detectors['Detector']
        detector2 = instrument2.detectors['Detector']
        self.assertEqual(detector1.current_collimator.name, detector2.current_collimator.name)
        np.testing.assert_array_almost_equal(detector1.positioner.configuration,
                                             detector2.positioner.configuration, decimal=5)
        for link1, link2 in zip(detector1.positioner.links, detector2.positioner.links):
            self.assertEqual(link1.ignore_limits, link2.ignore_limits)
            self.assertEqual(link1.locked, link2.locked)

    def testReadObj(self):
        # Write Obj file
        obj = ('# Demo\n'
               'v 0.5 0.5 0.0\n'
               'v -0.5 0.0 0.0\n'
               'v 0.0 0.0 0.0\n'
               '\n'
               'usemtl material_0\n'
               'f 1//1 2//2 3//3\n'
               '\n'
               '# End of file')

        filename = self.writeTestFile('test.obj', obj)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_3d_model(filename)
        np.testing.assert_array_almost_equal(mesh.vertices[mesh.indices], vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals[mesh.indices], normals, decimal=5)

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

        # cleaning the mesh will result in sorted vertices
        vertices = np.array([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_3d_model(filename)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals, normals, decimal=5)
        np.testing.assert_array_equal(mesh.indices, np.array([2, 0, 1]))

    def testReadAndWriteBinaryStl(self):
        vertices = np.array([[1, 2, 0], [4, 5, 0], [7, 28, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        mesh_to_write = Mesh(vertices, indices, normals)
        full_path = os.path.join(self.test_dir, 'test.stl')
        writer.write_binary_stl(full_path, mesh_to_write)

        mesh_read_from_file = reader.read_3d_model(full_path)
        np.testing.assert_array_almost_equal(mesh_to_write.vertices, mesh_read_from_file.vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh_to_write.normals, mesh_read_from_file.normals, decimal=5)
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
