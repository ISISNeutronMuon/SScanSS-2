import unittest
import unittest.mock as mock
import shutil
import tempfile
import os
import numpy as np
from sscanss.core.geometry import Mesh
from sscanss.core.instrument import read_instrument_description_file, Link
from sscanss.core.io import reader, writer
from sscanss.core.math import Matrix44
from sscanss.config import __version__
from tests.helpers import SAMPLE_IDF
import h5py


class TestIO(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    @mock.patch("sscanss.core.io.writer.settings", autospec=True)
    @mock.patch("sscanss.core.instrument.create.read_visuals", autospec=True)
    def testHDFReadWrite(self, visual_fn, setting_cls):

        visual_fn.return_value = Mesh(
            np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]),
            np.array([0, 1, 2]),
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        filename = self.writeTestFile("instrument.json", SAMPLE_IDF)
        instrument = read_instrument_description_file(filename)
        data = {
            "name": "Test Project",
            "instrument": instrument,
            "instrument_version": "1.0",
            "sample": {},
            "fiducials": np.recarray((0, ), dtype=[("points", "f4", 3), ("enabled", "?")]),
            "measurement_points": np.recarray((0, ), dtype=[("points", "f4", 3), ("enabled", "?")]),
            "measurement_vectors": np.empty((0, 3, 1), dtype=np.float32),
            "alignment": None,
        }

        filename = os.path.join(self.test_dir, "test.h5")

        writer.write_project_hdf(data, filename)
        result, instrument = reader.read_project_hdf(filename)

        self.assertEqual(__version__, result["version"])
        self.assertEqual(data["instrument_version"], result["instrument_version"])
        self.assertEqual(data["name"], result["name"], "Save and Load data are not Equal")
        self.assertEqual(data["instrument"].name, result["instrument"], "Save and Load data are not Equal")
        self.assertDictEqual(result["sample"], {})
        self.assertTrue(result["fiducials"][0].size == 0 and result["fiducials"][1].size == 0)
        self.assertTrue(result["measurement_points"][0].size == 0 and result["measurement_points"][1].size == 0)
        self.assertTrue(result["measurement_vectors"].size == 0)
        self.assertIsNone(result["alignment"])
        self.assertEqual(result["settings"], {})

        sample_key = "a mesh"
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        indices = np.array([0, 1, 2])
        mesh_to_write = Mesh(vertices, indices, normals)
        fiducials = np.rec.array(
            [([11.0, 12.0, 13.0], False), ([14.0, 15.0, 16.0], True), ([17.0, 18.0, 19.0], False)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )
        points = np.rec.array(
            [([1.0, 2.0, 3.0], True), ([4.0, 5.0, 6.0], False), ([7.0, 8.0, 9.0], True)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )
        vectors = np.zeros((3, 3, 2))
        vectors[:, :, 0] = [
            [0.0000076, 1.0000000, 0.0000480],
            [0.0401899, 0.9659270, 0.2556752],
            [0.1506346, 0.2589932, 0.9540607],
        ]

        vectors[:, :, 1] = [
            [0.1553215, -0.0000486, 0.9878640],
            [0.1499936, -0.2588147, 0.9542100],
            [0.0403915, -0.9658791, 0.2558241],
        ]
        base = Matrix44(np.random.random((4, 4)))
        stack_name = "Positioning Table + Huber Circle"
        new_collimator = "Snout 100mm"
        jaw_aperture = [7.0, 5.0]

        data = {
            "name": "demo",
            "instrument": instrument,
            "instrument_version": "1.1",
            "sample": {
                sample_key: mesh_to_write
            },
            "fiducials": fiducials,
            "measurement_points": points,
            "measurement_vectors": vectors,
            "alignment": np.identity(4),
        }

        instrument.loadPositioningStack(stack_name)
        instrument.positioning_stack.fkine([200.0, 0.0, 0.0, np.pi, 0.0])
        instrument.positioning_stack.links[0].ignore_limits = True
        instrument.positioning_stack.links[4].locked = True
        aux = instrument.positioning_stack.auxiliary[0]
        instrument.positioning_stack.changeBaseMatrix(aux, base)

        instrument.jaws.aperture = jaw_aperture
        instrument.jaws.positioner.fkine([-600.0])
        instrument.jaws.positioner.links[0].ignore_limits = True
        instrument.jaws.positioner.links[0].locked = True

        instrument.detectors["Detector"].current_collimator = new_collimator
        instrument.detectors["Detector"].positioner.fkine([np.pi / 2, 100.0])
        instrument.detectors["Detector"].positioner.links[0].ignore_limits = True
        instrument.detectors["Detector"].positioner.links[1].locked = True

        setting_cls.local = {"num": 1, "str": "string", "colour": (1, 1, 1, 1)}

        writer.write_project_hdf(data, filename)
        result, instrument2 = reader.read_project_hdf(filename)
        self.assertEqual(__version__, result["version"])
        self.assertEqual(data["name"], result["name"], "Save and Load data are not Equal")
        self.assertEqual(data["instrument_version"], result["instrument_version"])
        self.assertEqual(data["instrument"].name, result["instrument"], "Save and Load data are not Equal")
        self.assertTrue(sample_key in result["sample"])
        np.testing.assert_array_almost_equal(fiducials.points, result["fiducials"][0], decimal=5)
        np.testing.assert_array_almost_equal(points.points, result["measurement_points"][0], decimal=5)
        np.testing.assert_array_almost_equal(result["sample"][sample_key].vertices, vertices, decimal=5)
        np.testing.assert_array_almost_equal(result["sample"][sample_key].indices, indices, decimal=5)
        np.testing.assert_array_almost_equal(result["sample"][sample_key].normals, normals, decimal=5)
        np.testing.assert_array_almost_equal(fiducials.points, result["fiducials"][0], decimal=5)
        np.testing.assert_array_almost_equal(points.points, result["measurement_points"][0], decimal=5)
        np.testing.assert_array_equal(fiducials.enabled, result["fiducials"][1])
        np.testing.assert_array_equal(points.enabled, result["measurement_points"][1])
        np.testing.assert_array_almost_equal(vectors, result["measurement_vectors"], decimal=5)
        np.testing.assert_array_almost_equal(result["alignment"], np.identity(4), decimal=5)
        setting = result["settings"]
        self.assertEqual(setting["num"], 1)
        self.assertEqual(setting["str"], "string")
        self.assertEqual(tuple(setting["colour"]), (1, 1, 1, 1))

        self.assertEqual(instrument.positioning_stack.name, instrument2.positioning_stack.name)
        np.testing.assert_array_almost_equal(instrument.positioning_stack.configuration,
                                             instrument2.positioning_stack.configuration,
                                             decimal=5)
        for link1, link2 in zip(instrument.positioning_stack.links, instrument2.positioning_stack.links):
            self.assertEqual(link1.ignore_limits, link2.ignore_limits)
            self.assertEqual(link1.locked, link2.locked)
        for aux1, aux2 in zip(instrument.positioning_stack.auxiliary, instrument2.positioning_stack.auxiliary):
            np.testing.assert_array_almost_equal(aux1.base, aux2.base, decimal=5)

        np.testing.assert_array_almost_equal(instrument.jaws.aperture, instrument2.jaws.aperture, decimal=5)
        np.testing.assert_array_almost_equal(instrument.jaws.aperture_lower_limit,
                                             instrument2.jaws.aperture_lower_limit,
                                             decimal=5)
        np.testing.assert_array_almost_equal(instrument.jaws.aperture_upper_limit,
                                             instrument2.jaws.aperture_upper_limit,
                                             decimal=5)
        np.testing.assert_array_almost_equal(instrument.jaws.positioner.configuration,
                                             instrument2.jaws.positioner.configuration,
                                             decimal=5)
        for link1, link2 in zip(instrument.jaws.positioner.links, instrument2.jaws.positioner.links):
            self.assertEqual(link1.ignore_limits, link2.ignore_limits)
            self.assertEqual(link1.locked, link2.locked)

        detector1 = instrument.detectors["Detector"]
        detector2 = instrument2.detectors["Detector"]
        self.assertEqual(detector1.current_collimator.name, detector2.current_collimator.name)
        np.testing.assert_array_almost_equal(detector1.positioner.configuration,
                                             detector2.positioner.configuration,
                                             decimal=5)
        for link1, link2 in zip(detector1.positioner.links, detector2.positioner.links):
            self.assertEqual(link1.ignore_limits, link2.ignore_limits)
            self.assertEqual(link1.locked, link2.locked)

        data["measurement_vectors"] = np.ones((3, 3, 2))  # invalid normals
        writer.write_project_hdf(data, filename)
        self.assertRaises(ValueError, reader.read_project_hdf, filename)

        data["measurement_vectors"] = np.ones((3, 6, 2))  # more vector than detectors
        writer.write_project_hdf(data, filename)
        self.assertRaises(ValueError, reader.read_project_hdf, filename)

        data["measurement_vectors"] = np.ones((4, 3, 2))  # more vectors than points
        writer.write_project_hdf(data, filename)
        self.assertRaises(ValueError, reader.read_project_hdf, filename)

    def testReadTomoprocHdf(self):
        # Write nexus file
        data = {
            'entry/data/data/x': [0, 1],
            'entry/data/data/y': [3, 4],
            'entry/data/data/z': [6, 7],
            'entry/data/data/data': np.ones((2, 2, 2)),
            'entry/data/definition': b'NXtomoproc',
            'entry/definition': b'TOFRAW'
        }
        filename = os.path.join(self.test_dir, "test.h5")
        h = h5py.File(str(filename), 'w')
        for key, value in data.items():
            h.create_dataset(str(key), data=value)
        h['entry'].attrs['NX_class'] = u'NXentry'
        h.close()

        # Check data read correctly
        read_data = reader.read_tomoproc_hdf(filename)
        np.testing.assert_array_almost_equal(read_data.data, np.ones((2, 2, 2)), decimal=5)
        np.testing.assert_array_almost_equal(read_data.x, [0, 1], decimal=5)
        np.testing.assert_array_almost_equal(read_data.y, [3, 4], decimal=5)
        np.testing.assert_array_almost_equal(read_data.z, [6, 7], decimal=5)

        # Check that error is thrown when arrays don't match
        with h5py.File(filename, 'r+') as h:
            del h['entry/data/data/x']  # Needed as you can't change in place to a different size in h5py
            h['entry/data/data/x'] = [0, 1, 2]
        self.assertRaises(AttributeError, reader.read_tomoproc_hdf, filename)

        # Check that error is thrown when no NXtomoproc exists
        with h5py.File(filename, 'r+') as h:
            h['entry'].attrs['NX_class'] = u'noNXentry'
        self.assertRaises(AttributeError, reader.read_tomoproc_hdf, filename)

    def testvoxelToPitch(self):
        axis = reader.voxel_size_to_array(size=1, number_of_voxels=3)
        np.testing.assert_array_almost_equal(axis, [-1, 0, 1], decimal=5)

    @mock.patch('sscanss.core.io.reader.read_single_tiff', return_value=np.ones((100, 100)))
    @mock.patch('sscanss.core.io.reader.psutil.virtual_memory')
    def testTiffSizeVsMemory(self, mock_psutil, mock_single_tiff):
        mock_psutil.return_value.available = 1e5  # Don't want to have system specifics in test so assign ~0.1Gb memory
        filepath = self.test_dir
        true_state = reader.check_tiff_file_size_vs_memory(filepath, instances=1)
        self.assertTrue(true_state)
        false_state = reader.check_tiff_file_size_vs_memory(filepath, instances=100)
        self.assertFalse(false_state)

    @mock.patch('sscanss.core.io.reader.tiff.imread', return_value=np.ones((2, 2)))
    def testReadTiff(self, mock_tiff_imread):
        image = reader.read_single_tiff('dummy')
        np.testing.assert_array_almost_equal(image, np.ones((2, 2)))

    def testFileSortKey(self):
        list_of_strings = [f'test/00\testing{str(i)}' for i in range(4)]
        sorted_list = sorted(list_of_strings, key=reader.filename_sorting_key)
        self.assertEqual(sorted_list[0], 'test/00\testing0')
        self.assertEqual(sorted_list[3], 'test/00\testing3')

    @mock.patch('sscanss.core.io.reader.os.listdir', return_value=["test_file.png", "test_file.tiff", "test_file.tif"])
    def testFileWalker(self, mock_os_listdir):
        filepath = self.test_dir
        correct_name1 = os.path.join(filepath, "test_file.tiff")
        correct_name2 = os.path.join(filepath, "test_file.tif")
        wrong_name1 = os.path.join(filepath, "test_file.png")
        wrong_name2 = os.path.join(filepath, "test_file2.tiff")
        self.assertNotIn(wrong_name1, reader.file_walker(filepath))
        self.assertIn(wrong_name1, reader.file_walker(filepath, extension=(".tiff", ".tif", ".png")))
        self.assertNotIn(wrong_name2, reader.file_walker(filepath))
        self.assertIn(correct_name1, reader.file_walker(filepath))
        self.assertIn(correct_name2, reader.file_walker(filepath))

    @mock.patch('sscanss.core.io.reader.read_single_tiff', return_value=np.ones((2, 2)))
    def testDataFromTiffs(self, mock_read_single_tiff):
        with mock.patch('sscanss.core.io.reader.check_tiff_file_size_vs_memory', return_value=True):
            with mock.patch('sscanss.core.io.reader.file_walker',
                            return_value=["test_file1.tiff", "test_file2.tiff", "test_file3.tiff", "test_file4.tiff"]):
                # Test .tiff and non symmetric dataset
                volume_data = reader.create_data_from_tiffs("dummy/drive", 1, 1, 1)
                np.testing.assert_array_almost_equal(volume_data.x, [-0.5, 0.5], decimal=5)
                np.testing.assert_array_almost_equal(volume_data.z, [-1.5, -0.5, 0.5, 1.5], decimal=5)
                np.testing.assert_array_almost_equal(volume_data.data, np.ones((2, 2, 4)), decimal=5)

            with mock.patch('sscanss.core.io.reader.file_walker', return_value=["test_file1.tif", "test_file2.tif"]):
                # Test .tif and different voxel size
                volume_data = reader.create_data_from_tiffs("dummy/drive", 2, 2, 2)
                np.testing.assert_array_almost_equal(volume_data.x, [-1, 1], decimal=5)
                np.testing.assert_array_almost_equal(volume_data.z, [-1, 1], decimal=5)
                np.testing.assert_array_almost_equal(volume_data.data, np.ones((2, 2, 2)), decimal=5)

            with mock.patch('sscanss.core.io.reader.file_walker', return_value=[]):
                # Test empty folder
                with self.assertRaises(ValueError) as context:
                    _ = reader.create_data_from_tiffs("dummy/drive", 1, 1, 1)
                    self.assertTrue('There are no valid ".tiff" files in this folder' in str(context.exception))

            with mock.patch('sscanss.core.io.reader.check_tiff_file_size_vs_memory', return_value=False):
                # Test for files which are too large to load
                with mock.patch('sscanss.core.io.reader.file_walker', return_value=["test_file1.tif",
                                                                                    "test_file2.tif"]):
                    with self.assertRaises(MemoryError) as context:
                        _ = reader.create_data_from_tiffs("dummy/drive", 1, 1, 1)
                        self.assertTrue(
                            'The files are larger than the available memory on your machine' in str(context.exception))

    def testReadObj(self):
        # Write Obj file
        obj = ("# Demo\n"
               "v 0.5 0.5 0.0\n"
               "v -0.5 0.0 0.0\n"
               "v 0.0 0.0 0.0\n"
               "\n"
               "usemtl material_0\n"
               "f 1//1 2//2 3//3\n"
               "\n"
               "# End of file")

        filename = self.writeTestFile("test.obj", obj)

        vertices = np.array([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        mesh = reader.read_3d_model(filename)
        np.testing.assert_array_almost_equal(mesh.vertices[mesh.indices], vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh.normals[mesh.indices], normals, decimal=5)

    def testReadAsciiStl(self):
        # Write STL file
        stl = ("solid STL generated for demo\n"
               "facet normal 0.0 0.0 1.0\n"
               "  outer loop\n"
               "    vertex  0.5 0.5 0.0\n"
               "    vertex  -0.5 0.0 0.0\n"
               "    vertex  0.0 0.0 0.0\n"
               "  endloop\n"
               "endfacet\n"
               "endsolid demo\n")

        filename = self.writeTestFile("test.stl", stl)
        with open(filename, "w") as stl_file:
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
        full_path = os.path.join(self.test_dir, "test.stl")
        writer.write_binary_stl(full_path, mesh_to_write)

        mesh_read_from_file = reader.read_3d_model(full_path)
        np.testing.assert_array_almost_equal(mesh_to_write.vertices, mesh_read_from_file.vertices, decimal=5)
        np.testing.assert_array_almost_equal(mesh_to_write.normals, mesh_read_from_file.normals, decimal=5)
        np.testing.assert_array_equal(mesh_to_write.indices, mesh_read_from_file.indices)

    def testReadCsv(self):
        csvs = [
            "1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n",
            "1.0\t 2.0,3.0\n4.0, 5.0\t 6.0\n7.0, 8.0, 9.0\n",
            "1.0\t 2.0\t 3.0\n4.0\t 5.0\t 6.0\n7.0\t 8.0\t 9.0\n\n",
        ]

        for csv in csvs:
            filename = self.writeTestFile("test.csv", csv)

            data = reader.read_csv(filename)
            expected = [["1.0", "2.0", "3.0"], ["4.0", "5.0", "6.0"], ["7.0", "8.0", "9.0"]]

            np.testing.assert_array_equal(data, expected)

        filename = self.writeTestFile("test.csv", "")
        self.assertRaises(ValueError, reader.read_csv, filename)

    def testReadPoints(self):
        csv = "1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n"
        filename = self.writeTestFile("test.csv", csv)
        data = reader.read_points(filename)
        expected = ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [True, True, True])
        np.testing.assert_array_equal(data[1], expected[1])
        np.testing.assert_array_almost_equal(data[0], expected[0], decimal=5)

        csv = "1.0, 2.0, 3.0, false\n4.0, 5.0, 6.0, True\n7.0, 8.0, 9.0\n"
        filename = self.writeTestFile("test.csv", csv)
        data = reader.read_points(filename)
        expected = ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [False, True, True])
        np.testing.assert_array_equal(data[1], expected[1])
        np.testing.assert_array_almost_equal(data[0], expected[0], decimal=5)

        csv = "1.0, 3.9, 2.0, 3.0, false\n4.0, 5.0, 6.0, True\n7.0, 8.0, 9.0\n"  # first point has 4 values
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_points, filename)

        points = np.rec.array(
            [([11.0, 12.0, 13.0], True), ([14.0, 15.0, 16.0], False), ([17.0, 18.0, 19.0], True)],
            dtype=[("points", "f4", 3), ("enabled", "?")],
        )
        filename = os.path.join(self.test_dir, "test.csv")
        writer.write_points(filename, points)
        data, state = reader.read_points(filename)
        np.testing.assert_array_equal(state, points.enabled)
        np.testing.assert_array_almost_equal(data, points.points, decimal=5)

        csv = "nan, 2.0, 3.0, false\n4.0, 5.0, 6.0, True\n7.0, 8.0, 9.0\n"  # first point has NAN
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_points, filename)

    def testReadAngles(self):
        csv = "xyz\n1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n"
        filename = self.writeTestFile("test.csv", csv)
        data = reader.read_angles(filename)
        expected = ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "xyz")
        np.testing.assert_array_equal(data[1], expected[1])
        np.testing.assert_array_almost_equal(data[0], expected[0], decimal=5)

        csv = "1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n"  # missing order
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_angles, filename)

        csv = "xyz\n1.0, 2.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n"  # missing order
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_angles, filename)

        csv = "zyx\nnan, 2.0, 3.0\n4.0, 5.0, 6.0\n7.0, 8.0, 9.0\n"  # first point has NAN
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_angles, filename)

    def testReadVectors(self):
        # measurement vector column size must be a multiple of 3
        csv = "1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_vectors, filename)

        # NAN in data
        csv = "1.0,2.0,3.0,4.0,nan,6.0\n,1.0,2.0,3.0,4.0,5.0,6.0\n1.0,2.0,3.0,4.0,5.0,6.0\n\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_vectors, filename)

        # second and third row missing data
        csv = "1.0, 2.0, 3.0,4.0, 5.0, 6.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_vectors, filename)

        csv = "1.0,2.0,3.0,4.0,5.0,6.0\n,1.0,2.0,3.0,4.0,5.0,6.0\n1.0,2.0,3.0,4.0,5.0,6.0\n\n"
        filename = self.writeTestFile("test.csv", csv)
        data = reader.read_vectors(filename)
        expected = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

        csv = "1.0,2.0,3.0\n,1.0,2.0,3.0\n1.0,2.0,3.0\n"
        filename = self.writeTestFile("test.csv", csv)
        data = reader.read_vectors(filename)
        expected = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

    def testReadRobotWorldCalibrationFile(self):
        csv = "1,0,0,0,a,0\n1,0,0,0,50,prismatic,0"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_robot_world_calibration_file, filename)
        csv = "1,0,0,0\n1,0,0,0"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_robot_world_calibration_file, filename)
        csv = "1,0,0,0\n1,0,0,0,5"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_robot_world_calibration_file, filename)
        csv = "1,6,69.9,52.535,Nan,0,0,0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_robot_world_calibration_file, filename)
        csv = ("1,6,69.9,52.535,-583.339,0,0,0\n"
               "2,4,12.972,62.343,-423.562,90,-90,50\n"
               "3,1,42.946,74.268,-329.012,-90,90,-50")
        filename = self.writeTestFile("test.csv", csv)
        data = reader.read_robot_world_calibration_file(filename)
        np.testing.assert_array_equal(data[0], [0, 1, 2])
        np.testing.assert_array_almost_equal(data[1], [5, 3, 0])
        np.testing.assert_array_almost_equal(
            data[2], [[69.9, 52.535, -583.339], [12.972, 62.343, -423.562], [42.946, 74.268, -329.012]], decimal=5)
        np.testing.assert_array_almost_equal(data[3], [[0, 0, 0], [90, -90, 50], [-90, 90, -50]], decimal=5)

    def testReadKinematicCalibrationFile(self):
        csv = "1,0,0,0,a,0\n1,0,0,0,50,prismatic,0"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_kinematic_calibration_file, filename)
        csv = "1,0,0,0,a,prismatic,0\n1,0,0,0,50,prismatic,0"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_kinematic_calibration_file, filename)
        csv = "1,0,0,0,0,prismatic,0\n1,0,0,0,50,prismatic,0"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_kinematic_calibration_file, filename)
        csv = "1,0,0,0,0,prismatic,0\n1,0,0,0,50,prismatic,0\n1,0,0,0,100,prismatis,0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_kinematic_calibration_file, filename)
        csv = "1,0,0,0,0,prismatis,0\n1,0,0,0,50,prismatis,0\n1,0,0,0,100,prismatis,0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_kinematic_calibration_file, filename)
        csv = "1,0,0,0,0,prismatic,10\n1,0,0,0,50,prismatic,0\n1,0,0,0,100,prismatic,0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_kinematic_calibration_file, filename)
        csv = ("1,0,0,0,0,prismatic,0\n1,0,0,0,50,prismatic,0\n1,0,0,0,100,prismatic,0\n"
               "2,1.1,1.1,1.1,1,revolute,1\n2,1.1,1.1,1.1,51,revolute,1\n2,1.1,1.1,1.1,101,revolute,1")
        filename = self.writeTestFile("test.csv", csv)
        points, types, offsets, homes = reader.read_kinematic_calibration_file(filename)

        self.assertListEqual(types, [Link.Type.Prismatic, Link.Type.Revolute])
        np.testing.assert_array_almost_equal(homes, [0, 1], decimal=5)
        np.testing.assert_array_almost_equal(points[0], np.zeros((3, 3)), decimal=5)
        np.testing.assert_array_almost_equal(points[1], np.ones((3, 3)) * 1.1, decimal=5)
        np.testing.assert_array_almost_equal(offsets[0], [0, 50, 100], decimal=5)
        np.testing.assert_array_almost_equal(offsets[1], [1, 51, 101], decimal=5)

    def testReadTransMatrix(self):
        csv = "1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n"
        filename = self.writeTestFile("test.csv", csv)
        data = reader.read_trans_matrix(filename)
        expected = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

        csv = "1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n"  # missing last row
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_trans_matrix, filename)

        csv = "1.0, 2.0, 3.0,4.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,inf\n"  # INF in data
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_trans_matrix, filename)

        csv = "1.0, 2.0, 3.0\n, 1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n1.0, 2.0, 3.0,4.0\n"  # incorrect col size
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_trans_matrix, filename)

    def testReadFpos(self):
        csv = ("1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n, 2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n"
               "3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n")
        filename = self.writeTestFile("test.csv", csv)
        index, points, pose = reader.read_fpos(filename)
        expected = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ])
        np.testing.assert_equal(index, [0, 1, 2, 3])
        np.testing.assert_array_almost_equal(points, expected[:, 0:3], decimal=5)
        np.testing.assert_array_almost_equal(pose, expected[:, 3:], decimal=5)

        csv = "9, 1.0, 2.0, 3.0\n, 1, 1.0, 2.0, 3.0\n, 3, 1.0, 2.0, 3.0\n, 6, 1.0, 2.0, 3.0\n"
        filename = self.writeTestFile("test.csv", csv)
        index, points, pose = reader.read_fpos(filename)
        np.testing.assert_equal(index, [8, 0, 2, 5])
        np.testing.assert_array_almost_equal(points, expected[:, 0:3], decimal=5)
        self.assertEqual(pose.size, 0)

        csv = "1.0, 2.0, 3.0\n, 1.0, 2.0, 3.0\n1.0, 2.0, 3.0\n"  # missing index column
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_fpos, filename)

        csv = ("9, 1.0, 2.0, 3.0, 5.0\n, 1, 1.0, 2.0, 3.0\n, "
               "3, 1.0, 2.0, 3.0\n, 6, 1.0, 2.0, 3.0\n")  # incorrect col size
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_fpos, filename)

        csv = "1, nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n, 2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_fpos, filename)

        csv = "1, 1.0, 2.0, 3.0, 4.0, 5.0, -inf, 7.0\n, 2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0\n"
        filename = self.writeTestFile("test.csv", csv)
        self.assertRaises(ValueError, reader.read_fpos, filename)

    def testValidateVectorLength(self):
        vectors = np.ones((3, 3, 2))
        self.assertFalse(reader.validate_vector_length(vectors))

        vectors = np.zeros((3, 3, 2))
        self.assertTrue(reader.validate_vector_length(vectors))

        vectors[:, :, 0] = [
            [0.0000076, 1.0000000, 0.0000480],
            [0.0401899, 0.9659270, 0.2556752],
            [0.1506346, 0.2589932, 0.9540607],
        ]

        vectors[:, :, 1] = [
            [0.1553215, -0.0000486, 0.9878640],
            [0.1499936, -0.2588147, 0.9542100],
            [0.0403915, -0.9658791, 0.2558241],
        ]
        self.assertTrue(reader.validate_vector_length(vectors))

        vectors = np.zeros((3, 6, 1))
        vectors[:, :3, 0] = [
            [0.0000076, 1.0000000, 0.0000480],
            [0.00000000, 0.0000000, 0.0000000],
            [0.1506346, 0.2589932, 0.9540607],
        ]

        vectors[:, 3:, 0] = [
            [0.1553215, -0.0000486, 0.9878640],
            [0.1499936, -0.2588147, 0.9542100],
            [0.0403915, -0.9658791, 0.2558241],
        ]
        self.assertTrue(reader.validate_vector_length(vectors))

        vectors[0, 0, 0] = 10
        self.assertFalse(reader.validate_vector_length(vectors))

    def writeTestFile(self, filename, text):
        full_path = os.path.join(self.test_dir, filename)
        with open(full_path, "w") as text_file:
            text_file.write(text)
        return full_path


if __name__ == "__main__":
    unittest.main()
