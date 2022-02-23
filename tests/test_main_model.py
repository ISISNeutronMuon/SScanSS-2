import os
import shutil
import tempfile
import unittest
import unittest.mock as mock
import numpy as np
from sscanss.app.window.model import MainWindowModel, IDF
from sscanss.core.geometry import Mesh
from sscanss.core.instrument import Instrument
from sscanss.core.util import PointType, POINT_DTYPE, LoadVector
from tests.helpers import TestSignal


class TestMainWindowModel(unittest.TestCase):
    def setUp(self):
        self.model = MainWindowModel()
        self.instrument = mock.create_autospec(Instrument)
        self.instrument.detectors = []

        read_inst_function = self.createPatch("sscanss.app.window.model.read_instrument_description_file")
        read_inst_function.return_value = self.instrument

        validate_inst_function = self.createPatch("sscanss.app.window.model.validate_instrument_scene_size")
        validate_inst_function.return_value = True

        vertices = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 1]])
        normals = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        indices = np.array([0, 1, 2])
        self.mesh = Mesh(vertices, indices, normals)
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def createPatch(self, module):
        patcher = mock.patch(module, autospec=True)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def testCreateProjectData(self):
        self.assertIsNone(self.model.project_data)
        self.model.createProjectData("Test", "ENGIN-X")
        self.assertIsNotNone(self.model.project_data)

    @mock.patch("sscanss.app.window.model.settings", autospec=True)
    @mock.patch("sscanss.app.window.model.read_project_hdf", autospec=True)
    @mock.patch("sscanss.app.window.model.write_project_hdf", autospec=True)
    def testLoadAndSaveProject(self, write_fn, load_fn, settings):
        settings.local = {}
        self.model.saveProjectData("demo.hdf")
        write_fn.assert_called_once_with(None, "demo.hdf")
        data = {
            "name": "demo",
            "settings": {
                "colour": "w"
            },
            "instrument_version": "1.0.0",
            "sample": {
                "sample": None
            },
            "fiducials": ([[0, 1, 2]], [False]),
            "measurement_points": ([[3, 4, 5]], [True]),
            "measurement_vectors": np.array([[0.0, 1.0, 0.0]]),
            "alignment": np.identity(4),
        }

        instrument = mock.Mock()
        instrument.name = "some_instrument"
        load_fn.return_value = (data, instrument)
        self.model.loadProjectData("demo.hdf")
        self.assertIs(self.model.instrument, instrument)
        self.assertEqual(self.model.project_data["instrument_version"], data["instrument_version"])
        self.assertDictEqual(self.model.sample, data["sample"])
        self.assertDictEqual(settings.local, data["settings"])
        np.testing.assert_array_almost_equal(self.model.fiducials.points, data["fiducials"][0], decimal=5)
        np.testing.assert_equal(self.model.fiducials.enabled, data["fiducials"][1])
        np.testing.assert_array_almost_equal(self.model.measurement_points.points,
                                             data["measurement_points"][0],
                                             decimal=5)
        np.testing.assert_equal(self.model.measurement_points.enabled, data["measurement_points"][1])
        np.testing.assert_array_almost_equal(self.model.measurement_vectors, data["measurement_vectors"], decimal=5)
        np.testing.assert_array_almost_equal(self.model.alignment, data["alignment"], decimal=5)

        self.assertFalse(self.model.checkInstrumentVersion())
        self.model.instruments = {instrument.name: IDF(instrument.name, "", "2.0.0")}
        self.assertFalse(self.model.checkInstrumentVersion())
        self.model.instruments = {instrument.name: IDF(instrument.name, "", "1.0.0")}
        self.assertTrue(self.model.checkInstrumentVersion())

    def testAddAndRemoveMesh(self):
        self.model.createProjectData("Test", "ENGIN-X")

        self.model.addMeshToProject("demo", None)
        self.model.addMeshToProject("demo", None, "stl")  # should be added as 'demo [stl]'
        self.assertEqual(len(self.model.sample), 2)
        self.model.removeMeshFromProject("demo")
        self.assertEqual(len(self.model.sample), 1)

        self.assertEqual(self.model.uniqueKey("demo"), "demo")
        self.assertEqual(self.model.uniqueKey("demo [stl]"), "demo [stl] 1")
        self.assertEqual(self.model.uniqueKey("demo", "obj"), "demo [obj]")
        self.assertEqual(self.model.uniqueKey("demo", "stl"), "demo 1 [stl]")

    def testLoadAndSaveSample(self):
        self.model.createProjectData("Test", "ENGIN-X")
        self.model.sample = {"demo": self.mesh}

        path = os.path.join(self.test_dir, "test.stl")
        self.assertFalse(os.path.isfile(path))
        self.model.saveSample(path, "demo")
        self.assertTrue(os.path.isfile(path))
        self.model.loadSample(path)
        self.assertEqual(len(self.model.sample), 2)

    def testLoadAndSavePoints(self):
        self.model.createProjectData("Test", "ENGIN-X")
        path = os.path.join(self.test_dir, "measurement.csv")
        new_points = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], True)],
                                  dtype=POINT_DTYPE)
        self.model.measurement_points = new_points
        self.assertFalse(os.path.isfile(path))
        self.model.savePoints(path, PointType.Measurement)
        self.assertTrue(os.path.isfile(path))
        self.model.loadPoints(path, PointType.Measurement)
        self.assertEqual(len(self.model.measurement_points), 6)
        np.testing.assert_array_almost_equal(
            self.model.measurement_points.points[3:, ],
            new_points.points,
            decimal=5,
        )
        np.testing.assert_equal(
            self.model.measurement_points.enabled[3:, ],
            new_points.enabled,
        )

        path = os.path.join(self.test_dir, "fiducials.csv")
        self.model.fiducials = new_points
        self.assertFalse(os.path.isfile(path))
        self.model.savePoints(path, PointType.Fiducial)
        self.assertTrue(os.path.isfile(path))
        self.model.loadPoints(path, PointType.Fiducial)
        self.assertEqual(len(self.model.fiducials), 6)
        np.testing.assert_array_almost_equal(
            self.model.fiducials.points[3:, ],
            new_points.points,
            decimal=5,
        )
        np.testing.assert_equal(
            self.model.fiducials.enabled[3:, ],
            new_points.enabled,
        )

    def testVectorAlignmentCorrection(self):
        self.model.createProjectData("Test", None)
        self.assertEqual(self.model.measurement_vectors.shape, (0, 3, 1))
        self.model.createProjectData("Test", "ENGIN-X")
        self.instrument.detectors = [None, None]
        self.model.correctVectorDetectorSize()
        self.assertEqual(self.model.measurement_vectors.shape, (0, 6, 1))

        vectors = np.random.rand(3, 6, 1)
        self.model.measurement_vectors = vectors
        self.instrument.detectors = [None]  # 2 detectors to 1
        self.model.correctVectorDetectorSize()
        self.assertEqual(self.model.measurement_vectors.shape, (3, 3, 2))
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, :, 0], vectors[:, :3, 0], decimal=5)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, :, 1], vectors[:, 3:, 0], decimal=5)

        self.instrument.detectors = [None, None]  # 1 detectors to 2
        self.model.correctVectorDetectorSize()
        self.assertEqual(self.model.measurement_vectors.shape, (3, 6, 2))
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, :3, 0], vectors[:, :3, 0], decimal=5)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, :3, 1], vectors[:, 3:, 0], decimal=5)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, 3:, :], np.zeros((3, 3, 2)), decimal=5)

    def testLoadAndSaveVectors(self):
        self.model.createProjectData("Test", "ENGIN-X")
        self.instrument.detectors = [None, None]
        path = os.path.join(self.test_dir, "vectors.txt")
        points = np.rec.array([([0.0, 0.0, 0.0], False), ([2.0, 0.0, 1.0], True), ([0.0, 1.0, 1.0], True)],
                              dtype=POINT_DTYPE)

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

        self.model.measurement_points = points
        self.model.measurement_vectors = np.zeros((3, 3, 1))
        self.model.addVectorsToProject(vectors[:, :, 0], slice(None), alignment=0)
        self.model.addVectorsToProject(vectors[:, :, 1], slice(None), alignment=1)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, :3, :], vectors, decimal=5)
        self.assertFalse(os.path.isfile(path))
        self.model.saveVectors(path)
        self.assertTrue(os.path.isfile(path))
        self.model.measurement_vectors = None
        self.assertEqual(self.model.loadVectors(path), LoadVector.Larger)
        self.assertEqual(self.model.measurement_vectors.shape, (3, 6, 2))
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, :3, :], vectors, decimal=5)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:, 3:, :], np.zeros((3, 3, 2)), decimal=5)

        vectors = np.zeros((2, 6, 1))
        vectors[:, :, 0] = [
            [0.0000076, 1.0000000, 0.0000480, 0.1553215, -0.0000486, 0.9878640],
            [0.0401899, 0.9659270, 0.2556752, 0.1499936, -0.2588147, 2.9542100],
        ]  # bad vector

        self.model.measurement_vectors = vectors
        self.model.saveVectors(path)
        self.assertRaises(ValueError, self.model.loadVectors, path)
        self.instrument.detectors = [None]
        self.assertRaises(ValueError, self.model.loadVectors, path)
        self.model.measurement_vectors[1, 5, 0] = 0.9542100
        self.model.saveVectors(path)
        self.instrument.detectors = [None, None]
        self.model.measurement_vectors = None
        self.assertEqual(self.model.loadVectors(path), LoadVector.Smaller)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[:2, :, :], vectors, decimal=5)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors[2, :, :], np.zeros((6, 1)), decimal=5)

        vectors = np.zeros((3, 3, 1))
        vectors[:, :, 0] = [
            [0.0000076, 1.0000000, 0.0000480],
            [0.1499936, -0.2588147, 0.9542100],
            [0.1506346, 0.2589932, 0.9540607],
        ]

        self.instrument.detectors = [None]
        self.model.measurement_vectors = vectors
        self.model.saveVectors(path)
        self.model.measurement_vectors = None
        self.assertEqual(self.model.loadVectors(path), LoadVector.Exact)
        np.testing.assert_array_almost_equal(self.model.measurement_vectors, vectors, decimal=5)

        vectors = np.zeros((4, 3))
        vectors[:, :] = [
            [0.0000076, 1.0000000, 0.0000480],
            [0.0401899, 0.9659270, 0.2556752],
            [0.1506346, 0.2589932, 0.9540607],
            [0.1553215, -0.0000486, 0.9878640],
        ]

        np.savetxt(path, vectors, delimiter="\t")
        self.assertEqual(self.model.loadVectors(path), LoadVector.Larger)
        new_vectors = self.model.measurement_vectors
        np.testing.assert_array_almost_equal(
            new_vectors[:3, :, 0],
            vectors[:3, ],
            decimal=5,
        )
        np.testing.assert_array_almost_equal(new_vectors[:3, :, 1],
                                             np.row_stack((vectors[3, :], np.zeros((2, 3)))),
                                             decimal=5)

        vectors[0, 0] = 10
        np.savetxt(path, vectors, delimiter="\t")
        self.assertRaises(ValueError, self.model.loadVectors, path)

    @mock.patch("sscanss.app.window.model.Simulation", autospec=True)
    def testSimulationCreation(self, _simulation_model):
        self.model.createProjectData("Test", "ENGIN-X")
        self.model.sample = {"demo": self.mesh}

        mock_fn = mock.Mock()
        self.model.simulation_created = TestSignal()
        self.model.simulation_created.connect(mock_fn)

        self.model.createSimulation(True, False, True, False)

        self.assertTrue(self.model.simulation.compute_path_length)
        self.assertFalse(self.model.simulation.render_graphics)
        self.assertTrue(self.model.simulation.check_limits)
        self.assertFalse(self.model.simulation.check_collision)
        mock_fn.assert_called_once()
