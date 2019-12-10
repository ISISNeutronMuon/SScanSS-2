import unittest
import unittest.mock as mock
from sscanss.ui.window.model import MainWindowModel
from sscanss.core.instrument import Instrument


class TestMainWindowModel(unittest.TestCase):
    def setUp(self):
        self.model = MainWindowModel()
        self.instrument = mock.create_autospec(Instrument)
        self.instrument.detectors = []

        patcher = mock.patch('sscanss.ui.window.model.read_instrument_description_file', autospec=True)
        self.addCleanup(patcher.stop)
        mocked_function = patcher.start()
        mocked_function.return_value = self.instrument

    def testCreateProjectData(self):
        self.assertIsNone(self.model.project_data)
        self.model.createProjectData('Test', 'ENGIN-X')
        self.assertIsNotNone(self.model.project_data)

    def testAddAndRemoveMesh(self):
        self.model.createProjectData('Test', 'ENGIN-X')

        self.model.addMeshToProject('demo', None)
        self.model.addMeshToProject('demo', None)  # should be added as 'demo 1'
        self.assertEqual(len(self.model.sample), 2)
        self.model.removeMeshFromProject('demo')
        self.assertEqual(len(self.model.sample), 1)
