import unittest
import unittest.mock as mock
from sscanss.ui.windows.main.model import MainWindowModel
from sscanss.core.scene import Node
from sscanss.core.instrument import Instrument


class dummy():
    def model(self):
        return Node()

class TestMainWindowModel(unittest.TestCase):
    def setUp(self):
        self.model = MainWindowModel()

    @mock.patch('sscanss.ui.windows.main.model.read_instrument_description_file', autospec=True)
    def testCreateProjectData(self, mocked_function):
        i = Instrument('demo')
        i.fixed_positioner = '1'
        i.positioners = {'1': dummy()}
        i.jaws = dummy()
        i.detectors = {'South': [dummy(), dummy(), dummy()], 'North': [dummy(), dummy(), dummy()]}
        mocked_function.return_value = i

        self.assertIsNone(self.model.project_data)
        self.model.createProjectData('Test', 'ENGIN-X')
        self.assertIsNotNone(self.model.project_data)

    @mock.patch('sscanss.ui.windows.main.model.createSampleNode', autospec=True)
    def testAddAndRemoveMesh(self, mocked_function):
        mocked_function.return_value = Node()

        self.model.createProjectData('Test', 'ENGIN-X')

        self.model.addMeshToProject('demo', None)
        self.model.addMeshToProject('demo', None)  # should be added as 'demo 1'
        self.assertEqual(len(self.model.sample), 2)
        self.model.removeMeshFromProject('demo')
        self.assertEqual(len(self.model.sample), 1)
