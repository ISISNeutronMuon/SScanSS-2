import unittest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from sscanss.core.util import Primitives, TransformType
from sscanss.ui.windows.main.view import MainWindow


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])
        cls.window = MainWindow()
        cls.window.show()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def testMainView(self):
        if not QTest.qWaitForWindowActive(self.window):
            self.skipTest('Window is not ready!')

        self.assertTrue(self.window.isVisible())
        self.window.presenter.createProject('Test', 'ENGIN-X')

        self.window.docks.showTransformDialog(TransformType.Translate)
        self.window.docks.showInsertPrimitiveDialog(Primitives.Tube)
        self.window.docks.showSampleManager()

        self.window.populateRecentMenu()
        self.window.recent_projects = ['c://test.hdf']
        self.window.populateRecentMenu()

        self.window.showNewProjectDialog()
        self.assertTrue(self.window.project_dialog.isVisible())
        self.window.project_dialog.close()

        self.window.recent_projects = []

        self.window.showUndoHistory()
        self.assertTrue(self.window.undo_view.isVisible())
        self.window.close()

        self.window.showProgressDialog('Testing')
        self.assertTrue(self.window.progress_dialog.isVisible())
        self.window.progress_dialog.close()
