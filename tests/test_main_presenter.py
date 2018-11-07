import unittest
import unittest.mock as mock
from sscanss.ui.windows.main.presenter import MainWindowPresenter, MessageReplyType
import sscanss.ui.windows.main.view as view


class TestMainWindowPresenter(unittest.TestCase):
    @mock.patch('sscanss.ui.windows.main.presenter.MainWindowModel', autospec=True)
    def setUp(self, model_mock):
        self.view_mock = mock.create_autospec(view.MainWindow)
        self.model_mock = model_mock
        self.presenter = MainWindowPresenter(self.view_mock)

        self.test_project_data = {'name': 'Test Project', 'instrument': 'IMAT'}
        self.test_filename_1 = 'C:/temp/file.h5'
        self.test_filename_2 = 'C:/temp/file_2.h5'

    def testIsProjectCreated(self):
        self.model_mock.return_value.project_data = None
        self.assertFalse(self.presenter.isProjectCreated())

        self.model_mock.return_value.project_data = {}
        self.assertFalse(self.presenter.isProjectCreated())

        self.model_mock.return_value.project_data = self.test_project_data
        self.assertTrue(self.presenter.isProjectCreated())

    @unittest.skip('QThread used in function')
    def testCreateProject(self):
        name = self.test_project_data['name']
        instrument = self.test_project_data['instrument']
        self.presenter.createProject(name, instrument)
        self.model_mock.return_value.createProjectData.assert_called_with(name, instrument)

    def testSaveProjectWithDefaults(self):
        self.view_mock.recent_projects = []

        # When there is no project data save will not be called
        self.model_mock.return_value.project_data = None
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_not_called()

        # When there are no unsaved changes save will not be called
        self.model_mock.reset_mock()
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.unsaved = False
        self.model_mock.return_value.save_path = self.test_filename_1
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_not_called()

        # When there are unsaved changes save will be called
        self.model_mock.reset_mock()
        self.model_mock.return_value.unsaved = True
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_called_with(self.test_filename_1)
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_1])

        # When save_path is blank, filename is acquired from dialog
        self.model_mock.reset_mock()
        self.model_mock.return_value.unsaved = False
        self.model_mock.return_value.save_path = ''
        self.view_mock.showSaveDialog.return_value = self.test_filename_2
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_called_with(self.test_filename_2)
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_2, self.test_filename_1])

        # if dialog return empty filename (user cancels save), save will not be called
        self.model_mock.reset_mock()
        self.view_mock.showSaveDialog.return_value = ''
        self.presenter.saveProject()
        self.model_mock.return_value.saveProjectData.assert_not_called()

    def testSaveProjectWithSaveAs(self):
        self.view_mock.recent_projects = []
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.save_path = self.test_filename_1

        # Save_as opens dialog even though there are no unsaved changes
        self.model_mock.return_value.unsaved = True
        self.view_mock.showSaveDialog.return_value = self.test_filename_2
        self.presenter.saveProject(save_as=True)
        self.model_mock.return_value.saveProjectData.assert_called_with(self.test_filename_2)
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_2])

    def testOpenProjectWithDefaults(self):
        self.view_mock.recent_projects = []
        self.model_mock.return_value.unsaved = False
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.save_path = ''

        # When dialog return a non-empty filename, file should be loaded, recent list updated
        # and view name changed
        self.view_mock.showOpenDialog.return_value = self.test_filename_2
        self.presenter.openProject()
        self.model_mock.return_value.loadProjectData.assert_called_with(self.test_filename_2)
        self.view_mock.showProjectName.assert_called_with(self.test_project_data['name'])
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_2])

        # When dialog return a empty filename, file should not be loaded
        self.model_mock.reset_mock()
        self.view_mock.showOpenDialog.return_value = ''
        self.presenter.openProject()
        self.model_mock.return_value.loadProjectData.assert_not_called()

    def testOpenProjectWithFilename(self):
        self.view_mock.recent_projects = []
        self.model_mock.return_value.unsaved = False
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.save_path = ''

        # When non-empty filename is provided, dialog should not be called, file should be loaded,
        # recent list updated, and view name changed
        self.presenter.openProject(self.test_filename_2)
        self.view_mock.showOpenDialog.assert_not_called()
        self.model_mock.return_value.loadProjectData.assert_called_with(self.test_filename_2)
        self.view_mock.showProjectName.assert_called_with(self.test_project_data['name'])
        self.assertEqual(self.view_mock.recent_projects, [self.test_filename_2])

    def testUpdateRecentProjects(self):
        self.presenter.recent_list_size = 10

        self.view_mock.recent_projects = []
        self.presenter.updateRecentProjects('Hello World')
        self.assertEqual(self.view_mock.recent_projects, ['Hello World'])

        # Check new values are always placed in front
        self.view_mock.recent_projects = [1, 2, 3]
        self.presenter.updateRecentProjects(4)
        self.assertEqual(self.view_mock.recent_projects, [4, 1, 2, 3])

        # When max size is exceeded the last entry is removed
        self.view_mock.recent_projects = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.presenter.updateRecentProjects(10)
        self.assertEqual(self.view_mock.recent_projects, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        # When a value already exist in the list, it is push to the front
        self.view_mock.recent_projects = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.presenter.updateRecentProjects(3)
        self.assertEqual(self.view_mock.recent_projects, [3, 9, 8, 7, 6, 5, 4, 2, 1, 0])

    def testConfirmSave(self):
        # confirmSave should return True when there are no unsaved changes
        # and the save-discard message should not be called
        self.model_mock.return_value.unsaved = False
        self.assertTrue(self.presenter.confirmSave())
        self.view_mock.showSaveDiscardMessage.assert_not_called()

        # confirmSave should return False when user selects cancel on
        # the save-discard message box
        self.model_mock.return_value.project_data = self.test_project_data
        self.model_mock.return_value.unsaved = True
        self.view_mock.showSaveDiscardMessage.return_value = MessageReplyType.Cancel
        self.assertFalse(self.presenter.confirmSave())

        # confirmSave should return True when user selects discard on
        # the save-discard message box
        self.view_mock.showSaveDiscardMessage.return_value = MessageReplyType.Discard
        self.assertTrue(self.presenter.confirmSave())

        # confirmSave should call save (if save path exist) then return True
        # when user selects save on the save-discard message box
        self.model_mock.return_value.save_path = self.test_filename_1
        self.presenter.saveProject = mock.create_autospec(self.presenter.saveProject)
        self.view_mock.showSaveDiscardMessage.return_value = MessageReplyType.Save
        self.assertTrue(self.presenter.confirmSave())
        self.presenter.saveProject.assert_called_with()

        # confirmSave should call save_as (if save_path does not exist)
        self.model_mock.return_value.save_path = ''
        self.presenter.confirmSave()
        self.presenter.saveProject.assert_called_with(save_as=True)


if __name__ == '__main__':
    unittest.main()
