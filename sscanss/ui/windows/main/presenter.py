import logging
import os
from enum import Enum, unique
from .model import MainWindowModel
from sscanss.ui.commands import (ToggleRenderType, InsertPrimitive, DeleteSample, MergeSample,
                                 InsertSampleFromFile, RotateSample, TranslateSample,
                                 ChangeMainSample)
from sscanss.core.util import TransformType

@unique
class MessageReplyType(Enum):
    Save = 1
    Discard = 2
    Cancel = 3


class MainWindowPresenter:
    def __init__(self, view):
        self.view = view
        self.model = MainWindowModel()
        self.worker = None

        self.recent_list_size = 10  # Maximum size of the recent project list

    def isProjectCreated(self):
        return True if self.model.project_data else False

    def createProject(self, name, instrument):
        """
        This function creates the stub data for the project

        :param name: The name of the project
        :type name: str
        :param instrument: The name of the instrument used for the project
        :type instrument: str
        """
        self.model.createProjectData(name, instrument)
        self.view.showProjectName(name)

    def saveProject(self, save_as=False):
        """
        This function saves a project to a file. A file dialog will be opened for the first save
        after which the function will save to the same location. if save_as id True a dialog is
        opened every time

        :param save_as: A flag denoting whether to use file dialog or not
        :type save_as: bool
        """
        if not self.isProjectCreated():
            return

        # Avoids saving when there are no changes
        if not self.model.unsaved and self.model.save_path and not save_as:
            return

        filename = self.model.save_path
        if save_as or not filename:
            filename = self.view.showSaveDialog('hdf5 File (*.h5)',
                                                current_dir=filename)
            if not filename:
                return

        try:
            self.model.saveProjectData(filename)
            self.updateRecentProjects(filename)
        except OSError:
            msg = 'A error occurred while attempting to save this project ({})'.format(filename)
            logging.exception(msg)
            self.view.showErrorMessage(msg)

    def openProject(self, filename=''):
        """
        This function loads a project with the given filename. if filename is empty,
        a file dialog will be opened.

        :param filename: full path of file
        :type filename: str
        """
        if not self.confirmSave():
            return

        if not filename:
            filename = self.view.showOpenDialog('hdf5 File (*.h5)',
                                                title='Open Project',
                                                current_dir=self.model.save_path)
            if not filename:
                return

        try:
            self.model.loadProjectData(filename)
            self.updateRecentProjects(filename)
            self.view.showProjectName(self.model.project_data['name'])
        except (KeyError, AttributeError):
            msg = '{} could not open because it has an incorrect format.'
            msg = msg.format(os.path.basename(filename))
            logging.exception(msg)
            self.view.showErrorMessage(msg)
        except OSError:
            msg = 'An error occurred while opening this file.\nPlease check that ' \
                  'the file exist and also that this user has access privileges for this file.\n({})'

            msg = msg.format(filename)
            logging.exception(msg)
            self.view.showErrorMessage(msg)

    def confirmSave(self):
        """
        Checks if the project is saved and asks the user to save if necessary

        :return: True if the project is saved or user chose to discard changes
        :rtype: bool
        """
        if not self.model.unsaved:
            return True

        reply = self.view.showSaveDiscardMessage(self.model.project_data['name'])

        if reply == MessageReplyType.Save:
            if self.model.save_path:
                self.saveProject()
                return True
            else:
                self.saveProject(save_as=True)
                return False if self.model.unsaved else True

        elif reply == MessageReplyType.Discard:
            return True

        else:
            return False

    def updateRecentProjects(self, filename):
        """
        This function adds a filename entry to the front of the recent projects list
        if it does not exist in the list. if the entry already exist, it is moved to the
        front but not duplicated.

        :param filename: project path to add to recents lists
        :type filename: str
        """
        projects = self.view.recent_projects
        projects.insert(0, filename)
        projects = list(dict.fromkeys(projects))
        if len(projects) <= self.recent_list_size:
            self.view.recent_projects = projects
        else:
            self.view.recent_projects = projects[:self.recent_list_size]

    def importSample(self, filename=''):
        if not filename:
            filename = self.view.showOpenDialog('3D Files (*.stl *.obj)',
                                                title='Import Sample Model',
                                                current_dir=self.model.save_path)

            if not filename:
                return

        insert_command = InsertSampleFromFile(filename, self, self.confirmCombineSample())
        self.view.undo_stack.push(insert_command)

    def toggleRenderType(self, render_type):
        toggle_command = ToggleRenderType(render_type, self.view)
        self.view.undo_stack.push(toggle_command)

    def addPrimitive(self, primitive, args):
        insert_command = InsertPrimitive(primitive, args, self, combine=self.confirmCombineSample())
        self.view.undo_stack.push(insert_command)
        if len(self.model.sample) > 1:
            self.view.docks.showSampleManager()

    def transformSample(self, angles_or_offset, sample_key, transform_type):
        if transform_type == TransformType.Rotate:
            transform_command = RotateSample(angles_or_offset, sample_key, self)
        else:
            transform_command = TranslateSample(angles_or_offset, sample_key, self)

        self.view.undo_stack.push(transform_command)

    def deleteSample(self, sample_key):
        delete_command = DeleteSample(sample_key, self)
        self.view.undo_stack.push(delete_command)

    def mergeSample(self, sample_key):
        merge_command = MergeSample(sample_key, self)
        self.view.undo_stack.push(merge_command)

    def changeMainSample(self, sample_key):
        change_main_command = ChangeMainSample(sample_key, self)
        self.view.undo_stack.push(change_main_command)

    def confirmCombineSample(self):
        if self.model.sample:
            question = 'A sample model has already been added to the project.\n\n' \
                       'Do you want replace the model or combine them?'
            choice = self.view.showSelectChoiceMessage(question, ['combine', 'replace'], default_choice=1)

            if choice == 'combine':
                return True

        return False

    def importFiducials(self, filename=''):
        if not filename:
            filename = self.view.showOpenDialog('Fiducial File(*.fiducial)',
                                                title='Import Fiducial Points',
                                                current_dir=self.model.save_path)

            if not filename:
                return

        try:
            self.model.loadFiducials(filename)

        except:
            pass

        self.view.docks.showPointManager()
    def addFiducial(self, point):
        self.model.addPointsToProject([(point, True)])
