import logging
import os
from enum import Enum, unique
from .model import MainWindowModel
from sscanss.ui.commands import (ToggleRenderMode, InsertPrimitive, DeleteSample, MergeSample,
                                 InsertSampleFromFile, RotateSample, TranslateSample, TransformSample,
                                 ChangeMainSample, InsertPointsFromFile, InsertPoints, DeletePoints,
                                 MovePoints, EditPoints)
from sscanss.core.io import read_trans_matrix
from sscanss.core.util import TransformType


@unique
class MessageSeverity(Enum):
    Information = 1
    Warning = 2
    Critical = 3


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
            self.view.showMessage(msg)

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
            self.view.showMessage(msg)
        except OSError:
            msg = 'An error occurred while opening this file.\nPlease check that ' \
                  'the file exist and also that this user has access privileges for this file.\n({})'

            msg = msg.format(filename)
            logging.exception(msg)
            self.view.showMessage(msg)

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

    def importSample(self):
        filename = self.view.showOpenDialog('3D Files (*.stl *.obj)',
                                            title='Import Sample Model',
                                            current_dir=self.model.save_path)

        if not filename:
            return

        insert_command = InsertSampleFromFile(filename, self, self.confirmCombineSample())
        self.view.undo_stack.push(insert_command)

    def toggleRenderMode(self, render_mode):
        toggle_command = ToggleRenderMode(render_mode, self.view)
        self.view.undo_stack.push(toggle_command)

    def addPrimitive(self, primitive, args):
        insert_command = InsertPrimitive(primitive, args, self, combine=self.confirmCombineSample())
        self.view.undo_stack.push(insert_command)
        if len(self.model.sample) > 1:
            self.view.docks.showSampleManager()

    def transformSample(self, angles_or_offset, sample_key, transform_type):
        if transform_type == TransformType.Rotate:
            transform_command = RotateSample(angles_or_offset, sample_key, self)
        elif transform_type == TransformType.Translate:
            transform_command = TranslateSample(angles_or_offset, sample_key, self)
        else:
            transform_command = TransformSample(angles_or_offset, sample_key, self)

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

    def importPoints(self, point_type):
        if not self.model.sample:
            self.view.showMessage('A sample model should be added before {} points'.format(point_type.value.lower()),
                                  MessageSeverity.Information)
            return

        title = 'Import {} Points'.format(point_type.value)
        file_filter = '{} File(*.{})'.format(point_type.value, point_type.value.lower())

        filename = self.view.showOpenDialog(file_filter,
                                            title=title,
                                            current_dir=self.model.save_path)

        if not filename:
            return

        insert_command = InsertPointsFromFile(filename, point_type, self)
        self.view.undo_stack.push(insert_command)

    def addPoint(self, point, point_type):
        if not self.model.sample:
            self.view.showMessage('A sample model should be added before {} points'.format(point_type.value.lower()),
                                  MessageSeverity.Information)
            return

        points = [(point, True)]
        insert_command = InsertPoints(points, point_type, self)
        self.view.undo_stack.push(insert_command)
        self.view.docks.showPointManager(point_type)

    def deletePoints(self, indices, point_type):
        delete_command = DeletePoints(indices, point_type, self)
        self.view.undo_stack.push(delete_command)

    def movePoints(self, move_from, move_to, point_type):
        move_command = MovePoints(move_from, move_to, point_type, self)
        self.view.undo_stack.push(move_command)

    def editPoints(self, row, value, point_type):
        edit_command = EditPoints(row, value, point_type, self)
        self.view.undo_stack.push(edit_command)

    def importVectors(self):
        if not self.model.sample:
            self.view.showMessage('Sample model and measurement points should be added before vectors',
                                  MessageSeverity.Information)
            return

        if len(self.model.measurement_points) == 0:
            self.view.showMessage('Measurement points should be added before vectors', MessageSeverity.Information)
            return

        filename = self.view.showOpenDialog('Measurement Vector File(*.vecs)',
                                            title='Import Measurement Vectors',
                                            current_dir=self.model.save_path)

        if not filename:
            return

        self.model.loadVectors(filename)

    def importTransformMatrix(self):
        filename = self.view.showOpenDialog('Transform Matrix File(*.trans)',
                                            title='Import Transformation Vectors',
                                            current_dir=self.model.save_path)

        matrix = []
        if not filename:
            return matrix

        try:
            matrix = read_trans_matrix(filename)
        except:
            pass

        return matrix