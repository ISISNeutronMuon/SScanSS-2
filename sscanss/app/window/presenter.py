import os
import logging
import numpy as np
from enum import Enum, unique
from contextlib import suppress
from .model import MainWindowModel
from sscanss.config import INSTRUMENTS_PATH, settings
from sscanss.app.commands import (InsertPrimitive, DeleteSample, MergeSample,
                                  InsertSampleFromFile, RotateSample, TranslateSample, TransformSample,
                                  ChangeMainSample, InsertPointsFromFile, InsertPoints, DeletePoints, RemoveVectors,
                                  MovePoints, EditPoints, InsertVectorsFromFile, InsertVectors, LockJoint,
                                  IgnoreJointLimits, MovePositioner, ChangePositioningStack, ChangePositionerBase,
                                  ChangeCollimator, ChangeJawAperture, RemoveVectorAlignment, InsertAlignmentMatrix)
from sscanss.core.io import read_trans_matrix, read_fpos, read_robot_world_calibration_file
from sscanss.core.util import TransformType, MessageSeverity, Worker, toggleActionInGroup, PointType
from sscanss.core.instrument import robot_world_calibration
from sscanss.core.math import matrix_from_pose, find_3d_correspondence, rigid_transform, check_rotation, VECTOR_EPS


@unique
class MessageReplyType(Enum):
    Save = 1
    Discard = 2
    Cancel = 3


class MainWindowPresenter:
    """Presenter handles communication between View and Model

    :param view: Main window
    :type view: MainWindow
    """
    def __init__(self, view):
        self.view = view
        self.model = MainWindowModel()
        if not self.model.instruments:
            self.view.showMessage(f'No instrument description file was found. Check that "{INSTRUMENTS_PATH}"'
                                  'contains an instrument description file.')
            raise FileNotFoundError("No instrument description file was found.")

        self.worker = None

        self.recent_list_size = 10  # Maximum size of the recent project list

    def notifyError(self, message, exception):
        """Logs error and notifies user of them

        :param message: message to display to user and in the log
        :type message: str
        :param exception: exception to log
        :type exception: Exception
        """
        logging.error(message, exc_info=exception)
        self.view.showMessage(message)

    def useWorker(self, func, args, on_success=None, on_failure=None, on_complete=None):
        """Calls the given function from a new worker thread object

        :param func: function to run on ``QThread``
        :type func: Callable[..., Any]
        :param args: arguments of function ``func``
        :type args: Tuple[Any, ...]
        :param on_success: function to call if success
        :type on_success: Union[Callable[..., None], None]
        :param on_failure: function to call if failed
        :type on_failure: Union[Callable[..., None], None]
        :param on_complete: function to call when complete
        :type on_complete: Union[Callable[..., None], None]
        """
        self.worker = Worker.callFromWorker(func, args, on_success, on_failure, on_complete)

    def createProject(self, name, instrument):
        """Creates the stub data for the project

        :param name: The name of the project
        :type name: str
        :param instrument: The name of the instrument used for the project
        :type instrument: str
        """
        self.view.scenes.reset()
        self.resetSimulation()
        self.model.createProjectData(name, instrument)
        self.model.save_path = ''
        self.view.clearUndoStack()
        settings.reset()

    def updateView(self):
        """Updates view after project or instrument is changed"""
        self.view.showProjectName()
        self.view.toggleActiveInstrument()
        self.view.resetInstrumentMenu()
        for name, detector in self.model.instrument.detectors.items():
            show_more = detector.positioner is not None
            title = 'Detector' if name.lower() == 'detector' else f'{name} Detector'
            collimator_name = None if detector.current_collimator is None else detector.current_collimator.name
            self.view.addCollimatorMenu(name, detector.collimators.keys(), collimator_name, title, show_more)

        self.view.docks.closeAll()
        self.view.updateMenus()

    def projectCreationError(self, exception, args):
        """Handles errors from project creation or instrument change

        :param exception: raised exception
        :type exception: Exception
        :param args: arguments passed into function that threw exception
        :type args: Union[Tuple[str], Tuple[str, str]]
        """
        self.view.docks.closeAll()
        if self.model.project_data is None or self.model.instrument is None:
            self.model.project_data = None
            self.view.updateMenus()
            self.view.clearUndoStack()
        else:
            toggleActionInGroup(self.model.instrument.name, self.view.change_instrument_action_group)

        msg = 'An error occurred while parsing the instrument description file for {}.\n\n' \
              'Please contact the maintainer of the instrument model.'.format(args[-1])

        self.notifyError(msg, exception)

    def saveProject(self, save_as=False):
        """Saves a project to a file. A file dialog should be opened for the first save
        after which the function will save to the same location. if save_as is True a dialog is
        opened every time

        :param save_as: A flag denoting whether to use file dialog or not
        :type save_as: bool
        """
        # Avoids saving when there are no changes
        if self.view.undo_stack.isClean() and self.model.save_path and not save_as:
            return

        filename = self.model.save_path
        if save_as or not filename:
            filename = self.view.showSaveDialog('hdf5 File (*.h5)', title='Save Project')
            if not filename:
                return

        try:
            self.model.saveProjectData(filename)
            self.updateRecentProjects(filename)
            self.model.save_path = filename
            self.view.showProjectName()
            self.view.undo_stack.setClean()
        except OSError as e:
            self.notifyError(f'An error occurred while attempting to save this project ({filename}).', e)

    def openProject(self, filename):
        """This function loads a project with the given filename

        :param filename: filename
        :type filename: str
        """
        self.resetSimulation()
        self.model.loadProjectData(filename)
        self.updateRecentProjects(filename)
        self.model.save_path = filename
        self.view.clearUndoStack()

    def projectOpenError(self, exception, args):
        """Reports errors from opening project on a worker

        :param exception: raised exception
        :type exception: Exception
        :param args: arguments passed into function that threw exception
        :type args: Tuple[str]
        """
        self.view.docks.closeAll()
        filename = args[0]
        if isinstance(exception, ValueError):
            msg = f'Project data could not be read from {filename} because it has incorrect data: {exception}'
        elif isinstance(exception, (KeyError, AttributeError)):
            msg = f'{filename} could not open because it has an incorrect format.'
        elif isinstance(exception, OSError):
            msg = 'An error occurred while opening this file.\nPlease check that ' \
                  f'the file exist and also that this user has access privileges for this file.\n({filename})'
        else:
            msg = f'An unknown error occurred while opening {filename}.'

        self.notifyError(msg, exception)

    def confirmSave(self):
        """Checks if the project is saved and asks the user to save if necessary

        :return: True if the project is saved or user chose to discard changes
        :rtype: bool
        """
        if self.model.project_data is None or self.view.undo_stack.isClean():
            return True

        reply = self.view.showSaveDiscardMessage(self.model.project_data['name'])

        if reply == MessageReplyType.Save:
            if self.model.save_path:
                self.saveProject()
                return True
            else:
                self.saveProject(save_as=True)
                return True if self.view.undo_stack.isClean() else False

        elif reply == MessageReplyType.Discard:
            return True
        else:
            return False

    def updateRecentProjects(self, filename):
        """Adds a filename entry to the front of the recent projects list
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

    def confirmCombineSample(self):
        """Asks if new sample should be combined with old or replace it

        :return: True if sample should be combined
        :rtype: bool
        """
        if self.model.sample:
            question = 'A sample model has already been added to the project.\n\n' \
                       'Do you want replace the model or combine them?'
            choice = self.view.showSelectChoiceMessage(question, ['Combine', 'Replace'], default_choice=1)

            if choice == 'Combine':
                return True

        return False

    def confirmClearStack(self):
        """Asks if undo stack should cleared

        :return: True if stack should be cleared
        :rtype: bool
        """
        if self.view.undo_stack.count() == 0:
            return True

        question = 'This action cannot be undone, the undo history will be cleared.\n\n' \
                   'Do you want proceed with this action?'
        choice = self.view.showSelectChoiceMessage(question, ['Proceed', 'Cancel'], default_choice=1)

        if choice == 'Proceed':
            return True

        return False

    def importSample(self):
        """Adds command to insert sample from file into the view's undo stack"""
        filename = self.view.showOpenDialog('3D Files (*.stl *.obj)', title='Import Sample Model')

        if not filename:
            return

        insert_command = InsertSampleFromFile(filename, self, self.confirmCombineSample())
        self.view.undo_stack.push(insert_command)

    def exportSample(self):
        """Exports a sample as .stl file"""
        if not self.model.sample:
            self.view.showMessage('No samples have been added to the project', MessageSeverity.Information)
            return

        if len(self.model.sample) > 1:
            sample_key = self.view.showSampleExport(self.model.sample.keys())
        else:
            sample_key = list(self.model.sample.keys())[0]

        if not sample_key:
            return

        filename = self.view.showSaveDialog('Binary STL File(*.stl)', title=f'Export {sample_key}')

        if not filename:
            return

        try:
            self.model.saveSample(filename, sample_key)
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the sample ({sample_key}) to {filename}.', e)

    def addPrimitive(self, primitive, args):
        """Adds command to insert primitives as sample into the view's undo stack

        :param primitive: primitive type
        :type primitive: Primitives
        :param args: arguments for primitive creation
        :type args: Dict
        """
        insert_command = InsertPrimitive(primitive, args, self, combine=self.confirmCombineSample())
        self.view.undo_stack.push(insert_command)
        self.view.docks.showSampleManager()

    def transformSample(self, angles_or_offset, sample_key, transform_type):
        """Adds command to transform samples into the view's undo stack

        :param angles_or_offset: angles, offsets or matrix
        :type angles_or_offset: Union[List[float], List[List[float]]]
        :param sample_key: sample key
        :type sample_key: str
        :param transform_type: transform type
        :type transform_type: TransformType
        """
        if transform_type == TransformType.Rotate:
            transform_command = RotateSample(angles_or_offset, sample_key, self)
        elif transform_type == TransformType.Translate:
            transform_command = TranslateSample(angles_or_offset, sample_key, self)
        else:
            transform_command = TransformSample(angles_or_offset, sample_key, self)

        self.view.undo_stack.push(transform_command)

    def deleteSample(self, sample_keys):
        """Adds command to delete samples into the view's undo stack

        :param sample_keys: key(s) of sample(s)
        :type sample_keys: List[str]
        """
        delete_command = DeleteSample(sample_keys, self)
        self.view.undo_stack.push(delete_command)

    def mergeSample(self, sample_keys):
        """Adds command to merge sample(s) into the view's undo stack

        :param sample_keys: key(s) of sample(s)
        :type sample_keys: List[str]
        """
        merge_command = MergeSample(sample_keys, self)
        self.view.undo_stack.push(merge_command)

    def changeMainSample(self, sample_key):
        """Adds command to change main sample into the view's undo stack

        :param sample_key: key of sample
        :type sample_key: str
        """
        change_main_command = ChangeMainSample(sample_key, self)
        self.view.undo_stack.push(change_main_command)

    def importPoints(self, point_type):
        """Adds command to import fiducial or measurement points from file into the view's undo stack

        :param point_type: point type
        :type point_type: PointType
        """
        if not self.model.sample:
            self.view.showMessage('A sample model should be added before {} points'.format(point_type.value.lower()),
                                  MessageSeverity.Information)
            return

        filename = self.view.showOpenDialog(f'{point_type.value} File(*.{point_type.value.lower()})',
                                            title=f'Import {point_type.value} Points')

        if not filename:
            return

        insert_command = InsertPointsFromFile(filename, point_type, self)
        self.view.undo_stack.push(insert_command)

    def exportPoints(self, point_type):
        """Exports fiducial or measurement points to file"""
        points = self.model.fiducials if point_type == PointType.Fiducial else self.model.measurement_points
        if points.size == 0:
            self.view.showMessage('No {} points have been added to the project'.format(point_type.value.lower()),
                                  MessageSeverity.Information)
            return

        filename = self.view.showSaveDialog(f'{point_type.value} File(*.{point_type.value.lower()})',
                                            title=f'Export {point_type.value} Points')

        if not filename:
            return

        try:
            self.model.savePoints(filename, point_type)
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the {point_type.value} points to {filename}.', e)

    def addPoints(self, points, point_type, show_manager=True):
        """Adds command to insert fiducial or measurement points into the view's undo stack

        :param points: array of points
        :type points: List[Tuple[List[float], bool]]
        :param point_type: point type
        :type point_type: PointType
        :param show_manager: indicates point manager should be opened
        :type show_manager: bool
        """
        if not self.model.sample:
            self.view.showMessage('A sample model should be added before {} points'.format(point_type.value.lower()),
                                  MessageSeverity.Information)
            return

        insert_command = InsertPoints(points, point_type, self)
        self.view.undo_stack.push(insert_command)
        if show_manager:
            self.view.docks.showPointManager(point_type)

    def deletePoints(self, indices, point_type):
        """Adds command to delete fiducial or measurement points into the view's undo stack

        :param indices: indices of points
        :type indices: List[int]
        :param point_type: point type
        :type point_type: PointType
        """
        delete_command = DeletePoints(indices, point_type, self)
        self.view.undo_stack.push(delete_command)

    def movePoints(self, move_from, move_to, point_type):
        """Adds command to change order of fiducial or measurement points into the view's undo stack

        :param move_from: start index
        :type move_from: int
        :param move_to: destination index
        :type move_to: int
        :param point_type: point type
        :type point_type: PointType
        """
        move_command = MovePoints(move_from, move_to, point_type, self)
        self.view.undo_stack.push(move_command)

    def editPoints(self, values, point_type):
        """Adds command to edit fiducial or measurement points into the view's undo stack

        :param values: point array after edit
        :type values: numpy.recarray
        :param point_type: point type
        :type point_type: PointType
        """
        edit_command = EditPoints(values, point_type, self)
        self.view.undo_stack.push(edit_command)

    def importVectors(self):
        """Adds command to import measurement vectors from file into the view's undo stack"""
        if not self.model.sample:
            self.view.showMessage('Sample model and measurement points should be added before vectors',
                                  MessageSeverity.Information)
            return

        if self.model.measurement_points.size == 0:
            self.view.showMessage('Measurement points should be added before vectors', MessageSeverity.Information)
            return

        filename = self.view.showOpenDialog('Measurement Vector File(*.vecs)', title='Import Measurement Vectors')

        if not filename:
            return

        insert_command = InsertVectorsFromFile(filename, self)
        self.view.undo_stack.push(insert_command)

    def removeVectors(self, indices, detector, alignment):
        """Adds command to remove measurement vectors into the view's undo stack

        :param indices: indices of vectors
        :type indices: List[int]
        :param detector: index of detector
        :type detector: int
        :param alignment: index of alignment
        :type alignment: int
        """

        vectors = self.model.measurement_vectors[indices, slice(detector * 3, detector * 3 + 3), alignment]

        if (np.linalg.norm(vectors, axis=1) < VECTOR_EPS).all():
            return

        remove_command = RemoveVectors(indices, detector, alignment, self)
        self.view.undo_stack.push(remove_command)

    def removeVectorAlignment(self, index):
        """Adds command to remove measurement vector alignment into the view's undo stack

        :param index: index of alignment
        :type index: int
        """
        remove_command = RemoveVectorAlignment(index, self)
        self.view.undo_stack.push(remove_command)

    def addVectors(self, point_index, strain_component, alignment, detector, key_in=None, reverse=False):
        """Adds command to create measurement vectors into the view's undo stack

        :param point_index: index of measurement point, when index is -1 adds vectors for all points
        :type point_index: int
        :param strain_component: strain component method
        :type strain_component: StrainComponents
        :param alignment: index of alignment
        :type alignment: int
        :param detector: index of detector
        :type detector: int
        :param key_in: custom vector
        :type key_in: Union[None, List[float]]
        :param reverse: flag indicating vector should be reversed
        :type reverse: bool
        """
        if not self.model.sample:
            self.view.showMessage('Sample model and measurement points should be added before vectors',
                                  MessageSeverity.Information)
            return

        if self.model.measurement_points.size == 0:
            self.view.showMessage('Measurement points should be added before vectors', MessageSeverity.Information)
            return

        insert_command = InsertVectors(self, point_index, strain_component, alignment, detector, key_in, reverse)
        self.view.undo_stack.push(insert_command)

    def exportVectors(self):
        """Exports measurement vectors to .vecs file"""
        if self.model.measurement_vectors.shape[0] == 0:
            self.view.showMessage('No measurement vectors have been added to the project', MessageSeverity.Information)
            return

        filename = self.view.showSaveDialog('Measurement Vector File(*.vecs)', title='Export Measurement Vectors')

        if not filename:
            return

        try:
            self.model.saveVectors(filename)
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the measurement vector to {filename}.', e)

    def importTransformMatrix(self):
        """Imports transformation matrix from .trans file

        :return: imported matrix
        :rtype: Union[Matrix44, None]
        """
        filename = self.view.showOpenDialog('Transformation Matrix File(*.trans)',
                                            title='Import Transformation Matrix')

        if not filename:
            return None

        try:
            matrix = read_trans_matrix(filename)
            if not check_rotation(matrix):
                self.view.showMessage('The imported matrix is an invalid rotation. The rotation vectors should '
                                      f'have a magnitude of 1 (accurate to 7 decimal digits) - {filename}.',
                                      MessageSeverity.Critical)
                return None
            return matrix
        except (OSError, ValueError) as e:
            if isinstance(e, ValueError):
                msg = f'Alignment matrix could not be read from {filename} because it has incorrect data: {e}'
            else:
                msg = 'An error occurred while opening this file.\nPlease check that ' \
                      f'the file exist and also that this user has access privileges for this file.\n({filename})'

            self.notifyError(msg, e)

        return None

    def exportAlignmentMatrix(self):
        """Exports alignment matrix to .trans file"""
        if self.model.alignment is None:
            self.view.showMessage('Sample has not been aligned on instrument.', MessageSeverity.Information)
            return

        filename = self.view.showSaveDialog('Transformation Matrix File(*.trans)',
                                            title='Export Alignment Matrix')

        if not filename:
            return

        try:
            np.savetxt(filename, self.model.alignment, delimiter='\t', fmt='%.7f')
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the alignment matrix to {filename}.', e)

    def changeCollimators(self, detector_name, collimator_name):
        """Adds command to change collimator into the view's undo stack

        :param detector_name: name of detector
        :type detector_name: str
        :param collimator_name: new collimator name
        :type collimator_name: Union[str, None]
        """
        command = ChangeCollimator(detector_name, collimator_name, self)
        self.view.undo_stack.push(command)

    def lockPositionerJoint(self, positioner_name, index, value):
        """Adds command to lock or unlock specific positioner joint into the view's undo stack

        :param positioner_name: name of positioner
        :type positioner_name: str
        :param index: joint index
        :type index: int
        :param value: indicates if joint is locked
        :type value: bool
        """
        command = LockJoint(positioner_name, index, value, self)
        self.view.undo_stack.push(command)

    def ignorePositionerJointLimits(self, positioner_name, index, value):
        """Adds command to ignore or use limits of positioner joints into the view's undo stack

        :param positioner_name: name of positioner
        :type positioner_name: str
        :param index: joint index
        :type index: int
        :param value: indicates joint limit should be ignored
        :type value: bool
        """
        command = IgnoreJointLimits(positioner_name, index, value, self)
        self.view.undo_stack.push(command)

    def movePositioner(self, positioner_name, q, ignore_locks=False):
        """Adds command to move positioner joints into the view's undo stack

        :param positioner_name:  name of positioner
        :type positioner_name: str
        :param q: list of joint offsets to move to. The length must be equal to number of links
        :type q: List[float]
        :param ignore_locks: indicates that joint locks should be ignored
        :type ignore_locks: bool
        """
        command = MovePositioner(positioner_name, q, ignore_locks, self)
        self.view.undo_stack.push(command)

    def changePositioningStack(self, name):
        """Adds command to change positioning stack into the view's undo stack

        :param name: name of positioning stack
        :type name: str
        """
        command = ChangePositioningStack(name, self)
        self.view.undo_stack.push(command)

    def changePositionerBase(self, positioner, matrix):
        """Adds command to change positioner base matrix into the view's undo stack

        :param positioner: auxiliary positioner
        :type positioner: SerialManipulator
        :param matrix: base matrix
        :type matrix: sscanss.core.math.matrix.Matrix44
        """
        command = ChangePositionerBase(positioner, matrix, self)
        self.view.undo_stack.push(command)

    def changeJawAperture(self, aperture):
        """Adds command to change jaw aperture into the view's undo stack

        :param aperture: new aperture
        :type aperture: List[float]
        """
        command = ChangeJawAperture(aperture, self)
        self.view.undo_stack.push(command)

    def changeInstrument(self, instrument_name):
        """Switch from current to specified instrument

        :param instrument_name: name of instrument
        :type instrument_name: str
        """
        if self.model.instrument.name == instrument_name and self.model.checkInstrumentVersion():
            return

        if not self.confirmClearStack():
            toggleActionInGroup(self.model.instrument.name, self.view.change_instrument_action_group)
            return

        self.view.progress_dialog.show(f'Loading {instrument_name} Instrument')
        self.useWorker(self._changeInstrumentHelper, [instrument_name], self.updateView,
                       self.projectCreationError, self.view.progress_dialog.close)

    def _changeInstrumentHelper(self, instrument):
        self.resetSimulation()
        self.model.changeInstrument(instrument)
        self.model.save_path = ''
        self.view.clearUndoStack()
        self.view.undo_stack.resetClean()

    def alignSample(self, matrix):
        """Align sample on instrument using matrix

        :param matrix: alignment matrix
        :type matrix: Matrix44
        """
        command = InsertAlignmentMatrix(matrix, self)
        self.view.undo_stack.push(command)

    def alignSampleWithPose(self, pose):
        """Align sample on instrument using specified 6D pose. Pose contains 3D translation
        (X, Y, Z) and 3D orientation (XYZ euler angles)

        :param pose: position and orientation
        :type pose: List[float]
        """
        if not self.model.sample:
            self.view.showMessage('A sample model should be added before alignment', MessageSeverity.Information)
            return
        self.alignSample(matrix_from_pose(pose, order='zyx'))

    def alignSampleWithMatrix(self):
        """Align sample on instrument using matrix imported from .trans file"""
        if not self.model.sample:
            self.view.showMessage('A sample model should be added before alignment', MessageSeverity.Information)
            return

        matrix = self.importTransformMatrix()
        if matrix is None:
            return
        self.view.scenes.switchToInstrumentScene()
        self.alignSample(matrix)

    def alignSampleWithFiducialPoints(self):
        """Align sample on instrument using fiducial measurements imported from a .fpos file"""
        if not self.model.sample:
            self.view.showMessage('A sample model should be added before alignment', MessageSeverity.Information)
            return

        if self.model.fiducials.size < 3:
            self.view.showMessage('A minimum of 3 fiducial points is required for sample alignment.',
                                  MessageSeverity.Information)
            return

        count = self.model.fiducials.enabled.sum()
        if count < 3:
            self.view.showMessage('Less than 3 fiducial points are enabled. '
                                  f'Enable at least {3-count} point(s) from the point manager to proceed.',
                                  MessageSeverity.Information)
            return

        filename = self.view.showOpenDialog('Alignment Fiducial File(*.fpos)',
                                            title='Import Sample Alignment Fiducials')

        if not filename:
            return

        try:
            index, points, poses = read_fpos(filename)
        except (OSError, ValueError) as e:
            if isinstance(e, ValueError):
                msg = f'Fpos data could not be read from {filename} because it has incorrect data: {e}'
            else:
                msg = 'An error occurred while opening this file.\nPlease check that ' \
                      f'the file exist and also that this user has access privileges for this file.\n({filename})'

            self.notifyError(msg, e)
            return

        if index.size < 3:
            self.view.showMessage('A minimum of 3 points is required for sample alignment.')
            return

        count = self.model.fiducials.size
        if np.any(index < 0):
            self.view.showMessage('The fiducial point index should start at 1, negative point indices are not allowed.')
            return
        elif np.any(index >= count):
            self.view.showMessage(f'Point index {index.max()+1} exceeds the number of fiducial points {count}.')
            return

        positioner = self.model.instrument.positioning_stack
        link_count = len(positioner.links)
        if poses.size != 0 and poses.shape[1] != link_count:
            self.view.showMessage(f'Incorrect number of joint offsets in fpos file, received {poses.shape[1]} '
                                  f'but expected {link_count}')
            return
        q = positioner.set_points
        end_q = q
        if poses.size != 0:
            for i, pose in enumerate(poses):
                pose = positioner.fromUserFormat(pose)
                end_q = pose
                matrix = (positioner.fkine(pose, ignore_locks=True) @ positioner.tool_link).inverse()
                offset = matrix[0:3, 3].transpose()
                points[i, :] = points[i, :] @ matrix[0:3, 0:3].transpose() + offset

            positioner.fkine(q, ignore_locks=True)

        enabled = self.model.fiducials[index].enabled
        result = self.rigidTransform(index, points, enabled)

        self.view.showAlignmentError()
        self.view.alignment_error.updateModel(index, enabled, points, result)
        self.view.alignment_error.end_configuration = (positioner.name, end_q)

        with suppress(ValueError):
            new_index = find_3d_correspondence(self.model.fiducials.points, points)
            if np.any(new_index != index):
                self.view.alignment_error.indexOrder(new_index)

    def computePositionerBase(self, positioner):
        """Compute positioner base matrix using fiducial measurements imported from a .calib file

        :param positioner: auxiliary positioner
        :type positioner: SerialManipulator
        :return: base matrix
        :rtype: Matrix44
        """

        link_type = [link.type == link.Type.Revolute for link in positioner.links]
        if np.count_nonzero(link_type) < 2:
            self.view.showMessage('Positioners with less than 2 axes of rotation are not supported by the base '
                                  'computation algorithm.', MessageSeverity.Information)
            return

        if self.model.fiducials.size < 3:
            self.view.showMessage('A minimum of 3 fiducial points is required for base computation.',
                                  MessageSeverity.Information)
            return

        filename = self.view.showOpenDialog('Calibration Fiducial File(*.calib)', title='Import Calibration Fiducials')

        if not filename:
            return

        try:
            pose_index, fiducial_index, measured_points, poses = read_robot_world_calibration_file(filename)
        except (OSError, ValueError) as e:
            if isinstance(e, ValueError):
                msg = f'Calibration data could not be read from {filename} because it has incorrect data: {e}'
            else:
                msg = 'An error occurred while opening this file.\nPlease check that ' \
                      f'the file exist and also that this user has access privileges for this file.\n({filename})'

            self.notifyError(msg, e)
            return

        unique_pose_ids = np.unique(pose_index)
        number_of_poses = unique_pose_ids.size
        if number_of_poses < 3:
            self.view.showMessage('A minimum of 3 poses is required for base computation.')
            return
        elif unique_pose_ids.min() != 0 or unique_pose_ids.max() != number_of_poses - 1:
            self.view.showMessage('The pose index should start at 1 and be consecutive, negative pose indices '
                                  'are not allowed.')
            return

        count = self.model.fiducials.size
        if np.any(fiducial_index < 0):
            self.view.showMessage('The fiducial point index should start at 1, negative point indices are not allowed.')
            return
        elif np.any(fiducial_index >= count):
            self.view.showMessage(f'Point index {fiducial_index.max()+1} exceeds the number of fiducial '
                                  f'points {count}.')
            return
        elif np.any(~self.model.fiducials[fiducial_index].enabled):
            self.view.showMessage('All fiducial points used for the base computation must be enabled.')
            return

        indices = []
        points = []
        pool = []
        for i in range(number_of_poses):
            temp = np.where(pose_index == i)[0]
            if temp.shape[0] < 3:
                self.view.showMessage('Each pose must have a least 3 measured points.')
                return
            indices.append(fiducial_index[temp])
            points.append(measured_points[temp, :])
            pool.append(poses[temp[0], :])

        base_to_end = []
        sensor_to_tool = []

        link_count = len(positioner.links)
        if poses.shape[1] != link_count:
            self.view.showMessage(f'Incorrect number of joint offsets in calib file, received {poses.shape[1]} '
                                  f'but expected {link_count}')
            return

        fiducials = self.model.fiducials.points
        adj_fiducials = fiducials - np.mean(fiducials, axis=0)

        q = positioner.set_points

        for i in range(number_of_poses):
            pose = positioner.fromUserFormat(pool[i])
            base_to_end.append(positioner.fkine(pose, include_base=False, ignore_locks=True))
            sensor_to_tool.append(rigid_transform(adj_fiducials[indices[i], :], points[i]).matrix)

        positioner.fkine(q, ignore_locks=True)
        try:
            tool_matrix, base_matrix = robot_world_calibration(base_to_end, sensor_to_tool)
        except np.linalg.LinAlgError as e:
            msg = ('Base matrix computation failed! Check that the provided calibration data is properly labelled and '
                   'the positioner poses do not move the sample in a single plane.')
            self.notifyError(msg, e)
            return

        new_points = []
        for i in range(number_of_poses):
            matrix = (base_matrix @ base_to_end[i] @ tool_matrix).transpose()
            points = (adj_fiducials @ matrix[0:3, 0:3]) + matrix[3, 0:3]
            new_points.append(points[indices[i], :])

        new_points = np.vstack(new_points)
        error = new_points - measured_points
        if self.view.showCalibrationError(pose_index, fiducial_index, error):
            return base_matrix

    def exportBaseMatrix(self, matrix):
        """Exports base matrix to .trans file

        :param matrix: base matrix
        :type matrix: Matrix44
        """
        save_path = f'{os.path.splitext(self.model.save_path)[0]}_base_matrix' if self.model.save_path else ''
        filename = self.view.showSaveDialog('Transformation Matrix File(*.trans)', current_dir=save_path,
                                            title='Export Base Matrix')

        if not filename:
            return

        try:
            np.savetxt(filename, matrix, delimiter='\t', fmt='%.7f')
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the base matrix to {filename}.', e)

    def rigidTransform(self, index, points, enabled):
        """Computes rigid transformation between fiducial points and points
        from a fpos file which have specified index and are enabled.

        :param index: index of points
        :type index: numpy.ndarray[int]
        :param points: N x 3 array of points
        :type points: numpy.ndarray[float]
        :param enabled: indicates which points are enabled
        :type enabled: numpy.ndarray[bool]
        :return: rigid transform result
        :rtype: TransformResult
        """
        reference = self.model.fiducials[index].points
        return rigid_transform(reference[enabled], points[enabled])

    def runSimulation(self):
        """Create and start new simulation"""
        if self.model.alignment is None:
            self.view.showMessage('Sample must be aligned on the instrument for Simulation',
                                  MessageSeverity.Information)
            return

        if self.model.measurement_points.size == 0:
            self.view.showMessage('Measurement points should be added before Simulation', MessageSeverity.Information)
            return

        if not self.model.measurement_points.enabled.any():
            self.view.showMessage('No measurement points are enabled. Enable points from the point manager to proceed.',
                                  MessageSeverity.Information)
            return

        if settings.value(settings.Key.Skip_Zero_Vectors):
            vectors = self.model.measurement_vectors[self.model.measurement_points.enabled, :, :]
            if (np.linalg.norm(vectors, axis=1) < VECTOR_EPS).all():
                self.view.showMessage('No measurement vectors have been added and the software is configured to '
                                      '"Skip the measurement" when the measurement vector is unset. Change '
                                      'the behaviour in Preferences to proceed.', MessageSeverity.Information)
                return

        self.view.docks.showSimulationResults()
        if self.model.simulation is not None and self.model.simulation.isRunning():
            return

        compute_path_length = self.view.compute_path_length_action.isChecked()
        check_collision = self.view.check_collision_action.isChecked()
        render_graphics = self.view.show_sim_graphics_action.isChecked()
        check_limits = self.view.check_limits_action.isChecked()

        self.model.createSimulation(compute_path_length, render_graphics, check_limits, check_collision)
        # Start the simulation process. This can be slow due to pickling of arguments
        self.model.simulation.start()

    def stopSimulation(self):
        """Stops simulation"""
        if self.model.simulation is None or not self.model.simulation.isRunning():
            return

        self.model.simulation.abort()

    def resetSimulation(self):
        """Sets the simulation to None"""
        self.stopSimulation()
        self.model.simulation = None

    def exportScript(self, script_renderer):
        """Exports instrument script to text file. This function allows delayed generation of
        scripts until the filename is selected

        :param script_renderer: function to generate script
        :type script_renderer: Callable[None, Str]
        :return: indicates if export succeeded
        :rtype: bool
        """
        save_path = f'{os.path.splitext(self.model.save_path)[0]}_script' if self.model.save_path else ''
        filename = self.view.showSaveDialog('Text File (*.txt)', current_dir=save_path, title='Export Script')
        if filename:
            script_text = script_renderer()
            try:
                with open(filename, "w", newline="\n") as text_file:
                    text_file.write(script_text)
                return True
            except OSError as e:
                self.notifyError(f'A error occurred while attempting to save this project ({filename})', e)

        return False
