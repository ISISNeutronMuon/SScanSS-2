import os
import logging
import numpy as np
from contextlib import suppress
from .model import MainWindowModel
from sscanss.config import INSTRUMENTS_PATH, settings
from sscanss.app.commands import (InsertPrimitive, CreateVectorsWithEulerAngles, InsertAlignmentMatrix,
                                  InsertMeshFromFile, InsertVolumeFromFile, RotateSample, TranslateSample,
                                  TransformSample, InsertPointsFromFile, InsertPoints, DeletePoints, RemoveVectors,
                                  MovePoints, EditPoints, InsertVectorsFromFile, InsertVectors, LockJoint,
                                  IgnoreJointLimits, MovePositioner, ChangePositioningStack, ChangePositionerBase,
                                  ChangeCollimator, ChangeJawAperture, RemoveVectorAlignment, ChangeVolumeCurve)
from sscanss.core.io import read_trans_matrix, read_fpos, read_robot_world_calibration_file, write_fpos, read_csv
from sscanss.core.geometry import Mesh
from sscanss.core.util import (TransformType, MessageType, Worker, toggle_action_in_group, PointType, MessageReplyType,
                               InsertSampleOptions)
from sscanss.core.instrument import robot_world_calibration
from sscanss.core.math import matrix_from_pose, find_3d_correspondence, rigid_transform, check_rotation, VECTOR_EPS


class MainWindowPresenter:
    """Facilitates interaction between View and Model

    :param view: main window instance
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
        self.can_discard = False  # Need a better solution
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

    def useWorker(self, func, args, on_success=None, on_failure=None, on_complete=None, message=''):
        """Calls the given function from a new worker thread object

        :param func: function to run on ``QThread``
        :type func: Callable[..., Any]
        :param args: arguments of function ``func``
        :type args: Tuple[Any, ...]
        :param on_success: function to call if success
        :type on_success: Optional[Callable[..., None]]
        :param on_failure: function to call if failed
        :type on_failure: Optional[Callable[..., None]]
        :param on_complete: function to call when complete
        :type on_complete: Optional[Callable[..., None]]
        :param message: operation message
        :type message: str
        """
        self.worker = Worker.callFromWorker(func, args, on_success, on_failure, on_complete)
        if message:
            self.view.closeNonModalDialog()
            self.view.progress_dialog.showMessage(message)
            self.worker.finished.connect(self.view.progress_dialog.close)

    def createProject(self, name, instrument):
        """Creates an empty project and reset view

        :param name: name of the project
        :type name: str
        :param instrument: name of the instrument for the project
        :type instrument: str
        """
        self.resetSimulation()
        self.model.createProjectData(name, instrument)
        self.model.save_path = ''
        self.view.clearUndoStack()
        settings.reset()

    def updateView(self):
        """Updates the view after project or instrument is changed"""
        self.view.showProjectName()
        self.view.toggleActiveInstrument()
        self.view.resetInstrumentMenu()
        for name, detector in self.model.instrument.detectors.items():
            show_more = detector.positioner is not None
            title = 'Detector' if name.lower() == 'detector' else f'{name} Detector'
            collimator_name = None if detector.current_collimator is None else detector.current_collimator.name
            self.view.addCollimatorMenu(name, detector.collimators.keys(), collimator_name, title, show_more)

        self.view.docks.closeAll()
        self.view.closeNonModalDialog()
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
            toggle_action_in_group(self.model.instrument.name, self.view.change_instrument_action_group)

        msg = f'An error occurred while parsing the instrument description file for {args[-1]}.\n\n' \
              'Please contact the maintainer of the instrument model.'

        self.notifyError(msg, exception)

    def saveProject(self, save_as=False, callback=None):
        """Saves a project to a file. A file dialog should be opened for the first save
        after which the function will save to the same location. if save_as is True a dialog is
        opened every time

        :param save_as: indicates if file dialog should be used
        :type save_as: bool
        :param callback: callback to call after success save
        :type callback: Optional[Callable[None, None]]
        """
        # Avoids saving when there are no changes
        if self.view.undo_stack.isClean() and self.model.save_path and not save_as:
            return
        filename = self.model.save_path
        if save_as or not filename:
            filename = self.view.showSaveDialog('hdf5 File (*.h5)', title='Save Project')
            if not filename:
                return
        error_msg = f'An error occurred while attempting to save this project ({filename}).'

        def on_success():
            self._saveProjectSuccess()
            if callback is not None:
                callback()

        self.useWorker(self._saveProjectHelper, [filename],
                       message='Saving Project to File',
                       on_failure=lambda e: self.notifyError(error_msg, e),
                       on_success=on_success)

    def _saveProjectHelper(self, filename):
        self.model.saveProjectData(filename)
        self.updateRecentProjects(filename)
        self.model.save_path = filename

    def _saveProjectSuccess(self):
        self.view.showProjectName()
        self.view.undo_stack.setClean()

    def openProject(self, filename=''):
        """Loads a project with the given filename. if filename is empty,
        a file dialog will be opened

        :param filename: full path of file
        :type filename: str
        """
        if not filename:
            filename = self.view.showOpenDialog('hdf5 File (*.h5)',
                                                title='Open Project',
                                                current_dir=self.model.save_path)
            if not filename:
                return

        self.useWorker(self._openProjectHelper, [filename],
                       self.updateView,
                       self.projectOpenError,
                       message='Loading Project from File')

    def _openProjectHelper(self, filename):
        """Loads a project with the given filename

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

    def confirmSave(self, callback):
        """Checks if the project is saved and asks the user to save if necessary

        :param callback: callback to call if the project is saved or discarded
        :type callback: Callable[None, None]
        """
        if self.model.project_data is None or self.view.undo_stack.isClean():
            callback()
            return

        self.can_discard = False
        reply = self.view.showSaveDiscardMessage(self.model.project_data['name'])
        if reply == MessageReplyType.Save:
            if self.model.save_path:
                self.saveProject(callback=callback)
            else:
                self.saveProject(save_as=True, callback=callback)
        elif reply == MessageReplyType.Discard:
            self.can_discard = True
            callback()

    def updateRecentProjects(self, filename):
        """Adds a filename entry to the front of the recent projects list
        if it does not exist in the list. if the entry already exist, it is moved to the
        front but not duplicated.

        :param filename: project path to add to recent file lists
        :type filename: str
        """
        filename = os.path.normpath(filename)
        projects = self.view.recent_projects
        projects.insert(0, filename)
        projects = list(dict.fromkeys(projects))
        if len(projects) <= self.recent_list_size:
            self.view.recent_projects = projects
        else:
            self.view.recent_projects = projects[:self.recent_list_size]

    def confirmInsertSampleOption(self, is_mesh=True):
        """Asks if new sample should be combined with old or replace it

        :param is_mesh: indicates if new sample is a mesh
        :type is_mesh: bool
        :return: indicates if sample should be combined
        :rtype: Optional[InsertSampleOptions]
        """
        cancel_text = 'Cancel'
        if self.model.sample is None:
            return InsertSampleOptions.Replace
        if isinstance(self.model.sample, Mesh) and is_mesh:
            question = 'A sample model has already been added to the project.\n\n' \
                       'Do you want to replace the model or combine them?'
            options = [*[option.value for option in InsertSampleOptions], cancel_text]
        else:
            question = 'A sample model has already been added to the project.\n\n' \
                       'Do you want to replace the model?'
            options = [InsertSampleOptions.Replace.value, cancel_text]

        choice = self.view.showSelectChoiceMessage(question, options, cancel_choice=options.index(cancel_text))
        if choice == cancel_text:
            return

        return InsertSampleOptions(choice)

    def confirmClearStack(self):
        """Asks if undo stack should cleared

        :return: indicates if stack should be cleared
        :rtype: bool
        """
        if self.view.undo_stack.count() == 0:
            return True

        question = 'This action cannot be undone, the undo history will be cleared.\n\n' \
                   'Do you want proceed with this action?'
        choice = self.view.showSelectChoiceMessage(question, ['Proceed', 'Cancel'], cancel_choice=1)

        if choice == 'Proceed':
            return True

        return False

    def importMesh(self):
        """Adds a command to insert sample from file into the view's undo stack"""
        filename = self.view.showOpenDialog('3D Files (*.stl *.obj)', title='Import Sample Model')

        if not filename:
            return

        insert_option = self.confirmInsertSampleOption()
        if insert_option is None:
            return

        insert_command = InsertMeshFromFile(filename, self, insert_option)
        self.view.undo_stack.push(insert_command)

    def importVolume(self, filepath, voxel_size=None, centre=None):
        """Adds a command to insert volume from file into the view's undo stack

        :param filepath: path of file or path of the folder containing TIFF files
        :type filepath: str
        :param voxel_size: size of the volume's voxels in the x, y, and z axes
        :type voxel_size: Optional[List[float, float, float]]
        :param centre: coordinates of the volume centre in the x, y, and z axes
        :type centre: Optional[List[float, float, float]]
        """
        insert_option = self.confirmInsertSampleOption(False)
        if insert_option is None:
            return

        insert_command = InsertVolumeFromFile(self, filepath, voxel_size, centre)
        self.view.undo_stack.push(insert_command)

    def loadVolumeFromNexus(self):
        """Loads volume from nexus file"""
        filename = self.view.showOpenDialog(filters='Nexus Files (*.nxs *.nex)', title='Open Nexus File')
        if not filename:
            return

        self.importVolume(filename)

    def changeVolumeCurve(self, curve):
        """Adds a command to change a volume's curve into the view's undo stack

        :param curve: volume curve
        :type curve: Curve
        """
        change_command = ChangeVolumeCurve(curve, self)
        self.view.undo_stack.push(change_command)

    def exportSample(self):
        """Exports a sample as .stl file"""
        if self.model.sample is None:
            self.view.showMessage('No samples have been added to the project', MessageType.Information)
            return

        if isinstance(self.model.sample, Mesh):
            filters = 'Binary STL File(*.stl)'
            select_folder = False
        else:
            filters = ''
            select_folder = True

        path = self.view.showSaveDialog(filters=filters, title='Export Sample', select_folder=select_folder)

        if not path:
            return

        if os.path.isdir(path) and os.listdir(path):
            question = ('The selected folder is not empty, any existing files with the similar names as '
                        'the output will be overwritten. Do you want proceed with this action?')
            options = ['Proceed', 'Cancel']
            choice = self.view.showSelectChoiceMessage(question, options, cancel_choice=1)
            if choice == options[1]:
                return

        error_msg = f'An error occurred while exporting the sample to {path}.'
        self.useWorker(self.model.saveSample, [path],
                       message='Exporting Sample to File',
                       on_failure=lambda e: self.notifyError(error_msg, e))

    def addPrimitive(self, primitive, args):
        """Adds a command to insert primitives as sample into the view's undo stack

        :param primitive: primitive type
        :type primitive: Primitives
        :param args: arguments for primitive creation
        :type args: Dict
        """
        insert_option = self.confirmInsertSampleOption()
        if insert_option is None:
            return

        insert_command = InsertPrimitive(primitive, args, self, insert_option)
        self.view.undo_stack.push(insert_command)

    def transformSample(self, angles_or_offset, transform_type):
        """Adds a command to transform samples into the view's undo stack

        :param angles_or_offset: angles, offsets or matrix
        :type angles_or_offset: Union[List[float], List[List[float]]]
        :param transform_type: transform type
        :type transform_type: TransformType
        """
        if transform_type == TransformType.Rotate:
            transform_command = RotateSample(angles_or_offset, self)
        elif transform_type == TransformType.Translate:
            transform_command = TranslateSample(angles_or_offset, self)
        else:
            transform_command = TransformSample(angles_or_offset, self)

        self.view.undo_stack.push(transform_command)

    def importPoints(self, point_type):
        """Adds a command to import fiducial or measurement points from file into the view's undo stack

        :param point_type: point type
        :type point_type: PointType
        """
        if self.model.sample is None:
            self.view.showMessage(f'A sample model should be added before {point_type.value.lower()} points',
                                  MessageType.Information)
            return

        filename = self.view.showOpenDialog(f'{point_type.value} File(*.{point_type.value.lower()})',
                                            title=f'Import {point_type.value} Points')

        if not filename:
            return

        insert_command = InsertPointsFromFile(filename, point_type, self)
        self.view.undo_stack.push(insert_command)

    def exportPoints(self, point_type):
        """Exports the fiducial or measurement points to file"""
        points = self.model.fiducials if point_type == PointType.Fiducial else self.model.measurement_points
        if points.size == 0:
            self.view.showMessage(f'No {point_type.value.lower()} points have been added to the project',
                                  MessageType.Information)
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
        """Adds a command to insert fiducial or measurement points into the view's undo stack

        :param points: array of points
        :type points: List[Tuple[List[float], bool]]
        :param point_type: point type
        :type point_type: PointType
        :param show_manager: indicates point manager should be opened
        :type show_manager: bool
        """
        if self.model.sample is None:
            self.view.showMessage(f'A sample model should be added before {point_type.value.lower()} points',
                                  MessageType.Information)
            return

        insert_command = InsertPoints(points, point_type, self)
        self.view.undo_stack.push(insert_command)
        if show_manager:
            self.view.docks.showPointManager(point_type)

    def deletePoints(self, indices, point_type):
        """Adds a command to delete fiducial or measurement points into the view's undo stack

        :param indices: indices of points
        :type indices: List[int]
        :param point_type: point type
        :type point_type: PointType
        """
        delete_command = DeletePoints(indices, point_type, self)
        self.view.undo_stack.push(delete_command)

    def movePoints(self, move_from, move_to, point_type):
        """Adds a command to change order of fiducial or measurement points into the view's undo stack

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
        """Adds a command to edit fiducial or measurement points into the view's undo stack

        :param values: point array after edit
        :type values: numpy.recarray
        :param point_type: point type
        :type point_type: PointType
        """
        edit_command = EditPoints(values, point_type, self)
        self.view.undo_stack.push(edit_command)

    def importVectors(self):
        """Adds a command to import measurement vectors from file into the view's undo stack"""
        if not self.isSafeForVectors():
            return

        filename = self.view.showOpenDialog('Measurement Vector File(*.vecs)', title='Import Measurement Vectors')

        if not filename:
            return

        insert_command = InsertVectorsFromFile(filename, self)
        self.view.undo_stack.push(insert_command)

    def removeVectors(self, indices, detector, alignment):
        """Adds a command to remove measurement vectors into the view's undo stack

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
        """Adds a command to remove measurement vector alignment into the view's undo stack

        :param index: index of alignment
        :type index: int
        """
        remove_command = RemoveVectorAlignment(index, self)
        self.view.undo_stack.push(remove_command)

    def isSafeForVectors(self):
        """Checks if its prerequisite for adding vectors are met"""
        if self.model.sample is None:
            self.view.showMessage('Sample model and measurement points should be added before vectors',
                                  MessageType.Information)
            return False

        if self.model.measurement_points.size == 0:
            self.view.showMessage('Measurement points should be added before vectors', MessageType.Information)
            return False

        return True

    def createVectorsWithEulerAngles(self):
        """Adds a command to create measurement vectors sing Euler angles into the view's undo stack"""
        if not self.isSafeForVectors():
            return

        filename = self.view.showOpenDialog('Angles File (*.angles)', title='Import Euler Angles')

        if not filename:
            return

        insert_command = CreateVectorsWithEulerAngles(filename, self)
        self.view.undo_stack.push(insert_command)

    def addVectors(self, point_index, strain_component, alignment, detector, key_in=None, reverse=False):
        """Adds a command to create measurement vectors into the view's undo stack

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
        if not self.isSafeForVectors():
            return

        insert_command = InsertVectors(self, point_index, strain_component, alignment, detector, key_in, reverse)
        self.view.undo_stack.push(insert_command)

    def exportVectors(self):
        """Exports the measurement vectors to .vecs file"""
        if self.model.measurement_vectors.shape[0] == 0:
            self.view.showMessage('No measurement vectors have been added to the project', MessageType.Information)
            return

        filename = self.view.showSaveDialog('Measurement Vector File(*.vecs)', title='Export Measurement Vectors')

        if not filename:
            return

        try:
            self.model.saveVectors(filename)
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the measurement vector to {filename}.', e)

    def importTransformMatrix(self):
        """Imports a transformation matrix from .trans file

        :return: imported matrix
        :rtype: Optional[Matrix44]
        """
        filename = self.view.showOpenDialog('Transformation Matrix File(*.trans)', title='Import Transformation Matrix')

        if not filename:
            return None

        try:
            matrix = read_trans_matrix(filename)
            if not check_rotation(matrix):
                self.view.showMessage(
                    'The imported matrix is an invalid rotation. The rotation vectors should '
                    f'have a magnitude of 1 (accurate to 7 decimal digits) - {filename}.', MessageType.Error)
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
        """Exports the alignment matrix to .trans file"""
        if self.model.alignment is None:
            self.view.showMessage('Sample has not been aligned on instrument.', MessageType.Information)
            return

        filename = self.view.showSaveDialog('Transformation Matrix File(*.trans)', title='Export Alignment Matrix')

        if not filename:
            return

        try:
            np.savetxt(filename, self.model.alignment, delimiter='\t', fmt='%.7f')
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the alignment matrix to {filename}.', e)

    def changeCollimators(self, detector_name, collimator_name):
        """Adds a command to change collimator into the view's undo stack

        :param detector_name: name of detector
        :type detector_name: str
        :param collimator_name: new collimator name
        :type collimator_name: Union[str, None]
        """
        command = ChangeCollimator(detector_name, collimator_name, self)
        self.view.undo_stack.push(command)

    def lockPositionerJoint(self, positioner_name, index, value):
        """Adds a command to lock or unlock specific positioner joint into the view's undo stack

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
        """Adds a command to ignore or use limits of positioner joints into the view's undo stack

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
        """Adds a command to move positioner joints into the view's undo stack

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
        """Adds a command to change positioning stack into the view's undo stack

        :param name: name of positioning stack
        :type name: str
        """
        command = ChangePositioningStack(name, self)
        self.view.undo_stack.push(command)

    def changePositionerBase(self, positioner, matrix):
        """Adds a command to change positioner base matrix into the view's undo stack

        :param positioner: auxiliary positioner
        :type positioner: SerialManipulator
        :param matrix: base matrix
        :type matrix: Matrix44
        """
        command = ChangePositionerBase(positioner, matrix, self)
        self.view.undo_stack.push(command)

    def changeJawAperture(self, aperture):
        """Adds a command to change jaw aperture into the view's undo stack

        :param aperture: new aperture
        :type aperture: List[float]
        """
        command = ChangeJawAperture(aperture, self)
        self.view.undo_stack.push(command)

    def changeInstrument(self, instrument_name):
        """Changes the project instrument and to specified instrument

        :param instrument_name: name of instrument
        :type instrument_name: str
        """
        if self.model.instrument.name == instrument_name and self.model.checkInstrumentVersion():
            return

        if not self.confirmClearStack():
            toggle_action_in_group(self.model.instrument.name, self.view.change_instrument_action_group)
            return

        self.useWorker(self._changeInstrumentHelper, [instrument_name],
                       self.updateView,
                       self.projectCreationError,
                       message=f'Loading {instrument_name} Instrument')

    def _changeInstrumentHelper(self, instrument_name):
        """Changes the project instrument and to specified instrument and updates view to
        reflect change

        :param instrument_name: name of instrument
        :type instrument_name: str
        """
        self.resetSimulation()
        self.model.changeInstrument(instrument_name)
        self.model.save_path = ''
        self.view.clearUndoStack()
        self.view.undo_stack.resetClean()

    def alignSample(self, matrix):
        """Adds a command to align sample on instrument into the view's undo stack

        :param matrix: alignment matrix
        :type matrix: Matrix44
        """
        command = InsertAlignmentMatrix(matrix, self)
        self.view.undo_stack.push(command)

    def alignSampleWithPose(self, pose):
        """Aligns the sample on instrument using specified 6D pose. Pose contains 3D translation
        (X, Y, Z) and 3D orientation (ZYX euler angles)

        :param pose: position and orientation
        :type pose: List[float]
        """
        if self.model.sample is None:
            self.view.showMessage('A sample model should be added before alignment', MessageType.Information)
            return
        self.alignSample(matrix_from_pose(pose, order='zyx'))

    def alignSampleWithMatrix(self):
        """Aligns the sample on instrument using matrix imported from .trans file"""
        if self.model.sample is None:
            self.view.showMessage('A sample model should be added before alignment', MessageType.Information)
            return

        matrix = self.importTransformMatrix()
        if matrix is None:
            return
        self.view.scenes.switchToInstrumentScene()
        self.alignSample(matrix)

    def alignSampleWithFiducialPoints(self):
        """Aligns the sample on instrument using fiducial measurements imported from a .fpos file"""
        if self.model.sample is None:
            self.view.showMessage('A sample model should be added before alignment', MessageType.Information)
            return

        if self.model.fiducials.size < 3:
            self.view.showMessage('A minimum of 3 fiducial points is required for sample alignment.',
                                  MessageType.Information)
            return

        count = self.model.fiducials.enabled.sum()
        if count < 3:
            self.view.showMessage(
                'Less than 3 fiducial points are enabled. '
                f'Enable at least {3-count} point(s) from the point manager to proceed.', MessageType.Information)
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
        link_count = positioner.link_count
        if poses.size != 0 and poses.shape[1] != link_count:
            self.view.showMessage(f'Incorrect number of joint offsets in fpos file, received {poses.shape[1]} '
                                  f'but expected {link_count}')
            return
        q = positioner.set_points
        end_q = [link.default_offset for link in positioner.links]
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

        order_fix = None
        with suppress(ValueError):
            new_index = find_3d_correspondence(self.model.fiducials.points, points)
            if np.any(new_index != index):
                order_fix = new_index

        self.view.showAlignmentError(index, enabled, points, result, end_q, order_fix)

    def computePositionerBase(self, positioner):
        """Computes the positioner base matrix using fiducial measurements imported from a .calib file

        :param positioner: auxiliary positioner
        :type positioner: SerialManipulator
        :return: base matrix
        :rtype: Matrix44
        """
        link_type = [link.type == link.Type.Revolute for link in positioner.links]
        if np.count_nonzero(link_type) < 2:
            self.view.showMessage(
                'Positioners with less than 2 axes of rotation are not supported by the base '
                'computation algorithm.', MessageType.Information)
            return

        if self.model.fiducials.size < 3:
            self.view.showMessage('A minimum of 3 fiducial points is required for base computation.',
                                  MessageType.Information)
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
        """Exports the base matrix to .trans file

        :param matrix: base matrix
        :type matrix: Matrix44
        """
        save_path = f'{os.path.splitext(self.model.save_path)[0]}_base_matrix' if self.model.save_path else ''
        filename = self.view.showSaveDialog('Transformation Matrix File(*.trans)',
                                            current_dir=save_path,
                                            title='Export Base Matrix')

        if not filename:
            return

        try:
            np.savetxt(filename, matrix, delimiter='\t', fmt='%.7f')
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the base matrix to {filename}.', e)

    def exportCurrentFiducials(self, indices, points, poses=None):
        """Exports the given fiducial points and poses to .fpos file

        :param indices: point indices
        :type indices: numpy.ndarray
        :param points: fiducial point
        :type points: numpy.ndarray
        :param poses: positioner pose
        :type poses: numpy.ndarray
        """
        filename = self.view.showSaveDialog('Alignment Fiducial File (*.fpos)', title='Export Current Fiducials Points')

        if not filename:
            return

        try:
            write_fpos(filename, indices, points, poses)
        except OSError as e:
            self.notifyError(f'An error occurred while exporting the current fiducials positions to {filename}.', e)

    def rigidTransform(self, index, points, enabled):
        """Computes the rigid transformation between selected fiducial points and points from a
        fpos file.

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

    def importJointOffsets(self):
        """Imports a joint offsets from text file

        :return: imported matrix
        :rtype: Optional[Matrix44]
        """
        filename = self.view.showOpenDialog('Joint Offsets(*.txt)', title='Import Joint Offsets')

        if not filename:
            return None

        try:
            tmp = np.array(read_csv(filename), np.float32)
            positioner = self.model.instrument.positioning_stack
            link_count = positioner.link_count
            if tmp.size != 0 and tmp.shape[1] != link_count:
                self.view.showMessage(f'Incorrect number of joint offsets, received {tmp.shape[1]} '
                                      f'but expected {link_count}')
                return None
            joint_offsets = tmp.copy()
            joint_offsets[:, positioner.order] = tmp
            revolute_index = [link.type == link.Type.Revolute for link in positioner.links]
            if any(revolute_index):
                joint_offsets[:, revolute_index] = np.radians(joint_offsets[:, revolute_index])
            return joint_offsets
        except (OSError, ValueError) as e:
            if isinstance(e, ValueError):
                msg = f'Joint Offsets could not be read from {filename} because it has incorrect data: {e}'
            else:
                msg = 'An error occurred while opening this file.\nPlease check that ' \
                      f'the file exist and also that this user has access privileges for this file.\n({filename})'

            self.notifyError(msg, e)
        return None

    def runSimulation(self, use_joint_offsets=False):
        """Creates and starts a new simulation

        :param use_joint_offsets: indicates joint offsets should be used for a simulation
        :type use_joint_offsets: bool
        """
        joint_offsets = None
        if use_joint_offsets:
            joint_offsets = self.importJointOffsets()
            if joint_offsets is None:
                return
        else:
            if self.model.alignment is None:
                self.view.showMessage('Sample must be aligned on the instrument for Simulation',
                                      MessageType.Information)
                return

            if self.model.measurement_points.size == 0:
                self.view.showMessage('Measurement points should be added before Simulation', MessageType.Information)
                return

            if not self.model.measurement_points.enabled.any():
                self.view.showMessage(
                    'No measurement points are enabled. Enable points from the point manager '
                    'to proceed.', MessageType.Information)
                return

            if settings.value(settings.Key.Skip_Zero_Vectors):
                vectors = self.model.measurement_vectors[self.model.measurement_points.enabled, :, :]
                if (np.linalg.norm(vectors, axis=1) < VECTOR_EPS).all():
                    self.view.showMessage(
                        'No measurement vectors have been added and the software is configured to '
                        '"Skip the measurement" when the measurement vector is unset. Change '
                        'the behaviour in Preferences to proceed.', MessageType.Information)
                    return

        self.view.docks.showSimulationResults()
        if self.model.simulation is not None and self.model.simulation.isRunning():
            return

        compute_path_length = self.view.compute_path_length_action.isChecked()
        check_collision = self.view.check_collision_action.isChecked()
        render_graphics = self.view.show_sim_graphics_action.isChecked()
        check_limits = self.view.check_limits_action.isChecked()

        self.model.createSimulation(compute_path_length, render_graphics, check_limits, check_collision, joint_offsets)
        # Start the simulation process. This can be slow due to pickling of arguments
        self.model.simulation.start()

    def stopSimulation(self):
        """Stops the simulation"""
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
