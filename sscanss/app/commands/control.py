from PyQt5 import QtWidgets
from sscanss.core.util import CommandID, toggleActionInGroup, Attributes


class LockJoint(QtWidgets.QUndoCommand):
    """Sets lock state of specified joint

    :param positioner_name: name of positioner
    :type positioner_name: str
    :param index: joint index
    :type index: int
    :param value: indicates if joint is locked
    :type value: bool
    :param presenter: Mainwindow presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, positioner_name, index, value, presenter):
        super().__init__()

        self.model = presenter.model

        self.positioner_name = positioner_name
        stack = self.model.instrument.getPositioner(self.positioner_name)
        self.old_lock_state = [link.locked for link in stack.links]
        self.new_lock_state = self.old_lock_state.copy()
        self.new_lock_state[index] = value

        self.setText(f'Locked Joint in {positioner_name}')

    def redo(self):
        self.changeLockState(self.new_lock_state)

    def undo(self):
        self.changeLockState(self.old_lock_state)

    def changeLockState(self, lock_state):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        for state, link in zip(lock_state, stack.links):
            link.locked = state
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        """Merges consecutive change main commands

        :param command: command to merge
        :type command: QUndoCommand
        :return: True if merge was successful
        :rtype: bool
        """
        if self.positioner_name != command.positioner_name:
            return False

        if self.old_lock_state == command.new_lock_state:
            self.setObsolete(True)

        self.new_lock_state = command.new_lock_state
        return True

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.LockJoint


class IgnoreJointLimits(QtWidgets.QUndoCommand):
    """Sets joint limit ignore state of specified joint

    :param positioner_name: name of positioner
    :type positioner_name: str
    :param index: joint index
    :type index: int
    :param value: indicates joint limit should be ignored
    :type value: bool
    :param presenter: Mainwindow presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, positioner_name, index, value, presenter):
        super().__init__()

        self.model = presenter.model

        self.positioner_name = positioner_name
        stack = self.model.instrument.getPositioner(self.positioner_name)
        self.old_ignore_state = [link.ignore_limits for link in stack.links]
        self.new_ignore_state = self.old_ignore_state.copy()
        self.new_ignore_state[index] = value

        self.setText(f'Ignored Joint Limits in {positioner_name}')

    def redo(self):
        self.changeIgnoreLimitState(self.new_ignore_state)

    def undo(self):
        self.changeIgnoreLimitState(self.old_ignore_state)

    def changeIgnoreLimitState(self, ignore_state):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        for state, link in zip(ignore_state, stack.links):
            link.ignore_limits = state
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        """Merges consecutive change main commands

        :param command: command to merge
        :type command: QUndoCommand
        :return: True if merge was successful
        :rtype: bool
        """
        if self.positioner_name != command.positioner_name:
            return False

        if self.old_ignore_state == command.new_ignore_state:
            self.setObsolete(True)

        self.new_ignore_state = command.new_ignore_state
        return True

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.IgnoreJointLimits


class MovePositioner(QtWidgets.QUndoCommand):
    """Moves the stack to specified configuration. The first move will be animated but any repeats (redo)
    will be an instant move i.e. single step.

    :param positioner_name:  name of positioner
    :type positioner_name: str
    :param q: list of joint offsets to move to. The length must be equal to number of links
    :type q: List[float]
    :param ignore_locks: indicates that joint locks should be ignored
    :type ignore_locks: bool
    :param presenter: Mainwindow presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, positioner_name, q, ignore_locks, presenter):
        super().__init__()

        self.model = presenter.model
        self.view = presenter.view
        self.positioner_name = positioner_name
        stack = self.model.instrument.getPositioner(self.positioner_name)
        self.move_from = stack.set_points
        self.move_to = q

        self.animate = True
        self.ignore_locks = ignore_locks

        self.setText(f'Moved {positioner_name}')

    def redo(self):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        if self.animate:
            stack.set_points = self.move_to
            self.model.moveInstrument(lambda q, s=stack: s.fkine(q, setpoint=False, ignore_locks=self.ignore_locks),
                                      self.move_from, self.move_to, 500, 10)
            self.animate = False
        else:
            stack.fkine(self.move_to, ignore_locks=self.ignore_locks)
            self.model.notifyChange(Attributes.Instrument)
        self.model.instrument_controlled.emit(self.id())

    def undo(self):
        if self.view.scenes.sequence.isRunning():
            self.view.scenes.sequence.stop()
        stack = self.model.instrument.getPositioner(self.positioner_name)
        stack.set_point = self.move_from
        stack.fkine(self.move_from, ignore_locks=self.ignore_locks)
        self.model.notifyChange(Attributes.Instrument)
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        """Merges consecutive change main commands

        :param command: command to merge
        :type command: QUndoCommand
        :return: True if merge was successful
        :rtype: bool
        """
        if self.positioner_name != command.positioner_name or self.ignore_locks != command.ignore_locks:
            return False

        if self.move_from == command.move_to:
            self.setObsolete(True)

        self.move_to = command.move_to
        return True

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.MovePositioner


class ChangePositioningStack(QtWidgets.QUndoCommand):
    """Changes the active positioning stack of the instrument

    :param stack_name: name of positioning stack
    :type stack_name: str
    :param presenter: Mainwindow presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, stack_name, presenter):
        super().__init__()

        self.model = presenter.model

        stack = self.model.instrument.positioning_stack
        self.old_q = stack.set_points
        self.link_state = [(link.locked, link.ignore_limits) for link in stack.links]
        self.bases = [aux.base for aux in stack.auxiliary]

        self.old_stack = self.model.instrument.positioning_stack.name
        self.new_stack = stack_name

        self.setText('Changed Positioning Stack to {}'.format(stack_name))

    def redo(self):
        self.model.instrument.loadPositioningStack(self.new_stack)
        self.model.notifyChange(Attributes.Instrument)
        self.model.instrument_controlled.emit(self.id())

    def undo(self):
        self.model.instrument.loadPositioningStack(self.old_stack)
        stack = self.model.instrument.positioning_stack
        for base, aux in zip(self.bases, stack.auxiliary):
            aux.base = base

        for s, l in zip(self.link_state, stack.links):
            l.locked = s[0]
            l.ignore_limits = s[1]

        stack.fkine(self.old_q, True)
        self.model.notifyChange(Attributes.Instrument)
        self.model.instrument_controlled.emit(self.id())

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.ChangePositioningStack


class ChangePositionerBase(QtWidgets.QUndoCommand):
    """Changes the base matrix of an auxiliary positioner

    :param positioner: auxiliary positioner
    :type positioner: SerialManipulator
    :param matrix: new base matrix
    :type matrix: Matrix44
    :param presenter: Mainwindow presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, positioner, matrix, presenter):
        super().__init__()

        self.model = presenter.model
        self.aux = positioner
        self.old_matrix = positioner.base
        self.new_matrix = matrix

        self.setText('Changed Base Matrix of {}'.format(positioner.name))

    def redo(self):
        self.changeBase(self.new_matrix)

    def undo(self):
        self.changeBase(self.old_matrix)

    def changeBase(self, matrix):
        self.model.instrument.positioning_stack.changeBaseMatrix(self.aux, matrix)
        self.model.notifyChange(Attributes.Instrument)
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        """Merges consecutive change main commands

        :param command: command to merge
        :type command: QUndoCommand
        :return: True if merge was successful
        :rtype: bool
        """
        if self.aux is not command.aux:
            return False

        if self.old_matrix is command.new_matrix:
            self.setObsolete(True)

        self.new_matrix = command.new_matrix
        return True

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.ChangePositionerBase


class ChangeJawAperture(QtWidgets.QUndoCommand):
    """Sets the Jaws aperture

    :param aperture: new aperture
    :type aperture: List[float]
    :param presenter: Mainwindow presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, aperture, presenter):
        super().__init__()

        self.model = presenter.model
        jaws = self.model.instrument.jaws
        self.old_aperture = jaws.aperture.copy()
        self.new_aperture = aperture

        self.setText(f'Changed {jaws.name} Aperture')

    def redo(self):
        self.changeAperture(self.new_aperture)

    def undo(self):
        self.changeAperture(self.old_aperture)

    def changeAperture(self, aperture):
        self.model.instrument.jaws.aperture[0] = aperture[0]
        self.model.instrument.jaws.aperture[1] = aperture[1]
        self.model.notifyChange(Attributes.Instrument)
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        """Merges consecutive change main commands

        :param command: command to merge
        :type command: QUndoCommand
        :return: True if merge was successful
        :rtype: bool
        """
        if self.old_aperture == command.new_aperture:
            self.setObsolete(True)

        self.new_aperture = command.new_aperture
        return True

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.ChangeJawAperture


class ChangeCollimator(QtWidgets.QUndoCommand):
    """Changes the collimator of a given detector

    :param detector_name: name of detector
    :type detector_name: str
    :param collimator_name: new collimator name
    :type collimator_name: Union[str, None]
    :param presenter: Mainwindow presenter instance
    :type presenter: MainWindowPresenter
    """
    def __init__(self, detector_name, collimator_name, presenter):
        super().__init__()

        self.model = presenter.model
        self.detector_name = detector_name
        detector = self.model.instrument.detectors[self.detector_name]
        collimator = detector.current_collimator
        self.old_collimator_name = None if collimator is None else collimator.name
        self.new_collimator_name = collimator_name
        self.action_group = presenter.view.collimator_action_groups[detector_name]
        presenter.view.scenes.switchToInstrumentScene()

        self.setText(f"Changed {detector_name} Detector's Collimator to {collimator_name}")

    def redo(self):
        self.changeCollimator(self.new_collimator_name)

    def undo(self):
        self.changeCollimator(self.old_collimator_name)

    def changeCollimator(self, collimator_name):
        detector = self.model.instrument.detectors[self.detector_name]
        detector.current_collimator = collimator_name
        self.model.notifyChange(Attributes.Instrument)
        toggleActionInGroup(collimator_name, self.action_group)

    def mergeWith(self, command):
        """Merges consecutive change main commands

        :param command: command to merge
        :type command: QUndoCommand
        :return: True if merge was successful
        :rtype: bool
        """
        if self.detector_name != command.detector_name:
            return False

        if self.old_collimator_name == command.new_collimator_name:
            self.setObsolete(True)

        self.new_collimator_name = command.new_collimator_name
        self.setText(f"Changed {self.detector_name} Detector's Collimator to {self.new_collimator_name}")

        return True

    def id(self):
        """Returns ID used for notifying of or merging commands"""
        return CommandID.ChangeCollimator
