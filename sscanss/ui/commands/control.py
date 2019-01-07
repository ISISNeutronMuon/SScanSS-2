from PyQt5 import QtWidgets
from sscanss.core.util import CommandID, toggleActionInGroup


class LockJoint(QtWidgets.QUndoCommand):
    def __init__(self, positioner_name, index, value, presenter):
        super().__init__()

        self.model = presenter.model

        self.positioner_name = positioner_name
        stack = self.model.instrument.getPositioner(self.positioner_name)
        self.old_lock_state = [l.locked for l in stack.links]
        self.new_lock_state = self.old_lock_state.copy()
        self.new_lock_state[index] = value

        self.setText(f'Locked Joint in {positioner_name}')

    def redo(self):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        for state, link in zip(self.new_lock_state, stack.links):
            link.locked = state
        self.model.instrument_controlled.emit(self.id())

    def undo(self):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        for state, link in zip(self.old_lock_state, stack.links):
            link.locked = state
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        if self.positioner_name != command.positioner_name:
            return False

        if self.old_lock_state == command.new_lock_state:
            self.setObsolete(True)

        self.new_lock_state = command.new_lock_state
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.LockJoint


class IgnoreJointLimits(QtWidgets.QUndoCommand):
    def __init__(self, positioner_name, index, value, presenter):
        super().__init__()

        self.model = presenter.model

        self.positioner_name = positioner_name
        stack = self.model.instrument.getPositioner(self.positioner_name)
        self.old_ignore_state = [l.ignore_limits for l in stack.links]
        self.new_ignore_state = self.old_ignore_state.copy()
        self.new_ignore_state[index] = value

        self.setText(f'Ignored Joint Limits in {positioner_name}')

    def redo(self):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        for state, link in zip(self.new_ignore_state, stack.links):
            link.ignore_limits = state
        self.model.instrument_controlled.emit(self.id())

    def undo(self):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        for state, link in zip(self.old_ignore_state, stack.links):
            link.ignore_limits = state
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        if self.positioner_name != command.positioner_name:
            return False

        if self.old_ignore_state == command.new_ignore_state:
            self.setObsolete(True)

        self.new_ignore_state = command.new_ignore_state
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.IgnoreJointLimits


class MovePositioner(QtWidgets.QUndoCommand):
    def __init__(self, positioner_name, q, presenter):
        super().__init__()

        self.model = presenter.model

        self.positioner_name = positioner_name
        stack = self.model.instrument.getPositioner(self.positioner_name)
        self.move_from = stack.set_points
        self.move_to = q

        self.animate = True

        self.setText(f'Moved {positioner_name}')

    def redo(self):
        stack = self.model.instrument.getPositioner(self.positioner_name)
        if self.animate:
            stack.set_points = self.move_to
            self.model.animateInstrument(lambda q, s=stack: s.fkine(q, setpoint=False),
                                         self.move_from, self.move_to, 500, 10)
            self.animate = False
        else:
            stack.fkine(self.move_to)
            self.model.updateInstrumentScene()
        self.model.instrument_controlled.emit(self.id())

    def undo(self):
        if self.model.sequence.isRunning():
            self.model.sequence.stop()
        stack = self.model.instrument.getPositioner(self.positioner_name)
        stack.set_point = self.move_from
        stack.fkine(self.move_from)
        self.model.updateInstrumentScene()
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        if self.positioner_name != command.positioner_name:
            return False

        if self.move_from == command.move_to:
            self.setObsolete(True)

        self.move_to = command.move_to
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.MovePositioner


class ChangePositioningStack(QtWidgets.QUndoCommand):
    def __init__(self, stack_name, presenter):
        super().__init__()

        self.model = presenter.model

        stack = self.model.instrument.positioning_stack
        self.old_q = stack.set_points
        self.link_state = [(l.locked, l.ignore_limits) for l in stack.links]
        self.bases = [aux.base for aux in stack.auxiliary]

        self.old_stack = self.model.instrument.positioning_stack.name
        self.new_stack = stack_name

        self.setText('Changed Positioning Stack to {}'.format(stack_name))

    def redo(self):
        self.model.instrument.loadPositioningStack(self.new_stack)
        self.model.updateInstrumentScene()
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
        self.model.updateInstrumentScene()
        self.model.instrument_controlled.emit(self.id())

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.ChangePositioningStack


class ChangePositionerBase(QtWidgets.QUndoCommand):
    def __init__(self, positioner, matrix, presenter):
        super().__init__()

        self.model = presenter.model
        self.aux = positioner
        self.old_matrix = positioner.base
        self.new_matrix = matrix

        self.setText('Changed Base Matrix of {}'.format(positioner.name))

    def redo(self):
        self.model.instrument.positioning_stack.changeBaseMatrix(self.aux, self.new_matrix)
        self.model.updateInstrumentScene()
        self.model.instrument_controlled.emit(self.id())

    def undo(self):
        self.model.instrument.positioning_stack.changeBaseMatrix(self.aux, self.old_matrix)
        self.model.updateInstrumentScene()
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        if self.aux is not command.aux:
            return False

        if self.old_matrix is command.new_matrix:
            self.setObsolete(True)

        self.new_matrix = command.new_matrix
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.ChangePositionerBase


class ChangeJawAperture(QtWidgets.QUndoCommand):
    def __init__(self, aperture, presenter):
        super().__init__()

        self.model = presenter.model
        jaws = self.model.instrument.jaws
        self.old_aperture = jaws.aperture.copy()
        self.new_aperture = aperture

        self.setText(f'Changed {jaws.name} Aperture')

    def redo(self):
        self.model.instrument.jaws.aperture[0] = self.new_aperture[0]
        self.model.instrument.jaws.aperture[1] = self.new_aperture[1]
        self.model.instrument_controlled.emit(self.id())

    def undo(self):
        self.model.instrument.jaws.aperture[0] = self.old_aperture[0]
        self.model.instrument.jaws.aperture[1] = self.old_aperture[1]
        self.model.instrument_controlled.emit(self.id())

    def mergeWith(self, command):
        if self.old_aperture == command.new_aperture:
            self.setObsolete(True)

        self.new_aperture = command.new_aperture
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.ChangeJawAperture


class ChangeCollimator(QtWidgets.QUndoCommand):
    def __init__(self, detector_name, collimator_name, presenter):
        super().__init__()

        self.model = presenter.model
        self.detector_name = detector_name
        detector = self.model.instrument.detectors[self.detector_name]
        self.old_collimator_name = detector.current_collimator.name
        self.new_collimator_name = collimator_name
        self.action_group = presenter.view.collimator_action_groups[detector_name]
        self.model.switchSceneTo(self.model.instrument_scene)

        self.setText(f"Changed {detector_name} Detector's Collimator to {collimator_name}")

    def redo(self):
        detector = self.model.instrument.detectors[self.detector_name]
        detector.current_collimator = self.new_collimator_name
        self.model.updateInstrumentScene()
        toggleActionInGroup(self.new_collimator_name, self.action_group)

    def undo(self):
        detector = self.model.instrument.detectors[self.detector_name]
        detector.current_collimator = self.old_collimator_name
        self.model.updateInstrumentScene()
        toggleActionInGroup(self.old_collimator_name, self.action_group)

    def mergeWith(self, command):
        if self.detector_name != command.detector_name:
            return False

        if self.old_collimator_name == command.new_collimator_name:
            self.setObsolete(True)

        self.new_collimator_name = command.new_collimator_name
        self.setText(f"Changed {self.detector_name} Detector's Collimator to {self.new_collimator_name}")

        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.ChangeCollimator
