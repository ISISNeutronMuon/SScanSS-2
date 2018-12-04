from PyQt5 import QtWidgets
from sscanss.core.util import CommandID


class LockJoint(QtWidgets.QUndoCommand):
    def __init__(self, index, value, presenter):
        super().__init__()

        self.model = presenter.model

        stack = self.model.active_instrument.positioning_stack
        self.old_lock_state = [l.locked for l in stack.links]
        self.new_lock_state = self.old_lock_state.copy()
        self.new_lock_state[index] = value

        self.setText('Locked Joint in Positioning Stack')

    def redo(self):
        stack = self.model.active_instrument.positioning_stack
        for state, link in zip(self.new_lock_state, stack.links):
            link.locked = state
        self.model.positioner_updated.emit(self.id())

    def undo(self):
        stack = self.model.active_instrument.positioning_stack
        for state, link in zip(self.old_lock_state, stack.links):
            link.locked = state
        self.model.positioner_updated.emit(self.id())

    def mergeWith(self, command):
        if self.old_lock_state == command.new_lock_state:
            self.setObsolete(True)

        self.new_lock_state = command.new_lock_state
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.LockJoint.value


class IgnoreJointLimits(QtWidgets.QUndoCommand):
    def __init__(self, index, value, presenter):
        super().__init__()

        self.model = presenter.model

        stack = self.model.active_instrument.positioning_stack
        self.old_ignore_state = [l.ignore_limits for l in stack.links]
        self.new_ignore_state = self.old_ignore_state.copy()
        self.new_ignore_state[index] = value

        self.setText('Ignored Joint Limits in Positioning Stack')

    def redo(self):
        stack = self.model.active_instrument.positioning_stack
        for state, link in zip(self.new_ignore_state, stack.links):
            link.ignore_limits = state
        self.model.positioner_updated.emit(self.id())

    def undo(self):
        stack = self.model.active_instrument.positioning_stack
        for state, link in zip(self.old_ignore_state, stack.links):
            link.ignore_limits = state
        self.model.positioner_updated.emit(self.id())

    def mergeWith(self, command):
        if self.old_ignore_state == command.new_ignore_state:
            self.setObsolete(True)

        self.new_ignore_state = command.new_ignore_state
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.IgnoreJointLimits.value


class MovePositioner(QtWidgets.QUndoCommand):
    def __init__(self, q, presenter):
        super().__init__()

        self.model = presenter.model

        stack = self.model.active_instrument.positioning_stack
        self.move_from = stack.set_points
        self.move_to = q

        self.animate = True

        self.setText('Moved Positioning Stack')

    def redo(self):
        stack = self.model.active_instrument.positioning_stack
        if self.animate:
            stack.set_points = self.move_to
            self.model.animateInstrument(self.change_frame, self.move_from, self.move_to, 500, 10)
            self.animate = False
        else:
            stack.fkine(self.move_to)
            self.model.updateInstrumentScene()
        self.model.positioner_updated.emit(self.id())

    def undo(self):
        if self.model.sequence.isRunning():
            self.model.sequence.stop()
        stack = self.model.active_instrument.positioning_stack
        stack.set_point = self.move_from
        stack.fkine(self.move_from)
        self.model.updateInstrumentScene()
        self.model.positioner_updated.emit(self.id())

    def mergeWith(self, command):
        if self.move_from == command.move_to:
            self.setObsolete(True)

        self.move_to = command.move_to
        return True

    def change_frame(self, q):
        self.model.active_instrument.positioning_stack.fkine(q, setpoint=False)

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.MovePositioner.value


class ChangePositioningStack(QtWidgets.QUndoCommand):
    def __init__(self, stack_name, presenter):
        super().__init__()

        self.model = presenter.model

        stack = self.model.active_instrument.positioning_stack
        self.old_q = stack.set_points
        self.link_state = [(l.locked, l.ignore_limits) for l in stack.links]
        self.bases = [aux.base for aux in stack.auxiliary]

        self.old_stack = self.model.active_instrument.positioning_stack.name
        self.new_stack = stack_name

        self.setText('Changed Positioning Stack to {}'.format(stack_name))

    def redo(self):
        self.model.active_instrument.loadPositioningStack(self.new_stack)
        self.model.updateInstrumentScene()
        self.model.positioner_updated.emit(self.id())

    def undo(self):
        self.model.active_instrument.loadPositioningStack(self.old_stack)
        stack = self.model.active_instrument.positioning_stack
        for base, aux in zip(self.bases, stack.auxiliary):
            aux.base = base

        for s, l in zip(self.link_state, stack.links):
            l.locked = s[0]
            l.ignore_limits = s[1]

        stack.fkine(self.old_q, True)
        self.model.updateInstrumentScene()
        self.model.positioner_updated.emit(self.id())

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.ChangePositioningStack.value


class ChangePositionerBase(QtWidgets.QUndoCommand):
    def __init__(self, positioner, matrix, presenter):
        super().__init__()

        self.model = presenter.model
        self.aux = positioner
        self.old_matrix = positioner.base
        self.new_matrix = matrix

        self.setText('Changed Base Matrix of {}'.format(positioner.name))

    def redo(self):
        self.model.active_instrument.positioning_stack.changeBaseMatrix(self.aux, self.new_matrix)
        self.model.updateInstrumentScene()
        self.model.positioner_updated.emit(self.id())

    def undo(self):
        self.model.active_instrument.positioning_stack.changeBaseMatrix(self.aux, self.old_matrix)
        self.model.updateInstrumentScene()
        self.model.positioner_updated.emit(self.id())

    def mergeWith(self, command):
        if self.old_matrix is command.new_matrix:
            self.setObsolete(True)

        self.new_matrix = command.new_matrix
        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return CommandID.ChangePositionerBase.value
