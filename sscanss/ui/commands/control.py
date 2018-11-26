from PyQt5 import QtWidgets


class LockJoint(QtWidgets.QUndoCommand):
    def __init__(self, index, value, presenter):
        super().__init__()

        self.model = presenter.model
        self.stack = self.model.active_instrument.positioning_stack

        self.old_lock_state = [l.locked for l in self.stack.links]
        self.values = [l.locked for l in self.stack.links]
        self.values[index] = value

        self.setText('Locked Joint in Positioning Stack')

    def redo(self):
        for value, link in zip(self.values, self.stack.links):
            link.locked = value
        self.model.positioner_updated.emit()

    def undo(self):
        for value, link in zip(self.old_lock_state, self.stack.links):
            link.locked = value
        self.model.positioner_updated.emit()

    def mergeWith(self, command):
        self.values = command.values

        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return 1003


class IgnoreJointLimits(QtWidgets.QUndoCommand):
    def __init__(self, index, value, presenter):
        super().__init__()

        self.model = presenter.model
        self.stack = self.model.active_instrument.positioning_stack

        self.old_lock_state = [l.ignore_limits for l in self.stack.links]
        self.values = [l.ignore_limits for l in self.stack.links]
        self.values[index] = value

        self.setText('Ignored Joint Limits in Positioning Stack')

    def redo(self):
        for value, link in zip(self.values, self.stack.links):
            link.ignore_limits = value
        self.model.positioner_updated.emit()

    def undo(self):
        for value, link in zip(self.old_lock_state, self.stack.links):
            link.ignore_limits = value
        self.model.positioner_updated.emit()

    def mergeWith(self, command):
        self.values = command.values

        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return 1004


class MovePositioner(QtWidgets.QUndoCommand):
    def __init__(self, q, presenter):
        super().__init__()

        self.model = presenter.model
        self.stack = self.model.active_instrument.positioning_stack

        self.move_from = self.stack.configuration
        self.move_to = q

        self.animate = True

        self.setText('Moved Positioning Stack')

    def redo(self):
        if self.animate:
            self.model.animateInstrument(self.stack.fkine, self.move_from, self.move_to, 500, 10)
            self.animate = False
        else:
            self.stack.fkine(self.move_to)
            self.model.updateInstrumentScene()

    def undo(self):
        self.stack.fkine(self.move_from)
        self.model.updateInstrumentScene()

    def mergeWith(self, command):
        self.move_to = command.move_to

        return True

    def id(self):
        """ Returns ID used when merging commands"""
        return 1005
