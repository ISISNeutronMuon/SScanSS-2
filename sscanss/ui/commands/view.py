from PyQt5 import QtWidgets
from sscanss.core.util import toggleActionInGroup


class ToggleRenderMode(QtWidgets.QUndoCommand):

    def __init__(self, render_mode, view):
        super().__init__()
        self.gl_widget = view.gl_widget
        self.new_mode = render_mode
        self.old_mode = self.gl_widget.sampleRenderMode
        self.action_group = view.render_action_group

        self.setText('Show {}'.format(self.new_mode.value))

    def redo(self):
        self.gl_widget.sampleRenderMode = self.new_mode
        toggleActionInGroup(self.new_mode.value, self.action_group)

    def undo(self):
        self.gl_widget.sampleRenderMode = self.old_mode
        toggleActionInGroup(self.old_mode.value, self.action_group)
