from PyQt5 import QtWidgets
from sscanss.core.util import RenderType


class ToggleRenderType(QtWidgets.QUndoCommand):

    def __init__(self, render_type, gl_widget, action_group):
        super().__init__()
        self.gl_widget = gl_widget
        self.new_type = render_type
        self.old_type = gl_widget.sampleRenderType
        self.action_group = action_group

        self.setText('Show {}'.format(self.new_type.value))

    def redo(self):
        self.gl_widget.sampleRenderType = self.new_type
        self.toggleRenderActions(self.new_type)

    def undo(self):
        self.gl_widget.sampleRenderType = self.old_type
        self.toggleRenderActions(self.old_type)

    def toggleRenderActions(self, render_type):
        actions = self.action_group.actions()
        for action in actions:
            if action.text() == render_type.value:
                action.setChecked(True)
