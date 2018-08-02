from PyQt5 import QtWidgets


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
        self.toggleRenderActions(self.new_mode)

    def undo(self):
        self.gl_widget.sampleRenderMode = self.old_mode
        self.toggleRenderActions(self.old_mode)

    def toggleRenderActions(self, render_mode):
        actions = self.action_group.actions()
        for action in actions:
            if action.text() == render_mode.value:
                action.setChecked(True)
