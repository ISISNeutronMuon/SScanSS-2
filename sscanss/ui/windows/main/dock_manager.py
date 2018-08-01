from PyQt5 import QtCore, QtWidgets
from sscanss.core.util import DockFlag
from sscanss.ui.dialogs import (InsertPrimitiveDialog, SampleManager, TransformDialog,
                                InsertPointDialog, PointManager)


class DockManager:
    def __init__(self, parent):
        self.parent = parent
        self.createDockWindows()

    def createDockWindows(self):
        self.upper_dock = QtWidgets.QDockWidget(self.parent)
        self.upper_dock.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.upper_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.upper_dock)
        self.upper_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.upper_dock.setVisible(False)

        self.bottom_dock = QtWidgets.QDockWidget(self.parent)
        self.bottom_dock.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.bottom_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.bottom_dock)
        self.bottom_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.bottom_dock.setVisible(False)

    def addWidgetToDock(self, widget, dock):
        dock.setWindowTitle(widget.title)
        old_widget = dock.widget()
        if old_widget:
            old_widget.hide()
            old_widget.deleteLater()
        dock.setWidget(widget)
        dock.show()

    def showDockWidget(self, widget):
        if widget.dock_flag == DockFlag.Upper:
            self.addWidgetToDock(widget, self.upper_dock)
        elif widget.dock_flag == DockFlag.Bottom:
            self.addWidgetToDock(widget, self.bottom_dock)
        elif widget.dock_flag == DockFlag.Full:
            self.addWidgetToDock(widget, self.upper_dock)
            self.bottom_dock.setVisible(False)

    def showInsertPointDialog(self, point_type):
        widgets = InsertPointDialog(point_type, self.parent)
        self.showDockWidget(widgets)

    def showInsertPrimitiveDialog(self, primitive):
        widgets = InsertPrimitiveDialog(primitive, self.parent)
        self.showDockWidget(widgets)

    def showPointManager(self, point_type):
        widgets = PointManager(point_type, self.parent)
        self.showDockWidget(widgets)

    def showSampleManager(self):
        widgets = SampleManager(self.parent)
        self.showDockWidget(widgets)

    def showTransformDialog(self, transform_type):
        widgets = TransformDialog(transform_type, self.parent)
        self.showDockWidget(widgets)
