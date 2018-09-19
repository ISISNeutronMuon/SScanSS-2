from PyQt5 import QtCore, QtWidgets
from sscanss.core.util import DockFlag
from sscanss.ui.dialogs import (InsertPrimitiveDialog, SampleManager, TransformDialog,
                                InsertPointDialog, PointManager, InsertVectorDialog,
                                VectorManager, PickPointDialog)


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
        self.upper_dock.visibilityChanged.connect(lambda v: None if v else self.closeWidget(self.upper_dock))

        self.bottom_dock = QtWidgets.QDockWidget(self.parent)
        self.bottom_dock.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.bottom_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.bottom_dock)
        self.bottom_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.bottom_dock.setVisible(False)
        self.bottom_dock.visibilityChanged.connect(lambda v: None if v else self.closeWidget(self.bottom_dock))

    def addWidget(self, widget, dock):
        dock.setWindowTitle(widget.title)
        self.closeWidget(dock)
        dock.setWidget(widget)
        dock.show()

    def closeWidget(self, dock):
        widget = dock.widget()
        if not widget:
            return
        widget.hide()
        widget.deleteLater()

    def isWidgetDocked(self, widget_type, attr=None, value=None):
        if widget_type.dock_flag == DockFlag.Bottom:
            widget = self.bottom_dock.widget()
            found = isinstance(widget, widget_type)
        else:
            widget = self.upper_dock.widget()
            found = isinstance(widget, widget_type)

        if not found or attr is None or value is None:
            return found

        if getattr(widget, attr) == value:
            return True
        else:
            return False

    def showDockWidget(self, widget):
        if widget.dock_flag == DockFlag.Upper:
            self.addWidget(widget, self.upper_dock)
        elif widget.dock_flag == DockFlag.Bottom:
            upper = self.upper_dock.widget()
            if upper and upper.dock_flag == DockFlag.Full:
                self.upper_dock.setVisible(False)
            self.addWidget(widget, self.bottom_dock)
        elif widget.dock_flag == DockFlag.Full:
            self.addWidget(widget, self.upper_dock)
            self.bottom_dock.setVisible(False)

    def showInsertPointDialog(self, point_type):
        if not self.isWidgetDocked(InsertPointDialog, 'point_type', point_type):
            widget = InsertPointDialog(point_type, self.parent)
            self.showDockWidget(widget)

    def showInsertVectorDialog(self):
        if not self.isWidgetDocked(InsertVectorDialog):
            widget = InsertVectorDialog(self.parent)
            self.showDockWidget(widget)

    def showInsertPrimitiveDialog(self, primitive):
        if not self.isWidgetDocked(InsertPrimitiveDialog, 'primitive', primitive):
            widget = InsertPrimitiveDialog(primitive, self.parent)
            self.showDockWidget(widget)

    def showPointManager(self, point_type):
        if not self.isWidgetDocked(PointManager, 'point_type', point_type):
            widget = PointManager(point_type, self.parent)
            self.showDockWidget(widget)

    def showVectorManager(self):
        if not self.isWidgetDocked(VectorManager):
            widget = VectorManager(self.parent)
            self.showDockWidget(widget)

    def showSampleManager(self):
        if not self.isWidgetDocked(SampleManager):
            widget = SampleManager(self.parent)
            self.showDockWidget(widget)

    def showTransformDialog(self, transform_type):
        if not self.isWidgetDocked(TransformDialog, 'type', transform_type):
            widget = TransformDialog(transform_type, self.parent)
            self.showDockWidget(widget)

    def showPickPointDialog(self):
        if not self.isWidgetDocked(PickPointDialog):
            widget = PickPointDialog(self.parent)
            self.showDockWidget(widget)
