from PyQt5 import QtCore, QtWidgets
from sscanss.core.util import DockFlag
from sscanss.ui.dialogs import (InsertPrimitiveDialog, SampleManager, TransformDialog, SimulationDialog,
                                InsertPointDialog, PointManager, InsertVectorDialog, AlignSample,
                                VectorManager, PickPointDialog, JawControl, PositionerControl, DetectorControl)


class Dock(QtWidgets.QDockWidget):
    def __init__(self, parent):
        super().__init__(parent)

    def closeWidget(self):
        widget = self.widget()
        if widget and not widget.close():
            return False
        return True

    def closeEvent(self, event):
        if not self.closeWidget():
            event.ignore()
            return
        event.accept()


class DockManager(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.createDockWindows()

    def createDockWindows(self):
        self.upper_dock = Dock(self.parent)
        self.upper_dock.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.upper_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.upper_dock)
        self.upper_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.upper_dock.setVisible(False)

        self.bottom_dock = Dock(self.parent)
        self.bottom_dock.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.bottom_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.bottom_dock)
        self.bottom_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.bottom_dock.setVisible(False)

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

    def showDock(self, dock_flag):
        if dock_flag == DockFlag.Upper:
            self.upper_dock.show()
        elif dock_flag == DockFlag.Bottom:
            upper = self.upper_dock.widget()
            if upper and upper.dock_flag == DockFlag.Full:
                self.upper_dock.close()
            self.bottom_dock.show()
        elif dock_flag == DockFlag.Full:
            self.upper_dock.show()
            self.bottom_dock.close()

    def __showDockHelper(self, widget_class, params=None, attr_name=None, attr_value=None):
        if not self.isWidgetDocked(widget_class, attr_name, attr_value):
            _params = [] if params is None else params
            # Guarantees previous widget is close before new is created
            dock = self.bottom_dock if widget_class.dock_flag == DockFlag.Bottom else self.upper_dock
            dock.closeWidget()
            widget = widget_class(*_params, self.parent)
            widget.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            dock.setWindowTitle(widget.title)
            dock.setWidget(widget)
        self.showDock(widget_class.dock_flag)

    def showInsertPointDialog(self, point_type):
        self.__showDockHelper(InsertPointDialog, [point_type], 'point_type', point_type)

    def showInsertVectorDialog(self):
        self.__showDockHelper(InsertVectorDialog)

    def showInsertPrimitiveDialog(self, primitive):
        self.__showDockHelper(InsertPrimitiveDialog, [primitive], 'primitive', primitive)

    def showPointManager(self, point_type):
        self.__showDockHelper(PointManager, [point_type], 'point_type', point_type)

    def showVectorManager(self):
        self.__showDockHelper(VectorManager)

    def showSampleManager(self):
        self.__showDockHelper(SampleManager)

    def showTransformDialog(self, transform_type):
        self.__showDockHelper(TransformDialog, [transform_type], 'type', transform_type)

    def showPickPointDialog(self):
        self.__showDockHelper(PickPointDialog)

    def showJawControl(self):
        self.__showDockHelper(JawControl)

    def showDetectorControl(self, detector):
        self.__showDockHelper(DetectorControl, [detector], 'name', detector)

    def showPositionerControl(self):
        self.__showDockHelper(PositionerControl)

    def showAlignSample(self):
        self.__showDockHelper(AlignSample)

    def showSimulationResults(self):
        self.__showDockHelper(SimulationDialog, [], 'simulation', self.parent.presenter.model.simulation)

    def closeAll(self):
        self.upper_dock.close()
        self.bottom_dock.close()
