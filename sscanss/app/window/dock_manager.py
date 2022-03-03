from PyQt5 import QtCore, QtWidgets
from sscanss.core.util import DockFlag
from sscanss.app.dialogs import (InsertPrimitiveDialog, TransformDialog, SimulationDialog, PickPointDialog,
                                 InsertPointDialog, PointManager, InsertVectorDialog, AlignSample, VectorManager,
                                 JawControl, PositionerControl, DetectorControl)


class Dock(QtWidgets.QDockWidget):
    """Custom QDockWidget that closes contained widgets when it is closed.

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

    def closeWidget(self):
        """Calls close function of contained widget"""
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
    """"Manages upper and bottom docks.

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.createDockWindows()

    def createDockWindows(self):
        """Creates upper and bottom dock widgets"""
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

        # Fix dock widget snap https://bugreports.qt.io/browse/QTBUG-65592
        self.parent.resizeDocks((self.upper_dock, self.bottom_dock), (200, 200), QtCore.Qt.Horizontal)

    def isWidgetDocked(self, widget_class, attr_name=None, attr_value=None):
        """Checks if a widget of specified class that contains desired attribute value is
        docked in the upper or bottom dock. This is used to avoid recreating a widget if
        it already exists.

        :param widget_class class of widget
        :type widget_class: class
        :param attr_name: attribute name
        :type attr_name: Union(str, None)
        :param attr_value: attribute value
        :type attr_value: Union(Any, None)
        :return: indicate if widget is docked
        :rtype: bool
        """
        if widget_class.dock_flag == DockFlag.Bottom:
            widget = self.bottom_dock.widget()
        else:
            widget = self.upper_dock.widget()
        found = isinstance(widget, widget_class)

        if not found or attr_name is None or attr_value is None:
            return found

        return getattr(widget, attr_name) == attr_value

    def showDock(self, dock_flag):
        """Shows widget in full, upper or bottom dock in accordance with
        specified flag

        :param dock_flag: flag indicates how dock should be shown
        :type dock_flag: DockFlag
        """
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
        """Creates widget of specified class if it does not exist then shows widget in the
        appropriate dock.

        :param widget_class: class of widget
        :type widget_class: class
        :param params: parameters for init of class
        :type params: Union(Tuple[Any, ...], None)
        :param attr_name: attribute name
        :type attr_name: Union(str, None)
        :param attr_value: attribute value
        :type attr_value: Union(Any, None)
        """
        if not self.isWidgetDocked(widget_class, attr_name, attr_value):
            _params = [] if params is None else params
            # Guarantees previous widget is close before new is created
            dock = self.bottom_dock if widget_class.dock_flag == DockFlag.Bottom else self.upper_dock
            if not dock.closeWidget():
                return
            widget = widget_class(*_params, self.parent)
            widget.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            dock.setWindowTitle(widget.title)
            dock.setWidget(widget)
        self.showDock(widget_class.dock_flag)

    def showInsertPointDialog(self, point_type):
        """Opens the insert point dialog for the specified point type"""
        self.__showDockHelper(InsertPointDialog, [point_type], 'point_type', point_type)

    def showInsertVectorDialog(self):
        """Opens the insert measurement vector dialog"""
        self.__showDockHelper(InsertVectorDialog)

    def showInsertPrimitiveDialog(self, primitive):
        """Opens the insert primitive dialog for the specified type of primitive"""
        self.__showDockHelper(InsertPrimitiveDialog, [primitive], 'primitive', primitive)

    def showPointManager(self, point_type):
        """Opens the point manager dialog for the specified point type"""
        self.__showDockHelper(PointManager, [point_type], 'point_type', point_type)

    def showVectorManager(self):
        """Opens the measurement vector manager dialog"""
        self.__showDockHelper(VectorManager)

    def showTransformDialog(self, transform_type):
        """Opens the transform dialog for the specified type of transform tool"""
        self.__showDockHelper(TransformDialog, [transform_type], 'type', transform_type)

    def showPickPointDialog(self):
        """Opens the measurement point picking dialog"""
        self.__showDockHelper(PickPointDialog)

    def showJawControl(self):
        """Opens the jaw control dialog"""
        self.__showDockHelper(JawControl)

    def showDetectorControl(self, detector):
        """Opens the detector control dialog"""
        self.__showDockHelper(DetectorControl, [detector], 'name', detector)

    def showPositionerControl(self):
        """Opens the positioning system control dialog"""
        self.__showDockHelper(PositionerControl)

    def showAlignSample(self):
        """Opens the 6D pose alignment dialog"""
        self.__showDockHelper(AlignSample)

    def showSimulationResults(self):
        """Opens the simulation result dialog"""
        self.__showDockHelper(SimulationDialog, [], 'simulation', self.parent.presenter.model.simulation)

    def closeAll(self):
        """Closes upper and bottom dock"""
        self.upper_dock.close()
        self.bottom_dock.close()
