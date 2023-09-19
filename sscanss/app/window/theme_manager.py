import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from sscanss.config import settings, Key, Themes, load_stylesheet


class ThemeManager(QtWidgets.QWidget):
    """Manages light and dark themes using qt properties to get colours
    from stylesheet.

    :param parent: main window instance
    :type parent: MainWindow
    """
    theme_changed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.setObjectName('ColourPalette')
        self.loadStyleSheet()

    def reset(self):
        self._html_anchor = QtGui.QColor()
        self._scene_anchor = QtGui.QColor()
        self._scene_path = QtGui.QColor()
        self._scene_highlight = QtGui.QColor()
        self._scene_bounds = QtGui.QColor()
        self._scene_grid = QtGui.QColor()
        self._curve_face = QtGui.QColor()
        self._curve_line = QtGui.QColor()
        self._curve_label = QtGui.QColor()
        self._curve_plot = QtGui.QColor()

    def loadStyleSheet(self):
        """Loads the stylesheet for the last active theme"""
        self.reset()
        if settings.value(Key.Theme) == Themes.Light.value:
            if sys.platform == 'darwin':
                style = load_stylesheet("mac_style.css")
            else:
                style = load_stylesheet("style.css")
        else:
            style = load_stylesheet("dark_theme.css")
        self.parent.setStyleSheet(style)

    def toggleTheme(self):
        """Toggles the stylesheet of the app"""
        self.reset()
        if settings.value(Key.Theme) == Themes.Light.value:
            settings.system.setValue(Key.Theme.value, Themes.Dark.value)
            style = load_stylesheet("dark_theme.css")
        else:
            settings.system.setValue(Key.Theme.value, Themes.Light.value)
            if sys.platform == 'darwin':
                style = load_stylesheet("mac_style.css")
            else:
                style = load_stylesheet("style.css")
        self.parent.setStyleSheet(style)
        self.parent.updateImages()
        self.theme_changed.emit()

    def htmlAnchor(self):
        return self._html_anchor

    def setHtmlAnchor(self, value):
        self._html_anchor = value

    def scenePath(self):
        return self._scene_path

    def setScenePath(self, value):
        self._scene_path = value

    def sceneAnchor(self):
        return self._scene_anchor

    def setSceneAnchor(self, value):
        self._scene_anchor = value

    def sceneHighlight(self):
        return self._scene_highlight

    def setSceneHighlight(self, value):
        self._scene_highlight = value

    def sceneGrid(self):
        return self._scene_grid

    def setSceneGrid(self, value):
        self._scene_grid = value

    def sceneBounds(self):
        return self._scene_bounds

    def setSceneBounds(self, value):
        self._scene_bounds = value

    def curveFace(self):
        return self._curve_face

    def setCurveFace(self, value):
        self._curve_face = value

    def curveLabel(self):
        return self._curve_label

    def setCurveLabel(self, value):
        self._curve_label = value

    def curvePlot(self):
        return self._curve_plot

    def setCurvePlot(self, value):
        self._curve_plot = value

    def curveLine(self):
        return self._curve_line

    def setCurveLine(self, value):
        self._curve_line = value

    html_anchor = QtCore.pyqtProperty(QtGui.QColor, htmlAnchor, setHtmlAnchor)
    scene_anchor = QtCore.pyqtProperty(QtGui.QColor, sceneAnchor, setSceneAnchor)
    scene_path = QtCore.pyqtProperty(QtGui.QColor, scenePath, setScenePath)
    scene_highlight = QtCore.pyqtProperty(QtGui.QColor, sceneHighlight, setSceneHighlight)
    scene_bounds = QtCore.pyqtProperty(QtGui.QColor, sceneBounds, setSceneBounds)
    scene_grid = QtCore.pyqtProperty(QtGui.QColor, sceneGrid, setSceneGrid)
    curve_face = QtCore.pyqtProperty(QtGui.QColor, curveFace, setCurveFace)
    curve_line = QtCore.pyqtProperty(QtGui.QColor, curveLine, setCurveLine)
    curve_label = QtCore.pyqtProperty(QtGui.QColor, curveLabel, setCurveLabel)
    curve_plot = QtCore.pyqtProperty(QtGui.QColor, curvePlot, setCurvePlot)
