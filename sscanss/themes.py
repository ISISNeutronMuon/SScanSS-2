from contextlib import suppress
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from sscanss.config import settings, SOURCE_PATH

STATIC_PATH = SOURCE_PATH / 'static'
IMAGES_PATH = STATIC_PATH / 'images'


def path_for(filename):
    """Gets full path for the given image file

    :param filename: basename and extension of image
    :type filename: str
    :return: full path of image
    :rtype: str
    """
    if settings.value(settings.Key.Theme) == settings.DefaultThemes.Light.value:
        return (IMAGES_PATH / settings.DefaultThemes.Light.value / filename).as_posix()
    return (IMAGES_PATH / settings.DefaultThemes.Dark.value / filename).as_posix()


class IconEngine(QtGui.QIconEngine):
    """Creates the icons for the application

    :param file_name: the icon file
    :type file_name: str
    """
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name
        self.theme = settings.value(settings.Key.Theme)
        self.path = path_for(self.file_name)
        self.icon = QtGui.QIcon(self.path)

    def updateIcon(self):
        """Updates the Icon"""
        if self.theme != settings.value(settings.Key.Theme):
            self.path = path_for(self.file_name)
            self.icon = QtGui.QIcon(self.path)
            self.theme = settings.value(settings.Key.Theme)

    def pixmap(self, size, mode, state):
        """Creates the pixmap

        :param size: size
        :type size: QSize
        :param mode: mode
        :type mode: QIcon.Mode
        :param state: state
        :type state: QIcon.State
        """
        self.updateIcon()
        return self.icon.pixmap(size, mode, state)

    def paint(self, painter, rect, mode, state):
        """Paints the icon

        :param painter: painter
        :type painter: QPainter
        :param rect: rect
        :type rect: QRect
        :param mode: mode
        :type mode: QIcon.Mode
        :param state: state
        :type state: QIcon.State
        """
        self.updateIcon()
        return self.icon.pixmap.paint(painter, rect, mode, state)


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
        self.loadCurrentStyle()

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

    @staticmethod
    def loadStylesheet(name):
        """Loads qt stylesheet from file

        :param name: path to stylesheet file
        :type name: str
        :return: stylesheet
        :rtype: str
        """
        with suppress(FileNotFoundError):
            with open(STATIC_PATH / name, 'rt') as stylesheet:
                image_path = (IMAGES_PATH / settings.value(settings.Key.Theme)).as_posix()
                style = stylesheet.read().replace('@Path', image_path)
            return style
        return ''

    def loadCurrentStyle(self):
        """Loads the stylesheet for the last active theme"""
        self.reset()
        if settings.value(settings.Key.Theme) == settings.DefaultThemes.Light.value:
            if sys.platform == 'darwin':
                style = self.loadStylesheet("mac_style.css")
            else:
                style = self.loadStylesheet("style.css")
        else:
            style = self.loadStylesheet("dark_theme.css")
        self.parent.setStyleSheet(style)

    def toggleTheme(self):
        """Toggles the stylesheet of the app"""
        self.reset()
        if settings.value(settings.Key.Theme) == settings.DefaultThemes.Light.value:
            settings.system.setValue(settings.Key.Theme.value, settings.DefaultThemes.Dark.value)
            style = self.loadStylesheet("dark_theme.css")
        else:
            settings.system.setValue(settings.Key.Theme.value, settings.DefaultThemes.Light.value)
            if sys.platform == 'darwin':
                style = self.loadStylesheet("mac_style.css")
            else:
                style = self.loadStylesheet("style.css")
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
