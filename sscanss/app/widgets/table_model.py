import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.config import path_for
from sscanss.core.util import to_float


class CenteredBoxProxy(QtWidgets.QProxyStyle):
    """Ensures checkbox is centred in the table cell"""
    def __init__(self):
        super().__init__()

    def subElementRect(self, element, option, widget):
        rect = super().subElementRect(element, option, widget)
        if element == QtWidgets.QStyle.SE_ItemViewItemCheckIndicator:
            if option.index.flags() & QtCore.Qt.ItemIsUserCheckable != QtCore.Qt.NoItemFlags:
                text_margin = widget.style().pixelMetric(QtWidgets.QStyle.PM_FocusFrameHMargin) + 1
                rect = QtWidgets.QStyle.alignedRect(
                    option.direction, QtCore.Qt.AlignCenter,
                    QtCore.QSize(option.decorationSize.width() + 5, option.decorationSize.height()),
                    QtCore.QRect(option.rect.x() + text_margin, option.rect.y(),
                                 option.rect.width() - (2 * text_margin), option.rect.height()))

        return rect


class LimitTextDelegate(QtWidgets.QItemDelegate):
    """Changes the maximum length of a table view's editor widget

    :param max_length: maximum length of text editor
    :type max_length: int
    """
    def __init__(self, max_length=12):
        super().__init__()
        self.max_length = max_length

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QLineEdit(parent)
        editor.setMaxLength(self.max_length)
        return editor


class PointModel(QtCore.QAbstractTableModel):
    """Provides model for showing fiducial or measurement points in table view

    :param array: fiducial or measurement point
    :type array: numpy.recarray
    """
    editCompleted = QtCore.pyqtSignal(object)

    def __init__(self, array):
        super().__init__()

        self._data = array
        self.header_icon = ''
        self.title = ['X (mm)', 'Y (mm)', 'Z (mm)']
        self.setHeaderIcon()

    def update(self, array):
        """Updates model's data and layout

        :param array: fiducial or measurement point
        :type array: numpy.recarray
        """
        self.layoutAboutToBeChanged.emit()
        self._data = array
        self.setHeaderIcon()
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, 3)
        self.dataChanged.emit(top_left, bottom_right)
        self.layoutChanged.emit()

    def rowCount(self, parent=None):
        return self._data.points.shape[0]

    def columnCount(self, _parent=None):
        return self._data.points.shape[1] + 1

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return QtCore.QVariant()

        value = '' if index.column() == 3 else f'{self._data.points[index.row(), index.column()]:.3f}'

        if role == QtCore.Qt.EditRole:
            return value
        elif role == QtCore.Qt.DisplayRole:
            return value
        elif role == QtCore.Qt.CheckStateRole:
            if index.column() == 3:
                if self._data.enabled[index.row()]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter

        return QtCore.QVariant()

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid():
            return False

        row = index.row()
        self._data = self._data
        if role == QtCore.Qt.CheckStateRole and index.column() == 3:
            self._data.enabled[row] = True if value == QtCore.Qt.Checked else False
            self.editCompleted.emit(self._data)
            self.setHeaderIcon()

        elif role == QtCore.Qt.EditRole and index.column() != 3:
            col = index.column()
            value, value_is_float = to_float(value)

            if value_is_float and f'{value:.3f}' != f'{self._data.points[row, col]:.3f}':
                self._data.points[row, col] = value
                self.editCompleted.emit(self._data)
        else:
            return False

        return True

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags

        if index.column() == 3:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable

    def headerData(self, index, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DecorationRole:
            if index == 3:
                pixmap = QtGui.QPixmap(self.header_icon)
                pixmap = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                return QtCore.QVariant(pixmap)
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if index != 3:
                return QtCore.QVariant(self.title[index])

        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(index + 1)

        return QtCore.QVariant()

    def toggleCheckState(self, index):
        """Updates checked state of points when the header is clicked

        :param index: column index
        :type index: int
        """
        if index == 3 and self.rowCount() > 0:
            if np.all(self._data.enabled):
                self._data.enabled.fill(False)
            else:
                self._data.enabled.fill(True)

            self.editCompleted.emit(self._data)
            top_left = self.index(0, 3)
            bottom_right = self.index(self.rowCount() - 1, 3)
            self.dataChanged.emit(top_left, bottom_right)
            self.setHeaderIcon()

    def setHeaderIcon(self):
        """Updates header icons to match data enabled state"""
        if np.all(self._data.enabled):
            self.header_icon = path_for('checked.png')
        elif np.any(self._data.enabled):
            self.header_icon = path_for('intermediate.png')
        else:
            self.header_icon = path_for('unchecked.png')
        self.headerDataChanged.emit(QtCore.Qt.Horizontal, 3, 3)


class AlignmentErrorModel(QtCore.QAbstractTableModel):
    """Provides model for showing simplified alignment errors in table view

    :param index: N x 1 array containing indices of each point
    :type index: numpy.ndarray[int]
    :param error: N x 1 array containing distance error for each point
    :type error: numpy.ndarray[float]
    :param enabled: N x 1 array containing enabled state of each point
    :type enabled: numpy.ndarray[bool]
    :param tolerance: tolerance for acceptable error
    :type tolerance: float
    """
    def __init__(self, index=None, error=None, enabled=None, tolerance=0.1):
        QtCore.QAbstractTableModel.__init__(self)

        self.point_index = index if index is not None else np.empty(0)
        self.error = error if error is not None else np.empty(0)
        self.enabled = enabled if enabled is not None else np.empty(0)
        self.tolerance = tolerance
        self.title = ['Index', 'Error (mm)', 'Enabled']

    def update(self):
        """Updates model's data and layout"""
        self.layoutAboutToBeChanged.emit()
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, 2)
        self.dataChanged.emit(top_left, bottom_right)
        self.layoutChanged.emit()

    def rowCount(self, _parent=None):
        return len(self.point_index)

    def columnCount(self, _parent=None):
        return len(self.title)

    def headerData(self, index, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.title[index])

        return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return QtCore.QVariant()

        if index.column() == 2:
            value = ''
        elif index.column() == 1:
            value = self.error[index.row()]
            value = 'N/A' if np.isnan(value) else f'{value:.3f}'
        else:
            value = f'{self.point_index[index.row()] +  1}'

        if role == QtCore.Qt.EditRole:
            return value
        elif role == QtCore.Qt.DisplayRole:
            return value
        elif role == QtCore.Qt.CheckStateRole:
            if index.column() == 2:
                if self.enabled[index.row()]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter
        elif role == QtCore.Qt.ForegroundRole:
            if index.column() == 1:
                # error could also be NAN for disable points
                if self.error[index.row()] < self.tolerance:
                    return QtGui.QBrush(QtGui.QColor(50, 153, 95))
                elif self.error[index.row()] >= self.tolerance:
                    return QtGui.QBrush(QtGui.QColor(255, 00, 0))

        return QtCore.QVariant()

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid() or not (role == QtCore.Qt.CheckStateRole and index.column() == 2):
            return False

        row = index.row()
        self.enabled[row] = True if value == QtCore.Qt.Checked else False

        return True

    def flags(self, index):
        if index.isValid() and index.column() == 2:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable

        return QtCore.Qt.NoItemFlags


class ErrorDetailModel(QtCore.QAbstractTableModel):
    """Provides model for showing detailed alignment errors in table view

    :param index: N x 1 array containing indices of each point
    :type index: Union[None, numpy.ndarray[int]]
    :param details: M x 2 array containing pairwise distance errors
    :type details: Union[None, numpy.ndarray[float]]
    :param tolerance: tolerance for acceptable error
    :type tolerance: float
    """
    def __init__(self, index=None, details=None, tolerance=0.1):
        QtCore.QAbstractTableModel.__init__(self)

        self._index_pairs = []
        self.index_pairs = index if index is not None else np.empty(0)
        self.details = details if details is not None else np.empty(0)
        self.tolerance = tolerance
        self.title = [
            'Pair Indices', 'Pairwise \nDistances \nin Fiducials \n(mm)', 'Pairwise \nDistances \nin Measured \n(mm)',
            'Difference (mm)'
        ]

    @property
    def index_pairs(self):
        """Gets index pairs used in the pairwise distance analysis

        :return: list of index pairs
        :rtype: List[Tuple(str, str)]
        """
        return self._index_pairs

    @index_pairs.setter
    def index_pairs(self, index):
        """Sets index pairs

        :param index: N x 1 array containing indices of each point
        :type index: numpy.ndarray[int]]
        """
        size = len(index)
        self._index_pairs = [f'({index[x] + 1}, {index[y] + 1})' for x in range(size - 1) for y in range(x + 1, size)]

    def update(self):
        """Updates model's data and layout"""
        self.layoutAboutToBeChanged.emit()
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, 3)
        self.dataChanged.emit(top_left, bottom_right)
        self.layoutChanged.emit()

    def rowCount(self, _parent=None):
        return len(self.index_pairs)

    def columnCount(self, _parent=None):
        return len(self.title)

    def headerData(self, index, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.title[index])

        return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return QtCore.QVariant()

        if index.column() == 0:
            value = f'{self.index_pairs[index.row()]}'
        else:
            value = f'{self.details[index.row(), index.column()-1]:.3f}'

        if role == QtCore.Qt.DisplayRole:
            return value
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter
        elif role == QtCore.Qt.ForegroundRole:
            if index.column() == 3:
                if self.details[index.row(), index.column() - 1] < self.tolerance:
                    return QtGui.QBrush(QtGui.QColor(50, 153, 95))
                else:
                    return QtGui.QBrush(QtGui.QColor(255, 00, 0))

        return QtCore.QVariant()
