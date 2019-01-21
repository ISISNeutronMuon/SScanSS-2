import numpy as np
from PyQt5 import QtCore, QtGui


class NumpyModel(QtCore.QAbstractTableModel):
    editCompleted = QtCore.pyqtSignal(object)

    def __init__(self, array):
        QtCore.QAbstractTableModel.__init__(self)

        self._data = array.copy()
        self.header_icon = ''
        self.title = ['X', 'Y', 'Z']
        self.setHeaderIcon()

    def update(self, array):
        self._data = array.copy()

    def rowCount(self, _parent=None):
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
            if value.isdigit():
                self._data.points[row, col] = value
                self.editCompleted.emit(self._data)
                # TODO: Add range check to avoid input being too large

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
        if index == 3:
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
        if np.all(self._data.enabled):
            self.header_icon = '../static/images/checked.png'
        elif np.any(self._data.enabled):
            self.header_icon = '../static/images/intermediate.png'
        else:
            self.header_icon = '../static/images/unchecked.png'
        self.headerDataChanged.emit(QtCore.Qt.Horizontal, 3, 3)
