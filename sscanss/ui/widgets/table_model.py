import numpy as np
from PyQt5 import QtCore, QtGui


class NumpyModel(QtCore.QAbstractTableModel):
    editCompleted = QtCore.pyqtSignal(int, tuple)

    def __init__(self, array, parent):
        QtCore.QAbstractTableModel.__init__(self, parent)

        self._array = array.points
        self._enabled = array.enabled
        self.header_icon = ''
        self.title = ['X', 'Y', 'Z']
        self.setHeaderIcon()

        self.table_view = parent
        header = self.table_view.horizontalHeader()
        header.sectionClicked.connect(self.toggleCheckState)

    def rowCount(self, _parent=None):
        return self._array.shape[0]

    def columnCount(self, _parent=None):
        return self._array.shape[1] + 1

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if index.column() == 3:
            value = ''
        else:
            value = QtCore.QVariant('{:.3f}'.format(self._array[index.row(), index.column()]))
        if role == QtCore.Qt.EditRole:
            return value
        elif role == QtCore.Qt.DisplayRole:
            return value
        elif role == QtCore.Qt.CheckStateRole:
            if index.column() == 3:
                if self._enabled[index.row()]:
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
        point = np.copy(self._array[row, :])
        enabled = self._enabled[row]
        if role == QtCore.Qt.CheckStateRole and index.column() == 3:
            if value == QtCore.Qt.Checked:
                enabled = True
            else:
                enabled = False

            self.editCompleted.emit(row, (point, enabled))

        elif role == QtCore.Qt.EditRole and index.column() != 3:
            col = index.column()
            if value.isdigit():
                point[col] = value
                self.editCompleted.emit(row, (point, enabled))
                # TODO: Add range check to avoid input being too large

        return True

    def flags(self, index):
        if not index.isValid():
            return None

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
            if np.all(self._enabled):
                self._enabled.fill(False)
            else:
                self._enabled.fill(True)

            top_left = self.index(0, 3)
            bottom_right = self.index(self.rowCount(), 3)
            self.dataChanged.emit(top_left, bottom_right)
            self.setHeaderIcon()

    def setHeaderIcon(self):
        if np.all(self._enabled):
            self.header_icon = '../static/images/checked.png'
        elif np.any(self._enabled):
            self.header_icon = '../static/images/intermediate.png'
        else:
            self.header_icon = '../static/images/unchecked.png'
        self.headerDataChanged.emit(QtCore.Qt.Horizontal, 3, 3)
