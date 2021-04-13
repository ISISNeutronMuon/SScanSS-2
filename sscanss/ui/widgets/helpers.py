from enum import Enum, unique
import os
import re
from PyQt5 import QtGui, QtWidgets, QtCore
from sscanss.config import path_for


def create_icon(colour, size):
    """Creates an icon with a plain colour and fixed size

    :param colour: colour of icon
    :type colour: Tuple[float, float, float, float]
    :param size: dimension of icon
    :type size: QtCore.QSize
    :return: icon
    :rtype: QtGui.QIcon
    """
    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtGui.QColor.fromRgbF(*colour))

    return QtGui.QIcon(pixmap)


def create_header(text, name='h2'):
    """Creates a label with given name which is used to apply header style

    :param text: label text
    :type text: str
    :param name: style name
    :type name: str
    :return: header label
    :rtype: QtWidgets.QLabel
    """
    label = QtWidgets.QLabel(text)
    label.setObjectName(name)

    return label


def create_tool_button(checkable=False, checked=False, tooltip='', style_name='', icon_path='', hide=False,
                       text='', status_tip=''):
    """Creates tool button

    :param checkable: flag that indicates button can be checked
    :type checkable: bool
    :param checked: flag that indicates button is checked
    :type checked: Union[bool, numpy.bool_]
    :param tooltip: tooltip text
    :type tooltip: str
    :param style_name: style name
    :type style_name: str
    :param icon_path: path to icon
    :type icon_path: str
    :param hide: flag that indicates button is hidden
    :type hide: bool
    :param text: button text
    :type text: str
    :param status_tip: status bar text
    :type status_tip: str
    :return: tool button
    :rtype: QtWidgets.QToolButton
    """
    button = QtWidgets.QToolButton()
    button.setCheckable(checkable)
    button.setChecked(checked)
    button.setText(text)
    button.setToolTip(tooltip)
    button.setStatusTip(status_tip)
    button.setObjectName(style_name)
    if hide:
        button.setVisible(False)

    if icon_path:
        button.setIcon(QtGui.QIcon(icon_path))

    return button


def create_scroll_area(content, vertical_scroll=True, horizontal_scroll=False):
    """Wraps a widget in a QScrollArea

    :param content: widget
    :type content: QtWidgets.QWidget
    :param vertical_scroll: flag indicates vertical scrollbar be enabled
    :type vertical_scroll: bool
    :param horizontal_scroll: flag indicates horizontal scrollbar be enabled
    :type horizontal_scroll: bool
    :return: QScrollArea object with widget embedded
    :rtype: QtWidgets.QScrollArea
    """
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidget(content)
    scroll_area.setViewportMargins(0, 10, 0, 0)
    scroll_area.setAutoFillBackground(True)
    scroll_area.setWidgetResizable(True)
    scroll_policy = QtCore.Qt.ScrollBarAsNeeded if vertical_scroll else QtCore.Qt.ScrollBarAlwaysOff
    scroll_area.setVerticalScrollBarPolicy(scroll_policy)
    scroll_policy = QtCore.Qt.ScrollBarAsNeeded if horizontal_scroll else QtCore.Qt.ScrollBarAlwaysOff
    scroll_area.setHorizontalScrollBarPolicy(scroll_policy)
    scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

    return scroll_area


class Accordion(QtWidgets.QWidget):
    """Creates Accordion object"""

    def __init__(self):
        super().__init__()
        self.panes = []
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # Define Scroll Area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        pane_widget = QtWidgets.QWidget()
        self.pane_layout = QtWidgets.QVBoxLayout()
        self.pane_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.pane_layout.addStretch(1)
        pane_widget.setLayout(self.pane_layout)
        self.pane_layout.setSpacing(0)
        self.pane_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setWidget(pane_widget)
        main_layout.addWidget(scroll_area)

    def addPane(self, pane):
        """Adds pane to the Accordion

        :param pane: Pane object
        :type pane: Pane
        """
        if not isinstance(pane, Pane):
            raise TypeError("'pane' must be an instance of the Pane Object")

        self.panes.append(pane)
        self.pane_layout.insertWidget(self.pane_layout.count() - 1, pane)

    def clear(self):
        """Removes all panes from Accordion"""
        for pane in self.panes:
            self.pane_layout.removeWidget(pane)
            pane.hide()
            pane.deleteLater()

        self.panes = []


class Pane(QtWidgets.QWidget):
    """Creates an Accordion Pane that shows/hides content when clicked

    :param pane_widget: widget to embed in pane header
    :type pane_widget: QtWidgets.QLabel
    :param pane_content: widget to embed as pane content
    :type pane_content: QtWidgets.QLabel
    :param pane_type: pane type
    :type pane_type: Pane.Type
    """
    @unique
    class Type(Enum):
        Info = 1
        Warn = 2
        Error = 3

    def __init__(self, pane_widget, pane_content, pane_type=Type.Info):
        super().__init__()

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        self.header = QtWidgets.QWidget()
        self.header.setObjectName('pane-header')
        self.toggle_icon = QtWidgets.QLabel()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(pane_widget)
        layout.addStretch(1)
        layout.addWidget(self.toggle_icon)
        self.header.setLayout(layout)
        main_layout.addWidget(self.header)

        self.content = QtWidgets.QWidget()
        self.content.setObjectName('pane-content')
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(pane_content)
        self.content.setLayout(layout)
        main_layout.addWidget(self.content)

        self.toggle(True)
        self.setType(pane_type)

        self.context_menu = QtWidgets.QMenu()
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)

        super().paintEvent(event)

    def setType(self, pane_type):
        """Sets pane type which determines style

        :param pane_type: pane type
        :type pane_type: Pane.Type
        """
        style = ('QWidget#pane-header {{background-color:{};border-bottom: 1px solid;}} '
                 'QWidget#pane-content {{border-bottom: 1px solid;}}')
        if pane_type == self.Type.Error:
            style = style.format('#CD6155')
        elif pane_type == self.Type.Warn:
            style = style.format('#F4D03F')
        else:
            style = 'QWidget#pane-header, QWidget#pane-content {border-bottom: 1px solid gray;}'
        self.setStyleSheet(style)

    def toggle(self, visible):
        """Toggles visibility of Pane content

        :param visible: indicates if pane is visible
        :type visible: bool
        """
        if visible:
            self.content.hide()
            self.toggle_icon.setPixmap(QtGui.QPixmap(path_for('right_arrow.png')))
        else:
            self.content.show()
            self.toggle_icon.setPixmap(QtGui.QPixmap(path_for('down_arrow.png')))

    def addContextMenuAction(self, action):
        """Adds action to context menu

        :param action: QAction object
        :type action: QtWidgets.QAction
        """
        self.context_menu.addAction(action)

    def showContextMenu(self, pos):
        """Shows context menu at cursor position

        :param pos: cursor position
        :type pos: QtCore.QPoint
        """
        global_pos = self.mapToGlobal(pos)
        self.context_menu.popup(global_pos)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self.toggle(self.content.isVisible())

        super().mousePressEvent(event)


class ColourPicker(QtWidgets.QWidget):
    """Creates ColourPicker object used for colour selection

    :param colour: initial colour
    :type colour: QtGui.QColor
    """
    value_changed = QtCore.pyqtSignal(tuple)

    def __init__(self, colour):
        super().__init__()

        layout = QtWidgets.QHBoxLayout()
        self.style = "background-color: {}"

        self.colour_view = QtWidgets.QLabel()
        self.colour_view.setStyleSheet(self.style.format(colour.name()))
        self.colour_view.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Sunken)
        self.colour_view.setFixedSize(20, 20)
        layout.addWidget(self.colour_view)

        self.colour_name = QtWidgets.QLabel(colour.name())
        layout.addWidget(self.colour_name)
        self.setLayout(layout)

        self.__value = colour

    @property
    def value(self):
        """Returns current colour

        :return: current colour
        :rtype: QtGui.QColor
        """
        return self.__value

    def mousePressEvent(self, _):
        colour = QtWidgets.QColorDialog.getColor(self.value)

        if colour.isValid():
            self.__value = colour
            self.colour_view.setStyleSheet(self.style.format(colour.name()))
            self.colour_name.setText(colour.name())
            self.value_changed.emit(colour.getRgbF())


class FilePicker(QtWidgets.QWidget):
    """Creates FilePicker object used for file/folder selection

    :param path: initial path
    :type path: str
    :param select_folder: flag indicates if folder mode is enabled
    :type select_folder: bool
    :param filters: file filters
    :type filters: str
    """
    value_changed = QtCore.pyqtSignal(str)

    def __init__(self, path, select_folder=False, filters=''):
        super().__init__()

        self.select_folder = select_folder
        self.filters = filters

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(0)

        self.file_view = QtWidgets.QLineEdit()
        self.file_view.setReadOnly(True)
        layout.addWidget(self.file_view)

        self.browse_button = QtWidgets.QPushButton('Select')
        self.browse_button.clicked.connect(self.openFileDialog)
        layout.addWidget(self.browse_button)
        self.setLayout(layout)

        self.value = path

    @property
    def value(self):
        """Return current path

        :return: current path
        :rtype: str
        """
        return self.file_view.text()

    @value.setter
    def value(self, path):
        """Sets current path

        :param path: path
        :type path: str
        """
        if path and path != self.value:
            self.file_view.setText(path)
            self.file_view.setCursorPosition(0)
            self.value_changed.emit(path)

    def openFileDialog(self):
        """Opens file dialog """
        if not self.select_folder:
            self.value = FileDialog.getOpenFileName(self, 'Select File', self.value, self.filters)
        else:
            self.value = FileDialog.getExistingDirectory(self, 'Select Folder', self.value,
                                                         QtWidgets.QFileDialog.ShowDirsOnly |
                                                         QtWidgets.QFileDialog.DontResolveSymlinks)


class StatusBar(QtWidgets.QStatusBar):
    """Creates StatusBar object that allows widget to be added in left and right positions

    :param parent: parent widget
    :type parent: Union[None, QtWidgets.QWidget]
    """

    def __init__(self, parent=None):

        super().__init__(parent)

        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QGridLayout()
        main_layout.setContentsMargins(8, 1, 8, 1)
        main_layout.setHorizontalSpacing(20)
        widget.setLayout(main_layout)

        self.message_label = QtWidgets.QLabel()
        self.timer = QtCore.QTimer(self)

        self.left_layout = QtWidgets.QHBoxLayout()
        self.right_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(self.left_layout, 0, 0, QtCore.Qt.AlignLeft)
        main_layout.addWidget(self.message_label, 0, 1, QtCore.Qt.AlignLeft)
        main_layout.addLayout(self.right_layout, 0, 2, QtCore.Qt.AlignRight)
        main_layout.setColumnStretch(1, 1)
        super().addPermanentWidget(widget, 1)
        super().messageChanged.connect(self.showMessage)

    def addPermanentWidget(self, widget, stretch=0, alignment=QtCore.Qt.AlignRight):
        """Adds status bar widget

        :param widget: widget to remove
        :type widget: QtWidgets.QWidget
        :param stretch: layout stretch
        :type stretch: int
        :param alignment: widget alignment
        :type alignment: PyQt5.QtCore.AlignmentFlag
        """
        if alignment == QtCore.Qt.AlignLeft:
            self.left_layout.addWidget(widget, stretch)
        else:
            self.right_layout.addWidget(widget, stretch)

    def removeWidget(self, widget):
        """Removes status bar widget

        :param widget: widget to remove
        :type widget: QtWidgets.QWidget
        """
        self.left_layout.removeWidget(widget)
        self.right_layout.removeWidget(widget)

    def currentMessage(self):
        """Return current message

        :return: current message
        :rtype: str
        """
        return self.message_label.text()

    def showMessage(self, message, timeout=0):
        """Display message on status bar for a period of time

        :param message: message
        :type message: str
        :param timeout: milliseconds to show message
        :type timeout: int
        """
        self.message_label.setText(message)
        if timeout > 0:
            self.timer.singleShot(timeout, self.clearMessage)

    def clearMessage(self):
        """Clear status bar message """
        self.message_label.setText('')


class FileDialog(QtWidgets.QFileDialog):
    """Creates FileDialog object

    :param parent: parent widget
    :type parent: QtWidgets.QWidget
    :param caption: title text
    :type caption: str
    :param directory: initial path
    :type directory: str
    :param filters: file filters
    :type filters: str
    """
    def __init__(self, parent, caption, directory, filters):
        super().__init__(parent, caption, directory, filters)

        self.setOptions(QtWidgets.QFileDialog.DontConfirmOverwrite)

    def extractFilters(self, filters):
        """Extract file extensions from filter string

        :param filters:
        :type filters: str
        :return: list of file extensions
        :rtype: List[str]
        """
        filters = re.findall(r'\*(?:.\w+)?', filters)
        return [f[1:] for f in filters]

    @property
    def filename(self):
        """Returns selected file path

        :return: selected file path
        :rtype: str
        """
        filename = self.selectedFiles()[0]
        _, ext = os.path.splitext(filename)
        expected_ext = self.extractFilters(self.selectedNameFilter())
        if ext.lower() not in map(str.lower, expected_ext):
            filename = f'{filename}{expected_ext[0]}'

        return filename

    @staticmethod
    def getOpenFileName(parent, caption, directory, filters):
        """Shows open file dialog

        :param parent: parent widget
        :type parent: QtWidgets.QWidget
        :param caption: title text
        :type caption: str
        :param directory: initial path
        :type directory: str
        :param filters: file filters
        :type filters: str
        :return: selected file path
        :rtype: str
        """
        dialog = FileDialog(parent, caption, directory, filters)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if dialog.exec() != QtWidgets.QFileDialog.Accepted:
            return ''

        filename = dialog.filename

        if not os.path.isfile(filename):
            message = f'{filename} file not found.\nCheck the file name and try again.'
            QtWidgets.QMessageBox.warning(parent, caption, message, QtWidgets.QMessageBox.Ok,
                                          QtWidgets.QMessageBox.Ok)
            return ''

        return filename

    @staticmethod
    def getSaveFileName(parent, caption, directory, filters):
        """Shows save file dialog

        :param parent: parent widget
        :type parent: QtWidgets.QWidget
        :param caption: title text
        :type caption: str
        :param directory: initial path
        :type directory: str
        :param filters: file filters
        :type filters: str
        :return: selected file path
        :rtype: str
        """
        dialog = FileDialog(parent, caption, directory, filters)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        if dialog.exec() != QtWidgets.QFileDialog.Accepted:
            return ''

        filename = dialog.filename

        if os.path.isfile(filename):
            buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            message = f'{filename} already exists.\nDo want to replace it?'
            reply = QtWidgets.QMessageBox.warning(parent, caption, message, buttons,
                                                  QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.No:
                return ''

        return filename
