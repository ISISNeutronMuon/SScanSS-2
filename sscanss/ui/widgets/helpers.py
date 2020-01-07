from enum import Enum, unique
from PyQt5 import QtGui, QtWidgets, QtCore
from sscanss.config import path_for


def create_icon(colour, size):
    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtGui.QColor.fromRgbF(*colour))

    return QtGui.QIcon(pixmap)


def create_header(text, name='h2'):
    label = QtWidgets.QLabel(text)
    label.setObjectName(name)

    return label


def create_tool_button(checkable=False, checked=False, tooltip='', style_name='', icon_path='', hide=False,
                       text='', status_tip=''):
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
    """
        Accordion Class.
    """

    def __init__(self):
        super().__init__()
        self.panes = []
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # Define Scroll Area
        scroll_area = QtWidgets.QScrollArea()
        #scroll_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
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
        if not isinstance(pane, Pane):
            raise TypeError("'pane' must be an instance of the Pane Object")

        self.panes.append(pane)
        self.pane_layout.insertWidget(self.pane_layout.count() - 1, pane)

    def clear(self):
        for pane in self.panes:
            self.pane_layout.removeWidget(pane)
            pane.hide()
            pane.deleteLater()

        self.panes = []


class Pane(QtWidgets.QWidget):
    @unique
    class Type(Enum):
        Info = 1
        Warn = 2
        Error = 3

    def __init__(self, pane_widget, pane_content, ptype=Type.Info):
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
        self.setType(ptype)

        self.context_menu = QtWidgets.QMenu()
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)

        super().paintEvent(event)

    def setType(self, ptype):
        style = ('QWidget#pane-header {{background-color:{};border-bottom: 1px solid;}} '
                 'QWidget#pane-content {{border-bottom: 1px solid;}}')
        if ptype == self.Type.Error:
            style = style.format('#CD6155')
        elif ptype == self.Type.Warn:
            style = style.format('#F4D03F')
        else:
            style = 'QWidget#pane-header, QWidget#pane-content {border-bottom: 1px solid gray;}'
        self.setStyleSheet(style)

    def toggle(self, visible):
        if visible:
            self.content.hide()
            self.toggle_icon.setPixmap(QtGui.QPixmap(path_for('right_arrow.png')))
        else:
            self.content.show()
            self.toggle_icon.setPixmap(QtGui.QPixmap(path_for('down_arrow.png')))

    def addContextMenuAction(self, action):
        self.context_menu.addAction(action)

    def showContextMenu(self, pos):
        globalPos = self.mapToGlobal(pos)
        self.context_menu.popup(globalPos)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self.toggle(self.content.isVisible())

        super().mousePressEvent(event)


class ColourPicker(QtWidgets.QWidget):
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
        return self.__value

    def mousePressEvent(self, _):
        colour = QtWidgets.QColorDialog.getColor(self.value)

        if colour.isValid():
            self.__value = colour
            self.colour_view.setStyleSheet(self.style.format(colour.name()))
            self.colour_name.setText(colour.name())
            self.value_changed.emit(colour.getRgbF())


class StatusBar(QtWidgets.QStatusBar):
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
        if alignment == QtCore.Qt.AlignLeft:
            self.left_layout.addWidget(widget, stretch)
        else:
            self.right_layout.addWidget(widget, stretch)

    def removeWidget(self, widget):
        self.left_layout.removeWidget(widget)
        self.right_layout.removeWidget(widget)

    def currentMessage(self):
        return self.message_label.text()

    def showMessage(self, message, timeout=0):
        self.message_label.setText(message)
        if timeout > 0:
            self.timer.singleShot(timeout, self.clearMessage)

    def clearMessage(self):
        self.message_label.setText('')
