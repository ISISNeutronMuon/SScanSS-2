from enum import Enum, unique
from PyQt5 import QtGui, QtWidgets, QtCore


def create_tool_button(checkable=False, checked=False, tooltip='', style_name='', icon_path='', hide=False,
                       text=''):
    button = QtWidgets.QToolButton()
    button.setCheckable(checkable)
    button.setChecked(checked)
    if hide:
        button.setVisible(False)
    if text:
        button.setText(text)
    if tooltip:
        button.setToolTip(tooltip)
    if style_name:
        button.setObjectName(style_name)
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
        scroll_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
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
            self.toggle_icon.setPixmap(QtGui.QPixmap('../static/images/right_arrow.png'))
        else:
            self.content.show()
            self.toggle_icon.setPixmap(QtGui.QPixmap('../static/images/down_arrow.png'))

    def mousePressEvent(self, _):
        self.toggle(self.content.isVisible())


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
