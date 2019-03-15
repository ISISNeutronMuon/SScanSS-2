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
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setWidgetResizable(True)

        pane_widget = QtWidgets.QWidget()
        self.pane_layout = QtWidgets.QVBoxLayout()
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
        self.pane_layout.insertWidget(self.pane_layout.count() - 1, pane.content)

    def clear(self):
        for pane in self.panes:
            self.pane_layout.removeWidget(pane.content)
            self.pane_layout.removeWidget(pane)
            pane.content.hide()
            pane.content.deleteLater()
            pane.hide()
            pane.deleteLater()

        self.panes = []


class Pane(QtWidgets.QWidget):
    @unique
    class Type(Enum):
        Info = 1
        Warn = 2
        Error = 3

    def __init__(self, pane_widget, content, ptype=Type.Info):
        super().__init__()

        self.setContentsMargins(0, 0, 0, 0)
        main_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        self.toggle_icon = QtWidgets.QLabel()

        layout.addWidget(self.toggle_icon)
        layout.addWidget(pane_widget)
        main_layout.addLayout(layout)
        self.setLayout(main_layout)

        self.content = content
        self.content.hide()
        self.toggle_icon.setPixmap(QtGui.QPixmap('right_arrow.png'))
        self.setType(ptype)

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)

        super().paintEvent(event)

    def setType(self, ptype):
        style = 'Pane {{background-color:{};border-bottom: 1px solid {};}}'
        if ptype == self.Type.Error:
            style = style.format('#CD6155', '#CD6155')
        elif ptype == self.Type.Warn:
            style = style.format('#F4D03F', '#F4D03F')
        else:
            style = 'Pane {border-bottom: 1px solid gray;}'
        self.setStyleSheet(style)

    def toggle(self):
        if self.content.isVisible():
            self.content.hide()
            self.toggle_icon.setPixmap(QtGui.QPixmap('../static/images/right_arrow.png'))
        else:
            self.content.show()
            self.toggle_icon.setPixmap(QtGui.QPixmap('../static/images/down_arrow.png'))

    def mousePressEvent(self, _):
        self.toggle()
