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
