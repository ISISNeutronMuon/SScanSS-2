from PyQt5 import QtCore, QtWidgets
import InstrumentModel as im
from functools import partial

class ObjectStack(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.layout)
        self.object_stack = []

    def clearLayout(self):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

    def createUi(self):
        self.clearLayout()

        first = True
        for object, title in self.object_stack:
            if first:
                first = False
            else:
                self.layout.addWidget(QtWidgets.QLabel(">"))
            button = QtWidgets.QPushButton()
            button.setText(title)
            button.clicked.connect(partial(self.goDown, object, title))
            self.layout.addWidget(button)

    def addObject(self, new_object, new_title):
        self.object_stack.append((new_object, new_title))
        self.createUi()

        self.parent.setObject(new_object)

    def goDown(self, old_object, old_title):
        obj, title = self.object_stack[-1]
        while title != old_title:
            self.object_stack.pop(-1)
            obj, title = self.object_stack[-1]

        self.createUi()
        self.parent.setObject(obj)

class Designer(QtWidgets.QWidget):
    def __init__(self, parent, instrument_model):
        super().__init__(parent)

        self.object_stack = ObjectStack(self)
        self.attributes_panel = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.object_stack)
        self.layout.addWidget(self.attributes_panel)

        self.object_stack.addObject(instrument_model, "Instrument")

    def formatTitle(self, string):
        return ' '.join([word.capitalize() for word in string.split('_')])

    def setObject(self, new_object):
        self.currentObject = new_object
        self.createUi()

    def setAttribute(self, attribute, newValue):
        try:
            attribute.setValue(newValue)
        except ValueError as e:
            self.parent.setMessage(e)

    def createUi(self):
        self.layout.removeWidget(self.attributes_panel)
        self.attributes_panel = QtWidgets.QWidget()
        self.attributes_panel.layout = QtWidgets.QGridLayout()

        for count, attr in enumerate(self.currentObject.attributes.items()):
            name, attribute = attr
            name = self.formatTitle(name)
            self.attributes_panel.layout.addWidget(QtWidgets.QLabel(name), count, 0)

            if isinstance(attribute, im.JsonString):
                entry = QtWidgets.QLineEdit()
                entry.setText(attribute.value)
                entry.textChanged.connect(partial(attribute.setValue))
            elif isinstance(attribute, im.JsonFloat):
                entry = QtWidgets.QDoubleSpinBox()
                entry.setValue(attribute.value)
                entry.valueChanged.connect(partial(attribute.setValue))
            elif isinstance(attribute, im.JsonObject):
                entry = QtWidgets.QPushButton("Edit " + name)
                entry.clicked.connect(partial(self.object_stack.addObject, attribute, name))
            elif isinstance(attribute, im.JsonFloatVec):
                pass
            self.attributes_panel.layout.addWidget(entry, count, 1)

        self.attributes_panel.setLayout(self.attributes_panel.layout)
        self.layout.addWidget(self.attributes_panel)
