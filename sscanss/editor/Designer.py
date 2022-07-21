from PyQt5 import QtCore, QtWidgets
import InstrumentModel as im

class ObjectStack(QtWidgets.QWidget):
    stackChanged = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.layout)
        self.object_stack = []
        self.stackChanged.connect(self.createUi)

    def clearLayout(self):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

    def createUi(self):
        self.clearLayout()
        first = True
        for object in self.object_stack:
            if first:
                first = False
            else:
                self.layout.addWidget(QtWidgets.QLabel(">"))

            self.layout.addWidget(object.createStackWidget())

    def addObject(self, new_object):
        self.object_stack.append(new_object)
        self.stackChanged.emit()

    def goDown(self, selected_object):
        while self.top() != selected_object:
            self.object_stack.pop(-1)

        self.stackChanged.emit()

    def top(self):
        return self.object_stack[-1]

class Designer(QtWidgets.QWidget):

    def createSchema(self):
        detectorObjectAttr = {"name": im.JsonString(True),
                               "default_collimator": im.JsonString(True),
                               "diffracted_beam": im.JsonString(False),
                               "positioner": im.JsonString(True)}

        self.detectorObject = im.JsonObject("detector", detectorObjectAttr, self.object_stack)

        instrumentClassAttr = {"name": im.JsonString(True),
                               "version": im.JsonString(True),
                               "script_template": im.JsonString(False),
                               "gauge_volume": im.JsonAttributeArray(
                                   [im.JsonFloat(), im.JsonFloat(), im.JsonFloat()],
                                   True),
                               "incident_jaws": im.JsonString(True),
                               "detectors": self.detectorObject}

        self.instrument_model = im.JsonObject("instrument", instrumentClassAttr, self.object_stack)

    def __init__(self, parent, instrument_model):
        super().__init__(parent)

        self.object_stack = ObjectStack(self)
        self.object_stack.stackChanged.connect(self.createUi)

        self.attributes_panel = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.object_stack)
        self.layout.addWidget(self.attributes_panel)

        self.createSchema()

        self.object_stack.addObject(self.instrument_model)

    def createUi(self):
        self.layout.removeWidget(self.attributes_panel)
        self.attributes_panel = self.object_stack.top().createPanel()
        self.layout.addWidget(self.attributes_panel)
