from PyQt5 import QtCore, QtWidgets
import InstrumentModel as im
from functools import partial

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
        for title, object in self.object_stack:
            if first:
                first = False
            else:
                self.layout.addWidget(QtWidgets.QLabel(">"))

            button = QtWidgets.QPushButton()
            button.setText(title)
            button.clicked.connect(partial(self.goDown, object))
            self.layout.addWidget(button)

    def addObject(self, object_title, new_object):
        self.object_stack.append((object_title, new_object))
        self.stackChanged.emit()

    def goDown(self, selected_object):
        while self.top() != selected_object:
            self.object_stack.pop(-1)

        self.stackChanged.emit()

    def top(self):
        title, obj = self.object_stack[-1]
        return obj

class Designer(QtWidgets.QWidget):
    def __init__(self, parent, instrument_model):
        super().__init__(parent)

        self.object_stack = ObjectStack(self)
        self.object_stack.stackChanged.connect(self.createUi)

        self.attributes_panel = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.object_stack)
        self.layout.addWidget(self.attributes_panel)

        self.instrument_model = self.createSchema()

        self.object_stack.addObject("instrument", self.instrument_model)

    def createSchema(self):
        detectorObjectAttr = {"name": im.JsonString("Name1"),
                              "default_collimator": im.JsonString(),
                              "diffracted_beam": im.JsonString(),
                              "positioner": im.JsonString()}

        detectorObject = im.JsonObject(detectorObjectAttr, self.object_stack)

        detectorObjectAttr2 = {"name": im.JsonString("Name2"),
                               "default_collimator": im.JsonString(),
                               "diffracted_beam": im.JsonString(),
                               "positioner": im.JsonString()}

        detectorObject2 = im.JsonObject(detectorObjectAttr2, self.object_stack)

        visual_object_attr = {"pose": im.JsonAttributeArray([im.JsonFloat(), im.JsonFloat(), im.JsonFloat()]),
                              "colour": im.JsonColour(),
                              "mesh": im.JsonFile(self)}

        visual_object = im.JsonDirectlyEditableObject(visual_object_attr, self.object_stack)

        jaws_object_attr = {"name": im.JsonString("Name2"),
                            "aperture": im.JsonAttributeArray([im.JsonFloat(), im.JsonFloat(), im.JsonFloat()]),
                            "aperture_lower_limit": im.JsonAttributeArray([im.JsonFloat(), im.JsonFloat(), im.JsonFloat()]),
                            "aperture_upper_limit": im.JsonAttributeArray([im.JsonFloat(), im.JsonFloat(), im.JsonFloat()]),
                            "visual": visual_object
                            }

        jaws_object = im.JsonObject(jaws_object_attr, self.object_stack)

        instrumentClassAttr = {"name": im.JsonString(),
                               "version": im.JsonString(),
                               "script_template": im.JsonString(),
                               "gauge_volume": im.JsonAttributeArray([im.JsonFloat(), im.JsonFloat(), im.JsonFloat()]),
                               "incident_jaws": jaws_object,
                               "detectors": im.JsonObjectArray([detectorObject, detectorObject2], "name",
                                                               self.object_stack)}

        instrument_model = im.JsonObject(instrumentClassAttr, self.object_stack)

        return instrument_model

    def createUi(self):
        self.attributes_panel.setParent(None)
        self.attributes_panel = self.object_stack.top().createPanel()
        self.layout.addWidget(self.attributes_panel)