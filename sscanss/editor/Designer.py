from PyQt5 import QtCore, QtWidgets
from functools import partial
from sscanss.core.instrument.robotics import Link
import json
import InstrumentModel as im

class ObjectStack(QtWidgets.QWidget):
    """Holds the stack of selected objects to allow the user navigate the Json file and edit nested objects
    :param parent: parent widget
    :type parent: QWidget
    """
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
        """Recreates the UI based on the contents of the stack, creating a button for each object
        there to allow the user to return to any of the previous objects
        """
        self.clearLayout()
        first = True
        for title, object in self.object_stack:
            if first:
                first = False
            else:
                label = QtWidgets.QLabel(">")
                label.setMaximumSize(10, 50)
                #self.layout.addWidget(label)

            button = QtWidgets.QPushButton(title)
            button.clicked.connect(partial(self.goDown, object))
            button.setMaximumSize(100, 50)
            self.layout.addWidget(button, 0, QtCore.Qt.AlignLeft)

        for i in reversed(range(self.layout.count())):
            print(self.layout.itemAt(i).widget())

    def addObject(self, object_title, new_object):
        """Adds another object on top of the stack and updates UI accordingly
        :param object_title: title of the object to be shown in the button
        :type object_title: str
        :param new_object: the object to be added
        :type new_object: JsonObjectAttribute
        """
        self.object_stack.append((object_title, new_object))
        self.stackChanged.emit()

    def goDown(self, selected_object):
        """Goes down the stack removing all objects until the selected one
        :param selected_object: Object on top of which everything will be removed
        :type selected_object: JsonObjectAttribute
        """
        while self.top() != selected_object:
            self.object_stack.pop(-1)

        self.stackChanged.emit()

    def top(self):
        """Returns the stack on top of the stack - it is supposed to be currently active
        :return selected_object: the top object
        :rtype selected_object: JsonObjectAttribute
        """
        title, obj = self.object_stack[-1]
        return obj

class Designer(QtWidgets.QWidget):
    def __init__(self, parent):
        """Creates an instance of the designer widget to edit a Json file with a GUI
        :param parent: instance of the main window
        :type parent: MainWindow
        """

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

    def createVisualObject(self):
        visual_object_attr = {"pose": im.JsonAttributeArray([im.JsonFloat(), im.JsonFloat(), im.JsonFloat()]),
                              "colour": im.JsonColour(),
                              "mesh": im.JsonFile()}

        visual_object = im.JsonDirectlyEditableObject(visual_object_attr, self.object_stack)
        return visual_object

    def createAttributeArray(self, type, number):
        return im.JsonAttributeArray([type.defaultCopy() for i in range(number)])

    def addCyclicReferences(self, arr1, arr2, key1, key2):
        arr1.objects[0].attributes[key1] = im.JsonObjReference(arr2)
        arr2.objects[0].attributes[key2] = im.JsonObjReference(arr1)

    def createSchema(self):
        key = "name"
        fixed_hardware_attr = {key: im.JsonString("Name1"),
                               "visual": self.createVisualObject()}
        fixed_hardware_arr = im.JsonObjectArray([im.JsonObject(fixed_hardware_attr,
                                                 self.object_stack)], key, self.object_stack)

        link_object_attr = {key: im.JsonString("Name1"),
                            "visual": self.createVisualObject()}
        link_obj_arr = im.JsonObjectArray([im.JsonObject(link_object_attr, self.object_stack)], key, self.object_stack)

        joint_object_attr = {key: im.JsonString("Name1"),
                             "type": im.JsonEnum(Link.Type),
                             "parent": im.JsonObjReference(link_obj_arr),
                             "axis": self.createAttributeArray(im.JsonFloat(), 3),
                             "origin": self.createAttributeArray(im.JsonFloat(), 3),
                             "lower_limit": im.JsonFloat(),
                             "upper_limit": im.JsonFloat(),
                             "home_offset": im.JsonFloat()}
        joint_arr = im.JsonObjectArray([im.JsonObject(joint_object_attr, self.object_stack)], key, self.object_stack)

        positioner_attr = {key: im.JsonString("Name1"),
                           "base": self.createAttributeArray(im.JsonFloat(), 6),
                           "tool": self.createAttributeArray(im.JsonFloat(), 6),
                           "custom_order": self.createAttributeArray(im.JsonString(), 6),
                           "joints": im.JsonObjReference(joint_arr),
                           "links": im.JsonObjReference(link_obj_arr)}

        positioner_arr = im.JsonObjectArray([im.JsonObject(positioner_attr, self.object_stack)], key, self.object_stack)

        jaws_object_attr = {key: im.JsonString("Name2"),
                            "aperture": self.createAttributeArray(im.JsonFloat(), 3),
                            "aperture_lower_limit": self.createAttributeArray(im.JsonFloat(), 3),
                            "aperture_upper_limit": self.createAttributeArray(im.JsonFloat(), 3),
                            "beam_source": self.createAttributeArray(im.JsonFloat(), 3),
                            "positioner": im.JsonObjReference(positioner_arr),
                            "visual": self.createVisualObject()}
        jaws_object = im.JsonObject(jaws_object_attr, self.object_stack)

        detector_object_attr = {key: im.JsonString("Name1"),
                                "default_collimator": None,
                                "diffracted_beam": self.createAttributeArray(im.JsonFloat(), 3),
                                "positioner": im.JsonObjReference(positioner_arr)}
        detector_arr = im.JsonObjectArray([im.JsonObject(detector_object_attr, self.object_stack)],
                                          key, self.object_stack)

        collimator_object_attr = {key: im.JsonString("Name1"),
                                  "detector": None,
                                  "aperture": self.createAttributeArray(im.JsonFloat(), 2),
                                  "visual": self.createVisualObject()}
        collimator_arr = im.JsonObjectArray([im.JsonObject(collimator_object_attr, self.object_stack)],
                                            key, self.object_stack)

        self.addCyclicReferences(detector_arr, collimator_arr, "default_collimator", "detector")

        instrument_class_attr = {"name": im.JsonString(),
                                 "version": im.JsonString(),
                                 "script_template": im.JsonFile(),
                                 "gauge_volume": im.JsonAttributeArray([im.JsonFloat(), im.JsonFloat(), im.JsonFloat()]),
                                 "incident_jaws": jaws_object,
                                 "detectors": detector_arr,
                                 "collimators": collimator_arr,
                                 "positioning_stacks": im.JsonObjectArray([], key, self.object_stack),
                                 "positioners": positioner_arr,
                                 "fixed_hardware": fixed_hardware_arr}

        instrument_model = im.JsonObject(instrument_class_attr, self.object_stack)

        return instrument_model

    def getJsonFile(self):
        return self.instrument_model.getJsonValue()

    def setJsonFile(self, text):
        pass

    def createUi(self):
        """Updates the UI according to the top object in the stack"""
        self.attributes_panel.setParent(None)
        self.attributes_panel = self.object_stack.top().createPanel()
        self.layout.addWidget(self.attributes_panel)
