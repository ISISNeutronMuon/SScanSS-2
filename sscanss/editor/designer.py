from PyQt5 import QtCore, QtWidgets
from functools import partial
from sscanss.core.instrument.robotics import Link
import json
import sscanss.editor.json_attributes as ja


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
                self.layout.addWidget(label)

            button = QtWidgets.QPushButton(title)
            button.clicked.connect(partial(self.goDown, object))
            button.setMaximumSize(100, 50)
            self.layout.addWidget(button, 0, QtCore.Qt.AlignLeft)

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
    data_changed = QtCore.pyqtSignal(str)

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

        #self.instrument = self.createSchema()
        #self.instrument.been_set.connect(self.DataChanged)

        attributes2 = ja.JsonAttributes()
        attributes2.addAttribute("name", ja.StringValue("Name1"), mandatory=True)
        attributes2.addAttribute("enum", ja.EnumValue(Link.Type), mandatory=True)

        attributes = ja.JsonAttributes()
        attributes.addAttribute("name", ja.StringValue("Name1"), mandatory=True)
        attributes.addAttribute("float", ja.ValueArray([ja.FloatValue(), ja.FloatValue(), ja.FloatValue()]), mandatory=True)
        attributes.addAttribute("enum", ja.EnumValue(Link.Type), mandatory=False)
        attributes.addAttribute("object", ja.JsonObject(self.object_stack, attributes2), mandatory=False)
        object = ja.JsonObject(self.object_stack, attributes)
        self.instrument = object

        self.object_stack.addObject("instrument", self.instrument)
        self.instrument.been_set.connect(self.PrintValue)


    def PrintValue(self, value):
        print(self.instrument.json_value)

    def DataChanged(self):
        self.data_changed.emit(self.getJsonFile())

    def createAttributeArray(self, attribute, number, custom_title="", mandatory=False):
        return ja.ValueArray([attribute.defaultCopy() for i in range(number)], custom_title=custom_title,
                             mandatory=mandatory)

    def createVisualObject(self, mandatory=True):
        visual_attr = {"pose": self.createAttributeArray(ja.FloatValue(), 6, mandatory=False),
                       "colour": ja.ColourValue(mandatory=False),
                       "mesh": ja.FileValue('', '')}

        visual_object = ja.DirectlyEditableObject(visual_attr, self.object_stack, mandatory=mandatory,
                                                  custom_title="Model")
        return visual_object

    def createSchema(self):
        key = "name"
        fixed_hardware_attr = {key: ja.StringValue("Fixed Hardware"),
                               "visual": self.createVisualObject()}
        fixed_hardware_arr = ja.ObjectList([ja.JsonObject(fixed_hardware_attr,
                                                          self.object_stack)], key, self.object_stack, mandatory=False)

        link_attr = {key: ja.StringValue("Link"),
                     "visual": self.createVisualObject(mandatory=False)}
        link_arr = ja.ObjectList([ja.JsonObject(link_attr, self.object_stack)], key, self.object_stack)

        joint_attr = {key: ja.StringValue("Joint"),
                      "type": ja.EnumValue(Link.Type),
                      "parent": ja.SelectedObject("././links"),
                      "child": ja.SelectedObject("././links"),
                      "axis": self.createAttributeArray(ja.FloatValue(), 3),
                      "origin": self.createAttributeArray(ja.FloatValue(), 3),
                      "lower_limit": ja.FloatValue(custom_title="Min offset"),
                      "upper_limit": ja.FloatValue(custom_title="Max offset"),
                      "home_offset": ja.FloatValue(mandatory=False)}
        joint_arr = ja.ObjectList([ja.JsonObject(joint_attr, self.object_stack)], key, self.object_stack)

        positioner_attr = {key: ja.StringValue("Positioner"),
                           "base": self.createAttributeArray(ja.FloatValue(), 6, mandatory=False),
                           "tool": self.createAttributeArray(ja.FloatValue(), 6, mandatory=False),
                           "custom_order": ja.ObjectOrder("joints", mandatory=False),
                           "joints": joint_arr,
                           "links": link_arr}

        positioner_arr = ja.ObjectList([ja.JsonObject(positioner_attr, self.object_stack)], key, self.object_stack)

        jaws_attr = {"aperture": self.createAttributeArray(ja.FloatValue(), 2),
                     "aperture_lower_limit": self.createAttributeArray(ja.FloatValue(), 2),
                     "aperture_upper_limit": self.createAttributeArray(ja.FloatValue(), 2),
                     "beam_direction": self.createAttributeArray(ja.FloatValue(), 3),
                     "beam_source": self.createAttributeArray(ja.FloatValue(), 3),
                     "positioner": ja.SelectedObject("./positioners", mandatory=False),
                     "visual": self.createVisualObject()}
        jaws_object = ja.JsonObject(jaws_attr, self.object_stack)

        collimator_attr = {key: ja.StringValue("Collimatorf"),
                           "aperture": self.createAttributeArray(ja.FloatValue(), 2),
                           "visual": self.createVisualObject()}
        collimator_arr = ja.ObjectList([ja.JsonObject(collimator_attr, self.object_stack)],
                                       key, self.object_stack, mandatory=False)

        detector_attr = {key: ja.StringValue("Detector"),
                         "collimators": collimator_arr,
                         "default_collimator": ja.SelectedObject("collimators", mandatory=False),
                         "diffracted_beam": self.createAttributeArray(ja.FloatValue(), 3),
                         "positioner": ja.SelectedObject("././positioners", mandatory=False)}
        detector_arr = ja.ObjectList([ja.JsonObject(detector_attr, self.object_stack)],
                                     key, self.object_stack)

        positioning_stack_attr = {key: ja.StringValue("Positioning stack"),
                                  "positioners": ja.ObjectOrder("././positioners")}

        positioning_stack_arr = ja.ObjectList([ja.JsonObject(positioning_stack_attr, self.object_stack)], key,
                                              self.object_stack)

        instrument_attr = {"name": ja.StringValue("Instrument"),
                           "version": ja.StringValue(),
                           "script_template": ja.FileValue('', mandatory=False),
                           "gauge_volume": self.createAttributeArray(ja.FloatValue(), 3),
                           "incident_jaws": jaws_object,
                           "detectors": detector_arr,
                           "positioning_stacks": positioning_stack_arr,
                           "positioners": positioner_arr,
                           "fixed_hardware": fixed_hardware_arr}

        instrument_obj = ja.JsonObject(instrument_attr, self.object_stack)

        return instrument_obj

    def getJsonFile(self):
        """Returns dictionary, representing a json object created from the data in the designer
        :return: the json dictionary
        :rtype: dict{str: object}
        """
        json_dict = self.instrument.json_value
        json_dict["collimators"] = []
        for detector in json_dict["detectors"]:
            for collimator in detector["collimators"]:
                collimator["detector"] = detector["name"]
                json_dict["collimators"].append(collimator)
            del detector["collimators"]

        return json.dumps({"instrument": json_dict})

    def setJsonFile(self, text):
        """Loads the text representing json file into the schema to set the relevant data
        :param text: the json file text
        :type text: str
        """
        # Here also change the json schema by moving collimators into the detector objects
        instrument_dict = json.loads(text)["instrument"]

        for detector in instrument_dict["detectors"]:
            detector["collimators"] = [{key: value for key, value in collimator.items() if key != "detector"}
                                       for collimator in instrument_dict["collimators"] if collimator["detector"] ==
                                       detector["name"]]
        del instrument_dict["collimators"]

        self.instrument.json_value = instrument_dict
        self.createUi()

    def createUi(self):
        """Updates the UI according to the top object in the stack"""
        self.attributes_panel.setParent(None)
        self.attributes_panel = self.object_stack.top().createPanel()
        self.layout.addWidget(self.attributes_panel)
