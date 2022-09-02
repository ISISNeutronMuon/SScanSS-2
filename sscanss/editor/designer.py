from PyQt5 import QtCore, QtWidgets
from functools import partial
from sscanss.core.instrument.robotics import Link
import json
from sscanss.config import SOURCE_PATH
import sscanss.editor.json_attributes as ja


class ObjectStack(QtWidgets.QWidget):
    """Holds the stack of selected objects to allow the user navigate the Json file and edit nested objects
    :param parent: parent widget
    :type parent: QWidget
    """
    stack_changed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.layout)
        self.object_stack = []
        self.stack_changed.connect(self.createUi)

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
                label.setMinimumSize(10, 50)
                self.layout.addWidget(label)

            button = QtWidgets.QPushButton(title)
            button.clicked.connect(partial(self.goDown, object))
            button.setMaximumSize(140, 50)
            button.setMinimumSize(140, 50)
            self.layout.addWidget(button, 0, QtCore.Qt.AlignLeft)
        fill_widget = QtWidgets.QWidget()
        fill_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(fill_widget)

    def addObject(self, object_title, new_object):
        """Adds another object on top of the stack and updates UI accordingly
        :param object_title: title of the object to be shown in the button
        :type object_title: str
        :param new_object: the object to be added
        :type new_object: JsonObjectAttribute
        """
        self.object_stack.append((object_title, new_object))
        self.stack_changed.emit()

    def goDown(self, selected_object):
        """Goes down the stack removing all objects until the selected one
        :param selected_object: Object on top of which everything will be removed
        :type selected_object: JsonObjectAttribute
        """
        while self.top() != selected_object:
            self.object_stack.pop(-1)

        self.stack_changed.emit()

    def top(self):
        """Returns the stack on top of the stack - it is supposed to be currently active
        :return selected_object: the top object
        :rtype selected_object: JsonObjectAttribute
        """
        _, obj = self.object_stack[-1]
        return obj


class Designer(QtWidgets.QWidget):
    """Creates an instance of the designer widget to edit a Json file with a GUI
    :param parent: instance of the main window
    :type parent: MainWindow
    """

    data_changed = QtCore.pyqtSignal()
    new_relative_path = QtCore.pyqtSignal(str)

    def __init__(self, parent, update_thread):
        super().__init__(parent)

        self.object_stack = ObjectStack(self)
        self.object_stack.stack_changed.connect(self.createUi)
        self.update_thread = update_thread

        self.relative_path = ''
        self.attributes_panel = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.temp_widget = QtWidgets.QWidget()
        self.layout.addWidget(self.object_stack)
        self.layout.addWidget(self.attributes_panel)
        self.temp_widget = QtWidgets.QWidget()
        fill_widget = QtWidgets.QWidget()
        fill_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(fill_widget)
        self.resetInstrument()

    def dataChanged(self):
        """Method is called when any change occurs to the json file"""
        self.data_changed.emit()

    def createAttributeArray(self, attribute, number, format=None):
        """Creates attribute array with the given attribute copied given number of times"""
        return ja.ValueArray([attribute.defaultCopy() for i in range(number)], format)

    def createFileValue(self, relative_path, filter='', initial_value=''):
        file_value = ja.FileValue(relative_path=str(relative_path).replace('\\', '/'),
                                  filter=filter,
                                  initial_value=initial_value)
        self.new_relative_path.connect(file_value.updateRelativePath)
        return file_value

    def createVisualObject(self):
        visual_attr = ja.JsonAttributes()
        visual_attr.addAttribute("pose",
                                 self.createAttributeArray(ja.FloatValue(), 6, {
                                     "Translation (XYZ)": 3,
                                     "Orientation (XYZ)": 3
                                 }),
                                 mandatory=False)
        visual_attr.addAttribute("colour", ja.ColourValue(), mandatory=False)
        visual_attr.addAttribute("mesh", self.createFileValue(SOURCE_PATH, filter="Stl file (*.stl)"))
        visual_object = ja.DirectlyEditableObject(self.object_stack, visual_attr)
        return visual_object

    def updateSavePath(self, new_path):
        """Updates the relative path when the save location has been changes, should call appropriate attributes
        like fileValue
        :param new_path: the new relative path
        :type new_path: str
        """
        self.relative_path = new_path
        self.new_relative_path.emit(self.relative_path)

    def createSchema(self):
        key = "name"

        fixed_hardware_attr = ja.JsonAttributes()
        fixed_hardware_attr.addAttribute(key, ja.StringValue("Fixed Hardware"))
        fixed_hardware_attr.addAttribute("visual", self.createVisualObject())

        fixed_hardware_arr = ja.ObjectList(key, self.object_stack, ja.JsonObject(self.object_stack,
                                                                                 fixed_hardware_attr))

        link_attr = ja.JsonAttributes()
        link_attr.addAttribute(key, ja.StringValue("Link"))
        link_attr.addAttribute("visual", self.createVisualObject(), mandatory=False)

        link_arr = ja.ObjectList(key, self.object_stack, ja.JsonObject(self.object_stack, link_attr))

        joint_attr = ja.JsonAttributes()
        joint_attr.addAttribute(key, ja.StringValue("Joint"))
        joint_attr.addAttribute("type", ja.EnumValue(Link.Type))
        joint_attr.addAttribute("parent", ja.SelectedObject(ja.RelativeReference("././links")))
        joint_attr.addAttribute("child", ja.SelectedObject(ja.RelativeReference("././links")))
        joint_attr.addAttribute("axis", self.createAttributeArray(ja.FloatValue(), 3), title="Axis (XYZ)")
        joint_attr.addAttribute("origin", self.createAttributeArray(ja.FloatValue(), 3), title="Origin (XYZ)")
        joint_attr.addAttribute("lower_limit", ja.FloatValue(), title="Min offset")
        joint_attr.addAttribute("upper_limit", ja.FloatValue(), title="Max offset")
        joint_attr.addAttribute("home_offset", ja.FloatValue(), mandatory=False)

        joint_arr = ja.ObjectList(key, self.object_stack, ja.JsonObject(self.object_stack, joint_attr))

        positioner_attr = ja.JsonAttributes()
        positioner_attr.addAttribute(key, ja.StringValue("Positioner"))
        positioner_attr.addAttribute("base",
                                     self.createAttributeArray(ja.FloatValue(), 6, {
                                         "Translation (XYZ)": 3,
                                         "Orientation (XYZ)": 3
                                     }),
                                     mandatory=False)
        positioner_attr.addAttribute("tool",
                                     self.createAttributeArray(ja.FloatValue(), 6, {
                                         "Translation (XYZ)": 3,
                                         "Orientation (XYZ)": 3
                                     }),
                                     mandatory=False)
        positioner_attr.addAttribute("custom_order", ja.ObjectOrder(ja.RelativeReference("joints")), mandatory=False)
        positioner_attr.addAttribute("joints", joint_arr)
        positioner_attr.addAttribute("links", link_arr)

        positioner_arr = ja.ObjectList(key, self.object_stack, ja.JsonObject(self.object_stack, positioner_attr))

        jaws_attr = ja.JsonAttributes()
        jaws_attr.addAttribute("aperture", self.createAttributeArray(ja.FloatValue(), 2), title="Aperture (XY)")
        jaws_attr.addAttribute("aperture_lower_limit",
                               self.createAttributeArray(ja.FloatValue(), 2),
                               title="Aperture Min (XY)")
        jaws_attr.addAttribute("aperture_upper_limit",
                               self.createAttributeArray(ja.FloatValue(), 2),
                               title="Aperture Max (XY)")
        jaws_attr.addAttribute("beam_direction",
                               self.createAttributeArray(ja.FloatValue(), 3),
                               title="Beam Direction (XYZ)")
        jaws_attr.addAttribute("beam_source", self.createAttributeArray(ja.FloatValue(), 3), title="Beam Source (XYZ)")
        jaws_attr.addAttribute("positioner", ja.SelectedObject(ja.RelativeReference("./positioners")), mandatory=False)
        jaws_attr.addAttribute("visual", self.createVisualObject())

        jaws_object = ja.JsonObject(self.object_stack, jaws_attr)

        collimator_attr = ja.JsonAttributes()
        collimator_attr.addAttribute(key, ja.StringValue("Collimator"))
        collimator_attr.addAttribute("aperture", self.createAttributeArray(ja.FloatValue(), 2))
        collimator_attr.addAttribute("visual", self.createVisualObject())

        collimator_arr = ja.ObjectList(key, self.object_stack, ja.JsonObject(self.object_stack, collimator_attr))

        detector_attr = ja.JsonAttributes()
        detector_attr.addAttribute(key, ja.StringValue("Detector"))
        detector_attr.addAttribute("collimators", collimator_arr)
        detector_attr.addAttribute("default_collimator",
                                   ja.SelectedObject(ja.RelativeReference("collimators")),
                                   mandatory=False)
        detector_attr.addAttribute("diffracted_beam", self.createAttributeArray(ja.FloatValue(), 3))
        detector_attr.addAttribute("positioner",
                                   ja.SelectedObject(ja.RelativeReference("././positioners")),
                                   mandatory=False)

        detector_arr = ja.ObjectList(key, self.object_stack, ja.JsonObject(self.object_stack, detector_attr))

        positioning_stack_attr = ja.JsonAttributes()
        positioning_stack_attr.addAttribute(key, ja.StringValue("Positioning stack"))
        positioning_stack_attr.addAttribute("positioners",
                                            ja.ObjectOrder(ja.RelativeReference("././positioners"), include_all=False))

        positioning_stack_arr = ja.ObjectList(key, self.object_stack,
                                              ja.JsonObject(self.object_stack, positioning_stack_attr))

        instrument_attr = ja.JsonAttributes()
        instrument_attr.addAttribute("name", ja.StringValue("Instrument"))
        instrument_attr.addAttribute("version", ja.StringValue())
        instrument_attr.addAttribute("script_template", self.createFileValue(SOURCE_PATH), mandatory=False)
        instrument_attr.addAttribute("gauge_volume",
                                     self.createAttributeArray(ja.FloatValue(), 3),
                                     title="Gauge Volume (XYZ)")
        instrument_attr.addAttribute("incident_jaws", jaws_object)
        instrument_attr.addAttribute("detectors", detector_arr)
        instrument_attr.addAttribute("positioning_stacks", positioning_stack_arr)
        instrument_attr.addAttribute("positioners", positioner_arr)
        instrument_attr.addAttribute("fixed_hardware", fixed_hardware_arr, mandatory=False)

        instrument_obj = ja.JsonObject(self.object_stack, instrument_attr)

        return instrument_obj

    def resetInstrument(self):
        """Sets all values of the instrument to their defaults - usually empty"""
        self.instrument = self.createSchema()
        self.instrument.resolveReferences()
        self.instrument.been_set.connect(self.dataChanged)

        self.object_stack.object_stack = []
        self.object_stack.addObject("instrument", self.instrument)

    def getJsonFile(self):
        """Returns dictionary, representing a json object created from the data in the designer
        :return: the json dictionary
        :rtype: Dict[str: Any]
        """
        json_dict = self.instrument.json_value

        json_dict["collimators"] = []
        for detector in json_dict["detectors"]:
            for collimator in detector["collimators"]:
                collimator["detector"] = detector["name"]
                json_dict["collimators"].append(collimator)
            del detector["collimators"]

        return json.dumps({"instrument": json_dict}, indent=4)

    def setJsonFile(self, text):
        """Loads the text representing json file into the schema to set the relevant data
        :param text: the json file text
        :type text: str
        """
        # Here also change the json schema by moving collimators into the detector objects
        try:
            json_dict = json.loads(text)
        except Exception:
            return

        if "instrument" in json_dict:
            instrument_dict = json.loads(text)["instrument"]
        else:
            return

        if "detectors" in instrument_dict:
            for detector in instrument_dict["detectors"]:
                detector["collimators"] = [{key: value
                                            for key, value in collimator.items() if key != "detector"}
                                            for collimator in instrument_dict["collimators"]
                                            if collimator["detector"] == detector["name"]]
        del instrument_dict["collimators"]

        self.instrument.json_value = instrument_dict
        self.instrument.resolveReferences()
        self.createUi()
        print("Exit designer class")

    def createUi(self):
        """Updates the UI according to the top object in the stack"""
        self.attributes_panel.setParent(None)
        self.attributes_panel = self.object_stack.top().createPanel()
        self.layout.insertWidget(1, self.attributes_panel)
