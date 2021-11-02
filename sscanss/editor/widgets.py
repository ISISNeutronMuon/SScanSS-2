"""
Classes for instrument controls
"""
import math
import numpy as np
from PyQt5 import QtCore, QtWidgets


class ScriptWidget(QtWidgets.QWidget):
    """Creates a widget for viewing the result of the script renderer.

    :param parent: main window instance
    :type parent: MainWindow
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.instrument = self.parent.instrument
        self.template = self.instrument.script
        self.results = np.random.rand(10, self.instrument.positioning_stack.numberOfLinks)
        self.createTemplateKeys()

        main_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        show_mu_amps = self.template.Key.mu_amps.value in self.template.keys
        layout.addWidget(QtWidgets.QLabel("Duration of Measurements (microamps):"))
        self.micro_amp_textbox = QtWidgets.QDoubleSpinBox()
        value = self.template.keys[self.template.Key.mu_amps.value] if show_mu_amps else "0.000"
        self.micro_amp_textbox.valueFromText(value)
        self.micro_amp_textbox.setDecimals(3)
        self.micro_amp_textbox.valueChanged.connect(self.renderScript)

        if not show_mu_amps:
            self.micro_amp_textbox.setVisible(show_mu_amps)
        layout.addWidget(self.micro_amp_textbox)
        main_layout.addLayout(layout)

        self.preview_label = QtWidgets.QTextEdit()
        self.preview_label.setReadOnly(True)
        main_layout.addWidget(self.preview_label)
        self.renderScript()
        self.setLayout(main_layout)

    def updateScript(self):
        """Updates rendered script"""
        show_mu_amps = self.template.Key.mu_amps.value in self.template.keys
        if self.micro_amp_textbox.isVisible() != show_mu_amps:
            self.micro_amp_textbox.setVisible(show_mu_amps)
        self.micro_amp_textbox.setVisible(self.template.Key.mu_amps.value in self.template.keys)
        self.results = np.random.rand(10, self.instrument.positioning_stack.numberOfLinks)
        self.createTemplateKeys()
        self.renderScript()

    def createTemplateKeys(self):
        """Initializes the keys from the script template"""
        temp = {
            self.template.Key.script.value: [],
            self.template.Key.position.value: "",
            self.template.Key.filename.value: self.parent.filename,
            self.template.Key.mu_amps.value: "0.000",
            self.template.Key.count.value: len(self.results),
        }

        header = "\t".join(self.template.header_order)
        links = self.instrument.positioning_stack.links
        joint_labels = [links[order].name for order in self.instrument.positioning_stack.order]
        temp[self.template.Key.header.value] = header.replace(
            self.template.Key.position.value, "\t".join(joint_labels), 1
        )

        for key in self.template.keys:
            self.template.keys[key] = temp[key]

    def renderScript(self):
        """Renders script with template"""
        key = self.template.Key
        script = []
        for i in range(len(self.results)):
            script.append({key.position.value: "\t".join("{:.3f}".format(res) for res in self.results[i])})

        if self.template.Key.mu_amps.value in self.template.keys:
            self.template.keys[key.mu_amps.value] = self.micro_amp_textbox.text()
        self.template.keys[key.script.value] = script
        self.preview_label.setText(self.template.render())


class DetectorWidget(QtWidgets.QWidget):
    """Creates a widget for changing a detector's collimator and controlling the
    positioner if available.

    :param parent: main window instance
    :type parent: MainWindow
    :param detector_name: name of detector
    :type detector_name: str
    """

    collimator_changed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent, detector_name):
        super().__init__(parent)
        self.parent = parent

        self.instrument = self.parent.instrument
        self.main_layout = QtWidgets.QVBoxLayout()

        self.name = detector_name
        self.detector = self.instrument.detectors[detector_name]

        layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(layout)
        layout.addWidget(QtWidgets.QLabel("Collimators: "))
        self.combobox = QtWidgets.QComboBox()
        self.combobox.setView(QtWidgets.QListView())
        self.combobox.addItems([str(None), *self.detector.collimators.keys()])
        self.combobox.activated.connect(self.changeCollimator)
        self.combobox.setCurrentText(self.collimator_name)

        layout.addWidget(self.combobox)
        self.main_layout.addSpacing(10)

        if self.detector.positioner is not None:
            self.createPositionerForm()

        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

    @property
    def collimator_name(self):
        """Gets name of current collimator

        :return: name of collimator
        :rtype: str
        """
        return str(None) if self.detector.current_collimator is None else self.detector.current_collimator.name

    def changeCollimator(self):
        """Changes collimator to the selected one"""
        index = self.combobox.currentIndex()
        collimator = None if index == 0 else self.combobox.currentText()
        self.detector.current_collimator = collimator
        self.parent.scene.updateInstrumentScene()
        self.collimator_changed.emit(self.name, self.combobox.currentText())

    def createPositionerForm(self):
        """Creates form inputs for the detector's positioner"""
        self.position_forms = []
        title = QtWidgets.QLabel(f"{self.detector.name} Position")
        self.main_layout.addWidget(title)
        self.position_form = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.position_form.setLayout(layout)

        for index in self.detector.positioner.order:
            link = self.detector.positioner.links[index]
            if link.type == link.Type.Revolute:
                unit = "degrees"
                offset = math.degrees(link.set_point)
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                unit = "mm"
                offset = link.set_point
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            sub_layout = QtWidgets.QHBoxLayout()
            layout.addLayout(sub_layout)
            sub_layout.addWidget(QtWidgets.QLabel(f"{link.name.title()} ({unit}):"))
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(lower_limit, upper_limit)
            control.setDecimals(3)
            control.setValue(offset)
            sub_layout.addWidget(control)
            self.position_forms.append(control)

        self.main_layout.addWidget(self.position_form)
        button_layout = QtWidgets.QHBoxLayout()
        self.move_detector_button = QtWidgets.QPushButton("Move Detector")
        self.move_detector_button.clicked.connect(self.moveDetectorsButtonClicked)
        button_layout.addWidget(self.move_detector_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def moveDetectorsButtonClicked(self):
        values = [control.value() for control in self.position_forms]
        move_to = self.detector.positioner.fromUserFormat(values)

        move_from = self.detector.positioner.set_points
        if move_to != move_from:
            stack = self.detector.positioner
            stack.set_points = move_to
            self.parent.moveInstrument(lambda q, s=stack: s.fkine(q, setpoint=False), move_from, move_to)


class JawsWidget(QtWidgets.QWidget):
    """Creates a widget for setting the jaws' aperture and controlling the
    positioner if available.

    :param parent: main window instance
    :type parent: MainWindow
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.instrument = self.parent.instrument
        self.main_layout = QtWidgets.QVBoxLayout()
        self.createApertureForm()
        self.main_layout.addSpacing(40)
        if self.instrument.jaws.positioner is not None:
            self.createPositionerForm()

        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

    def createPositionerForm(self):
        """Creates form inputs for the jaws' positioner"""
        self.position_forms = []
        title = QtWidgets.QLabel(f"{self.instrument.jaws.name} Position")
        self.main_layout.addWidget(title)
        self.position_form = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.position_form.setLayout(layout)

        for index in self.instrument.jaws.positioner.order:
            link = self.instrument.jaws.positioner.links[index]
            if link.type == link.Type.Revolute:
                unit = "degrees"
                offset = math.degrees(link.set_point)
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                unit = "mm"
                offset = link.set_point
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            sub_layout = QtWidgets.QHBoxLayout()
            layout.addLayout(sub_layout)
            sub_layout.addWidget(QtWidgets.QLabel(f"{link.name.title()} ({unit}):"))
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(lower_limit, upper_limit)
            control.setDecimals(3)
            control.setValue(offset)
            sub_layout.addWidget(control)
            self.position_forms.append(control)

        self.main_layout.addWidget(self.position_form)
        button_layout = QtWidgets.QHBoxLayout()
        self.move_jaws_button = QtWidgets.QPushButton("Move Jaws")
        self.move_jaws_button.clicked.connect(self.moveJawsButtonClicked)
        button_layout.addWidget(self.move_jaws_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def createApertureForm(self):
        """Creates form inputs for jaws aperture size"""
        self.aperture_forms = []
        title = QtWidgets.QLabel(f"{self.instrument.jaws.name} Aperture Size")
        self.main_layout.addWidget(title)
        aperture = self.instrument.jaws.aperture
        upper_limit = self.instrument.jaws.aperture_upper_limit
        lower_limit = self.instrument.jaws.aperture_lower_limit

        self.aperture_form = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.aperture_form.setLayout(layout)

        sub_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(sub_layout)
        sub_layout.addWidget(QtWidgets.QLabel(f"Horizontal Aperture Size (mm):"))
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(lower_limit[0], upper_limit[0])
        control.setDecimals(3)
        control.setValue(aperture[0])
        sub_layout.addWidget(control)
        self.aperture_forms.append(control)

        sub_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(sub_layout)
        sub_layout.addWidget(QtWidgets.QLabel(f"Vertical Aperture Size (mm):"))
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(lower_limit[1], upper_limit[1])
        control.setDecimals(3)
        control.setValue(aperture[1])
        sub_layout.addWidget(control)
        self.aperture_forms.append(control)

        self.main_layout.addWidget(self.aperture_form)

        button_layout = QtWidgets.QHBoxLayout()
        self.change_aperture_button = QtWidgets.QPushButton("Change Aperture Size")
        self.change_aperture_button.clicked.connect(self.changeApertureButtonClicked)
        button_layout.addWidget(self.change_aperture_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def moveJawsButtonClicked(self):
        values = [control.value() for control in self.position_forms]
        move_to = self.instrument.jaws.positioner.fromUserFormat(values)

        move_from = self.instrument.jaws.positioner.set_points
        if move_to != move_from:
            stack = self.instrument.jaws.positioner
            stack.set_points = move_to
            self.parent.moveInstrument(lambda q, s=stack: s.fkine(q, setpoint=False), move_from, move_to)

    def changeApertureButtonClicked(self):
        self.instrument.jaws.aperture = [self.aperture_forms[0].value(), self.aperture_forms[1].value()]
        self.parent.scene.updateInstrumentScene()


class PositionerWidget(QtWidgets.QWidget):
    """Creates a widget for setting and controlling the positioning stacks.

    :param parent: main window instance
    :type parent: MainWindow
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.instrument = self.parent.instrument
        self.main_layout = QtWidgets.QVBoxLayout()

        stack_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(stack_layout)

        self.stack_combobox = QtWidgets.QComboBox()
        self.stack_combobox.setView(QtWidgets.QListView())
        self.stack_combobox.addItems(self.instrument.positioning_stacks.keys())
        self.stack_combobox.setCurrentText(self.instrument.positioning_stack.name)
        self.stack_combobox.activated[str].connect(self.changeStack)

        if len(self.instrument.positioning_stacks) > 1:
            stack_layout.addWidget(QtWidgets.QLabel("Positioning Stack:"))
            stack_layout.addWidget(self.stack_combobox)
            stack_layout.addStretch(1)
            self.main_layout.addSpacing(10)

        self.positioner_forms_layout = QtWidgets.QVBoxLayout()
        self.positioner_forms_layout.setContentsMargins(0, 0, 0, 0)

        self.createForms()
        self.main_layout.addLayout(self.positioner_forms_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.move_joints_button = QtWidgets.QPushButton("Move Joints")
        self.move_joints_button.clicked.connect(self.moveJointsButtonClicked)
        button_layout.addWidget(self.move_joints_button)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

    def changeStack(self, selected):
        """Changes the active positioning stack

        :param selected: name of selected stack
        :type selected: str
        """
        if selected != self.instrument.positioning_stack.name:
            self.parent.instrument.loadPositioningStack(selected)
            self.createForms()
            self.parent.scene.updateInstrumentScene()

    def createForms(self):
        """Creates form inputs for main and auxiliary positioners in the positioning stack"""
        for i in range(self.positioner_forms_layout.count()):
            widget = self.positioner_forms_layout.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()

        self.positioner_forms = []
        self.positioner_form_controls = []
        self.base_reset_buttons = {}

        positioner = self.instrument.positioning_stack.fixed
        widget = self.createPositionerWidget(positioner)
        self.positioner_forms_layout.addWidget(widget)

        for aux in self.instrument.positioning_stack.auxiliary:
            widget = self.createPositionerWidget(aux)
            self.positioner_forms_layout.addWidget(widget)

    def createPositionerWidget(self, positioner):
        """Creates a form widget for the given positioner that is used to set the
        values of the joints of the positioner.

        :param positioner: positioner
        :type positioner: SerialManipulator
        :return: positioner widget
        :rtype: QtWidgets.QWidget
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        title = QtWidgets.QLabel(positioner.name)
        layout.addWidget(title)

        for index in positioner.order:
            link = positioner.links[index]
            if link.type == link.Type.Revolute:
                unit = "degrees"
                offset = math.degrees(link.set_point)
                lower_limit = math.degrees(link.lower_limit)
                upper_limit = math.degrees(link.upper_limit)
            else:
                unit = "mm"
                offset = link.set_point
                lower_limit = link.lower_limit
                upper_limit = link.upper_limit

            sub_layout = QtWidgets.QHBoxLayout()
            sub_layout.addWidget(QtWidgets.QLabel(f"{link.name.title()} ({unit})"))
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(lower_limit, upper_limit)
            control.setDecimals(3)
            control.setValue(offset)
            sub_layout.addWidget(control)

            layout.addLayout(sub_layout)
            self.positioner_form_controls.append(control)

        widget.setLayout(layout)

        return widget

    def moveJointsButtonClicked(self):
        values = [control.value() for control in self.positioner_form_controls]
        move_to = self.instrument.positioning_stack.fromUserFormat(values)

        move_from = self.instrument.positioning_stack.set_points
        if move_to != move_from:
            stack = self.instrument.positioning_stack
            stack.set_points = move_to
            self.parent.moveInstrument(lambda q, s=stack: s.fkine(q, setpoint=False), move_from, move_to)
