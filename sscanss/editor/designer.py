import contextlib
from enum import Enum, unique
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.util import Accordion, ColourPicker, FilePicker, to_float, FormTitle, Pane


class Designer(QtWidgets.QWidget):
    """Creates a UI for modifying components of the instrument description"""
    json_updated = QtCore.pyqtSignal(object)

    @unique
    class Component(Enum):
        """Instrument Description Components"""
        General = 'General'
        Jaws = 'Incident Jaws'
        Visual = 'Visual'
        Detector = 'Detector'
        Collimator = 'Collimator'
        FixedHardware = 'Fixed Hardware'
        PositioningStacks = 'Positioning Stacks'
        Positioners = 'Positioners'

    def __init__(self, parent):
        super().__init__(parent)

        self.json = {}
        self.folder_path = '.'

        self.layout = QtWidgets.QVBoxLayout()

        self.add_update_button = QtWidgets.QPushButton('Update Entry')
        self.add_update_button.clicked.connect(self.updateJson)
        self.component = None

        self.title = FormTitle('')
        self.title.addHeaderControl(self.add_update_button)

        self.layout.addWidget(self.title)
        self.setLayout(self.layout)

    def clear(self):
        self.json = {}
        self.folder_path = '.'
        if self.component is not None:
            # RESET FIX IS HERE
            # self.component.reset()
            self.component.updateValue(self.json, self.folder_path)

    def setComponent(self, component_type):
        """Sets the current component to one of the specified type

        :param component_type: component type
        :type component_type: Designer.Component
        """
        if self.component is not None:
            if self.component.type == component_type:
                return

            self.layout.removeWidget(self.component)
            self.component.hide()
            self.component.deleteLater()

        self.title.label.setText(component_type.value)

        # Replace stretch if necessary
        if not self.layout.itemAt(1):
            self.layout.insertStretch(1, 1)

        if component_type == Designer.Component.Jaws:
            self.component = JawComponent()
        elif component_type == Designer.Component.General:
            self.component = GeneralComponent()
        elif component_type == Designer.Component.Detector:
            self.component = DetectorComponent()
        elif component_type == Designer.Component.Collimator:
            self.component = CollimatorComponent()
        elif component_type == Designer.Component.FixedHardware:
            self.component = FixedHardwareComponent()
        elif component_type == Designer.Component.PositioningStacks:
            self.component = PositioningStacksComponent()
        elif component_type == Designer.Component.Positioners:
            self.component = PositionersComponent()
            # Remove stretch to allow accordion object to take up all remaining space
            self.layout.removeItem(self.layout.itemAt(1))

        self.layout.insertWidget(1, self.component)

    def setJson(self, json_data):
        """Sets the json data of the designer

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        """
        if self.component is None:
            return

        self.json = json_data
        self.component.updateValue(self.json, self.folder_path)

    def updateJson(self):
        """Updates the json from the current component"""
        if self.component is None:
            return

        if not self.component.validate():
            return

        json_data = self.json
        if json_data.get('instrument') is None:
            json_data = {'instrument': {}}

        json_data['instrument'].update(self.component.value())
        self.json_updated.emit(json_data)


class VisualSubComponent(QtWidgets.QWidget):
    """Creates a UI for modifying visual subcomponent of the instrument description"""
    def __init__(self):
        super().__init__()

        self.key = 'visual'

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        box = QtWidgets.QGroupBox('Visual')
        main_layout.addWidget(box)
        layout = QtWidgets.QGridLayout()
        box.setLayout(layout)

        self.x_translation = create_validated_line_edit(3, '0.0')
        self.y_translation = create_validated_line_edit(3, '0.0')
        self.z_translation = create_validated_line_edit(3, '0.0')
        sub_layout = xyz_hbox_layout(self.x_translation, self.y_translation, self.z_translation)

        layout.addWidget(QtWidgets.QLabel('Pose (Translation): '), 0, 0)
        layout.addLayout(sub_layout, 0, 1)

        self.x_orientation = create_validated_line_edit(3, '0.0')
        self.y_orientation = create_validated_line_edit(3, '0.0')
        self.z_orientation = create_validated_line_edit(3, '0.0')
        sub_layout = xyz_hbox_layout(self.x_orientation, self.y_orientation, self.z_orientation)

        layout.addWidget(QtWidgets.QLabel('Pose (Orientation): '), 1, 0)
        layout.addLayout(sub_layout, 1, 1)

        self.colour_picker = ColourPicker(QtGui.QColor(QtCore.Qt.black))
        layout.addWidget(QtWidgets.QLabel('Colour: '), 2, 0)
        layout.addWidget(self.colour_picker, 2, 1)
        self.file_picker = FilePicker('', filters='3D Files (*.stl *.obj)', relative_source='.')
        layout.addWidget(QtWidgets.QLabel('Mesh: '), 3, 0)
        layout.addWidget(self.file_picker, 3, 1)

        self.validation_label = QtWidgets.QLabel()
        self.validation_label.setStyleSheet('color: red')
        layout.addWidget(self.validation_label, 3, 2)

    def reset(self):
        """Reset widgets to default values and validation state"""
        self.file_picker.file_view.clear()
        self.file_picker.file_view.setStyleSheet('')
        self.validation_label.setText('')

        self.colour_picker.value = QtGui.QColor(QtCore.Qt.black)

        self.x_translation.setText('0.0')
        self.y_translation.setText('0.0')
        self.z_translation.setText('0.0')
        self.x_orientation.setText('0.0')
        self.y_orientation.setText('0.0')
        self.z_orientation.setText('0.0')

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        if not self.file_picker.value:
            self.file_picker.file_view.setStyleSheet('border: 1px solid red;')
            self.validation_label.setText('Required!')
            return False

        self.file_picker.file_view.setStyleSheet('')
        self.validation_label.setText('')
        return True

    def updateValue(self, json_data, folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        :param folder_path: path to instrument file folder
        :type folder_path: str
        """
        self.reset()

        pose = json_data.get('pose')
        if pose is not None:
            self.x_translation.setText(f"{safe_get_value(pose, 0, '0.0')}")
            self.y_translation.setText(f"{safe_get_value(pose, 1, '0.0')}")
            self.z_translation.setText(f"{safe_get_value(pose, 2, '0.0')}")
            self.x_orientation.setText(f"{safe_get_value(pose, 3, '0.0')}")
            self.y_orientation.setText(f"{safe_get_value(pose, 4, '0.0')}")
            self.z_orientation.setText(f"{safe_get_value(pose, 5, '0.0')}")

        colour = json_data.get('colour')
        if colour is not None:
            tmp = [0., 0., 0.]
            tmp[0] = safe_get_value(colour, 0, 0)
            tmp[1] = safe_get_value(colour, 1, 0)
            tmp[2] = safe_get_value(colour, 2, 0)
            colour = QtGui.QColor.fromRgbF(*tmp)
            self.colour_picker.value = colour

        mesh_path = json_data.get('mesh', '')
        self.file_picker.relative_source = folder_path
        if mesh_path and isinstance(mesh_path, str):
            self.file_picker.value = mesh_path

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        tx, ty, tz = self.x_translation.text(), self.y_translation.text(), self.z_translation.text()
        ox, oy, oz = self.x_orientation.text(), self.y_orientation.text(), self.z_orientation.text()
        if tx and ty and tz and ox and oy and oz:
            pose = [float(tx), float(ty), float(tz), float(ox), float(oy), float(oz)]
            if pose != [0] * 6:
                json_data['pose'] = pose

        colour = self.colour_picker.value
        if colour.name() != '#000000':
            json_data['colour'] = [round(colour.redF(), 2), round(colour.greenF(), 2), round(colour.blueF(), 2)]

        mesh = self.file_picker.value
        if mesh:
            json_data['mesh'] = self.file_picker.value

        return {self.key: json_data}


def create_validated_line_edit(decimal=3, text=''):
    """Creates a line edit with a number validator

    :param decimal: number of decimal place allowed
    :type decimal: int
    :param text: initial text in line edit
    :type text: str
    :return: line edit with validator
    :rtype: QtWidgets.QLineEdit
    """
    line_edit = QtWidgets.QLineEdit()
    validator = QtGui.QDoubleValidator()
    validator.setDecimals(decimal)
    validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
    line_edit.setValidator(validator)
    line_edit.setText(text)

    return line_edit


def create_required_label():
    """Creates label with red text

    :return: label
    :rtype: PyQt5.QtWidgets.QLabel
    """
    label = QtWidgets.QLabel('')
    label.setStyleSheet('color: red')

    return label


def safe_get_value(array, index, default):
    """Gets given index from a floating array that could contain bad type
    or have the wrong length. The default value is returned in case of bad input.

    :param array: inputted array
    :type array: List[Any]
    :param index: index to get
    :type index: int
    :param default: default value
    :type default: Any
    :return: floating point value from array or default
    :rtype: Any
    """
    if len(array) > index:
        value = to_float(array[index])
        if value is not None:
            return value
    return default


def xy_hbox_layout(x_widget, y_widget, spacing=50):
    """Creates a horizontal sub layout consisting of two widgets labelled "X" and "Y".

    :param x_widget: The widget following the label "X"
    :type x_widget: PyQt5.QtWidgets.QWidget
    :param y_widget: The widget following the label "Y"
    :type y_widget: PyQt5.QtWidgets.QWidget
    :param spacing: spacing between the X and Y widgets. Default: 50
    :type spacing: int
    :return: A horizontal layout of the widgets with labels "X" and "Y"
    :rtype: QtWidgets.QHBoxLayout
    """
    sub_layout = QtWidgets.QHBoxLayout()
    sub_layout.addWidget(QtWidgets.QLabel('X: '))
    sub_layout.addWidget(x_widget)
    sub_layout.addSpacing(spacing)
    sub_layout.addWidget(QtWidgets.QLabel('Y: '))
    sub_layout.addWidget(y_widget)

    return sub_layout


def xyz_hbox_layout(x_widget, y_widget, z_widget, spacing=50):
    """Creates a horizontal sub layout consisting of three widgets labelled "X", "Y" and "Z".

    :param x_widget: The widget following the label "X"
    :type x_widget: PyQt5.QtWidgets.QWidget
    :param y_widget: The widget following the label "Y"
    :type y_widget: PyQt5.QtWidgets.QWidget
    :param z_widget: The widget following the label "Z"
    :type z_widget: PyQt5.QtWidgets.QWidget
    :param spacing: spacing between the X and Y, and Y and Z widgets. Default: 50
    :type spacing: int
    :return: A horizontal layout of the widgets with labels "X", "Y" and "Z"
    :rtype: QtWidgets.QHBoxLayout
    """
    sub_layout = QtWidgets.QHBoxLayout()
    sub_layout.addWidget(QtWidgets.QLabel('X: '))
    sub_layout.addWidget(x_widget)
    sub_layout.addSpacing(spacing)
    sub_layout.addWidget(QtWidgets.QLabel('Y: '))
    sub_layout.addWidget(y_widget)
    sub_layout.addSpacing(spacing)
    sub_layout.addWidget(QtWidgets.QLabel('Z: '))
    sub_layout.addWidget(z_widget)

    return sub_layout


class JawComponent(QtWidgets.QWidget):
    """Creates a UI for modifying jaws component of the instrument description"""
    def __init__(self):
        super().__init__()

        self.type = Designer.Component.Jaws
        self.key = 'incident_jaws'

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.x_aperture = create_validated_line_edit(3)
        self.y_aperture = create_validated_line_edit(3)
        sub_layout = xy_hbox_layout(self.x_aperture, self.y_aperture)

        layout.addWidget(QtWidgets.QLabel('Aperture: '), 0, 0)
        layout.addLayout(sub_layout, 0, 1)
        self.aperture_validation_label = create_required_label()
        layout.addWidget(self.aperture_validation_label, 0, 2)

        self.x_aperture_lower_limit = create_validated_line_edit(3)
        self.y_aperture_lower_limit = create_validated_line_edit(3)
        sub_layout = xy_hbox_layout(self.x_aperture_lower_limit, self.y_aperture_lower_limit)

        layout.addWidget(QtWidgets.QLabel('Aperture Lower Limit: '), 1, 0)
        layout.addLayout(sub_layout, 1, 1)
        self.aperture_lo_validation_label = create_required_label()
        layout.addWidget(self.aperture_lo_validation_label, 1, 2)

        self.x_aperture_upper_limit = create_validated_line_edit(3)
        self.y_aperture_upper_limit = create_validated_line_edit(3)
        sub_layout = xy_hbox_layout(self.x_aperture_upper_limit, self.y_aperture_upper_limit)

        layout.addWidget(QtWidgets.QLabel('Aperture Upper Limit: '), 2, 0)
        layout.addLayout(sub_layout, 2, 1)
        self.aperture_up_validation_label = create_required_label()
        layout.addWidget(self.aperture_up_validation_label, 2, 2)

        self.x_beam_source = create_validated_line_edit(3)
        self.y_beam_source = create_validated_line_edit(3)
        self.z_beam_source = create_validated_line_edit(3)
        sub_layout = xyz_hbox_layout(self.x_beam_source, self.y_beam_source, self.z_beam_source)

        layout.addWidget(QtWidgets.QLabel('Beam Source: '), 3, 0)
        layout.addLayout(sub_layout, 3, 1)
        self.beam_src_validation_label = create_required_label()
        layout.addWidget(self.beam_src_validation_label, 3, 2)

        self.x_beam_direction = create_validated_line_edit(7)
        self.y_beam_direction = create_validated_line_edit(7)
        self.z_beam_direction = create_validated_line_edit(7)
        sub_layout = xyz_hbox_layout(self.x_beam_direction, self.y_beam_direction, self.z_beam_direction)

        layout.addWidget(QtWidgets.QLabel('Beam Direction: '), 4, 0)
        layout.addLayout(sub_layout, 4, 1)
        self.beam_dir_validation_label = create_required_label()
        layout.addWidget(self.beam_dir_validation_label, 4, 2)

        self.positioner_combobox = QtWidgets.QComboBox()
        self.positioner_combobox.addItems(['None'])
        layout.addWidget(QtWidgets.QLabel('Positioner: '), 5, 0)
        layout.addWidget(self.positioner_combobox, 5, 1)

        self.visuals = VisualSubComponent()
        layout.addWidget(self.visuals, 6, 0, 1, 3)

    @property
    def __required_widgets(self):
        """Generates dict of required widget for validation. The key is the validation
        label and the value is a list of widget in the same row as the validation label

        :return: dict of labels and input widgets
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {
            self.aperture_validation_label: [self.x_aperture, self.y_aperture],
            self.aperture_lo_validation_label: [self.x_aperture_lower_limit, self.y_aperture_lower_limit],
            self.aperture_up_validation_label: [self.x_aperture_upper_limit, self.y_aperture_upper_limit],
            self.beam_src_validation_label: [self.x_beam_source, self.y_beam_source, self.z_beam_source],
            self.beam_dir_validation_label: [self.x_beam_direction, self.y_beam_direction, self.z_beam_direction]
        }

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, line_edits in self.__required_widgets.items():
            label.setText('')
            for line_edit in line_edits:
                line_edit.clear()
                line_edit.setStyleSheet('')

        self.visuals.reset()

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        widgets = self.__required_widgets
        valid = True
        for label, line_edits in widgets.items():
            row_valid = True
            for line_edit in line_edits:
                if not line_edit.text():
                    line_edit.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    line_edit.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        visual_valid = self.visuals.validate()

        if valid and visual_valid:
            for label, line_edits in widgets.items():
                label.setText('')
                for line_edit in line_edits:
                    line_edit.setStyleSheet('')
            return True
        return False

    def updateValue(self, json_data, folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        :param folder_path: path to instrument file folder
        :type folder_path: str
        """
        self.reset()
        instrument_data = json_data.get('instrument', {})
        jaws_data = instrument_data.get('incident_jaws', {})

        aperture = jaws_data.get('aperture')
        if aperture is not None:
            self.x_aperture.setText(f"{safe_get_value(aperture, 0, '')}")
            self.y_aperture.setText(f"{safe_get_value(aperture, 1, '')}")

        lo_limit = jaws_data.get('aperture_lower_limit')
        if lo_limit is not None:
            self.x_aperture_lower_limit.setText(f"{safe_get_value(lo_limit, 0, '')}")
            self.y_aperture_lower_limit.setText(f"{safe_get_value(lo_limit, 1, '')}")

        up_limit = jaws_data.get('aperture_upper_limit')
        if up_limit is not None:
            self.x_aperture_upper_limit.setText(f"{safe_get_value(up_limit, 0, '')}")
            self.y_aperture_upper_limit.setText(f"{safe_get_value(up_limit, 1, '')}")

        beam_dir = jaws_data.get('beam_direction')
        if beam_dir is not None:
            self.x_beam_direction.setText(f"{safe_get_value(beam_dir, 0, '')}")
            self.y_beam_direction.setText(f"{safe_get_value(beam_dir, 1, '')}")
            self.z_beam_direction.setText(f"{safe_get_value(beam_dir, 2, '')}")

        beam_src = jaws_data.get('beam_source')
        if beam_src is not None:
            self.x_beam_source.setText(f"{safe_get_value(beam_src, 0, '')}")
            self.y_beam_source.setText(f"{safe_get_value(beam_src, 1, '')}")
            self.z_beam_source.setText(f"{safe_get_value(beam_src, 2, '')}")

        positioners = []
        positioners_data = instrument_data.get('positioners', [])
        for data in positioners_data:
            name = data.get('name', '')
            if name:
                positioners.append(name)
        self.positioner_combobox.clear()
        self.positioner_combobox.addItems(['None', *positioners])
        positioner = jaws_data.get('positioner', 'None')
        if isinstance(positioner, str):
            self.positioner_combobox.setCurrentText(positioner)
        else:
            self.positioner_combobox.setCurrentIndex(0)

        self.visuals.updateValue(jaws_data.get('visual', {}), folder_path)

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        x, y = self.x_aperture.text(), self.y_aperture.text()
        if x and y:
            json_data['aperture'] = [float(x), float(y)]

        x, y = self.x_aperture_lower_limit.text(), self.y_aperture_lower_limit.text()
        if x and y:
            json_data['aperture_lower_limit'] = [float(x), float(y)]

        x, y = self.x_aperture_upper_limit.text(), self.y_aperture_upper_limit.text()
        if x and y:
            json_data['aperture_upper_limit'] = [float(x), float(y)]

        x, y, z = self.x_beam_source.text(), self.y_beam_source.text(), self.z_beam_source.text()
        if x and y and z:
            json_data['beam_source'] = [float(x), float(y), float(z)]

        x, y, z = self.x_beam_direction.text(), self.y_beam_direction.text(), self.z_beam_direction.text()
        if x and y and z:
            json_data['beam_direction'] = [float(x), float(y), float(z)]

        name = self.positioner_combobox.currentText()
        if name and name != 'None':
            json_data['positioner'] = name

        visual_data = self.visuals.value()
        if visual_data[self.visuals.key]:
            json_data.update(visual_data)
        return {self.key: json_data}


class GeneralComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the general component of the instrument description"""
    def __init__(self):
        super().__init__()

        self.type = Designer.Component.General
        self.key = ''

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.instrument_name = QtWidgets.QLineEdit()
        layout.addWidget(QtWidgets.QLabel('Name: '), 0, 0)
        layout.addWidget(self.instrument_name, 0, 1)
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 0, 2)

        self.file_version = QtWidgets.QLineEdit()
        layout.addWidget(QtWidgets.QLabel('Version: '), 1, 0)
        layout.addWidget(self.file_version, 1, 1)
        self.version_validation_label = create_required_label()
        layout.addWidget(self.version_validation_label, 1, 2)

        self.x_gauge_volume = create_validated_line_edit(3)
        self.y_gauge_volume = create_validated_line_edit(3)
        self.z_gauge_volume = create_validated_line_edit(3)
        sub_layout = xyz_hbox_layout(self.x_gauge_volume, self.y_gauge_volume, self.z_gauge_volume)

        layout.addWidget(QtWidgets.QLabel('Gauge Volume: '), 2, 0)
        layout.addLayout(sub_layout, 2, 1)
        self.gauge_vol_validation_label = create_required_label()
        layout.addWidget(self.gauge_vol_validation_label, 2, 2)

        self.script_picker = FilePicker('', filters='All Files (*)', relative_source='.')
        layout.addWidget(QtWidgets.QLabel('Script Template: '), 3, 0)
        layout.addWidget(self.script_picker, 3, 1)

    @property
    def __required_widgets(self):
        """Generates dict of required widget for validation. The key is the validation
        label and the value is a list of widget in the same row as the validation label

        :return: dict of labels and input widgets
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {
            self.name_validation_label: [self.instrument_name],
            self.version_validation_label: [self.file_version],
            self.gauge_vol_validation_label: [self.x_gauge_volume, self.y_gauge_volume, self.z_gauge_volume]
        }

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, line_edits in self.__required_widgets.items():
            label.setText('')
            for line_edit in line_edits:
                line_edit.clear()
                line_edit.setStyleSheet('')

        self.script_picker.file_view.clear()

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        widgets = self.__required_widgets
        valid = True
        for label, line_edits in widgets.items():
            row_valid = True
            for line_edit in line_edits:
                if not line_edit.text():
                    line_edit.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    line_edit.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        if valid:
            for label, line_edits in widgets.items():
                label.setText('')
                for line_edit in line_edits:
                    line_edit.setStyleSheet('')

        return valid

    def updateValue(self, json_data, folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        :param folder_path: path to instrument file folder
        :type folder_path: str
        """
        self.reset()
        instrument_data = json_data.get('instrument', {})

        name = instrument_data.get('name')
        if name is not None:
            self.instrument_name.setText(name)

        version = instrument_data.get('version')
        if version is not None:
            self.file_version.setText(version)

        gauge_volume = instrument_data.get('gauge_volume')
        if gauge_volume is not None:
            self.x_gauge_volume.setText(f"{safe_get_value(gauge_volume, 0, '')}")
            self.y_gauge_volume.setText(f"{safe_get_value(gauge_volume, 1, '')}")
            self.z_gauge_volume.setText(f"{safe_get_value(gauge_volume, 2, '')}")

        script_path = instrument_data.get('script_template', '')
        self.script_picker.relative_source = '.' if not folder_path else folder_path
        if script_path and isinstance(script_path, str):
            self.script_picker.value = script_path

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        name = self.instrument_name.text()
        if name:
            json_data['name'] = name

        version = self.file_version.text()
        if version:
            json_data['version'] = version

        x, y, z = self.x_gauge_volume.text(), self.y_gauge_volume.text(), self.z_gauge_volume.text()
        if x and y and z:
            json_data['gauge_volume'] = [float(x), float(y), float(z)]

        file_path = self.script_picker.value
        if file_path:
            json_data['script_template'] = file_path
        return json_data


class DetectorComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the detector component of the instrument description"""
    def __init__(self):
        super().__init__()

        self.type = Designer.Component.Detector
        self.key = 'detectors'
        self.collimator_key = 'collimators'

        self.json = {}
        self.folder_path = '.'
        self.previous_name = ''
        self.add_new_text = 'Add New...'
        self.detector_list = []
        self.collimator_list = []

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Name field - string, required -- combobox chooses between detectors, and allows renaming
        self.detector_name_combobox = QtWidgets.QComboBox()
        self.detector_name_combobox.setEditable(True)
        layout.addWidget(QtWidgets.QLabel('Name: '), 0, 0)
        layout.addWidget(self.detector_name_combobox, 0, 1)
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 0, 2)

        # When the detector is changed, connect to a slot that updates the detector parameters in the component.
        # The "activated" signal is emitted only when the user selects an option (not programmatically) and is also
        # emitted when the user re-selects the same option.
        self.detector_name_combobox.activated.connect(lambda: self.updateValue(self.json, self.folder_path))

        # Default Collimator field - string from list, optional
        self.default_collimator_combobox = QtWidgets.QComboBox()
        self.default_collimator_combobox.addItems(['None'])
        layout.addWidget(QtWidgets.QLabel('Default Collimator: '), 1, 0)
        layout.addWidget(self.default_collimator_combobox, 1, 1)

        # Diffracted Beam field - array of floats, required
        self.x_diffracted_beam = create_validated_line_edit(3)
        self.y_diffracted_beam = create_validated_line_edit(3)
        self.z_diffracted_beam = create_validated_line_edit(3)
        sub_layout = xyz_hbox_layout(self.x_diffracted_beam, self.y_diffracted_beam, self.z_diffracted_beam)

        layout.addWidget(QtWidgets.QLabel('Diffracted Beam: '), 2, 0)
        layout.addLayout(sub_layout, 2, 1)
        self.diffracted_beam_validation_label = create_required_label()
        layout.addWidget(self.diffracted_beam_validation_label, 2, 2)

        # Positioner field - string from list, optional
        self.positioner_combobox = QtWidgets.QComboBox()
        self.positioner_combobox.addItems(['None'])
        layout.addWidget(QtWidgets.QLabel('Positioner: '), 3, 0)
        layout.addWidget(self.positioner_combobox, 3, 1)

    @property
    def __required_widgets(self):
        """Generates dict of required widget for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input widgets
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {
            self.diffracted_beam_validation_label:
            [self.x_diffracted_beam, self.y_diffracted_beam, self.z_diffracted_beam]
        }

    @property
    def __required_comboboxes(self):
        """Generates dict of required comboboxes for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input comboboxes
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.name_validation_label: [self.detector_name_combobox]}

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, line_edits in self.__required_widgets.items():
            label.setText('')
            for line_edit in line_edits:
                line_edit.clear()
                line_edit.setStyleSheet('')

        for label, comboboxes in self.__required_comboboxes.items():
            label.setText('')
            for combobox in comboboxes:
                combobox.setStyleSheet('')

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        valid = True

        widgets = self.__required_widgets
        for label, line_edits in widgets.items():
            row_valid = True
            for line_edit in line_edits:
                if not line_edit.text():
                    line_edit.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    line_edit.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        comboboxes = self.__required_comboboxes
        for label, boxes in comboboxes.items():
            row_valid = True
            for combobox in boxes:
                if not combobox.currentText():
                    combobox.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    combobox.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        if valid:
            for label, line_edits in widgets.items():
                label.setText('')
                for line_edit in line_edits:
                    line_edit.setStyleSheet('')
            for label, boxes in comboboxes.items():
                label.setText('')
                for combobox in boxes:
                    combobox.setStyleSheet('')

        return valid

    def setNewDetector(self):
        """ When the 'Add New...' option is chosen in the detector name combobox, clear the text."""
        self.detector_name_combobox.clearEditText()
        self.x_diffracted_beam.clear()
        self.y_diffracted_beam.clear()
        self.z_diffracted_beam.clear()

    def updateValue(self, json_data, _folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        """
        self.reset()
        self.json = json_data
        instrument_data = json_data.get('instrument', {})
        self.detector_list = instrument_data.get('detectors', [])
        self.collimator_list = instrument_data.get('collimators', [])

        try:
            detector_data = self.detector_list[max(self.detector_name_combobox.currentIndex(), 0)]
        except IndexError:
            detector_data = {}

        # Name combobox
        name = detector_data.get('name')
        # Need to track the name of the detector in case it changes when "value()" is next called
        self.previous_name = name
        if name is not None:
            self.detector_name_combobox.setCurrentText(name)

        detectors = []
        for data in self.detector_list:
            name = data.get('name', '')
            if name:
                detectors.append(name)

        # Rewrite the combobox to contain the new list of detectors, and reset the index to the current value
        index = max(self.detector_name_combobox.currentIndex(), 0)
        self.detector_name_combobox.clear()
        self.detector_name_combobox.addItems([*detectors, self.add_new_text])
        self.detector_name_combobox.setCurrentIndex(index)
        if self.detector_name_combobox.currentText() == self.add_new_text:
            self.setNewDetector()

        # Default collimator combobox
        # NOTE -- if the detector name is changed in the JSON directly, the list of collimators for the detector will
        #         NOT be updated. However, if "value()" is called immediately prior to this routine (via the
        #         "Update Entry" button in the editor) then the collimators for this detector WILL have been updated.
        collimators = []
        collimators_data = instrument_data.get('collimators', [])
        for data in collimators_data:
            collimator_name = data.get('name', '')
            detector = data.get('detector', '')
            if collimator_name and detector == self.detector_name_combobox.currentText():
                collimators.append(collimator_name)
        self.default_collimator_combobox.clear()
        self.default_collimator_combobox.addItems(['None', *collimators])
        default_collimator = detector_data.get('default_collimator', 'None')
        if isinstance(default_collimator, str):
            self.default_collimator_combobox.setCurrentText(default_collimator)
        else:
            self.default_collimator_combobox.setCurrentIndex(0)

        # Diffracted Beam line edit
        diffracted_beam = detector_data.get('diffracted_beam')
        if diffracted_beam is not None:
            self.x_diffracted_beam.setText(f"{safe_get_value(diffracted_beam, 0, '')}")
            self.y_diffracted_beam.setText(f"{safe_get_value(diffracted_beam, 1, '')}")
            self.z_diffracted_beam.setText(f"{safe_get_value(diffracted_beam, 2, '')}")

        # Positioners combobox
        positioners = []
        positioners_data = instrument_data.get('positioners', [])
        for data in positioners_data:
            positioner_name = data.get('name', '')
            if positioner_name:
                positioners.append(positioner_name)
        self.positioner_combobox.clear()
        self.positioner_combobox.addItems(['None', *positioners])
        positioner = detector_data.get('positioner', 'None')
        if isinstance(positioner, str):
            self.positioner_combobox.setCurrentText(positioner)
        else:
            self.positioner_combobox.setCurrentIndex(0)

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        name = self.detector_name_combobox.currentText()
        if name:
            json_data['name'] = name

            # Also update the detector name in each collimator
            for collimator in self.collimator_list:
                if collimator['detector'] == self.previous_name:
                    collimator['detector'] = name

            # With the detector and collimators correctly matched, we can reset this variable
            self.previous_name = name

        default_collimator = self.default_collimator_combobox.currentText()
        if default_collimator and default_collimator != 'None':
            json_data['default_collimator'] = default_collimator

        x, y, z = self.x_diffracted_beam.text(), self.y_diffracted_beam.text(), self.z_diffracted_beam.text()
        if x and y and z:
            json_data['diffracted_beam'] = [float(x), float(y), float(z)]

        positioner = self.positioner_combobox.currentText()
        if positioner and positioner != 'None':
            json_data['positioner'] = positioner

        # Place edited detector within the list of detectors
        try:
            self.detector_list[self.detector_name_combobox.currentIndex()] = json_data
        except IndexError:
            self.detector_list.append(json_data)

        # Return updated set of detectors and, if necessary, collimators
        if self.collimator_list:
            return {self.key: self.detector_list, self.collimator_key: self.collimator_list}
        else:
            return {self.key: self.detector_list}


class CollimatorComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the collimator component of the instrument description"""
    def __init__(self):
        super().__init__()

        self.type = Designer.Component.Collimator
        self.key = 'collimators'

        self.json = {}
        self.folder_path = '.'
        self.add_new_text = 'Add New...'
        self.collimator_list = []

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # The combobox chooses between collimators
        self.collimator_combobox = QtWidgets.QComboBox()
        layout.addWidget(QtWidgets.QLabel('Collimator: '), 0, 0)
        layout.addWidget(self.collimator_combobox, 0, 1)

        # When the collimator is changed, connect to a slot that updates the collimator parameters in the component.
        # The "activated" signal is emitted only when the user selects an option (not programmatically) and is also
        # emitted when the user re-selects the same option.
        self.collimator_combobox.activated.connect(lambda: self.updateValue(self.json, self.folder_path))

        # Name field - string, required
        self.collimator_name = QtWidgets.QLineEdit()
        layout.addWidget(QtWidgets.QLabel('Name: '), 1, 0)
        layout.addWidget(self.collimator_name, 1, 1)
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 1, 2)

        # Detector field - string from list, required
        self.detector_combobox = QtWidgets.QComboBox()
        self.detector_combobox.setEditable(True)
        layout.addWidget(QtWidgets.QLabel('Detector: '), 2, 0)
        layout.addWidget(self.detector_combobox, 2, 1)
        self.detector_validation_label = create_required_label()
        layout.addWidget(self.detector_validation_label, 2, 2)

        # The "activated" signal is emitted when the user re-selects the same option,
        # so we can ensure the "Add New..." text is cleared each time it is selected.
        self.detector_combobox.activated.connect(self.setNewDetector)

        # Aperture field - array of floats, required
        self.x_aperture = create_validated_line_edit(3)
        self.y_aperture = create_validated_line_edit(3)
        sub_layout = xy_hbox_layout(self.x_aperture, self.y_aperture)

        layout.addWidget(QtWidgets.QLabel('Aperture: '), 3, 0)
        layout.addLayout(sub_layout, 3, 1)
        self.aperture_validation_label = create_required_label()
        layout.addWidget(self.aperture_validation_label, 3, 2)

        # Visual field - visual object, required
        # The visual object contains: pose, colour, and mesh parameters
        self.visuals = VisualSubComponent()
        layout.addWidget(self.visuals, 4, 0, 1, 3)

    @property
    def __required_widgets(self):
        """Generates dict of required widget for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input widgets
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {
            self.name_validation_label: [self.collimator_name],
            self.aperture_validation_label: [self.x_aperture, self.y_aperture]
        }

    @property
    def __required_comboboxes(self):
        """Generates dict of required comboboxes for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input comboboxes
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.detector_validation_label: [self.detector_combobox]}

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, line_edits in self.__required_widgets.items():
            label.setText('')
            for line_edit in line_edits:
                line_edit.clear()
                line_edit.setStyleSheet('')

        for label, comboboxes in self.__required_comboboxes.items():
            label.setText('')
            for combobox in comboboxes:
                combobox.setStyleSheet('')

        self.visuals.reset()

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        valid = True

        widgets = self.__required_widgets
        for label, line_edits in widgets.items():
            row_valid = True
            for line_edit in line_edits:
                if not line_edit.text():
                    line_edit.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    line_edit.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        comboboxes = self.__required_comboboxes
        for label, boxes in comboboxes.items():
            row_valid = True
            for combobox in boxes:
                if not combobox.currentText():
                    combobox.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    combobox.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        visual_valid = self.visuals.validate()

        if valid and visual_valid:
            for label, line_edits in widgets.items():
                label.setText('')
                for line_edit in line_edits:
                    line_edit.setStyleSheet('')
            for label, boxes in comboboxes.items():
                label.setText('')
                for combobox in boxes:
                    combobox.setStyleSheet('')
            return True
        return False

    def setNewCollimator(self):
        """ When the 'Add New...' option is chosen in the collimator combobox, clear the text."""
        self.collimator_name.clear()
        self.detector_combobox.setCurrentIndex(self.detector_combobox.count() - 1)
        self.x_aperture.clear()
        self.y_aperture.clear()
        self.visuals.reset()

    def setNewDetector(self):
        """ When the 'Add New...' option is chosen in the detector combobox, clear the text."""
        if self.detector_combobox.currentText() == self.add_new_text:
            self.detector_combobox.clearEditText()

    def updateValue(self, json_data, folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        :param folder_path: path to instrument file folder
        :type folder_path: str
        """
        self.reset()
        self.json = json_data
        instrument_data = json_data.get('instrument', {})
        self.collimator_list = instrument_data.get('collimators', [])

        try:
            collimator_data = self.collimator_list[max(self.collimator_combobox.currentIndex(), 0)]
        except IndexError:
            collimator_data = {}

        # Collimators combobox
        collimators = []
        for index, data in enumerate(self.collimator_list):
            name = data.get('name', '')
            if name:
                collimators.append(f"Collimator {index + 1}")

        # Rewrite the combobox to contain the new list of collimators, and reset the index to the current value
        index = max(self.collimator_combobox.currentIndex(), 0)
        self.collimator_combobox.clear()
        self.collimator_combobox.addItems([*collimators, self.add_new_text])
        self.collimator_combobox.setCurrentIndex(index)
        if self.collimator_combobox.currentText() == self.add_new_text:
            self.setNewCollimator()

        # Name field
        name = collimator_data.get('name')
        if name is not None:
            self.collimator_name.setText(name)

        # Detectors combobox
        detectors = []
        detectors_data = instrument_data.get('detectors', [])
        for data in detectors_data:
            detector_name = data.get('name', '')
            if detector_name:
                detectors.append(detector_name)

        # Rewrite the combobox to contain the new list of detectors, and reset the index to the current value
        index = max(self.detector_combobox.currentIndex(), 0)
        self.detector_combobox.clear()
        self.detector_combobox.addItems([*detectors, self.add_new_text])
        self.detector_combobox.setCurrentIndex(index)
        if self.detector_combobox.currentText() == self.add_new_text:
            self.detector_combobox.clearEditText()

        detector = collimator_data.get('detector')
        if detector is not None:
            self.detector_combobox.setCurrentText(detector)

        # Aperture line edit
        aperture = collimator_data.get('aperture')
        if aperture is not None:
            self.x_aperture.setText(f"{safe_get_value(aperture, 0, '')}")
            self.y_aperture.setText(f"{safe_get_value(aperture, 1, '')}")

        # Visual object
        self.visuals.updateValue(collimator_data.get('visual', {}), folder_path)

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        name = self.collimator_name.text()
        if name:
            json_data['name'] = name

        detector = self.detector_combobox.currentText()
        if detector:
            json_data['detector'] = detector

        x, y = self.x_aperture.text(), self.y_aperture.text()
        if x and y:
            json_data['aperture'] = [float(x), float(y)]

        visual_data = self.visuals.value()
        if visual_data[self.visuals.key]:
            json_data.update(visual_data)

        # Place edited collimator within the list of detectors
        try:
            self.collimator_list[self.collimator_combobox.currentIndex()] = json_data
        except IndexError:
            self.collimator_list.append(json_data)

        # Return updated set of collimators
        return {self.key: self.collimator_list}


class FixedHardwareComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the fixed hardware component of the instrument description"""
    def __init__(self):
        super().__init__()

        self.type = Designer.Component.FixedHardware
        self.key = 'fixed_hardware'

        self.json = {}
        self.folder_path = '.'
        self.add_new_text = 'Add New...'
        self.hardware_list = []

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Name field - string, required
        self.name_combobox = QtWidgets.QComboBox()
        self.name_combobox.setEditable(True)
        layout.addWidget(QtWidgets.QLabel('Name: '), 0, 0)
        layout.addWidget(self.name_combobox, 0, 1)
        layout.setColumnStretch(1, 2)  # Stretches middle column to double width
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 0, 2)

        # When the fixed hardware component is changed, connect to a slot that updates the visual object in the
        # component. The "activated" signal is emitted only when the user selects an option (not programmatically)
        # and is also emitted when the user re-selects the same option.
        self.name_combobox.activated.connect(lambda: self.updateValue(self.json, self.folder_path))

        # Visual field - visual object, required
        # The visual object contains: pose, colour, and mesh parameters
        self.visuals = VisualSubComponent()
        layout.addWidget(self.visuals, 1, 0, 1, 3)

    @property
    def __required_comboboxes(self):
        """Generates dict of required comboboxes for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input comboboxes
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.name_validation_label: [self.name_combobox]}

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, comboboxes in self.__required_comboboxes.items():
            label.setText('')
            for combobox in comboboxes:
                combobox.setStyleSheet('')

        self.visuals.reset()

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        valid = True

        comboboxes = self.__required_comboboxes
        for label, boxes in comboboxes.items():
            row_valid = True
            for combobox in boxes:
                if not combobox.currentText():
                    combobox.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    combobox.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        visual_valid = self.visuals.validate()

        if valid and visual_valid:
            for label, boxes in comboboxes.items():
                label.setText('')
                for combobox in boxes:
                    combobox.setStyleSheet('')
            return True
        return False

    def updateValue(self, json_data, folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        :param folder_path: path to instrument file folder
        :type folder_path: str
        """
        self.reset()
        self.json = json_data
        instrument_data = json_data.get('instrument', {})
        self.hardware_list = instrument_data.get('fixed_hardware', [])

        try:
            hardware_data = self.hardware_list[max(self.name_combobox.currentIndex(), 0)]
        except IndexError:
            hardware_data = {}

        # Name combobox
        hardware = []
        for data in self.hardware_list:
            name = data.get('name', '')
            if name:
                hardware.append(name)

        # Rewrite the combobox to contain the new list of hardware, and reset the index to the current value
        index = max(self.name_combobox.currentIndex(), 0)
        self.name_combobox.clear()
        self.name_combobox.addItems([*hardware, self.add_new_text])
        self.name_combobox.setCurrentIndex(index)
        if self.name_combobox.currentText() == self.add_new_text:
            self.name_combobox.clearEditText()
            self.visuals.reset()

        name = hardware_data.get('name')
        if name is not None:
            self.name_combobox.setCurrentText(name)

        # Visual object
        self.visuals.updateValue(hardware_data.get('visual', {}), folder_path)

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        name = self.name_combobox.currentText()
        if name:
            json_data['name'] = name

        visual_data = self.visuals.value()
        if visual_data[self.visuals.key]:
            json_data.update(visual_data)

        # Place edited hardware within the list of fixed hardware components
        try:
            self.hardware_list[self.name_combobox.currentIndex()] = json_data
        except IndexError:
            self.hardware_list.append(json_data)

        # Return updated set of fixed hardware components
        return {self.key: self.hardware_list}


class PositioningStacksComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the positioning stacks component of the instrument description"""
    def __init__(self):
        super().__init__()

        self.type = Designer.Component.PositioningStacks
        self.key = 'positioning_stacks'

        self.json = {}
        self.folder_path = '.'
        self.add_new_text = 'Add New...'
        self.positioning_stack_list = []
        self.positioners_list = []

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Name field - string, required -- combobox chooses between stacks, and allows renaming
        self.name_combobox = QtWidgets.QComboBox()
        self.name_combobox.setEditable(True)
        layout.addWidget(QtWidgets.QLabel('Name: '), 0, 0)
        layout.addWidget(self.name_combobox, 0, 1)
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 0, 2)

        # When the positioning stack is changed, connect to a slot that updates the list of positioners in the
        # component. The "activated" signal is emitted only when the user selects an option (not programmatically)
        # and is also emitted when the user re-selects the same option.
        self.name_combobox.activated.connect(lambda: self.updateValue(self.json, self.folder_path))

        # Positioners field - string(s) from list
        self.positioners_combobox = QtWidgets.QComboBox()
        self.positioners_combobox.setEditable(True)
        layout.addWidget(QtWidgets.QLabel('Positioners: '), 1, 0)
        layout.addWidget(self.positioners_combobox, 1, 1)

        # The "activated" signal is emitted when the user re-selects the same option,
        # so we can ensure the "Add New..." text is cleared each time it is selected.
        self.positioners_combobox.activated.connect(self.setNewPositioner)
        # Need to include index change so programmatic changes of index are also accounted for
        self.positioners_combobox.currentIndexChanged.connect(self.setNewPositioner)

        # Display list of positioners in a QListWidget
        self.positioning_stack_box = QtWidgets.QListWidget()
        self.positioning_stack_box.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        layout.addWidget(self.positioning_stack_box, 2, 1)
        self.positioning_stack_validation_label = create_required_label()
        layout.addWidget(self.positioning_stack_validation_label, 2, 2)

        # Create buttons to add and remove entries from the positioners list
        self.add_button = QtWidgets.QPushButton('Add')
        self.add_button.clicked.connect(self.addNewItem)
        layout.addWidget(self.add_button, 1, 2)
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.clear_button.clicked.connect(self.clearList)
        layout.addWidget(self.clear_button, 2, 2, alignment=QtCore.Qt.AlignTop)

    @property
    def __required_widgets(self):
        """Generates dict of required widget for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input widgets
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.positioning_stack_validation_label: [self.positioning_stack_box]}

    @property
    def __required_comboboxes(self):
        """Generates dict of required comboboxes for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input comboboxes
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.name_validation_label: [self.name_combobox]}

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, line_edits in self.__required_widgets.items():
            label.setText('')
            for line_edit in line_edits:
                line_edit.clear()
                line_edit.setStyleSheet('')

        for label, comboboxes in self.__required_comboboxes.items():
            label.setText('')
            for combobox in comboboxes:
                combobox.setStyleSheet('')

        self.positioning_stack_box.clear()

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        valid = True

        widgets = self.__required_widgets
        for label, list_widgets in widgets.items():
            row_valid = True
            for list_widget in list_widgets:
                if list_widget.count() == 0:
                    list_widget.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    list_widget.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        comboboxes = self.__required_comboboxes
        for label, boxes in comboboxes.items():
            row_valid = True
            for combobox in boxes:
                if not combobox.currentText():
                    combobox.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    combobox.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        if valid:
            for label, line_edits in widgets.items():
                label.setText('')
                for line_edit in line_edits:
                    line_edit.setStyleSheet('')
            for label, boxes in comboboxes.items():
                label.setText('')
                for combobox in boxes:
                    combobox.setStyleSheet('')

        return valid

    def addNewItem(self):
        """ When the 'Add' button is clicked, add the chosen positioner to the list and remove it from the combobox."""
        # Remove the positioner if it is already included in the list, then add it to the end of the list
        for item in self.positioning_stack_box.findItems(self.positioners_combobox.currentText(),
                                                         QtCore.Qt.MatchFixedString):
            self.positioning_stack_box.takeItem(self.positioning_stack_box.row(item))
        self.positioning_stack_box.addItem(self.positioners_combobox.currentText())
        # Remove the positioner from the combobox or clear the "Add New..." text as necessary
        if self.positioners_combobox.currentIndex() != (self.positioners_combobox.count() - 1):
            self.positioners_combobox.removeItem(self.positioners_combobox.currentIndex())
        else:
            self.positioners_combobox.clearEditText()

    def clearList(self):
        """ When the 'Clear' button is clicked, clear the list of positioners and repopulate the combobox."""
        self.positioning_stack_box.clear()
        self.positioners_combobox.clear()
        self.positioners_combobox.addItems([*self.positioners_list, self.add_new_text])
        self.positioners_combobox.setCurrentIndex(0)

    def setNewPositioner(self):
        """ When the 'Add New...' option is chosen in the positioner combobox, clear the text."""
        if self.positioners_combobox.currentText() == self.add_new_text:
            self.positioners_combobox.clearEditText()

    def updateValue(self, json_data, _folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        """
        self.reset()
        self.json = json_data
        instrument_data = json_data.get('instrument', {})
        self.positioning_stack_list = instrument_data.get('positioning_stacks', [])

        try:
            positioning_stack_data = self.positioning_stack_list[max(self.name_combobox.currentIndex(), 0)]
        except IndexError:
            positioning_stack_data = {}

        # Name combobox
        positioning_stacks = []
        for data in self.positioning_stack_list:
            name = data.get('name', '')
            if name:
                positioning_stacks.append(name)

        # Rewrite the combobox to contain the new list of positioning stacks, and reset the index to the current value
        index = max(self.name_combobox.currentIndex(), 0)
        self.name_combobox.clear()
        self.name_combobox.addItems([*positioning_stacks, self.add_new_text])
        self.name_combobox.setCurrentIndex(index)
        if self.name_combobox.currentText() == self.add_new_text:
            self.name_combobox.clearEditText()

        name = positioning_stack_data.get('name')
        if name is not None:
            self.name_combobox.setCurrentText(name)

        positioners = []
        positioners_data = instrument_data.get('positioners', [])
        for data in positioners_data:
            positioner_name = data.get('name', '')
            if positioner_name:
                positioners.append(positioner_name)

        self.positioners_list = positioners.copy()

        # Positioners list widget
        stack_positioners = positioning_stack_data.get('positioners', [])
        self.positioning_stack_box.clear()
        # Add positioners in this stack to the box, and remove from the list to be used for the combobox
        for positioner in stack_positioners:
            self.positioning_stack_box.addItem(positioner)
            with contextlib.suppress(ValueError):
                positioners.remove(positioner)

        # Positioners combobox
        # Rewrite the combobox to contain the remaining positioners, and reset the index to the current value
        index = max(self.positioners_combobox.currentIndex(), 0)
        self.positioners_combobox.clear()
        self.positioners_combobox.addItems([*positioners, self.add_new_text])
        self.positioners_combobox.setCurrentIndex(index)
        if self.positioners_combobox.currentText() == self.add_new_text:
            self.positioners_combobox.clearEditText()

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        name = self.name_combobox.currentText()
        if name:
            json_data['name'] = name

        positioners = []
        for index in range(self.positioning_stack_box.count()):
            positioners.append(self.positioning_stack_box.item(index).text())

        if positioners:
            json_data['positioners'] = positioners

        # Place edited positioning stack within the list of positioning stacks
        try:
            self.positioning_stack_list[self.name_combobox.currentIndex()] = json_data
        except IndexError:
            self.positioning_stack_list.append(json_data)

        # Return updated set of positioning stacks
        return {self.key: self.positioning_stack_list}


class PositionersComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the positioners component of the instrument description"""
    def __init__(self):
        super().__init__()

        self.type = Designer.Component.Positioners
        self.key = 'positioners'

        self.json = {}
        self.folder_path = '.'
        self.add_new_text = 'Add New...'
        self.positioners_list = []

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Name field - string, required -- combobox chooses between positioners, and allows renaming
        self.name_combobox = QtWidgets.QComboBox()
        self.name_combobox.setEditable(True)
        layout.addWidget(QtWidgets.QLabel('Name: '), 0, 0)
        layout.addWidget(self.name_combobox, 0, 1)
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 0, 2)

        # When the positioner is changed, connect to a slot that updates the positioner parameters in the component.
        # The "activated" signal is emitted only when the user selects an option (not programmatically)
        # and is also emitted when the user re-selects the same option.
        self.name_combobox.activated.connect(lambda: self.updateValue(self.json, self.folder_path))

        # Base field - array of floats, optional -- array is: xyz translation, xyz orientation in degrees
        self.base_x_translation = create_validated_line_edit(3, '0.0')
        self.base_y_translation = create_validated_line_edit(3, '0.0')
        self.base_z_translation = create_validated_line_edit(3, '0.0')
        sub_layout = xyz_hbox_layout(self.base_x_translation, self.base_y_translation, self.base_z_translation)

        layout.addWidget(QtWidgets.QLabel('Base (Translation): '), 1, 0)
        layout.addLayout(sub_layout, 1, 1)

        self.base_x_orientation = create_validated_line_edit(3, '0.0')
        self.base_y_orientation = create_validated_line_edit(3, '0.0')
        self.base_z_orientation = create_validated_line_edit(3, '0.0')
        sub_layout = xyz_hbox_layout(self.base_x_orientation, self.base_y_orientation, self.base_z_orientation)

        layout.addWidget(QtWidgets.QLabel('Base (Orientation): '), 2, 0)
        layout.addLayout(sub_layout, 2, 1)

        # Tool field - array of floats, optional -- array is: xyz translation, xyz orientation in degrees
        self.tool_x_translation = create_validated_line_edit(3, '0.0')
        self.tool_y_translation = create_validated_line_edit(3, '0.0')
        self.tool_z_translation = create_validated_line_edit(3, '0.0')
        sub_layout = xyz_hbox_layout(self.tool_x_translation, self.tool_y_translation, self.tool_z_translation)

        layout.addWidget(QtWidgets.QLabel('Tool (Translation): '), 3, 0)
        layout.addLayout(sub_layout, 3, 1)

        self.tool_x_orientation = create_validated_line_edit(3, '0.0')
        self.tool_y_orientation = create_validated_line_edit(3, '0.0')
        self.tool_z_orientation = create_validated_line_edit(3, '0.0')
        sub_layout = xyz_hbox_layout(self.tool_x_orientation, self.tool_y_orientation, self.tool_z_orientation)

        layout.addWidget(QtWidgets.QLabel('Tool (Orientation): '), 4, 0)
        layout.addLayout(sub_layout, 4, 1)

        # Custom Order field - array of strings, optional
        # Display list of joint objects in a QListWidget
        sub_layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(QtWidgets.QLabel('Custom Order: '))
        button_layout.addStretch(1)
        self.add_button = QtWidgets.QPushButton('Add Order')
        self.clear_button = QtWidgets.QPushButton('Clear Order')

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.clear_button)

        self.custom_order_box = QtWidgets.QListWidget()
        self.custom_order_box.setSpacing(2)
        self.custom_order_box.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.custom_order_box.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        sub_layout.addLayout(button_layout)
        sub_layout.addWidget(self.custom_order_box)
        self.add_button.clicked.connect(self.addOrder)
        self.clear_button.clicked.connect(self.custom_order_box.clear)
        layout.addLayout(sub_layout, 5, 0, 1, 2)

        # Joints Subcomponents
        sub_layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()
        self.add_joint_button = QtWidgets.QPushButton('Add Joint')
        self.add_joint_button.clicked.connect(self.addJoint)
        self.remove_joint_button = QtWidgets.QPushButton('Remove Joint')
        self.remove_joint_button.clicked.connect(self.removeJoint)
        button_layout.addWidget(QtWidgets.QLabel('Joints:'))
        button_layout.addStretch(1)
        button_layout.addWidget(self.add_joint_button)
        button_layout.addWidget(self.remove_joint_button)
        self.joint_components = []
        self.joint_accordion = Accordion()
        self.joint_accordion.setMinimumHeight(200)
        sub_layout.addLayout(button_layout)
        sub_layout.addWidget(self.joint_accordion)
        layout.addLayout(sub_layout, 9, 0, 1, 3)
        layout.setRowStretch(9, 2)

        # Links Subcomponents
        sub_layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()
        self.add_link_button = QtWidgets.QPushButton('Add Link')
        self.add_link_button.clicked.connect(self.addLink)
        self.remove_link_button = QtWidgets.QPushButton('Remove Link')
        self.remove_link_button.clicked.connect(self.removeLink)
        button_layout.addWidget(QtWidgets.QLabel('Links:'))
        button_layout.addStretch(1)
        button_layout.addWidget(self.add_link_button)
        button_layout.addWidget(self.remove_link_button)
        self.link_components = []
        self.link_accordion = Accordion()
        self.link_accordion.setMinimumHeight(200)
        sub_layout.addLayout(button_layout)
        sub_layout.addWidget(self.link_accordion)
        layout.addLayout(sub_layout, 10, 0, 1, 3)
        layout.setRowStretch(10, 2)

    def addLink(self):
        index = len(self.link_accordion.panes) + 1
        self.link_components.append((QtWidgets.QLabel(f'Link #{index}'),
                                     QtWidgets.QCheckBox(), LinkSubComponent()))
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.link_components[-1][1])  # add checkbox
        layout.addSpacing(10)
        layout.addWidget(self.link_components[-1][0])  # add label
        layout.addStretch(1)
        widget.setLayout(layout)
        self.link_accordion.addPane(Pane(widget, self.link_components[-1][2]))

    def removeLink(self):
        selection = [link[1].isChecked() for link in self.link_components]
        for index, selected in reversed(list(enumerate(selection))):
            if selected:
                pane = self.link_accordion.panes[index]
                del self.link_components[index]
                self.link_accordion.removePane(pane)

        for index, link in enumerate(self.link_components):
            link[0].setText(f'Link #{index + 1}')

    def removeJoint(self):
        selection = [joint[1].isChecked() for joint in self.joint_components]
        for index, selected in reversed(list(enumerate(selection))):
            if selected:
                pane = self.joint_accordion.panes[index]
                del self.joint_components[index]
                self.joint_accordion.removePane(pane)

        for index, joint in enumerate(self.joint_components):
            joint[0].setText(f'Joint #{index + 1}')

        # Repopulate the custom order box to reflect new joints
        if self.custom_order_box.count() > 0:
            self.addOrder()

    def addJoint(self):
        index = len(self.joint_accordion.panes) + 1
        self.joint_components.append((QtWidgets.QLabel(f'Joint #{index}'),
                                      QtWidgets.QCheckBox(), JointSubComponent()))
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.joint_components[-1][1])  # add checkbox
        layout.addSpacing(10)
        layout.addWidget(self.joint_components[-1][0])  # add label
        layout.addStretch(1)
        widget.setLayout(layout)
        self.joint_accordion.addPane(Pane(widget, self.joint_components[-1][2]))
        # Repopulate the custom order box to reflect new joints
        if self.custom_order_box.count() > 0:
            self.custom_order_box.addItem(f'Joint #{index} [No Name]')

    @property
    def __required_comboboxes(self):
        """Generates dict of required comboboxes for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input comboboxes
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.name_validation_label: [self.name_combobox]}

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, comboboxes in self.__required_comboboxes.items():
            label.setText('')
            for combobox in comboboxes:
                combobox.setStyleSheet('')

        self.base_x_translation.setText('0.0')
        self.base_y_translation.setText('0.0')
        self.base_z_translation.setText('0.0')
        self.base_x_orientation.setText('0.0')
        self.base_y_orientation.setText('0.0')
        self.base_z_orientation.setText('0.0')

        self.tool_x_translation.setText('0.0')
        self.tool_y_translation.setText('0.0')
        self.tool_z_translation.setText('0.0')
        self.tool_x_orientation.setText('0.0')
        self.tool_y_orientation.setText('0.0')
        self.tool_z_orientation.setText('0.0')

        self.custom_order_box.clear()
        self.joint_components.clear()
        self.link_components.clear()
        self.joint_accordion.clear()
        self.link_accordion.clear()

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        valid = True

        comboboxes = self.__required_comboboxes
        for label, boxes in comboboxes.items():
            row_valid = True
            for combobox in boxes:
                if not combobox.currentText():
                    combobox.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    combobox.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        joint_valid = True
        for index, pane in enumerate(self.joint_accordion.panes):
            component = self.joint_components[index][2]
            valid = component.validate()
            if valid:
                pane.setStyleSheet('')
            else:
                pane.setStyleSheet('Pane > QWidget{border: 1px solid red;}')
                # Show the accordion pane if the subcomponent is not valid
                # Note that passing "False" to the "toggle()" routine toggles the pane's visibility to "True"
                self.joint_accordion.panes[index].toggle(False)
                joint_valid = False

        link_valid = True
        for index, pane in enumerate(self.link_accordion.panes):
            component = self.link_components[index][2]
            valid = component.validate()
            if valid:
                pane.setStyleSheet('')
            else:
                pane.setStyleSheet('Pane > QWidget{border: 1px solid red;}')
                # Show the accordion pane if the subcomponent is not valid
                # Note that passing "False" to the "toggle()" routine toggles the pane's visibility to "True"
                self.link_accordion.panes[index].toggle(False)
                link_valid = False

        if valid and joint_valid and link_valid:
            for label, boxes in comboboxes.items():
                label.setText('')
                for combobox in boxes:
                    combobox.setStyleSheet('')
            return True
        return False

    def addOrder(self):
        """ When the 'Add' button is clicked, add the set of joint objects to the custom order list."""
        self.custom_order_box.clear()
        joint_list = []
        for index, joint in enumerate(self.joint_components):
            joint_name = joint[2].joint_name.text()
            joint_list.append(joint_name if joint_name else f'Joint #{index+1} [No Name]')
        self.custom_order_box.addItems(joint_list)

    def updateValue(self, json_data, folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        :param folder_path: path to instrument file folder
        :type folder_path: str
        """
        self.reset()
        self.json = json_data
        instrument_data = json_data.get('instrument', {})
        self.positioners_list = instrument_data.get('positioners', [])

        try:
            positioner_data = self.positioners_list[max(self.name_combobox.currentIndex(), 0)]
        except IndexError:
            positioner_data = {}

        # Name combobox
        positioners = []
        for data in self.positioners_list:
            name = data.get('name', '')
            if name:
                positioners.append(name)

        # Rewrite the combobox to contain the new list of positioners
        index = max(self.name_combobox.currentIndex(), 0)
        self.name_combobox.clear()
        self.name_combobox.addItems([*positioners, self.add_new_text])
        self.name_combobox.setCurrentIndex(index)
        if self.name_combobox.currentText() == self.add_new_text:
            self.name_combobox.clearEditText()

        name = positioner_data.get('name')
        if name is not None:
            self.name_combobox.setCurrentText(name)

        # Base field
        base = positioner_data.get('base')
        if base is not None:
            self.base_x_translation.setText(f"{safe_get_value(base, 0, '0.0')}")
            self.base_y_translation.setText(f"{safe_get_value(base, 1, '0.0')}")
            self.base_z_translation.setText(f"{safe_get_value(base, 2, '0.0')}")
            self.base_x_orientation.setText(f"{safe_get_value(base, 3, '0.0')}")
            self.base_y_orientation.setText(f"{safe_get_value(base, 4, '0.0')}")
            self.base_z_orientation.setText(f"{safe_get_value(base, 5, '0.0')}")

        # Tool field
        tool = positioner_data.get('tool')
        if tool is not None:
            self.tool_x_translation.setText(f"{safe_get_value(tool, 0, '0.0')}")
            self.tool_y_translation.setText(f"{safe_get_value(tool, 1, '0.0')}")
            self.tool_z_translation.setText(f"{safe_get_value(tool, 2, '0.0')}")
            self.tool_x_orientation.setText(f"{safe_get_value(tool, 3, '0.0')}")
            self.tool_y_orientation.setText(f"{safe_get_value(tool, 4, '0.0')}")
            self.tool_z_orientation.setText(f"{safe_get_value(tool, 5, '0.0')}")

        # Joint object
        # Send a deepcopy of the JSON so that removing joints is not permanent
        # if we change positioner prior to updating the instrument description file
        joints_list = positioner_data.get('joints', [])
        for joint in joints_list:
            self.addJoint()
            self.joint_components[-1][2].updateValue(joint, folder_path)
            joint_name = self.joint_components[-1][2].joint_name.text()
            if joint_name:
                # Update pane label
                self.joint_components[-1][0].setText(joint_name)

        # Link object
        # Send a deepcopy of the JSON so that removing links is not permanent
        # if we change positioner prior to updating the instrument description file
        links_list = positioner_data.get('links', [])
        for link in links_list:
            self.addLink()
            self.link_components[-1][2].updateValue(link, folder_path)
            link_name = self.link_components[-1][2].link_name.text()
            if link_name:
                # Update pane label
                self.link_components[-1][0].setText(link_name)

        # Custom Order field -- need to do after joints are initialised
        custom_order = positioner_data.get('custom_order')
        if custom_order is not None:
            self.custom_order_box.addItems(custom_order)


    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        # Begin with obtaining data from subcomponents to use for the positioners
        name = self.name_combobox.currentText()
        if name:
            json_data['name'] = name

        # Base field
        btx, bty, btz = self.base_x_translation.text(), self.base_y_translation.text(), self.base_z_translation.text()
        box, boy, boz = self.base_x_orientation.text(), self.base_y_orientation.text(), self.base_z_orientation.text()
        if btx and bty and btz and box and boy and boz:
            base = [float(btx), float(bty), float(btz), float(box), float(boy), float(boz)]
            if base != [0] * 6:
                json_data['base'] = base

        # Tool field
        ttx, tty, ttz = self.tool_x_translation.text(), self.tool_y_translation.text(), self.tool_z_translation.text()
        tox, toy, toz = self.tool_x_orientation.text(), self.tool_y_orientation.text(), self.tool_z_orientation.text()
        if ttx and tty and ttz and tox and toy and toz:
            tool = [float(ttx), float(tty), float(ttz), float(tox), float(toy), float(toz)]
            if tool != [0] * 6:
                json_data['tool'] = tool

        # Need to obtain joints data before determining custom order, will update json afterwards
        joints_data = []
        for joint in self.joint_components:
            joints_data.extend(joint[2].value()['joints'])

        # Create a custom order of all joints from the subcomponent
        # Need to do this after joints are updated
        custom_order = [self.custom_order_box.item(i).text() for i in range(self.custom_order_box.count())]
        if custom_order:

            # Ensure joint names are updated
            for i, joint_string in enumerate(custom_order):
                if 'Joint #' in joint_string:
                    joint_index = int(joint_string[7:-10]) - 1
                    custom_order[i] = joints_data[joint_index]['name']

            json_data['custom_order'] = custom_order

        # Now update json with joints data
        json_data.update({'joints': joints_data})

        links_data = []
        for link in self.link_components:
            links_data.extend(link[2].value()['links'])
        json_data.update({'links': links_data})

        # Place edited positioner within the list of positioners
        try:
            self.positioners_list[self.name_combobox.currentIndex()] = json_data
        except IndexError:
            self.positioners_list.append(json_data)

        # Return updated set of positioners
        return {self.key: self.positioners_list}


class JointSubComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the joint subcomponent of the instrument description"""
    def __init__(self):
        super().__init__()

        self.key = 'joints'

        self.json = {}
        self.folder_path = '.'

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Name field - string, required -- combobox chooses between joints, and allows renaming
        self.joint_name = QtWidgets.QLineEdit()
        # self.name_combobox.setEditable(True)
        layout.addWidget(QtWidgets.QLabel('Name: '), 0, 0)
        layout.addWidget(self.joint_name, 0, 1)
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 0, 2)

        # Type field - string from list, required
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.addItems(['prismatic', 'revolute'])
        layout.addWidget(QtWidgets.QLabel('Type: '), 1, 0)
        layout.addWidget(self.type_combobox, 1, 1)
        self.type_validation_label = create_required_label()
        layout.addWidget(self.type_validation_label, 1, 2)

        # Parent field - string from list, required
        self.parent_name = QtWidgets.QLineEdit()
        layout.addWidget(QtWidgets.QLabel('Parent Link: '), 2, 0)
        layout.addWidget(self.parent_name, 2, 1)
        self.parent_validation_label = create_required_label()
        layout.addWidget(self.parent_validation_label, 2, 2)

        # Child field - string from list, required
        self.child_name = QtWidgets.QLineEdit()
        layout.addWidget(QtWidgets.QLabel('Child Link: '), 3, 0)
        layout.addWidget(self.child_name, 3, 1)
        self.child_validation_label = create_required_label()
        layout.addWidget(self.child_validation_label, 3, 2)

        # Axis field - array of floats, required
        self.x_axis = create_validated_line_edit(3)
        self.y_axis = create_validated_line_edit(3)
        self.z_axis = create_validated_line_edit(3)
        sub_layout = xyz_hbox_layout(self.x_axis, self.y_axis, self.z_axis)

        layout.addWidget(QtWidgets.QLabel('Axis: '), 4, 0)
        layout.addLayout(sub_layout, 4, 1)
        self.axis_validation_label = create_required_label()
        layout.addWidget(self.axis_validation_label, 4, 2)

        # Origin field - array of floats, required
        self.x_origin = create_validated_line_edit(3)
        self.y_origin = create_validated_line_edit(3)
        self.z_origin = create_validated_line_edit(3)
        sub_layout = xyz_hbox_layout(self.x_origin, self.y_origin, self.z_origin)

        layout.addWidget(QtWidgets.QLabel('Origin: '), 5, 0)
        layout.addLayout(sub_layout, 5, 1)
        self.origin_validation_label = create_required_label()
        layout.addWidget(self.origin_validation_label, 5, 2)

        # Lower Limit field - float, required
        self.lower_limit = create_validated_line_edit(3)
        layout.addWidget(QtWidgets.QLabel('Lower limit: '), 6, 0)
        layout.addWidget(self.lower_limit, 6, 1)
        self.lower_limit_validation_label = create_required_label()
        layout.addWidget(self.lower_limit_validation_label, 6, 2)

        # Upper Limit field - float, required
        self.upper_limit = create_validated_line_edit(3)
        layout.addWidget(QtWidgets.QLabel('Upper limit: '), 7, 0)
        layout.addWidget(self.upper_limit, 7, 1)
        self.upper_limit_validation_label = create_required_label()
        layout.addWidget(self.upper_limit_validation_label, 7, 2)

        # Home Offset field - float, optional
        self.home_offset = create_validated_line_edit(3)
        layout.addWidget(QtWidgets.QLabel('Home Offset: '), 8, 0)
        layout.addWidget(self.home_offset, 8, 1)

    @property
    def __required_widgets(self):
        """Generates dict of required widgets for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input widgets
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {
            self.name_validation_label: [self.joint_name],
            self.parent_validation_label: [self.parent_name],
            self.child_validation_label: [self.child_name],
            self.axis_validation_label: [self.x_axis, self.y_axis, self.z_axis],
            self.origin_validation_label: [self.x_origin, self.y_origin, self.z_origin],
            self.lower_limit_validation_label: [self.lower_limit],
            self.upper_limit_validation_label: [self.upper_limit]
        }

    @property
    def __required_comboboxes(self):
        """Generates dict of required comboboxes for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input comboboxes
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.type_validation_label: [self.type_combobox]}

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, line_edits in self.__required_widgets.items():
            label.setText('')
            for line_edit in line_edits:
                line_edit.clear()
                line_edit.setStyleSheet('')

        for label, comboboxes in self.__required_comboboxes.items():
            label.setText('')
            for combobox in comboboxes:
                combobox.setStyleSheet('')

        self.home_offset.setText('')

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        valid = True

        widgets = self.__required_widgets
        for label, line_edits in widgets.items():
            row_valid = True
            for line_edit in line_edits:
                if not line_edit.text():
                    line_edit.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    line_edit.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        comboboxes = self.__required_comboboxes
        for label, boxes in comboboxes.items():
            row_valid = True
            for combobox in boxes:
                if not combobox.currentText():
                    combobox.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    combobox.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        if valid:
            for label, line_edits in widgets.items():
                label.setText('')
                for line_edit in line_edits:
                    line_edit.setStyleSheet('')
            for label, boxes in comboboxes.items():
                label.setText('')
                for combobox in boxes:
                    combobox.setStyleSheet('')

        return valid

    def updateValue(self, json_data, _folder_path):
        """Updates the json data of the component

        :param json_data: positioner json
        :type json_data: Dict[str, Any]
        """
        self.reset()
        joint_data = json_data

        # Name field
        name = joint_data.get('name')
        if name is not None:
            self.joint_name.setText(name)

        # Type combobox
        joint_type = joint_data.get('type')
        if joint_type is not None:
            self.type_combobox.setCurrentText(joint_type)

        parent = joint_data.get('parent')
        if parent is not None:
            self.parent_name.setText(parent)

        child = joint_data.get('child')
        if child is not None:
            self.child_name.setText(child)

        # Axis field
        axis = joint_data.get('axis')
        if axis is not None:
            self.x_axis.setText(f"{safe_get_value(axis, 0, '')}")
            self.y_axis.setText(f"{safe_get_value(axis, 1, '')}")
            self.z_axis.setText(f"{safe_get_value(axis, 2, '')}")

        # Origin field
        origin = joint_data.get('origin')
        if origin is not None:
            self.x_origin.setText(f"{safe_get_value(origin, 0, '')}")
            self.y_origin.setText(f"{safe_get_value(origin, 1, '')}")
            self.z_origin.setText(f"{safe_get_value(origin, 2, '')}")

        # Lower Limit field
        lower_limit = joint_data.get('lower_limit')
        if lower_limit is not None:
            self.lower_limit.setText(f"{safe_get_value([lower_limit], 0, '')}")

        # Upper Limit field
        upper_limit = joint_data.get('upper_limit')
        if upper_limit is not None:
            self.upper_limit.setText(f"{safe_get_value([upper_limit], 0, '')}")

        # Home Offset field
        home_offset = joint_data.get('home_offset')
        if home_offset is not None:
            self.home_offset.setText(f"{safe_get_value([home_offset], 0, '')}")

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        name = self.joint_name.text()
        if name:
            json_data['name'] = name

        joint_type = self.type_combobox.currentText()
        if joint_type:
            json_data['type'] = joint_type

        parent = self.parent_name.text()
        if parent:
            json_data['parent'] = parent

        child = self.child_name.text()
        if child:
            json_data['child'] = child

        ax, ay, az = self.x_axis.text(), self.y_axis.text(), self.z_axis.text()
        if ax and ay and az:
            json_data['axis'] = [float(ax), float(ay), float(az)]

        ox, oy, oz = self.x_origin.text(), self.y_origin.text(), self.z_origin.text()
        if ox and oy and oz:
            json_data['origin'] = [float(ox), float(oy), float(oz)]

        lower_limit = self.lower_limit.text()
        if lower_limit:
            json_data['lower_limit'] = float(lower_limit)

        upper_limit = self.upper_limit.text()
        if upper_limit:
            json_data['upper_limit'] = float(upper_limit)

        home_offset = self.home_offset.text()
        if home_offset:
            json_data['home_offset'] = float(home_offset)

        # Return updated set of joints
        return {self.key: [json_data]}


class LinkSubComponent(QtWidgets.QWidget):
    """Creates a UI for modifying the link subcomponent of the instrument description"""
    def __init__(self):
        super().__init__()

        self.key = 'links'

        self.json = {}
        self.folder_path = '.'

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Name field - string, required -- combobox chooses between links, and allows renaming
        self.link_name = QtWidgets.QLineEdit()
        layout.addWidget(QtWidgets.QLabel('Name: '), 0, 0)
        layout.addWidget(self.link_name, 0, 1)
        layout.setColumnStretch(1, 2)  # Stretches middle column to double width
        self.name_validation_label = create_required_label()
        layout.addWidget(self.name_validation_label, 0, 2)

        # Visual field - visual object, optional
        # The visual object contains: pose, colour, and mesh parameters
        self.visuals = VisualSubComponent()
        layout.addWidget(self.visuals, 1, 0, 1, 3)

    @property
    def __required_widgets(self):
        """Generates dict of required widgets for validation. The key is the validation
        label and the value is a list of widgets in the same row as the validation label

        :return: dict of labels and input widgets
        :rtype: Dict[QtWidgets.QLabel, QtWidgets.QWidget]
        """
        return {self.name_validation_label: [self.link_name]}

    def reset(self):
        """Reset widgets to default values and validation state"""
        for label, line_edits in self.__required_widgets.items():
            label.setText('')
            for line_edit in line_edits:
                line_edit.setStyleSheet('')

        self.visuals.reset()

    def validate(self):
        """Validates the required inputs in the component are filled

        :return: indicates the required inputs are filled
        :rtype: bool
        """
        widgets = self.__required_widgets
        valid = True
        for label, line_edits in widgets.items():
            row_valid = True
            for line_edit in line_edits:
                if not line_edit.text():
                    line_edit.setStyleSheet('border: 1px solid red;')
                    label.setText('Required!')
                    valid = False
                    row_valid = False
                else:
                    line_edit.setStyleSheet('')
                    if row_valid:
                        label.setText('')

        if valid:
            for label, line_edits in widgets.items():
                label.setText('')
                for line_edit in line_edits:
                    line_edit.setStyleSheet('')

        return valid

    def updateValue(self, json_data, folder_path):
        """Updates the json data of the component

        :param json_data: instrument json
        :type json_data: Dict[str, Any]
        :param folder_path: path to instrument file folder
        :type folder_path: str
        """
        self.reset()

        # Name field
        link_data = json_data

        name = link_data.get('name')
        if name is not None:
            self.link_name.setText(name)

        # Visual object
        self.visuals.updateValue(link_data.get('visual', {}), folder_path)

    def value(self):
        """Returns the updated json from the component's inputs

        :return: updated instrument json
        :rtype: Dict[str, Any]
        """
        json_data = {}

        name = self.link_name.text()
        if name:
            json_data['name'] = name

        visual_data = self.visuals.value()
        if visual_data[self.visuals.key]:
            json_data.update(visual_data)

        # Return updated set of links
        return {self.key: [json_data]}
