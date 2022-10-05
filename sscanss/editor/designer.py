from enum import Enum, unique
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.util import ColourPicker, FilePicker, to_float, FormTitle


class Designer(QtWidgets.QWidget):
    """Creates a UI for modifying components of the instrument description"""
    json_updated = QtCore.pyqtSignal(object)

    @unique
    class Component(Enum):
        """Instrument Description Components"""
        General = 'General'
        Jaws = 'Incident Jaws'
        Visual = 'Visual'

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
        self.layout.addStretch(1)
        self.setLayout(self.layout)

    def clear(self):
        self.json = {}
        self.folder_path = '.'
        if self.component is not None:
            self.component.reset()

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
        if component_type == Designer.Component.Jaws:
            self.component = JawComponent()
        elif component_type == Designer.Component.General:
            self.component = GeneralComponent()

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
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_translation)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_translation)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Z: '))
        sub_layout.addWidget(self.z_translation)

        layout.addWidget(QtWidgets.QLabel('Pose (Translation): '), 0, 0)
        layout.addLayout(sub_layout, 0, 1)

        self.x_orientation = create_validated_line_edit(3, '0.0')
        self.y_orientation = create_validated_line_edit(3, '0.0')
        self.z_orientation = create_validated_line_edit(3, '0.0')
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_orientation)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_orientation)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Z: '))
        sub_layout.addWidget(self.z_orientation)

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
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_aperture)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_aperture)

        layout.addWidget(QtWidgets.QLabel('Aperture: '), 0, 0)
        layout.addLayout(sub_layout, 0, 1)
        self.aperture_validation_label = create_required_label()
        layout.addWidget(self.aperture_validation_label, 0, 2)

        self.x_aperture_lower_limit = create_validated_line_edit(3)
        self.y_aperture_lower_limit = create_validated_line_edit(3)
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_aperture_lower_limit)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_aperture_lower_limit)

        layout.addWidget(QtWidgets.QLabel('Aperture Lower Limit: '), 1, 0)
        layout.addLayout(sub_layout, 1, 1)
        self.aperture_lo_validation_label = create_required_label()
        layout.addWidget(self.aperture_lo_validation_label, 1, 2)

        self.x_aperture_upper_limit = create_validated_line_edit(3)
        self.y_aperture_upper_limit = create_validated_line_edit(3)
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_aperture_upper_limit)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_aperture_upper_limit)

        layout.addWidget(QtWidgets.QLabel('Aperture Upper Limit: '), 2, 0)
        layout.addLayout(sub_layout, 2, 1)
        self.aperture_up_validation_label = create_required_label()
        layout.addWidget(self.aperture_up_validation_label, 2, 2)

        self.x_beam_source = create_validated_line_edit(3)
        self.y_beam_source = create_validated_line_edit(3)
        self.z_beam_source = create_validated_line_edit(3)
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_beam_source)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_beam_source)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Z: '))
        sub_layout.addWidget(self.z_beam_source)

        layout.addWidget(QtWidgets.QLabel('Beam Source: '), 3, 0)
        layout.addLayout(sub_layout, 3, 1)
        self.beam_src_validation_label = create_required_label()
        layout.addWidget(self.beam_src_validation_label, 3, 2)

        self.x_beam_direction = create_validated_line_edit(7)
        self.y_beam_direction = create_validated_line_edit(7)
        self.z_beam_direction = create_validated_line_edit(7)
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_beam_direction)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_beam_direction)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Z: '))
        sub_layout.addWidget(self.z_beam_direction)

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

        self.positioner_combobox.clear()
        self.positioner_combobox.addItems(['None'])
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
    """Creates a UI for modifying jaws component of the instrument description"""
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
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(QtWidgets.QLabel('X: '))
        sub_layout.addWidget(self.x_gauge_volume)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Y: '))
        sub_layout.addWidget(self.y_gauge_volume)
        sub_layout.addSpacing(50)
        sub_layout.addWidget(QtWidgets.QLabel('Z: '))
        sub_layout.addWidget(self.z_gauge_volume)

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
