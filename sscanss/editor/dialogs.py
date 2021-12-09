import json
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.instrument import Link, circle_point_analysis, generate_description
from sscanss.core.math import clamp
from sscanss.core.util import create_scroll_area
from .widgets import ScriptWidget, JawsWidget, PositionerWidget, DetectorWidget
#from sscanss.app.window.view import MainWindow


class Controls(QtWidgets.QDialog):
    """Creates a widget that creates and manages the instrument control widgets.
    The widget creates instrument controls if the instrument description file is correct,
    otherwise the widget will be blank.

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.setWindowTitle('Instrument Control')

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(600)
        self.tabs.setMinimumHeight(600)
        self.tabs.tabBarClicked.connect(self.updateTabs)
        layout.addWidget(self.tabs)

        self.last_tab_index = 0
        self.last_stack_name = ''
        self.last_collimator_name = {}

    def createWidgets(self):
        """Creates widgets for positioner, jaws, detector, and script"""
        self.tabs.clear()
        positioner_widget = PositionerWidget(self.parent)
        if self.last_stack_name and self.last_stack_name in self.parent.instrument.positioning_stacks.keys():
            positioner_widget.changeStack(self.last_stack_name)
        positioner_widget.stack_combobox.activated[str].connect(self.setStack)

        self.tabs.addTab(create_scroll_area(positioner_widget), 'Positioner')
        self.tabs.addTab(create_scroll_area(JawsWidget(self.parent)), 'Jaws')

        collimator_names = {}
        for name in self.parent.instrument.detectors:
            pretty_name = name if name.lower() == 'detector' else f'{name} Detector'
            detector_widget = DetectorWidget(self.parent, name)
            self.tabs.addTab(create_scroll_area(detector_widget), pretty_name)
            collimator_name = self.last_collimator_name.get(name, '')
            if collimator_name:
                collimator_names[name] = collimator_name
                detector_widget.combobox.setCurrentText(collimator_name)
                detector_widget.changeCollimator()
            detector_widget.collimator_changed.connect(self.setCollimator)
        self.last_collimator_name = collimator_names

        self.script_widget = ScriptWidget(self.parent)
        self.tabs.addTab(create_scroll_area(self.script_widget), 'Script')
        self.tabs.setCurrentIndex(clamp(self.last_tab_index, 0, self.tabs.count()))

    def reset(self):
        """Resets stored states"""
        self.last_tab_index = 0
        self.last_stack_name = ''
        self.last_collimator_name = {}

    def setStack(self, stack_name):
        """Stores the last loaded positioning stack. This preserves the active stack
        between description file modifications

        :param stack_name: name of active positioning stack
        :type stack_name: str
        """
        self.last_stack_name = stack_name

    def setCollimator(self, detector, collimator_name):
        """Stores the last loaded collimator on a detector. This preserves the active
        collimator between description file modifications

        :param detector: name of detector
        :type detector: str
        :param collimator_name: name of active collimator
        :type collimator_name: str
        """
        self.last_collimator_name[detector] = collimator_name

    def updateTabs(self, index):
        """Stores the last open tab.

        :param index: tab index
        :type index: int
        """
        self.last_tab_index = index
        if self.tabs.tabText(index) == 'Script':
            self.script_widget.updateScript()


class CalibrationWidget(QtWidgets.QDialog):
    """Creates a widget for performing kinematic calibration and displaying the residual errors.

    :param parent: main window instance
    :type parent: MainWindow
    :param points: measured 3D points for each joint
    :type points: List[numpy.ndarray]
    :param joint_types: types of each joint
    :type joint_types: List[Link.Type]
    :param joint_offsets: measured offsets for each measurement
    :type joint_offsets: List[numpy.ndarray]
    :param joint_homes: home position for each measurement
    :type joint_homes: List[float]
    """
    def __init__(self, parent, points, joint_types, joint_offsets, joint_homes):
        super().__init__(parent)

        self.points = points
        self.offsets = joint_offsets
        self.robot_name = 'Positioning System'
        self.order = list(range(len(points)))
        self.names = [f'Joint {i + 1}' for i in range(len(points))]
        self.homes = joint_homes
        self.types = joint_types

        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        # create stacked layout
        self.stack = QtWidgets.QStackedLayout()
        main_layout.addLayout(self.stack)
        self.stack1 = QtWidgets.QWidget()
        self.stack2 = QtWidgets.QWidget()
        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)

        self.createCalibrationForm()
        self.createResultTable()

        self.setLayout(main_layout)
        self.setMinimumSize(800, 720)
        self.setWindowTitle('Kinematic Calibration')

    def calibrate(self):
        """Creates kinematic model of a robot from measurement and displays result"""
        self.results = circle_point_analysis(self.points, self.types, self.offsets, self.homes)
        self.json = generate_description(self.robot_name, self.results.base, self.results.tool, self.order, self.names,
                                         self.types, self.results.joint_axes, self.results.joint_origins, self.homes,
                                         self.offsets)
        self.displayResiduals()
        self.stack.setCurrentIndex(1)

    def changeRobotName(self, value):
        """Changes name of the robot

        :param value: name of robot
        :type value: str
        """
        self.robot_name = value
        self.validateForm()

    def changeOrder(self, values):
        """Changes the display order of joints

        :param values: joint indices arranged in the display order
        :type values: List[int]
        """
        size = len(self.points)
        try:
            order = [int(value) - 1 for value in values.split(',')]
            if len(set(order)) != size:
                raise ValueError

            if min(order) != 0 or max(order) != size - 1:
                raise ValueError

            self.order = order
        except ValueError:
            self.order = []

        self.validateForm()

    def changeJointNames(self, index, value):
        """Changes the name of the joint at given index

        :param index: joint index
        :type index: int
        :param value: joint name
        :type value: str
        """
        self.names[index] = value
        self.validateForm()

    def changeHome(self, index, value):
        """Changes the home position of the joint at given index

        :param index: joint index
        :type index: int
        :param value: joint home position
        :type value: str
        """
        self.homes[index] = value

    def changeType(self, index, value):
        """Changes the type of the joint at given index

        :param index: joint index
        :type index: int
        :param value: joint type
        :type value: str
        """
        self.types[index] = Link.Type(value.lower())

    def createCalibrationForm(self):
        """Creates inputs for the calibration arguments"""
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        row_layout = QtWidgets.QHBoxLayout()
        self.error_text = QtWidgets.QLabel()
        self.error_text.setWordWrap(True)
        self.error_text.setStyleSheet('color: red;')
        self.calibrate_button = QtWidgets.QPushButton('Generate Model')
        self.calibrate_button.clicked.connect(self.calibrate)
        row_layout.addStretch(1)
        row_layout.addWidget(self.error_text, 4)
        row_layout.addStretch(1)
        row_layout.addWidget(self.calibrate_button)
        layout.addLayout(row_layout)
        layout.addSpacing(10)

        row_layout = QtWidgets.QHBoxLayout()
        row_layout.addWidget(QtWidgets.QLabel('Name of Positioner:\t'))
        name_line_edit = QtWidgets.QLineEdit(self.robot_name)
        name_line_edit.textChanged.connect(self.changeRobotName)
        row_layout.addWidget(name_line_edit, 2)
        row_layout.addStretch(1)
        layout.addLayout(row_layout)

        row_layout = QtWidgets.QHBoxLayout()
        order_line_edit = QtWidgets.QLineEdit(','.join(str(x + 1) for x in self.order))
        order_line_edit.textChanged.connect(self.changeOrder)
        row_layout.addWidget(QtWidgets.QLabel('Custom Order:\t'))
        row_layout.addWidget(order_line_edit, 2)
        row_layout.addStretch(1)
        layout.addLayout(row_layout)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        divider.setFrameShadow(QtWidgets.QFrame.Sunken)

        layout.addSpacing(10)
        layout.addWidget(QtWidgets.QLabel('<p style="font-size:14px">Joint Information</p>'))
        layout.addWidget(divider)
        layout.addSpacing(5)

        # Define Scroll Area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        widget = QtWidgets.QWidget()
        sub_layout = QtWidgets.QVBoxLayout()
        sub_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)

        widget.setLayout(sub_layout)
        sub_layout.setSpacing(5)
        sub_layout.setContentsMargins(10, 0, 10, 0)
        scroll_area.setWidget(widget)
        layout.addWidget(scroll_area)

        for i in range(len(self.points)):
            name_line_edit = QtWidgets.QLineEdit(self.names[i])
            name_line_edit.textChanged.connect(lambda value, index=i: self.changeJointNames(index, value))

            joint_type_combobox = QtWidgets.QComboBox()
            joint_type_combobox.setView(QtWidgets.QListView())
            joint_type_combobox.addItems([t.value.title() for t in Link.Type])
            joint_type_combobox.setCurrentText(self.types[i].value.title())
            joint_type_combobox.currentTextChanged.connect(lambda value, index=i: self.changeType(index, value))

            joint_home_spinbox = QtWidgets.QDoubleSpinBox()
            joint_home_spinbox.setDecimals(3)
            joint_home_spinbox.setRange(-10000, 10000)
            joint_home_spinbox.setValue(self.homes[i])
            joint_home_spinbox.valueChanged.connect(lambda value, index=i: self.changeHome(index, value))

            row_layout = QtWidgets.QHBoxLayout()
            column_layout = QtWidgets.QVBoxLayout()
            column_layout.addWidget(QtWidgets.QLabel(f'Name of Joint {i + 1}:\t'))
            column_layout.addWidget(name_line_edit)
            column_layout.addStretch(1)
            row_layout.addLayout(column_layout, 4)
            row_layout.addStretch(1)
            sub_layout.addLayout(row_layout)

            row_layout = QtWidgets.QHBoxLayout()
            column_layout = QtWidgets.QVBoxLayout()
            column_layout.addWidget(QtWidgets.QLabel(f'Type of Joint {i + 1}:\t'))
            column_layout.addWidget(joint_type_combobox)
            column_layout.addStretch(1)
            row_layout.addLayout(column_layout, 2)

            column_layout = QtWidgets.QVBoxLayout()
            column_layout.addWidget(QtWidgets.QLabel(f'Home for Joint {i + 1}:\t'))
            column_layout.addWidget(joint_home_spinbox)
            column_layout.addStretch(1)
            row_layout.addLayout(column_layout, 2)
            row_layout.addStretch(1)
            sub_layout.addLayout(row_layout)

            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.HLine)
            divider.setFrameShadow(QtWidgets.QFrame.Sunken)
            sub_layout.addWidget(divider)

        sub_layout.addStretch(1)
        self.stack1.setLayout(layout)

    def copyModel(self):
        """Copies json description of robot to the clipboard"""
        QtWidgets.QApplication.clipboard().setText(json.dumps(self.json, indent=2))

    def saveModel(self):
        """Saves json description of the robot to file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose a file name", '.', "JSON File (*.json)")
        if not filename:
            return

        with open(filename, 'w') as json_file:
            json.dump(self.json, json_file, indent=2)

    def createResultTable(self):
        """Creates widget to show calibration errors"""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        sub_layout = QtWidgets.QHBoxLayout()
        self.filter_combobox = QtWidgets.QComboBox()
        self.filter_combobox.setView(QtWidgets.QListView())
        self.filter_combobox.addItems(['All', *[f'{i + 1}' for i in range(len(self.points))]])
        self.filter_combobox.currentIndexChanged.connect(self.displayResiduals)
        self.copy_model_button = QtWidgets.QPushButton('Copy Model')
        self.copy_model_button.clicked.connect(self.copyModel)
        self.save_model_button = QtWidgets.QPushButton('Save Model')
        self.save_model_button.clicked.connect(self.saveModel)
        sub_layout.addWidget(QtWidgets.QLabel('Show Joint: '))
        sub_layout.addWidget(self.filter_combobox)
        sub_layout.addStretch(1)
        sub_layout.addWidget(self.copy_model_button)
        sub_layout.addWidget(self.save_model_button)
        layout.addLayout(sub_layout)
        layout.addSpacing(10)

        self.result_label = QtWidgets.QLabel()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)

        self.model_error_table = QtWidgets.QTableWidget()
        self.model_error_table.setColumnCount(4)
        self.model_error_table.setHorizontalHeaderLabels(['X', 'Y', 'Z', 'Norm'])
        self.model_error_table.setAlternatingRowColors(True)
        self.model_error_table.setMinimumHeight(300)
        self.model_error_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.model_error_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.model_error_table.horizontalHeader().setMinimumSectionSize(40)
        self.model_error_table.horizontalHeader().setDefaultSectionSize(40)
        self.tabs.addTab(self.model_error_table, 'Model Error')

        self.fit_error_table = QtWidgets.QTableWidget()
        self.fit_error_table.setColumnCount(4)
        self.fit_error_table.setHorizontalHeaderLabels(['X', 'Y', 'Z', 'Norm'])
        self.fit_error_table.setAlternatingRowColors(True)
        self.fit_error_table.setMinimumHeight(300)
        self.fit_error_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.fit_error_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.fit_error_table.horizontalHeader().setMinimumSectionSize(40)
        self.fit_error_table.horizontalHeader().setDefaultSectionSize(40)
        self.tabs.addTab(self.fit_error_table, 'Fitting Error')

        self.tabs.currentChanged.connect(self.displayResiduals)
        layout.addWidget(self.result_label)
        layout.addWidget(self.tabs)
        self.stack2.setLayout(layout)

    def displayResiduals(self):
        """Populates table widgets with residual error"""
        tol = 0.1
        active_tab = self.tabs.currentIndex()
        joint_to_show = self.filter_combobox.currentIndex() - 1
        if active_tab == 0:
            table = self.model_error_table
            residuals = self.results.model_errors
        else:
            table = self.fit_error_table
            residuals = self.results.fit_errors

        residuals = np.vstack(residuals) if joint_to_show == -1 else residuals[joint_to_show]
        norm = np.linalg.norm(residuals, axis=1)
        result_text = '<p style="font-size:14px">The Average {} is ' \
                      '<span style="color:{};font-weight:500;">{:.3f}</span> mm</p>'
        mean = np.mean(norm)
        colour = 'Tomato' if mean > tol else 'SeaGreen'
        self.result_label.setText(result_text.format(self.tabs.tabText(active_tab), colour, mean))
        table.setRowCount(residuals.shape[0])
        for row, vector in enumerate(residuals):
            x = QtWidgets.QTableWidgetItem(f'{vector[0]:.3f}')
            x.setTextAlignment(QtCore.Qt.AlignCenter)
            y = QtWidgets.QTableWidgetItem(f'{vector[1]:.3f}')
            y.setTextAlignment(QtCore.Qt.AlignCenter)
            z = QtWidgets.QTableWidgetItem(f'{vector[2]:.3f}')
            z.setTextAlignment(QtCore.Qt.AlignCenter)
            n = QtWidgets.QTableWidgetItem(f'{norm[row]:.3f}')
            n.setTextAlignment(QtCore.Qt.AlignCenter)

            tomato = QtGui.QBrush(QtGui.QColor('Tomato'))
            if abs(vector[0]) > tol:
                x.setData(QtCore.Qt.BackgroundRole, tomato)
            if abs(vector[1]) > tol:
                y.setData(QtCore.Qt.BackgroundRole, tomato)
            if abs(vector[2]) > tol:
                z.setData(QtCore.Qt.BackgroundRole, tomato)
            if norm[row] > tol:
                n.setData(QtCore.Qt.BackgroundRole, tomato)
            table.setItem(row, 0, x)
            table.setItem(row, 1, y)
            table.setItem(row, 2, z)
            table.setItem(row, 3, n)

    def validateForm(self):
        """Validates calibration inputs"""
        error = []
        size = len(self.names)

        if not self.robot_name:
            error.append('"Name of Positioner" cannot be blank')

        if not self.order:
            error.append(f'"Custom order" should contain comma separated indices for joints 1 to {size}.')

        for index, name in enumerate(self.names):
            if not name:
                error.append(f'"Name of Joint {index + 1}" cannot be blank')

        if len(set(self.names)) != len(self.names):
            error.append('Joint names must be unique')

        self.error_text.setText('\n'.join(error))
        self.calibrate_button.setDisabled(len(error) != 0)


class FindWidget(QtWidgets.QDialog):
    """Creates a widget that searches the Instrument file text and highlights the next occurrence.
    Can chose to match case, or require search to be the whole word
        :param parent: main window instance
        :type parent: MainWindow
        """
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle('Find')

        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search..")

        self.match_case = QtWidgets.QCheckBox()
        self.match_case.setText("Match Case")

        self.whole_word = QtWidgets.QCheckBox()
        self.whole_word.setText("Match whole word")

        self.find_button = QtWidgets.QPushButton("Find")
        self.search_box.textChanged.connect(self.resetSearch)
        self.match_case.stateChanged.connect(self.resetSearch)
        self.whole_word.stateChanged.connect(self.resetSearch)
        self.find_button.clicked.connect(self.search)

        self.status_box = QtWidgets.QLabel()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.search_box, 0, 0)
        layout.addWidget(self.match_case, 1, 0)
        layout.addWidget(self.whole_word, 2, 0)
        layout.addWidget(self.find_button, 2, 1)
        layout.addWidget(self.status_box, 3, 0)

        self.setLayout(layout)

    def search(self):
        """Performs a search for the input_text in the editor window"""
        input_text = self.search_box.text()
        case = self.match_case.isChecked()
        whole_word = self.whole_word.isChecked()

        if self.fist_search_flag:
            findable = self.parent().editor.findFirst(input_text, False, case, whole_word, False, True, 0, 0)
            self.fist_search_flag = False
        else:
            findable = self.parent().editor.findFirst(input_text, False, case, whole_word, False)
        if not findable:
            self.status_box.setText("No more entries found.")
        else:
            self.status_box.clear()

    def resetSearch(self):
        """Resets the FindWidget window"""
        self.fist_search_flag = True
        self.status_box.setText("")

