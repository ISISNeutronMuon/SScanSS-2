import datetime
from enum import Enum, unique
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sscanss.config import path_for, __version__, settings
from sscanss.core.geometry import Curve, Volume
from sscanss.core.instrument import IKSolver
from sscanss.core.math import trunc
from sscanss.core.util import (DockFlag, Attributes, Accordion, Pane, create_tool_button, Banner, compact_path,
                               StyledTabWidget, MessageType, CommandID, create_scroll_area, FileDialog)
from sscanss.app.widgets import AlignmentErrorModel, ErrorDetailModel, CenteredBoxProxy


class AboutDialog(QtWidgets.QDialog):
    """Creates a UI that displays information about the software

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setMinimumSize(640, 560)
        self.setModal(True)
        self.main_layout.setContentsMargins(1, 1, 1, 1)

        img = QtWidgets.QLabel()
        img.setPixmap(QtGui.QPixmap(path_for('banner.png')))
        self.main_layout.addWidget(img)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 0, 20, 0)
        self.main_layout.addLayout(layout)
        self.main_layout.addStretch(1)

        text = ('<div>'
                f'<h2 style="text-align:center">Version {__version__}</h2>'
                '<p >SScanSS 2 (pronounced “<b>scans two</b>”) provides a virtual laboratory for planning, visualizing,'
                ' and setting-up strain scanning experiments on engineering beam-line instruments. SScanSS 2 which is '
                'an acronym for <b>S</b>train <b>Scan</b>ning <b>S</b>imulation <b>S</b>oftware uses a computer model '
                'of the instrument i.e. jaws, collimators, positioning system and 3D model of the sample to simulate '
                'the measurement procedure.</p> <p>SScanSS 2 is a Python rewrite of the SScanSS application written '
                'in IDL by <b>Dr. Jon James</b> at the Open University, in collaboration with the ISIS Neutron and '
                'Muon Source.</p>'
                '<h3>Reference</h3>'
                '<ol><li>J. A. James, J. R. Santisteban, L. Edwards and M. R. Daymond, “A virtual laboratory for '
                'neutron and synchrotron strain scanning,” Physica B: Condensed Matter, vol. 350, no. 1-3, '
                'p. 743–746, 2004.</li>'
                '<li>Nneji Stephen, & Cooper Joshaniel FK. (2022). SScanSS 2—a redesigned strain scanning simulation '
                'software (Version 2.0.0). <a href="http://doi.org/10.5281/zenodo.6516966">'
                'http://doi.org/10.5281/zenodo.6516966</a>.</li>'
                '</ol>'
                '<h3>Credit</h3>'
                '<ul><li>Icons from FontAwesome</li></ul>'
                '<hr/>'
                '<p style="text-align:center">Distributed under the BSD 3-Clause License</p>'
                f'<p style="text-align:center">Copyright &copy; 2018-{datetime.date.today().year}, ISIS Neutron and '
                'Muon Source. All rights reserved.</p>'
                '</div>')

        label = QtWidgets.QLabel(text)
        label.setOpenExternalLinks(True)
        label.setWordWrap(True)
        layout.addWidget(label)


class ProjectDialog(QtWidgets.QDialog):
    """Creates a UI for creating and loading projects

    :param parent: main window instance
    :type parent: MainWindow
    """
    # max number of recent projects to show in dialog is fixed because of the
    # dimensions of the dialog window
    max_recent_size = 5

    def __init__(self, recent, parent):
        super().__init__(parent)

        self._busy = False
        self.parent = parent
        self.recent = recent
        self.instruments = sorted(parent.presenter.model.instruments.keys())
        data = parent.presenter.model.project_data
        self.selected_instrument = None if data is None else data['instrument'].name

        if len(self.recent) > self.max_recent_size:
            self.recent_list_size = self.max_recent_size
        else:
            self.recent_list_size = len(self.recent)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setMinimumSize(640, 500)
        self.main_layout.setContentsMargins(1, 1, 1, 1)

        self.createImageHeader()
        self.createTabWidgets()
        self.createNewProjectWidgets()
        self.createRecentProjectWidgets()
        self.createLoadingIndicator()
        self.create_project_button.clicked.connect(self.createProjectButtonClicked)

        self.project_name_textbox.setFocus()

    @property
    def is_busy(self):
        return self._busy

    @is_busy.setter
    def is_busy(self, value):
        self._busy = value
        self.loading_bar.setMaximum(0 if value else 1)

    def createImageHeader(self):
        """Adds banner image to dialog"""
        img = QtWidgets.QLabel()
        img.setPixmap(QtGui.QPixmap(path_for('banner.png')))
        self.main_layout.addWidget(img)

    def createTabWidgets(self):
        """Creates the tab widget"""
        self.tabs = StyledTabWidget()
        self.tabs.addTab('Create New Project')
        self.tabs.addTab('Open Existing Project')

        self.main_layout.addWidget(self.tabs)

    def createLoadingIndicator(self):
        self.loading_bar = QtWidgets.QProgressBar()
        self.loading_bar.setMinimum(0)
        self.loading_bar.setMaximum(1)
        self.main_layout.addWidget(self.loading_bar)

    def createNewProjectWidgets(self):
        """Creates the inputs for creating a new project"""
        layout = QtWidgets.QVBoxLayout()
        layout.addStretch(1)

        layout.addWidget(QtWidgets.QLabel('Project Name:'))
        self.project_name_textbox = QtWidgets.QLineEdit()
        layout.addWidget(self.project_name_textbox)
        self.validator_textbox = QtWidgets.QLabel('')
        self.validator_textbox.setObjectName('Error')
        layout.addWidget(self.validator_textbox)
        layout.addStretch(1)

        layout.addWidget(QtWidgets.QLabel('Select Instrument:'))
        self.instrument_combobox = QtWidgets.QComboBox()
        self.instrument_combobox.setView(QtWidgets.QListView())
        self.instrument_combobox.addItems(self.instruments)
        self.instrument_combobox.setCurrentText(self.selected_instrument)
        layout.addWidget(self.instrument_combobox)
        layout.addStretch(1)

        button_layout = QtWidgets.QHBoxLayout()
        self.create_project_button = QtWidgets.QPushButton('Create')
        button_layout.addWidget(self.create_project_button)
        button_layout.addStretch(1)

        layout.addLayout(button_layout)
        layout.addStretch(1)

        widget = self.tabs.stack.widget(0)
        widget.setContentsMargins(30, 0, 30, 0)
        widget.setLayout(layout)

    def createProjectButtonClicked(self):
        if self.is_busy:
            return

        name = self.project_name_textbox.text().strip()
        instrument = self.instrument_combobox.currentText()
        if not name:
            self.validator_textbox.setText('Project name cannot be left blank.')
            return

        self.parent.presenter.useWorker(self.parent.presenter.createProject, [name, instrument], self.onSuccess,
                                        self.onFailure)
        self.is_busy = True

    def createRecentProjectWidgets(self):
        """Creates the widget for open recent projects"""
        layout = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setObjectName('Recents')
        self.list_widget.setSpacing(10)

        for i in range(self.recent_list_size):
            item = QtWidgets.QListWidgetItem(compact_path(self.recent[i], 70))
            item.setData(QtCore.Qt.UserRole, self.recent[i])
            item.setToolTip(self.recent[i])
            item.setIcon(QtGui.QIcon(path_for('file-black.png')))
            self.list_widget.addItem(item)

        item = QtWidgets.QListWidgetItem('Open ...')
        item.setIcon(QtGui.QIcon(path_for('folder-open.png')))
        self.list_widget.addItem(item)

        self.list_widget.itemDoubleClicked.connect(self.projectItemDoubleClicked)
        layout.addWidget(self.list_widget)

        widget = self.tabs.stack.widget(1)
        widget.setContentsMargins(30, 0, 30, 0)
        widget.setLayout(layout)

    def projectItemDoubleClicked(self, item):
        if self.is_busy:
            return

        index = self.list_widget.row(item)

        if index == self.recent_list_size:
            filename = self.parent.showOpenDialog('hdf5 File (*.h5)',
                                                  title='Open Project',
                                                  current_dir=self.parent.presenter.model.save_path)
            if not filename:
                return
        else:
            filename = item.data(QtCore.Qt.UserRole)

        self.parent.presenter.useWorker(self.parent.presenter._openProjectHelper, [filename], self.onSuccess,
                                        self.onFailure)
        self.is_busy = True

    def onSuccess(self):
        self.parent.presenter.updateView()
        self.accept()

    def onFailure(self, exception, args):
        if len(args) == 1:
            self.parent.presenter.projectOpenError(exception, args)
        else:
            self.parent.presenter.projectCreationError(exception, args)
        self.is_busy = False

    def keyPressEvent(self, event):
        """This ensure the user cannot close the dialog box with the Esc key"""
        if not self.is_busy:
            super().keyPressEvent(event)

    def reject(self):
        if self.parent.presenter.model.project_data is None:
            self.parent.statusBar().showMessage("Most menu options are disabled until you create/open a project.")
        super().reject()


class ProgressDialog(QtWidgets.QDialog):
    """Creates a UI that informs the user that the software is busy

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setTextVisible(False)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        self.message = QtWidgets.QLabel('')
        self.message.setAlignment(QtCore.Qt.AlignCenter)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addStretch(1)
        main_layout.addWidget(progress_bar)
        main_layout.addWidget(self.message)
        main_layout.addStretch(1)

        self.setLayout(main_layout)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setMinimumSize(300, 120)
        self.setModal(True)

    def showMessage(self, message):
        """Shows the progress bar along with the given message

        :param message: message to show
        :type message: str
        """
        self.message.setText(message)
        self.show()

    def keyPressEvent(self, _):
        """This ensure the user cannot close the dialog box with the Esc key"""


class AlignmentErrorDialog(QtWidgets.QDialog):
    """Creates a UI for presenting sample alignment with fiducial position errors

    :param parent: main window instance
    :type parent: MainWindow
    :param indices: N x 1 array containing indices of each point
    :type indices: numpy.ndarray[int]
    :param enabled: N x 1 array containing enabled state of each point
    :type enabled: numpy.ndarray[bool]
    :param points: N X 3 array of measured fiducial points
    :type points: numpy.ndarray[float]
    :param transform_result: initial alignment result
    :type transform_result: TransformResult
    :param end_configuration: final configuration of the positioning system
    :type end_configuration: List[float]
    :param order_fix: suggested indices for wrong order correction
    :type order_fix: Union[numpy.ndarray[int], None]
    """
    def __init__(self, parent, indices, enabled, points, transform_result, end_configuration, order_fix=None):
        super().__init__(parent)

        self.measured_points = points
        self.transform_result = transform_result
        self.end_configuration = end_configuration

        self.main_layout = QtWidgets.QVBoxLayout()
        self.banner = Banner(MessageType.Information, self)
        self.main_layout.addWidget(self.banner)
        self.banner.hide()

        self.main_layout.addSpacing(5)
        self.result_text = '<p style="font-size:14px">The Average Distance Error is ' \
                           '<span style="color:{};font-weight:500;">{:.3f}</span> {}</p>'
        self.result_label = QtWidgets.QLabel()
        self.result_label.setTextFormat(QtCore.Qt.RichText)
        self.updateResultText(0.0)
        self.main_layout.addWidget(self.result_label)
        self.main_layout.addSpacing(10)
        self.createTabWidgets()
        self.createSummaryTable()
        self.createDetailTable()

        self.main_layout.addSpacing(10)
        layout = QtWidgets.QHBoxLayout()
        self.check_box = QtWidgets.QCheckBox('Move positioner to the last sample pose')
        self.check_box.setChecked(True)
        layout.addWidget(self.check_box)
        self.main_layout.addLayout(layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addStretch(1)

        button_layout = QtWidgets.QHBoxLayout()
        self.accept_button = QtWidgets.QPushButton('Accept')
        self.accept_button.clicked.connect(self.submit)
        self.recalculate_button = QtWidgets.QPushButton('Recalculate')
        self.recalculate_button.clicked.connect(self.recalculate)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.close)
        self.cancel_button.setDefault(True)

        button_layout.addWidget(self.recalculate_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        self.main_layout.addLayout(button_layout)
        self.setLayout(self.main_layout)

        self.setMinimumWidth(450)
        self.setWindowTitle('Error Report for Sample Alignment')

        self.updateModel(indices, enabled)
        if order_fix is not None:
            self.banner.actionButton('FIX', lambda ignore, n=order_fix: self.__correctIndexOrder(n))
            self.banner.showMessage('Incorrect Point Indices have been detected.', MessageType.Warning, False)

    def createTabWidgets(self):
        """Creates the tab widget"""
        self.tabs = StyledTabWidget()
        self.tabs.addTab('Summary')
        self.tabs.addTab('Detailed Analysis')

        self.main_layout.addWidget(self.tabs)

    def createSummaryTable(self):
        """Creates the table view for the summary of the alignment errors"""
        layout = QtWidgets.QVBoxLayout()
        self.summary_table_view = QtWidgets.QTableView()
        self.summary_table_model = AlignmentErrorModel()
        self.summary_table_view.setStyle(CenteredBoxProxy())
        self.summary_table_view.setModel(self.summary_table_model)
        self.summary_table_view.verticalHeader().setVisible(False)
        self.summary_table_view.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.summary_table_view.setFocusPolicy(QtCore.Qt.NoFocus)
        self.summary_table_view.setAlternatingRowColors(True)
        self.summary_table_view.setMinimumHeight(300)
        self.summary_table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.summary_table_view.horizontalHeader().setMinimumSectionSize(40)
        self.summary_table_view.horizontalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.summary_table_view)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs.stack.widget(0).setLayout(layout)

    def createDetailTable(self):
        """Creates the table view for the detailed view of the alignment error"""
        layout = QtWidgets.QVBoxLayout()
        self.detail_table_view = QtWidgets.QTableView()
        self.detail_table_model = ErrorDetailModel()
        self.detail_table_view.setModel(self.detail_table_model)
        self.detail_table_view.verticalHeader().setVisible(False)
        self.detail_table_view.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.detail_table_view.setFocusPolicy(QtCore.Qt.NoFocus)
        self.detail_table_view.setAlternatingRowColors(True)
        self.detail_table_view.setMinimumHeight(300)
        self.detail_table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.detail_table_view.horizontalHeader().setMinimumSectionSize(40)
        self.detail_table_view.horizontalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.detail_table_view)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs.stack.widget(1).setLayout(layout)

    def recalculate(self):
        """Recalculates the alignment error"""
        if np.count_nonzero(self.summary_table_model.enabled) < 3:
            self.banner.showMessage('A minimum of 3 points is required for sample alignment.', MessageType.Error)
            return

        self.transform_result = self.parent().presenter.rigidTransform(self.summary_table_model.point_index,
                                                                       self.measured_points,
                                                                       self.summary_table_model.enabled)
        error = np.full(self.summary_table_model.enabled.size, np.nan)
        error[self.summary_table_model.enabled] = self.transform_result.error
        self.summary_table_model.error = error
        self.updateResultText(self.transform_result.average)

        self.detail_table_model.details = self.transform_result.distance_analysis
        self.detail_table_model.index_pairs = self.summary_table_model.point_index[self.summary_table_model.enabled]
        self.updateTable()

    def updateTable(self):
        """Updates the summary and detailed table view"""
        self.detail_table_model.update()
        self.detail_table_view.update()
        self.summary_table_model.update()
        self.summary_table_view.update()

    def __correctIndexOrder(self, new_indices):
        """Updates the point indices and recomputes the alignment matrix

        :param new_indices: new indices
        :type new_indices: numpy.ndarray
        """
        self.summary_table_model.point_index = new_indices
        self.recalculate()

    def updateResultText(self, average_error):
        """Displays the average alignment error in a formatted label

        :param average_error: average of alignment errors
        :type average_error: float
        """
        if average_error < 0.1:
            colour = 'SeaGreen'
            self.banner.hide()
        else:
            colour = 'firebrick'
            self.banner.showMessage(
                'If the errors are larger than desired, '
                'Try disabling points with the large errors then recalculate.', MessageType.Information)
        self.result_label.setText(self.result_text.format(colour, average_error, 'mm'))

    def updateModel(self, indices, enabled):
        """Updates the summary and detailed table model

        :param indices: N x 1 array containing indices of each point
        :type indices: numpy.ndarray[int]
        :param enabled: N x 1 array containing enabled state of each point
        :type enabled: numpy.ndarray[bool]
        """
        error = np.full(enabled.size, np.nan)
        error[enabled] = self.transform_result.error

        self.summary_table_model.point_index = indices
        self.summary_table_model.error = error
        self.summary_table_model.enabled = enabled
        self.updateResultText(self.transform_result.average)
        self.detail_table_model.details = self.transform_result.distance_analysis
        self.detail_table_model.index_pairs = indices[enabled]
        self.updateTable()

    def submit(self):
        """Accepts the alignment matrix"""
        self.parent().scenes.switchToInstrumentScene()
        self.parent().presenter.alignSample(self.transform_result.matrix)
        if self.check_box.isChecked() and self.end_configuration is not None:
            positioner_name = self.parent().presenter.model.instrument.positioning_stack.name
            self.parent().presenter.movePositioner(positioner_name, self.end_configuration, ignore_locks=True)

        self.accept()


class CalibrationErrorDialog(QtWidgets.QDialog):
    """Creates a UI for presenting base computation with fiducial position errors

    :param pose_id: array of pose index
    :type pose_id: numpy.ndarray
    :param fiducial_id: array of fiducial point index
    :type fiducial_id: numpy.ndarray
    :param error: difference between measured point and computed point
    :type error: numpy.ndarray
    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent, pose_id, fiducial_id, error):
        super().__init__(parent)

        self.main_layout = QtWidgets.QVBoxLayout()

        self.main_layout.addSpacing(5)
        self.result_label = QtWidgets.QLabel()
        self.result_label.setTextFormat(QtCore.Qt.RichText)
        self.main_layout.addWidget(self.result_label)
        self.main_layout.addSpacing(10)

        self.error_table = QtWidgets.QTableWidget()
        self.error_table.setColumnCount(6)
        self.error_table.setHorizontalHeaderLabels(
            ['Pose ID', 'Fiducial ID', '\u0394X', '\u0394Y', '\u0394Z', 'Euclidean\n Norm'])
        self.error_table.horizontalHeaderItem(0).setToolTip('Index of pose')
        self.error_table.horizontalHeaderItem(1).setToolTip('Index of fiducial point')
        self.error_table.horizontalHeaderItem(2).setToolTip('Difference between computed and measured points (X)')
        self.error_table.horizontalHeaderItem(3).setToolTip('Difference between computed and measured points (Y)')
        self.error_table.horizontalHeaderItem(4).setToolTip('Difference between computed and measured points (Z)')
        self.error_table.horizontalHeaderItem(5).setToolTip('Euclidean distance between and measured points')

        self.error_table.setAlternatingRowColors(True)
        self.error_table.setMinimumHeight(600)
        self.error_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.error_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.error_table.horizontalHeader().setMinimumSectionSize(40)
        self.error_table.horizontalHeader().setDefaultSectionSize(40)
        self.main_layout.addWidget(self.error_table)

        self.main_layout.addStretch(1)

        button_layout = QtWidgets.QHBoxLayout()
        self.accept_button = QtWidgets.QPushButton('Accept')
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setDefault(True)

        button_layout.addStretch(1)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        self.main_layout.addLayout(button_layout)
        self.setLayout(self.main_layout)

        self.updateTable(pose_id, fiducial_id, error)
        self.setMinimumSize(800, 720)
        self.setWindowTitle('Error Report for Base Computation')

    def updateTable(self, pose_id, fiducial_id, error):
        """Updates the table widget with calibration error

        :param pose_id: array of pose index
        :type pose_id: numpy.ndarray
        :param fiducial_id: array of fiducial point index
        :type fiducial_id: numpy.ndarray
        :param error: difference between measured point and computed point
        :type error: numpy.ndarray
        """
        tol = 0.1
        norm = np.linalg.norm(error, axis=1)
        result_text = '<p style="font-size:14px">The Average Distance Error is ' \
                      '<span style="color:{};font-weight:500;">{:.3f}</span> mm</p>'
        mean = np.mean(norm)
        colour = 'Tomato' if mean > tol else 'SeaGreen'
        self.result_label.setText(result_text.format(colour, mean))
        self.error_table.setRowCount(error.shape[0])
        for row, vector in enumerate(error):
            pose = QtWidgets.QTableWidgetItem(f'{pose_id[row]}')
            pose.setTextAlignment(QtCore.Qt.AlignCenter)
            fiducial = QtWidgets.QTableWidgetItem(f'{fiducial_id[row]}')
            fiducial.setTextAlignment(QtCore.Qt.AlignCenter)
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
            self.error_table.setItem(row, 0, pose)
            self.error_table.setItem(row, 1, fiducial)
            self.error_table.setItem(row, 2, x)
            self.error_table.setItem(row, 3, y)
            self.error_table.setItem(row, 4, z)
            self.error_table.setItem(row, 5, n)


class SimulationDialog(QtWidgets.QWidget):
    """Creates a UI for displaying the simulation result

    :param parent: main window instance
    :type parent: MainWindow
    """
    dock_flag = DockFlag.Full

    @unique
    class State(Enum):
        """State of simulation"""
        Starting = 0
        Running = 1
        Stopped = 2

    @unique
    class ResultKey(Enum):
        """Simulation result keys"""
        Good = 0
        Warn = 1
        Fail = 2
        Skip = 3

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.parent_model = parent.presenter.model
        main_layout = QtWidgets.QVBoxLayout()
        self.robot_configuration = self.parent_model.instrument.positioning_stack.set_points

        self.banner = Banner(MessageType.Information, self)
        main_layout.addWidget(self.banner)
        self.banner.hide()
        main_layout.addSpacing(10)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.setSpacing(0)
        self.createfilterButtons(button_layout)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        button_layout.addSpacing(5)
        button_layout.addWidget(divider)
        button_layout.addSpacing(5)

        self.path_length_button = create_tool_button(tooltip='Plot Path Length',
                                                     style_name='ToolButton',
                                                     status_tip='Plot calculated path length for current simulation',
                                                     icon_path=path_for('line-chart.png'))
        self.path_length_button.clicked.connect(self.parent.showPathLength)

        self.export_button = create_tool_button(tooltip='Export Script',
                                                style_name='ToolButton',
                                                status_tip='Export script for current simulation',
                                                icon_path=path_for('export.png'))
        self.export_button.clicked.connect(self.parent.showScriptExport)

        button_layout.addWidget(self.path_length_button)
        button_layout.addSpacing(5)
        button_layout.addWidget(self.export_button)
        main_layout.addLayout(button_layout)

        self.progress_label = QtWidgets.QLabel()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.progress_bar)

        self.result_list = Accordion()
        main_layout.addWidget(self.result_list)
        self.setLayout(main_layout)

        self.title = 'Simulation Result'
        self.setMinimumWidth(450)
        self.render_graphics = False
        self.check_collision = False
        self.default_vector_alignment = self.parent.scenes.rendered_alignment

        self.setup(no_render=True)
        self.parent_model.simulation_created.connect(self.setup)
        self.parent.scenes.switchToInstrumentScene()
        self.showResult()

    @property
    def simulation(self):
        """Gets the simulation instance in the main window model

        :return: simulation instance
        :rtype: Simulation
        """
        return self.parent_model.simulation

    def setup(self, no_render=False):
        """Setups the dialog for new simulation"""
        if self.simulation is not None:
            self.banner.hide()
            self.render_graphics = False if no_render else self.simulation.render_graphics
            self.check_collision = self.simulation.check_collision
            self.default_vector_alignment = self.parent.scenes.rendered_alignment
            self.parent.scenes.changeVisibility(Attributes.Beam, True)
            self.renderSimualtion()
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(self.simulation.count)
            self.updateProgress(SimulationDialog.State.Starting)
            self.result_list.clear()
            self.resetFilterButtons()
            self.simulation.result_updated.connect(self.showResult)
            self.simulation.stopped.connect(lambda: self.updateProgress(SimulationDialog.State.Stopped))

    def updateProgress(self, state, increment=False):
        """Updates the progress bar and text

        :param state: simulation state
        :type state: SimulationDialog.State
        :param increment: indicates progress should be increased by one
        :type increment: bool
        """
        if increment:
            self.progress_bar.setValue(self.progress_bar.value() + 1)

        completion = f'<p>Completed {self.progress_bar.value()} of {self.progress_bar.maximum()}</p>'
        if state == SimulationDialog.State.Starting:
            self.progress_label.setText(f'<h3>Simulation Starting</h3>{completion}')
        elif state == SimulationDialog.State.Running:
            self.progress_label.setText(f'<h3>Simulation In-Progress</h3>{completion}')
        elif state == SimulationDialog.State.Stopped:
            self.progress_label.setText(f'<h3>Simulation Stopped</h3>{completion}')

        if self.progress_bar.value() == self.progress_bar.maximum():
            self.progress_label.setText(f'<h3>Simulation Finished</h3>{completion}')

    def createfilterButtons(self, filter_layout):
        """Creates the filter buttons

        :param filter_layout: button layout
        :type filter_layout: QtWidgets.QLayout
        """
        self.filter_button_group = QtWidgets.QButtonGroup()
        self.filter_button_group.setExclusive(False)
        self.filter_button_group.buttonClicked.connect(self.toggleResults)

        hide_good_result = create_tool_button(checkable=True,
                                              checked=True,
                                              tooltip='0 results succeeded',
                                              text='0',
                                              show_text=True,
                                              style_name='ToolButton',
                                              icon_path=path_for('check-circle.png'))
        hide_warn_result = create_tool_button(checkable=True,
                                              checked=True,
                                              tooltip='0 results with warnings',
                                              text='0',
                                              show_text=True,
                                              style_name='ToolButton',
                                              icon_path=path_for('exclamation-circle.png'))
        hide_failed_result = create_tool_button(checkable=True,
                                                checked=True,
                                                tooltip='0 results failed',
                                                text='0',
                                                show_text=True,
                                                style_name='ToolButton',
                                                icon_path=path_for('times-circle.png'))
        hide_skipped_result = create_tool_button(checkable=True,
                                                 checked=True,
                                                 tooltip='0 results skipped',
                                                 hide=True,
                                                 text='0',
                                                 show_text=True,
                                                 style_name='ToolButton',
                                                 icon_path=path_for('minus-circle.png'))

        self.filter_button_group.addButton(hide_good_result, self.ResultKey.Good.value)
        self.filter_button_group.addButton(hide_warn_result, self.ResultKey.Warn.value)
        self.filter_button_group.addButton(hide_failed_result, self.ResultKey.Fail.value)
        self.filter_button_group.addButton(hide_skipped_result, self.ResultKey.Skip.value)
        filter_layout.addWidget(hide_good_result)
        filter_layout.addWidget(hide_warn_result)
        filter_layout.addWidget(hide_failed_result)
        filter_layout.addWidget(hide_skipped_result)

    def toggleResults(self):
        """Shows/Hides the result panes based on the filters"""
        for i, pane in enumerate(self.result_list.panes):
            key = self.getResultKey(self.simulation.results[i])
            self.__updateFilterButton(key)
            pane.setVisible(self.filter_button_group.button(key.value).isChecked())

    def getResultKey(self, result):
        """Returns the key for the given simulation result

        :param result: simulation result
        :type result: SimulationResult
        :return: result key
        :rtype: SimulationDialog.ResultKey
        """
        key = self.ResultKey.Warn
        show_collision = self.simulation.check_collision and np.any(result.collision_mask)
        if result.skipped:
            key = self.ResultKey.Skip
        elif result.ik.status == IKSolver.Status.Failed:
            key = self.ResultKey.Fail
        elif result.ik.status == IKSolver.Status.Converged and not show_collision:
            key = self.ResultKey.Good

        return key

    def resetFilterButtons(self):
        self.result_counts = {key: 0 for key in self.ResultKey}
        for key in self.ResultKey:
            self.result_counts[key] = 0
            self.filter_button_group.button(key.value).setText('0')
        self.filter_button_group.button(self.ResultKey.Skip.value).setVisible(False)

    def __updateFilterButton(self, result_key):
        """Updates the filter button for the given key

        :param result_key: result key
        :type result_key: SimulationDialog.ResultKey
        """
        button_id = result_key.value
        count = self.result_counts[result_key]
        button = self.filter_button_group.button(button_id)
        tool_tip = button.toolTip().split(' ', 1)
        button.setText(f'{count}')
        button.setToolTip(f'{count} {tool_tip[1]}')
        if button_id == 3:
            button.setVisible(count > 0)

    def showResult(self, error=False):
        """Adds the simulation results into the result list and notifies user of fatal error

        :param error: indicates if fatal error occurred
        :type error: bool
        """
        if self.simulation is None:
            return

        state = SimulationDialog.State.Running if self.simulation.isRunning() else SimulationDialog.State.Stopped
        results = self.simulation.results[len(self.result_list.panes):]

        for result in results:
            header = QtWidgets.QLabel()
            header.setTextFormat(QtCore.Qt.RichText)

            details = QtWidgets.QLabel()
            details.setTextFormat(QtCore.Qt.RichText)

            if result.skipped:
                header = self.__createPaneHeader(
                    f'<span>{result.id}</span><br/> '
                    f'<span><b>SKIPPED:</b> {result.note}.</span>', None)
                style = MessageType.Information
            elif result.ik.status == IKSolver.Status.Failed:
                header = self.__createPaneHeader(
                    f'<span>{result.id}</span><br/><span> A runtime error occurred. '
                    'Check logs for more Information.</span>', result.ik.status)
                style = MessageType.Error
            else:
                result_text = '\n'.join('{:<20}{:>12.3f}'.format(*t)
                                        for t in zip(result.joint_labels, result.formatted))
                pos_style = '' if result.ik.position_converged else 'style="color:red"'
                orient_style = '' if result.ik.orientation_converged else 'style="color:red"'
                pos_err, orient_err = result.ik.position_error, result.ik.orientation_error
                info = (f'<span>{result.id}</span><br/>'
                        f'<span {pos_style}><b>Position Error (mm):</b> (X.) {pos_err[0]:.3f}, (Y.) '
                        f'{pos_err[1]:.3f}, (Z.) {pos_err[2]:.3f}</span><br/>'
                        f'<span {orient_style}><b>Orientation Error (degrees):</b> (X.) {orient_err[0]:.3f}, (Y.) '
                        f'{orient_err[1]:.3f}, (Z.) {orient_err[2]:.3f}</span>')

                if self.simulation.compute_path_length:
                    labels = self.simulation.detector_names
                    path_length_info = ', '.join('({}) {:.3f}'.format(*l) for l in zip(labels, result.path_length))
                    info = f'{info}<br/><span><b>Path Length:</b> {path_length_info}</span>'

                show_collision = self.simulation.check_collision and np.any(result.collision_mask)
                if result.ik.status == IKSolver.Status.Converged and not show_collision:
                    style = MessageType.Information
                else:
                    style = MessageType.Warning

                header = self.__createPaneHeader(info, result.ik.status, show_collision)
                details.setText(f'<pre>{result_text}</pre>')
                if self.render_graphics:
                    self.renderSimualtion(result)

            self.result_list.addPane(self.__createPane(header, details, style, result))
            self.updateProgress(state, True)

        if error:
            self.updateProgress(SimulationDialog.State.Stopped)
            self.parent.showMessage('An error occurred while running the simulation.')

    def __createPane(self, panel, details, style, result):
        """Creates an accordion pane for the simulation result

        :param panel: pane header
        :type panel: QtWidgets.QWidget
        :param details: pane details
        :type details: QtWidgets.QWidget
        :param style: pane type
        :type style: MessageType
        :param result: simulation result
        :type result: SimulationResult
        :return: accordion pane
        :rtype: Pane
        """
        pane = Pane(panel, details, style)
        key = self.getResultKey(result)
        self.result_counts[key] += 1
        self.__updateFilterButton(key)
        if not self.filter_button_group.button(key.value).isChecked():
            pane.setHidden(True)

        if style == MessageType.Error or result.skipped:
            pane.setDisabled(True)
            pane.toggle_icon.hide()
            return pane
        action = QtWidgets.QAction('Copy', pane)
        action.setStatusTip('Copy positioner offsets to clipboard')
        action_text = '\t'.join('{:.3f}'.format(t) for t in result.formatted)
        action.triggered.connect(lambda ignore, q=action_text: QtWidgets.QApplication.clipboard().setText(q))
        pane.addContextMenuAction(action)

        action = QtWidgets.QAction('Visualize', pane)
        action.setStatusTip('Visualize selected simulation result in the graphics window')
        action.triggered.connect(lambda ignore, r=result: self.__visualize(result))
        pane.addContextMenuAction(action)

        return pane

    def __createPaneHeader(self, header_text, status, show_collision=False):
        """Creates the pane header widget

        :param header_text: header text
        :type header_text: str
        :param status: inverse kinematics solver status
        :type status: IKSolver.Status
        :param show_collision: indicates if collision icon should be shown
        :type show_collision: bool
        :return: pane header widget
        :rtype: QtWidgets.QWidget
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        header = QtWidgets.QLabel(header_text)
        header.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(header)

        sub_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(sub_layout)

        tool_tip_duration = 1800000
        if show_collision:
            icon = QtWidgets.QLabel()
            icon.setToolTip('Collision Detected')
            icon.setToolTipDuration(tool_tip_duration)
            icon.setPixmap(QtGui.QPixmap(path_for('collision.png')).scaled(25, 25))
            sub_layout.addWidget(icon)

        tool_tip = ''
        icon_path = ''
        if status == IKSolver.Status.HardwareLimit:
            tool_tip = 'Hardware limits violation'
            icon_path = 'limit_hit.png'
        elif status == IKSolver.Status.Unreachable:
            tool_tip = 'Orientation is not reachable by the positioner'
            icon_path = 'unreachable.png'
        elif status == IKSolver.Status.DeformedVectors:
            tool_tip = 'Angle between measurement vectors does not match q-vectors'
            icon_path = 'deformed.png'

        if icon_path and tool_tip:
            icon = QtWidgets.QLabel()
            icon.setToolTip(tool_tip)
            icon.setPixmap(QtGui.QPixmap(path_for(icon_path)).scaled(25, 25))
            icon.setToolTipDuration(tool_tip_duration)
            sub_layout.addWidget(icon)
        sub_layout.addStretch(1)

        return widget

    def __visualize(self, result):
        """Renders the simulation result when run is complete

        :param result: simulation result
        :type result: SimulationResult
        """
        if not self.simulation.isRunning():
            self.renderSimualtion(result)

    def renderSimualtion(self, result=None):
        """Renders the simulation result

        :param result: simulation result
        :type result: Union[SimulationResult, None]
        """
        positioner = self.parent_model.instrument.positioning_stack
        if result is None:
            self.parent.scenes.changeRenderedAlignment(self.default_vector_alignment)
            self.__movePositioner(self.robot_configuration)
            self.parent.scenes.resetCollision()
        else:
            self.parent.scenes.changeRenderedAlignment(result.alignment)

            if positioner.name != self.simulation.positioner_name:
                self.banner.showMessage(
                    f'Simulation cannot be visualized because positioning system '
                    f'("{self.simulation.positioner_name}") was changed.', MessageType.Error)
                return

            self.__movePositioner(result.ik.q)

            if result.collision_mask is None:
                return

            if (self.simulation.scene_size != len(result.collision_mask)
                    or not self.simulation.validateInstrumentParameters(self.parent_model.instrument)):
                self.banner.showMessage(
                    'Collision results could be incorrect because sample, jaw or detector was'
                    ' moved/changed', MessageType.Warning)
                return

            self.parent.scenes.renderCollision(result.collision_mask)

    def __movePositioner(self, end):
        """Moves the positioner from current position to end configuration

        :param end: end configuration
        :type end: numpy.ndarray
        """
        time = 50
        step = 2
        start = None
        positioner = self.parent_model.instrument.positioning_stack
        start = positioner.configuration if start is None else start
        positioner.set_points = end
        self.parent_model.moveInstrument(lambda q, s=positioner: s.fkine(q, set_point=False), start, end, time, step)
        self.parent_model.instrument_controlled.emit(CommandID.MovePositioner)

    def closeEvent(self, event):
        if self.simulation is not None:
            if self.simulation.isRunning():
                options = ['Stop', 'Cancel']
                res = self.parent.showSelectChoiceMessage(
                    'The current simulation will be terminated before dialog is '
                    'closed.\n\nDo you want proceed with this action?',
                    options,
                    cancel_choice=1)
                if res == options[1]:
                    event.ignore()
                    return

                self.simulation.abort()

            self.parent.scenes.changeVisibility(Attributes.Beam, False)
            self.renderSimualtion()
        event.accept()


class ScriptExportDialog(QtWidgets.QDialog):
    """Creates a UI for rendering and exporting instrument script

    :param simulation: simulation result
    :type simulation: Simulation
    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, simulation, parent):
        super().__init__(parent)

        self.parent = parent
        self.parent_model = parent.presenter.model
        self.results = simulation.results

        self.template = self.parent_model.instrument.script
        self.show_mu_amps = self.template.Key.mu_amps.value in self.template.keys
        self.createTemplateKeys()

        main_layout = QtWidgets.QVBoxLayout()
        if self.show_mu_amps:
            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(QtWidgets.QLabel('Duration of Measurements (microamps):'))
            self.micro_amp_textbox = QtWidgets.QLineEdit(self.template.keys[self.template.Key.mu_amps.value])
            validator = QtGui.QDoubleValidator(self.micro_amp_textbox)
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            validator.setDecimals(3)
            self.micro_amp_textbox.setValidator(validator)
            self.micro_amp_textbox.setMaxLength(12)
            self.micro_amp_textbox.textEdited.connect(self.preview)
            layout.addStretch(1)
            layout.addWidget(self.micro_amp_textbox)
            main_layout.addLayout(layout)

        self.preview_label = QtWidgets.QTextEdit()
        self.preview_label.setReadOnly(True)
        self.preview_label.setMinimumHeight(350)
        main_layout.addWidget(self.preview_label)
        self.preview()

        layout = QtWidgets.QHBoxLayout()
        self.export_button = QtWidgets.QPushButton('Export')
        self.export_button.clicked.connect(self.export)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.close)
        layout.addStretch(1)
        layout.addWidget(self.export_button)
        layout.addWidget(self.cancel_button)
        main_layout.addLayout(layout)

        self.setLayout(main_layout)
        self.setMinimumSize(640, 560)
        self.setWindowTitle('Export Script')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def createTemplateKeys(self):
        """Creates the initial template keys"""
        temp = {
            self.template.Key.script.value: [],
            self.template.Key.position.value: '',
            self.template.Key.filename.value: self.parent_model.save_path,
            self.template.Key.mu_amps.value: '0.000',
            self.template.Key.count.value: len(self.results)
        }

        header = '\t'.join(self.template.header_order)
        temp[self.template.Key.header.value] = header.replace(self.template.Key.position.value,
                                                              '\t'.join(self.results[0].joint_labels), 1)

        for key in self.template.keys:
            self.template.keys[key] = temp[key]

    def renderScript(self, preview=False):
        """Renders the simulation result into a script using template. For a preview,
        only 10 results are rendered

        :param preview: indicates if render is a preview
        :type preview: bool
        :return: rendered script
        :rtype: str
        """
        key = self.template.Key
        script = []
        size = 0
        for result in self.results:
            if size == 10 and preview:
                break

            if result.skipped or result.ik.status == IKSolver.Status.Failed:
                continue

            script.append({key.position.value: '\t'.join('{:.3f}'.format(value) for value in result.formatted)})
            size += 1

        if self.show_mu_amps:
            self.template.keys[key.mu_amps.value] = self.micro_amp_textbox.text()
        self.template.keys[key.script.value] = script
        self.template.keys[key.count.value] = len(script)

        return self.template.render()

    def preview(self):
        """Renders and displays a preview of the script"""
        script = self.renderScript(preview=True)
        if len(self.results) > 10:
            self.preview_label.setText(f'[Maximum of 10 points shown in preview] \n\n{script}')
        else:
            self.preview_label.setText(script)

    def export(self):
        """Exports the simulation result in a script"""
        if self.parent.presenter.exportScript(self.renderScript):
            self.accept()


class PathLengthPlotter(QtWidgets.QDialog):
    """Creates a UI for displaying the path lengths from the simulation result

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.setStyleSheet('QComboBox { min-width: 60px; }')
        self.simulation = self.parent.presenter.model.simulation

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        tool_layout = QtWidgets.QHBoxLayout()
        tool_layout.setSpacing(20)
        tool_layout.addStretch(1)
        tool_layout.setAlignment(QtCore.Qt.AlignBottom)
        self.main_layout.addLayout(tool_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(1)
        self.alignment_combobox = QtWidgets.QComboBox()
        self.alignment_combobox.setView(QtWidgets.QListView())
        self.alignment_combobox.addItems(['All', *[f'{i}' for i in range(1, self.simulation.shape[2] + 1)]])
        self.alignment_combobox.setMinimumWidth(100)
        self.alignment_combobox.activated.connect(self.plot)
        if self.simulation.shape[2] > 1:
            layout.addWidget(QtWidgets.QLabel('Alignment: '))
            layout.addWidget(self.alignment_combobox)
            tool_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(1)
        self.detector_combobox = QtWidgets.QComboBox()
        self.detector_combobox.setView(QtWidgets.QListView())
        self.detector_combobox.addItems(['Both', *self.simulation.detector_names])
        self.detector_combobox.setMinimumWidth(100)
        self.detector_combobox.activated.connect(self.plot)
        if self.simulation.shape[1] > 1:
            layout.addWidget(QtWidgets.QLabel('Detector: '))
            layout.addWidget(self.detector_combobox)
            tool_layout.addLayout(layout)

        # layout = QtWidgets.QVBoxLayout()
        # layout.setSpacing(1)
        self.grid_checkbox = QtWidgets.QCheckBox('Show Grid')
        self.grid_checkbox.clicked.connect(self.plot)
        # layout.addSpacing(15)
        tool_layout.addWidget(self.grid_checkbox)
        # tool_layout.addLayout(layout)

        tool_layout.addStretch(1)
        self.export_button = QtWidgets.QPushButton('Export')
        self.export_button.clicked.connect(self.export)
        tool_layout.addWidget(self.export_button)

        self.figure = Figure((6.0, 8.0), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.axes = self.figure.subplots()
        self.main_layout.addWidget(self.canvas, 1)

        self.setMinimumSize(800, 800)
        self.setWindowTitle('Path Length')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.plot()

    def export(self):
        """Exports a path length as image or text file"""
        save_path = self.parent.presenter.model.save_path
        save_path = f'{os.path.splitext(save_path)[0]}_path_length' if save_path else ''
        filename = self.parent.showSaveDialog('Image File (*.png);; Text Files (*.txt)',
                                              current_dir=save_path,
                                              title='Export Path Length')

        if not filename:
            return

        try:
            if os.path.splitext(filename)[1].lower() == '.txt':
                header = 'Index'
                data = [self.axes.lines[0].get_xdata()]
                fmt = ['%d']
                for line in self.axes.lines:
                    data.append(line.get_ydata())
                    fmt.append('%.3f')
                    header = f'{header}\t{line.get_label()}'

                header = f'{self.axes.get_title()}\n{header}'
                np.savetxt(filename, np.column_stack(data), fmt=fmt, delimiter='\t', header=header, comments='')
            else:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
        except OSError as e:
            self.parent.presenter.notifyError(f'An error occurred while exporting the path lengths to {filename}.', e)

    def plot(self):
        """Plots the path lengths"""
        self.axes.clear()
        self.axes.grid(self.grid_checkbox.isChecked(), which='both')

        alignment_index = self.alignment_combobox.currentIndex() - 1
        detector_index = self.detector_combobox.currentIndex() - 1

        names = self.simulation.detector_names
        path_length = self.simulation.path_lengths
        colours = [None] * len(names)
        colours[:2] = [settings.value(settings.Key.Vector_1_Colour), settings.value(settings.Key.Vector_2_Colour)]
        if detector_index > -1:
            names = names[detector_index:detector_index + 1]
            path_length = path_length[:, detector_index:detector_index + 1, :]
            colours = colours[detector_index:detector_index + 1]

        if alignment_index > -1:
            path_length = path_length[:, :, alignment_index]
            title = f'Path lengths for measurements in alignment {alignment_index + 1}'
        else:
            if self.simulation.args['align_first_order']:
                path_length = np.column_stack(np.dsplit(path_length, path_length.shape[2])).reshape(-1, len(names))
            else:
                path_length = np.row_stack(np.dsplit(path_length, path_length.shape[2])).reshape(-1, len(names))
            title = 'Path lengths for measurements'

        label = np.arange(1, path_length.shape[0] + 1)
        step = 1 if path_length.shape[0] < 10 else path_length.shape[0] // 10

        self.axes.set_xticks(label[::step])
        for i in range(path_length.shape[1]):
            self.axes.plot(label, path_length[:, i], 'o-', label=names[i], color=colours[i])

        self.axes.legend()
        self.axes.set_title(title)
        self.axes.set_xlabel('Measurement Index', labelpad=10)
        self.axes.set_ylabel('Path Length (mm)')
        self.axes.minorticks_on()
        self.canvas.draw()


class InstrumentCoordinatesDialog(QtWidgets.QDialog):
    """Creates a dialog for displaying the fiducial coordinates in the instrument coordinate system
     after the instrument has moved position and the instrument positioner matrix

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setWindowTitle('Instrument Coordinates')
        self.setMinimumSize(480, 500)
        self.banner = Banner(MessageType.Warning, self)
        self.banner.hide()
        self.main_layout.addWidget(self.banner)

        self.export_fiducials_action = QtWidgets.QAction("Fiducial Points", self)
        self.export_fiducials_action.triggered.connect(self.exportFiducials)

        self.export_matrix_action = QtWidgets.QAction("Positioning Stack Pose", self)
        self.export_matrix_action.triggered.connect(self.exportMatrix)
        export_menu = QtWidgets.QMenu()
        export_menu.addAction(self.export_matrix_action)
        export_menu.addAction(self.export_fiducials_action)

        self.export_button = create_tool_button(text='Export', style_name='TextButton', tooltip='Export to file')
        self.export_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.export_button.setMenu(export_menu)
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.export_button)
        self.main_layout.addLayout(layout)

        self.createControlPanel()
        self.parent.presenter.model.fiducials_changed.connect(self.setFiducialsData)
        self.parent.presenter.model.instrument_model_updated.connect(self.checkIfPositionerMoved)
        self.parent.presenter.model.instrument_controlled.connect(self.checkIfPositionerMoved)

        if self.parent.presenter.model.project_data is not None:
            self.setMatrixData()
            self.setFiducialsData()

    def createControlPanel(self):
        """Creates the control panel widgets"""
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.createMatrixTab()
        self.createFiducialsTab()
        self.main_layout.addWidget(self.tabs)

    def createFiducialsTab(self):
        """Creates a tab for displaying the sample and fiducial coordinates in the instrument coordinate system
        after the instrument has moved position and a button to export those to file
        """
        layout = QtWidgets.QVBoxLayout()

        self.fiducial_table_widget = QtWidgets.QTableWidget()
        self.fiducial_table_widget.setColumnCount(3)
        self.fiducial_table_widget.setShowGrid(True)
        self.fiducial_table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.fiducial_table_widget)

        fiducials_tab = QtWidgets.QWidget()
        fiducials_tab.setLayout(layout)
        self.tabs.addTab(create_scroll_area(fiducials_tab), 'Fiducial Points')

    def createMatrixTab(self):
        """Creates a tab for displaying the instrument pose matrix and a button for exporting that to file"""
        layout = QtWidgets.QVBoxLayout()

        self.matrix_table_widget = QtWidgets.QTableWidget()
        self.matrix_table_widget.setColumnCount(4)
        self.matrix_table_widget.setRowCount(4)
        self.matrix_table_widget.setShowGrid(True)
        self.matrix_table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.matrix_table_widget)

        matrix_tab = QtWidgets.QWidget()
        matrix_tab.setLayout(layout)
        self.tabs.addTab(create_scroll_area(matrix_tab), 'Positioning Stack Pose')

    def checkIfPositionerMoved(self, command_id):
        """If the sample stack has been moved or changed then update the coordinates of the fiducials

        :param command_id: Value of the enum describing which command has been sent
        :type command_id: Union[int, None]
        """
        commands = [CommandID.MovePositioner, CommandID.ChangePositionerBase, CommandID.ChangePositioningStack]
        if command_id is None or command_id in commands:  # Positioning stack moved, changed, or updated
            self.setFiducialsData()
            self.setMatrixData()

    def setFiducialsData(self):
        """Sets the table header and inserts the data values into the cells"""
        self.fiducial_table_widget.clear()
        self.fiducial_table_widget.setHorizontalHeaderLabels(['X (mm)', 'Y (mm)', 'Z (mm)'])
        self.fiducial_table_widget.setRowCount(len(self.parent.presenter.model.fiducials['points']))
        self.fiducial_table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.fiducials_coordinates = self.fiducialToPosition()

        for row, entry in enumerate(self.fiducials_coordinates):
            for column in range(3):
                item = QtWidgets.QTableWidgetItem(f'{entry[column]:.3f}')
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.fiducial_table_widget.setItem(row, column, item)

    def fiducialToPosition(self):
        """Converts fiducial positions from sample coordinate frame to instrument frame

        :return: Fiducial coordinates in instrument frame
        :rtype: np.ndarray
        """
        base_data = self.parent.presenter.model.fiducials
        self.export_fiducials_action.setDisabled(self.parent.presenter.model.alignment is None)
        if self.parent.presenter.model.alignment is None:
            self.banner.showMessage(
                'Sample has not been aligned on instrument, coordinates are in sample coordinate frame.',
                MessageType.Warning,
                no_close=True)
            return base_data.points

        self.banner.hide()
        stack = self.parent.presenter.model.instrument.positioning_stack
        pose = stack.fkine(stack.set_points, set_point=False) @ stack.tool_link
        transform = pose @ self.parent.presenter.model.alignment
        matrix = transform[0:3, 0:3].transpose()
        offset = transform[0:3, 3].transpose()

        return base_data.points @ matrix + offset

    def exportFiducials(self):
        """Writes out the data to a .fpos file if the sample has been aligned on the instrument"""
        if self.parent.presenter.model.alignment is None:
            return

        count = len(self.fiducials_coordinates)
        positioner = self.parent.presenter.model.instrument.positioning_stack
        poses = np.tile(positioner.toUserFormat(positioner.configuration), (count, 1))
        self.parent.presenter.exportCurrentFiducials(np.arange(count, dtype=int), self.fiducials_coordinates, poses)

    def setMatrixData(self):
        """Sets the table header and inserts the data values into the cells"""
        self.matrix_table_widget.clear()
        self.matrix_table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for row, entry in enumerate(self.parent.presenter.model.instrument.positioning_stack.pose):
            for column in range(4):
                item = QtWidgets.QTableWidgetItem(f'{entry[column]:.8f}')
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.matrix_table_widget.setItem(row, column, item)

    def exportMatrix(self):
        """Writes out the data to a .trans file"""
        name = FileDialog.getSaveFileName(self, 'Export Current Positioning Stack Pose', '',
                                          'Transformation Matrix File (*.trans)')
        if not name:
            return

        np.savetxt(name, self.parent.presenter.model.instrument.positioning_stack.pose, fmt='%.7f')


class CurveEditor(QtWidgets.QDialog):
    """Creates a UI for modifying a volumes transfer function curve

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.last_pos = None
        self._selected_index = 0

        main_layout = QtWidgets.QVBoxLayout()
        self.banner = Banner(MessageType.Warning, self)
        self.banner.hide()
        main_layout.addWidget(self.banner)
        self.setLayout(main_layout)
        sub_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(sub_layout)

        self.figure = Figure((5.0, 5.0), dpi=100)
        self.figure.patch.set_facecolor('None')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet('background-color: transparent;')
        self.canvas.mpl_connect('button_press_event', self.canvasMousePressEvent)
        self.canvas.mpl_connect('button_release_event', self.canvasMouseReleaseEvent)
        self.canvas.mpl_connect('motion_notify_event', self.canvasMouseMoveEvent)
        self.canvas.setParent(self)
        self.axes = self.figure.subplots()
        self.cursor_pos = self.axes.text(0.05, 0.95, "(0.00, 0.00)")
        sub_layout.addWidget(self.canvas, 5)

        group_layout = QtWidgets.QVBoxLayout()
        sub_layout.addLayout(group_layout, 2)

        self.group_box = QtWidgets.QGroupBox('Curve Options')
        group_layout.addWidget(self.group_box)
        group_layout.addStretch(1)

        control_layout = QtWidgets.QVBoxLayout()
        self.group_box.setLayout(control_layout)

        self.input_spinbox = QtWidgets.QDoubleSpinBox()
        self.input_spinbox.setDecimals(3)
        self.input_spinbox.valueChanged.connect(self.updateSelectedPoint)
        self.output_spinbox = QtWidgets.QDoubleSpinBox()
        self.output_spinbox.setDecimals(3)
        self.output_spinbox.setRange(0.0, 1.0)
        self.output_spinbox.valueChanged.connect(self.updateSelectedPoint)
        control_layout.addWidget(QtWidgets.QLabel('Input Intensity: '))
        control_layout.addWidget(self.input_spinbox)
        control_layout.addSpacing(10)
        control_layout.addWidget(QtWidgets.QLabel('Output Alpha: '))
        control_layout.addWidget(self.output_spinbox)
        control_layout.addSpacing(10)
        label = QtWidgets.QLabel('Curve Type: ')
        control_layout.addWidget(label)
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.setView(QtWidgets.QListView())
        self.type_combobox.addItems([t.value for t in Curve.Type])
        self.type_combobox.activated.connect(self.createCurve)
        control_layout.addWidget(self.type_combobox)
        control_layout.addSpacing(10)
        tool_layout = QtWidgets.QHBoxLayout()
        control_layout.addLayout(tool_layout)
        self.delete_button = create_tool_button(tooltip='Delete selected point',
                                                style_name='MidToolButton',
                                                icon_path=path_for('cross.png'))
        self.delete_button.clicked.connect(self.deleteSelected)
        tool_layout.addWidget(self.delete_button)
        self.reset_button = create_tool_button(tooltip='Resets curve to default',
                                               style_name='MidToolButton',
                                               icon_path=path_for('refresh.png'))
        self.reset_button.clicked.connect(self.reset)
        tool_layout.addWidget(self.reset_button)
        tool_layout.addStretch(1)

        button_layout = QtWidgets.QHBoxLayout()
        self.accept_button = QtWidgets.QPushButton('Accept')
        self.accept_button.clicked.connect(self.submit)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setDefault(True)

        button_layout.addStretch(1)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addSpacing(20)
        main_layout.addLayout(button_layout)

        self.setMinimumSize(800, 600)
        self.setWindowTitle('Curve Editor')

        self.updateVolume()
        self.parent.presenter.model.sample_changed.connect(self.updateVolume)

    def updateVolume(self):
        sample = self.parent.presenter.model.sample
        sample_is_volume = isinstance(sample, Volume)
        if sample_is_volume:
            self.banner.hide()
            histogram, edge = sample.histogram
            self.histogram = histogram / histogram.max()
            self.histogram = np.concatenate((self.histogram[:1], self.histogram, self.histogram[-1:]))
            self.edge = (edge[:-1] + edge[1:]) / 2
            self.edge = np.concatenate((edge[:1], self.edge, edge[-1:]))
            self.default_curve = sample.curve
            self.inputs = self.default_curve.inputs.copy()
            self.outputs = self.default_curve.outputs.copy()

        else:
            self.banner.showMessage('No volume has been added to the project.', MessageType.Warning, no_close=True)
            self.histogram = np.array([0., 1.])
            self.edge = np.array([0., 1.])
            self.inputs = np.array([0., 1.])
            self.outputs = np.array([0., 1.])
            self.default_curve = Curve(self.inputs, self.outputs, self.edge, Curve.Type.Cubic)

        self.canvas.setEnabled(sample_is_volume)
        self.group_box.setEnabled(sample_is_volume)
        self.accept_button.setEnabled(sample_is_volume)
        self.input_spinbox.setRange(self.edge[0], self.edge[-1])
        self.reset()

    def deleteSelected(self):
        """Deletes selected point from the curve"""
        if len(self.inputs) < 2:
            return

        self.inputs = np.delete(self.inputs, self.selected_index)
        self.outputs = np.delete(self.outputs, self.selected_index)
        self.selected_index = 0
        self.createCurve()

    def getSelectedPoint(self, event):
        """Determines the index of the selected point in the canvas

        :param event: mouse event
        :type event: matplotlib.backend_bases.MouseEvent
        :return: index of selected point
        :rtype: Union[int, None]
        """
        points = np.vstack((self.inputs, self.outputs)).T
        xy_pixels = self.axes.transData.transform(points)
        dist = np.sqrt((xy_pixels[:, 0] - event.x)**2 + (xy_pixels[:, 1] - event.y)**2)
        ind = dist.argmin()

        if dist[ind] >= 5:
            ind = None

        return ind

    def updateSelectedPoint(self):
        """Updates the  intensity and alpha value of the selected point"""
        self.__updateAndMerge(self.input_spinbox.value(), self.output_spinbox.value(), self.selected_index)
        self.createCurve()
        self.parent.scenes.previewVolumeCurve(self.curve)

    def __updateAndMerge(self, x, y, selected_index):
        """Updates the  intensity and alpha value of the selected point and removes any points in
        between the selected point's old position and its new position. This gives the impression
        of merging the removed points with the selected point as it is dragged to the new location.


        :param x: new intensity value for the selected point
        :type x: float
        :param y: new alpha value for the selected point
        :type y: float
        :param selected_index: selected index
        :type selected_index: int
        :return: new selected index
        :rtype: int
        """
        self.inputs[selected_index] = x
        self.outputs[selected_index] = y

        intensity, index, inv_index = np.unique(self.inputs, return_index=True, return_inverse=True)
        alpha = self.outputs[index]
        new_index = inv_index[selected_index]

        if new_index > selected_index:
            start = selected_index if len(self.inputs) > len(intensity) else selected_index + 1
            intensity = np.delete(self.inputs, slice(start, new_index + 1))
            alpha = np.delete(self.outputs, slice(start, new_index + 1))
        elif new_index < selected_index:
            intensity = np.delete(self.inputs, slice(new_index, selected_index))
            alpha = np.delete(self.outputs, slice(new_index, selected_index))
            selected_index = new_index

        self.inputs = intensity
        self.outputs = alpha
        self.selected_index = selected_index

        return selected_index

    def canvasMousePressEvent(self, event):
        """Matplotlib canvas mouse press event handler

        :param event: mouse event
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if event.button != 1 or event.inaxes is None:
            return

        selected_index = self.getSelectedPoint(event)
        if selected_index is None:
            x = max(min(trunc(event.xdata, self.input_spinbox.decimals()), self.edge[-1]), self.edge[0])
            y = max(min(trunc(event.ydata, self.output_spinbox.decimals()), 1.0), 0.0)
            self.inputs = np.append(self.inputs, x)
            self.outputs = np.append(self.outputs, y)
            self.inputs, index, inv_index = np.unique(self.inputs, return_index=True, return_inverse=True)
            self.outputs = self.outputs[index]
            self.selected_index = inv_index[-1]
        else:
            self.selected_index = selected_index
        self.last_pos = self.selected_index
        self.createCurve()

    def canvasMouseReleaseEvent(self, event):
        """Matplotlib canvas mouse release event handler

        :param event: mouse event
        :type event: matplotlib.backend_bases.MouseEvent
        """
        if event.button != 1:
            return

        self.last_pos = None
        self.parent.scenes.previewVolumeCurve(self.curve)

    def canvasMouseMoveEvent(self, event):
        """Matplotlib canvas mouse move event handler

        :param event: mouse event
        :type event: matplotlib.backend_bases.MouseEvent
        """

        if event.inaxes is None:
            return

        x = max(min(trunc(event.xdata, self.input_spinbox.decimals()), self.edge[-1]), self.edge[0])
        y = max(min(trunc(event.ydata, self.output_spinbox.decimals()), 1.0), 0.0)
        self.cursor_pos.set_text(f'({x:.2f}, {y:.2f})')

        if self.last_pos is None or event.button != 1:
            self.canvas.draw()
            return

        self.last_pos = self.__updateAndMerge(x, y, self.last_pos)

        self.createCurve()

    def createCurve(self):
        """Creates a new Curve object with selected inputs"""
        self.curve = Curve(self.inputs, self.outputs, (self.edge[0], self.edge[-1]),
                           Curve.Type(self.type_combobox.currentText()))
        self.plot()

    def plot(self):
        """Plots the curve"""
        margin = 0.05
        cursor_txt = self.cursor_pos.get_text()
        self.axes.clear()
        self.cursor_pos = self.axes.text(0.05,
                                         0.95,
                                         cursor_txt,
                                         transform=self.axes.transAxes,
                                         bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.2))

        self.axes.set_xlabel('Intensity')
        self.axes.set_ylabel('Normalized Alpha')
        self.axes.set_xlim(self.edge[0] - (margin * self.edge[-1]), self.edge[-1] + (margin * self.edge[-1]))
        self.axes.set_ylim(-0.05, 1.05)

        line_x = np.array([self.edge[0], *self.inputs, self.edge[-1]])
        if len(self.inputs) > 1:
            line_x = np.sort(np.concatenate((line_x, np.linspace(self.inputs[0], self.inputs[-1], 1000))))
        line_y = self.curve.evaluate(line_x)

        self.axes.fill_between(self.edge, self.histogram, -margin, alpha=0.2)
        self.axes.plot(line_x, line_y, 'b')  # plot lines
        self.axes.plot(self.inputs, self.outputs, color='red', marker='o', linestyle='none', markerfacecolor='none')
        self.axes.plot(self.inputs[self.selected_index], self.outputs[self.selected_index], 'ro')
        self.canvas.draw()
        self.figure.tight_layout()

    def reset(self):
        """Resets the curve"""
        self.inputs = self.default_curve.inputs.copy()
        self.outputs = self.default_curve.outputs.copy()
        self.curve = self.default_curve
        self.type_combobox.setCurrentText(self.curve.type.value)
        self.selected_index = 0
        self.plot()
        self.parent.scenes.previewVolumeCurve(self.default_curve)

    @property
    def selected_index(self):
        """Gets the index of the selected point in the canvas"""
        return self._selected_index

    @selected_index.setter
    def selected_index(self, index):
        """Sets the index of the selected point in the canvas

        :param index: index of selected point
        :type index: int
        """
        self._selected_index = index
        self.input_spinbox.blockSignals(True)
        self.input_spinbox.setValue(self.inputs[self.selected_index])
        self.input_spinbox.blockSignals(False)

        self.output_spinbox.blockSignals(True)
        self.output_spinbox.setValue(self.outputs[self.selected_index])
        self.output_spinbox.blockSignals(False)

    def submit(self):
        """Changes the volume curve"""
        self.parent.scenes.previewVolumeCurve(self.default_curve)
        if self.curve is not self.default_curve:
            self.parent.presenter.changeVolumeCurve(self.curve)
        self.accept()

    def reject(self):
        self.reset()
        super().reject()
