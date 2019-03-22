import os
import re
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.util import DockFlag
from sscanss.ui.widgets import AlignmentErrorModel, ErrorDetailModel, Banner, Accordion, Pane


class ProjectDialog(QtWidgets.QDialog):

    formSubmitted = QtCore.pyqtSignal(str, str)
    recentItemDoubleClicked = QtCore.pyqtSignal(str)
    # max number of recent projects to show in dialog is fixed because of the
    # dimensions of the dialog window
    max_recent_size = 5

    def __init__(self, recent, parent):
        super().__init__(parent)

        self.recent = recent
        self.instruments = list(parent.presenter.model.instruments.keys())
        data = parent.presenter.model.project_data
        self.selected_instrument = None if data is None else data['instrument'].name

        if len(self.recent) > self.max_recent_size:
            self.recent_list_size = self.max_recent_size
        else:
            self.recent_list_size = len(self.recent)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setMinimumSize(640, 480)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.createImageHeader()
        self.createTabWidgets()
        self.createStackedWidgets()
        self.createNewProjectWidgets()
        self.createRecentProjectWidgets()
        self.create_project_button.clicked.connect(self.createProjectButtonClicked)

        main_window_view = self.parent()
        self.formSubmitted.connect(main_window_view.presenter.createProject)
        self.recentItemDoubleClicked.connect(main_window_view.presenter.openProject)

        self.project_name_textbox.setFocus()

    def createImageHeader(self):
        img = QtWidgets.QLabel()
        img.setPixmap(QtGui.QPixmap('../static/images/banner.png'))
        self.main_layout.addWidget(img)

    def createTabWidgets(self):
        self.group = QtWidgets.QButtonGroup()
        self.button_layout = QtWidgets.QHBoxLayout()
        self.new_project_button = QtWidgets.QPushButton('Create New Project')
        self.new_project_button.setObjectName('CustomTab')
        self.new_project_button.setCheckable(True)
        self.new_project_button.setChecked(True)

        self.load_project_button = QtWidgets.QPushButton('Open Existing Project')
        self.load_project_button.setObjectName('CustomTab')
        self.load_project_button.setCheckable(True)

        self.group.addButton(self.new_project_button, 0)
        self.group.addButton(self.load_project_button, 1)
        self.button_layout.addWidget(self.new_project_button)
        self.button_layout.addWidget(self.load_project_button)
        self.button_layout.setSpacing(0)

        self.main_layout.addLayout(self.button_layout)

    def createStackedWidgets(self):
        self.stack = QtWidgets.QStackedLayout()
        self.main_layout.addLayout(self.stack)

        self.stack1 = QtWidgets.QWidget()
        self.stack1.setContentsMargins(30, 0, 30, 0)
        self.stack2 = QtWidgets.QWidget()
        self.stack2.setContentsMargins(30, 0, 30, 0)
        self.group.buttonClicked[int].connect(self.stack.setCurrentIndex)

        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)

    def createNewProjectWidgets(self):
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

        self.stack1.setLayout(layout)

    def createProjectButtonClicked(self):
        name = self.project_name_textbox.text().strip()
        instrument = self.instrument_combobox.currentText()
        if name:
            self.formSubmitted.emit(name, instrument)
        else:
            self.validator_textbox.setText('Project name cannot be left blank.')

    def createRecentProjectWidgets(self):
        layout = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setObjectName('Recents')
        self.list_widget.setSpacing(10)

        for i in range(self.recent_list_size):
            item = QtWidgets.QListWidgetItem(self.recent[i])
            item.setIcon(QtGui.QIcon('../static/images/file-black.png'))
            self.list_widget.addItem(item)

        item = QtWidgets.QListWidgetItem('Open ...')
        item.setIcon(QtGui.QIcon('../static/images/folder-open.png'))
        self.list_widget.addItem(item)

        self.list_widget.itemDoubleClicked.connect(self.projectItemDoubleClicked)
        layout.addWidget(self.list_widget)

        self.stack2.setLayout(layout)

    def projectItemDoubleClicked(self, item):
        index = self.list_widget.row(item)

        if index == self.recent_list_size:
            self.recentItemDoubleClicked.emit('')
        else:
            self.recentItemDoubleClicked.emit(item.text())
        self.accept()


class ProgressDialog(QtWidgets.QDialog):

    def __init__(self, message, parent):
        super().__init__(parent)

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setTextVisible(False)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        message = QtWidgets.QLabel(message)
        message.setAlignment(QtCore.Qt.AlignCenter)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addStretch(1)
        main_layout.addWidget(progress_bar)
        main_layout.addWidget(message)
        main_layout.addStretch(1)

        self.setLayout(main_layout)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setMinimumSize(300, 120)

    def keyPressEvent(self, _):
        """
        This ensure the user cannot close the dialog box with the Esc key
        """
        pass


class AlignmentErrorDialog(QtWidgets.QDialog):

    def __init__(self, parent):
        super().__init__(parent)

        self.measured_points = np.empty(0)
        self.transform_result = None
        self.main_layout = QtWidgets.QVBoxLayout()
        self.banner = Banner(Banner.Type.Info, self)
        self.main_layout.addWidget(self.banner)
        self.banner.hide()

        self.main_layout.addSpacing(5)
        self.result_text = '<p style="font-size:14px">The Average (RMS) Error is ' \
                           '<span style="color:{};font-weight:500;">{:.3f}</span> {}</p>'
        self.result_label = QtWidgets.QLabel()
        self.result_label.setTextFormat(QtCore.Qt.RichText)
        self.updateResultText(0.0)
        self.main_layout.addWidget(self.result_label)
        self.main_layout.addSpacing(10)
        self.createTabWidgets()
        self.createStackedWidgets()
        self.createSummaryTable()
        self.createDetailTable()

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

    def createTabWidgets(self):
        self.tabs = QtWidgets.QButtonGroup()
        tab_layout = QtWidgets.QHBoxLayout()
        self.summary_tab = QtWidgets.QPushButton('Summary')
        self.summary_tab.setObjectName('CustomTab')
        self.summary_tab.setCheckable(True)
        self.summary_tab.setChecked(True)

        self.detail_tab = QtWidgets.QPushButton('Detailed Analysis')
        self.detail_tab.setObjectName('CustomTab')
        self.detail_tab.setCheckable(True)

        self.tabs.addButton(self.summary_tab, 0)
        self.tabs.addButton(self.detail_tab, 1)
        tab_layout.addWidget(self.summary_tab)
        tab_layout.addWidget(self.detail_tab)
        tab_layout.setSpacing(0)

        self.main_layout.addLayout(tab_layout)

    def createStackedWidgets(self):
        self.stack = QtWidgets.QStackedLayout()
        self.main_layout.addLayout(self.stack)
        self.stack1 = QtWidgets.QWidget()
        self.stack2 = QtWidgets.QWidget()

        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)
        self.tabs.buttonClicked[int].connect(self.changeTab)

    def createSummaryTable(self):
        layout = QtWidgets.QVBoxLayout()
        self.summary_table_view = QtWidgets.QTableView()
        self.summary_table_model = AlignmentErrorModel()
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
        self.stack1.setLayout(layout)

    def createDetailTable(self):
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
        self.stack2.setLayout(layout)

    def recalculate(self):
        if self.measured_points.size == 0:
            return

        if np.count_nonzero(self.summary_table_model.enabled) < 3:
            self.banner.showMessage('A minimum of 3 points is required for sample alignment.',
                                    Banner.Type.Error)
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
        self.detail_table_model.update()
        self.detail_table_view.update()
        self.summary_table_model.update()
        self.summary_table_view.update()

    def indexOrder(self, new_order):
        self.banner.actionButton('FIX', lambda ignore, n=new_order: self.__correctIndexOrder(n))
        self.banner.showMessage('Incorrect Point Indices have been detected.',
                                Banner.Type.Warn, False)

    def __correctIndexOrder(self, new_index):
        self.summary_table_model.point_index = new_index
        self.recalculate()

    def updateResultText(self, average_error):
        if average_error < 0.1:
            colour = 'SeaGreen'
            self.banner.hide()
        else:
            colour = 'firebrick'
            self.banner.showMessage('If the errors are larger than desired, '
                                    'Try disabling points with the large errors then recalculate.',
                                    Banner.Type.Info)
        self.result_label.setText(self.result_text.format(colour, average_error, 'mm'))

    def updateModel(self, index, enabled, measured_points, transform_result):
        error = np.full(enabled.size, np.nan)
        error[enabled] = transform_result.error

        self.summary_table_model.point_index = index
        self.summary_table_model.error = error
        self.summary_table_model.enabled = enabled
        self.transform_result = transform_result
        self.measured_points = measured_points
        self.updateResultText(self.transform_result.average)
        self.detail_table_model.details = self.transform_result.distance_analysis
        self.detail_table_model.index_pairs = index[enabled]
        self.updateTable()

    def changeTab(self, index):
        self.stack.setCurrentIndex(index)
        self.cancel_button.setDefault(True)

    def submit(self):
        if self.transform_result is None:
            return
        self.parent().scenes.switchToInstrumentScene()
        self.parent().presenter.alignSample(self.transform_result.matrix)

        self.accept()


class FileDialog(QtWidgets.QFileDialog):
    def __init__(self, parent, caption, directory, filters):
        super().__init__(parent, caption, directory, filters)

        self.filters = self.extractFilters(filters)
        self.setOptions(QtWidgets.QFileDialog.DontConfirmOverwrite)

    def extractFilters(self, filters):
        filters = re.findall(r'\*.\w+', filters)
        return [f[1:] for f in filters]

    @property
    def filename(self):
        filename = self.selectedFiles()[0]
        _, ext = os.path.splitext(filename)
        expected_ext = self.extractFilters(self.selectedNameFilter())[0]
        if ext not in self.filters:
            filename = f'{filename}{expected_ext}'

        return filename

    @staticmethod
    def getOpenFileName(parent, caption, directory, filters):
        dialog = FileDialog(parent, caption, directory, filters)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if dialog.exec() != QtWidgets.QFileDialog.Accepted:
            return ''

        filename = dialog.filename

        if not os.path.isfile(filename):
            message = f'{filename} file not found.\nCheck the file name and try again.'
            QtWidgets.QMessageBox.warning(parent, caption, message, QtWidgets.QMessageBox.Ok,
                                          QtWidgets.QMessageBox.Ok)
            return ''

        return filename

    @staticmethod
    def getSaveFileName(parent, caption, directory, filters):
        dialog = FileDialog(parent, caption, directory, filters)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        if dialog.exec() != QtWidgets.QFileDialog.Accepted:
            return ''

        filename = dialog.filename

        if os.path.isfile(filename):
            buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            message = f'{filename} already exists.\nDo want to replace it?'
            reply = QtWidgets.QMessageBox.warning(parent, caption, message, buttons,
                                                  QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.No:
                return ''

        return filename


class SampleExportDialog(QtWidgets.QDialog):

    def __init__(self, sample_list, parent):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_widget.setMinimumHeight(200)
        self.list_widget.setSpacing(2)
        self.list_widget.addItems(sample_list)
        self.list_widget.setCurrentRow(0)
        self.list_widget.itemClicked.connect(self.deselection)
        layout.addWidget(self.list_widget)

        button_layout = QtWidgets.QHBoxLayout()
        self.accept_button = QtWidgets.QPushButton('OK')
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setDefault(True)

        button_layout.addStretch(1)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setWindowTitle('Select Sample to Save ...')

    def deselection(self, item):
        """prevent deselection by ensuring the clicked item is always selected"""
        if not item.isSelected():
            self.list_widget.setCurrentItem(item)

    @property
    def selected(self):
        return self.list_widget.currentItem().text()


class SimulationDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Full

    def __init__(self, parent):
        super().__init__(parent)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.simulation = parent.presenter.model.simulation
        if self.simulation is not None and self.simulation.isRunning():
            parent.scenes.switchToInstrumentScene()

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)
        main_layout.addSpacing(10)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.addStretch(1)
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.export_button = QtWidgets.QPushButton('Export Script')
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.export_button)
        main_layout.addLayout(button_layout)

        self.result_list = Accordion()
        main_layout.addWidget(self.result_list)
        self.setLayout(main_layout)

        self.title = 'Simulation Result'
        self.setMinimumWidth(400)
