import datetime
from enum import Enum, unique
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sscanss.config import path_for, __version__
from sscanss.core.instrument import IKSolver
from sscanss.core.util import DockFlag, Attributes
from sscanss.ui.widgets import (AlignmentErrorModel, ErrorDetailModel, Banner, Accordion, Pane, create_tool_button,
                                CenteredBoxProxy)


class AboutDialog(QtWidgets.QDialog):
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
                'the measurement procedure.</p> <p>SScanSS 2 is a rewrite of the SScanSS application developed by '
                '<b>Dr. Jon James</b> at the Open University in collaboration with the ISIS neutron facility.</p>'
                '<h3>Reference</h3>'
                '<ol><li>J. A. James, J. R. Santisteban, L. Edwards and M. R. Daymond, “A virtual laboratory for '
                'neutron and synchrotron strain scanning,” Physica B: Condensed Matter, vol. 350, no. 1-3, '
                'p. 743–746, 2004.</li></ol>'
                '<h3>Author</h3>'
                '<ul><li>Stephen Nneji</li></ul>'
                '<h3>Credit</h3>'
                '<ul><li>Icons from FontAwesome</li></ul>'
                '<hr/>'
                '<p style="text-align:center">Distributed under the BSD 3-Clause License</p>'
                f'<p style="text-align:center">Copyright &copy; 2018-{datetime.date.today().year}, ISIS Neutron and '
                'Muon Source. All rights reserved.</p>'
                '</div>')

        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)


class ProjectDialog(QtWidgets.QDialog):

    formSubmitted = QtCore.pyqtSignal(str, str)
    recentItemDoubleClicked = QtCore.pyqtSignal(str)
    # max number of recent projects to show in dialog is fixed because of the
    # dimensions of the dialog window
    max_recent_size = 5

    def __init__(self, recent, parent):
        super().__init__(parent)

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
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setMinimumSize(640, 480)
        self.main_layout.setContentsMargins(1, 1, 1, 1)

        self.createImageHeader()
        self.createTabWidgets()
        self.createStackedWidgets()
        self.createNewProjectWidgets()
        self.createRecentProjectWidgets()
        self.create_project_button.clicked.connect(self.createProjectButtonClicked)

        presenter = self.parent.presenter
        self.formSubmitted.connect(lambda name, inst: presenter.useWorker(presenter.createProject, [name, inst],
                                                                          presenter.updateView,
                                                                          presenter.projectCreationError, self.accept))
        self.recentItemDoubleClicked.connect(lambda name: presenter.useWorker(presenter.openProject, [name],
                                                                              presenter.updateView,
                                                                              presenter.projectOpenError, self.accept))

        self.project_name_textbox.setFocus()

    def createImageHeader(self):
        img = QtWidgets.QLabel()
        img.setPixmap(QtGui.QPixmap(path_for('banner.png')))
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
            item.setIcon(QtGui.QIcon(path_for('file-black.png')))
            self.list_widget.addItem(item)

        item = QtWidgets.QListWidgetItem('Open ...')
        item.setIcon(QtGui.QIcon(path_for('folder-open.png')))
        self.list_widget.addItem(item)

        self.list_widget.itemDoubleClicked.connect(self.projectItemDoubleClicked)
        layout.addWidget(self.list_widget)

        self.stack2.setLayout(layout)

    def projectItemDoubleClicked(self, item):
        index = self.list_widget.row(item)

        if index == self.recent_list_size:
            filename = self.parent.showOpenDialog('hdf5 File (*.h5)', title='Open Project',
                                                  current_dir=self.parent.presenter.model.save_path)
            if not filename:
                return
        else:
            filename = item.text()

        self.recentItemDoubleClicked.emit(filename)

    def reject(self):
        if self.parent.presenter.model.project_data is None:
            self.parent.statusBar().showMessage("Most menu options are disabled until you create/open a project.")
        super().reject()


class ProgressDialog(QtWidgets.QDialog):

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

    def show(self, message):
        self.message.setText(message)
        super().show()

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
        self.end_configuration = None
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
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
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
        if self.check_box.isChecked() and self.end_configuration is not None:
            self.parent().presenter.movePositioner(*self.end_configuration, ignore_locks=True)

        self.accept()


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
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)

    def deselection(self, item):
        """prevent deselection by ensuring the clicked item is always selected"""
        if not item.isSelected():
            self.list_widget.setCurrentItem(item)

    @property
    def selected(self):
        if self.list_widget.count() == 0:
            return ''
        return self.list_widget.currentItem().text()


class SimulationDialog(QtWidgets.QWidget):
    dock_flag = DockFlag.Full

    @unique
    class State(Enum):
        Starting = 0
        Running = 1
        Stopped = 2

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.parent_model = parent.presenter.model
        main_layout = QtWidgets.QVBoxLayout()

        self.banner = Banner(Banner.Type.Info, self)
        main_layout.addWidget(self.banner)
        self.banner.hide()
        main_layout.addSpacing(10)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        self.path_length_button = create_tool_button(tooltip='Plot Path Length', style_name='ToolButton',
                                                     status_tip='Plot calculated path length for current simulation',
                                                     icon_path=path_for('line-chart.png'))
        self.path_length_button.clicked.connect(self.parent.showPathLength)

        self.export_button = create_tool_button(tooltip='Export Script', style_name='ToolButton',
                                                status_tip='Export script for current simulation',
                                                icon_path=path_for('export.png'))
        self.export_button.clicked.connect(self.parent.showScriptExport)

        button_layout.addWidget(self.path_length_button)
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

        self.loadSimulation(no_render=True)
        self.parent_model.simulation_created.connect(self.loadSimulation)
        self.parent.scenes.switchToInstrumentScene()
        self.showResult()

    @property
    def simulation(self):
        return self.parent_model.simulation

    def loadSimulation(self, no_render=False):
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
            self.simulation.result_updated.connect(self.showResult)
            self.simulation.stopped.connect(lambda: self.updateProgress(SimulationDialog.State.Stopped))

    def updateProgress(self, state, increment=False):
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

    def showResult(self, error=False):
        if self.simulation is None:
            return

        state = SimulationDialog.State.Running if self.simulation.isRunning() else SimulationDialog.State.Stopped
        results = self.simulation.results[len(self.result_list.panes):]

        for result in results:
            result_text = '\n'.join('{:<20}{:>12.3f}'.format(*t) for t in zip(result.joint_labels, result.formatted))
            header = QtWidgets.QLabel()
            header.setTextFormat(QtCore.Qt.RichText)

            details = QtWidgets.QLabel()
            details.setTextFormat(QtCore.Qt.RichText)

            if result.ik.status == IKSolver.Status.Failed:
                header.setText(f'<span>{result.id}</span><br/> '
                               f'<span> A runtime error occurred. Check logs for more Information.</span>')
                style = Pane.Type.Error
            else:
                style = Pane.Type.Info if result.ik.status == IKSolver.Status.Converged else Pane.Type.Warn
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

                if self.simulation.check_collision and np.any(result.collision_mask):
                    info = (f'{info}<table cellspacing="0" border="0" style="color:red"><tr>'
                            f'<td><img src="{path_for("collision.png")}" height="25" width="25"></td>'
                            '<td style="vertical-align:middle;"><b>Collision Detected</b></td>'
                            '</tr></table>')

                header.setText(info)
                details.setText(f'<pre>{result_text}</pre>')
                if self.render_graphics:
                    self.renderSimualtion(result)

            self.result_list.addPane(self.__createPane(header, details, style, result))
            self.updateProgress(state, True)

        if error:
            self.updateProgress(SimulationDialog.State.Stopped)
            self.parent.showMessage('An error occurred while running the simulation.')

    def __createPane(self, panel, details, style, result):
        pane = Pane(panel, details, style)
        action = QtWidgets.QAction('Copy', pane)
        action.setStatusTip('Copy positioner offsets to clipboard')
        action_text = '\t'.join('{:.3f}'.format(t) for t in result.formatted)
        action.triggered.connect(lambda ignore, q=action_text:
                                 QtWidgets.QApplication.clipboard().setText(q))
        pane.addContextMenuAction(action)

        action = QtWidgets.QAction('Visualize', pane)
        action.setStatusTip('Visualize selected simulation result in the graphics window')
        action.triggered.connect(lambda ignore, r=result: self.__visualize(result))
        pane.addContextMenuAction(action)

        return pane

    def __visualize(self, result):
        if not self.simulation.isRunning():
            self.renderSimualtion(result)

    def renderSimualtion(self, result=None):
        positioner = self.parent_model.instrument.positioning_stack
        if result is None:
            self.parent.scenes.changeRenderedAlignment(self.default_vector_alignment)
            self.__movePositioner(positioner.set_points)
            self.parent.scenes.resetCollision()
        else:
            self.parent.scenes.changeRenderedAlignment(result.alignment)

            if positioner.name != self.simulation.positioner.name:
                self.banner.showMessage(f'Simulation cannot be visualized because positioning system '
                                        f'("{self.simulation.positioner.name}") was changed.', Banner.Type.Error)
                return

            self.__movePositioner(result.ik.q)

            if result.collision_mask is None:
                return

            if (self.simulation.scene_size != len(result.collision_mask) or
                    not self.simulation.validateInstrumentParameters(self.parent_model.instrument)):
                self.banner.showMessage('Collision results could be incorrect because sample, jaw or detector was'
                                        ' moved/changed', Banner.Type.Warn)
                return

            self.parent.scenes.renderCollision(result.collision_mask)

    def __movePositioner(self, end):
        time = 50
        step = 2
        start = None
        positioner = self.parent_model.instrument.positioning_stack
        start = positioner.configuration if start is None else start
        self.parent_model.moveInstrument(lambda q, s=positioner: s.fkine(q, setpoint=False), start, end, time, step)

    def closeEvent(self, event):
        if self.simulation is not None:
            if self.simulation.isRunning():
                options = ['Stop', 'Cancel']
                res = self.parent.showSelectChoiceMessage('The current simulation will be terminated before dialog is '
                                                          'closed.\n\nDo you want proceed with this action?',
                                                          options, default_choice=1)
                if res == options[1]:
                    event.ignore()
                    return

                self.simulation.abort()

        self.parent.scenes.changeVisibility(Attributes.Beam, False)
        self.renderSimualtion()
        event.accept()


class ScriptExportDialog(QtWidgets.QDialog):
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
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)

    def createTemplateKeys(self):
        temp = {self.template.Key.script.value: [],
                self.template.Key.position.value: '',
                self.template.Key.filename.value: self.parent_model.save_path,
                self.template.Key.mu_amps.value: '0.000',
                self.template.Key.count.value: len(self.results)}

        header = '\t'.join(self.template.header_order)
        temp[self.template.Key.header.value] = header.replace(self.template.Key.position.value,
                                                              '\t'.join(self.results[0].joint_labels), 1)

        for key in self.template.keys:
            self.template.keys[key] = temp[key]

    def renderScript(self, preview=False):
        key = self.template.Key
        count = len(self.results)
        size = 10 if count > 10 and preview else count
        script = []
        for i in range(size):
            script.append({key.position.value: '\t'.join('{:.3f}'.format(l) for l in self.results[i].formatted)})

        if self.show_mu_amps:
            self.template.keys[key.mu_amps.value] = self.micro_amp_textbox.text()
        self.template.keys[key.script.value] = script

        return self.template.render()

    def preview(self):
        script = self.renderScript(preview=True)
        if len(self.results) > 10:
            self.preview_label.setText(f'[Maximum of 10 points shown in preview] \n\n{script}')
        else:
            self.preview_label.setText(script)

    def export(self):
        if self.parent.presenter.exportScript(self.renderScript):
            self.accept()


class PathLengthPlotter(QtWidgets.QDialog):

    def __init__(self, parent):
        super().__init__(parent)

        self.simulation = parent.presenter.model.simulation

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        tool_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(tool_layout)
        if self.simulation.shape[2] > 1:
            tool_layout.addStretch(1)
            self.alignment_combobox = QtWidgets.QComboBox()
            self.alignment_combobox.setView(QtWidgets.QListView())
            self.alignment_combobox.addItems([f'{k + 1}' for k in range(self.simulation.shape[2])])
            self.alignment_combobox.activated.connect(self.plot)
            tool_layout.addWidget(QtWidgets.QLabel('Select Alignment: '))
            tool_layout.addWidget(self.alignment_combobox)

        self.figure = Figure((5.0, 6.0), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.axes = self.figure.subplots()
        self.main_layout.addWidget(self.canvas)

        self.setMinimumSize(800, 800)
        self.setWindowTitle('Path Length')
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.plot()

    def plot(self, index=0):
        self.axes.clear()

        if self.simulation.compute_path_length:
            path_length = self.simulation.path_lengths[:, :, index]
            names = self.simulation.detector_names
            step = 1 if path_length.shape[0] < 10 else path_length.shape[0]//10
            label = np.arange(1, path_length.shape[0] + 1)
            self.axes.set_xticks(label[::step])
            for i in range(path_length.shape[1]):
                self.axes.plot(label, path_length[:, i], '+--', label=names[i])

            self.axes.legend()
        else:
            self.axes.set_xticks([1, 2, 3, 4])
        self.axes.set_xlabel('Measurement Point', labelpad=10)
        self.axes.set_ylabel('Path Length (mm)')
        self.axes.grid(True, which='both')
        self.axes.set_ylim(bottom=0.)
        self.axes.minorticks_on()
        self.canvas.draw()
