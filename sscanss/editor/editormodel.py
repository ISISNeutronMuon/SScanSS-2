from PyQt5 import QtCore


class InstrumentWorker(QtCore.QThread):
    """Creates worker thread for updating instrument from the description file.

    :param parent: main window instance
    :type parent: MainWindow
    """
    job_succeeded = QtCore.pyqtSignal(object)
    job_failed = QtCore.pyqtSignal(Exception)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.job_succeeded.connect(self.parent.setInstrumentSuccess)
        self.job_failed.connect(self.parent.setInstrumentFailed)

    def run(self):
        """Updates instrument from description file"""
        try:
            result = self.parent.setInstrument()
            self.job_succeeded.emit(result)
        except Exception as e:
            self.job_failed.emit(e)


class EditorModel:
    """The model of the application, responsible for the computation"""
    def __init__(self):
        self.file_watcher = QtCore.QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(lambda: self.lazyInstrumentUpdate())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.useWorker)
        self.worker = InstrumentWorker(self)


    def lazyInstrumentUpdate(self, interval=300):
        """Updates instrument after the wait time elapses

        :param interval: wait time (milliseconds)
        :type interval: int
        """
        self.initialized = True
        self.timer.stop()
        self.timer.setSingleShot(True)
        self.timer.setInterval(interval)
        self.timer.start()


    def saveFile(self, save_as=False):
        """Saves the instrument description file. A file dialog should be opened for the first save
        after which the function will save to the same location. If save_as is True a dialog is
        opened every time

        :param save_as: A flag denoting whether to use file dialog or not
        :type save_as: bool
        """
        if not self.unsaved and not save_as:
            return

        filename = self.filename
        if save_as or not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Instrument Description File', '',
                                                                'Json File (*.json)')

        if not filename:
            return

        try:
            with open(filename, 'w') as idf:
                self.filename = filename
                text = self.editor.text()
                idf.write(text)
                self.saved_text = text
                self.updateWatcher(os.path.dirname(filename))
                self.setTitle()
            if save_as:
                self.resetInstrument()
        except OSError as e:
            self.message.setText(f'An error occurred while attempting to save this file ({filename}). \n{e}')