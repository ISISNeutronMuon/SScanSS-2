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
        self.filename = ''
        self.saved_text = ''
        self.initialized = False
        self.unsaved = False
        """self.file_watcher = QtCore.QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(lambda: self.lazyInstrumentUpdate())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.useWorker)
        self.worker = InstrumentWorker(self)"""

    def getSavedText(self):
        return self.saved_text

    def createNewFile(self):
        self.saved_text = ''
        self.filename = ''
        self.initialized = False
        #self.updateWatcher(self.filename)
        #self.scene.reset()
        #self.controls.close()
        self.message.setText('')

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




