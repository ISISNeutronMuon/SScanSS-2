from PyQt5 import QtCore
import os


class InstrumentWorker(QtCore.QThread):
    """Creates worker thread for updating instrument from the description file.

    :param parent: main window instance
    :type parent: MainWindow
    """
    job_succeeded = QtCore.pyqtSignal(object, object)
    job_failed = QtCore.pyqtSignal(Exception)

    def __init__(self, parent, presenter):
        super().__init__(parent)
        self.presenter = presenter
        self.model = None

    def run(self):
        """Updates instrument from description file"""
        try:
            widget_to_update = self.model.widget_to_update
            result = self.presenter.createInstrument()
            self.job_succeeded.emit(result, widget_to_update)
        except Exception as e:
            self.job_failed.emit(e)


class EditorModel(QtCore.QObject):
    """The model of the application, responsible for the computation"""
    def __init__(self, worker):
        super().__init__()

        self.current_file = ''
        self.saved_text = ''
        self.current_text = ''
        self.initialized = False
        self.widget_to_update = None
        self.instrument = None

        self.file_watcher = QtCore.QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(lambda: self.lazyInstrumentUpdate())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.useWorker)

        self.worker = worker
        self.worker.model = self

    def resetAddresses(self):
        """Resets the file addresses"""

        self.saved_text = ''
        self.current_file = ''
        self.initialized = False
        self.updateWatcher(self.current_file)

    def openFile(self, file_address):
        """Opens the file at given address and returns it
        :param file_address: opens the file address
        :type file_address: str
        :return: the text in the open file
        :rtype: str
        """
        with open(file_address, 'r') as idf:
            self.current_file = file_address
            self.saved_text = idf.read()
            self.updateWatcher(os.path.dirname(self.current_file))
            return self.saved_text

    def saveFile(self, text, filename):
        """saves the given text in given file
        :param text: the text which should be saved in the file
        :type text: str
        :param filename: address at which the file should be saved
        :type filename: str
        """
        with open(filename, 'w') as idf:
            idf.write(text)
            self.saved_text = text
            self.updateWatcher(os.path.dirname(filename))

    def updateWatcher(self, path):
        """Adds path to the file watcher, which monitors the path for changes to
        model or template files.

        :param path: file path of the instrument description file
        :type path: str
        """
        if self.file_watcher.directories():
            self.file_watcher.removePaths(self.file_watcher.directories())
        if path:
            self.file_watcher.addPaths([path, *[f.path for f in os.scandir(path) if f.is_dir()]])

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

    def useWorker(self):
        """Uses worker thread to create instrument from description"""
        if self.worker is not None and self.worker.isRunning():
            self.lazyInstrumentUpdate(100)
            return
        self.worker.start()
