import os
from PyQt6 import QtCore


class InstrumentWorker(QtCore.QThread):
    """Creates worker thread for updating instrument from the description file.

    :param parent: main window instance
    :type parent: MainWindow
    """
    job_succeeded = QtCore.pyqtSignal(object)
    job_failed = QtCore.pyqtSignal(Exception)

    def __init__(self, parent, presenter):
        super().__init__(parent)
        self.presenter = presenter

    def run(self):
        """Updates instrument from description file"""
        try:
            json_text = self.presenter.view.editor.text()
            folder_path = os.path.dirname(self.presenter.model.filename)
            result = self.presenter.parser.parse(json_text, folder_path)
            self.job_succeeded.emit(result)
        except Exception as e:
            self.job_failed.emit(e)


class EditorModel(QtCore.QObject):
    """The model of the application, responsible for the computation"""
    def __init__(self, worker):
        super().__init__()

        self._filename = ''
        self.file_directory = '.'
        self.saved_text = ''
        self.initialized = False
        self.instrument = None

        self.file_watcher = QtCore.QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(lambda: self.lazyInstrumentUpdate())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.useWorker)

        self.worker = worker

    def reset(self):
        """Resets the model"""
        self.initialized = False
        self.saved_text = ''
        self.filename = ''

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value
        self.file_directory = os.path.dirname(value) if value else '.'
        self.updateWatcher(self.file_directory)

    def openFile(self, filename):
        """Opens the file at given address and returns it

        :param filename: path of file to open
        :type filename: str
        :return: text in the open file
        :rtype: str
        """
        with open(filename, 'r') as idf:
            self.filename = filename
            self.saved_text = idf.read()
            return self.saved_text

    def saveFile(self, text, filename):
        """saves the given text in given file

        :param text: the text which should be saved in the file
        :type text: str
        :param filename: path of file to save
        :type filename: str
        """
        with open(filename, 'w') as idf:
            idf.write(text)
            self.saved_text = text
            self.filename = filename

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
