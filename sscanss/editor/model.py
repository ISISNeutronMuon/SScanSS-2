from PyQt5 import QtCore
from jsonschema.exceptions import ValidationError
import os


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
            result = self.presenter.createInstrument()
            self.job_succeeded.emit(result)
        except Exception as e:
            self.job_failed.emit(e)


class EditorModel(QtCore.QObject):
    """args: exception type, error message"""
    error_occurred = QtCore.pyqtSignal(Exception, str)

    """The model of the application, responsible for the computation"""
    def __init__(self, worker):
        super().__init__()

        self.current_file = ''
        self.saved_text = ''
        self.initialized = False
        self.unsaved = False

        self.file_watcher = QtCore.QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(lambda: self.lazyInstrumentUpdate())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.useWorker)
        self.worker = worker

    def getSavedText(self):
        return self.saved_text

    def getCurrentFile(self):
        return self.current_file

    def createNewFile(self):
        self.saved_text = ''
        self.current_file = ''
        self.initialized = False
        self.updateWatcher(self.current_file)

    def openFile(self,fileAddress):
        with open(fileAddress, 'r') as idf:
            self.current_file = fileAddress
            self.saved_text = idf.read()
            self.updateWatcher(os.path.dirname(self.current_file))
            return self.saved_text

    def saveFile(self,  text, filename):
        with open(filename, 'w') as idf:
            idf.write(text)
            self.saved_text = text
            self.updateWatcher(os.path.dirname(filename))

    def setInstrumentFailed(self, e):
        """Reports errors from instrument update worker

        :param e: raised exception
        :type e: Exception
        """
        if self.initialized:
            if isinstance(e, ValidationError):
                path = ''
                for p in e.absolute_path:
                    if isinstance(p, int):
                        path = f'{path}[{p}]'
                    else:
                        path = f'{path}.{p}' if path else p

                path = path if path else 'instrument description file'
                error_message = f'{e.message} in {path}'
            else:
                error_message = str(e).strip("'")

            self.error_occurred.emit(e, error_message)

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

