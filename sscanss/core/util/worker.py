from PyQt5 import QtCore


class Worker(QtCore.QThread):
    job_succeeded = QtCore.pyqtSignal()
    job_failed = QtCore.pyqtSignal(Exception)

    def __init__(self, _exec, args):
        super().__init__()
        self._exec = _exec
        self._args = args

    def run(self):
        try:
            self._exec(*self._args)
            self.job_succeeded.emit()
        except Exception as e:
            self.job_failed.emit(e)
