from PyQt5 import QtCore


class Worker(QtCore.QThread):
    job_succeeded = QtCore.pyqtSignal('PyQt_PyObject')
    job_failed = QtCore.pyqtSignal(Exception, 'PyQt_PyObject')

    def __init__(self, _exec, args):
        super().__init__()
        self._exec = _exec
        self._args = args

    def run(self):
        try:
            result = self._exec(*self._args)
            self.job_succeeded.emit(result)
        except Exception as e:
            self.job_failed.emit(e, self._args)

    @classmethod
    def callFromWorker(cls, func, args, on_success=None, on_failure=None, on_complete=None):
        worker = cls(func, args)
        if on_success is not None:
            worker.job_succeeded.connect(on_success)
        if on_failure is not None:
            worker.job_failed.connect(on_failure)
        if on_complete is not None:
            worker.finished.connect(on_complete)
        worker.start()

        return worker

