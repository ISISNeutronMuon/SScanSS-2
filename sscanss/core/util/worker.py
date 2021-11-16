"""
Class for worker thread object
"""
from PyQt5 import QtCore


class Worker(QtCore.QThread):
    """Creates worker thread object

    :param _exec: function to run on ``QThread``
    :type _exec: Callable[..., Any]
    :param args: arguments of function ``_exec``
    :type args: Tuple[Any, ...]
    """
    job_succeeded = QtCore.pyqtSignal('PyQt_PyObject')
    job_failed = QtCore.pyqtSignal(Exception, 'PyQt_PyObject')

    def __init__(self, _exec, args):
        super().__init__()
        self._exec = _exec
        self._args = args

    def run(self):
        """This function is executed on worker thread when the ``QThread.start``
        method is called."""
        try:
            result = self._exec(*self._args)
            self.job_succeeded.emit(result)
        except Exception as e:
            self.job_failed.emit(e, self._args)

    @classmethod
    def callFromWorker(cls, func, args, on_success=None, on_failure=None, on_complete=None):
        """Calls the given function from a new worker thread object

        :param func: function to run on ``QThread``
        :type func: Callable[..., Any]
        :param args: arguments of function ``func``
        :type args: Tuple[Any, ...]
        :param on_success: function to call if success
        :type on_success: Union[Callable[..., None], None]
        :param on_failure: function to call if failed
        :type on_failure: Union[Callable[..., None], None]
        :param on_complete: function to call when complete
        :type on_complete: Union[Callable[..., None], None]
        :return: worker thread running ``func``
        :rtype: Worker
        """
        worker = cls(func, args)
        if on_success is not None:
            worker.job_succeeded.connect(on_success)
        if on_failure is not None:
            worker.job_failed.connect(on_failure)
        if on_complete is not None:
            worker.finished.connect(on_complete)
        worker.start()

        return worker
