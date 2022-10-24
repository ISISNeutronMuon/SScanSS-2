"""
Class for worker thread object
"""
from PyQt5 import QtCore


class Singleton(type(QtCore.QObject), type):
    """
    Metaclass used to create a PyQt singleton
    """
    def __init__(cls, name, bases, cls_dict):
        super().__init__(name, bases, cls_dict)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class ProgressReport(QtCore.QObject, metaclass=Singleton):
    """Create singleton class to update progress bar"""
    progress_updated = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.percentage = 0

    def updateProgress(self, progress):
        self.percentage = progress
        self.progress_updated.emit(progress)


class Worker(QtCore.QThread):
    """Creates worker thread object

    :param _exec: function to run on ``QThread``
    :type _exec: Callable[..., Any]
    :param args: arguments of function ``_exec``
    :type args: Tuple[Any, ...]
    """
    job_succeeded = QtCore.pyqtSignal('PyQt_PyObject')
    job_failed = QtCore.pyqtSignal(Exception, 'PyQt_PyObject')
    job_cancelled = QtCore.pyqtSignal()

    def __init__(self, _exec, args):
        super().__init__()
        self._exec = _exec
        self._args = args

    def printer(self, value):
        print(f'{value:.3f}%')

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
