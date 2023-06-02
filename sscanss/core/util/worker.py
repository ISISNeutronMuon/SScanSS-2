"""
Class for worker thread object
"""
from PyQt6 import QtCore


class Singleton(type(QtCore.QObject), type):
    """Metaclass used to create a PyQt singleton"""
    def __init__(cls, name, bases, cls_dict):
        super().__init__(name, bases, cls_dict)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class ProgressReport(QtCore.QObject, metaclass=Singleton):
    """Creates a singleton class to update progress bar. This is designed to report for
    multistep operations"""
    progress_updated = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.initialized = False

    def start(self, message, total_steps=1):
        """Starts the report

        :param message: operation message
        :type message: str
        :param total_steps: number of steps for the report
        :type total_steps: int
        """
        self.initialized = True
        self.message = message
        self.total_steps = total_steps
        self.chunk_percentage = [0.0] * self.total_steps
        self.current_step = 1
        self.current_chunk_size = 1
        self.progress_updated.emit(0.0)

    def beginStep(self, message=''):
        """Starts the report if not started. This avoids starting an already in-progress
        report in a nested function call

        :param message: operation message
        :type message: str
        """
        if not self.initialized:
            self.start(message)

    def nextStep(self):
        """Moves to the next step"""
        next_step = self.current_step + 1
        if next_step > self.total_steps:
            return

        self.current_chunk_size = next_step - sum(self.chunk_percentage)
        self.current_step = next_step

    def updateProgress(self, percentage):
        """Updates progress for the current step

        :param percentage: percentage of progress for current step [0, 1]
        :type percentage: float
        """
        self.chunk_percentage[self.current_step - 1] = percentage * self.current_chunk_size
        self.progress_updated.emit(self.percentage)

    @property
    def percentage(self):
        """Gets the percentage progress for all steps

        :return: percentage [0, 1]
        :rtype: float
        """
        return sum(self.chunk_percentage) / self.total_steps

    def completeStep(self):
        """Completes the current step of the report"""
        self.chunk_percentage[self.current_step - 1] = self.current_chunk_size
        self.progress_updated.emit(self.percentage)
        if self.total_steps == self.current_step:
            self.initialized = False

    def complete(self):
        """Completes the report for all steps"""
        self.chunk_percentage = [1.0] * self.total_steps
        self.progress_updated.emit(self.percentage)
        self.initialized = False


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
