from PyQt5.QtWidgets import QMainWindow


def do_nothing(*_args, **_kwargs):
    pass


class TestSignal:
    def __init__(self):
        self.call = do_nothing

    def connect(self, call):
        self.call = call

    def emit(self, *args):
        self.call(*args)


class TestView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.presenter = None
        self.scenes = None
        self.showSelectChoiceMessage = None
        self.showMessage = do_nothing
        self.showPathLength = do_nothing
        self.showScriptExport = do_nothing
