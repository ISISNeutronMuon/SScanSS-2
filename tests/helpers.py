def do_nothing(*args, **kwargs):
    pass


class TestSignal:
    def connect(self, call):
        self.call = call

    def emit(self, *args):
        self.call(*args)