def do_nothing(*_args, **_kwargs):
    pass


class TestSignal:
    def connect(self, call):
        self.call = call

    def emit(self, *args):
        self.call(*args)
