

class Presenter:
    """Main presenter for the editor app

    :param view: main window instance
    :type view: MainWindow
    :param model: main model instance
    :type model: Model
    """
    def __init__(self, view, model):
        self.view = view
        self.model = model


