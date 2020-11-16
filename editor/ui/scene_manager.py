from PyQt5 import QtCore
from sscanss.core.scene import createInstrumentNode, createBeamNode, Scene
from sscanss.core.util import Attributes
from sscanss.core.instrument import Sequence


class SceneManager(QtCore.QObject):

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.scene = Scene(Scene.Type.Instrument)
        self.sequence = None
        self.parent.animate_instrument.connect(self.animateInstrument)

    def reset(self):
        self.scene = Scene(Scene.Type.Instrument)
        self.drawScene()

    def animateInstrument(self, func, start_var, stop_var, duration=1000, step=10):
        if self.sequence is not None:
            self.sequence.stop()

        self.sequence = Sequence(func, start_var, stop_var, duration, step)
        self.sequence.frame_changed.connect(self.animateInstrumentScene)
        self.sequence.start()

    def animateInstrumentScene(self):
        self.addInstrumentToScene()
        self.addBeamToScene()
        self.parent.gl_widget.update()

    def updateInstrumentScene(self):
        old_extent = self.scene.extent
        self.addInstrumentToScene()
        self.addBeamToScene()
        self.drawScene(self.scene.extent > old_extent)

    def drawScene(self, zoom_to_fit=True):
        self.parent.gl_widget.loadScene(self.scene, zoom_to_fit)

    def addBeamToScene(self):
        node = createBeamNode(self.parent.instrument, self.scene.bounding_box, True)
        self.scene.addNode(Attributes.Beam, node)

    def addInstrumentToScene(self):
        self.scene.addNode(Attributes.Instrument, createInstrumentNode(self.parent.instrument))
