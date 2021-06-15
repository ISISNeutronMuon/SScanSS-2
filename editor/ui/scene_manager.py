"""
Class for the Editor's Scene Manager
"""

from PyQt5 import QtCore
from sscanss.core.scene import InstrumentEntity, BeamEntity, Scene
from sscanss.core.util import Attributes
from sscanss.core.instrument import Sequence


class SceneManager(QtCore.QObject):
    """Manages the scene and draws the scene in the MainWindow's OpenGL widget.

    :param parent: MainWindow object
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.scene = Scene(Scene.Type.Instrument)
        self.sequence = None
        self.parent.animate_instrument.connect(self.animateInstrument)

    def reset(self):
        """Resets the scenes"""
        self.scene = Scene(Scene.Type.Instrument)
        self.drawScene()

    def animateInstrument(self, func, start, stop, duration=1000, step=10):
        """Initiates animation sequence for the instrument scene

        :param func: function to generate frame at each way point
        :type func: method
        :param start: inclusive start joint configuration/offsets
        :type start: List[float]
        :param stop: inclusive stop joint configuration/offsets
        :type stop: List[float]
        :param duration: time duration in milliseconds
        :type duration: int
        :param step: number of steps
        :type step: int
        """
        if self.sequence is not None:
            self.sequence.stop()

        self.sequence = Sequence(func, start, stop, duration, step)
        self.sequence.frame_changed.connect(self.animateInstrumentScene)
        self.sequence.start()

    def animateInstrumentScene(self):
        """Renders each frame of the instrument scene during animation. It faster than
        'updateInstrumentScene' and avoids zooming on the camera."""
        self.addInstrumentToScene()
        self.parent.gl_widget.update()

    def updateInstrumentScene(self):
        """Updates the instrument scene"""
        old_extent = self.scene.extent
        self.addInstrumentToScene()
        self.drawScene(abs(self.scene.extent - old_extent) > 10)

    def drawScene(self, zoom_to_fit=True):
        """Draws the scene in the OpenGL widget

        :param zoom_to_fit: indicates scene camera should be zoomed to fit content
        :type zoom_to_fit: bool
        """
        self.parent.gl_widget.loadScene(self.scene, zoom_to_fit)

    def addBeamToScene(self, bounds):
        """Adds beam model to the scene

        :param bounds: scene bounds
        :type bounds: BoundingBox
        """
        node = BeamEntity(self.parent.instrument, bounds, True).node()
        self.scene.addNode(Attributes.Beam, node)

    def addInstrumentToScene(self):
        """Adds instrument model to the scene"""
        instrument_node = InstrumentEntity(self.parent.instrument).node()
        self.scene.addNode(Attributes.Instrument, instrument_node)
        self.addBeamToScene(instrument_node.bounding_box)
