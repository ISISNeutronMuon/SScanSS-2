"""
Class for Scene object
"""
from contextlib import suppress
from collections import OrderedDict
import numpy as np
from PyQt6 import QtCore
from .camera import Camera
from .node import Node, VolumeNode
from .entity import (InstrumentEntity, PlaneEntity, BeamEntity, SampleEntity, FiducialEntity, MeasurementVectorEntity,
                     MeasurementPointEntity)
from ..util.misc import Attributes
from ..geometry.mesh import BoundingBox
from ..geometry.volume import Volume


def validate_instrument_scene_size(instrument):
    """Checks that the instrument scene size is within maximum extents

    :param instrument: instrument object
    :type instrument: Instrument
    :return: indicates if scene size is valid
    :rtype: bool
    """
    s = Scene()
    node = Node()
    node.vertices = InstrumentEntity(instrument).vertices
    s.addNode('', node)
    return not s.invalid


class Scene:
    """Creates a container for the nodes in the scene"""

    max_extent = 5000000  # cap for scene radius in mm

    def __init__(self):
        self._data = OrderedDict()
        self.bounding_box = None
        self.camera = Camera(1.0, 60, [-0.473, -0.788, -0.394], [0.0, 0.0, 1.0])
        self.invalid = False
        self.extent = 0.0

    @property
    def nodes(self):
        """Gets the top-level nodes in scene

        :return: top-level nodes
        :rtype: List[Node]
        """
        nodes = self._data.values()
        render_mode = self._data.get(Attributes.Sample, Node()).render_mode
        if render_mode == Node.RenderMode.Transparent:
            nodes = reversed(nodes)
        return list(nodes)

    def addNode(self, key, node):
        """Adds a non-empty node to the scene and updates bounding box

        :param key: name of node
        :type key: Any
        :param node: node
        :type node: Node
        """
        if node.isEmpty():
            self.removeNode(key)
            return

        self._data[key] = node
        # Ensures that the sample is drawn last so transparency is rendered properly
        if Attributes.Sample in self._data:
            self._data.move_to_end(Attributes.Sample, False)
        self.updateBoundingBox()

    def removeNode(self, key):
        """Removes node with specified key from the scene

        :param key: key of the node to remove
        :type key: Any
        """
        with suppress(KeyError):
            del self._data[key]
            self.updateBoundingBox()

    def updateBoundingBox(self):
        """Recalculates the bounding box after a node is added or removed"""

        if self.isEmpty():
            self.bounding_box = None
            self.extent = 0.0
            return

        self.bounding_box = BoundingBox.merge([node.bounding_box for node in self.nodes])
        self.extent = self.bounding_box.center.length + self.bounding_box.radius
        if not np.isfinite(self.extent) or self.extent > self.max_extent:
            self.invalid = True
        else:
            self.invalid = False

    def isEmpty(self):
        """Checks if Scene is empty

        :return: indicates scene is empty
        :rtype: bool
        """
        if self._data:
            return False
        return True

    def __contains__(self, key):
        if key in self._data:
            return True
        return False

    def __getitem__(self, key):
        return self._data.get(key, Node())


class SceneManager(QtCore.QObject):
    """Manages the instrument and sample scene and draws the active scene in the
    MainWindow's OpenGL widget.

    :param model: main window instance or model with instrument and sample data
    :type model: Union[MainWindow, MainWindowModel]
    :param renderer: graphics renderer
    :type renderer: OpenGLRenderer
    :param use_sample_scene: indicates if the sample scene should be used
    :type use_sample_scene: bool
    """
    rendered_alignment_changed = QtCore.pyqtSignal(int)

    def __init__(self, model, renderer, use_sample_scene=True):
        super().__init__()

        self.model = model
        self.renderer = renderer
        self.use_sample_scene = use_sample_scene

        self.active_scene = None
        self.sample_scene = Scene()
        self.instrument_scene = Scene()
        self.active_scene = self.sample_scene if use_sample_scene else self.instrument_scene

        self.sequence = None

        self.plane_entity = None
        self.instrument_entity = None
        self._rendered_alignment = 0
        self.visible_state = {attr: True for attr in Attributes}
        self.visible_state[Attributes.Beam] = False
        self.sample_render_mode = Node.RenderMode.Transparent

    @property
    def rendered_alignment(self):
        """Gets and Sets the index of the rendered vector alignment

        :return: index of rendered vector alignment
        :rtype: int
        """
        return self._rendered_alignment

    @rendered_alignment.setter
    def rendered_alignment(self, value):
        self.changeRenderedAlignment(value)

    def toggleScene(self):
        """Toggles the active scene between sample and instrument scenes"""
        if self.active_scene is self.sample_scene:
            self.active_scene = self.instrument_scene
        else:
            self.active_scene = self.sample_scene

        self.drawActiveScene()

    def switchToSampleScene(self):
        """Sets the active scene to the sample scene"""
        self.switchSceneTo(self.sample_scene)

    def switchToInstrumentScene(self):
        """Sets the active scene to the instrument scene"""
        self.switchSceneTo(self.instrument_scene)

    def switchSceneTo(self, scene):
        """Sets the active scene to the given scene object

        :param scene: scene object
        :type scene: Scene
        """
        if self.active_scene is not scene:
            self.active_scene = scene
            self.drawActiveScene()

    def previewVolumeCurve(self, curve):
        """Sets the transfer function for viewing a volume node

        :param curve: volume curve
        :type curve: Curve
        """
        for scene in [self.sample_scene, self.instrument_scene]:
            if Attributes.Sample in scene:
                node = scene[Attributes.Sample]
                if isinstance(node, VolumeNode):
                    node.updateTransferFunction(curve.transfer_function)

        self.drawActiveScene(False)

    def reset(self):
        """Resets the instrument and sample scenes and makes sample scene active"""

        self.sample_scene = Scene()
        self.instrument_scene = Scene()
        self.active_scene = self.sample_scene if self.use_sample_scene else self.instrument_scene
        self.drawActiveScene()

    def drawScene(self, scene, zoom_to_fit=True):
        """Draws the given scene if it is the active scene

        :param scene: scene to draw
        :type scene: Scene
        :param zoom_to_fit: indicates if scene camera should be zoomed to fit content
        :type zoom_to_fit: bool
        """
        if self.active_scene is scene:
            self.drawActiveScene(zoom_to_fit)

    def drawActiveScene(self, zoom_to_fit=True):
        """Draws the active scene in the OpenGL widget

        :param zoom_to_fit: indicates if scene camera should be zoomed to fit content
        :type zoom_to_fit: bool
        """
        self.renderer.loadScene(self.active_scene, zoom_to_fit)

    def changeRenderMode(self, render_mode):
        """Sets the render mode of the sample in both scenes

        :param render_mode: sample render mode
        :type render_mode: Node.RenderMode
        """
        self.sample_render_mode = render_mode
        if Attributes.Sample in self.sample_scene:
            self.sample_scene[Attributes.Sample].render_mode = render_mode
        if Attributes.Sample in self.instrument_scene:
            self.instrument_scene[Attributes.Sample].render_mode = render_mode

        self.drawActiveScene(False)

    def changeVisibility(self, key, visible):
        """Sets the visibility of an attribute in both scenes

        :param key: scene attribute
        :type key: Attributes
        :param visible: indicates if attribute is visible
        :type visible: bool
        """
        self.visible_state[key] = visible
        if key in self.sample_scene:
            self.sample_scene[key].visible = visible
        if key in self.instrument_scene:
            self.instrument_scene[key].visible = visible

        self.drawActiveScene(False)

    def changeSelected(self, key, selections):
        """Sets the selected property of children of an attribute node in the sample scene.
        This is used to highlight sample, points and vectors.

        :param key: scene attribute
        :type key: Attributes
        :param selections: indicates if node is selected
        :type selections: List[bool]
        """
        if key in self.sample_scene:
            node = self.sample_scene[key]
            node.selected = selections

        self.drawScene(self.sample_scene, False)

    def changeRenderedAlignment(self, alignment):
        """Sets the index of measurement vector alignment to draw

        :param alignment: index of measurement vector
        :type alignment: int
        """
        if self.model.measurement_vectors.size == 0:
            return

        align_count = self.model.measurement_vectors.shape[2]
        if alignment == self._rendered_alignment or alignment >= align_count:
            return

        # self.active_scene[Attributes.Vectors].visible = self.parent.show_vectors_action.isChecked()
        old_alignment = self._rendered_alignment
        self._rendered_alignment = alignment
        children = self.active_scene[Attributes.Vectors].children

        children[alignment].visible = True
        if old_alignment < align_count:
            children[old_alignment].visible = False

        self.rendered_alignment_changed.emit(alignment)
        self.drawActiveScene(False)

    def drawPlane(self, plane, width, height):
        """Draws a plane in the sample scene

        :param plane: plane to draw
        :type plane: Plane
        :param width: width of plane
        :type width: float
        :param height: height of plane
        :type height: float
        """
        self.plane_entity = PlaneEntity(plane, width, height)
        self.sample_scene.addNode(Attributes.Plane, self.plane_entity.node())
        self.drawScene(self.sample_scene, False)

    def movePlane(self, offset):
        """Shifts existing plane by offset

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: Union[numpy.ndarray, Vector3]
        """
        if self.plane_entity is None:
            return

        self.plane_entity.offset = offset
        self.sample_scene.addNode(Attributes.Plane, self.plane_entity.node())
        self.drawScene(self.sample_scene, False)

    def removePlane(self):
        """Removes plane attribute from the sample scene"""
        self.plane_entity = None
        self.sample_scene.removeNode(Attributes.Plane)
        self.drawScene(self.sample_scene)

    def animateInstrument(self, sequence):
        """Initiates an animation sequence for the instrument scene

        :param sequence: animation sequence
        :type sequence: Sequence
        """
        if self.sequence is not None:
            self.sequence.stop()

        self.sequence = sequence
        self.sequence.frame_changed.connect(self.animateInstrumentScene)
        if self.active_scene is self.instrument_scene:
            self.sequence.start()
        else:
            self.sequence.setFrame(-1)

    def animateInstrumentScene(self):
        """Renders each frame of the instrument scene during animation. It's faster than
        'updateInstrumentScene' and avoids zooming on the camera."""
        instrument_node = self.instrument_scene[Attributes.Instrument]
        self.instrument_entity.updateTransforms(instrument_node)
        self.addBeamToScene(instrument_node.bounding_box)
        if self.use_sample_scene:
            alignment = self.model.alignment
            if alignment is not None:
                pose = self.model.instrument.positioning_stack.tool_pose
                transform = pose @ alignment
                self.instrument_scene[Attributes.Sample].transform = transform
                self.instrument_scene[Attributes.Fiducials].transform = transform
                self.instrument_scene[Attributes.Measurements].transform = transform
                self.instrument_scene[Attributes.Vectors].transform = transform

        self.drawScene(self.instrument_scene, False)

    def updateInstrumentScene(self, sequence=None):
        """Creates the instrument scene or initiates an animation sequence for the instrument scene

        :param sequence: animation sequence
        :type sequence: Union[Sequence, None]
        """
        if sequence is None:
            self.addInstrumentToScene()
        else:
            self.animateInstrument(sequence)

    def addInstrumentToScene(self):
        """Adds instrument model to the instrument scene"""
        old_extent = self.instrument_scene.extent
        self.resetCollision()
        self.instrument_entity = InstrumentEntity(self.model.instrument)
        instrument_node = self.instrument_entity.node()
        self.instrument_scene.addNode(Attributes.Instrument, instrument_node)
        self.addBeamToScene(instrument_node.bounding_box)
        if self.use_sample_scene:
            alignment = self.model.alignment
            if alignment is not None:
                pose = self.model.instrument.positioning_stack.tool_pose
                transform = pose @ alignment
                self.instrument_scene.addNode(Attributes.Sample, self.sample_scene[Attributes.Sample].copy(transform))
                self.instrument_scene.addNode(Attributes.Fiducials,
                                              self.sample_scene[Attributes.Fiducials].copy(transform))
                self.instrument_scene.addNode(Attributes.Measurements,
                                              self.sample_scene[Attributes.Measurements].copy(transform))
                self.instrument_scene.addNode(Attributes.Vectors, self.sample_scene[Attributes.Vectors].copy(transform))
            else:
                self.instrument_scene.removeNode(Attributes.Sample)
                self.instrument_scene.removeNode(Attributes.Fiducials)
                self.instrument_scene.removeNode(Attributes.Measurements)
                self.instrument_scene.removeNode(Attributes.Vectors)

        self.drawScene(self.instrument_scene, abs(self.instrument_scene.extent - old_extent) > 10)

    def updateSampleScene(self, key):
        """Adds sample elements with specified key to the sample scene and updates instrument scene if needed"""
        exception = None
        visible = self.visible_state[key]

        if key == Attributes.Sample:
            try:
                self.sample_scene.addNode(Attributes.Sample,
                                          SampleEntity(self.model.sample).node(self.sample_render_mode))
            except (MemoryError, OSError) as e:
                if isinstance(self.model.sample, Volume):
                    self.sample_scene.addNode(Attributes.Sample,
                                              SampleEntity(self.model.sample.asMesh()).node(self.sample_render_mode))
                    exception = e

        elif key == Attributes.Fiducials:
            self.sample_scene.addNode(Attributes.Fiducials, FiducialEntity(self.model.fiducials, visible).node())
        elif key == Attributes.Measurements:
            self.sample_scene.addNode(Attributes.Measurements,
                                      MeasurementPointEntity(self.model.measurement_points, visible).node())
        elif key == Attributes.Vectors:
            self.sample_scene.addNode(
                Attributes.Vectors,
                MeasurementVectorEntity(self.model.measurement_points, self.model.measurement_vectors,
                                        self.rendered_alignment, visible).node())

        if self.model.alignment is not None:
            self.updateInstrumentScene()

        self.drawScene(self.sample_scene)
        if exception is not None:
            self.renderer.reportError(exception)

    def addBeamToScene(self, bounds):
        """Adds beam model to the instrument scene

        :param bounds: scene bounds
        :type bounds: BoundingBox
        """
        node = BeamEntity(self.model.instrument, bounds, self.visible_state[Attributes.Beam]).node()
        self.instrument_scene.addNode(Attributes.Beam, node)

    def resetCollision(self):
        """Removes collision highlights"""
        self.instrument_scene[Attributes.Sample].resetOutline()
        self.instrument_scene[Attributes.Instrument].resetOutline()

    def renderCollision(self, collisions):
        """Highlights colliding objects

        :param collisions: indicates which objects are colliding
        :type collisions: List[bool]
        """
        if self.sequence is not None and self.sequence.isRunning():
            self.sequence.stop()

        sample_node = self.instrument_scene[Attributes.Sample]
        sample_node.outlined = collisions[0]
        node = self.instrument_scene[Attributes.Instrument]
        node.outlined = collisions[-len(node.outlined):]

        self.drawScene(self.instrument_scene, False)
