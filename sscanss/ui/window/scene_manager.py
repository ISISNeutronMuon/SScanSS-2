from PyQt5 import QtCore
from sscanss.core.util import Attributes
from sscanss.core.scene import (FiducialEntity, MeasurementPointEntity, MeasurementVectorEntity, SampleEntity,
                                InstrumentEntity, PlaneEntity, BeamEntity, Scene)


class SceneManager(QtCore.QObject):
    """Manages the instrument and sample scene and draws the active scene in the
    MainWindow's OpenGL widget.

    :param parent: MainWindow object
    :type parent: MainWindow
    """
    
    rendered_alignment_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.parent_model = parent.presenter.model

        self.instrument_scene = Scene(Scene.Type.Instrument)
        self.sample_scene = Scene()
        self.active_scene = self.sample_scene
        self.plane_entity = None
        self.sequence = None
        self._rendered_alignment = 0
        self.parent_model.sample_scene_updated.connect(self.updateSampleScene)
        self.parent_model.instrument_scene_updated.connect(self.updateInstrumentScene)
        self.parent_model.animate_instrument.connect(self.animateInstrument)

    @property
    def rendered_alignment(self):
        return self._rendered_alignment

    def reset(self):
        """Resets the instrument and sample scenes and makes sample scene active"""
        self.instrument_scene = Scene(Scene.Type.Instrument)
        self.sample_scene = Scene()
        self.active_scene = self.sample_scene
        self.drawActiveScene()

    def switchToSampleScene(self):
        """Sets active scene to the sample scene"""
        self.switchSceneTo(self.sample_scene)

    def switchToInstrumentScene(self):
        """Sets active scene to the instrument scene"""
        self.switchSceneTo(self.instrument_scene)

    def switchSceneTo(self, scene):
        """Set active scene to the given scene object

        :param scene: scene object
        :type scene: Scene
        """
        if self.active_scene is not scene:
            self.active_scene = scene
            self.drawActiveScene()

    def changeRenderMode(self, render_mode):
        """Sets the render mode of the sample in both scenes

        :param render_mode: sample render mode
        :type render_mode: Node.RenderMode
        """
        Scene.sample_render_mode = render_mode
        if Attributes.Sample in self.sample_scene:
            self.sample_scene[Attributes.Sample].render_mode = render_mode
        if Attributes.Sample in self.instrument_scene:
            self.instrument_scene[Attributes.Sample].render_mode = render_mode

        self.drawActiveScene(False)

    def changeRenderedAlignment(self, alignment):
        """Sets the index of measurement vector alignment to draw

        :param alignment: index of measurement vector
        :type alignment: int
        """
        align_count = self.parent_model.measurement_vectors.shape[2]
        if alignment == self.rendered_alignment or alignment >= align_count:
            return
        self.active_scene[Attributes.Vectors].visible = self.parent.show_vectors_action.isChecked()
        old_alignment = self.rendered_alignment
        self._rendered_alignment = alignment
        children = self.active_scene[Attributes.Vectors].children

        children[alignment].visible = True
        children[old_alignment].visible = False

        self.rendered_alignment_changed.emit(alignment)
        self.drawActiveScene(False)

    def toggleScene(self):
        """Toggles the active scene between sample and instrument scenes"""
        if self.active_scene is self.sample_scene:
            self.active_scene = self.instrument_scene
        else:
            self.active_scene = self.sample_scene

        self.drawActiveScene()

    def changeVisibility(self, key, visible):
        """Sets the visibility of an attribute in both scenes

        :param key: scene attribute
        :type key: Attributes
        :param visible: indicates if attribute is visible
        :type visible: bool
        """
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
            # for selected, node in zip(selections, nodes):
            node.selected = selections

        self.drawScene(self.sample_scene, False)

    def animateInstrument(self, sequence):
        """Initiates animation sequence for the instrument scene

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
            self.sequence.animate(-1)

    def animateInstrumentScene(self):
        """Renders each frame of the instrument scene during animation. It faster than
        'updateInstrumentScene' and avoids zooming on the camera."""

        self.addInstrumentToScene()

        alignment = self.parent_model.alignment
        if alignment is not None:
            pose = self.parent_model.instrument.positioning_stack.tool_pose
            transform = pose @ alignment
            self.instrument_scene[Attributes.Sample].transform = transform
            self.instrument_scene[Attributes.Fiducials].transform = transform
            self.instrument_scene[Attributes.Measurements].transform = transform
            self.instrument_scene[Attributes.Vectors].transform = transform

        self.drawScene(self.instrument_scene, False)

    def updateInstrumentScene(self):
        """Creates the static instrument scene and adds/removes the sample elements as needed"""
        old_extent = self.instrument_scene.extent
        self.addInstrumentToScene()

        alignment = self.parent_model.alignment
        if alignment is not None:
            pose = self.parent_model.instrument.positioning_stack.tool_pose
            transform = pose @ alignment
            self.instrument_scene.addNode(Attributes.Sample, self.sample_scene[Attributes.Sample].copy(transform))
            self.instrument_scene.addNode(Attributes.Fiducials, self.sample_scene[Attributes.Fiducials].copy(transform))
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
        if key == Attributes.Sample:
            self.sample_scene.addNode(Attributes.Sample,
                                      SampleEntity(self.parent_model.sample).node(Scene.sample_render_mode))
        elif key == Attributes.Fiducials:
            visible = self.parent.show_fiducials_action.isChecked()
            self.sample_scene.addNode(Attributes.Fiducials, FiducialEntity(self.parent_model.fiducials, visible).node())
        elif key == Attributes.Measurements:
            visible = self.parent.show_measurement_action.isChecked()
            self.sample_scene.addNode(Attributes.Measurements,
                                      MeasurementPointEntity(self.parent_model.measurement_points, visible).node())
        elif key == Attributes.Vectors:
            visible = self.parent.show_vectors_action.isChecked()
            self.sample_scene.addNode(Attributes.Vectors,
                                      MeasurementVectorEntity(self.parent_model.measurement_points,
                                                              self.parent_model.measurement_vectors,
                                                              self.rendered_alignment, visible).node())

        if self.parent_model.alignment is not None:
            self.updateInstrumentScene()
        self.drawScene(self.sample_scene)

    def drawScene(self, scene, zoom_to_fit=True):
        """Draws the given scene if it is the active scene in the OpenGL widget

        :param scene: scene to draw
        :type scene: Scene
        :param zoom_to_fit: indicates scene camera should be zoomed to fit content
        :type zoom_to_fit: bool
        """
        if self.active_scene is scene:
            self.drawActiveScene(zoom_to_fit)

    def drawActiveScene(self, zoom_to_fit=True):
        """Draws the active scene in the OpenGL widget

        :param zoom_to_fit: indicates scene camera should be zoomed to fit content
        :type zoom_to_fit: bool
        """
        self.parent.gl_widget.loadScene(self.active_scene, zoom_to_fit)

    def drawPlane(self, plane, width, height):
        """Draws a plane in the sample scene or shifts existing plane by offset

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
        :type offset: Union[numpy.ndarray, sscanss.core.scene.Vector3]
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

    def addBeamToScene(self, bounds):
        """Adds beam model to the instrument scene

        :param bounds: scene bounds
        :type bounds: BoundingBox
        """
        instrument = self.parent_model.instrument
        node = self.instrument_scene[Attributes.Beam]
        visible = False if node.isEmpty() else node.visible
        node = BeamEntity(instrument, bounds, visible).node()
        self.instrument_scene.addNode(Attributes.Beam, node)

    def addInstrumentToScene(self):
        """Adds instrument model to the instrument scene"""
        self.resetCollision()
        instrument_node = InstrumentEntity(self.parent_model.instrument).node()
        self.instrument_scene.addNode(Attributes.Instrument, instrument_node)
        self.addBeamToScene(instrument_node.bounding_box)

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
        sample_node.outlined = collisions[:len(sample_node.batch_offsets)]
        node = self.instrument_scene[Attributes.Instrument]
        node.outlined = collisions[len(sample_node.batch_offsets):]

        self.drawScene(self.instrument_scene, False)
