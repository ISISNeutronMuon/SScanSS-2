from PyQt5 import QtCore
from sscanss.core.scene import (createSampleNode, createFiducialNode, createMeasurementPointNode,
                                createMeasurementVectorNode, createPlaneNode, createBeamNode, Scene)
from sscanss.core.util import Attributes


class SceneManager(QtCore.QObject):
    rendered_alignment_changed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.parent_model = parent.presenter.model

        self.instrument_scene = Scene(Scene.Type.Instrument)
        self.sample_scene = Scene()
        self.active_scene = self.sample_scene
        self.sequence = None
        self._rendered_alignment = 0
        self.parent_model.sample_scene_updated.connect(self.updateSampleScene)
        self.parent_model.instrument_scene_updated.connect(self.updateInstrumentScene)
        self.parent_model.animate_instrument.connect(self.animateInstrument)

    @property
    def rendered_alignment(self):
        return self._rendered_alignment

    @rendered_alignment.setter
    def rendered_alignment(self, value):
        self._rendered_alignment = value
        self.rendered_alignment_changed.emit()

    def reset(self):
        self.instrument_scene = Scene(Scene.Type.Instrument)
        self.sample_scene = Scene()
        self.active_scene = self.sample_scene
        self.drawActiveScene()

    def switchToSampleScene(self):
        self.switchSceneTo(self.sample_scene)

    def switchToInstrumentScene(self):
        self.switchSceneTo(self.instrument_scene)

    def switchSceneTo(self, scene):
        if self.active_scene is not scene:
            self.active_scene = scene
            self.drawActiveScene()

    def changeRenderMode(self, render_mode):
        if Attributes.Sample in self.sample_scene:
            self.sample_scene[Attributes.Sample].render_mode = render_mode
        if Attributes.Sample in self.instrument_scene:
            self.instrument_scene[Attributes.Sample].render_mode = render_mode

        self.drawActiveScene(False)

    def changeRenderedAlignment(self, alignment):
        align_count = self.parent_model.measurement_vectors.shape[2]
        if alignment == self.rendered_alignment:
            self.active_scene[Attributes.Vectors].visible = self.parent.show_vectors_action.isChecked()
        elif alignment >= align_count:
            self.active_scene[Attributes.Vectors].visible = False
        else:
            self.active_scene[Attributes.Vectors].visible = self.parent.show_vectors_action.isChecked()
            old_alignment = self.rendered_alignment
            self.rendered_alignment = alignment
            children = self.active_scene[Attributes.Vectors].children
            if not children:
                return
            detectors = len(self.parent_model.instrument.detectors)
            for index in range(detectors):
                children[alignment * detectors + index].visible = True
                children[old_alignment * detectors + index].visible = False

        self.drawActiveScene(False)

    def toggleScene(self):
        if self.active_scene is self.sample_scene:
            self.active_scene = self.instrument_scene
        else:
            self.active_scene = self.sample_scene

        self.drawActiveScene()

    def toggleVisibility(self, key, visible):
        if key in self.sample_scene:
            self.sample_scene[key].visible = visible
        if key in self.instrument_scene:
            self.instrument_scene[key].visible = visible

        self.drawActiveScene(False)

    def changeSelected(self, key, selections):
        if key in self.sample_scene:
            nodes = self.sample_scene[key].children
            for selected, node in zip(selections, nodes):
                node.selected = selected

        self.drawScene(self.sample_scene, False)

    def animateInstrument(self, sequence):
        if self.sequence is not None:
            self.sequence.stop()

        self.sequence = sequence
        self.sequence.frame_changed.connect(self.animateInstrumentScene)
        if self.active_scene is self.instrument_scene:
            self.sequence.start()
        else:
            self.sequence.animate(-1)

    def animateInstrumentScene(self):
        self.instrument_scene.addNode(Attributes.Instrument, self.parent_model.instrument.model())
        self.addBeamToScene()

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
        old_extent = self.instrument_scene.extent
        self.instrument_scene.addNode(Attributes.Instrument, self.parent_model.instrument.model())
        self.addBeamToScene()

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

        self.drawScene(self.instrument_scene, self.instrument_scene.extent > old_extent)

    def updateSampleScene(self, key):
        old_extent = self.sample_scene.extent
        if key == Attributes.Sample:
            self.addSampleToScene(self.sample_scene, self.parent_model.sample)
        elif key == Attributes.Fiducials:
            self.addFiducialsToScene(self.sample_scene, self.parent_model.fiducials)
        elif key == Attributes.Measurements:
            self.addMeasurementsToScene(self.sample_scene, self.parent_model.measurement_points)
        elif key == Attributes.Vectors:
            self.addVectorsToScene(self.sample_scene, self.parent_model.measurement_points,
                                   self.parent_model.measurement_vectors)
        self.updateInstrumentScene()
        self.drawScene(self.sample_scene, self.sample_scene.extent > old_extent)

    def drawScene(self, scene, zoom_to_fit=True):
        if self.active_scene is scene:
            self.drawActiveScene(zoom_to_fit)

    def drawActiveScene(self, zoom_to_fit=True):
        self.parent.gl_widget.loadScene(self.active_scene, zoom_to_fit)

    def drawPlane(self, plane=None, width=None, height=None, shift_by=None):
        if shift_by is not None and Attributes.Plane in self.sample_scene:
            self.sample_scene[Attributes.Plane].translate(shift_by)
        elif plane is not None and width is not None and height is not None:
            self.sample_scene.addNode(Attributes.Plane, createPlaneNode(plane, width, height))
        self.drawScene(self.sample_scene)

    def removePlane(self):
        self.sample_scene.removeNode(Attributes.Plane)
        self.drawScene(self.sample_scene)

    def addSampleToScene(self, scene, sample, transform=None):
        render_mode = self.parent.selected_render_mode
        scene.addNode(Attributes.Sample, createSampleNode(sample, render_mode, transform))

    def addFiducialsToScene(self, scene, fiducials, transform=None):
        visible = self.parent.show_fiducials_action.isChecked()
        scene.addNode(Attributes.Fiducials, createFiducialNode(fiducials, visible, transform))

    def addMeasurementsToScene(self, scene, points, transform=None):
        visible = self.parent.show_measurement_action.isChecked()
        scene.addNode(Attributes.Measurements, createMeasurementPointNode(points, visible, transform))

    def addVectorsToScene(self, scene, points, vectors, transform=None):
        visible = self.parent.show_vectors_action.isChecked()
        scene.addNode(Attributes.Vectors, createMeasurementVectorNode(points, vectors,
                                                                      self.rendered_alignment, visible, transform))

    def addBeamToScene(self):
        instrument = self.parent_model.instrument
        node = self.instrument_scene[Attributes.Beam]
        visible = False if node.isEmpty() else node.visible

        node = createBeamNode(instrument, self.instrument_scene.bounding_box, visible)

        self.instrument_scene.addNode(Attributes.Beam, node)
