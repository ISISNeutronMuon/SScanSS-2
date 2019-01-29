from PyQt5 import QtCore
from sscanss.core.scene import (createSampleNode, createFiducialNode, createMeasurementPointNode,
                                createMeasurementVectorNode, createPlaneNode, Scene)
from sscanss.core.util import Attributes


class SceneManager(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.parent_model = parent.presenter.model

        self.instrument_scene = Scene(Scene.Type.Instrument)
        self.sample_scene = Scene()
        self.active_scene = self.sample_scene
        self.sequence = None
        self.rendered_alignment = 0
        self.parent_model.scene_updated.connect(self.updateScene)
        self.parent_model.animate_instrument.connect(self.animateInstrument)

    def switchToSampleScene(self):
        self.switchSceneTo(self.sample_scene)

    def switchToInstrumentScene(self):
        self.switchSceneTo(self.instrument_scene)

    def switchSceneTo(self, scene):
        if self.active_scene is not scene:
            self.active_scene = scene
            self.drawScene()

    def changeRenderMode(self, render_mode):
        if Attributes.Sample in self.sample_scene:
            self.sample_scene[Attributes.Sample].render_mode = render_mode
            self.drawScene()

    def toggleScene(self):
        if self.active_scene is self.sample_scene:
            self.active_scene = self.instrument_scene
        else:
            self.active_scene = self.sample_scene

        self.drawScene()

    def toggleVisibility(self, key, visible):
        if key in self.sample_scene:
            self.sample_scene[key].visible = visible

        self.drawScene()

    def animateInstrument(self, sequence):
        if self.sequence is not None:
            self.sequence.stop()

        self.sequence = sequence
        self.sequence.frame_changed.connect(lambda: self.updateScene(Attributes.Instrument))
        if self.active_scene is self.instrument_scene:
            self.sequence.start()
        else:
            self.sequence.animate(-1)

    def updateScene(self, key):
        if key == Attributes.Instrument:
            self.instrument_scene.addNode('instrument', self.parent_model.instrument.model())
            if self.active_scene is self.instrument_scene:
                self.drawScene()
            return

        if key == Attributes.Sample:
            render_mode = self.parent.selected_render_mode
            self.sample_scene.addNode(key, createSampleNode(self.parent_model.sample, render_mode))
        elif key == Attributes.Fiducials:
            visible = self.parent.show_fiducials_action.isChecked()
            self.sample_scene.addNode(key, createFiducialNode(self.parent_model.fiducials, visible))
        elif key == Attributes.Measurements:
            visible = self.parent.show_measurement_action.isChecked()
            self.sample_scene.addNode(key, createMeasurementPointNode(self.parent_model.measurement_points, visible))
        elif key == Attributes.Vectors:
            visible = self.parent.show_vectors_action.isChecked()
            self.sample_scene.addNode(key, createMeasurementVectorNode(self.parent_model.measurement_points,
                                                                       self.parent_model.measurement_vectors,
                                                                       self.rendered_alignment, visible))
        if self.active_scene is self.sample_scene:
            self.drawScene()

    def drawScene(self):
           self.parent.gl_widget.loadScene(self.active_scene)

    def drawPlane(self, plane=None, width=None, height=None, shift_by=None):
        if shift_by is not None and Attributes.Plane in self.sample_scene:
            self.sample_scene[Attributes.Plane].translate(shift_by)
        elif plane is not None and width is not None and height is not None:
            self.sample_scene.addNode(Attributes.Plane, createPlaneNode(plane, width, height))
        self.drawScene()

    def removePlane(self):
        self.sample_scene.removeNode(Attributes.Plane)
        self.drawScene()
