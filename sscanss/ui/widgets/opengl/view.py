import numpy as np
from OpenGL import GL
from PyQt5 import QtCore, QtWidgets
from pyrr import Vector3, Vector4
from sscanss.core.util import Camera, Colour, RenderType, SceneType

SAMPLE_KEY = 'sample'


class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.camera = Camera(self.width()/self.height(), 60)

        self.scene = {}
        self.scene_type = SceneType.Sample
        self.bounding_box = {'min': 0.0, 'max': 0.0, 'radius': 0.0, 'center':  0.0}

        self.render_colour = Colour.black()
        self.render_type = RenderType.Solid

        self.parent_model.sample_changed.connect(self.loadScene)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def initializeGL(self):
        GL.glClearColor(*Colour.white())

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glColorMaterial(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glShadeModel(GL.GL_SMOOTH)

        self.initLights()

    def initLights(self):
        # set up light colour
        ambient = Vector4([0.0, 0.0, 0.0, 1.0])
        diffuse = Vector4([1.0, 1.0, 1.0, 1.0])
        specular = Vector4([1.0, 1.0, 1.0, 1.0])

        # set up light direction
        front = Vector4([0.0, 0.0, 1.0, 0.0])
        back = Vector4([0.0, 0.0, -1.0, 0.0])
        left = Vector4([-1.0, 0.0, 0.0, 0.0])
        right = Vector4([1.0, 0.0, 0.0, 0.0])
        top = Vector4([0.0, 1.0, 0.0, 0.0])
        bottom = Vector4([0.0, -1.0, 0.0, 0.0])

        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, front)

        GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, back)

        GL.glLightfv(GL.GL_LIGHT2, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_POSITION, left)

        GL.glLightfv(GL.GL_LIGHT3, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_POSITION, right)

        GL.glLightfv(GL.GL_LIGHT4, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT4, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT4, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT4, GL.GL_POSITION, top)

        GL.glLightfv(GL.GL_LIGHT5, GL.GL_AMBIENT, ambient)
        GL.glLightfv(GL.GL_LIGHT5, GL.GL_DIFFUSE, diffuse)
        GL.glLightfv(GL.GL_LIGHT5, GL.GL_SPECULAR, specular)
        GL.glLightfv(GL.GL_LIGHT5, GL.GL_POSITION, bottom)

        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHT1)
        GL.glEnable(GL.GL_LIGHT2)
        GL.glEnable(GL.GL_LIGHT3)
        GL.glEnable(GL.GL_LIGHT4)
        GL.glEnable(GL.GL_LIGHT5)
        GL.glEnable(GL.GL_LIGHTING)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        self.camera.aspect = width / height
        GL.glLoadMatrixf(self.camera.perspective.transpose())

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(self.camera.perspective.transpose())

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(self.camera.model_view.transpose())

        for _, node in self.scene.items():
            self.recursive_draw(node)

    def recursive_draw(self, node):

        GL.glPushMatrix()
        GL.glMultMatrixf(node.transform.transpose())
        if node.colour is not None:
            self.render_colour = node.colour

        if node.render_type is not None:
            self.render_type = node.render_type

        GL.glColor4f(*self.render_colour.rgbaf())

        if self.render_type == RenderType.Solid:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        elif self.render_type == RenderType.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_ONE_MINUS_SRC_COLOR)
            inverted_colour = self.render_colour.invert()
            GL.glColor4f(*inverted_colour.rgbaf())

        if node.vertices.size != 0 and node.indices.size != 0:
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointerf(node.vertices)
            if node.normals.size != 0:
                GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
                GL.glNormalPointerf(node.normals)

            GL.glDrawElementsui(GL.GL_TRIANGLES, node.indices)

            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        # reset OpenGL State
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        for child in node.children:
            self.recursive_draw(child)

        GL.glPopMatrix()

    def boundingBox(self):
        max_pos = [np.nan, np.nan, np.nan]
        min_pos = [np.nan, np.nan, np.nan]
        for child in self.scene[SAMPLE_KEY].children:
            max_pos = np.fmax(max_pos, np.max(child.vertices, axis=0))
            min_pos = np.fmin(min_pos, np.min(child.vertices, axis=0))
        self.bounding_box['max'] = Vector3(max_pos)
        self.bounding_box['min'] = Vector3(min_pos)
        self.bounding_box['center'] = Vector3(self.bounding_box['max'] + self.bounding_box['min']) / 2
        self.bounding_box['radius'] = np.linalg.norm(self.bounding_box['max'] - self.bounding_box['min']) / 2

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        translation_speed = 0.001

        if event.buttons() & QtCore.Qt.LeftButton:
            p1 = (self.lastPos.x() / self.width() * 2, self.lastPos.y() / self.height() * 2)
            p2 = (event.x() / self.width() * 2, event.y() / self.height() * 2)
            self.camera.rotate(p1, p2)

        elif event.buttons() & QtCore.Qt.RightButton:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            x_offset = -dx * translation_speed
            y_offset = -dy * translation_speed
            self.camera.pan(x_offset, y_offset)

        self.lastPos = event.pos()
        self.update()

    def wheelEvent(self, event):
        zoom_scale = 0.05
        delta = 0.0
        num_degrees = event.angleDelta() / 8
        if not num_degrees.isNull():
            delta = num_degrees.y() / 15

        self.camera.zoom(delta * zoom_scale)
        self.update()

    @property
    def sampleRenderType(self):
        if SAMPLE_KEY in self.scene:
            return self.scene[SAMPLE_KEY].render_type
        else:
            return RenderType.Solid

    @sampleRenderType.setter
    def sampleRenderType(self, render_type):
        if SAMPLE_KEY in self.scene:
            self.scene[SAMPLE_KEY].render_type = render_type
            self.update()

    def loadScene(self):
        if self.scene_type == SceneType.Sample:
            self.scene = self.parent_model.sampleScene

        self.boundingBox()
        self.camera.zoomToFit(self.bounding_box['center'], self.bounding_box['radius'])
        self.update()
