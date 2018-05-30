import numpy as np
from OpenGL import GL
from PyQt5 import QtCore, QtWidgets
from pyrr import Vector3, Vector4
from sscanss.core.util import Camera, Colour, RenderType

SAMPLE_KEY = 'sample'


class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)

        self.camera = Camera(self.width(), self.height(), 60)
        # self.camera.setDistance(5)
        # self.camera.setRotation(Vector3([45.0, -45.0, 0.0]))

        self.scene = {}
        self.bounding_box = {'min': 0.0, 'max': 0.0, 'radius': 0.0, 'center':  0.0}

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
        front = Vector4([1.0, 0.0, 0.0, 0.0])
        back = Vector4([-1.0, 0.0, 0.0, 0.0])
        left = Vector4([0.0, 1.0, 0.0, 0.0])
        right = Vector4([0.0, -1.0, 0.0, 0.0])
        top = Vector4([0.0, 0.0, 1.0, 0.0])
        bottom = Vector4([0.0, 0.0, -1.0, 0.0])

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
        self.camera.aspect = width / height
        GL.glViewport(0, 0, width, height)
        self.camera.setPerspective()

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_MODELVIEW)

        GL.glLoadMatrixf(self.camera.matrix.transpose())

        for _, node in self.scene.items():
            self.recursive_draw(node)

    def recursive_draw(self, node):

        GL.glPushMatrix()
        GL.glMultMatrixf(node.transform.transpose())

        GL.glColor4f(*node.colour.rgbaf())

        if node.render_type == RenderType.Solid:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        elif node.render_type == RenderType.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_ONE_MINUS_SRC_COLOR)
            inverted_colour = node.colour.invert()
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
        temp = self.scene[SAMPLE_KEY].vertices
        self.bounding_box.max = Vector3(np.max(temp, axis=0))
        self.bounding_box.min = Vector3(np.min(temp, axis=0))
        self.bounding_box.center = Vector3(self.bounding_box.max + self.bounding_box.min) / 2
        self.bounding_box.radius = np.linalg.norm(self.bounding_box.max - self.bounding_box.min) / 2

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        speed = 0.2
        speedt = 0.001
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if event.buttons() & QtCore.Qt.LeftButton:
            self.camera.rotate(Vector3([dy * speed, dx * speed, 0.0]))

        elif event.buttons() & QtCore.Qt.RightButton:
            distance = self.camera.distance if self.camera.distance != 0.0 else 0.1
            x_offset = -dx * speedt * distance
            y_offset = -dy * speedt * distance
            self.camera.pan(Vector3([x_offset, y_offset, 0.0]))

        self.lastPos = event.pos()
        self.update()

    def wheelEvent(self, event):

        zoom_scale = 0.05
        delta = 0.0
        numDegrees = event.angleDelta() / 8
        if not numDegrees.isNull():
            delta = numDegrees.y() / 15

        distance = self.camera.distance if self.camera.distance != 0.0 else 0.1
        distance -= (delta * zoom_scale * distance)
        self.camera.zoom(distance)
        self.update()