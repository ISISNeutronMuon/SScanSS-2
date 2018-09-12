from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets
from sscanss.core.math import Vector4
from sscanss.core.util import Camera, Colour, RenderMode, RenderPrimitive, SceneType, world_to_screen

SAMPLE_KEY = 'sample'


class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent_model = parent.presenter.model

        self.camera = Camera(self.width()/self.height(), 60)

        self.scene = {}
        self.scene_type = SceneType.Sample

        self.render_colour = Colour.black()
        self.render_mode = RenderMode.Solid

        self.default_font = QtGui.QFont("Times", 10)

        self.parent_model.scene_updated.connect(self.loadScene)
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
        ambient = [0.0, 0.0, 0.0, 1.0]
        diffuse = [0.5, 0.5, 0.5, 1.0]
        specular = [0.2, 0.2, 0.2, 1.0]

        # set up light direction
        front = [0.0, 0.0, 1.0, 0.0]
        back = [0.0, 0.0, -1.0, 0.0]
        left = [-1.0, 0.0, 0.0, 0.0]
        right = [1.0, 0.0, 0.0, 0.0]
        top = [0.0, 1.0, 0.0, 0.0]
        bottom = [0.0, -1.0, 0.0, 0.0]

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

        for _, node in list(self.scene.items()):
            self.recursive_draw(node)

    def recursive_draw(self, node):

        GL.glPushMatrix()
        GL.glMultMatrixf(node.transform.transpose())
        if node.colour is not None:
            self.render_colour = node.colour

        if node.render_mode is not None:
            self.render_mode = node.render_mode

        GL.glColor4f(*self.render_colour.rgbaf)

        if self.render_mode == RenderMode.Solid:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        elif self.render_mode == RenderMode.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ZERO, GL.GL_ONE_MINUS_SRC_COLOR)
            inverted_colour = self.render_colour.invert()
            GL.glColor4f(*inverted_colour.rgbaf)

        if node.vertices.size != 0 and node.indices.size != 0:
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointerf(node.vertices)
            if node.normals.size != 0:
                GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
                GL.glNormalPointerf(node.normals)

            render_primitive = GL.GL_TRIANGLES if node.render_primitive == RenderPrimitive.Triangles else GL.GL_LINES
            GL.glDrawElementsui(render_primitive, node.indices)

            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        # reset OpenGL State
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

        for child in node.children:
            self.recursive_draw(child)

        GL.glPopMatrix()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        translation_speed = 0.001

        if event.buttons() == QtCore.Qt.LeftButton:
            p1 = (self.lastPos.x() / self.width() * 2, self.lastPos.y() / self.height() * 2)
            p2 = (event.x() / self.width() * 2, event.y() / self.height() * 2)
            self.camera.rotate(p1, p2)

        elif event.buttons() == QtCore.Qt.RightButton:
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
    def sampleRenderMode(self):
        if SAMPLE_KEY in self.scene:
            return self.scene[SAMPLE_KEY].render_mode
        else:
            return RenderMode.Solid

    @sampleRenderMode.setter
    def sampleRenderMode(self, render_mode):
        if SAMPLE_KEY in self.scene:
            self.scene[SAMPLE_KEY].render_mode = render_mode
            self.update()

    def loadScene(self):
        if self.scene_type == SceneType.Sample:
            self.scene = self.parent_model.sample_scene

        if SAMPLE_KEY in self.scene:
            bounding_box = self.scene[SAMPLE_KEY].bounding_box
            if bounding_box:
                self.camera.zoomToFit(bounding_box.center, bounding_box.radius)
            else:
                self.camera.reset()
        self.update()

    def project(self, x, y, z):
        world_point = Vector4([x, y, z, 1])
        model_view = self.camera.model_view
        perspective = self.camera.perspective

        return world_to_screen(world_point, model_view, perspective, self.width(), self.height())

    def renderText(self, x, y, z, text, font=QtGui.QFont(), font_colour=QtGui.QColor(0,0,0)):
        text_pos, ok = self.project(x, y, z)
        if not ok:
            return

        # Render text
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        painter = QtGui.QPainter(self)
        painter.setPen(font_colour)
        painter.setFont(font)
        painter.drawText(text_pos[0], self.height() - text_pos[1], text)
        painter.end()
        GL.glPopAttrib()
