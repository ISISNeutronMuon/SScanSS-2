import logging
import numpy as np
from OpenGL import GL, error
from PyQt5 import QtCore, QtGui, QtWidgets
from .camera import Camera, world_to_screen, screen_to_world
from .node import Node, InstanceRenderNode, BatchRenderNode, TextNode
from .scene import Scene
from .shader import DefaultShader, GouraudShader, VolumeShader, TextShader
from ..geometry.colour import Colour
from ..math.matrix import Matrix44
from ..math.vector import Vector3
from ..util.misc import Attributes, MessageType
from ...config import settings


class OpenGLRenderer(QtWidgets.QOpenGLWidget):
    """Provides OpenGL widget for draw 3D scene for the sample setup and instrument

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.scene = Scene()
        self.interactor = SceneInteractor(self)
        self.show_bounding_box = False
        self.show_coordinate_frame = True
        self.picks = []
        self.default_font = QtGui.QFont("Times", 10)
        self.error = False
        self.custom_error_handler = None
        self.shader_programs = {}
        self.gpu_info = {'vendor': '', 'version': '', 'pbo_support': False}

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def cleanup(self):
        self.makeCurrent()
        del self.scene
        for program in self.shader_programs.values():
            program.destroy()
        self.doneCurrent()

    def initializeGL(self):
        try:
            GL.glClearColor(*Colour.white())
            GL.glColor4f(*Colour.black())

            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glEnable(GL.GL_MULTISAMPLE)

            number_of_lights = self.initLights()
            # Create and compile GLSL shaders program
            self.shader_programs['mesh'] = GouraudShader(number_of_lights)
            self.shader_programs['default'] = DefaultShader()
            self.shader_programs['text'] = TextShader()
            self.shader_programs['volume'] = VolumeShader()

            self.context().aboutToBeDestroyed.connect(self.cleanup)

            self.gpu_info['vendor'] = GL.glGetString(GL.GL_RENDERER).decode()
            self.gpu_info['version'] = GL.glGetString(GL.GL_VERSION).decode()
            extensions = GL.glGetString(GL.GL_EXTENSIONS).split()
            self.gpu_info['pbo_support'] = b'GL_ARB_pixel_buffer_object' in extensions
            logging.info(f"GPU: {self.gpu_info['vendor']}, Driver Version: {self.gpu_info['version']}, PBO Support: "
                         f"{self.gpu_info['pbo_support']}")
        except error.GLError:
            self.parent.showMessage('An error occurred during OpenGL initialization. '
                                    'The minimum OpenGL requirement for this software is version 2.0.\n\n'
                                    'This error may be caused by:\n'
                                    '* A missing or faulty graphics driver installation.\n'
                                    '* Accessing SScanSS 2 from a remote connection with GPU rendering disabled.\n\n'
                                    'The software will be closed now.')
            raise

    def reportError(self, exception):
        message = ("This device has insufficient memory for rendering the volume. "
                   "A simple mesh representation will be rendered instead.\n\n"
                   f"The active graphic card is the {self.gpu_info['vendor']} {self.gpu_info['version']}")
        self.parent.showMessage(message, MessageType.Error)
        logging.error(message, exc_info=exception)

    def initLights(self):
        """Sets up light properties"""
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
        directions = {
            GL.GL_LIGHT0: front,
            GL.GL_LIGHT1: back,
            GL.GL_LIGHT2: left,
            GL.GL_LIGHT3: right,
            GL.GL_LIGHT4: top,
            GL.GL_LIGHT5: bottom
        }

        for light, direction in directions.items():
            GL.glLightfv(light, GL.GL_AMBIENT, ambient)
            GL.glLightfv(light, GL.GL_DIFFUSE, diffuse)
            GL.glLightfv(light, GL.GL_SPECULAR, specular)
            GL.glLightfv(light, GL.GL_POSITION, direction)

            GL.glEnable(light)

        GL.glEnable(GL.GL_LIGHTING)

        return len(directions)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        self.scene.camera.aspect = width / height
        GL.glLoadTransposeMatrixf(self.scene.camera.projection)

    def paintGL(self):
        if self.scene.invalid:
            if self.error:
                return

            self.error = True

            if self.custom_error_handler is not None:
                self.custom_error_handler()
                self.scene.camera.reset()
            return

        self.error = False

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadTransposeMatrixf(self.scene.camera.projection)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadTransposeMatrixf(self.scene.camera.model_view)

        for node in self.scene.nodes:
            node.draw(self)

        if self.show_bounding_box:
            self.renderBoundingBox()

        if self.picks:
            self.renderPicks()

        # draw last due to blending
        if self.show_coordinate_frame:
            self.renderAxis()

    def renderPicks(self):
        """Renders picked points in the scene"""
        size = settings.value(settings.Key.Measurement_Size)

        node = InstanceRenderNode(len(self.picks))
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines

        vertices = np.array(
            [[-size, 0., 0.], [size, 0., 0.], [0., -size, 0.], [0., size, 0.], [0., 0., -size], [0., 0., size]],
            dtype=np.float32)
        indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        node.vertices = vertices
        node.indices = indices
        for index, pick in enumerate(self.picks):
            point, selected = pick
            node.selected[index] = selected
            node.per_object_colour[index] = Colour(0.9, 0.4, 0.4)
            node.per_object_transform[index] = Matrix44.fromTranslation(point)

        node.buildVertexBuffer()
        node.draw(self)

    def showCoordinateFrame(self, state):
        """Sets visibility of the coordinate frame in the widget

        :param state: indicates if the coordinate frame should be visible
        :type state: bool
        """
        self.show_coordinate_frame = state
        self.update()

    def showBoundingBox(self, state):
        """Sets visibility of the sample bounding box frame in the widget

        :param state: indicates if the bounding box should be visible
        :type state: bool
        """
        self.show_bounding_box = state
        self.update()

    def loadScene(self, scene, zoom_to_fit=True):
        """Loads a scene into the widget and adjust the camera

        :param scene: sample or instrument scene
        :type scene: Scene
        :param zoom_to_fit: indicates that the scene should be zoomed to fit window
        :type zoom_to_fit: bool
        """
        self.scene = scene
        self.scene.camera.aspect = self.width() / self.height()

        if not self.scene.isEmpty():
            bounding_box = self.scene.bounding_box
            if zoom_to_fit:
                self.scene.camera.zoomToFit(bounding_box.center, bounding_box.radius)
            else:
                self.scene.camera.updateView(bounding_box.center, bounding_box.radius)

        self.update()

    def unproject(self, x, y, z):
        """Converts point in screen coordinate to point in world coordinate

        :param x: x coordinate
        :type x: float
        :param y: y coordinate
        :type y: float
        :param z: z coordinate
        :type z: float
        :return: point in screen coordinates and flag indicating the new point is valid
        :rtype: Tuple[Vector3, bool]
        """
        y = self.height() - y  # invert y to match screen coordinate
        screen_point = Vector3([x, y, z])
        model_view = self.scene.camera.model_view
        projection = self.scene.camera.projection

        world_point, valid = screen_to_world(screen_point, model_view, projection, self.width(), self.height())
        return world_point, valid

    def project(self, x, y, z):
        """Converts point in world coordinate to point in screen coordinate

        :param x: x coordinate
        :type x: float
        :param y: y coordinate
        :type y: float
        :param z: z coordinate
        :type z: float
        :return: point in screen coordinates and flag indicating the new point is valid
        :rtype: Tuple[Vector3, bool]
        """
        world_point = Vector3([x, y, z])
        model_view = self.scene.camera.model_view
        projection = self.scene.camera.projection

        screen_point, valid = world_to_screen(world_point, model_view, projection, self.width(), self.height())
        screen_point.y = self.height() - screen_point.y  # invert y to match screen coordinate
        return screen_point, valid

    def renderBoundingBox(self):
        """Draws the axis aligned bounding box of the sample"""
        if Attributes.Sample not in self.scene:
            return

        bounding_box = self.scene[Attributes.Sample].bounding_box
        max_x, max_y, max_z = bounding_box.max
        min_x, min_y, min_z = bounding_box.min

        node = Node()
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines

        node.vertices = np.array(
            [[min_x, min_y, min_z], [min_x, max_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z],
             [min_x, min_y, max_z], [min_x, max_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z]],
            dtype=np.float32)
        node.indices = np.array([0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7],
                                dtype=np.uint32)
        node.colour = Colour(0.9, 0.4, 0.4)
        node.buildVertexBuffer()
        node.draw(self)

    def renderAxis(self):
        """Draws the X, Y and Z axis lines and centre point"""
        if self.scene.isEmpty():
            return

        scale = self.scene.bounding_box.radius * 0.97

        node = BatchRenderNode(3)
        node.render_mode = Node.RenderMode.Solid
        node.render_primitive = Node.RenderPrimitive.Lines
        node.vertices = np.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, scale, 0.0],
                                  [0.0, 0.0, 0.0], [0.0, 0.0, scale]],
                                 dtype=np.float32)

        node.indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        node.per_object_colour = [Colour(1.0, 0.0, 0.0), Colour(0.0, 1.0, 0.0), Colour(0.0, 0.0, 1.0)]
        node.batch_offsets = [2, 4, 6]
        node.buildVertexBuffer()
        node.draw(self)

        axes = [((1, 0, 0), 'X'), ((0, 1, 0), 'Y'), ((0, 0, 1), 'Z')]
        for axis, text in axes:
            text_pos = np.array(axis) * scale * 1.02
            text_node = TextNode(text, text_pos, QtGui.QColor.fromRgbF(*axis), self.default_font)
            text_node.buildVertexBuffer()
            text_node.draw(self)

    def viewFrom(self, direction):
        """Changes view direction of scene camera

        :param direction: direction to view from
        :type direction: Direction
        """
        self.scene.camera.mode = Camera.Projection.Orthographic
        self.scene.camera.viewFrom(direction)
        self.update()

    def resetCamera(self):
        """Resets scene camera"""
        self.scene.camera.reset()
        self.update()


class SceneInteractor(QtCore.QObject):
    """A tool class that provides the camera manipulations like panning, zooming and orbiting, and
    point picking for the OpenGL renderer

    :param renderer: renderer instance
    :type renderer: OpenGLRenderer
    """
    ray_picked = QtCore.pyqtSignal(object, object)

    def __init__(self, renderer):
        super().__init__()

        self.renderer = renderer
        renderer.installEventFilter(self)

        self._picking = False
        self.last_pos = QtCore.QPointF()

    @property
    def picking(self):
        return self._picking

    @picking.setter
    def picking(self, value):
        """Enables/Disables point picking

        :param value: indicates if point picking is enabled
        :type value: bool
        """
        self._picking = value
        if value:
            self.renderer.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.renderer.setCursor(QtCore.Qt.ArrowCursor)

    @property
    def camera(self):
        """Gets active camera from the scene"""
        return self.renderer.scene.camera

    def __isLeftButtonPressed(self, event):
        """Checks if the left mouse button is pressed

         :param event: mouse event
         :type event: QtGui.QMouseEvent
         :return: indicate the left button is pressed
         :rtype: bool
         """
        return ((event.button() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.LeftButton)
                and event.modifiers() == QtCore.Qt.NoModifier)

    def isRotating(self, event):
        """Checks if the selected mouse button and keybind is for rotation

        :param event: mouse event
        :type event: QtGui.QMouseEvent
        :return: indicate the event is for rotation
        :rtype: bool
        """
        return self.__isLeftButtonPressed(event) and not self.picking

    def isPicking(self, event):
        """Checks if the selected mouse button and keybind is for picking

        :param event: mouse event
        :type event: QtGui.QMouseEvent
        :return: indicate the event is for picking
        :rtype: bool
        """
        return self.__isLeftButtonPressed(event) and self.picking

    def isPanning(self, event):
        """Checks if the selected mouse button and keybind is for panning

        :param event: mouse event
        :type event: QtGui.QMouseEvent
        :return: indicate the event is for panning
        :rtype: bool
        """
        return ((event.button() == QtCore.Qt.RightButton or event.buttons() == QtCore.Qt.RightButton)
                and event.modifiers() == QtCore.Qt.NoModifier)

    def createPickRay(self, pos):
        """Creates the start and stop position for the ray used for point picking

        :param pos: mouse position
        :type pos: QtCore.QPointF
        """
        start_pos, valid_1 = self.renderer.unproject(pos.x(), pos.y(), 0.0)
        stop_pos, valid_2 = self.renderer.unproject(pos.x(), pos.y(), 1.0)

        if not valid_1 or not valid_2:
            return

        self.ray_picked.emit(start_pos, stop_pos)

    def rotate(self, pos, viewport_size):
        """Rotates camera based on mouse movements

        :param pos: mouse position
        :type pos: QtCore.QPointF
        :param viewport_size: viewport dimension i.e. width, height
        :type viewport_size: Tuple[int, int]
        """
        width, height = viewport_size
        p1 = (self.last_pos.x() / width * 2, self.last_pos.y() / height * 2)
        p2 = (pos.x() / width * 2, pos.y() / height * 2)
        self.camera.rotate(p1, p2)
        self.renderer.update()

    def pan(self, pos):
        """Pans camera based on mouse movements

        :param pos: mouse position
        :type pos: QtCore.QPointF
        """
        translation_speed = 0.001
        dx = pos.x() - self.last_pos.x()
        dy = pos.y() - self.last_pos.y()
        x_offset = -dx * translation_speed
        y_offset = -dy * translation_speed
        self.camera.pan(x_offset, y_offset)
        self.renderer.update()

    def zoom(self, angle_delta):
        """Zooms camera based on mouse movements

        :param angle_delta: relative amount that the wheel was rotated
        :type angle_delta: QtCore.QPointF
        """
        zoom_scale = 0.05
        delta = 0.0
        num_degrees = angle_delta / 8
        if not num_degrees.isNull():
            delta = num_degrees.y() / 15
        self.camera.zoom(delta * zoom_scale)
        self.renderer.update()

    def eventFilter(self, obj, event):
        """Intercepts the mouse events and computes anchor snapping based on mouse movements

        :param obj: widget
        :type obj: QtWidgets.QWidget
        :param event: Qt events
        :type event: QtCore.QEvent
        :return: indicates if event was handled
        :rtype: bool
        """
        if event.type() == QtCore.QEvent.MouseButtonPress:
            if self.isRotating(event) or self.isPanning(event):
                self.last_pos = event.pos()
            elif self.isPicking(event):
                self.createPickRay(event.pos())
            else:
                return False
            return True
        if event.type() == QtCore.QEvent.MouseMove:
            if self.isRotating(event):
                self.camera.mode = Camera.Projection.Perspective
                self.rotate(event.pos(), (self.renderer.width(), self.renderer.height()))
            elif self.isPanning(event):
                self.pan(event.pos())
            else:
                return False

            self.last_pos = event.pos()
            return True

        if (event.type() == QtCore.QEvent.Wheel and event.buttons() == QtCore.Qt.NoButton
                and event.modifiers() == QtCore.Qt.NoModifier):
            self.zoom(event.angleDelta())
            return True

        return False
