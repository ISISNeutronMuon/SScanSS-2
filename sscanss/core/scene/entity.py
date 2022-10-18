import numpy as np
from .node import Node, BatchRenderNode, InstanceRenderNode, VolumeNode
from ..geometry.colour import Colour
from ..geometry.mesh import Mesh
from ..geometry.primitive import create_sphere, create_plane, create_cuboid
from ..geometry.volume import Volume
from ..math.matrix import Matrix44
from ..math.transform import rotation_btw_vectors
from ..util import Attributes
from ...config import settings


class Entity:
    """Base class for rendered entities"""
    def __init__(self):
        self.visible = True

    def node(self):
        """Returns scene node of entity"""


class SampleEntity(Entity):
    """Creates entity for sample

    :param sample: sample mesh or volume
    :type sample: Optional[Union[Mesh, Volume]]
    """
    def __init__(self, sample):
        super().__init__()

        self._sample = sample

    def node(self, render_mode=Node.RenderMode.Solid):
        """Creates scene node for samples

        :param render_mode: render mode
        :type render_mode: Node.RenderMode
        :return: node containing sample
        :rtype: Union[Node, VolumeNode]
        """
        sample_node = Node()

        if isinstance(self._sample, Mesh):
            sample_node = Node(self._sample)
            sample_node.colour = Colour(*settings.value(settings.Key.Sample_Colour))
        elif isinstance(self._sample, Volume):
            sample_node = VolumeNode(self._sample)

        sample_node.render_mode = render_mode
        sample_node.buildVertexBuffer()
        return sample_node


class FiducialEntity(Entity):
    """Creates entity for fiducial points

    :param fiducials: fiducial points
    :type fiducials: numpy.recarray
    :param visible: indicates node is visible
    :type visible: bool
    """
    def __init__(self, fiducials, visible=True):
        super().__init__()
        self.visible = visible
        self.colours = []
        self.transforms = []
        size = settings.value(settings.Key.Fiducial_Size)

        enabled_colour = Colour(*settings.value(settings.Key.Fiducial_Colour))
        disabled_colour = Colour(*settings.value(settings.Key.Fiducial_Disabled_Colour))
        for point, enabled in fiducials:
            colour = enabled_colour if enabled else disabled_colour
            self.transforms.append(Matrix44.fromTranslation(point))
            self.colours.append(colour)

        fiducial_mesh = create_sphere(size, 32, 32)
        self.vertices = fiducial_mesh.vertices
        self.indices = fiducial_mesh.indices
        self.normals = fiducial_mesh.normals

    def node(self):
        """Creates scene node for fiducial points

        :return: node containing fiducial points
        :rtype: Node
        """
        fiducial_node = InstanceRenderNode(len(self.transforms))
        fiducial_node.visible = self.visible
        fiducial_node.render_mode = Node.RenderMode.Solid

        if len(self.transforms) == 0:
            return fiducial_node

        fiducial_node.vertices = self.vertices
        fiducial_node.indices = self.indices
        fiducial_node.normals = self.normals
        fiducial_node.per_object_colour = self.colours
        fiducial_node.per_object_transform = self.transforms
        fiducial_node.buildVertexBuffer()

        return fiducial_node


class MeasurementPointEntity(Entity):
    """Creates entity for measurement points

    :param points: measurement points
    :type points: numpy.recarray
    :param visible: indicates node is visible
    :type visible: bool
    """
    def __init__(self, points, visible=True):

        super().__init__()
        self.visible = visible
        self.colours = []
        self.transforms = []
        enabled_colour = Colour(*settings.value(settings.Key.Measurement_Colour))
        disabled_colour = Colour(*settings.value(settings.Key.Measurement_Disabled_Colour))
        size = settings.value(settings.Key.Measurement_Size)
        for point, enabled in points:
            colour = enabled_colour if enabled else disabled_colour
            self.transforms.append(Matrix44.fromTranslation(point))
            self.colours.append(colour)

        self.vertices = np.array(
            [[-size, 0., 0.], [size, 0., 0.], [0., -size, 0.], [0., size, 0.], [0., 0., -size], [0., 0., size]],
            dtype=np.float32)
        self.indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)

    def node(self):
        """Creates scene node for measurement points

        :return: node containing measurement points
        :rtype: Node
        """
        measurement_point_node = InstanceRenderNode(len(self.transforms))
        measurement_point_node.visible = self.visible
        measurement_point_node.render_mode = Node.RenderMode.Solid
        measurement_point_node.render_primitive = Node.RenderPrimitive.Lines

        if len(self.transforms) == 0:
            return measurement_point_node

        measurement_point_node.vertices = self.vertices
        measurement_point_node.indices = self.indices
        measurement_point_node.per_object_colour = self.colours
        measurement_point_node.per_object_transform = self.transforms
        measurement_point_node.buildVertexBuffer()

        return measurement_point_node


class MeasurementVectorEntity(Entity):
    """Creates entity for measurement vectors

    :param points: measurement points
    :type points: numpy.recarray
    :param vectors: measurement vectors
    :type vectors: numpy.ndarray
    :param alignment: vector alignment
    :type alignment: int
    :param visible: indicates node is visible
    :type visible: bool
    """
    def __init__(self, points, vectors, alignment, visible=True):

        super().__init__()
        self.visible = visible
        self.vertices = []

        size = settings.value(settings.Key.Vector_Size)
        colours = [
            Colour(*settings.value(settings.Key.Vector_1_Colour)),
            Colour(*settings.value(settings.Key.Vector_2_Colour))
        ]

        if len(vectors) == 0:
            return

        self.alignment = 0 if alignment >= vectors.shape[2] else alignment

        for k in range(vectors.shape[2]):
            start_point = points.points

            vertices = []
            colour = []
            ranges = []
            count = 0
            for j in range(0, vectors.shape[1] // 3):
                end_point = start_point + size * vectors[:, j * 3:j * 3 + 3, k]

                vertices.append(np.column_stack((start_point, end_point)).reshape(-1, 3))
                if j < 2:
                    colour.append(colours[j])
                else:
                    np.random.seed(j)
                    colour.append(Colour(*np.random.random(3)))
                count += len(vertices[-1])
                ranges.append(count)
            self.colours = colour
            self.offsets = ranges
            self.vertices.append(np.array(np.row_stack(vertices), dtype=np.float32))
        self.indices = np.arange(0, len(self.vertices[-1]), dtype=np.uint32)

    def node(self):
        """Creates scene node for measurement vectors

        :return: node containing measurement vectors
        :rtype: Node
        """
        measurement_vector_node = Node()
        measurement_vector_node.visible = self.visible
        measurement_vector_node.render_mode = Node.RenderMode.Solid

        if not self.vertices:
            return measurement_vector_node

        for index, vertices in enumerate(self.vertices):
            child = BatchRenderNode(len(self.offsets))
            child.vertices = vertices
            child.indices = self.indices
            child.per_object_colour = self.colours
            child.render_mode = None
            child.visible = self.alignment == index
            child.render_primitive = Node.RenderPrimitive.Lines
            child.batch_offsets = self.offsets
            child.buildVertexBuffer()

            measurement_vector_node.addChild(child)

        return measurement_vector_node


class InstrumentEntity(Entity):
    """Creates entity for a given instrument.

    :param instrument: instrument
    :type instrument: Instrument
    """
    def __init__(self, instrument):
        super().__init__()

        self._count = 0
        self._index_offset = 0
        self.instrument = instrument
        self._vertices = []
        self._indices = []
        self._normals = []
        self.offsets = []
        self.colours = []
        self.transforms = []
        for mesh, transform in self.instrument.positioning_stack.model():
            self._updateParams(mesh, transform)
        self.keys = {Attributes.Positioner.value: len(self.offsets)}

        for detector in self.instrument.detectors.values():
            for mesh, transform in detector.model():
                self._updateParams(mesh, transform)
            self.keys[f'{Attributes.Detector.value}_{detector.name}'] = len(self.offsets)

        for mesh, transform in self.instrument.jaws.model():
            self._updateParams(mesh, transform)
        self.keys[Attributes.Jaws.value] = len(self.offsets)

        for name, mesh in self.instrument.fixed_hardware.items():
            self._updateParams(mesh, Matrix44.identity())
            self.keys[f'{Attributes.Fixture.value}_{name}'] = len(self.offsets)

        self.vertices = np.row_stack(self._vertices)
        self.indices = np.concatenate(self._indices)
        self.normals = np.row_stack(self._normals)

    def _updateParams(self, mesh, transform):
        self._vertices.append(mesh.vertices)
        self._indices.append(mesh.indices + self._count)
        self._normals.append(mesh.normals)
        self.transforms.append(transform)
        self.colours.append(mesh.colour)
        self._count += len(mesh.vertices)
        self._index_offset += len(mesh.indices)
        self.offsets.append(self._index_offset)

    def updateTransforms(self, node):
        """Updates transformation matrices in an instrument node

        :param: node containing model of instrument
        :type: BatchRenderNode
        """
        transforms = []
        for _, transform in self.instrument.positioning_stack.model():
            transforms.append(transform)

        for detector in self.instrument.detectors.values():
            for _, transform in detector.model():
                transforms.append(transform)

        for _, transform in self.instrument.jaws.model():
            transforms.append(transform)

        node.per_object_transform[:len(transforms)] = transforms

    def node(self):
        """Creates scene node for a given instrument.

        :return: node containing model of instrument
        :rtype: BatchRenderNode
        """
        instrument_node = BatchRenderNode(len(self.offsets))

        if len(self.vertices) == 0:
            return instrument_node

        instrument_node.vertices = self.vertices
        instrument_node.indices = self.indices
        instrument_node.per_object_colour = self.colours
        instrument_node.normals = self.normals
        instrument_node.per_object_transform = self.transforms
        instrument_node.batch_offsets = self.offsets
        instrument_node.buildVertexBuffer()

        return instrument_node


class PlaneEntity(Entity):
    """Creates entity for cross-sectional plane

    :param plane: plane normal and point
    :type plane: Plane
    :param width: plane width
    :type width: float
    :param height: plane height
    :type height: float
    """
    def __init__(self, plane, width, height):

        super().__init__()

        self.mesh = create_plane(plane, width, height)
        self.plane = plane
        self._offset = np.zeros(3)
        self.transform = Matrix44.identity()

    @property
    def offset(self):
        """Returns plane offset

        :rtype: [float, float, float]
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value
        self.transform @= Matrix44.fromTranslation(value)

    def node(self):
        """Creates scene node for cross-sectional plane

        :return: node containing plane
        :rtype: Node
        """
        node = Node(self.mesh)
        node.transform = self.transform
        node.render_mode = Node.RenderMode.Transparent
        node.colour = Colour(*settings.value(settings.Key.Cross_Sectional_Plane_Colour))
        node.buildVertexBuffer()

        return node


class BeamEntity(Entity):
    """Creates entity for beam

    :param instrument: instrument object
    :type instrument: Instrument
    :param bounds: bounding box of the instrument scene
    :type bounds: BoundingBox
    :param visible: indicates node is visible
    :type visible: bool
    """
    def __init__(self, instrument, bounds, visible=False):
        super().__init__()

        self.visible = visible
        jaws = instrument.jaws
        detectors = instrument.detectors
        q_vectors = instrument.q_vectors
        gauge_volume = instrument.gauge_volume

        width, height = jaws.aperture
        beam_source = jaws.beam_source
        beam_direction = jaws.beam_direction
        cuboid_axis = np.array([0., 1., 0.])

        bound_max = np.dot(bounds.max - beam_source, beam_direction)
        bound_min = np.dot(bounds.min - beam_source, beam_direction)
        depth = max(bound_min, bound_max)

        self.beam_mesh = create_cuboid(width, height, depth)
        m = Matrix44.fromTranslation(beam_source)
        m[0:3, 0:3] = rotation_btw_vectors(beam_direction, cuboid_axis)
        m = m @ Matrix44.fromTranslation([0., -depth / 2, 0.])
        self.beam_mesh.transform(m)

        self.q_vertices = []
        if instrument.beam_in_gauge_volume:
            for index, detector in enumerate(detectors.values()):
                if detector.current_collimator is None:
                    continue
                bound_max = np.dot(bounds.max - gauge_volume, detector.diffracted_beam)
                bound_min = np.dot(bounds.min - gauge_volume, detector.diffracted_beam)
                depth = max(bound_min, bound_max)
                sub_mesh = create_cuboid(width, height, depth)
                m = Matrix44.fromTranslation(gauge_volume)
                m[0:3, 0:3] = rotation_btw_vectors(cuboid_axis, detector.diffracted_beam)
                m = m @ Matrix44.fromTranslation([0., depth / 2, 0.])
                self.beam_mesh.append(sub_mesh.transformed(m))

                # draw q_vector
                end_point = gauge_volume + q_vectors[index] * depth / 2
                self.q_vertices.extend((gauge_volume, end_point))

    def node(self):
        """Creates scene node for beam

        :return: node containing beam
        :rtype: Node
        """
        node = Node(self.beam_mesh)
        node.render_mode = Node.RenderMode.Solid
        node.colour = Colour(0.80, 0.45, 0.45)
        node.visible = self.visible
        node.buildVertexBuffer()

        if len(self.q_vertices) > 0:
            child = Node()
            child.vertices = np.array(self.q_vertices)
            child.indices = np.arange(len(self.q_vertices))
            child.colour = Colour(0.60, 0.25, 0.25)
            child.render_primitive = Node.RenderPrimitive.Lines
            child.buildVertexBuffer()
            node.addChild(child)

        return node
