"""
Classes for Mesh and Bounding-Box objects
"""
import numpy as np
from .colour import Colour
from ..math.vector import Vector3


def compute_face_normals(vertices):
    """ Calculates the face normals by determining the edges of the face
    and finding the cross product of the edges. The function assumes that every 3
    consecutive vertices belong to the same face.

    :param vertices: N x 3 array of vertices
    :type vertices: numpy.ndarray
    :return: M x 3 array of normals. M = N // 3
    :rtype: numpy.ndarray
    """
    face_vertices = vertices.reshape(-1, 9)
    edge_1 = face_vertices[:, 0:3] - face_vertices[:, 3:6]
    edge_2 = face_vertices[:, 3:6] - face_vertices[:, 6:9]

    normals = np.cross(edge_1, edge_2)
    row_sums = np.linalg.norm(normals, axis=1)
    return normals / row_sums[:, np.newaxis]


class Mesh:
    """ Creates a Mesh object. Calculates the bounding box
    of the Mesh and calculates normals if not provided.

    :param vertices: N x 3 array of vertices
    :type vertices: numpy.ndarray
    :param indices: N X 1 array of indices
    :type indices: numpy.ndarray
    :param normals: N x 3 array of normals
    :type normals: Union[numpy.ndarray, None]
    """
    def __init__(self, vertices, indices, normals=None, colour=None):
        self.vertices = vertices
        self.indices = indices

        if normals is not None:
            self.normals = normals
        else:
            self.computeNormals()

        self.colour = Colour.black() if colour is None else Colour(*colour)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value
        self.computeBoundingBox()

    def append(self, mesh):
        """ Append a given mesh to this mesh. Indices are offset to
        ensure the correct vertices and normals are used

        :param mesh: mesh to append
        :type mesh: Mesh
        """
        count = self.vertices.shape[0]
        self.vertices = np.vstack((self.vertices, mesh.vertices))
        self.indices = np.concatenate((self.indices, mesh.indices + count))
        self.normals = np.vstack((self.normals, mesh.normals))

    def splitAt(self, index):
        """ Split this mesh into two parts using the given index. The operation
        is not the exact opposite of Mesh.append() as vertices and normals could
        be duplicated or rearranged and indices changed but the mesh will be valid.
        The first split is retained while the second is returned by the function

        :param index: index to split from
        :type index: int
        :return: split mesh
        :rtype: Mesh
        """
        temp_indices = np.split(self.indices, [index])
        temp_vertices = self.vertices[temp_indices[1], :]
        temp_normals = self.normals[temp_indices[1], :]

        self.vertices = self.vertices[temp_indices[0], :]
        self.indices = np.arange(index)
        self.normals = self.normals[temp_indices[0], :]

        return Mesh(temp_vertices, np.arange(temp_indices[1].size), temp_normals, Colour(*self.colour))

    def rotate(self, matrix):
        """ performs in-place rotation of mesh.

        :param matrix: 3 x 3 rotation matrix
        :type matrix: Union[numpy.ndarray, Matrix33]
        """
        _matrix = matrix[0:3, 0:3].transpose()
        self.vertices = self.vertices @ _matrix
        self.normals = self.normals @ _matrix

    def translate(self, offset):
        """ performs in-place translation of mesh.
        Don't use a Vector3 for offset. it causes vertices to become an
        array of Vector3's which leads to other problems

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: Union[numpy.ndarray, Vector3]
        """
        self.vertices = self.vertices + offset

    def transform(self, matrix):
        """ performs in-place transformation of mesh

        :param matrix: 4 x 4 transformation matrix
        :type matrix: numpy.ndarray
        """
        mesh = self.transformed(matrix)
        self._vertices = mesh.vertices
        self.normals = mesh.normals
        self.bounding_box = mesh.bounding_box

    def transformed(self, matrix):
        """ performs a transformation of mesh

        :param matrix: 4 x 4 transformation matrix
        :type matrix: numpy.ndarray
        """
        _matrix = matrix[0:3, 0:3].transpose()
        offset = matrix[0:3, 3].transpose()

        vertices = self.vertices @ _matrix + offset
        normals = self.normals @ _matrix

        return Mesh(vertices, np.copy(self.indices), normals, Colour(*self.colour))

    def computeBoundingBox(self):
        """ Calculates the axis aligned bounding box of the mesh """
        self.bounding_box = BoundingBox.fromPoints(self.vertices)

    def computeNormals(self):
        """ Computes normals fo the mesh """
        vertices = self.vertices[self.indices]
        face_normals = compute_face_normals(vertices)
        self.normals = np.repeat(face_normals, 3, axis=0)

    def copy(self):
        """ Deep copies the mesh

        :return: deep copy of the mesh
        :rtype: Mesh
        """
        vertices = np.copy(self.vertices)
        indices = np.copy(self.indices)
        normals = np.copy(self.normals)

        return Mesh(vertices, indices, normals, Colour(*self.colour))


class BoundingBox:
    """Creates an Axis Aligned Bounding box

    :param max_position: maximum position
    :type max_position: Union[numpy.ndarray, Vector3]
    :param min_position: minimum position
    :type min_position: Union[numpy.ndarray, Vector3]
    """
    def __init__(self, max_position, min_position):
        self.max = Vector3(max_position)
        self.min = Vector3(min_position)
        self.center = (self.max + self.min) / 2
        self.radius = np.linalg.norm(self.max - self.min) / 2

    @classmethod
    def fromPoints(cls, points):
        """compute the bounding box for an array of points

        :param points: N x 3 array of point
        :type points: numpy.ndarray
        :return: bounding box
        :rtype: BoundingBox
        """
        max_pos = np.max(points, axis=0)
        min_pos = np.min(points, axis=0)
        return cls(max_pos, min_pos)

    @property
    def bounds(self):
        """property that returns max and min bounds of box (in that order)

        :return: max and min bounds of box
        :rtype: Tuple[float, float]
        """
        return self.max, self.min

    def translate(self, offset):
        """ Performs in-place translation of bounding box by
        given offset

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: Union[numpy.ndarray, Vector3]
        """
        self.max += offset
        self.min += offset
        self.center += offset
