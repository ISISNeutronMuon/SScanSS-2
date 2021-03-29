"""
Classes for Mesh and Bounding-Box objects
"""
import numpy as np
from .colour import Colour
from ..math.constants import VECTOR_EPS
from ..math.vector import Vector3


def compute_face_normals(vertices, remove_degenerate=False):
    """Calculates the face normals by determining the edges of the face and finding the
    cross product of the edges. The function expects vertices to be a N x 3 array where
    consecutive vertices belong to the same face or a N x 9 array where each row contains
    vertices of a face. The function can also remove degenerate (zero area) faces.

    :param vertices: array of vertices
    :type vertices: numpy.ndarray
    :param remove_degenerate: flag that specifies degenerate faces be removed
    :type remove_degenerate: bool
    :return: array of normals or array of vertices and normals when remove_degenerate is True.
    :rtype: Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]
    """
    reshape = False if vertices.shape[1] == 9 else True
    face_vertices = vertices.reshape(-1, 9) if reshape else vertices
    edge_1 = face_vertices[:, 0:3] - face_vertices[:, 3:6]
    edge_2 = face_vertices[:, 3:6] - face_vertices[:, 6:9]

    normals = np.cross(edge_1, edge_2)
    row_sums = np.linalg.norm(normals, axis=1)

    if remove_degenerate:
        good_index = row_sums >= VECTOR_EPS
        face_vertices = face_vertices[good_index, :]
        normals = normals[good_index, :] / row_sums[good_index, np.newaxis]
        return (face_vertices.reshape(-1, 3), np.repeat(normals, 3, axis=0)) if reshape else (face_vertices, normals)

    row_sums[row_sums < VECTOR_EPS] = 1
    normals = normals / row_sums[:, np.newaxis]

    return np.repeat(normals, 3, axis=0) if reshape else normals


class Mesh:
    """Creates a Mesh object. Calculates the bounding box of the Mesh and calculates normals
     if not provided. Removes unused vertices, degenerate faces and duplicate vertices when clean is True.
     The vertices are sorted when clean is performed as a consequence of duplicate removal.

    :param vertices: N x 3 array of vertices
    :type vertices: numpy.ndarray
    :param indices: N X 1 array of indices
    :type indices: numpy.ndarray
    :param normals: N x 3 array of normals
    :type normals: Union[numpy.ndarray, None]
    :param colour: render colour of mesh
    :type colour: Colour
    :param clean: flag that specifies mesh should be cleaned
    :type clean: bool
    """
    def __init__(self, vertices, indices, normals=None, colour=None, clean=False):

        if not np.isfinite(vertices).all():
            raise ValueError('Non-finite value present in mesh vertices')

        self.vertices = vertices
        self.indices = indices

        if normals is not None and not clean:
            self.normals = normals
            if not np.isfinite(normals).all():
                raise ValueError('Non-finite value present in mesh normals')
        else:
            self.computeNormals()

        self.colour = Colour.black() if colour is None else Colour(*colour)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value
        self.bounding_box = BoundingBox.fromPoints(self.vertices)

    def append(self, mesh):
        """Appends a given mesh to this mesh. Indices are offset to ensure the correct
        vertices and normals are used

        :param mesh: mesh to append
        :type mesh: Mesh
        """
        count = self.vertices.shape[0]
        self.vertices = np.vstack((self.vertices, mesh.vertices))
        self.indices = np.concatenate((self.indices, mesh.indices + count))
        self.normals = np.vstack((self.normals, mesh.normals))

    def remove(self, index):
        """Splits this mesh into two parts using the given index. This operation can be used as an inverse
        of Mesh.append() but the split is not guaranteed to be a valid mesh if vertices have been rearranged.
        The first split is retained while the second is returned by the function.

        :param index: index to split from
        :type index: int
        :return: split mesh
        :rtype: Mesh
        """
        temp_indices = np.split(self.indices, [index])
        cut_off = temp_indices[0].max() + 1

        temp_vertices = self.vertices[cut_off:, :]
        temp_normals = self.normals[cut_off:, :]

        self.indices = temp_indices[0]
        self.vertices = self.vertices[0:cut_off, :]
        self.normals = self.normals[0:cut_off, :]

        return Mesh(temp_vertices, temp_indices[1] - cut_off, temp_normals, Colour(*self.colour))

    def rotate(self, matrix):
        """performs in-place rotation of mesh.

        :param matrix: 3 x 3 rotation matrix
        :type matrix: Union[numpy.ndarray, Matrix33]
        """
        _matrix = matrix[0:3, 0:3].transpose()
        self.vertices = self.vertices @ _matrix
        self.normals = self.normals @ _matrix

    def translate(self, offset):
        """performs in-place translation of mesh.
        Don't use a Vector3 for offset. it causes vertices to become an
        array of Vector3's which leads to other problems

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: Union[numpy.ndarray, Vector3]
        """
        self.vertices = self.vertices + offset

    def transform(self, matrix):
        """performs in-place transformation of mesh

        :param matrix: 4 x 4 transformation matrix
        :type matrix: Union[numpy.ndarray, Matrix44]
        """
        mesh = self.transformed(matrix)
        self._vertices = mesh.vertices
        self.normals = mesh.normals
        self.bounding_box = mesh.bounding_box

    def transformed(self, matrix):
        """performs a transformation of mesh

        :param matrix: 4 x 4 transformation matrix
        :type matrix: Union[numpy.ndarray, Matrix44]
        """
        _matrix = matrix[0:3, 0:3].transpose()
        offset = matrix[0:3, 3].transpose()

        vertices = self.vertices @ _matrix + offset
        normals = self.normals @ _matrix

        return Mesh(vertices, np.copy(self.indices), normals, Colour(*self.colour))

    def computeNormals(self):
        """Computes normals for the mesh and removes unused vertices, degenerate
        faces and duplicate vertices
        """
        vertices = self.vertices[self.indices]

        # Also removes unused vertices because of indexed vertices
        vn = compute_face_normals(vertices, remove_degenerate=True)
        vn, inverse = np.unique(np.hstack(vn), return_inverse=True, axis=0)

        self._vertices = vn[:, 0:3]  # bounds should not be changed by cleaning
        self.indices = inverse
        self.normals = vn[:, 3:]

    def copy(self):
        """Deep copies the mesh

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

    @classmethod
    def merge(cls, bounding_boxes):
        """compute the bounding box for an array of points

        :param bounding_boxes: list of bounding boxes
        :type bounding_boxes: List[BoundingBox]
        :return: bounding box
        :rtype: BoundingBox
        """
        if not bounding_boxes:
            raise ValueError('bounding_boxes cannot be empty')

        max_pos = min_pos = None
        for box in bounding_boxes:
            if max_pos is None:
                max_pos, min_pos = box.bounds
            else:
                max_pos = np.maximum(box.max, max_pos)
                min_pos = np.minimum(box.min, min_pos)

        return cls(max_pos, min_pos)

    @property
    def bounds(self):
        """property that returns max and min bounds of box (in that order)

        :return: max and min bounds of box
        :rtype: Tuple[float, float]
        """
        return self.max, self.min

    def translate(self, offset):
        """Performs in-place translation of bounding box by
        given offset

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: Union[numpy.ndarray, Vector3]
        """
        self.max += offset
        self.min += offset
        self.center += offset

    def transform(self, matrix):
        """performs a transformation of Bounding Box. The transformed box is not
        guaranteed to be a tight box (i.e it could be bigger than actual bounding box)

        :param matrix: transformation matrix
        :type matrix: Union[numpy.ndarray, Matrix44]
        """
        bound_min = [matrix[0, 3], matrix[1, 3], matrix[2, 3]]
        bound_max = [matrix[0, 3], matrix[1, 3], matrix[2, 3]]

        for i in range(3):
            for j in range(3):

                a = matrix[i, j] * self.min[j]
                b = matrix[i, j] * self.max[j]

                if a < b:
                    bound_min[i] += a
                    bound_max[i] += b

                else:
                    bound_min[i] += b
                    bound_max[i] += a

        return BoundingBox(bound_max, bound_min)
