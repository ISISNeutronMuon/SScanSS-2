import numpy as np
from ..util.scene import BoundingBox


class Mesh:
    def __init__(self, vertices, indices, normals=None):
        """ Creates a Mesh object. Calculates the bounding box
        of the Mesh and calculates normals if not provided.

        :param vertices: N x 3 array of vertices
        :type vertices: numpy.ndarray
        :param indices: N X 1 array of indices
        :type indices: numpy.ndarray
        :param normals: N x 3 array of normals
        :type normals: Union[numpy.ndarray, None]
        """
        self.vertices = vertices
        self.indices = indices

        if normals is not None:
            self.normals = normals
        else:
            self.computeNormals()

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
        :type mesh: sscanss.core.mesh.utility.Mesh
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
        :rtype: sscanss.core.mesh.utility.Mesh
        """
        temp_indices = np.split(self.indices, [index])
        temp_vertices = self.vertices[temp_indices[1], :]
        temp_normals = self.normals[temp_indices[1], :]

        self.vertices = self.vertices[temp_indices[0], :]
        self.indices = np.arange(index)
        self.normals = self.normals[temp_indices[0], :]

        return Mesh(temp_vertices, np.arange(temp_indices[1].size), temp_normals)

    def rotate(self, matrix):
        """ performs in-place rotation of mesh.

        :param matrix: 3 x 3 rotation matrix
        :type matrix: Union[numpy.ndarray, pyrr.objects.matrix33.Matrix33]
        """
        _matrix = matrix[0:3, 0:3]
        self.vertices = self.vertices.dot(_matrix.transpose())
        self.normals = self.normals.dot(_matrix.transpose())

    def translate(self, offset):
        """ performs in-place translation of mesh.
        Don't use a Vector3 for offset. it causes vertices to become an
        array of Vector3's which leads to other problems

        :param offset: 3 x 1 array of offsets for X, Y and Z axis
        :type offset: numpy.ndarray
        """
        self.vertices = self.vertices + offset

    def transform(self, matrix):
        """ performs in-place transformation of mesh

        :param matrix: 4 x 4 transformation matrix
        :type matrix: numpy.ndarray
        """
        _matrix = matrix[0:3, 0:3]
        offset = matrix[0:3, 3].transpose()

        self.vertices = self.vertices.dot(_matrix.transpose()) + offset
        self.normals = self.normals.dot(_matrix.transpose())

    def computeBoundingBox(self):
        """ Calculates the axis aligned bounding box of the mesh """
        bb_max = np.max(self.vertices, axis=0)
        bb_min = np.min(self.vertices, axis=0)
        center = (bb_max + bb_min) / 2
        radius = np.linalg.norm(bb_max - bb_min) / 2

        self.bounding_box = BoundingBox(bb_max, bb_min, center, radius)

    def computeNormals(self):
        """ Calculates the vertex normals by determining the edges of the face
        and finding the cross product of the edges. The function assumes that every 3
        consecutive vertices belong to the same face.
        """
        face_vertices = self.vertices.reshape(-1, 9)
        edge_1 = face_vertices[:, 0:3] - face_vertices[:, 3:6]
        edge_2 = face_vertices[:, 3:6] - face_vertices[:, 6:9]

        normals = np.cross(edge_1, edge_2)
        row_sums = np.linalg.norm(normals, axis=1)
        normals = normals / row_sums[:, np.newaxis]

        self.normals = np.repeat(normals, 3, axis=0)
