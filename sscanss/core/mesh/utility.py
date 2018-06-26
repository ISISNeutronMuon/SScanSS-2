import numpy as np
from sscanss.core.util import BoundingBox


class Mesh:
    def __init__(self, vertices, indices, normals=None):

        self.vertices = vertices
        self.indices = indices

        if normals is not None:
            self.normals = normals
        else:
            self.computeNormals()

        self.computeBoundingBox()

    def append(self, mesh):
        count = self.vertices.shape[0]
        self.vertices = np.vstack((self.vertices, mesh.vertices))
        self.indices = np.concatenate((self.indices, mesh.indices + count))
        self.normals = np.vstack((self.normals, mesh.normals))

    def splitAt(self, index):
        pass

    def rotate(self, matrix):
        _matrix = matrix[0:3, 0:3]
        self.vertices = self.vertices.dot(_matrix.transpose())
        self.normals = self.normals.dot(_matrix.transpose())

    def translate(self, offset):
        self.vertices = self.vertices + offset

    def transform(self, matrix):
        _matrix = matrix[0:3, 0:3]
        offset = matrix[0:3, 3].transpose()

        self.vertices = self.vertices.dot(_matrix.transpose()) + offset
        self.normals = self.normals.dot(_matrix.transpose())

    def computeBoundingBox(self):
        bb_max = np.max(self.vertices, axis=0)
        bb_min = np.min(self.vertices, axis=0)
        center = (bb_max + bb_min) / 2
        radius = np.linalg.norm(bb_max - bb_min) / 2

        self.bounding_box = BoundingBox(bb_max, bb_min, center, radius)

    def computeNormals(self):
        """ calculates the vertex normals by determining the edges of the face
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
