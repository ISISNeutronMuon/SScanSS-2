import numpy as np


def compute_normals(vertices):
    """ calculates the vertex normals by determining the edges of the face
    and finding the cross product of the edges. The function assumes that every 3
    consecutive vertices belong to the same face.

    :param vertices: array of vertices (N x 3)
    :type vertices: np.array
    :return: array of normals (N x 3)
    :rtype: np.array
    """
    face_vertices = vertices.reshape(-1, 9)
    edge_1 = face_vertices[:, 0:3] - face_vertices[:, 3:6]
    edge_2 = face_vertices[:, 3:6] - face_vertices[:, 6:9]

    normals = np.cross(edge_1, edge_2)
    row_sums = np.linalg.norm(normals, axis=1)
    normals = normals / row_sums[:, np.newaxis]
    normals = np.repeat(normals, 3, axis=0)

    return normals
