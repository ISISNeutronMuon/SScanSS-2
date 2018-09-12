import numpy as np

eps = 0.000001


def closest_point_on_triangle(vertex_a, vertex_b, vertex_c, point):
    a = vertex_a
    b = vertex_b
    c = vertex_c

    ab = b - a
    ac = c - a
    ap = point - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1/ (d1 - d3)
        return a + v * ab

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2/ (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / (d4 - d3 + d5 -d6)
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def closest_triangle_to_point(vertices, point):
    min_dist = np.NaN
    for i in range(vertices.shape[0]):
        a = vertices[i, 0:3]
        b = vertices[i, 3:6]
        c = vertices[i, 6:9]
        closest_pt = closest_point_on_triangle(a, b, c, point)
        dist = closest_pt - point
        dist = np.dot(dist, dist)
        if np.isnan(min_dist) or dist < min_dist:
            closest_face = vertices[i]
            min_dist = dist

    return closest_face, min_dist


def mesh_plane_intersection(mesh, plane):
    """Gets the intersection between a triangular mesh and a plane. The algorithm returns
    a set of lines is points pairs where each even indexed point is the start of a line
    and the next point is the end. An empty list implies no intersection.
    Based on code from Real-Time Collision Detection (1st Edition) By Christer Ericson

    :param mesh: a triangular mesh
    :type mesh: sscanss.core.mesh.Mesh
    :param plane: a plane
    :type plane: sscanss.core.math.Plane
    :return: array of points pairs
    :rtype: list[numpy.ndarray]
    """
    # The algorithm checks if all vertices on each mesh face are on the same side of the plane
    # if this is true, the face does not intersect the plane. Once the faces that intersect
    # the plane are determined, the segments that form the face are tested for intersection with the plane.
    segments = []
    all_vertices = mesh  # get vertices
    all_dist = np.dot(all_vertices - plane.point, plane.normal)
    offsets = range(0, all_vertices.shape[0], 3)
    gg = np.logical_and.reduceat(all_dist > 0, offsets)
    ll = np.logical_and.reduceat(all_dist < 0, offsets)
    indices = np.where(~ll & ~gg)[0] * 3
    if indices.size == 0:
        return segments

    for index in indices:
        dist = all_dist[index:index + 3]
        vertices = all_vertices[index:index + 3, :]

        points = []
        zeros = np.nonzero(dist)[0]
        if zeros.size == 0:
            # all vertices on the plane we currently don't
            # care about this condition
            continue
        elif zeros.size == 1:
            # edge lies on the plane
            a = vertices[zeros[0], :]
            b = vertices[zeros[1], :]
            segments.extend([a, b])
            continue
        elif zeros.size == 2:
            # point lies on plane so check intersection for a single line
            indices = (0, 1, 2)
            points.append(vertices[zeros[0], :])
            face_indices = [np.delete(indices, zeros[0])]
        else:
            face_indices = [(0, 1), (1, 2), (0, 2)]

        for i, j in face_indices:
            a = vertices[i, :]
            b = vertices[j, :]

            point = segment_plane_intersection(a, b, plane)
            if point is not None:
                points.append(point)

            if len(points) == 2:
                segments.extend([points[0], points[1]])
                break

    return segments


def segment_plane_intersection(point_a, point_b, plane):
    """Gets the intersection between a line segment and a plane

    :param point_a:  3D starting point of the segment
    :type point_a:numpy.ndarray
    :param point_b: 3D ending point of the segment
    :type point_b: numpy.ndarray
    :param plane: the plane
    :type plane: sscanss.core.math.Plane
    :return: point of intersection or None if no intersection
    :rtype: Union[numpy.ndarray, NoneType]
    """
    ab = point_b - point_a
    n = - np.dot(plane.normal, point_a - plane.point)
    d = np.dot(plane.normal, ab)
    if -eps < d < eps:
        # ignore case where line lies on plane
        return None
    t = n / d
    if 0.0 <= t <= 1.0:
        q = point_a + t * ab
        return q

    return None
