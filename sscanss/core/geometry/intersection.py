"""
Functions for geometry intersection and path length calculation
"""
import numpy as np
import math

import scipy.ndimage
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from ..math.transform import view_from_plane
from sscanss.core.geometry.primitive import create_cuboid
from ..math.matrix import Matrix44
from ..math.vector import Vector3, Vector4
eps = 0.000001


def closest_triangle_to_point(faces, points):
    """Computes the closest face to a given 3D point. Assumes face is triangular.
    Based on code from http://www.iquilezles.org/www/articles/triangledistance/triangledistance.htm

    :param faces: faces: N x 9 array of triangular face vertices
    :type faces: numpy.ndarray
    :param points: M x 3 array of points to find closest faces
    :type points: numpy.ndarray
    :return: M x 9 array of faces corresponding to points
    :rtype: numpy.ndarray
    """
    result = []

    v1 = faces[:, 0:3]
    v2 = faces[:, 3:6]
    v3 = faces[:, 6:9]
    v21 = v2 - v1
    v32 = v3 - v2
    v13 = v1 - v3
    nor = np.cross(v21, v13)
    c13 = np.cross(v13, nor)
    c32 = np.cross(v32, nor)
    c21 = np.cross(v21, nor)
    dot_v21 = 1.0 / np.einsum('ij,ij->i', v21, v21)
    dot_v32 = 1.0 / np.einsum('ij,ij->i', v32, v32)
    dot_v13 = 1.0 / np.einsum('ij,ij->i', v13, v13)
    dot_nor = 1.0 / np.einsum('ij,ij->i', nor, nor)

    for point in points:
        dist = np.zeros(v1.shape[0], v1.dtype)

        p1 = point - v1
        p2 = point - v2
        p3 = point - v3

        mask = (np.sign(np.einsum('ij,ij->i', c21, p1)) + np.sign(np.einsum('ij,ij->i', c32, p2)) +
                np.sign(np.einsum('ij,ij->i', c13, p3)))

        mask = mask < 2.0

        oo = v21[mask]
        ooo = p1[mask]
        temp = oo * np.clip(np.einsum('ij,ij->i', oo, ooo) * dot_v21[mask], 0.0, 1.0)[:, np.newaxis] - ooo
        temp = np.einsum('ij,ij->i', temp, temp)

        oo = v32[mask]
        ooo = p2[mask]
        temp_2 = oo * np.clip(np.einsum('ij,ij->i', oo, ooo) * dot_v32[mask], 0.0, 1.0)[:, np.newaxis] - ooo
        temp_2 = np.einsum('ij,ij->i', temp_2, temp_2)

        oo = v13[mask]
        ooo = p3[mask]
        temp_3 = oo * np.clip(np.einsum('ij,ij->i', oo, ooo) * dot_v13[mask], 0.0, 1.0)[:, np.newaxis] - ooo
        temp_3 = np.einsum('ij,ij->i', temp_3, temp_3)

        inv_mask = ~mask
        temp_4 = np.einsum('ij,ij->i', nor[inv_mask], p1[inv_mask])
        dist[inv_mask] = temp_4 * temp_4 * dot_nor[inv_mask]

        dist[mask] = np.minimum(temp, np.minimum(temp_2, temp_3))

        closest_face = faces[dist.argmin()]
        result.append(closest_face)

    return np.array(result)


def mesh_plane_intersection(mesh, plane):
    """Gets the intersection between a triangular mesh and a plane. The algorithm returns
    a set of lines is points pairs where each even indexed point is the start of a line
    and the next point is the end. An empty list implies no intersection.
    Based on code from *Real-Time Collision Detection (1st Edition) By Christer Ericson*

    :param mesh: a triangular mesh
    :type mesh: Mesh
    :param plane: plane normal and point
    :type plane: Plane
    :return: array of points pairs
    :rtype: List[numpy.ndarray]
    """
    # The algorithm checks if all vertices on each mesh face are on the same side of the plane
    # if this is true, the face does not intersect the plane. Once the faces that intersect
    # the plane are determined, the segments that form the face are tested for intersection with the plane.
    segments = []
    all_vertices = mesh.vertices[mesh.indices]
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
        non_zeros = np.nonzero(dist)[0]
        if non_zeros.size == 0:
            # all vertices on the plane we currently don't
            # care about this condition
            continue
        elif non_zeros.size == 1:
            # edge lies on the plane
            tmp = np.delete((0, 1, 2), non_zeros[0])
            a = vertices[tmp[0], :]
            b = vertices[tmp[1], :]
            segments.extend([a, b])
            continue
        elif non_zeros.size == 2:
            # point lies on plane so check intersection for a single line
            tmp = np.delete((0, 1, 2), non_zeros)
            points.append(vertices[tmp[0], :])
            face_indices = [(non_zeros[0], non_zeros[1])]
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

    :param point_a: 3D starting point of the segment
    :type point_a: numpy.ndarray
    :param point_b: 3D ending point of the segment
    :type point_b: numpy.ndarray
    :param plane: the plane
    :type plane: Plane
    :return: point of intersection or None if no intersection
    :rtype: Union[numpy.ndarray, None]
    """
    ab = point_b - point_a
    n = -np.dot(plane.normal, point_a - plane.point)
    d = np.dot(plane.normal, ab)
    if -eps < d < eps:
        # ignore case where line lies on plane
        return None
    t = n / d
    if 0.0 <= t <= 1.0:
        q = point_a + t * ab
        return q

    return None


def segment_triangle_intersection(origin, direction, length, faces, tol=1e-5):
    """Calculates the distance along the specified direction from the origin to
    intersection points on the triangles in a mesh.

    :param origin: origin of segment
    :type origin: Vector3
    :param direction: normalized direction of segment
    :type direction: Vector3
    :param length: length of segment
    :type length: float
    :param faces: faces: N x 9 array of triangular face vertices
    :type faces: numpy.ndarray
    :param tol: tolerance to determine if distance is unique
    :type tol: float
    :return: sorted distances of intersection
    :rtype: List[float]
    """
    p0 = faces[:, 0:3]
    p1 = faces[:, 3:6]
    p2 = faces[:, 6:9]
    e1 = p1 - p0
    e2 = p2 - p0

    q = np.cross(np.expand_dims(direction, axis=0), e2)
    a = np.einsum('ij,ij->i', q, e1)

    mask = np.where(np.logical_or(a < -eps, a > eps))[0]
    if mask.size == 0:
        return []

    e1 = e1[mask, :]
    e2 = e2[mask, :]
    a = a[mask]
    q = q[mask, :]

    f = 1 / a
    s = -p0[mask] + origin
    u = f * np.einsum('ij,ij->i', q, s)

    mask = np.where(u >= 0.0)[0]
    if mask.size == 0:
        return []

    s = s[mask, :]
    e1 = e1[mask, :]
    e2 = e2[mask, :]
    f = f[mask]
    u = u[mask]

    r = np.cross(s, e1)
    v = f * np.einsum('ij,ij->i', r, np.expand_dims(direction, axis=0))

    mask = np.where(np.logical_and(v >= 0.0, (u + v) <= 1.0))[0]
    if mask.size == 0:
        return []

    t = f[mask] * np.einsum('ij,ij->i', r[mask, :], e2[mask, :])

    mask = np.where(np.logical_and(t >= 0, t <= length))
    t = t[mask]

    t = np.sort(t)
    distances = []

    for i in range(t.size):
        if distances and abs(distances[-1] - t[i]) < tol:
            continue

        distances.append(t[i])

    return distances


def path_length_calculation(mesh, gauge_volume, beam_axis, diff_axis):
    """Calculates the path length of the beam through a sample model. It assumes that the beam starts outside the
    sample and every pair of face intersections is taken as beam entry and exit from the sample. The path length is
    set to zero if beam hits the gauge volume outside the sample or an entry/exit face pair is not found.
    This technique could give incorrect results if the sample has internal faces or spurious faces from bad scanning
    intersect with the beam.

    :param mesh: a triangular mesh
    :type mesh: Mesh
    :param gauge_volume: The centre of the gauge volume
    :type gauge_volume: Vector3
    :param beam_axis: The direction of the beam
    :type beam_axis: Vector3
    :param diff_axis: The direction of the beam
    :type diff_axis: List[Vector3]
    :return: Path length from beam origin to each detector
    :rtype: Tuple[float]
    """

    length = mesh.bounding_box.radius + 100  # mesh radius + fudge value
    num_of_detectors = len(diff_axis)
    vertices = mesh.vertices[mesh.indices]
    v = vertices.reshape(-1, 9)

    # incoming beam from beam source to gauge volume
    distances = segment_triangle_intersection(gauge_volume, -beam_axis, length, v)
    # flag for when beam is in the sample, beam starts outside the sample
    if not distances or len(distances) % 2 == 0:
        return [0.0] * num_of_detectors

    d = np.array([0.0, *distances, length])
    beam_to_gauge = np.sum((d[1:] - d[:-1])[::2])

    path_lengths = []
    # outgoing beam from gauge volume to collimator
    for axis in diff_axis:
        distances = segment_triangle_intersection(gauge_volume, axis, length, v)
        if not distances or len(distances) % 2 == 0:
            path_lengths.append(0.0)
            continue

        d = np.array([0.0, *distances, length])
        gauge_to_detector = np.sum((d[1:] - d[:-1])[::2])

        path_lengths.append(beam_to_gauge + gauge_to_detector)

    return path_lengths


def point_selection(start, end, faces, volumes=None):
    """Calculates the intersection points between a line segment and triangle mesh.

    :param start: line segment start point
    :type start: Vector3
    :param end: line segment end point
    :type end: Vector3
    :param faces: faces: N x 9 array of triangular face vertices
    :type faces: numpy.ndarray
    :return: array of intersection points
    :rtype: numpy.ndarray
    """
    direction = end - start
    length = direction.length
    direction /= length

    if volumes:
        distances = volume_ray_intersection(start, end, volumes)
    else:
        if length < eps or faces is None:
            return np.array([])
        distances = segment_triangle_intersection(start, direction, length, faces)

    if not distances:
        return np.array([])
    distances = np.reshape(distances, (len(distances), 1))

    return start + direction * distances


class VolumeSlice:
    """Data class for the volume slice and rectangle with position and size of slice on the plane

    :param image: slice image
    :type image: numpy.ndarray
    :param rect: rectangle which indicates the x, y position and width, height of the slice
    :type rect: Tuple[float, float, float float]
    """
    def __init__(self, image, rect):

        self.image = image
        self.rect = rect


def volume_plane_intersection(volume, plane, resolution=1024):
    """Gets the intersection between a volume and a plane. The algorithm generates indices using the
    intersection between the plane and the volume box then the intensity values for the slice are
    determined using linear interpolation (cubic is too slow).

    :param volume: volume
    :type volume: Volume
    :param plane: plane normal and point
    :type plane: Plane
    :param resolution: number of pixel in x and y dimensions of slice
    :type resolution: int
    :return: volume slice
    :rtype: Optional[VolumeSlice]
    """
    volume_shape = np.array(volume.shape)
    center = volume_shape / 2
    scale = (volume_shape - 1) / volume_shape
    segments = mesh_plane_intersection(volume.asMesh(), plane)
    if len(segments) == 0:
        return

    view_matrix = view_from_plane(plane.normal)
    matrix = np.identity(4)
    matrix[:3, :3] = view_matrix.transpose()
    matrix = np.linalg.inv(matrix @ volume.transform_matrix @ np.diag([*volume.voxel_size, 1]))

    rotated_segments = np.row_stack(segments) @ view_matrix
    min_limits = rotated_segments.min(axis=0)
    max_limits = rotated_segments.max(axis=0)

    xs = np.linspace(min_limits[0], max_limits[0], resolution, dtype=np.float32)
    ys = np.linspace(min_limits[1], max_limits[1], resolution, dtype=np.float32)
    idx = np.zeros((4, resolution, resolution), dtype=np.float32)
    idx[0, :, :] = xs.reshape(resolution, 1)
    idx[1, :, :] = ys.reshape(1, resolution)
    idx[2, :, :] = max_limits[2]
    idx[3, :, :] = 1

    idx = np.einsum('il, ljk->ijk', matrix, idx)[:3, :, :] + center.reshape(3, 1, 1)
    idx *= scale.reshape(3, 1, 1)
    image_slice = ndimage.map_coordinates(volume.data, idx, order=1, mode='constant', cval=0)

    rect = (*min_limits[:2], *(max_limits[:2] - min_limits[:2]))
    return VolumeSlice(image_slice.transpose().copy(), rect)


def volume_ray_intersection(start, end, volume):
    """Finds point where a ray intersects a volume - uses same algorithm as in the fragment shader for volumes
    :param volume: the volume to find intersection with
    :type volume: Volume
    :param start: start of the ray
    :type start: Vector3
    :param end: end of the ray
    :type end: Vector3
    :return: the point where ray intersects the volume
    :rtype: Vector3
    """

    # First get the mesh of bounding box of the volume
    width = volume.voxel_size[0] * volume.data.shape[0]
    height = volume.voxel_size[1] * volume.data.shape[1]
    depth = volume.voxel_size[2] * volume.data.shape[2]

    bounding_box_mesh = create_cuboid(width, depth, height)
    bounding_box_mesh.transform(volume.transform_matrix)

    # Find where does the ray intersect the volume's planes (should be 2 points)
    line_segment_direction = end - start
    line_segment_length = line_segment_direction.length
    line_segment_direction /= line_segment_length
    faces = bounding_box_mesh.vertices[bounding_box_mesh.indices].reshape(-1, 9)

    intersection_distances = segment_triangle_intersection(start, line_segment_direction, line_segment_length, faces)
    intersection_distances.sort()
    if len(intersection_distances) == 2:
        points = []
        translate = Vector3([width / 2, height / 2, depth / 2])
        for distance in intersection_distances:
            point = start + line_segment_direction * distance
            point = Vector3((np.linalg.inv(volume.transform_matrix) @ Vector4([point.x, point.y, point.z, 0]))[:3])
            point = point + translate
            points.append(point)

        inside_segment = points[1] - points[0]
        step_vector = 0.001 * inside_segment / inside_segment.length
        num_of_steps = int(math.ceil(inside_segment.length / step_vector.length))
        if num_of_steps > 10000:
            num_of_steps = 10000

        volume_space = [np.linspace(0, volume.data.shape[i]*volume.voxel_size[i], volume.data.shape[i])
                        for i in range(3)]

        sample_points = [list(points[0] + step_vector * i) for i in range(num_of_steps)]
        sampled_points = scipy.ndimage.map_coordinates(volume.data, sample_points, order=1)
        interpolated_values = volume.curve.evaluate(sampled_points)

        for i, value in enumerate(interpolated_values):
            if value > 0.0:
                position = sample_points[i] - translate
                return [(start - position).length]
        return None
    return None

