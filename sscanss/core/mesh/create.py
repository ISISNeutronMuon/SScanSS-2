import numpy as np
from .utility import Mesh
from ..math import rotation_btw_vectors


def create_cuboid(width=1.0, height=1.0, depth=1.0):
    """ generates the vertices, normals, and indices for a cuboid mesh

    :param width: cuboid width
    :type width: float
    :param height: cuboid height
    :type height: float
    :param depth: cuboid depth
    :type depth: float
    :return: The vertices, normals and index array of the mesh
    :rtype: sscanss.core.mesh.Mesh
    """
    # centre cube at the origin
    w = width / 2
    h = height / 2
    d = depth / 2

    ftl = [-w, d, h]  # front top left vertex
    ftr = [w, d, h]  # front top right vertex
    fbl = [-w, -d, h]  # front bottom left vertex
    fbr = [w, -d, h]  # front bottom right vertex
    btl = [-w, d, -h]  # back top left vertex
    btr = [w, d, -h]  # back top right vertex
    bbl = [-w, -d, -h]  # back bottom left vertex
    bbr = [w, -d, -h]  # back bottom right vertex

    vertices = [fbl, fbr, ftl, ftr, ftl, ftr,
                btl, btr, btl, btr, bbl, bbr,
                bbl, bbr, fbl, fbr, fbr, bbr,
                ftr, btr, bbl, fbl, btl, ftl]

    # normals for each face
    top = [0, 1, 0]
    bottom = [0, -1, 0]
    left = [-1, 0, 0]
    right = [1, 0, 0]
    front = [0, 0, 1]
    back = [0, 0, -1]

    normals = [front, front, front, front,
               top, top, top, top,
               back, back, back, back,
               bottom, bottom, bottom, bottom,
               right, right, right, right,
               left, left, left, left]

    indices = [0, 1, 2, 2, 1, 3,
               4, 5, 6, 6, 5, 7,
               8, 9, 10, 10, 9, 11,
               12, 13, 14, 14, 13, 15,
               16, 17, 18, 18, 17, 19,
               20, 21, 22, 22, 21, 23]

    return Mesh(np.array(vertices, dtype=np.float32), np.array(indices),
                np.array(normals, dtype=np.float32))


def create_cylinder(radius=1.0, height=1.0, slices=64, stacks=64, closed=True):
    """ generates the vertices, normals, and indices for a cylinder mesh

    :param radius: cylinder radius
    :type radius: float
    :param height: cylinder height
    :type height: float
    :param slices: number of radial segments to use
    :type slices: int
    :param stacks: number of height segments to use
    :type stacks: int
    :param closed: indicates if mesh should be closed at both ends
    :type closed: bool
    :return: The vertices, normals and index array of the mesh
    :rtype: sscanss.core.mesh.Mesh
    """
    half_height = height / 2

    # get angles (from 0 to 360) for each slice
    theta = np.linspace(0, 2 * np.pi, slices + 1)[:-1]

    # Add the cylinder torso and normals
    x = np.tile(np.sin(theta), stacks + 1)
    y = np.tile(np.cos(theta), stacks + 1)
    z_div = np.linspace(0.0, 1.0, stacks + 1)
    z = np.repeat(-1 * z_div * height + half_height, slices)


    vertices = np.column_stack((radius * x, radius * y, z))
    normals = np.column_stack((x, y, np.zeros(x.size)))

    a = np.fromiter((i for i in range(slices * stacks)), int)
    b = slices + a
    c = np.fromiter((i if i % slices else i - slices for i in range(1, slices * stacks + 1)), int)
    d = slices + c
    indices = np.column_stack((a, b, c, b, d, c)).flatten()

    if closed:
        # Add the cylinder caps
        cap_directions = [-1, 1]
        for sign in cap_directions:
            vertices = np.vstack((vertices, [0.0, 0.0, sign * half_height]))
            normals = np.vstack((normals, [0.0, sign, 0.0]))

            x = radius * np.sin(theta)
            y = radius * np.cos(theta)
            z = np.full(slices, [half_height * sign])

            vertex = np.column_stack((x, y, z))
            normal = np.tile([0, sign, 0], (slices, 1))
            vertices = np.vstack((vertices, vertex))
            normals = np.vstack((normals, normal))

            bottom_row = len(vertices) - slices
            a = bottom_row + np.arange(slices)
            b = bottom_row + (np.arange(1, slices + 1) % slices)
            c = np.full(len(a), bottom_row - 1)

            order = [b, a, c] if sign == -1 else [a, b, c]
            temp = np.column_stack(order).flatten()
            indices = np.concatenate((indices, temp))

    return Mesh(vertices.astype(np.float32), indices,
                normals.astype(np.float32))


def create_tube(inner_radius=0.5, outer_radius=1.0, height=1.0, slices=64, stacks=64):
    """ generates the vertices, normals, and indices for a tube mesh

    :param inner_radius: tube inner radius
    :type inner_radius: float
    :param outer_radius: tube outer radius
    :type outer_radius: float
    :param height: tube height
    :type height: float
    :param slices: number of radial segments to use
    :type slices: int
    :param stacks: number of height segments to use
    :type stacks: int
    :return: The vertices, normals and index array of the mesh
    :rtype: sscanss.core.mesh.Mesh
    """
    inner_cylinder = create_cylinder(inner_radius, height, slices, stacks, closed=False)
    outer_cylinder = create_cylinder(outer_radius, height, slices, stacks, closed=False)

    v_1 = outer_cylinder.vertices
    v_2 = inner_cylinder.vertices
    n_1 = outer_cylinder.normals
    n_2 = inner_cylinder.normals
    i_1 = outer_cylinder.indices
    # fix face windings for inner cylinder
    temp = inner_cylinder.indices.reshape(-1, 3)
    i_2 = temp[:, ::-1].flatten()

    vertex_count = slices * (stacks + 1)

    vertices = np.vstack((v_1, v_2,
                          v_1[:slices, :], v_2[:slices, :],  # vertices for top face
                          v_1[-slices:, :], v_2[-slices:, :]))  # vertices for bottom face

    normals = np.vstack((n_1, -1 * n_2,
                         np.tile([0.0, 1.0, 0.0], (slices * 2, 1)),  # normals for top face
                         np.tile([0.0, -1.0, 0.0], (slices * 2, 1))))  # normals for bottom face
    indices = np.concatenate((i_1, vertex_count + i_2))

    vertex_count *= 2

    # Add caps to the pipe
    for x in range(2):
        a = vertex_count + np.arange(slices)
        b = slices + a
        d = vertex_count + (np.arange(1, slices + 1) % slices)
        c = slices + d

        order = [d, b, a, d, c, b] if x == 0 else [a, b, d, b, c, d]
        temp = np.column_stack(order).flatten()
        indices = np.concatenate((indices, temp))
        vertex_count += slices * 2

    return Mesh(vertices.astype(np.float32), indices,
                normals.astype(np.float32))


def create_sphere(radius=1.0, slices=64, stacks=64):
    """ generates the vertices, normals, and indices for a sphere mesh

    :param radius: sphere radius
    :type radius: float
    :param slices: number of radial segments used
    :type slices: int
    :param stacks: number of height segments used
    :type stacks: int
    :return: The vertices, normals and index array of the mesh
    :rtype: sscanss.core.mesh.Mesh
    """
    # get inclination angles (from 0 to 180) for each stack
    theta = np.linspace(0, np.pi, stacks + 1)
    # get azimuth angles (from 0 to 360) for each slice
    phi = np.linspace(0, 2 * np.pi, slices + 1)[:-1]

    cos_phi = np.tile(np.cos(phi), stacks + 1)
    sin_phi = np.tile(np.sin(phi), stacks + 1)
    cos_theta = np.repeat(np.cos(theta), slices)
    sin_theta = np.repeat(np.sin(theta), slices)

    x = -radius * cos_phi * sin_theta
    y = radius * sin_phi * sin_theta
    z = radius * cos_theta

    vertices = np.column_stack((x, y, z))
    # normals are the same as vertices just normalized
    normals = np.copy(vertices)
    # TODO: create function to compute magnitude then replace the lines below
    row_sums = np.linalg.norm(normals, axis=1)
    normals = normals / row_sums[:, np.newaxis]

    # get index for the mesh
    b = np.array([i for i in range(slices * stacks)])
    a = np.array([i if i % slices else i - slices for i in range(1, slices * stacks + 1)])
    c = slices + b
    d = slices + a

    # indices for the top face
    top = np.column_stack((b[:slices], c[:slices], d[:slices])).flatten()
    # indices for the mid-section
    middle = np.column_stack((a[slices:-slices],
                              b[slices:-slices],
                              d[slices:-slices],
                              b[slices:-slices],
                              c[slices:-slices],
                              d[slices:-slices])).flatten()
    # indices for the bottom face
    bottom = np.column_stack((a[-slices:], b[-slices:], d[-slices:])).flatten()

    indices = np.concatenate((top, middle, bottom))

    return Mesh(vertices.astype(np.float32), indices,
                normals.astype(np.float32))


def create_plane(plane, width=1.0, height=1.0, slices=1, stacks=1):
    """ generates the vertices, normals, and indices for a plane mesh

    :param plane: the plane normal and point
    :type plane: sscanss.core.math.Plane
    :param width: plane width
    :type width: float
    :param height: plane height
    :type height: float
    :param slices: number of width segments used
    :type slices:
    :param stacks: number of height segments used
    :type stacks: int
    :return: The vertices, normals and index array of the mesh
    :rtype: sscanss.core.mesh.Mesh
    """

    h = height / 2
    w = width / 2
    x = np.tile(np.linspace(-w, w, slices + 1), stacks + 1)
    y = np.repeat(np.linspace(-h, h, stacks + 1), slices + 1)

    vertices = np.column_stack((x, y, np.zeros(x.size)))
    matrix = rotation_btw_vectors(plane.normal, np.array([0., 0., 1.]))

    vertices = vertices.dot(matrix) + plane.point
    normals = np.tile(plane.normal, (x.size, 1))

    a = np.fromiter((i for i in range((slices + 1) * stacks) if (i + 1) % (slices + 1) != 0), int)
    b = a + slices + 1
    c = b + 1
    d = a + 1

    indices = np.column_stack([d, b, a, d, c, b]).flatten()

    return Mesh(vertices.astype(np.float32), indices, normals.astype(np.float32))
