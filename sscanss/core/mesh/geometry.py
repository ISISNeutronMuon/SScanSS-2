import numpy as np


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
