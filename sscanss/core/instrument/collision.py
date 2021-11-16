"""
Classes for collision detection
"""
from enum import Enum, unique
from bitarray import bitarray
import gimpact
import numpy as np


class Collider:
    """Represents a geometry that can be collided with.

    :param identifier: unique identifier
    :type identifier: int
    :param vertices: N x 3 array of vertices
    :type vertices: numpy.ndarray
    :param indices: N X 1 array of indices
    :type indices: numpy.ndarray
    :param mask_size: size of exclusion mask
    :type mask_size: int
    :param transform: transformation matrix
    :type transform: Union[None, Matrix44]
    """
    def __init__(self, identifier, vertices, indices, mask_size=32, transform=None):

        self.id = identifier
        self.geometry = gimpact.TriMesh(vertices, indices).decimate(50000)
        self.excludes = bitarray(mask_size)
        self.excludes.setall(False)
        self.excludes[identifier] = True

        if transform is not None:
            self.geometry.transform(np.array(transform, np.float32))


class CollisionManager:
    """Manages the collision objects and handles collision queries

    :param max_size: maximum number of colliders
    :type max_size: int
    """
    @unique
    class Exclude(Enum):
        All = 0
        Consecutive = 1
        Nothing = 2

    def __init__(self, max_size=32):
        self.max_size = max_size
        self.colliders = []
        self.collider_aabbs = None
        self.queries = []
        self.query_aabbs = None

    def createAABBSets(self):
        """Creates and initialises AABBSets for the colliders"""
        if self.collider_aabbs is None:
            self.collider_aabbs = gimpact.AABBSet(len(self.colliders))

        for i, c in enumerate(self.colliders):
            self.collider_aabbs[i] = c.geometry.bounds

        if self.query_aabbs is None:
            self.query_aabbs = gimpact.AABBSet(len(self.queries))

        for i, c in enumerate(self.queries):
            self.query_aabbs[i] = c.geometry.bounds

    def clear(self):
        """Clears the manager"""
        self.queries.clear()
        self.colliders.clear()
        self.collider_aabbs = None
        self.query_aabbs = None

    def addColliders(self, geometry, transform=None, exclude=Exclude.Nothing, movable=False):
        """Adds collider geometry to the manager. This function creates a collider from
        a list of scene nodes, specifies how they should collide (e.g. Exclude.Consecutive
        indicates that consecutive nodes cannot collide with each other), and indicates if the
        collider can move.

        :param geometry: scene nodes containing vertices and indices of colliders
        :type geometry: Union[List[Node], List[Mesh]]
        :param transform: transformations of the geometries
        :type transform: List[Matrix44]
        :param exclude: indicates which node should be exclude from collision checks
        :type exclude: CollisionManager.Exclude
        :param movable: flag indicating the collider can move
        :type movable: bool
        """
        object_count = len(self.colliders)
        node_count = len(geometry)
        for index, geom in enumerate(geometry):
            t_matrix = None if transform is None else transform[index]
            obj = Collider(index + object_count, geom.vertices, geom.indices, self.max_size, t_matrix)
            if exclude == CollisionManager.Exclude.All:
                for i in range(object_count, object_count + node_count):
                    obj.excludes[i] = True
            elif exclude == CollisionManager.Exclude.Consecutive:
                if index - 1 >= 0:
                    obj.excludes[obj.id - 1] = True
                if index + 1 < node_count:
                    obj.excludes[obj.id + 1] = True

            self.colliders.append(obj)
            if movable:
                self.queries.append(obj)

    def collide(self):
        """Checks for colliding objects

        :return: indicates which colliders are colliding
        :rtype: List[bool]
        """
        collisions = [False] * len(self.colliders)
        intersecting = self.collider_aabbs.find_intersections(self.query_aabbs)
        for i, j in intersecting:
            collider = self.colliders[i]
            query = self.queries[j]

            if collider.excludes[query.id]:
                continue

            contacts = gimpact.trimesh_trimesh_collision(collider.geometry, query.geometry, True)
            if contacts:
                collisions[query.id] = True
                collisions[collider.id] = True

        return collisions
