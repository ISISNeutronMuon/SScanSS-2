import unittest
from sscanss.core.geometry import create_cuboid
from sscanss.core.instrument import CollisionManager
from sscanss.core.math import Matrix44


class TestCollisionClass(unittest.TestCase):

    def testManager(self):
        manager = CollisionManager(5)
        self.assertEqual(manager.max_size, 5)

        geometry = [create_cuboid(), create_cuboid()]
        transform = [Matrix44.identity(), Matrix44.fromTranslation([0, 0, 0.5])]
        manager.addColliders(geometry, transform, movable=True)

        self.assertEqual(len(manager.queries), 2)
        self.assertEqual(len(manager.colliders), 2)

        manager.addColliders([create_cuboid()], [Matrix44.fromTranslation([0, 0, 2.])])

        self.assertEqual(len(manager.queries), 2)
        self.assertEqual(len(manager.colliders), 3)

        manager.createAABBSets()
        self.assertListEqual(manager.collide(), [True, True, False])

        manager.clear()
        self.assertEqual(len(manager.queries), 0)
        self.assertEqual(len(manager.colliders), 0)

        geometry = [create_cuboid(), create_cuboid(), create_cuboid()]
        transform = [Matrix44.identity(), Matrix44.fromTranslation([0, 0, 0.5]), Matrix44.fromTranslation([0, 0, 1.5])]
        manager.addColliders(geometry, transform, CollisionManager.Exclude.Consecutive, movable=True)
        manager.createAABBSets()
        self.assertListEqual(manager.collide(), [False, False, False])

        manager.clear()
        transform = [Matrix44.identity(), Matrix44.identity(), Matrix44.identity()]
        manager.addColliders(geometry, transform, CollisionManager.Exclude.All, movable=True)
        manager.createAABBSets()
        self.assertListEqual(manager.collide(), [False, False, False])
