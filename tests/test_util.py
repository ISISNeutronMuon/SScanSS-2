import unittest
import numpy as np
from sscanss.core.util import Colour, to_float, clamp


class TestUtil(unittest.TestCase):
    def testColourClass(self):
        colour = Colour.black()
        np.testing.assert_array_almost_equal(colour[:], [0.0, 0.0, 0.0, 1.0], decimal=5)

        colour = Colour.normalize(255, 0.0, 0.0)
        np.testing.assert_array_almost_equal(colour[:], [1.0, 0.0, 0.0, 1.0], decimal=5)
        np.testing.assert_array_equal(colour.rgba, [255, 0, 0, 255])

        colour = Colour.white()
        np.testing.assert_array_almost_equal(colour.rgbaf, [1.0, 1.0, 1.0, 1.0], decimal=5)

        colour = Colour(127, 0.5, 0.3, -1)
        self.assertAlmostEqual(colour.r, 1.0, 5)
        self.assertAlmostEqual(colour.g, 0.5, 5)
        self.assertAlmostEqual(colour.b, 0.3, 5)
        self.assertAlmostEqual(colour.a, 0.0, 5)
        np.testing.assert_array_almost_equal(colour.invert()[:], [0.0, 0.5, 0.7, 0.0], decimal=5)

        self.assertEqual(str(colour.white()), 'rgba(1.0, 1.0, 1.0, 1.0)')
        self.assertEqual(repr(colour.white()), 'Colour(1.0, 1.0, 1.0, 1.0)')

    def testToFloat(self):
        value, ok = to_float('21')
        self.assertAlmostEqual(value, 21, 5)
        self.assertTrue(ok)

        value, ok = to_float('2.1e3')
        self.assertAlmostEqual(value, 2100, 5)
        self.assertTrue(ok)

        value, ok = to_float('-1001.67845')
        self.assertAlmostEqual(value, -1001.67845, 5)
        self.assertTrue(ok)

        value, ok = to_float('Hello')
        self.assertEqual(value, None)
        self.assertFalse(ok)

    def testClamp(self):
        value = clamp(-1, 0, 1)
        self.assertEqual(value, 0)

        value = clamp(100, 0, 50)
        self.assertEqual(value, 50)

        value = clamp(20, 0, 50)
        self.assertEqual(value, 20)


if __name__ == '__main__':
    unittest.main()
