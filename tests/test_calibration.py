import json
import unittest
import numpy as np
from sscanss.core.instrument import circle_point_analysis, generate_description, Link
from sscanss.core.instrument.create import extract_positioner
from sscanss.core.instrument.calibration import correct_line, correct_circle_axis


class TestCalibration(unittest.TestCase):
    def testGeometryCorrection(self):
        axis = np.array([0., 0., -1.])
        center = np.zeros(3)
        offsets = np.array([180, 0, 90, 70])
        points = np.array([[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])

        new_axis = correct_circle_axis(axis, center, points, offsets)
        np.testing.assert_array_almost_equal(new_axis, [0, 0, 1], decimal=5)
        np.testing.assert_array_almost_equal(correct_circle_axis(new_axis, center, points, offsets),
                                             new_axis, decimal=5)

        axis = np.array([-1., 0., 0.])
        center = np.zeros(3)
        offsets = np.array([100, 0, 50])
        points = np.array([[50., 0., 0.], [-50., 0., 0.], [0., 0., 0.]])
        new_center, new_axis = correct_line(axis, center, points, offsets)
        np.testing.assert_array_almost_equal(new_axis, [1, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(new_center, [-50, 0, 0], decimal=5)
        new_center, new_axis = correct_line(new_axis, center, points, offsets)
        np.testing.assert_array_almost_equal(new_axis, [1, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(new_center, [-50, 0, 0], decimal=5)

    def testCPA(self):
        points = [np.array([[12., 0., 1.5], [11.41421356, 1.41421356, 1.5], [10., 2., 1.5],
                            [8.58578644, 1.41421356, 1.5], [8., 0., 1.5]]),
                  np.array([[10., 0., 1.5], [10.29289322, -0.70710678, 1.5], [11., -1., 1.5],
                            [11.70710678, -0.70710678, 1.5], [12., 0., 1.5]])]

        offsets = [np.array([0.0, 45.0, 90.0, 135.0, 180.0]), np.array([-180.0, -135.0, -90.0, -45.0, 0.0])]
        types = [Link.Type.Revolute, Link.Type.Revolute]
        homes = [0.0, 0.0]

        result = circle_point_analysis(points, types, offsets, homes)

        np.testing.assert_array_almost_equal(result.joint_axes[0], [0., 0., 1.], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_axes[1], [0., 0., 1.], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_origins[0], [0., 0., 0.], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_origins[1], [1., 0., 0.], decimal=5)
        base, tool = np.identity(4), np.identity(4)
        base[:3, 3] = [10., 0., 1.5]
        tool[:3, 3] = [1., 0., 0.]
        np.testing.assert_array_almost_equal(result.base, base, decimal=5)
        np.testing.assert_array_almost_equal(result.tool, tool, decimal=5)
        np.testing.assert_array_almost_equal(np.vstack(result.fit_errors), np.zeros((10, 3)), decimal=5)
        np.testing.assert_array_almost_equal(np.vstack(result.model_errors), np.zeros((10, 3)), decimal=5)

        offsets = [np.array([0., 100., 200., 300., 400, 500.]), np.array([-180.0, -108.0, -36.0, 36.0, 108.0, 180.0]),
                   np.array([-200., -120., -40., 40., 120, 200.]), np.array([-200., -120., -40., 40., 120, 200.])]

        points = [np.array([[0, 0, 0], [0.004324125, 0.007919232, 100.0577353],
                            [0.00519, 0.009611275, 200.0346462], [0.00936, 0.0229328, 299.9897745],
                            [0.016288942, -0.00449079, 399.9475168], [-0.019166718, 0.01355, 499.934]]),

                  np.array([[-37.702407, -100.3246853, 0.060174943], [-72.4308, -47.72367282, 0.02528772],
                            [-33.11670555, 1.571, -0.002], [25.89742938, -20.58587784, -0.0142],
                            [23.05826241, -83.51255368, 0.032545921], [-37.6133, -100.3567116, 0.048915878]]),

                  np.array([[-0.008, -199.7817, 0.173440527], [-0.03388, -119.9040682, 0.132],
                            [-0.012866725, -40.03464456, 0.0608], [0.02147, 40.09068065, -0.003563545],
                            [-0.001905469, 120.1877634, -0.077537662], [0.0085, 200.1388357, -0.126678]]),

                  np.array([[-200.1472381, 0.04181174, 0.048689129], [-120.0620691, 0.035838747, 0.044916269],
                            [-40.039, 0.029215491, 0.015246372], [40.04469207, 0.020326262, -0.001128861],
                            [120.0471608, 0.030719316, -0.00639], [200.0948445, 0.045893343, -0.055839322]])]

        types = [Link.Type.Prismatic, Link.Type.Revolute, Link.Type.Prismatic, Link.Type.Prismatic]
        homes = [0.0, 0.0, 0.0, 0.0]
        result = circle_point_analysis(points, types, offsets, homes)

        np.testing.assert_array_almost_equal(result.joint_axes[0], [-1.59357e-05, 1.25323e-05, 1.], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_axes[1], [-2.27729e-04, -6.01415e-04, -1.], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_axes[2], [7.59974e-05,  1., -7.83437e-04], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_axes[3], [1., -1.37606e-06, -2.47331e-04], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_origins[0], [0., 0., 0.], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_origins[1], [-18.87706, -50.12061, 0.0254302], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_origins[2], [-0.0111, 0.11102, 0.03246], decimal=5)
        np.testing.assert_array_almost_equal(result.joint_origins[3], [-0.01692, 0.02885, 0.01364], decimal=5)
        base = np.identity(4)
        base[:3, 3] = [0.00665, 0.00512, -0.00605]
        np.testing.assert_array_almost_equal(result.base, base, decimal=5)
        np.testing.assert_array_almost_equal(result.tool, np.identity(4), decimal=5)
        self.assertAlmostEqual(np.vstack(result.fit_errors).max(), 0.0203117, 5)
        self.assertAlmostEqual(np.vstack(result.model_errors).max(), 0.18427, 5)

    def testDescriptionGeneration(self):
        robot_name = 'Two Link'
        joint_names = ['a', 'b']
        types = [Link.Type.Revolute, Link.Type.Prismatic]
        homes = [0, 50]
        order = [1, 0]
        base = np.array([[0, -1, 0, 10.], [0, 0, -1, 0], [1, 0, 0, 1.5], [0, 0, 0, 1]])
        tool = np.array([[0, 0, -1, 1.], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        axes = [np.array([1., 0., 0.]), np.array([0., 0., 1.])]
        origins = [np.array([0., 0., 0.]), np.array([0., 0., 1.])]
        offsets = [np.array([-180, 0., 180]), np.array([100., 50., 0.])]
        lower_limits = [-np.pi, 0]
        upper_limits = [np.pi, 100]

        desc = generate_description(robot_name, base, tool, order, joint_names, types, axes, origins, homes, offsets)
        robot = extract_positioner(desc)

        self.assertEqual(robot.name, robot_name)
        self.assertListEqual(robot.order, order)
        np.testing.assert_array_almost_equal(robot.base, base, decimal=5)
        np.testing.assert_array_almost_equal(robot.tool, tool, decimal=5)
        for index, link in enumerate(robot.links):
            self.assertEqual(link.name, joint_names[index])
            self.assertEqual(link.type, types[index])
            self.assertEqual(link.offset, homes[index])
            self.assertEqual(link.lower_limit, lower_limits[index])
            self.assertEqual(link.upper_limit, upper_limits[index])
            np.testing.assert_array_almost_equal(link.joint_axis, axes[index], decimal=5)
            np.testing.assert_array_almost_equal(link.home, origins[::-1][index], decimal=5)

        self.assertNotEqual(json.dumps(desc), '')

