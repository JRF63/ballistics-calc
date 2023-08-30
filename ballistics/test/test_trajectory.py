from ballistics.trajectory import *
from ballistics.integration import *

import unittest

import numpy as np


class TestTrajectoryCalcs(unittest.TestCase):
    def test_vacuum_trajectory(self):
        'Example 8.1 of Modern Exterior Ballistics'

        pm_traj = PointMassTrajectory(
            parse_drag_table('ballistics/data/mcg7.txt'))
        muzzle_speed = 80 / 0.3048  # 80 m/s
        angle = np.radians(2005.03 / 60)
        bc = float('inf')
        max_range = 600  # 600 meters
        ranges = [x / 0.3048 for x in range(0, max_range + 10, 10)]

        x0 = np.zeros(3)
        v0 = np.array([muzzle_speed * np.cos(angle),
                      0.0, muzzle_speed * np.sin(angle)])

        result = pm_traj.calculate_trajectory(x0, v0, bc, ranges=ranges)

        def analytical_pos(t): return v0 * t + ACCEL_GRAVITY / 2.0 * t**2
        def analytical_vel(t): return v0 + ACCEL_GRAVITY * t

        for t, y in zip(result.t_events, result.y_events):
            pos = y[0][:3]
            pos_diff = analytical_pos(t) - pos
            pos_residual = np.linalg.norm(pos_diff)

            vel = y[0][3:]
            vel_diff = analytical_vel(t) - vel
            vel_residual = np.linalg.norm(vel_diff)

            self.assertAlmostEqual(pos_residual, 0.0, places=5)
            self.assertAlmostEqual(vel_residual, 0.0, places=5)
