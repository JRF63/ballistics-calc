from ballistics.trajectory import *
from ballistics.integration import *

import unittest

import numpy as np
from scipy.interpolate import make_interp_spline


class TestTrajectoryCalcs(unittest.TestCase):
    def test_vacuum_trajectory(self):
        'Example 8.1 of Modern Exterior Ballistics'

        pm_traj = PointMassTrajectory(
            parse_drag_table('ballistics/data/mcg7.txt'))
        muzzle_speed = 80 / 0.3048  # 80 m/s
        angle = 2005.03 / 60
        bc = float('inf')
        max_range = 700 / 0.3048  # 650 meters
        integrator_type = RungeKuttaMethodIntegrator
        dt = 1/60

        x0 = np.zeros(3)
        v0 = np.array([muzzle_speed * np.cos(angle),
                      0.0, muzzle_speed * np.sin(angle)])

        ts, xs, vs = pm_traj.calculate_trajectory(
            x0, v0, bc, max_range, integrator_type, dt)
        pos_curve = make_interp_spline(ts, xs)
        vel_curve = make_interp_spline(ts, vs)

        def analytical_pos(t): return v0 * t + ACCEL_GRAVITY / 2.0 * t**2
        def analytical_vel(t): return v0 + ACCEL_GRAVITY * t

        for t in np.arange(0.0, 9.0, dt):
            pos_diff = analytical_pos(t) - pos_curve(t)
            pos_residual = np.linalg.norm(pos_diff)

            vel_diff = analytical_vel(t) - vel_curve(t)
            vel_residual = np.linalg.norm(vel_diff)

            self.assertAlmostEqual(pos_residual, 0.0, places=5)
            self.assertAlmostEqual(vel_residual, 0.0, places=5)
