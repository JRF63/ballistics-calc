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

    def test_trajectory_on_calm_winds(self):
        EPSILON = 1e-3

        # JBM Ballistics result
        reference = {
            0: (-1.5, 3000.0, 0.000),
            100: (-3.5, 2806.5, 0.103),
            200: (-10.0, 2621.3, 0.214),
            300: (-21.5, 2443.6, 0.333),
            400: (-38.8, 2272.8, 0.460),
            500: (-62.9, 2108.8, 0.597),
            600: (-94.8, 1951.8, 0.745),
            700: (-135.8, 1802.3, 0.905),
            800: (-187.6, 1661.0, 1.078),
            900: (-252.0, 1529.2, 1.266),
            1000: (-331.3, 1408.3, 1.471),
            1100: (-428.2, 1299.9, 1.693),
            1200: (-545.7, 1205.9, 1.933),
            1300: (-687.1, 1128.2, 2.191),
            1400: (-855.9, 1066.2, 2.465),
            1500: (-1055.2, 1016.6, 2.753),
            1600: (-1288.3, 975.7, 3.055),
            1700: (-1558.1, 940.7, 3.370),
            1800: (-1867.4, 909.8, 3.695),
            1900: (-2219.0, 882.0, 4.032),
            2000: (-2615.8, 856.6, 4.379)
        }

        pm_traj = PointMassTrajectory(
            parse_drag_table('ballistics/data/mcg1.txt'))
        muzzle_speed = 3000.0
        bc = 0.5
        sight_height = 1.5 / 12.0
    
        max_range = 2000  # 200 yards
        ranges = [3.0 * x for x in range(0, max_range + 100, 100)]

        method = 'DOP853'
        
        x0 = np.array([0.0, 0.0, -sight_height])
        v0 = np.array([muzzle_speed, 0.0, 0.0])

        result = pm_traj.calculate_trajectory(
            x0,
            v0,
            bc,
            method=method,
            ranges=ranges
        )

        for t, y in zip(result.t_events, result.y_events):
            distance = round(y[0, 0] / 3)
            drop = round(12.0 * y[0, 2], 1)
            speed = round(np.linalg.norm(y[0][3:]), 1)
            t = round(t[0], 3)
            
            drop_ref, speed_ref, t_ref = reference[distance]

            self.assertAlmostEqual(drop, drop_ref, delta=EPSILON*abs(drop_ref))
            self.assertAlmostEqual(speed, speed_ref, delta=EPSILON*abs(speed_ref))
            self.assertAlmostEqual(t, t_ref, delta=EPSILON*abs(t_ref))

    def test_trajectory_on_low_winds(self):
        EPSILON = 1e-3

        # JBM Ballistics result
        reference = {
            0: (-1.5, 0.0, 3000.0, 0.000),
            100: (-3.5, 0.6, 2806.5, 0.103),
            200: (-10.0, 2.5, 2621.3, 0.214),
            300: (-21.5, 5.7, 2443.6, 0.333),
            400: (-38.8, 10.5, 2272.8, 0.460),
            500: (-62.9, 17.1, 2108.8, 0.597),
            600: (-94.8, 25.5, 1951.8, 0.745),
            700: (-135.8, 36.0, 1802.3, 0.905),
            800: (-187.6, 49.0, 1661.0, 1.078),
            900: (-252.0, 64.5, 1529.3, 1.266),
            1000: (-331.3, 82.9, 1408.3, 1.471),
            1100: (-428.2, 104.4, 1299.9, 1.693),
            1200: (-545.7, 129.0, 1205.9, 1.933),
            1300: (-687.1, 156.7, 1128.2, 2.191),
            1400: (-855.9, 187.4, 1066.2, 2.465),
            1500: (-1055.2, 220.6, 1016.7, 2.754),
            1600: (-1288.4, 256.2, 975.8, 3.056),
            1700: (-1558.1, 293.9, 940.8, 3.370),
            1800: (-1867.4, 333.6, 909.9, 3.695),
            1900: (-2219.1, 375.2, 882.1, 4.032),
            2000: (-2615.8, 418.7, 856.6, 4.379)
        }

        pm_traj = PointMassTrajectory(
            parse_drag_table('ballistics/data/mcg1.txt'))
        muzzle_speed = 3000.0
        bc = 0.5
        sight_height = 1.5 / 12.0
        wind_speed = 10 * 5280 / 3600 # mph
        wind = wind_speed * np.array([0.0, 1.0, 0.0])
        
        max_range = 2000  # 200 yards
        ranges = [3.0 * x for x in range(0, max_range + 100, 100)]

        method = 'DOP853'
        
        x0 = np.array([0.0, 0.0, -sight_height])
        v0 = np.array([muzzle_speed, 0.0, 0.0])

        result = pm_traj.calculate_trajectory(
            x0,
            v0,
            bc,
            wind=wind,
            method=method,
            ranges=ranges
        )

        for t, y in zip(result.t_events, result.y_events):
            distance = round(y[0, 0] / 3)
            windage = round(12.0 * y[0, 1], 1)
            drop = round(12.0 * y[0, 2], 1)
            speed = round(np.linalg.norm(y[0][3:]), 1)
            t = round(t[0], 3)
            
            drop_ref, windage_ref, speed_ref, t_ref = reference[distance]

            self.assertAlmostEqual(drop, drop_ref, delta=EPSILON*abs(drop_ref))
            self.assertAlmostEqual(windage, windage_ref, delta=EPSILON*abs(windage_ref))
            self.assertAlmostEqual(speed, speed_ref, delta=EPSILON*abs(speed_ref))
            self.assertAlmostEqual(t, t_ref, delta=EPSILON*abs(t_ref))