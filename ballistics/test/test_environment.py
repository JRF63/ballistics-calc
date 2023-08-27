from ballistics.environment import *

import unittest

class TestEnvironmentCalcs(unittest.TestCase):
    def test_water_vapor_pressure(self):
        # Table 8.2 of Modern Exterior Ballistics
        table = [
            (-40, 0.006),   # These two does not seem to agree with
            (0, 0.045),     # the common empirical equations
            (32, 0.18),
            (59, 0.50),
            (70, 0.74),
            (100, 1.93),
            (130, 4.53)
        ]

        for temp, pwv_ref in table:
            pwv = arden_buck_equation(temp)
            self.assertAlmostEqual(pwv, pwv_ref, delta=0.05)

    def test_env_data_wrt_to_humidity(self):
        # Table 8.3 of Modern Exterior Ballistics
        table = [
            (0, 0, 1.128, 1051.0),
            (0, 50, 1.128, 1051.2),
            (0, 78, 1.128, 1051.2),
            (0, 100, 1.128, 1051.3),

            (32, 0, 1.055, 1087.0),
            (32, 50, 1.054, 1087.5),
            (32, 78, 1.053, 1087.7),
            (32, 100, 1.053, 1087.9),

            (59, 0, 1.000, 1116.45),
            (59, 50, 0.997, 1117.8),
            (59, 78, 0.995, 1118.5),
            (59, 100, 0.994, 1119.1),

            (70, 0, 0.979, 1128.2),
            (70, 50, 0.975, 1130.2),
            (70, 78, 0.972, 1131.3),
            (70, 100, 0.970, 1132.1),

            (100, 0, 0.927, 1159.7),
            (100, 50, 0.915, 1165.0),
            (100, 78, 0.909, 1167.9),
            (100, 100, 0.904, 1170.2),

            (130, 0, 0.880, 1190.4),
            (130, 50, 0.854, 1203.0),
            (130, 78, 0.840, 1210.1),
            (130, 100, 0.829, 1215.7)
        ]

        pressure_std = 29.92
        density_std = 0.0764742
        altitude = 0
        for temp, rh, density_ratio_ref, speed_ref in table:
            density = air_density(temp, pressure_std, rh, altitude)
            density_ratio = density / density_std
            speed = speed_sound(temp, rh, altitude)

            self.assertAlmostEqual(density_ratio, density_ratio_ref, delta=0.5)
            self.assertAlmostEqual(speed, speed_ref, delta=0.5)

    def test_env_data_wrt_to_elevation(self):
        # Excerpt from Table 8.1 of Modern Exterior Ballistics
        table = [
            (2000, 52.8, 27.82, 0.943),
            (5000, 43.6, 24.90, 0.862),
            (8000, 34.6, 22.23, 0.786),
            (10000, 28.7, 20.58, 0.739)
        ]

        temp_std = 59
        density_std = 0.0764742
        for altitude, temp_ref, _, density_ratio_ref in table:

            temp = temp_at_altitude(temp_std, altitude)
            
            density = air_density_at_altitude(density_std, altitude)
            density_ratio = density / density_std

            self.assertAlmostEqual(temp, temp_ref, delta=(0.2 * max(temp, temp_ref)))
            self.assertAlmostEqual(density_ratio, density_ratio_ref, delta=0.05)