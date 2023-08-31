from ballistics.trajectory import *

import numpy as np

def main():
    for method in ('DOP853', 'LSODA', 'BeemansAlgorithm'):
        pm_traj = PointMassTrajectory(parse_drag_table('ballistics/data/mcg7.txt'))
        muzzle_speed = 2970
        bc = 0.371
        sight_height = 1.5 / 12.0
        barrel_length = 0.0
        zero_range = 100.0 * 3.0
        zero_elevation = 0.0
        wind_speed = 10 * 5280 / 3600 # mph
        wind = wind_speed * np.array([0.0, 1.0, 0.0])
        rh = 50
        ranges = [x * 3.0 for x in (100, 200, 600, 1000, 1600, 2000)]
        
        x0 = np.array([barrel_length, 0.0, -sight_height])
        ver_angle, hor_angle = pm_traj.solve_for_initial_velocity(
            x0,
            muzzle_speed,
            bc,
            zero_range,
            zero_elevation,
            wind=wind,
            rh=rh,
            method=method
        )
        
        v0 = muzzle_speed * np.array([
            np.cos(ver_angle) * np.cos(hor_angle),
            np.sin(hor_angle),
            np.sin(ver_angle) * np.cos(hor_angle)
        ])

        result = pm_traj.calculate_trajectory(
            x0,
            v0,
            bc,
            wind=wind,
            rh=rh,
            method=method,
            ranges=ranges
        )

        print(f'{method}\nG7 bullet, BC = {bc} lb/in2, Muzzle speed = {muzzle_speed} ft/s')
        for t, y in zip(result.t_events, result.y_events):
            distance = y[0, 0] / 3
            drop = 12 * y[0, 2]
            windage = 12 * y[0, 1]
            speed = np.linalg.norm(y[0][3:])
            print(
                '%4.0f, %8.2f, %8.2f, %8.3f, %6.3f' %
                (
                    distance,
                    drop,
                    windage,
                    speed,
                    t
                )
            )
        print('')
        
    # shooterscalculator
    #  100,     0.00, 2843, 0.10
    #  200,    -2.80, 2720, 0.21
    #  600,   -65.78, 2255, 0.70
    # 1000,  -239.87, 1836, 1.29
    # 1600,  -844.97, 1291, 2.46
    # 2000, -1646.26, 1056, 3.50

    # Hornady
    #  200,    -2.80, 2717
    #  600,      -66, 2246
    # 1000,   -241.3, 1823
    # 1600,   -853.9, 1273
    # 2000,  -1670.2, 1049


if __name__ == '__main__':
    main()
