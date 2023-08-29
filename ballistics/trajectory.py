from environment import *
from integration import *

import numpy as np
from scipy.interpolate import make_interp_spline


MAX_SIMULATION_TIME = 20.0

def parse_drag_table(filename: str):
    table = []
    with open(filename) as f:
        for line in f:
            table.append(tuple(map(float, line.strip().split())))
    return table


class PointMassTrajectory:
    
    def __init__(self, integrator: type[NumericalIntegrator]) -> None:
        g1 = parse_drag_table('data/mcg1.txt')
        g7 = parse_drag_table('data/mcg7.txt')

        self.g1_curve = make_interp_spline(*zip(*g1), k=3)
        self.g7_curve = make_interp_spline(*zip(*g7), k=3)
        self.cd_func = self.g7_curve

        self.integrator = integrator

        self.g = np.array([0.0, 0.0, -32.17405])

    def calculate_acceleration(
        self,
        v: np.ndarray,
        v_sound: float,
        bc: float,
        density_air: float,
        wind: np.ndarray
    ) -> np.ndarray:

        # 8 * 144, the 144 comes from converting in2 to ft2
        k = 1152.0

        vw = v - wind
        speed = np.linalg.norm(vw)
        m = speed / v_sound
        cd_star = density_air * np.pi * self.cd_func(m) / (k * bc)
        decel = -cd_star * speed * vw + self.g
        return decel

    def firing_angle_at_zero(
        self,
        muzzle_speed: float,
        bc: float,
        sight_height: float,
        barrel_length: float,
        distance_of_zero: float,
        elevation_of_zero: float,
        dt: float,
        wind: np.ndarray = np.zeros(3),
        temp: float = 59.0,
        pressure: float = 29.92,
        rh: float = 0.0,
    ):
        
        NUM_CONVERGENCE_STEPS = 100

        density_air = air_density(temp, pressure, rh, 0.0)
        v_sound = speed_sound(temp, rh, 0.0)

        def accel_func(v):
            return self.calculate_acceleration(v, v_sound, bc, density_air, wind)
        
        # Where the bullet trajectory starts
        x0 = np.array([barrel_length, 0.0, -sight_height])

        # Initial guess of vertical angle
        angle = np.arctan(sight_height / distance_of_zero)

        vert_angle_low = angle - np.radians(45)
        vert_angle_high = angle + np.radians(45)

        # Solve for vertical angle
        for _ in range(NUM_CONVERGENCE_STEPS):
            angle = (vert_angle_low + vert_angle_high) / 2.0
            
            v0 = np.array([muzzle_speed * np.cos(angle),
                          0.0, muzzle_speed * np.sin(angle)])

            integrator = self.integrator(x0, v0, accel_func)

            x_prev = x0.copy()

            for _ in np.arange(0.0, MAX_SIMULATION_TIME, dt):
                x, _ = integrator.step(accel_func, dt)
                if x[0] > distance_of_zero:

                    drop = np.interp(
                        distance_of_zero,
                        [x_prev[0], x[0]],
                        [x_prev[2], x[2]]
                    )

                    if drop > elevation_of_zero:
                        # Aiming too high
                        vert_angle_high = angle
                    else:
                        # Aiming too low
                        vert_angle_low = angle
                    break
                x_prev = x.copy()
            else:
                # Did not reach the zero target, either angle too high or bullet is not fast enough
                vert_angle_high = angle

        return angle

    def calculate_trajectory(
        self,
        x0: np.ndarray,
        v0: np.ndarray,
        bc: float,
        max_range: float,
        dt: float,
        wind: np.ndarray = np.zeros(3),
        temp: float = 59.0,
        pressure: float = 29.92,
        rh: float = 0.0,
    ) -> (list[float], list[np.ndarray], list[np.ndarray]):
        
        density_air = air_density(temp, pressure, rh, 0.0)
        v_sound = speed_sound(temp, rh, 0.0)

        def accel_func(v):
            return self.calculate_acceleration(v, v_sound, bc, density_air, wind)
        
        ts = [0.0]
        xs = [x0.copy()]
        vs = [v0.copy()]
        
        integrator = self.integrator(x0, v0, accel_func)
        for t in np.arange(dt, MAX_SIMULATION_TIME, dt):
            x, v = integrator.step(accel_func, dt)

            ts.append(t)
            xs.append(x.copy())
            vs.append(v.copy())

            if x[0] > max_range:
                break

        return ts, xs, vs


def main():
    pm_traj = PointMassTrajectory(RungeKuttaMethodIntegrator)
    muzzle_speed = 2970
    bc = 0.371
    sight_height = 1.5 / 12.0
    barrel_length = 0.0
    distance_of_zero = 100.0 * 3.0
    elevation_of_zero = 0.0
    max_range = 2500.0 * 3.0
    dt = 1/60
    
    angle = pm_traj.firing_angle_at_zero(
        muzzle_speed,
        bc,
        sight_height,
        barrel_length,
        distance_of_zero,
        elevation_of_zero,
        dt,
        rh = 50
    )


    x0 = np.array([barrel_length, 0.0, -sight_height])
    v0 = np.array([muzzle_speed * np.cos(angle), 0.0, muzzle_speed * np.sin(angle)])

    # def sight_line(x):
    #     return np.interp(x, [0.0, 0.0], [0.0, elevation_of_zero])
    
    ts, xs, vs = pm_traj.calculate_trajectory(x0, v0, bc, max_range, dt, rh=50)

    rs = []
    ds = []
    ss = []
    for x, v in zip(xs, vs):
        rs.append(x[0])
        ds.append(x[2])
        ss.append(np.linalg.norm(v))

    drop_curve = make_interp_spline(rs, ds, k=3)
    speed_curve = make_interp_spline(rs, ss, k=3)
    time_curve = make_interp_spline(rs, ts, k=3)

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

    print(f'G7 bullet, BC = {bc} lb/in2, Muzzle speed = {muzzle_speed} ft/s')
    for x in (100, 200, 600, 1000, 1600, 2000):
        x *= 3
        print(
            '%4d, %8.2f, %8.3f, %6.3f' %
            (
                x / 3,
                12 * drop_curve(x),
                speed_curve(x),
                time_curve(x)
            )
        )

    


if __name__ == '__main__':
    main()
