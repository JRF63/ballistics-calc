from .environment import *
from .integration import *

from collections import deque

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

MAX_SIMULATION_TIME = 20.0
ACCEL_GRAVITY = np.array([0.0, 0.0, -32.17405])


def parse_drag_table(filename: str):
    table = []
    with open(filename) as f:
        for line in f:
            table.append(tuple(map(float, line.strip().split())))
    return table


class PointMassTrajectory:

    def __init__(self, table: list[(float, float)]) -> None:
        self.cd_func = make_interp_spline(*zip(*table), k=3)

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
        decel = -cd_star * speed * vw + ACCEL_GRAVITY
        return decel

    def solve_for_initial_velocity(
        self,
        muzzle_speed: float,
        bc: float,
        sight_height: float,
        barrel_length: float,
        distance_of_zero: float,
        elevation_of_zero: float,
        integrator_type: type[NumericalIntegrator],
        dt: float,
        wind: np.ndarray = np.zeros(3),
        temp: float = 59.0,
        pressure: float = 29.92,
        rh: float = 0.0,
    ) -> np.ndarray:

        MAX_CONVERGENCE_STEPS = 100
        EPSILON = 1e-5

        density_air = air_density(temp, pressure, rh, 0.0)
        v_sound = speed_sound(temp, rh, 0.0)

        def accel_func(v):
            return self.calculate_acceleration(v, v_sound, bc, density_air, wind)

        # Where the bullet trajectory starts
        x0 = np.array([barrel_length, 0.0, -sight_height])
        v0 = np.array([muzzle_speed, 0.0, 0.0])

        # Initial guess of vertical angle
        ver_angle = np.arctan(sight_height / distance_of_zero)
        ver_angle_low = ver_angle - np.radians(45)
        ver_angle_high = ver_angle + np.radians(45)

        # Solve for vertical angle
        converged = False
        for _ in range(MAX_CONVERGENCE_STEPS):
            if converged:
                break

            ver_angle = (ver_angle_low + ver_angle_high) / 2.0

            # Need to negate to match the definition of extrinsic rotation
            v_guess = Rotation.from_euler('y', -ver_angle).apply(v0)

            integrator = integrator_type(x0, v_guess, accel_func)

            x_hist = deque([x0.copy()], maxlen=4)

            for _ in np.arange(0.0, MAX_SIMULATION_TIME, dt):
                x, _ = integrator.step(accel_func, dt)
                x_hist.append(x.copy())

                if x[0] > distance_of_zero:
                    # Add one more data point for cubic spline interpolation
                    x, _ = integrator.step(accel_func, dt)
                    x_hist.append(x.copy())

                    drop_func = make_interp_spline(
                        [a[0] for a in x_hist],
                        [a[2] for a in x_hist]
                    )
                    drop = drop_func(distance_of_zero)

                    # Second zero should be attained at the specified distance
                    if abs(drop - elevation_of_zero) < EPSILON:
                        converged = True
                        break
                    elif drop > elevation_of_zero:
                        # Aiming too high
                        ver_angle_high = ver_angle
                    else:
                        # Aiming too low
                        ver_angle_low = ver_angle
                    break
            else:
                raise Exception('Unable to solve for vertical firing angle')
        else:
            raise Exception('Unable to solve for vertical firing angle')

        v0 = Rotation.from_euler('y', -ver_angle).apply(v0)

        hor_angle = 0.0
        hor_angle_left = -np.radians(45)
        hor_angle_right = np.radians(45)

        # Solve for horizontal angle
        converged = False
        for _ in range(MAX_CONVERGENCE_STEPS):
            if converged:
                break

            hor_angle = (hor_angle_left + hor_angle_right) / 2.0
            v_guess = Rotation.from_euler('z', hor_angle).apply(v0)

            integrator = integrator_type(x0, v_guess, accel_func)

            x_hist = deque([x0.copy()], maxlen=4)

            for _ in np.arange(0.0, MAX_SIMULATION_TIME, dt):
                x, _ = integrator.step(accel_func, dt)
                x_hist.append(x.copy())

                if x[0] > distance_of_zero:
                    x, _ = integrator.step(accel_func, dt)
                    x_hist.append(x.copy())

                    deflection_func = make_interp_spline(
                        [a[0] for a in x_hist],
                        [a[1] for a in x_hist]
                    )
                    deflection = deflection_func(distance_of_zero)

                    if abs(deflection) < EPSILON:
                        converged = True
                        break
                    elif deflection > 0.0:
                        # Aiming too far to the right
                        hor_angle_right = hor_angle
                    else:
                        # Aiming too far to the left
                        hor_angle_left = hor_angle
                    break
            else:
                raise Exception('Unable to solve for horizontal firing angle')

        else:
            raise Exception('Unable to solve for horizontal firing angle')

        v0 = Rotation.from_euler('z', hor_angle).apply(v0)
        return v0

    def calculate_trajectory(
        self,
        x0: np.ndarray,
        v0: np.ndarray,
        bc: float,
        max_range: float,
        integrator_type: type[NumericalIntegrator],
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

        integrator = integrator_type(x0, v0, accel_func)
        for t in np.arange(dt, MAX_SIMULATION_TIME, dt):
            x, v = integrator.step(accel_func, dt)

            ts.append(t)
            xs.append(x.copy())
            vs.append(v.copy())

            if x[0] > max_range:
                break

        return ts, xs, vs
