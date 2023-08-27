import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy.interpolate import make_interp_spline

def numerical_solve(v0, bc, time_of_flight, cd_func):
    dt = 1 / 60

    temp = 59
    pressure = 29.92
    rh = 50
    wv_pressure = 0.50

    v_sound = 49.0223*(temp + 459.67)**0.5

    density_air_std = 0.0764742
    density_air = (pressure / 29.92) * (518.67 / (temp + 459.67)) * density_air_std

    v_sound_corr = 1 + 0.0014 * rh * wv_pressure / 29.92
    density_air_corr = 1 - 0.00378 * rh * wv_pressure / 29.92

    v_sound *= v_sound_corr
    density_air *= density_air_corr
    
    def cd_star(speed):
        m = speed / v_sound
        result = density_air * math.pi * cd_func(m) / (1152.0 * bc)
        return result
    
    ts = []
    xs = []
    vs = []

    t = 0.0
    x = np.array([0.0, 0.0, 0.0])
    v = v0.copy()
    g = np.array([0.0, 0.0, -32.17405])

    speed = linalg.norm(v)
    accel = -cd_star(speed) * speed * v + g

    while t < time_of_flight:
        # Runge-Kutta 4th order
        # x += v * dt
        # speed = linalg.norm(v)
        # k1 = -cd_star(speed) * speed * v + g
        # y = v + k1/2 * dt
        # speed = linalg.norm(y)
        # k2 = -cd_star(speed) * speed * y + g
        # y = v + k2/2 * dt
        # speed = linalg.norm(y)
        # k3 = -cd_star(speed) * speed * y + g
        # y = v + k3 * dt
        # speed = linalg.norm(y)
        # k4 = -cd_star(speed) * speed * y + g
        # v += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        # t += dt

        # Euler
        # x += v * dt
        # speed = linalg.norm(v)
        # accel = -cd_star(speed) * speed * v + g
        # v += accel * dt
        # t += dt

        # Heun
        # speed = linalg.norm(v)
        # accel = -cd_star(speed) * speed * v + g
        # v_next = v + accel * dt
        # x += (v + v_next) * dt / 2
        # v = v_next
        # t += dt

        # Verlet
        x = x + v * dt + accel * 0.5 * dt * dt
        speed = linalg.norm(v)
        new_accel = -cd_star(speed) * speed * v + g
        v = v + (accel + new_accel) * 0.5 * dt
        accel = new_accel
        t += dt

        ts.append(t)
        xs.append(x)
        vs.append(v)

    return ts, xs, vs

def parse_drag_table(filename: str):
    table = []
    with open(filename) as f:
        for line in f:
            table.append(tuple(map(float, line.strip().split())))
    return table

def main():
    g7 = parse_drag_table('data/mcg7.txt')
    cd_func = make_interp_spline(*zip(*g7), k=3)

    bc = 0.371
    muzzle_speed = 2970
    v0 = np.array([muzzle_speed, 0.0, 0.0])
    time_of_flight = 4.0

    ts, xs, vs = numerical_solve(v0, bc, time_of_flight, cd_func)
    rs = []
    ds = []
    ss = []
    for x, v in zip(xs, vs):
        rs.append(x[0])
        ds.append(x[2])
        ss.append(linalg.norm(v))

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

    bc = float('inf')
    muzzle_speed = 262.467
    angle = 2005.03 / 60
    v0 = np.array([muzzle_speed * np.cos(angle * math.pi / 180), 0.0, muzzle_speed * np.sin(angle * math.pi / 180)])
    time_of_flight = 10.0

    ts, xs, vs = numerical_solve(v0, bc, time_of_flight, cd_func)
    rs = []
    ds = []
    ss = []
    for x, v in zip(xs, vs):
        rs.append(x[0])
        ds.append(x[2])
        ss.append(linalg.norm(v))

    drop_curve = make_interp_spline(rs, ds, k=3)
    speed_curve = make_interp_spline(rs, ss, k=3)
    time_curve = make_interp_spline(rs, ts, k=3)

    print('')
    print('Vacuum trajectory')
    for x in range(0, 650, 50):
        x /= 0.3048
        print(
            '%4d, %8.2f, %8.3f, %6.3f' %
            (
                x * 0.3048,
                12 * drop_curve(x),
                speed_curve(x),
                time_curve(x)
            )
        )

main()
