import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy.interpolate import make_interp_spline

G7 = [
    (0.00, 0.1198),
    (0.05, 0.1197),
    (0.10, 0.1196),
    (0.15, 0.1194),
    (0.20, 0.1193),
    (0.25, 0.1194),
    (0.30, 0.1194),
    (0.35, 0.1194),
    (0.40, 0.1193),
    (0.45, 0.1193),
    (0.50, 0.1194),
    (0.55, 0.1193),
    (0.60, 0.1194),
    (0.65, 0.1197),
    (0.70, 0.1202),
    (0.725, 0.1207),
    (0.75, 0.1215),
    (0.775, 0.1226),
    (0.80, 0.1242),
    (0.825, 0.1266),
    (0.85, 0.1306),
    (0.875, 0.1368),
    (0.90, 0.1464),
    (0.925, 0.1660),
    (0.95, 0.2054),
    (0.975, 0.2993),
    (1.0, 0.3803),
    (1.025, 0.4015),
    (1.05, 0.4043),
    (1.075, 0.4034),
    (1.10, 0.4014),
    (1.125, 0.3987),
    (1.15, 0.3955),
    (1.20, 0.3884),
    (1.25, 0.3810),
    (1.30, 0.3732),
    (1.35, 0.3657),
    (1.40, 0.3580),
    (1.50, 0.3440),
    (1.55, 0.3376),
    (1.60, 0.3315),
    (1.65, 0.3260),
    (1.70, 0.3209),
    (1.75, 0.3160),
    (1.80, 0.3117),
    (1.85, 0.3078),
    (1.90, 0.3042),
    (1.95, 0.3010),
    (2.00, 0.2980),
    (2.05, 0.2951),
    (2.10, 0.2922),
    (2.15, 0.2892),
    (2.20, 0.2864),
    (2.25, 0.2835),
    (2.30, 0.2807),
    (2.35, 0.2779),
    (2.40, 0.2752),
    (2.45, 0.2725),
    (2.50, 0.2697),
    (2.55, 0.2670),
    (2.60, 0.2643),
    (2.65, 0.2615),
    (2.70, 0.2588),
    (2.75, 0.2561),
    (2.80, 0.2533),
    (2.85, 0.2506),
    (2.90, 0.2479),
    (2.95, 0.2451),
    (3.00, 0.2424),
    (3.10, 0.2368),
    (3.20, 0.2313),
    (3.30, 0.2258),
    (3.40, 0.2205),
    (3.50, 0.2154),
    (3.60, 0.2106),
    (3.70, 0.2060),
    (3.80, 0.2017),
    (3.90, 0.1975),
    (4.00, 0.1935),
    (4.20, 0.1861),
    (4.40, 0.1793),
    (4.60, 0.1730),
    (4.80, 0.1672),
    (5.00, 0.1618)
]

G7_OLD = [
    (0.0, 0.120),
    (0.5, 0.119),
    (0.6, 0.119),
    (0.7, 0.120),
    (0.8, 0.124),
    (0.9, 0.146),
    (0.95, 0.205),
    (1.0, 0.380),
    (1.05, 0.404),
    (1.1, 0.401),
    (1.2, 0.388),
    (1.3, 0.373),
    (1.4, 0.358),
    (1.5, 0.344),
    (1.6, 0.332),
    (1.8, 0.312),
    (2.0, 0.298),
    (2.2, 0.286),
    (2.5, 0.270),
    (3.0, 0.242),
    (3.5, 0.215),
    (4.0, 0.194)
]

def cd(a, b, c, m):
    smooth = 0.5 + 0.5 * np.tanh(c * (m - 1))
    # return a * (1 - smooth) + b / m**0.5 * smooth
    t1 = a
    # t2 = b / m**0.5
    b1 = 0
    b2 = -0.04
    t2 = b / (m + b1) ** 0.5 + b2
    return t1 * (1 - smooth) + t2 * smooth

def siacci(v):
    return 1/v**2 * 0.896 * (
            0.284746 * v
            - 224.221
            + ((0.234396 * v -  223.754)**2 + 209.043)**0.5
            + (0.019161 * v * (v - 984.261))/(371 + (v/656.174)**10) )

def siacci2(v):
    s = 1e-10
    if v > 2600:
        a = 7.6090480
        p = 1.55
    elif v > 1800:
        a = 7.0961978
        p = 1.7
    elif v > 1370:
        a = 6.1192596
        p = 2
    elif v > 1230:
        a = 2.9809023
        p = 3
    elif v > 970:
        a = 6.8018712
        p = 5
        s = 1e-20
    elif v > 790:
        a = 2.7734430
        p = 3
    else:
        a = 5.6698914
        p = 2
    return (a * s) * v**p

def siacci1(vp):
    if (vp > 4200):
        a = 1.29081656775919e-09
        p = 3.24121295355962
    elif (vp > 3000):
        a = 0.0171422231434847
        p = 1.27907168025204
    elif (vp > 1470):
        a = 2.33355948302505e-03
        p = 1.52693913274526
    elif (vp > 1260):
        a = 7.97592111627665e-04
        p = 1.67688974440324
    elif (vp > 1110):
        a = 5.71086414289273e-12
        p = 4.3212826264889
    elif (vp > 960):
        a = 3.02865108244904e-17 
        p = 5.99074203776707
    elif (vp > 670):
        a = 7.52285155782535e-06
        p = 2.1738019851075
    elif (vp > 540):
        a = 1.31766281225189e-05
        p = 2.08774690257991
    else:
        a = 1.34504843776525e-05
        p = 2.08702306738884
    return a * vp**p

def numerical_solve(v0, bc, time_of_flight, cd_func):
    dt = 1 / 600

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

    x = np.array([0.0, 0.0, 0.0])
    g = np.array([0.0, 0.0, -32.17405])
    v = v0.copy()

    t = 0.0
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
        speed = linalg.norm(v)
        accel = -cd_star(speed) * speed * v + g
        v_next = v + accel * dt
        x += (v + v_next) * dt / 2
        v = v_next
        t += dt

        
    return t, x, v

def find_angle_at_zero_drop(muzzle_speed, bc, zero_range, cd_func):
    ITERATIONS = 50
    high = 45.0 * math.pi / 180.0
    low = 0.0
    angle = 0.0

    for _i in range(ITERATIONS):
        angle = (high + low) / 2.0
        test_v = np.array([muzzle_speed * np.cos(angle), 0.0, muzzle_speed * np.sin(angle)])
        x, _v = numerical_solve(test_v, bc, zero_range, cd_func)
        drop = x[2]
        if drop > 0.0:
            high = angle
        else:
            low = angle
    return angle

def main():
    # m = np.linspace(0.01, 5, 100)
    # a = 0.1198
    # b = 0.49
    # c = 20
    # y = cd(a, b, c, m)

    cd_func = make_interp_spline(*zip(*G7), k=3)
    # cd_func = lambda x: np.interp(x, *zip(*G7))

    # bc = 0.371
    # muzzle_speed = 2970
    # mass = 62
    # zero_range = 100 # feet
    # dist = 2000 # feet

    # angle = find_angle_at_zero_drop(muzzle_speed, bc, zero_range * 3, cd_func)
    # print(f'Angle: {angle * 180 / math.pi}')
    # v0 = np.array([muzzle_speed * np.cos(angle), 0.0, muzzle_speed * np.sin(angle)])

    bc = 0.371
    muzzle_speed = 2970
    v0 = np.array([muzzle_speed, 0.0, 0.0])

    # x, v = numerical_solve(v0, bc, dist * 3, cd_func)
    # print('%4d, %8.2f, %4.0f' % (x[0] / 3, 12 * x[2], linalg.norm(v)))

    for i in (0.10, 0.21, 0.70, 1.29, 2.46, 3.50):
        t, x, v = numerical_solve(v0, bc, i, cd_func)
        print('%4d, %8.2f, %5f, %6.3f' % (x[0] / 3, 12 * x[2], linalg.norm(v), t))

    # bc = float('inf')
    # muzzle_speed = 262.467
    # angle = 2005.03 / 60
    # v0 = np.array([muzzle_speed * np.cos(angle * math.pi / 180), 0.0, muzzle_speed * np.sin(angle * math.pi / 180)])
    # for i in (0.749, 3.744, 6.739, 8.985):
    #     t, x, v = numerical_solve(v0, bc, i, cd_func)
    #     print('%4d, %8.2f, %4.2f, %4.2f, %4.2f, %6.3f' % (x[0] * 0.3048, 12 * x[2], v[0], v[1], v[2], t))

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

main()
