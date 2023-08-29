from ballistics.trajectory import *

def main():
    pm_traj = PointMassTrajectory(parse_drag_table('ballistics/data/mcg7.txt'))
    muzzle_speed = 2970
    bc = 0.371
    sight_height = 1.5 / 12.0
    barrel_length = 0.0
    distance_of_zero = 100.0 * 3.0
    elevation_of_zero = 0.0
    max_range = 2500.0 * 3.0
    integrator_type = RungeKuttaMethodIntegrator
    integrator_type = TwoStepAdamsBashforthIntegrator
    dt = 1/60

    x0 = np.array([barrel_length, 0.0, -sight_height])
    v0 = pm_traj.solve_for_initial_velocity(
        muzzle_speed,
        bc,
        sight_height,
        barrel_length,
        distance_of_zero,
        elevation_of_zero,
        integrator_type,
        dt,
        rh=50
    )

    ts, xs, vs = pm_traj.calculate_trajectory(
        x0, v0, bc, max_range, integrator_type, dt, rh=50)

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
