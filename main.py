from ballistics.trajectory import *

import numpy as np
import pandas as pd
import streamlit as st

st.title('Ballistic calculator')

# Inputs to the calculator
muzzle_speed = st.number_input('Muzzle speed (ft/s)', 500.0, 10000.0, value=3000.0)
ballistic_coefficient = st.number_input('Ballistic coefficient (lbm/in2)', 0.01, 10.0, value=0.3)
drag_model = st.selectbox('Drag model', ['G1', 'G7'], index=1)
sight_height = st.number_input('Sight height (in)', 0.0, 24.0, value=1.5) / 12.0
barrel_length = st.number_input('Barrel length (in)', 0.0, 60.0, value=18.0) / 12.0
zero_range = st.number_input('Zero range (yd)', 0.0, 10000.0, value=100.0) * 3.0
zero_elevation = st.number_input('Zero elevation (yd)', -100.0, 100.0, value=0.0) * 3.0
wind_speed = st.number_input('Wind speed (mph)', 0.0, 20.0, value=10.0) * 5280 / 3600
wind_angle = st.number_input('Wind angle (deg)', 0.0, 360.0, value=90.0)

temperature = st.number_input('Temperature (F)', -20.0, 120.0, value=59.0)
pressure = st.number_input('Pressure (in. Hg)', 15.0, 50.0, value=29.92)
relative_humidity = st.number_input('Relative humidity (RH)', 0.0, 100.0, value=50.0)

max_range = st.number_input('Max range (yd)', 100.0, 10000.0, value=2000.0)
range_interval = st.number_input('Range intervals (yd)', 10.0, 1000.0, value=100.0)


if st.button('Calculate'):
    # DOP853 is often the fastest for a given accuracy target
    method = 'DOP853'

    drag_table = 'ballistics/data/mcg1.txt' if drag_model == 'G1' else 'ballistics/data/mcg7.txt'
    pm_traj = PointMassTrajectory(parse_drag_table(drag_table))

    x0 = np.array([barrel_length, 0.0, -sight_height])
    wind = wind_speed * np.array([np.cos(np.radians(wind_angle)), np.sin(np.radians(wind_angle)), 0.0])

    # Intentionally exclude distance == 0.0 to prevent NaNs
    ranges = list(3.0 * np.arange(range_interval, max_range + range_interval, range_interval))

    ver_angle, hor_angle = pm_traj.solve_for_initial_velocity(
        x0,
        muzzle_speed,
        ballistic_coefficient,
        zero_range,
        zero_elevation,
        wind=wind,
        temp=temperature,
        pressure=pressure,
        rh=relative_humidity,
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
        ballistic_coefficient,
        wind=wind,
        temp=temperature,
        pressure=pressure,
        rh=relative_humidity,
        method=method,
        ranges=ranges
    )

    events = [result.sol(0.0).T] # First row at time == 0.0
    events.extend(result.y_events)

    # Combine the list into one array
    data = np.vstack(events)

    # Calculate minute of angle from coordinates
    # Ugly use of ufuncs is to avoid divide by zero
    windage_ratio = np.divide(data[:,1], data[:,0], out=np.zeros_like(data[:,1]), where=(data[:,0] != 0))
    drop_ratio = np.divide(data[:,2], data[:,0], out=np.zeros_like(data[:,1]), where=(data[:,0] != 0))
    windage_moa = np.arctan(windage_ratio, out=np.zeros_like(windage_ratio), where=(windage_ratio != 0)) * 10800.0 / np.pi
    drop_moa = np.arctan(drop_ratio, out=np.zeros_like(drop_ratio), where=(drop_ratio != 0)) * 10800.0 / np.pi

    # Convert ft to yd
    data[:,0] /= 3
    # Convert ft to in
    data[:,1:3] *= 12
    # Combine velocities to speed
    speed = np.linalg.norm(data[:,3:], axis=1)

    columns = ['Range (yd)', 'Windage (in)', 'Drop (in)']
    df = pd.DataFrame(data[:,:3], columns=columns)
    
    time = [0.0]
    time.extend(result.t_events)
    df.insert(3, 'Time (s)', np.vstack(time))
    df.insert(3, 'Speed (ft/s)', speed)
    df.insert(3, 'Drop (MOA)', drop_moa)
    df.insert(2, 'Windage (MOA)', windage_moa)

    st.subheader('Trajectory table')
    st.dataframe(df, use_container_width=True)

    t_last = float(result.t_events[-1][0])
    t = np.linspace(0.0, t_last, 1000)

    # Output of `OdeSolution` is transposed so transpose it back
    sol = result.sol(t).T
    # ft to yd
    sol[:,0] /= 3
    # ft to in
    sol[:,2] *= 12
    
    plot_df = pd.DataFrame(sol[:,:3], columns=columns)

    st.subheader('Bullet drop')
    st.line_chart(plot_df, x=columns[0], y=columns[2])