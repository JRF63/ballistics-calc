import numpy as np

def arden_buck_equation(temp: float) -> float:
    """Calculates the vapor pressure of water using the Arden Buck
    equation.
    https://en.wikipedia.org/wiki/Arden_Buck_equation

    Parameters
    ----------
    temp : float
        Temperature in Fahrenheit

    Returns
    -------
    pwv : float
        The saturation vapor pressure [inHg] at the given temperature
    """

    temp_celsius = (temp - 32.0) * 5.0 / 9.0
    if temp > 0:
        a = 6.1121
        b = 18.678
        c = 234.5
        d = 257.14
    else:
        a = 6.1115
        b = 23.036
        c = 333.7
        d = 279.82
    pwv = a * np.exp((b - temp_celsius/c) *
                     (temp_celsius/(d + temp_celsius)))
    result = pwv / 33.7685  # Convert hPa to inHg
    return result


def temp_at_altitude(temp: float, altitude: float) -> float:
    """Calculates the temperature above the initial point. Values are based upon the ICAO
    Atmosphere.

    Parameters
    ----------
    temp : float
        Temperature in Fahrenheit
    altitude : float
        Altitude above the point of measurement in ft

    Returns
    -------
    ty : float
        Temperature in degF at the specified altitude
    """

    # Equation 8.10 of Modern Exterior Ballistics
    k = 6.858e-6 + 2.776e-11 * altitude
    ty = (temp + 459.67) * np.exp(-k * altitude) - 459.67

    return ty


def air_density_at_altitude(density: float, altitude: float) -> float:
    """Calculates the air density at the altitude given the current air density. Values are based
    upon the ICAO Atmosphere.

    Parameters
    ----------
    density : float
        Air density in lb/ft3
    altitude : float
        Altitude above the point of measurement in ft

    Returns
    -------
    dy : float
        Air density in lb/ft3 at the specified altitude
    """

    # Equation 8.15 of Modern Exterior Ballistics
    h = 2.926e-5 + 1e-10 * altitude
    dy = density * np.exp(-h * altitude)

    return dy


def air_density(temp: float, pressure: float, rh: float, altitude: float) -> float:
    """Calculates the air density. Assumes the ICAO Atmosphere is being used.

    Parameters
    ----------
    temp : float
        Temperature in Fahrenheit
    pressure : float
        Air pressure in inHg
    rh : float
        Percent relative humidity
    altitude : float
        Altitude above the point of measurement in ft

    Returns
    -------
    density : float
        Air density in lb/ft3
    """

    density_std = 0.0764742
    density = (pressure / 29.92) * (518.67 / (temp + 459.67)) * density_std
    pwv = arden_buck_equation(temp)

    # Equation 8.24 of Modern Exterior Ballistics
    correction_factor = 1 - 0.00378 * rh * pwv / 29.92
    density *= correction_factor

    density = air_density_at_altitude(density, altitude)

    return density


def speed_sound(temp: float, rh: float, altitude: float) -> float:
    """Calculates the speed of sound. Assumes the ICAO Atmosphere is being used.

    Parameters
    ----------
    temp : float
        Temperature in Fahrenheit
    rh : float
        Percent relative humidity
    altitude : float
        Altitude above the point of measurement in ft

    Returns
    -------
    speed : float
        Speed of sound in ft/s
    """

    ty = temp_at_altitude(temp, altitude)

    speed = 49.0223*(ty + 459.67)**0.5
    pwv = arden_buck_equation(ty)

    # Equation 8.26 of Modern Exterior Ballistics
    correction_factor = 1 + 0.0014 * rh * pwv / 29.92
    speed *= correction_factor

    return speed