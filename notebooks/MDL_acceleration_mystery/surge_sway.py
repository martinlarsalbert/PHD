import numpy as np


def surge_sway(t, x0, y0, psi, **kwargs):
    """
    Compute surge and sway based on input coordinates and heading.

    Parameters:
    t      : array-like, time [s]
    x0     : array-like, x-coordinate [m]
    y0     : array-like, y-coordinate [m]
    psi    : array-like, heading [rad]

    Returns:
    surge  : ndarray, surge displacement [m]
    sway   : ndarray, sway displacement [m]
    """
    # Convert inputs to numpy arrays
    t = np.asarray(t)
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    psi = np.asarray(psi)

    # Shift x0 so it starts at zero
    x0 = x0 - x0[0]

    # Mean heading
    psi_mean = np.mean(psi)

    # Transform coordinates to surge-sway frame
    x = x0 * np.cos(psi_mean) + y0 * np.sin(psi_mean)
    y = y0 * np.cos(psi_mean) - x0 * np.sin(psi_mean)

    # Compute mean velocities
    velx = np.max(x) / np.max(t)
    vely = np.max(y) / np.max(t)

    # Deviations from mean path
    x = x - velx * t
    y = y - vely * t

    surge = x
    sway = y

    return surge, sway
