import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _get_corners(angle: float, df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    angle : float
        zigzag angle [rad]
    df : pd.DataFrame
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """
    angle = np.abs(angle)
    mask = (df["delta"].abs() - angle).abs() < np.deg2rad(0.3)
    slices = df.loc[mask]

    mask = slices["delta"].diff().abs() > angle
    corners = slices.loc[mask]
    corners = pd.concat((corners,df.iloc[[-1]],),axis=0)

    assert len(corners) > 0, "Could not find any corners in the rudder signal"

    return corners


def _get_maximums(df: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    maximums = pd.DataFrame()
    for i in range(len(corners) - 1):
        index1 = corners.index[i]
        index2 = corners.index[i + 1]
        slice = df.loc[index1:index2]
        if slice['delta'].mean() > 0:
            index = slice["psi"].idxmax()
        else:
            index = slice["psi"].idxmin()
        maximum = slice.loc[[index]]
        maximums = pd.concat((maximums,maximum))

    return maximums


def get_overshoots(
    angle: float, df: pd.DataFrame, heading_deviation=None
) -> pd.DataFrame:
    """Get the overshoot angles from a simulation

    Parameters
    ----------
    angle : float
        ZigZag angle: zigzag(10)/10 in [rad]
    df : pd.DataFrame
        Simulation results as a pandas DataFrame with time [s] as index
        Must contain: "psi" and "delta"
    heading_deviation : float, default None --> angle
        heading deviation [rad]

    Returns
    -------
    pd.DataFrame
        overshoot angles [deg] and times as index [s]
    """

    ## Checks:
    assert "delta" in df, "rudder angle 'delta' must exist in df"
    assert "psi" in df, "heading angle 'psi' must exist in df"
    assert df["delta"].abs().max() < np.deg2rad(90), "'delta should be in radians"

    angle = np.abs(angle)
    corners = _get_corners(angle=angle, df=df)
    maximums = _get_maximums(df=df, corners=corners)

    if heading_deviation is None:
        heading_deviation = angle

    if len(maximums) > 0:
        overshoots = maximums["psi"].abs() - np.abs(heading_deviation)
    else:
        overshoots = pd.Series([np.NaN, np.NaN])

    return np.rad2deg(overshoots)


def calculate_distance(df: pd.DataFrame) -> np.ndarray:
    """Calculate the distance in the x0,y0 plane.

    Parameters
    ----------
    df : pd.DataFrame
        [description]

    Returns
    -------
    np.ndarray
        distance for each time stamp
    """

    dx = df["x0"].diff()
    dy = df["y0"].diff()
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    return s


def get_initial_turning_distance(df: pd.DataFrame, angle: float) -> float:
    """Calculate the distance travelled till the zigzag angle deviation

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    angle : float
        ZigZag angle: zigzag(10)/10 in [rad]

    Returns
    -------
    float
        distance travelled till the zigzag angle deviation [m]
    """

    angle = np.abs(angle)
    df["distance"] = calculate_distance(df=df)
    corners = _get_corners(angle=angle, df=df)
    maximums = _get_maximums(df=df, corners=corners)

    if len(maximums) > 0:
        df_cut = df.loc[0 : maximums.index[0]]
        initial_turning_distance = np.interp(
            angle, df_cut["psi"].abs(), df_cut["distance"]
        )
    else:
        initial_turning_distance = pd.Series([np.NaN])

    return initial_turning_distance
