"""
This is a boilerplate pipeline 'load_7m'
generated using Kedro 0.18.7
"""
import pandas as pd
import re
import numpy as np
import geopandas
from numpy import sin, cos

regexp = re.compile(r"\(.*\)")


def load(time_series_raw: dict, GPS_position: dict) -> dict:
    """_summary_

    Parameters
    ----------
    time_series_raw : dict
        _description_
    GPS_position : dict
        GPS position in ship reference frame:
            'x' position from lpp/2 -> fwd
            'y' position from center line -> stbd

    Returns
    -------
    dict
        time_series

    pd.DataFrame
        time_series_meta_data
    """
    time_series = {}
    _ = []
    for key, loader in time_series_raw.items():
        time_series[key] = data = _load(loader=loader, GPS_position=GPS_position)
        statistics = run_statistics(data=data)
        statistics.name = key
        _.append(statistics)

    time_series_meta_data = pd.DataFrame(_)
    return time_series, time_series_meta_data


def _load(loader, GPS_position: dict):
    data = loader()
    # data.index = pd.to_datetime(data.index, unit="s")  # This was used in the first batch
    data.index = pd.to_datetime(data.index, unit="us")
    data["date"] = data.index
    data.index = (data.index - data.index[0]).total_seconds()
    data["delta"] = -np.deg2rad(
        data["rudderAngle(deg)"]
    )  # (Change the convention of rudder angle)

    # Remove the units from column names:
    renames = {column: regexp.sub("", column) for column in data.columns}
    data.rename(columns=renames, inplace=True)

    data["V"] = V_ = data["sog"]
    cog_ = np.unwrap(data["cog"])
    psi_ = np.unwrap(data["yaw"])
    data["beta"] = beta_ = psi_ - cog_  # Drift angle
    data["u"] = V_ * np.cos(beta_)
    data["v"] = -V_ * np.sin(beta_)

    data["psi"] = psi_
    data["phi"] = data["heelAngle"]

    data = add_xy_from_latitude_and_longitude(data=data)

    data = move_GPS_to_origo(data=data, GPS_position=GPS_position)

    return data


def run_statistics(data) -> pd.Series:
    statistics = data.select_dtypes(include=float).abs().max()
    statistics["date"] = data.iloc[0]["date"]
    return statistics


def add_xy_from_latitude_and_longitude(data: pd.DataFrame, epsg=3015) -> pd.DataFrame:
    """Adding x0, y0 from latitude and longitude

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    epsg : int, optional
        _description_, by default 3015

    Returns
    -------
    pd.DataFrame
        _description_
    """

    df = geopandas.GeoDataFrame(
        data.copy(),
        geometry=geopandas.points_from_xy(
            data.longitude, data.latitude, crs="EPSG:4326"
        ),
    )
    df = df.to_crs(epsg=epsg)
    data["y_GPS"] = (
        df.geometry.x - df.geometry.x[0]
    )  # Position of GPS in global North direction
    data["x_GPS"] = (
        df.geometry.y - df.geometry.y[0]
    )  # Position of GPS in global East direction

    return data


def move_GPS_to_origo(data: pd.DataFrame, GPS_position: dict) -> pd.DataFrame:
    """Calculate the global position of the ship's origo from the GPS local and global position.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    GPS_position : dict
        GPS position in ship reference frame:
            'x' position from aft -> fwd
            'y' position from center line -> stbd

    Returns
    -------
    pd.DataFrame
        _description_
    """

    psi = data["psi"]
    # p is a vector from origo to GPS, the component of this vector is GPS_position["x"] and GPS_position["y"] in the ship reference frame.
    # In the global frame the components can be calculated as:
    pX = GPS_position["x"] * cos(psi) - GPS_position["y"] * sin(psi)
    pY = GPS_position["x"] * sin(psi) + GPS_position["y"] * cos(psi)

    # The position of the ship origo can now be calculated as GPS position - p:
    data["x0"] = data["x_GPS"] - pX
    data["y0"] = data["y_GPS"] - pY

    return data
