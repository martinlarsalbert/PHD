"""
This is a boilerplate pipeline 'load_7m'
generated using Kedro 0.18.7
"""
import pandas as pd
import re
import numpy as np
import geopandas
from numpy import sin, cos
from typing import Tuple

regexp = re.compile(r"\(.*\)")


def load(time_series_raw: dict, GPS_position: dict, missions: dict) -> dict:
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
        missions = missions[key]()
        time_series[key] = data = _load(
            loader=loader, GPS_position=GPS_position, missions=missions
        )
        statistics = run_statistics(data=data)
        statistics.name = key
        _.append(statistics)

    time_series_meta_data = pd.DataFrame(_)
    return time_series, time_series_meta_data


def _load(loader, GPS_position: dict, missions: str):
    data = loader()
    # data.index = pd.to_datetime(data.index, unit="s")  # This was used in the first batch
    data.index = pd.to_datetime(data.index, unit="us")

    add_missions(data=data, missions=missions)

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


def add_missions(data: pd.DataFrame, missions: str) -> pd.DataFrame:

    s_mission = parse_mission_string(missions=missions)
    data["mission"] = sync_and_merge_missions(data=data, s_mission=s_mission)

    # no return data is updated...


def parse_mission_string(missions: str) -> pd.Series:
    _missions = []
    _times = []

    for row in missions.split("\n"):
        parts = row.split(" ", 1)
        _times.append(int(parts[0]))
        _missions.append(parts[1])

    s_mission = pd.Series(_missions, index=_times, name="mission")
    s_mission.index = pd.to_datetime(s_mission.index, unit="us")
    s_mission.index.name = "time(us)"
    assert s_mission.index.is_unique

    return s_mission


def sync_and_merge_missions(data: pd.DataFrame, s_mission: pd.Series) -> pd.Series:
    """Sync the mission log messages with the data.
    If more than one message is associated (close in time) with the same time stamp in the data,
    the messages are concatenated, separated by a comma.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    s_mission : pd.Series
        _description_

    Returns
    -------
    pd.Series
        series with closest data time stamp for each message(s) (if merged)
    """

    _ = {}
    for time, message in s_mission.items():
        i = np.abs((data.index - time).total_seconds()).argmin()
        index = data.index[i]

        if not index in _:
            _[index] = ""

        _[index] += f"{message},"

    s_mission_merged = pd.Series(_)
    return s_mission_merged


def divide_into_tests(data: dict) -> Tuple[dict, pd.DataFrame]:
    """Divide time series into tests (zigzag tests etc.)

    Parameters
    ----------
    data : dict
        partitioned dataset with time series

    Returns
    -------
    tests, time_series_meta_data
        tests: partitioned dataset (dict) with dataframes,
        time_series_meta_data: data frame with meta data
    """

    tests = {}
    _ = []
    id0 = 0
    for key, loader in data.items():

        data_ = loader()
        tests_, time_series_meta_data_ = _divide_into_tests(data=data_, id0=id0)
        time_series_meta_data_["time_series"] = key
        tests.update(tests_)
        _.append(time_series_meta_data_)
        id0 = np.max(list(tests.keys())) + 1

    time_series_meta_data = pd.concat(_, axis=0)
    time_series_meta_data.index.name = "id"

    # Make some fixes...
    for id, df in tests.items():
        df["global time"] = df.index
        df.index -= df.index[0]
        df.drop(columns=["zigzag_test_id", "inbetween_zigzags_id"], inplace=True)

    tests = {str(int(key)): value for key, value in tests.items()}

    return tests, time_series_meta_data


def _divide_into_tests(data: pd.DataFrame, id0=0) -> Tuple[dict, pd.DataFrame]:

    find_zigzags(data=data, id0=id0)

    tests = {}
    _ = []
    # ZigZags:
    for id, df in data.groupby(by="zigzag_test_id"):

        if pd.isnull(id):
            continue

        tests[id] = df
        run_statistics(df)
        statistics = run_statistics(data=df)
        statistics.name = id
        statistics["type"] = "zigzag"
        _.append(statistics)

    # Inbetweens:
    for id, df in data.groupby(by="inbetween_zigzags_id"):

        if pd.isnull(id):
            continue

        tests[id] = df
        run_statistics(df)
        statistics = run_statistics(data=df)
        statistics.name = id
        statistics["type"] = "inbetween"
        _.append(statistics)

    time_series_meta_data = pd.DataFrame(_)

    return tests, time_series_meta_data


def find_zigzags(data: pd.DataFrame, id0=0):

    ## Zigzags:
    data["zigzag_test_id"] = np.NaN
    mission_rows = data["mission"].dropna()
    mask = mission_rows.str.contains("ZigZag: start")
    zigzag_starts = mission_rows.loc[mask].index
    data.loc[zigzag_starts, "zigzag_test_id"] = np.arange(0, len(zigzag_starts))

    mask = mission_rows.str.contains("ZigZag: stop")
    zigzag_stops = mission_rows.loc[mask].index
    data.loc[zigzag_stops, "zigzag_test_id"] = np.arange(0, len(zigzag_stops))

    for i, zigzag_start in enumerate(zigzag_starts):

        if i == len(zigzag_stops):
            data.loc[zigzag_start:, "zigzag_test_id"] = i
        else:
            zigzag_stop = zigzag_stops[i]
            data.loc[zigzag_start:zigzag_stop, "zigzag_test_id"] = i

    ## Inbetweens:
    data["inbetween_zigzags_id"] = np.NaN
    for i, zigzag_stop in enumerate(zigzag_stops[0:-1]):

        next_zigzag_start = zigzag_starts[i + 1]
        zigzag_stop_i = data.index.get_loc(zigzag_stop)
        next_zigzag_start_i = data.index.get_loc(next_zigzag_start)
        start = data.index[zigzag_stop_i + 1]
        end = data.index[next_zigzag_start_i - 1]

        data.loc[start:end, "inbetween_zigzags_id"] = (
            i + data["zigzag_test_id"].max() + 2
        )

    zigzag_start_i = data.index.get_loc(zigzag_starts[0])
    end = data.index[zigzag_start_i - 1]
    data["inbetween_zigzags_id"].loc[data.index[0] : end] = (
        data["inbetween_zigzags_id"].min() - 1
    )

    zigzag_stop_i = data.index.get_loc(zigzag_stops[-1])
    start = data.index[zigzag_stop_i + 1]
    data["inbetween_zigzags_id"].loc[start:] = data["inbetween_zigzags_id"].max() + 1

    data["zigzag_test_id"] += id0
    data["inbetween_zigzags_id"] += id0

    # Check that no duplicated samples exist:
    assert (
        (data["zigzag_test_id"].notnull() & data["inbetween_zigzags_id"].isnull())
        | (data["zigzag_test_id"].isnull() & data["inbetween_zigzags_id"].notnull())
    ).all()
