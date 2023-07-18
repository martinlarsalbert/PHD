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
import logging
from vessel_manoeuvring_models.angles import smallest_signed_angle
from vessel_manoeuvring_models.angles import mean_angle
from vessel_manoeuvring_models.apparent_wind import (
    apparent_wind_angle_to_true,
    apparent_wind_speed_to_true,
)
from vessel_manoeuvring_models.prime_system import PrimeSystem     

log = logging.getLogger(__name__)


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
    units_all = {}
    _ = []
    for key, loader in time_series_raw.items():
        missions = missions[key]()
        data, units = _load(loader=loader, GPS_position=GPS_position, missions=missions)
        units_all.update(units)
        time_series[key] = data
        statistics = run_statistics(data=data, units=units)
        statistics.name = key
        _.append(statistics)

    time_series_meta_data = pd.DataFrame(_)
    return time_series, time_series_meta_data, units_all


def derivative(df, key):
    d = np.diff(df[key]) / np.diff(df.index)
    d = np.concatenate((d, [d[-1]]))
    return d


regexp = re.compile(r"\((.*)\)")


def _load(loader, GPS_position: dict, missions: str, replace_velocities=True):
    data = loader()
    # data.index = pd.to_datetime(data.index, unit="s")  # This was used in the first batch
    data.index = pd.to_datetime(data.index, unit="us")

    add_missions(data=data, missions=missions)

    data["date"] = data.index
    data.index = (data.index - data.index[0]).total_seconds()
    data["delta"] = -np.deg2rad(
        data["rudderAngle(deg)"]
    )  # (Change the convention of rudder angle)
    units = {"delta": "rad"}
    # Extract and remove the units from column names:
    renames = {}
    for column in data.columns:
        result = regexp.search(column)
        if result:
            remove = result.group(0)
            new_column = column.replace(remove, "")
            renames[column] = new_column
            if len(result.groups()) > 0:
                units[new_column] = result.group(1)

    data.rename(columns=renames, inplace=True)

    data["thrusterTarget"] = data["thrusterTarget"].fillna(method="ffill")

    psi = np.unwrap(data["yaw"])
    data["psi"] = psi
    data["phi"] = data["heelAngle"]
    units["psi"] = "rad"
    units["phi"] = "rad"

    data = add_xy_from_latitude_and_longitude(data=data)
    units["x_GPS"] = "m"
    units["y_GPS"] = "m"
    data = move_GPS_to_origo(data=data, GPS_position=GPS_position)
    units["x0"] = "m"
    units["y0"] = "m"

    dxdt = derivative(data, "x0")
    dydt = derivative(data, "y0")
    psi = data["psi"]

    if not "u" in data or replace_velocities:
        data["u"] = dxdt * np.cos(psi) + dydt * np.sin(psi)

    if not "v" in data or replace_velocities:
        data["v"] = v = -dxdt * np.sin(psi) + dydt * np.cos(psi)
    units["u"] = "m/s"
    units["v"] = "m/s"

    if not "r" in data or replace_velocities:
        data["r"] = r = derivative(data, "psi")
    units["r"] = "rad/s"

    data["u1d"] = derivative(data, "u")
    data["v1d"] = derivative(data, "v")
    data["r1d"] = derivative(data, "r")
    units["u1d"] = "m/s2"
    units["v1d"] = "m/s2"
    units["r1d"] = "rad/s2"

    estimate_apparent_wind(data=data)
    units["aws"] = "m/s"
    units["awa"] = "rad"

    calculated_signals(data=data)
    units["beta"] = "rad"
    units["tws"] = "m/s"
    units["twa"] = "rad"

    # data = fix_interpolated_angle(data=data, key="awaBowRAW")
    # data = fix_interpolated_angle(data=data, key="awaSternRAW")
    # data = fix_interpolated_angle(data=data, key="awaBow")
    # data = fix_interpolated_angle(data=data, key="awaStern")

    return data, units


def calculated_signals(data: pd.DataFrame) -> pd.DataFrame:
    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    data["beta"] = -np.arctan2(data["v"], data["u"])  # Drift angle
    data["cog"] = smallest_signed_angle(data["psi"]) - smallest_signed_angle(
        data["beta"]
    )

    calculate_true_wind(data=data)


def fix_interpolated_angle(
    data: pd.DataFrame,
    key: str,
    max_change=0.4,
) -> pd.DataFrame:
    """Some of the angles in the data have been interpolated numerically so that false points exist in around 0 degrees.
    This method tried to remove these false points (but it is not perfect as some points are hard to distinquish)

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    key : str
        _description_
    limit : float, optional
        maximum change of angle between two time steps, by default 0.4

    Returns
    -------
    pd.DataFrame
        _description_
    """

    log.info(f"fix_interpolated_angle for {key}")

    y = data[key].values.copy()

    condition = np.abs(np.diff(np.unwrap(y))) > max_change
    data_ = data.copy()
    ok = False
    for _ in range(len(data)):
        if not np.max(condition):
            ok = True
            break
        i = np.argmax(condition)
        data_.drop(index=data_.index[i + 1], inplace=True)
        y = data_[key].values
        condition = np.abs(np.diff(np.unwrap(y))) > max_change
        # print(i)
    data_[key] = np.unwrap(data_[key])

    assert (
        data.index[-1] == data_.index[-1]
    ), f"Something went wrong in the fix_interpolation_angle for {key} (increasing 'max_change' parameter might help)"

    assert ok, f"Max number of iterations exceeded for {key}"

    # New interpolation:
    x = np.cos(data_[key])
    y = np.sin(data_[key])
    x_interp = np.interp(data.index, data_.index, x)
    y_interp = np.interp(data.index, data_.index, y)
    data[key] = np.unwrap(np.arctan2(y_interp, x_interp))

    return data


def run_statistics(data: pd.DataFrame, units: dict) -> pd.Series:
    statistics = data.select_dtypes(include=float).mean()

    for key, unit in units.items():
        if unit == "rad":
            statistics[key] = mean_angle(data[key])

    statistics["date"] = data.iloc[0]["date"]

    mask = data["mission"].notnull()
    statistics["missions"] = "".join(
        f"{mission}," for mission in data.loc[mask, "mission"]
    )

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


def estimate_apparent_wind(data: pd.DataFrame):
    """Estimate the apparent wind, taking the average between bow and stern anomemeter.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    """
    data["aws"] = data[["awsBowRAW", "awsSternRAW"]].mean(axis=1)
    data["awa"] = np.unwrap(mean_angle(data[["awaBowRAW", "awaSternRAW"]], axis=1))


def calculate_true_wind(data: pd.DataFrame):
    """Calculate true wind speed and angle from apparent wind.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    """

    data["tws"] = apparent_wind_speed_to_true(**data)
    data["twa"] = apparent_wind_angle_to_true(**data)


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


def zero_angle(angle: pd.Series) -> pd.Series:
    return angle - (angle.iloc[0] - smallest_signed_angle(angle.iloc[0]))


def divide_into_tests(data: dict, units: dict) -> Tuple[dict, pd.DataFrame]:
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
        tests_, time_series_meta_data_ = _divide_into_tests(
            data=data_, units=units, id0=id0
        )
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

        # angles = [
        #    "psi",
        #    "cog",
        #    "twa",
        #    "awa",
        # ]
        # for key in angles:
        #    df[key] = zero_angle(df[key])

        df.drop(columns=["zigzag_test_id", "inbetween_zigzags_id"], inplace=True)

    tests = {str(int(key)): value for key, value in tests.items()}

    return tests, time_series_meta_data


def _divide_into_tests(
    data: pd.DataFrame, units: dict, id0=0
) -> Tuple[dict, pd.DataFrame]:

    find_zigzags(data=data, id0=id0)

    tests = {}
    _ = []
    # ZigZags:
    for id, df in data.groupby(by="zigzag_test_id"):

        if pd.isnull(id):
            continue

        tests[id] = df
        statistics = run_statistics(data=df, units=units)
        statistics.name = id
        statistics["type"] = "zigzag"
        _.append(statistics)

    # Inbetweens:
    for id, df in data.groupby(by="inbetween_zigzags_id"):

        if pd.isnull(id):
            continue

        tests[id] = df
        statistics = run_statistics(data=df, units=units)
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

def scale_ship_data(ship_data_wPCC:dict,scale_factor:float, rho:float)-> dict:
    
    lpp = ship_data_wPCC['L']*ship_data_wPCC['scale_factor']/scale_factor
    prime_system = PrimeSystem(L=lpp, rho=rho)
    ship_data_7m =prime_system.unprime(ship_data_wPCC)
    
    ship_data_7m['x_G'] = -4.784/100*ship_data_7m['L']  # wPCC has x_G=0 because motions are given at CG.
    ship_data_7m['x_r'] -=0.05  # E-mail from Ulysse: "...This means that the rudders stick out a little bit aft of the boat (something like 5-6 cm). "  
    return ship_data_7m
    