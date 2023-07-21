"""
This is a boilerplate pipeline 'resimulate_with_autopilot'
generated using Kedro 0.18.7
"""

import pandas as pd
import numpy as np
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.IMO_simulations import zigzag

import logging

log = logging.getLogger(__name__)


def resimulate_all(
    datas: dict,
    model: ModularVesselSimulator,
    exclude=["wpcc.updated.joined.ek_smooth"],
) -> pd.DataFrame:
    results = {}
    log.info("Resimulating:")
    for key, loader in datas.items():
        if key in exclude:
            log.info(f"Exluded: {key}")
            continue
        else:
            log.info(f"Simulate: {key}")
        data = loader()
        results[key] = resimulate(data=data, model=model)

    return results


def resimulate(
    data: pd.DataFrame,
    model: ModularVesselSimulator,
) -> pd.DataFrame:
    data = data.copy()
    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    if not "rev" in data:
        data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

    if not "twa" in data:
        data["twa"] = 0

    if not "tws" in data:
        data["tws"] = 0

    u0 = data.iloc[0]["u"]
    # rev=float(data['rev'].mean())
    rev = data["rev"]
    rudder_rate = 2.32 * np.sqrt(model.ship_parameters["scale_factor"])
    angle = 10 if data["delta"].idxmin() > data["delta"].idxmax() else -10

    # twa = mean_angle(data['twa'])
    twa = data["twa"]

    # tws = data['tws'].mean()
    tws = data["tws"]

    neutral_rudder_angle = np.rad2deg(data["delta"].max() + data["delta"].min()) / 2
    result = zigzag(
        model=model,
        u0=u0,
        rev=rev,
        rudder_rate=rudder_rate,
        angle=angle,
        twa=twa,
        tws=tws,
        neutral_rudder_angle=neutral_rudder_angle,
    )
    return result
