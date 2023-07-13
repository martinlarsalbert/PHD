"""
This is a boilerplate pipeline 'resimulate'
generated using Kedro 0.18.7
"""
import pandas as pd
import numpy as np
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
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

    result = model.simulate(data)
    return result
