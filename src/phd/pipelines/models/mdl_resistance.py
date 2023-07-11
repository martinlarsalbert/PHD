import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import numpy as np


def reference_test_resistance(
    model: ModularVesselSimulator,
    time_series_meta_data: pd.DataFrame,
    time_series_smooth: dict,
) -> pd.DataFrame:

    _ = []
    for id, row in (
        time_series_meta_data.groupby(by=["test type"])
        .get_group("reference speed")
        .iterrows()
    ):
        try:
            data_ = time_series_smooth[f"wpcc.updated.{id}.ek_smooth"]()
        except:
            continue

        data_["V"] = data_["U"] = np.sqrt(data_["u"] ** 2 + data_["v"] ** 2)
        data_["rev"] = data_[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)
        _.append(pd.Series(data_.mean(), name=id))

    df_reference_speed = pd.concat(_, axis=1).transpose()
    df_reference_speed["X_H"] = (
        -(1 - model.ship_parameters["tdf"]) * df_reference_speed["thrust"]
    )
    return df_reference_speed
