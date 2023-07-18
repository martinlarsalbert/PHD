"""
This is a boilerplate pipeline 'predict'
generated using Kedro 0.18.7
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import logging


log = logging.getLogger(__name__)

dofs = ["X_D", "Y_D", "N_D"]
def force_prediction_score(models: dict, time_series: dict) -> pd.DataFrame:
    _ = []
    for id, data_loader in time_series.items():
        log.info(f"_____________ Making prediction for time series:{id} ____________")
        data = data_loader()
        if not "rev" in data:
            data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

        if not "twa" in data:
            data["twa"] = 0

        if not "tws" in data:
            data["tws"] = 0

        for model_name, model_loader in models.items():
            log.info(f"   * {model_name}")
            model = model_loader()

            try:
                df_force = model.forces_from_motions(data=data)
            except AttributeError as e:
                log.warning(f"Skipping model:{model_name} (Too old API)")
                continue

            df_force["X_D"] = df_force["fx"]
            df_force["Y_D"] = df_force["fy"]
            df_force["N_D"] = df_force["mz"]

            try:
                df_force_predicted = pd.DataFrame(
                    model.calculate_forces(
                        states_dict=data[model.states_str],
                        control=data[model.control_keys],
                    )
                )
            except KeyError as e:
                if "thrust" in e.args[0]:
                    log.warning(
                        f"Skipping model:{model_name} (Since it does not have a propeller model to predict thrust)"
                    )
                    continue
                else:
                    raise e
            except AttributeError as e:
                if "states_str" in e.args[0]:
                    log.warning(f"Skipping model:{model_name} (Too old API)")
                    continue
                else:
                    raise e

            s = score(df_force=df_force, df_force_predicted=df_force_predicted)
            s["model"] = model_name
            s["test_id"] = id
            _.append(s)

    force_prediction_scores = pd.DataFrame(_)
    r2_keys = [f"r2({dof})" for dof in dofs]
    rmse_keys = [f"rmse({dof})" for dof in dofs]
    force_prediction_scores["mean(r2)"] = force_prediction_scores[r2_keys].mean(axis=1)
    force_prediction_scores["mean(rmse)"] = force_prediction_scores[rmse_keys].mean(
        axis=1
    )

    return force_prediction_scores
   
def score(df_force:pd.DataFrame, df_force_predicted:pd.DataFrame):
    s = pd.Series()
    

    for dof in dofs:
        s[f"r2({dof})"] = r2_score(
            y_true=df_force[dof], y_pred=df_force_predicted[dof]
        )
        s[f"rmse({dof})"] = np.sqrt(
            mean_squared_error(
                y_true=df_force[dof], y_pred=df_force_predicted[dof]
            )
        )

    return s

def select_prediction_dataset_7m(
    time_series: dict, test_meta_data: pd.DataFrame
) -> dict:
    mask = test_meta_data["type"] == "zigzag"
    test_meta_data_selected = test_meta_data.loc[mask]

    log.info(f"selecting tests:{str(test_meta_data_selected.index)}")

    time_series_prediction = {
        id: time_series[str(id)] for id in test_meta_data_selected.index
    }

    return time_series_prediction
