import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy.optimize import least_squares
import numpy as np
from phd.visualization.plot_prediction import predict


def residual(x, model: ModularVesselSimulator, data: pd.DataFrame):
    data["delta"] = x
    df_force_predicted = predict(model=model, data=data)
    residual = df_force_predicted["N_D"]

    return residual


def fit(model: ModularVesselSimulator, data: pd.DataFrame):
    kwargs = {
        "model": model,
        "data": data.copy(),
    }

    x0 = np.zeros(len(data))

    return least_squares(
        fun=residual,
        x0=x0,
        # bounds=((0, 3 * x_r), (1, -3 * x_r)),
        kwargs=kwargs,
    )
