from vct.regression_ols import Regression, RegressionPipeline
import symseaman as ss
from symseaman.seaman_symbols import *
from symseaman.shipdict import ShipDict
from vct.regression_ols import data_to_bis
import pandas as pd


class Regression_yuv(Regression):
    _eq = ss.equations.hull.sway.eq_expanded.subs(
        [
            (phi.bis, 0),
            (r_w.bis, 0),
            (delta, 0),
            (Y_uuv, 0),  # Note!
        ]
    )


def fit(df_VCT: pd.DataFrame, shipdict: ShipDict):
    pre_set_derivatives_0 = {}
    pipeline = RegressionPipeline(
        shipdict=shipdict, pre_set_derivatives=pre_set_derivatives_0
    )

    units = {
        "fx_hull": "force",
        "fy_hull": "force",
        "mz_hull": "moment",
    }

    interesting = [
        "fx_hull",
        "fy_hull",
        "mz_hull",
    ]

    df_bis = data_to_bis(
        df=df_VCT, shipdict=shipdict, units=units, interesting=interesting
    )

    mask = df_bis["test type"].isin(["Drift angle"])
    df_ = df_bis.loc[mask]

    ## __________________________
    ## Drift angles
    # One run to get nonlinearities:
    i = 0
    for V_, df__ in df_.groupby(by="V"):

        if len(df__) < 2:
            continue

        pipeline[f"yvav_{i}"] = Regression_yuv(df=df__, derivatives_suffix=f"_{i}")

        i += 1
    pipeline.fit()

    i = 0
    for V_, df__ in df_.groupby(by="V"):
        if len(df__) < 2:
            continue
        pipeline[f"yuv_{i}"] = Regression_yuv(df=df__, derivatives_suffix=f"_{i}")
        i += 1
    pipeline.fit()

    return pipeline, df_bis