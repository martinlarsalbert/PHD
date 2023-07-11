import pandas as pd
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
import logging

log = logging.getLogger(__name__)


def pipeline(df_VCT_prime: pd.DataFrame, hull: PrimeEquationSubSystem) -> dict:

    tests = df_VCT_prime.groupby(by="test type")

    regression_pipeline = {
        "resistance": {
            "eq": hull.equations["X_H"].subs(
                [
                    (v, 0),
                    (r, 0),
                ]
            ),
            "data": tests.get_group("resistance"),
        },
        "Drift angle fx": {
            "eq": hull.equations["X_H"].subs(
                [
                    (r, 0),
                ]
            ),
            "data": tests.get_group("Drift angle"),
        },
        "Drift angle fy": {
            "eq": hull.equations["Y_H"].subs(
                [
                    (r, 0),
                ]
            ),
            "data": tests.get_group("Drift angle"),
        },
        "Drift angle mz": {
            "eq": hull.equations["N_H"].subs(
                [
                    (r, 0),
                ]
            ),
            "data": tests.get_group("Drift angle"),
        },
        "Circle fx": {
            "eq": hull.equations["X_H"].subs(
                [
                    (v, 0),
                ]
            ),
            "data": tests.get_group("Circle"),
        },
        "Circle fy": {
            "eq": hull.equations["Y_H"].subs(
                [
                    (v, 0),
                ]
            ),
            "data": tests.get_group("Circle"),
        },
        "Circle mz": {
            "eq": hull.equations["N_H"].subs(
                [
                    (v, 0),
                ]
            ),
            "data": tests.get_group("Circle"),
        },
        "Circle + Drift fx": {
            "eq": hull.equations["X_H"],
            "data": tests.get_group("Circle + Drift"),
        },
        "Circle + Drift fy": {
            "eq": hull.equations["Y_H"],
            "data": tests.get_group("Circle + Drift"),
        },
        "Circle + Drift mz": {
            "eq": hull.equations["N_H"],
            "data": tests.get_group("Circle + Drift"),
        },
    }

    return regression_pipeline


def fit(regression_pipeline: dict):
    models = {}
    exclude_parameters = {}
    new_parameters = {}
    for name, regression in regression_pipeline.items():
        log.info(f"Fitting:{name}")
        eq = regression["eq"]
        if eq.rhs == 0:
            print(f"skipping:{name}")
            continue

        label = regression.get("label", eq.lhs)
        eq_to_matrix = DiffEqToMatrix(
            eq,
            label=label,
            base_features=[u, v, r, thrust],
            exclude_parameters=exclude_parameters,
        )

        data = regression["data"]
        assert len(data) > 0
        key = eq_to_matrix.acceleration_equation.lhs.name
        X, y = eq_to_matrix.calculate_features_and_label(data=data, y=data[key])

        if len(X.columns) == 0:
            print(f"skipping:{name}")
            continue

        ols = sm.OLS(y, X)
        models[name] = ols_fit = ols.fit()
        new_parameters.update(ols_fit.params)
        exclude_parameters.update(new_parameters)

    return models, new_parameters
