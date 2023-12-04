import pandas as pd
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
import logging

log = logging.getLogger(__name__)


def pipeline(df_VCT_prime: pd.DataFrame, model: ModularVesselSimulator) -> dict:
    hull = model.subsystems["hull"]
    assert isinstance(hull, PrimeEquationSubSystem)

    tests = df_VCT_prime.groupby(by="test type")

    regression_pipeline = {
        "resistance": {
            "eq": hull.equations["X_H"].subs(
                [
                    (v, 0),
                    (r, 0),
                ]
            ),
            "data": tests.get_group("resistance")
            if "resistance" in tests.groups
            else tests.get_group("self propulsion"),
        },
        "resistance fy": {
            "eq": hull.equations["Y_H"].subs(
                [
                    (v, 0),
                    (r, 0),
                ]
            ),
            "data": tests.get_group("resistance")
            if "resistance" in tests.groups
            else tests.get_group("self propulsion"),
        },
        "resistance mz": {
            "eq": hull.equations["N_H"].subs(
                [
                    (v, 0),
                    (r, 0),
                ]
            ),
            "data": tests.get_group("resistance")
            if "resistance" in tests.groups
            else tests.get_group("self propulsion"),
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


def pipeline_with_rudder(
    df_VCT_prime: pd.DataFrame, model: ModularVesselSimulator
) -> dict:
    regression_pipeline = pipeline(df_VCT_prime=df_VCT_prime, model=model)

    rudders = model.subsystems["rudders"]
    assert isinstance(rudders, PrimeEquationSubSystem)
    tests = df_VCT_prime.groupby(by="test type")

    rudder_pipeline = {
        "Rudder fy": {
            "eq": rudders.equations["Y_R"],
            "data": tests.get_group("Rudder angle resistance (no propeller)"),
        },
        "Rudder mz": {
            "eq": rudders.equations["N_R"],
            "data": tests.get_group("Rudder angle resistance (no propeller)"),
        },
    }
    regression_pipeline.update(rudder_pipeline)
    return regression_pipeline


def fit(regression_pipeline: dict, exclude_parameters={}):
    models = {}
    exclude_parameters = exclude_parameters.copy()
    new_parameters = exclude_parameters.copy()
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
            base_features=[u, v, r, thrust, delta],
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
        try:
            models[name] = ols_fit = ols.fit()
        except Exception:
            raise ValueError(f"Failed in regression:{name}")
        ols_fit.X = X
        ols_fit.y = y
        new_parameters.update(ols_fit.params)
        exclude_parameters.update(new_parameters)

    return models, new_parameters
