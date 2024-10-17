import pandas as pd
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
import logging
from vessel_manoeuvring_models.substitute_dynamic_symbols import run


log = logging.getLogger(__name__)


def pipeline_RHI(df_VCT_prime: pd.DataFrame, model: ModularVesselSimulator) -> dict:
    hull = model.subsystems["hull"]
    assert isinstance(hull, PrimeEquationSubSystem)

    tests = df_VCT_prime.groupby(by="test type")

    rudder_hull_interaction = model.subsystems["rudder_hull_interaction"]

    data_rudder = tests.get_group("Rudder angle").copy()

    regression_pipeline = {
        "rudder hull interaction aH": {
            "eq": rudder_hull_interaction.equations["Y_RHI"].subs(Y_RHI, Y_D_),
            # "eq": sp.Eq(Y_D_,rudder_hull_interaction.equations['Y_RHI'].rhs + Y_R),
            "data": data_rudder,
        },
        "rudder hull interaction xH": {
            # "eq": rudder_hull_interaction.equations['N_RHI'].subs([(N_RHI,N_D_),(L,1),(a_H*x_H,a_x_H)]),
            # "eq": sp.Eq(N_D_,rudder_hull_interaction.equations['N_RHI'].rhs.subs([(L,1),(a_H*x_H,a_x_H)]) + N_R),
            "eq": sp.Eq(N_D_, x_H * N_R),
            "data": data_rudder,
        },
    }

    return regression_pipeline


def pipeline(df_VCT_prime: pd.DataFrame, model: ModularVesselSimulator) -> dict:
    hull = model.subsystems["hull"]
    assert isinstance(hull, PrimeEquationSubSystem)
        
    tests = df_VCT_prime.groupby(by="test type")

    # Adding the rudder hull interaction to the hull forces (otherwise they will be included both in the hull forces and the RHI forces):
    # eq_Y_H = sp.Eq(
    #    hull.equations["Y_H"].lhs, hull.equations["Y_H"].rhs + (a_H - 1) * Y_R
    # )
    eq_Y_H = hull.equations["Y_H"]

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
        # "resistance fy": {
        #    "eq": eq_Y_H.subs(
        #        [
        #            (v, 0),
        #            (r, 0),
        #        ]
        #    ),
        #    "data": tests.get_group("resistance")
        #    if "resistance" in tests.groups
        #    else tests.get_group("self propulsion"),
        # },
        # "resistance mz": {
        #    "eq": hull.equations["N_H"].subs(
        #        [
        #            (v, 0),
        #            (r, 0),
        #        ]
        #    ),
        #    "data": tests.get_group("resistance")
        #    if "resistance" in tests.groups
        #    else tests.get_group("self propulsion"),
        # },
        "Drift angle fx": {
            "eq": hull.equations["X_H"].subs(
                [
                    (r, 0),
                ]
            ),
            "data": tests.get_group("Drift angle"),
            "const": True,
        },
        "Drift angle fy": {
            "eq": eq_Y_H.subs(
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
            "const": True,
        },
        "Circle fy": {
            "eq": eq_Y_H.subs(
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
            "const": True,
        },
        "Circle + Drift fy": {
            "eq": eq_Y_H,
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
    eq_X = model.expand_subsystemequations(model.X_D_eq)
    eq_Y = model.expand_subsystemequations(model.Y_D_eq)
    eq_N = model.expand_subsystemequations(model.N_D_eq)

    hull = model.subsystems["hull"]
    assert isinstance(hull, PrimeEquationSubSystem)

    tests = df_VCT_prime.groupby(by="test type")
    rudders = model.subsystems["rudders"]

    mask = df_VCT_prime["test type"].isin(
        [
            "Circle + rudder angle",
            "Rudder and drift angle",
        ]
    )
    data_flow_straightening = df_VCT_prime.loc[mask]
    data_rudder = tests.get_group("Rudder angle").copy()
    regression_pipeline = {
        "Rudder fx": {
            "eq": rudders.equations["X_R"],
            "data": data_rudder,
            "const": True,
        },
        "Rudder fy": {
            "eq": eq_Y.subs([(v, 0), (r, 0)]),
            "data": data_rudder,
        },
        "Rudder mz": {
            "eq": eq_N.subs(
                [
                    (v, 0),
                    (r, 0),
                ]
            ),
            "data": data_rudder,
        },
        "resistance": {
            "eq": eq_X.subs(
                [
                    (v, 0),
                    (r, 0),
                ]
            ),
            "data": tests.get_group("resistance")
            if "resistance" in tests.groups
            else tests.get_group("self propulsion"),
        },
        # "resistance fy": {
        #    "eq": eq_Y.subs(
        #        [
        #            (v, 0),
        #            (r, 0),
        #        ]
        #    ),
        #    "data": tests.get_group("resistance")
        #    if "resistance" in tests.groups
        #    else tests.get_group("self propulsion"),
        # },
        # "resistance mz": {
        #    "eq": eq_N.subs(
        #        [
        #            (v, 0),
        #            (r, 0),
        #        ]
        #    ),
        #    "data": tests.get_group("resistance")
        #    if "resistance" in tests.groups
        #    else tests.get_group("self propulsion"),
        # },
        "Drift angle fx": {
            "eq": eq_X.subs([(r, 0), (delta, 0)]),
            "data": tests.get_group("Drift angle"),
            "const": True,
        },
        "Drift angle fy": {
            "eq": eq_Y.subs([(r, 0), (delta, 0)]),
            "data": tests.get_group("Drift angle"),
        },
        "Drift angle mz": {
            "eq": eq_N.subs([(r, 0), (delta, 0)]),
            "data": tests.get_group("Drift angle"),
        },
        "Circle fx": {
            "eq": eq_X.subs([(v, 0), (delta, 0)]),
            "data": tests.get_group("Circle"),
            "const": True,
        },
        "Circle fy": {
            "eq": eq_Y.subs([(v, 0), (delta, 0)]),
            "data": tests.get_group("Circle"),
        },
        "Circle mz": {
            "eq": eq_N.subs([(v, 0), (delta, 0)]),
            "data": tests.get_group("Circle"),
        },
        "Circle + Drift fx": {
            "eq": eq_X.subs(delta, 0),
            "data": tests.get_group("Circle + Drift"),
            "const": True,
        },
        "Circle + Drift fy": {
            "eq": eq_Y.subs(delta, 0),
            "data": tests.get_group("Circle + Drift"),
        },
        "Circle + Drift mz": {
            "eq": eq_N.subs(delta, 0),
            "data": tests.get_group("Circle + Drift"),
        },
        # "Rudder and drift angle fy": {
        #    "eq": eq_Y,
        #    "data": tests.get_group('Rudder and drift angle'),
        #    "const": True,
        # },
        # "Rudder and drift angle mz": {
        #    "eq": eq_N,
        #    "data": tests.get_group('Rudder and drift angle'),
        #    "const": True,
        # },
    }

    return regression_pipeline


def fit(
    regression_pipeline: dict, model: ModularVesselSimulator, exclude_parameters={}, simplify_names=True, feature_name_subs={}
):
    models = {}
    exclude_parameters = exclude_parameters.copy()
    new_parameters = exclude_parameters.copy()
    log.info(f"Excluding:{exclude_parameters}")
    for name, regression in regression_pipeline.items():
        log.info(f"Fitting:{name}")
        eq = regression["eq"]

        const = regression.get("const", False)
        if eq.rhs == 0:
            print(f"skipping:{name}")
            continue

        label = regression.get("label", eq.lhs)
        eq_to_matrix = DiffEqToMatrix(
            eq,
            label=label,
            base_features=[
                u,
                v,
                r,
                thrust,
                delta,
                thrust_port,
                thrust_stbd,
                y_p_port,
                y_p_stbd,
                Y_R,
                N_R,
            ],
            exclude_parameters=exclude_parameters,
            feature_name_subs=feature_name_subs
        )

        data = regression["data"]
        assert len(data) > 0
        key = eq_to_matrix.acceleration_equation.lhs.name
        X, y = eq_to_matrix.calculate_features_and_label(
            data=data, y=data[key], parameters=model.ship_parameters, simplify_names=simplify_names
        )

        if const:
            X["const"] = 1

        if len(X.columns) == 0:
            print(f"skipping:{name} with equation: {eq}")
            continue

        ols = sm.OLS(y, X)
        try:
            models[name] = ols_fit = ols.fit()
        except Exception:
            raise ValueError(f"Failed in regression:{name}")
        ols_fit.X = X
        ols_fit.y = y
        ols_fit.eq = eq
        new_parameters.update(ols_fit.params)
        exclude_parameters.update(new_parameters)
        # data[str(eq.rhs)] = ols_fit.predict(X)  # So that Y_R can be used again...

    return models, new_parameters
