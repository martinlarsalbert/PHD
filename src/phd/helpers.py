import pandas as pd
import numpy as np
from vessel_manoeuvring_models.angles import mean_angle

angle_columns = [
    "psi",
    "cog",
    "twa",
    "twaBowRAW",
    "twaSternRAW",
]


def mean(item):

    if isinstance(item, pd.core.window.rolling.Rolling):
        return mean_rolling(item)
    elif isinstance(item, pd.DataFrame):
        return mean_df(item)
    else:
        raise ValueError(f"Mean not implemented for type:{type(item)}")


def mean_df(df: pd.DataFrame) -> pd.Series:

    s = df.mean()
    angles = list(set(s.index) & set(angle_columns))
    for key in angles:
        s[key] = mean_angle(df[key])

    return s


def mean_rolling(rolling: pd.core.window.rolling.Rolling) -> pd.DataFrame:

    _ = []
    for window in rolling:
        s = mean_df(window)
        s.name = np.mean(window.index)
        _.append(s)

    df = pd.DataFrame(_)
    return df
