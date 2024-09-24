import pandas as pd
import numpy as np
from vessel_manoeuvring_models.angles import mean_angle
from numpy import sqrt, sin, cos, arctan2, pi
import inspect
from vessel_manoeuvring_models.differentiation import derivative

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


def apparent_wind_speed_to_true(
    U: np.ndarray,
    awa: np.ndarray,
    aws: np.ndarray,
    cog: np.ndarray,
    psi: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return sqrt(U**2 - 2 * U * aws * cos(awa - cog + psi) + aws**2)


def apparent_wind_angle_to_true(
    U: np.ndarray,
    awa: np.ndarray,
    aws: np.ndarray,
    cog: np.ndarray,
    psi: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return (
        arctan2(
            U * sin(cog) - aws * sin(awa + psi), U * cos(cog) - aws * cos(awa + psi)
        )
        + pi
    )


def true_wind_speed_to_apparent(
    U: np.ndarray, cog: np.ndarray, twa: np.ndarray, tws: np.ndarray, **kwargs
) -> np.ndarray:
    return sqrt(U**2 + 2 * U * tws * cos(cog - twa) + tws**2)


def true_wind_angle_to_apparent(
    U: np.ndarray,
    cog: np.ndarray,
    psi: np.ndarray,
    twa: np.ndarray,
    tws: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return (
        arctan2(
            -U * sin(cog - psi) + tws * sin(psi - twa),
            -U * cos(cog - psi) - tws * cos(psi - twa),
        )
        + pi
    )


def identity_decorator(wrapped):
    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(wrapped)  # the magic is here!
    return wrapper

def lazy(loaders:dict, function, **kwargs):
    """
    Apply lazy loading to a node to save memory
       
    
    Args:
        loaders (dict): all the loaders, which will be loaded and passed to the function
        function (_type_): the function where the loaded variables will be inserted.
    
    Example:
    
    Instead of doing this (which consumes a lot of memory):
    ```
    def node(models:dict)->dict:
        
        new_models = {}
        for name, loader in models.items():
            model = loader()
            new_models[name] = the_function(model=model, a=1)
        
        return new_models
    ```
    
    We can instead use the lazy method:        
    ```
    def node(models:dict)->dict:
        
        new_models = {}
        for name, loader in models.items():
            new_models[name] = lazy(loaders={'model':loader}, function=the_function, a=1)
        
        return new_models
    
    
    ```
    """
    def run():
        inputs = {key:loader() for key,loader in loaders.items()}
        return function(**inputs, **kwargs)
    return run

def lazy_iteration(loaders, var_name:str, function, **kwargs):
    
    savers = {}
        
    for name, loader in loaders.items():
        savers[name] = lazy(loaders={var_name:loader}, function=function, **kwargs)
        
    return savers