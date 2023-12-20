"""
This is a boilerplate pipeline 'resistance_MDL'
generated using Kedro 0.18.7
"""

import pandas as pd


def resistance(
    time_series_meta_data: pd.DataFrame, tests: dict, ship_data: dict
) -> pd.DataFrame:
    # mask = time_series_meta_data["test_type"] == "reference speed"
    # ids = time_series_meta_data.loc[mask].index

    ids = [22611, 22635, 22639]

    _ = []
    for id in ids:
        df = tests[str(id)]()
        s = df.iloc[500:].mean(axis=0)
        s.name = id

        if s["V"] > 0.6:
            _.append(s)

    df_resistance = pd.concat(_, axis=1).transpose()

    #df_resistance["Rtot"] = df_resistance["thrust"] / (1 - ship_data["tdf"])
    #df_resistance["X_D"] = -df_resistance["Rtot"]
    
    df_resistance["X_D"] = 0
    
    df_resistance.sort_values(by="u", inplace=True)

    return df_resistance
