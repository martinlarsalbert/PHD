from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import pandas as pd
import phd
from matplotlib.pyplot import cm
import numpy as np

# If you are creating a session outside of a Kedro project (i.e. not using
# `kedro run` or `kedro jupyter`), you need to run `bootstrap_project` to
# let Kedro find your configuration.
project_path = Path(phd.__path__[0]).parents[1]
bootstrap_project(project_path)
with KedroSession.create(project_path=project_path) as session:
    context = session.load_context()
    model_loaders = context.catalog.load("wPCC.models")


def get_colors(models):
    return pd.Series(colors)[models].values


def simple_model_name(model_name: str, **kwargs):
    vmm = model_name.split(".")[0]
    regression = model_name.split(".")[-1]

    return (
        f"{vmm_simple_names[vmm]}.{regression_simple_names.get(regression,regression)}"
    )


def simple_model_names(row: pd.Series) -> str:
    return simple_model_name(**row)


model_names = pd.DataFrame(index=model_loaders.keys())
model_names["model_name"] = model_names.index
model_names["vmm"] = model_names["model_name"].apply(lambda x: x.split(".")[0])
model_names["regression"] = model_names["model_name"].apply(lambda x: x.split(".")[-1])
model_names.sort_values(by=["vmm", "regression"], inplace=True)

vmm_simple_names = {key: f"M{i+1}" for i, key in enumerate(model_names["vmm"].unique())}
regression_simple_names = {
    "MDL_hull_inverse_dynamics": "MDL_HID",
    "MDL_inverse_dynamics": "MDL_ID",
}

model_names["model_name_simple"] = model_names.apply(simple_model_names, axis=1)
model_names["wind"] = model_names["model_name"].str.contains("_wind")

color_map = cm.tab20(np.linspace(0, 1, len(model_names)))
colors = {key: color for key, color in zip(model_names["model_name_simple"], color_map)}
