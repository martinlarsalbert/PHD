"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# Instantiated project hooks.
# from phd.hooks import ProjectHooks

# HOOKS = (ProjectHooks(),)

from kedro.config import OmegaConfigLoader  # noqa: import-outside-toplevel

CONFIG_LOADER_CLASS = OmegaConfigLoader

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.shelvestore import ShelveStore
# SESSION_STORE_CLASS = ShelveStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
# CONFIG_LOADER_CLASS = ConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
# CONFIG_LOADER_ARGS = {
#    "config_patterns": {
#        "spark": ["spark*/"],
#        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
#    }
# }
# from kedro.config import TemplatedConfigLoader  # new import
#
# CONFIG_LOADER_CLASS = TemplatedConfigLoader
## CONFIG_LOADER_ARGS = {
##    "globals_pattern": "*globals.yml",
## }
## Class that manages the Data Catalog.
## from kedro.io import DataCatalog
## DATA_CATALOG_CLASS = DataCatalog
#
# from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore
# from pathlib import Path
#
# SESSION_STORE_CLASS = SQLiteStore
# SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2] / "data")}
## CONFIG_LOADER_ARGS = {"globals_pattern": "*globals.yml"}
