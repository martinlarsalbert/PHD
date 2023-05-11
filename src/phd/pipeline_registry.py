"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines import load_7m


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["load_7m"] = load_7m.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
