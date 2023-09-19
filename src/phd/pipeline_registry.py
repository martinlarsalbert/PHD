"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline
from .pipelines import load_7m, filter, filter_wPCC


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["load_7m"] = load_7m.create_pipeline()
    pipelines["filter"] = pipeline(pipe=filter.create_pipeline(), namespace="7m")
    pipelines["filter_wPCC"] = pipeline(
        pipe=filter_wPCC.create_pipeline(), namespace="wPCC"
    )

    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
