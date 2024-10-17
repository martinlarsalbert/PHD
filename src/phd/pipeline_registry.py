"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline
from .pipelines import (
    load_7m,
    filter,
    filter_wPCC,
    load_wPCC,
    regression_VCT,
    regression_ID,
    models,
    models_7m,
    resistance_MDL,
    added_mass_from_inverse_dynamics,
    regression_VCT_7m,
    captive,
    add_propeller,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    pipelines = {}
    ships=['wPCC','optiwise','DM']
    for ship_name in ships:
        pipelines[ship_name] = ship_pipeline(ship_name=ship_name)

    ### KVLCC2
    #pipelines["models_kvlcc2"] = pipeline(
    #    models.create_pipeline(), namespace="kvlcc2_hsva"
    #)
#
    ### 7m
    #pipelines["load_7m"] = load_7m.create_pipeline()
#
    ## pipelines["models_7m"] = pipeline(models_7m.create_pipeline())  # (Scaling MDL model to 7m scale)
    #pipelines["models_7m"] = pipeline(models_7m.create_pipeline(), namespace="7m")
#
    #pipelines["filter_7m"] = pipeline(filter.create_pipeline(), namespace="7m")
    #pipelines["regression_VCT_7m"] = pipeline(
    #    regression_VCT_7m.create_pipeline(), namespace="7m"
    #)

    ## TT (captive)
    pipelines["captive"] = pipeline(
        captive.create_pipeline(),
        inputs={"ship_data": "wPCC.ship_data"},
        namespace="TT",
    )

    # -------------------
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines

def ship_pipeline(ship_name:str):
    
    ## ship pipeline (wPCC)
    the_pipeline = (
        pipeline(models.create_pipeline(), namespace=ship_name)

        + pipeline(regression_VCT.create_pipeline(ship_name=ship_name), namespace=ship_name)

        + pipeline(add_propeller.create_pipeline(), namespace=ship_name)

        + pipeline(regression_ID.create_pipeline(), namespace=ship_name)
        
        + pipeline(pipe=load_wPCC.create_pipeline(), namespace=ship_name)
    
        + pipeline(pipe=filter_wPCC.create_pipeline(), namespace=ship_name)

        + pipeline(resistance_MDL.create_pipeline(), namespace=ship_name)
    
    )
    return the_pipeline