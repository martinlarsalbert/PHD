"""
This is a boilerplate pipeline 'predict'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import force_prediction_score, select_prediction_dataset_7m
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline


def the_pipeline(tags=[]):
    return pipeline(
        [
            node(
                func=force_prediction_score,
                inputs=[
                    f"models",
                    f"time_series",
                ],
                outputs="force_prediction_scores",
                name=f"force_prediction_scores",
                tags=["score", "predict"] + tags,
            )
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    wPCC_pipeline = modular_pipeline(
        the_pipeline(tags=["wPCC"]),
        namespace="wPCC",
        inputs={"time_series": "wPCC.time_series_preprocessed.ek_smooth"},
    )

    select_prediction_dataset_7m_pipeline = pipeline(
        [
            node(
                func=select_prediction_dataset_7m,
                inputs=[
                    f"7m.tests_ek_smooth",
                    f"7m.test_meta_data",
                ],
                outputs="7m.time_series_prediction",
                name=f"select_prediction_dataset_7m",
                tags=["score", "predict", "select"],
            )
        ]
    )

    _7m_pipeline = modular_pipeline(
        the_pipeline(tags=["7m"]),
        namespace="7m",
        inputs={"time_series": "7m.time_series_prediction"},
    )

    return wPCC_pipeline + select_prediction_dataset_7m_pipeline + _7m_pipeline
