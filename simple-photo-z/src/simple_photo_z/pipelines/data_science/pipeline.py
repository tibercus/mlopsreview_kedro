from kedro.pipeline import node, Pipeline
from simple_photo_z.pipelines.data_science import nodes as ds_nodes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                ds_nodes.get_model_features,
                inputs="features_trainX",
                outputs=["X_train", "y_train"]
            ),
            node(
                ds_nodes.train_model,
                inputs=["X_train", "y_train"],
                outputs="model"
            ),
            node(
                ds_nodes.get_model_features,
                inputs="features_testS82X",
                outputs=["X_test", "y_test"]
            ),
            node(
                ds_nodes.evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs=None
            )
        ]
    )
