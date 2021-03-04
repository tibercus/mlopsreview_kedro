from kedro.pipeline import node, Pipeline
import simple_photo_z.pipelines.feature_engineering.nodes as fe_nodes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                fe_nodes.calculate_features_on_dataset,
                inputs="dataset_trainX",
                outputs="features_trainX",
                name="calculating_features_trainX"
            ),
            node(
                fe_nodes.calculate_features_on_dataset,
                inputs="dataset_testS82X",
                outputs="features_testS82X",
                name="calculating_features_testS82X"
            )
        ]
    )