from kedro.pipeline import node, Pipeline
import simple_photo_z.pipelines.data_engineering.nodes as de_nodes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # trainX
            node(
                de_nodes.preprocess_xray_catalog,
                inputs="trainX_sample",
                outputs="preprocessed_trainX_sample",
                name="preprocessing_trainX_sample"
            ),
            node(
                de_nodes.preprocess_desilis,
                inputs="trainX_ls",
                outputs="preprocessed_trainX_ls",
                name="preprocessing_trainX_ls"
            ),
            node(
                de_nodes.preprocess_panstarrs,
                inputs="trainX_ps",
                outputs="preprocessed_trainX_ps",
                name="preprocessing_trainX_ps"
            ),
            node(
                de_nodes.preprocess_sdss,
                inputs="trainX_sdss",
                outputs="preprocessed_trainX_sdss",
                name="preprocessing_trainX_sdss"
            ),
            node(
                de_nodes.create_master_dataset,
                inputs=["preprocessed_trainX_sample",
                        "preprocessed_trainX_ls",
                        "preprocessed_trainX_ps",
                        "preprocessed_trainX_sdss"],
                outputs="dataset_trainX"
            ),

            # Stripe82X
            node(
                de_nodes.preprocess_xray_catalog,
                inputs="testS82X_sample",
                outputs="preprocessed_testS82X_sample",
                name="preprocessing_testS82X_sample"
            ),
            node(
                de_nodes.preprocess_desilis,
                inputs="testS82X_ls",
                outputs="preprocessed_testS82X_ls",
                name="preprocessing_testS82X_ls"
            ),
            node(
                de_nodes.preprocess_panstarrs,
                inputs="testS82X_ps",
                outputs="preprocessed_testS82X_ps",
                name="preprocessing_testS82X_ps"
            ),
            node(
                de_nodes.preprocess_sdss,
                inputs="testS82X_sdss",
                outputs="preprocessed_testS82X_sdss",
                name="preprocessing_testS82X_sdss"
            ),
            node(
                de_nodes.create_master_dataset,
                inputs=["preprocessed_testS82X_sample",
                        "preprocessed_testS82X_ls",
                        "preprocessed_testS82X_ps",
                        "preprocessed_testS82X_sdss"],
                outputs="dataset_testS82X"
            )
        ]
    )
