from kedro.pipeline import node, Pipeline
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner

# prepare catalog
data_catalog = DataCatalog({"ex_data": MemoryDataSet()})


# first node
def return_greeting():
    return "Hello, Space"


return_greeting_node = node(
    func=return_greeting, inputs=None, outputs="my_salutation"
)


# second node
def join_statements(greeting):
    return f"{greeting} Explorers!"


join_statements_node = node(
    join_statements, inputs="my_salutation", outputs="my_message"
)

# pipeline
pipeline = Pipeline([return_greeting_node, join_statements_node])

# run
runner = SequentialRunner()
print(runner.run(pipeline, data_catalog))
