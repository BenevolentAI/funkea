import pyspark.sql.functions as F
import pytest

from funkea.components import locus_definition
from funkea.core import method, pipeline, workflow


@pytest.fixture(scope="function")
def mock_abc(mock_abstract_methods):
    mock_abstract_methods(pipeline.DataPipeline)
    mock_abstract_methods(method.EnrichmentMethod)
    mock_abstract_methods(workflow.Workflow)


@pytest.fixture(scope="function")
def mock_component_transforms(mock_transform):
    mock_transform(pipeline.DataPipeline)
    mock_transform(method.EnrichmentMethod, lambda _, df: df.withColumn("p_value", F.lit(0.05)))


@pytest.fixture(scope="function")
def example_workflow(mock_abc, mock_component_transforms) -> workflow.Workflow:
    return workflow.Workflow(
        pipeline=pipeline.DataPipeline(locus_definition.Identity()),
        method=method.EnrichmentMethod(),
    )


class TestWorkflow:
    def test__init_workflow(self, example_workflow, sumstats):
        with example_workflow.partition_by("hello"):
            res = example_workflow._init_workflow(sumstats)

        assert set(res.columns).difference(sumstats.columns) == {"hello"}
        assert res.select("hello").distinct().count() == 1

    def test__transform(self, example_workflow, sumstats):
        with example_workflow.partition_by("hello"):
            res = example_workflow._transform(sumstats)
        assert "p_value" in res.columns
        assert "hello" not in res.columns

    def test_raises_value_error(self, example_workflow, sumstats):
        with pytest.raises(ValueError, match="Do not set empty partitioning"):
            with example_workflow.partition_by():
                example_workflow._transform(sumstats)
