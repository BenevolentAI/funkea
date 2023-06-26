import abc
import copy
import inspect
from typing import Any, Generic, Type, TypeVar

import pyspark.sql.functions as F
import pyspark.sql.types as T
import quinn
from py4j import java_gateway
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from typing_extensions import Self

from funkea.core import data
from funkea.core import method as method_
from funkea.core import params as params_
from funkea.core import pipeline as pipeline_
from funkea.core.utils import partition

Pipeline = TypeVar("Pipeline", bound=pipeline_.DataPipeline)
Method = TypeVar("Method", bound=method_.EnrichmentMethod)
Params = TypeVar("Params", bound=params_.Params)

Component = TypeVar("Component", bound=Transformer | data.DataComponent)


def _has_component(workflow: "Workflow", component: Type[Component]) -> bool:
    try:
        # JAX has a lot of issues being installed on various architectures, so we only import it
        # here if we actually need it.
        # Moreover, this feature is not _super_ important, so we can just skip it if JAX is not
        # installed.
        import jax
        import jax.tree_util
    except (ImportError, ModuleNotFoundError):
        return False

    def _f(obj: Any) -> bool | dict[str, bool]:
        if isinstance(obj, component):
            return True
        # `py4j.java_gateway` objects can have any attributes on them, even if they do not actually
        # exist. This can lead to infinite recursions.
        if not hasattr(obj, "__dict__") or inspect.getmodule(obj) is java_gateway:
            return False
        return jax.tree_map(_f, vars(obj))

    return any(jax.tree_util.tree_flatten(jax.tree_map(_f, vars(workflow)))[0])


class Workflow(
    Transformer,
    Generic[Pipeline, Method, Params],
    partition.PartitionByMixin,
    metaclass=abc.ABCMeta,
):
    """Base class for workflows.

    Args:
        pipeline: The data pipeline, which prepares the sumstats and annotation data for enrichment.
        method: The enrichment method, which uses the annotated loci to calculate enrichment for the
            :math:`K` partitions and compute the corresponding :math:`p`-value.
    """

    pipeline: Pipeline
    method: Method

    _expected_input_schema: T.StructType = data.Sumstats.spark_schema()

    def __init__(self, pipeline: Pipeline, method: Method):
        super(Workflow, self).__init__()
        self.pipeline = pipeline
        self.method = method
        self._expected_input_schema = self._get_expected_schema()

    def _get_expected_schema(self) -> T.StructType:
        # in the future we should delegate the required schema to each component and then just
        # collect these here.
        schema = copy.deepcopy(self._expected_input_schema)
        if _has_component(self, data.LDComponent):
            # if we are using LD components in the workflow, we should make sure to have an ancestry
            # column
            schema = schema.add(T.StructField(data.ANCESTRY_COL, T.StringType()))
        return schema

    def _init_workflow(self, dataset: DataFrame) -> DataFrame:
        for column in self.partition_cols:
            if column not in dataset.columns:
                # create dummy partitions for missing partition columns
                # this allows for generality, for example, when we only pass a single study
                # we add an artificial column for study ID, and then the same Spark code can work
                # for one or more studies
                # equally, we can then partition by any arbitrary set of columns
                dataset = dataset.withColumn(column, F.lit("dummy"))
        return dataset

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if len(self.partition_cols) == 0:
            raise ValueError(
                "Do not set empty partitioning, as this will break the workflow. Any missing "
                "partitioning columns will be added as dummies automatically. Hence, if no "
                "partitioning is necessary (i.e. only one study being run), it is recommended to "
                "just set a single (arbitrary) partitioning column (the default)."
            )
        quinn.validate_schema(dataset, self._expected_input_schema, ignore_nullable=True)

        # missing partition columns are added with `dummy` values, such that all partitioning parts
        # still pass
        # however, we need to make sure to remove these after completing the workflow
        missing_partition_cols: set[str] = set(self.partition_cols).difference(dataset.columns)
        return (
            self._init_workflow(dataset)
            .transform(self.pipeline.transform)
            .transform(self.method.transform)
            .sort(*self.partition_cols, "p_value")
            .drop(*missing_partition_cols)
        )

    @classmethod
    @abc.abstractmethod
    def default(
        cls,
        annotation: data.AnnotationComponent | None = None,
        params: Params = params_.Params(),  # type: ignore[assignment]
    ) -> Self:
        pass
