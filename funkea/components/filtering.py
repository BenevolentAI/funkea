r"""Set of filtering transforms.

The filtering transforms are used to filter the (annotation) input data. The filtering is done in a
composable way, so that multiple filters can be applied in sequence. The filtering transforms are
also useful for building more complex pipelines.
"""
import abc

import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame, Window

from funkea.core.utils import functions

__all__ = [
    "FilterOperation",
    "Identity",
    "QuantileCutoff",
    "MakeFull",
]

PrimitiveType = int | float | bool


class FilterOperation(Transformer, metaclass=abc.ABCMeta):
    r"""Base filter operation class."""
    pass


class Identity(FilterOperation):
    r"""The identity transform.

    This is useful for testing purposes or when no filtering is required.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>>
        >>> spark: SparkSession
        >>> df = spark.createDataFrame(
        ...     [("a", "x"), ("b", "x"), ("b", "z")],
        ...     "annotation string, partition string"
        ... )
        >>> Identity().transform(df).show()
        +----------+---------+
        |annotation|partition|
        +----------+---------+
        |         a|        x|
        |         b|        x|
        |         b|        z|
        +----------+---------+
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset


class QuantileCutoff(FilterOperation):
    r"""Keep only the records with a value above a certain percentile.

    Args:
        value_col: Column name for input values from which the percentile will be computed and
            filtering is performed.
        threshold: The percentile defining the lower bound of all values.
        partition_by: A sequence of column names by which to partition when computing the
            percentile.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>>
        >>> spark: SparkSession
        >>> df = spark.createDataFrame(
        ...     [("a", "x", 1), ("b", "x", 2), ("b", "z", 3)],
        ...     "annotation string, partition string, value int"
        ... )
        >>> (
        ...     QuantileCutoff(value_col="value", threshold=1.0, partition_by=("partition",))
        ...     .transform(df)
        ...     .show()
        ... )
        +----------+---------+-----+
        |annotation|partition|value|
        +----------+---------+-----+
        |         b|        x|    2|
        |         b|        z|    3|
        +----------+---------+-----+
    """

    def __init__(self, value_col: str, threshold: float, partition_by: tuple[str, ...] = ("id",)):
        super(QuantileCutoff, self).__init__()
        self.value_col = value_col
        self.threshold = threshold
        self.partition_by = partition_by

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return (
            dataset.withColumn(
                "quantile",
                F.percentile_approx(self.value_col, self.threshold).over(
                    Window.partitionBy(*self.partition_by)
                ),
            )
            .filter(F.col(self.value_col) >= F.col("quantile"))
            .drop("quantile")
        )


def _flatten(cols: tuple[str | list[str], ...]) -> list[str]:
    out: list[str] = []
    for col in cols:
        if isinstance(col, list):
            out.extend(col)
            continue
        out.append(col)
    return out


class MakeFull(FilterOperation):
    """Adds missing combinations.

    This is mostly useful for hard-partitioned annotation data. Since the data are in long table
    format, only partition-annotation pairs which exist will be in the table. However, in some cases
    having the negative cases is as important.

    Args:
        combination_cols: A sequence of column names or groups of column names, for which exhaustive
            combinations are required.
        fill_value: Optional value to fill for columns in the input dataset which are not in the
            combination columns. If None, no filling is applied. See
            ``pyspark.sql.DataFrame.fillna`` for more details. (``default=None``)

    Examples:
        >>> from pyspark.sql import SparkSession
        >>>
        >>> spark: SparkSession
        >>> df = spark.createDataFrame(
        ...     [("a", "x"), ("b", "x"), ("b", "z")],
        ...     "annotation string, partition string"
        ... )
        >>> (
        ...     MakeFull(combination_cols=("annotation", "partition"), fill_value=False)
        ...     .transform(df.withColumn("presence", F.lit(True)))
        ...     .show()
        ... )
        +----------+---------+--------+
        |annotation|partition|presence|
        +----------+---------+--------+
        |         a|        x|    true|
        |         b|        x|    true|
        |         a|        z|   false|
        |         b|        z|    true|
        +----------+---------+--------+
    """

    def __init__(
        self,
        combination_cols: tuple[str | list[str], ...],
        fill_value: PrimitiveType | str | None = None,
    ):
        super(MakeFull, self).__init__()
        self.combination_cols = combination_cols
        self.fill_value = fill_value

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if len(self.combination_cols) < 2:
            raise ValueError(
                "`combination_cols` has to have at least two elements to combine (currently "
                f"{len(self.combination_cols)})"
            )

        comb_col = "__comb_id__"
        combs: DataFrame = functions.get_session().range(1).withColumnRenamed("id", comb_col)
        for cols in self.combination_cols:
            combs = combs.crossJoin(dataset.select(cols).distinct()).drop(comb_col)

        comb_cols = _flatten(self.combination_cols)
        dataset = dataset.join(combs, on=comb_cols, how="right")
        if self.fill_value is not None:
            fill_cols: list[str] = sorted(set(dataset.columns).difference(comb_cols))
            dataset = dataset.fillna(value=self.fill_value, subset=fill_cols)
        return dataset


class Compose(FilterOperation):
    r"""Compose a sequence of filter operations.

    Args:
        steps: A sequence of filter operations to apply in sequence.

    Examples:
        Below is an example of how to use this class to filter out lowly active
        annotations-partition pairs and then fill in the missing combinations.

        >>> from pyspark.sql import SparkSession
        >>>
        >>> spark: SparkSession
        >>> df = spark.createDataFrame(
        ...     [("a", "x", 1), ("b", "x", 1), ("b", "z", 1)],
        ...     "annotation string, partition string, value int"
        ... )
        >>> (
        ...     Compose(
        ...         QuantileCutoff(value_col="value", threshold=0.5, partition_by=("partition",)),
        ...         MakeFull(combination_cols=("annotation", "partition"), fill_value=0),
        ...     )
        ...     .transform(df.withColumn("value", F.lit(1)))
        ...     .show()
        ... )
        +----------+---------+-----+
        |annotation|partition|value|
        +----------+---------+-----+
        |         a|        x|    1|
        |         b|        x|    1|
        |         a|        z|    0|
        |         b|        z|    1|
        +----------+---------+-----+
    """

    def __init__(self, *steps: FilterOperation):
        super(Compose, self).__init__()
        self.steps = steps

    def _transform(self, dataset: DataFrame) -> DataFrame:
        for step in self.steps:
            dataset = step.transform(dataset)
        return dataset
