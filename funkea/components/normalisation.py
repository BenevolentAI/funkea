r"""Set of normalisation transforms."""
import abc

import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame, Window

__all__ = [
    "Normalisation",
    "Identity",
    "StandardScaler",
    "QuantileNorm",
    "EuclidNorm",
    "Compose",
]


class Normalisation(Transformer, metaclass=abc.ABCMeta):
    r"""Base class for normalisation transforms."""

    def __init__(self, values_col: str, output_col: str, partition_by: tuple[str, ...] = ("id",)):
        super(Normalisation, self).__init__()
        self.values_col = values_col
        self.output_col = output_col
        self.partition_by = partition_by


class Identity(Normalisation):
    r"""The identity transform."""

    def __init__(self):
        super(Identity, self).__init__(
            values_col="",
            output_col="",
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset


class StandardScaler(Normalisation):
    r"""Computes :math:`z`-scores.

    Unlike the official implementation in ``pyspark.ml``, this version allows for partitioning the
    dataframe arbitrarily for computing the mean and standard deviation. It is defined in the
    standard way:

    .. math::

        z = \frac{x - \mu}{\sigma}

    where :math:`\mu` and :math:`\sigma` are the sample mean and standard deviation, respectively.

    Notes:
        :math:`\sigma` is the *unbiased* estimator of the population standard deviation.
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        window = Window.partitionBy(*self.partition_by)
        x = F.col(self.values_col)
        mu = F.mean(x).over(window)
        sigma = F.stddev_samp(x).over(window)
        return dataset.withColumn(self.output_col, (x - mu) / sigma).fillna(
            value=0.0, subset=self.output_col
        )


class QuantileNorm(Normalisation):
    r"""Performs quantile normalisation.

    Quantile normalisation [1]_ is a method for matching the quantiles of multiple distributions.
    Briefly, it first assigns the values in each partition to their quantiles, takes mean of the
    quantiles over the partitions, and then replaces the original values with the averaged
    quantiles.

    References:
        .. [1] https://en.wikipedia.org/wiki/Quantile_normalization

    Examples:
        >>> from funkea.components.normalisation import QuantileNorm
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = spark.createDataFrame(
        ...     [
        ...         (1, 1, 1.0),
        ...         (1, 2, 2.0),
        ...         (1, 3, 3.0),
        ...         (2, 1, 4.0),
        ...         (2, 2, 5.0),
        ...         (2, 3, 6.0),
        ...     ],
        ...     ["id", "feature", "value"],
        ... )
        >>> df.show()
        +---+-------+-----+
        | id|feature|value|
        +---+-------+-----+
        |  1|      1|  1.0|
        |  1|      2|  2.0|
        |  1|      3|  3.0|
        |  2|      1|  4.0|
        |  2|      2|  5.0|
        |  2|      3|  6.0|
        +---+-------+-----+
        >>> QuantileNorm("value", "norm_value").transform(df).show()
        +---+-------+-----+----------+
        | id|feature|value|norm_value|
        +---+-------+-----+----------+
        |  1|      1|  1.0|       2.5|
        |  1|      2|  2.0|       3.5|
        |  1|      3|  3.0|       4.5|
        |  2|      1|  4.0|       2.5|
        |  2|      2|  5.0|       3.5|
        |  2|      3|  6.0|       4.5|
        +---+-------+-----+----------+
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        window = Window.partitionBy(*self.partition_by).orderBy(self.values_col)
        return (
            dataset.withColumn(
                "rank",
                F.rank().over(window),
            )
            # if `output_col` already exists in `dataset`, we need to drop it to avoid duplicated
            # column names. If it does not exist, `drop` will have no effect
            .drop(self.output_col)
            .join(
                dataset.withColumn("rank", F.row_number().over(window))
                .groupBy("rank")
                .agg(F.mean(self.values_col).alias(self.output_col)),
                on="rank",
                how="inner",
            )
            .drop("rank")
        )


class EuclidNorm(Normalisation):
    r"""Normalises partitions into unit-length.

    For a given partition represented as vector :math:`\mathbf{x}`, the euclid normalisation is
    defined in the standard way:

    .. math::
        \bar{\mathbf{x}} = \frac{1}{||\mathbf{x}||_2} \mathbf{x}

    Notes:
        If a certain partition produces :math:`||\mathbf{x}||_2 = 0`, :math:`\hat{\mathbf{x}}` will
        be set to :math:`0` for each element.
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        x = F.col(self.values_col)
        return dataset.withColumn(
            self.output_col, x / F.sqrt(F.sum(x**2).over(Window.partitionBy(*self.partition_by)))
        ).fillna(value=0.0, subset=[self.output_col])


class Compose(Normalisation):
    """Composes a sequence of normalisation steps.

    This is a convenience class that allows for composing multiple normalisation steps into a single
    transformer.

    Args:
        *steps: A sequence of normalisation steps.

    Examples:
        >>> from funkea.components.normalisation import Compose, StandardScaler, QuantileNorm
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = spark.createDataFrame(
        ...     [
        ...         (1, 1, 1.0),
        ...         (1, 2, 2.0),
        ...         (1, 3, 3.0),
        ...         (2, 1, 4.0),
        ...         (2, 2, 5.0),
        ...         (2, 3, 6.0),
        ...     ],
        ...     ["id", "feature", "value"],
        ... )
        >>> df.show()
        +---+-------+-----+
        | id|feature|value|
        +---+-------+-----+
        |  1|      1|  1.0|
        |  1|      2|  2.0|
        |  1|      3|  3.0|
        |  2|      1|  4.0|
        |  2|      2|  5.0|
        |  2|      3|  6.0|
        +---+-------+-----+
        >>> normaliser = Compose(
        ...     StandardScaler(values_col="value", output_col="scaled", partition_by=("id",)),
        ...     QuantileNorm(values_col="scaled", output_col="quantile_norm", partition_by=("id",)),
        ... )
        >>> normaliser.transform(df).show()
        +---+-------+-----+------+-------------+
        | id|feature|value|scaled|quantile_norm|
        +---+-------+-----+------+-------------+
        |  1|      1|  1.0|  -1.0|         -1.0|
        |  1|      2|  2.0|   0.0|          0.0|
        |  1|      3|  3.0|   1.0|          1.0|
        |  2|      1|  4.0|  -1.0|         -1.0|
        |  2|      2|  5.0|   0.0|          0.0|
        |  2|      3|  6.0|   1.0|          1.0|
        +---+-------+-----+------+-------------+
    """

    def __init__(self, *steps: Normalisation):
        self.steps = steps
        super(Compose, self).__init__(
            values_col="",
            output_col="",
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        for step in self.steps:
            dataset = step.transform(dataset)
        return dataset
