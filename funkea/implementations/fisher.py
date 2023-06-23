from typing import Final, cast

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame
from scipy import special
from typing_extensions import Self

from funkea.components import filtering
from funkea.components import locus_definition as locus_def
from funkea.components import variant_selection
from funkea.core import data, method
from funkea.core import params as params_
from funkea.core import pipeline, workflow
from funkea.core.utils import types


class Pipeline(pipeline.DataPipeline):
    """A pipeline for the Fisher method.

    This pipeline performs the following steps:

    1. Selects variants based on the provided parameters.
    2. Defines the locus based on the provided parameters.
    3. Collects the annotations for across all loci for a given study.
    """

    query_col: Final[str] = "query"
    library_col: Final[str] = "library"
    space_size_col: Final[str] = "space_size"

    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = self.get_variant_selector().transform(dataset)

        annotation = self.locus_definition.annotation
        if annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure to set this during initialisation of "
                "`locus_definition`."
            )

        cols = annotation.columns
        queries: DataFrame = (
            self.locus_definition.transform(dataset)
            .groupBy(*self.partition_cols)
            .agg(F.collect_set(cols.annotation_id).alias(self.query_col))
        )
        library: DataFrame = (
            annotation.load()
            .select(cols.partition_id, cols.annotation_id)
            .groupBy(cols.partition_id)
            .agg(F.collect_set(cols.annotation_id).alias(self.library_col))
            # getting the total number of distinct annotations is easier in the pipeline
            .withColumn(self.space_size_col, F.lit(annotation.get_annotations_count()))
        )
        return queries.crossJoin(library).withColumn(
            "entry_id", F.concat(*self.partition_cols, cols.partition_id)
        )


def hyper_geom_test(contingency_table: DataFrame) -> DataFrame:
    """Performs a hypergeometric test on a set of contingency tables.

    The alternative hypothesis is that the odds ratio of the underlying population is greater than
    1.

    Args:
        contingency_table: A Spark dataframe containing the cells of the contingency table, where
            each row is a contingency table. It requires the following columns:
            ``a, b, c, d, entry_id``. The ``entry_id`` column should be a unique identifier for each
            contingency table. Note that it is *not* advised to use the
            ``pyspark.sql.monotonically_increasing_id`` function, as it is not reliable for join
            operations.

    Returns:
        The input dataframe along with a ``p_value`` column.

    Examples:
        >>> import pyspark.sql.functions as F
        >>> from pyspark.sql import SparkSession
        >>> spark: SparkSession
        >>>
        >>> contingency_table = spark.createDataFrame(
        ...     [
        ...         (1, 2, 3, 4, "a"),
        ...         (5, 6, 7, 8, "b"),
        ...         (9, 10, 11, 12, "c"),
        ...     ],
        ...     ["a", "b", "c", "d", "entry_id"],
        ... )
        >>> hyper_geom_test(contingency_table).show()
        +---+---+---+---+------------------+
        |  a|  b|  c|  d|           p_value|
        +---+---+---+---+------------------+
        |  1|  2|  3|  4|0.8333333333333344|
        |  5|  6|  7|  8|0.6759052362363788|
        |  9| 10| 11| 12| 0.632599492245731|
        +---+---+---+---+------------------+
    """

    @F.pandas_udf("double")  # type: ignore[call-overload]
    def log_factorial(e: pd.Series) -> pd.Series:
        return special.gammaln(e + 1)

    f = log_factorial
    a, b, c, d = F.col("a"), F.col("b"), F.col("c"), F.col("d")
    N = a + b + c + d

    numerator: Column = (f(a + b) + f(c + d) + f(a + c) + f(b + d)).alias("numerator")
    fN: Column = f(N).alias("fN")

    bi = F.col("bi")
    p_values = (
        contingency_table.select(
            a,
            b,
            c,
            d,
            fN,
            numerator,
            # the +1 is to account for the fact that the range is inclusive
            # we need to ensure that `b` is of type int, otherwise the range function will fail
            F.posexplode(F.array_repeat("entry_id", b.cast("int") + 1)).alias("bi", "entry_id"),
        )
        .groupBy("entry_id")
        .agg(
            F.sum(
                F.exp(numerator - (f(bi) + f(a + b - bi) + f(b + d - bi) + f(c - b + bi) + fN))
            ).alias("p_value")
        )
    )
    return contingency_table.join(p_values, on="entry_id", how="inner").drop("entry_id")


class Method(method.EnrichmentMethod):
    """A method for the Fisher method.

    Thin wrapper around the ``hyper_geom_test`` function. Takes the overlapped annotations and then
    generates the contingency table for each partition. The contingency table is then passed to the
    ``hyper_geom_test`` function, which returns the p-value for each partition.
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        a, b, c = F.col("a"), F.col("b"), F.col("c")
        contingency_table = (
            dataset.withColumn(
                "a", F.size(F.array_intersect(Pipeline.query_col, Pipeline.library_col))
            )
            .withColumn("b", F.size(Pipeline.library_col) - a)
            .withColumn("c", F.size(Pipeline.query_col) - a)
            .withColumn("d", F.col(Pipeline.space_size_col) - (a + b + c))
            .drop(Pipeline.query_col, Pipeline.library_col, Pipeline.space_size_col)
        )
        return (
            hyper_geom_test(contingency_table)
            .drop("b", "c", "d")
            .withColumnRenamed("a", "enrichment")
        )


class Params(params_.Params):
    """Parameters for the default configuration of ``Fisher``.

    Args:
        association_threshold: The :math:`p`-value threshold used to select genome-wide significant
            variants. (``default=5e-8``)
    """

    association_threshold: types.UnitFloat = cast(
        types.UnitFloat, variant_selection.DEFAULT_ASSOCIATION_THRESHOLD
    )


class Fisher(workflow.Workflow[Pipeline, Method, Params]):
    r"""Computes functional enrichment using Fisher's exact test.

    The hypergeomtric test, or Fisher's exact test [1]_, is a naive functional enrichment method,
    which computes the significance in the overlap between "enriched" subset
    :math:`\hat{\mathcal{S}} \in \mathcal{G}` and "true" subsets
    :math:`\mathcal{S}_i \in \mathcal{G}, \forall \, i \in \{1, \dots, K\}`. Hence, the enrichment
    :math:`e_i` of a given study for partition :math:`i` of :math:`\mathcal{G}` is defined as

    .. math::
        e_i = |\hat{\mathcal{S}} \cap \mathcal{S}_i|

    where :math:`\hat{\mathcal{S}}` is the set of :math:`s_{ij}` overlapped by all the genome-wide
    significant variants in the study. The significance of :math:`e_i` is assumed to be distributed
    hypergeomtrically, and hence

    .. math::
        a &= e_i \\
        b &= |\mathcal{S}_i| - e_i \\
        c &= |\hat{\mathcal{S}}| - e_i \\
        d &= |\mathcal{G}| - (a + b + c) \\
        p_i &= \sum_{j = 0}^b \frac{(a + b)! (a + c)! (c + d)! (b + d)!}{j! (a + b - j)! (b + d - j)! (c - b + j)! (a + b + c + d)!}

    The Fisher's method can be extended with linkage disequilibrium (LD) pruning (for variant
    selection) and / or FDR correction (for enrichment selection). However, we have found these
    modifications generally did not improve enrichment results.

    References:
        .. [1] Fisher, R.A., 1922. On the interpretation of χ 2 from contingency tables, and the
            calculation of P. Journal of the royal statistical society, 85(1), pp.87-94.
    """

    @classmethod
    def default(
        cls, annotation: data.AnnotationComponent | None = None, params: Params = Params()
    ) -> Self:
        """Returns the default workflow for Fisher's exact test.

        Args:
            annotation: The annotation component to use for the enrichment. If ``None``, then the
                default annotation component is used.
            params: The parameters for the workflow.

        Returns:
            The default workflow for Fisher's exact test.
        """
        if annotation is None:
            annotation = data.GTEx(
                filter_operation=filtering.QuantileCutoff(
                    value_col="pem", threshold=0.9, partition_by=(data.GTEx.columns.partition_id,)
                )
            )
            annotation.columns.values = "pem"
        return cls(
            pipeline=Pipeline(
                locus_def.Overlap(annotation),
                variant_selection.Compose(
                    variant_selection.AssociationThreshold(
                        threshold=params.association_threshold,
                    ),
                    variant_selection.DropHLA(),
                    variant_selection.DropComplement(),
                    variant_selection.DropIndel(),
                ),
            ),
            method=Method(),
        )
