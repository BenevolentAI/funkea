import abc
import functools
from typing import Final, Type, cast

import pandas as pd
import pydantic
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Transformer
from pyspark.sql import Column, DataFrame, Window
from scipy import special
from typing_extensions import Self

from funkea.components import locus_definition as locus_def
from funkea.components import normalisation as norm
from funkea.components import variant_selection
from funkea.core import data, method
from funkea.core import params as params_
from funkea.core import pipeline, workflow
from funkea.core.utils import files, functions, partition, types
from funkea.implementations.depict import _create_permutation

_DEFAULT_N_PERMUTATIONS: Final[pydantic.NonNegativeInt] = cast(pydantic.NonNegativeInt, 500)
_DEFAULT_EXACT_MATCH_CUTOFF: Final[pydantic.NonNegativeInt] = cast(pydantic.NonNegativeInt, 10)


class CollectExpandedIfEmpty(locus_def.Collect):
    """Collects annotations from expanded locus if locus does not overlap with any annotation.

    This transform is similar to :class:`.Collect`, but collects annotations from the expanded locus
    if the locus does not overlap with any annotation. This is useful for methods that require at
    least one annotation per locus, such as :class:`.SNPsea`.

    Args:
        extension: The number of base pairs to extend the locus by.
        annotation: The annotation component to collect annotations from.

    Raises:
        ValueError: If ``annotation`` is ``None``.
        RuntimeError: If either ``start`` or ``end`` is missing from the input dataframe.
    """

    def __init__(
        self,
        extension: tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt],
        annotation: data.AnnotationComponent | None = None,
    ):
        self.extension = extension
        super(CollectExpandedIfEmpty, self).__init__(what="annotation", annotation=annotation)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.annotation is None:
            raise ValueError("`annotation` is None. Make sure to set this during initialisation.")

        cols = self.locus_columns
        ann_cols = self.annotation.columns
        start, end = self.extension

        pos_col = {ann_cols.start, ann_cols.end}
        if pos_col.intersection(dataset.columns) != pos_col:
            raise RuntimeError(
                f"Either {ann_cols.start!r} or {ann_cols!r} is missing from the input dataframe. "
                "Make sure to apply `Overlap` before applying this transform, to associate "
                "variants with annotations."
            )

        exp_col = "expanded_overlap"
        dataset = dataset.withColumn(
            exp_col,
            ~functions.overlap(
                F.col(ann_cols.start),
                F.col(ann_cols.end),
                F.col(cols.start) + start,
                F.col(cols.end) - end,
            ),
        )
        with self.partition_by(*self.get_partition_cols(exp_col)):
            dataset = super(CollectExpandedIfEmpty, self)._transform(dataset)
        return (
            dataset.filter(F.size(self.collected_col) > 0)
            .withColumn(
                "rank",
                F.rank().over(self.get_window(cols.id).orderBy(F.col(exp_col).cast("byte"))),
            )
            .filter(F.col("rank") == 1)
            .drop("rank", exp_col)
        )


class PercentileAssignment(norm.Normalisation):
    """Assigns percentiles to a column.

    Args:
        values_col: The column to assign percentiles to.
        output_col: The column to store the percentiles in.
        partition_by: The columns to partition by. Percentiles are assigned within each partition.
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return (
            dataset.withColumn(
                "rank",
                F.rank().over(Window.partitionBy(*self.partition_by).orderBy(self.values_col)),
            )
            .join(
                dataset.groupBy(*self.partition_by).count(),
                on=list(self.partition_by),
                how="inner",
            )
            .withColumn(self.output_col, 1 - ((F.col("rank") - 1) / F.col("count")))
            .drop("rank", "count")
        )


class BackgroundVariants(data.DataComponent, partition.PartitionByMixin):
    """A component that loads background variants.

    This component loads background variants from a dataset. The dataset is used to determine the
    significance of the observed overlap between variants and annotations.

    Args:
        dataset: The dataset to load background variants from.
    """

    def __init__(self, dataset: data.Dataset | None = files.default.snpsea_background_variants):
        super(BackgroundVariants, self).__init__(dataset=dataset)

    def _load(self) -> DataFrame:
        dataset = self._get_dataset()
        for part in self.partition_cols:
            dataset = dataset.withColumn(part, F.lit("dummy"))
        return dataset


class Pipeline(pipeline.DataPipeline):
    """A data pipeline for SNPsea.

    Args:
        locus_definition: The locus definition component.
        variant_selector: The variant selection component.
        background_variants: The background variants component. The background variants are used to
            determine the significance of the observed overlap between variants and annotations.
        n_permutations: The number of permutations to perform. The default is 500. Increasing this
            number will increase the accuracy of the p-values, but will also increase the runtime.
    """

    id_col: str = "locus_id"
    locus_type: str = "locus_type"
    score_col: str = "score"

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        locus_definition: locus_def.LocusDefinition,
        variant_selector: variant_selection.VariantSelection | None = None,
        background_variants: BackgroundVariants = BackgroundVariants(),
        n_permutations: pydantic.NonNegativeInt = _DEFAULT_N_PERMUTATIONS,
        exact_match_cutoff: pydantic.NonNegativeInt = _DEFAULT_EXACT_MATCH_CUTOFF,
    ):
        self.background_variants = background_variants
        self.n_permutations = n_permutations
        self.exact_match_cutoff = exact_match_cutoff
        super(Pipeline, self).__init__(
            locus_definition=locus_definition,
            variant_selector=variant_selector,
        )

    def get_matched_null_loci(self, loci: DataFrame) -> DataFrame:
        """Get the null loci that match the given loci.

        Matching is performed by matching the number of annotations in each locus. The null loci are
        matched exactly if the number of annotations in the locus is less than or equal to the
        `exact_match_cutoff`. Otherwise, the null loci are matched by sampling from the null loci
        with the closest number of annotations.

        Args:
            loci: The loci to match null loci to.
        """
        null_loci = self.locus_definition.transform(self.background_variants.load())

        collection_col = CollectExpandedIfEmpty.collected_col
        exact_match_col = "match_exactly"

        def _exact_match(dataframe: DataFrame) -> DataFrame:
            return dataframe.withColumn("size", F.size(collection_col)).withColumn(
                exact_match_col, F.col("size") <= self.exact_match_cutoff
            )

        nil = null_loci.transform(_exact_match)
        true = loci.transform(_exact_match)

        cols = self.locus_definition.locus_columns
        return (
            true.filter(F.col(exact_match_col))
            .select(*self.get_partition_cols(cols.id), "size")
            .join(
                nil.filter(F.col(exact_match_col)).drop(*self.get_partition_cols(cols.id)),
                on="size",
            )
            .unionByName(
                true.filter(~F.col(exact_match_col))
                .select(*self.get_partition_cols(cols.id))
                .crossJoin(
                    nil.filter(~F.col(exact_match_col)).drop(*self.get_partition_cols(cols.id))
                )
            )
            .groupBy(*self.partition_cols, cols.id)
            .agg(
                F.collect_list(CollectExpandedIfEmpty.collected_col).alias(
                    CollectExpandedIfEmpty.collected_col
                )
            )
            .crossJoin(
                functions.get_session()
                .range(0, self.n_permutations)
                .select(F.concat(F.lit("null"), F.col("id").cast("string")).alias(self.locus_type))
            )
            .groupBy(*self.partition_cols, self.locus_type)
            .applyInPandas(
                functools.partial(
                    _create_permutation,
                    locus_id_col=cols.id,
                    null_locus_col=locus_def.Collect.collected_col,
                    partition_cols=self.partition_cols + (self.locus_type,),
                ),
                loci.select(
                    *self.partition_cols, cols.id, locus_def.Collect.collected_col
                ).schema.add(T.StructField(self.locus_type, T.StringType())),
            )
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        annotation = self.locus_definition.annotation
        if annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure to set this during initialisation of "
                "`locus_definition`."
            )

        dataset = self.get_variant_selector().transform(dataset)
        loci = self.locus_definition.transform(dataset)
        if locus_def.Collect.collected_col not in loci.columns:
            raise RuntimeError(
                f"{locus_def.Collect.collected_col!r} not in the loci dataframe. Make sure to "
                "add a `Collect` step to `locus_definition`."
            )
        null_loci = self.get_matched_null_loci(loci)

        cols = self.locus_definition.locus_columns
        ann_cols = annotation.columns

        annotation_data = annotation.load()
        values_col = ann_cols.values or self.score_col

        space_size: str | Column = BinarySpecificity.space_size_col
        if annotation.hard_partitions:
            # Hard partitioned data could have a `values` column present. This would cause ambiguity
            # downstream and cause a crash. Hence, we drop it here in order to replace (no operation
            # if there is not such column)
            annotation_data = annotation_data.drop(values_col).join(
                annotation_data.groupBy(ann_cols.partition_id)
                .count()
                .withColumnRenamed("count", values_col),
                on=ann_cols.partition_id,
                how="inner",
            )
            space_size = F.lit(annotation.get_annotations_count()).alias(
                BinarySpecificity.space_size_col
            )
        return (
            # subset columns in `loci` to avoid potential discrepancy with `null_loci`
            loci.select(*self.partition_cols, cols.id, locus_def.Collect.collected_col)
            .withColumn(self.locus_type, F.lit("true"))
            .unionByName(null_loci)
            .select(
                *self.partition_cols,
                F.col(cols.id).alias(self.id_col),
                self.locus_type,
                F.explode(CollectExpandedIfEmpty.collected_col).alias(ann_cols.annotation_id),
            )
            .join(
                annotation_data.select(
                    ann_cols.partition_id,
                    ann_cols.annotation_id,
                    F.col(values_col).alias(self.score_col),
                    *(space_size,) * annotation.hard_partitions,
                ),
                on=ann_cols.annotation_id,
            )
        )


class SpecificityScore(Transformer, partition.PartitionByMixin, metaclass=abc.ABCMeta):
    """Base class for specificity scores.

    This class is not meant to be used directly. Instead, use one of the subclasses. Specificity
    scores are computed for each locus and partition combination.
    """

    score_col: str = "score"

    def __init__(
        self,
        locus_id_col: str,
        partition_id_col: str,
        annotation_id_col: str,
        values_col: str,
    ):
        self.locus_id_col = locus_id_col
        self.partition_id_col = partition_id_col
        self.annotation_id_col = annotation_id_col
        self.values_col = values_col
        super(SpecificityScore, self).__init__()


class ContinuousSpecificity(SpecificityScore):
    r"""Compute specificity scores for continuous annotations (i.e. soft partitions).

    The specificity score is computed as the Bonferroni-corrected minimum value of the annotations
    in a locus:

    .. math::

        K_{il} &= 1 - \left(1 - \langle \bar{\mathbf{X}}_i, \mathbf{L}_l \rangle \right) ^ {c_l} \\
        \langle \bar{\mathbf{X}}_{i}, \mathbf{L}_{l} \rangle &:= \min_j \bar{X}_{ij} ^ {L_{jl}} \\
        c_l &= \sum_j^{|\mathcal{G}|} L_{jl}

    where :math:`\bar{\mathbf{X}} \in \mathbb{R} ^ {K \times |\mathcal{G}|}` and
    :math:`\mathbf{L} \in \{0, 1\} ^ {|\mathcal{G}| \times M}` are the (normalised) annotation and
    locus matrices, respectively, and :math:`\mathcal{G}` is the set of all possible annotations.

    Args:
        locus_id_col: The name of the locus ID column.
        partition_id_col: The name of the partition ID column.
        annotation_id_col: The name of the annotation ID column.
        values_col: The name of the values column.
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.groupBy(
            *self.get_partition_cols(self.partition_id_col, self.locus_id_col)
        ).agg(
            # NOTE: the assumption here is that we have all partition-annotation combinations.
            # This is not necessarily the case, in which case the exponent below is _wrong_. It
            # should be the total number of unique annotations in a given locus (rather than the
            # number of annotations in a locus of a particular partition -- though assuming
            # "fullness", we would get the same number)
            (1 - (1 - F.min(self.values_col)) ** F.count(self.annotation_id_col)).alias(
                self.score_col
            )
        )


class BinarySpecificity(SpecificityScore):
    r"""Compute specificity scores for binary annotations (i.e. hard partitions).

    The specificity score is computed as:

    .. math::

        K_{il} =
        \begin{cases}
            1 - \frac{{|\mathcal{G}| - n_i \choose c_l}}{{|\mathcal{G}| \choose c_l}} & \text{if } \sum_j^{|\mathcal{G}|} X_{ij} L_{jl} \, > 0 \\
            1 & \text{otherwise}
        \end{cases}

    where :math:`n_i = \sum_j^{|\mathcal{G}|} X_{ij}` (the number of annotations in a given
    partition :math:`i`). The binomial coefficient is computed using the log-gamma function
    :math:`\log \Gamma(x) = \log((x - 1)!)`. This is done to avoid overflow when computing the
    binomial coefficient.

    Args:
        locus_id_col: The name of the locus ID column.
        partition_id_col: The name of the partition ID column.
        annotation_id_col: The name of the annotation ID column.
        values_col: The name of the values column.
    """
    space_size_col: str = "space_size"

    def _transform(self, dataset: DataFrame) -> DataFrame:
        @F.pandas_udf("double")  # type: ignore[call-overload]
        def log_factorial(e: pd.Series) -> pd.Series:
            return special.gammaln(e + 1)

        f = log_factorial
        M = F.col("M")
        n = F.col("n")
        # By default, Spark column names are not case-sensitive.
        # Hence, `n` and `N` will be interpreted as the same, and give an error when referenced.
        N = F.col("N_")
        unique_locus_parts = self.get_partition_cols(self.locus_id_col)
        return (
            dataset.select(
                *unique_locus_parts,
                self.partition_id_col,
                F.col(self.space_size_col).alias("M"),
                F.col(self.values_col).alias("n"),
            )
            .distinct()
            .join(
                dataset.groupBy(
                    *unique_locus_parts,
                )
                .count()
                .withColumnRenamed("count", "N_"),
                on=list(unique_locus_parts),
            )
            .withColumn(self.score_col, 1 - F.exp(f(M - n) + f(M - N) - f(M) - f(M - n - N)))
            .drop("M", "n", "N_")
        )


class Method(method.EnrichmentMethod):
    """Compute specificity scores for SNPsea.

    Args:
        partition_id_col: The name of the partition ID column.
        annotation_id_col: The name of the annotation ID column.
        partition_type: The type of partition.
        locus_id_col: The name of the locus ID column.
        score_col: The name of the score column. Defaults to "score". Score here refers to the
            to the annotation-partition value.
        locus_type_col: The name of the locus type column. Defaults to "locus_type". Locus type
            refers to the type of locus (e.g. "true", "null").
        n_permutations: The number of permutations to perform. Defaults to 500. Sets the number of
            permutations to perform when computing the empirical p-value. Higher values will
            increase the accuracy of the p-value, but will also increase the computation time.
    """

    def __init__(
        self,
        partition_id_col: str,
        annotation_id_col: str,
        partition_type: data.PartitionType,
        locus_id_col: str = Pipeline.id_col,
        score_col: str = Pipeline.score_col,
        locus_type_col: str = Pipeline.locus_type,
        n_permutations: pydantic.NonNegativeInt = _DEFAULT_N_PERMUTATIONS,
    ):
        self.partition_id_col = partition_id_col
        self.annotation_id_col = annotation_id_col
        self.partition_type = partition_type
        self.locus_id_col = locus_id_col
        self.score_col = score_col
        self.locus_type_col = locus_type_col
        self.n_permutations = n_permutations
        super(Method, self).__init__()

    def _transform(self, dataset: DataFrame) -> DataFrame:
        specificity_scorer: Type[SpecificityScore] = {  # type: ignore[type-abstract]
            data.PartitionType.SOFT: ContinuousSpecificity,
            data.PartitionType.HARD: BinarySpecificity,
        }[self.partition_type]

        with self.partition_by(*self.partition_cols, Pipeline.locus_type):
            enrichment = (
                specificity_scorer(
                    locus_id_col=self.locus_id_col,
                    partition_id_col=self.partition_id_col,
                    annotation_id_col=self.annotation_id_col,
                    values_col=self.score_col,
                )
                .transform(dataset)
                .groupBy(*self.get_partition_cols(self.partition_id_col))
                .agg((-F.sum(F.log(specificity_scorer.score_col))).alias("enrichment"))
            )

        return (
            enrichment.filter(F.col(self.locus_type_col) == "true")
            .drop(self.locus_type_col)
            .join(
                enrichment.filter(F.col(self.locus_type_col) != "true").select(
                    *self.get_partition_cols(self.partition_id_col),
                    F.col("enrichment").alias("null_enrichment"),
                ),
                on=list(self.get_partition_cols(self.partition_id_col)),
                how="left",
            )
            .fillna(value=0.0, subset=["null_enrichment"])
            .groupBy(*self.get_partition_cols(self.partition_id_col))
            .agg(
                F.first("enrichment").alias("enrichment"),
                (
                    (F.sum((F.col("enrichment") <= F.col("null_enrichment")).cast("byte")) + 1)
                    / (self.n_permutations + 1)
                ).alias("p_value"),
            )
            .select(*self.partition_cols, self.partition_id_col, "enrichment", "p_value")
        )


class Params(params_.Params):
    """Parameters for SNPsea.

    Args:
        expansion_window: The window size to use when expanding the locus set. Defaults to 10,000.
            This is the number of base pairs to expand the locus set by in each direction. If the
            locus does not contain any annotations, then the locus will be expanded once more.
        expansion_r2_threshold: The R2 threshold to use when expanding the locus set. Defaults to
            0.5. This is the R2 threshold to use when expanding the locus set, i.e. the minimum
            R2 value between the lead variant and the variants in the locus set.
        association_threshold: The association threshold to use when selecting the lead variants.
            Defaults to :math:`5e-8`.
        n_permutations: The number of permutations to perform. Defaults to 500. Sets the number of
            permutations to perform when computing the empirical p-value. Higher values will
            increase the accuracy of the p-value, but will also increase the computation time.
    """

    expansion_window: pydantic.NonNegativeInt = cast(pydantic.NonNegativeInt, 10_000)
    expansion_r2_threshold: types.UnitFloat = cast(types.UnitFloat, 0.5)
    association_threshold: types.UnitFloat = cast(
        types.UnitFloat, variant_selection.DEFAULT_ASSOCIATION_THRESHOLD
    )
    n_permutations: pydantic.NonNegativeInt = _DEFAULT_N_PERMUTATIONS


class SNPsea(workflow.Workflow[Pipeline, Method, Params]):
    r"""Computes functional enrichment using ``SNPsea``.

    Originally developed for tissue and cell-type enrichment [1]_ [2]_, ``SNPsea`` has also been
    applied to pathway enrichment [2]_. It uses activity matrix :math:`\mathbf{X}` to allow for
    definitions of "soft" partitions; that is, distributing each annotation over the :math:`K`
    partitions. For example, in tissue enrichment :math:`\mathbf{X}` may be a gene expression
    matrix such that :math:`X_{ij}` is the expression of gene :math:`j` in tissue :math:`i`. A nice
    property of this method, is that it is defined both for
    :math:`\mathbf{X} \in \mathbb{R}^{K \times |G|}` and
    :math:`\mathbf{X} \in \{0, 1\}^{K \times |G|}`, i.e. it works for soft *and* hard partitions.
    If :math:`\mathbf{X} \in \mathbb{R}^{K \times |\mathcal{G}|}`, then it is normalised in the
    following way:

    .. math::
        \bar{\mathbf{X}} = (h \circ g \circ f)(\mathbf{X})

    where

    .. math::
        f &:= \texttt{quantile_norm} \label{eqn:f} \\
        g &:= \texttt{euclid_norm} \label{eqn:g} \\
        h &:= \texttt{percentile_assignment} \label{eqn:h}

    \eqref{eqn:g} is the easiest to explain mathematically, and simply normalises each column
    vector in :math:`\mathbf{X}` to unit length; that is, each annotation :math:`j` will have
    length 1 over all :math:`K` partitions. In the example of gene expression, this amplifies
    specifically expressed genes and surpresses ubiquitously expressed ones. To better understand
    \eqref{eqn:f}, consider the following code snippet:

    >>> def quantile_norm(X):
    ...    # subtract 1 to get 0-based indexes
    ...    rank = scipy.stats.rankdata(X, axis=1, method="min") - 1
    ...    return np.sort(X, axis=1).mean(axis=0)[rank]

    i.e. we average the *observed* quantiles in each partition over the :math:`K` partitions and
    then assign each annotation in each partition to a quantile. See [3]_ [4]_ for a reference.

    Similarly, \eqref{eqn:h} assigns :math:`1` minus the percentile of each annotation in a given
    partition to that annotation, such that the final activity matrix
    :math:`\bar{\mathbf{X}} \in \left[ \frac{1}{|\mathcal{G}|}, 1 \right]^{K \times |\mathcal{G}|}`.
    It is important to emphasise that this means that *higher* values in :math:`\mathbf{X}` will be
    assigned to *lower* values in :math:`\bar{\mathbf{X}}`.

    Next, ``SNPsea`` defines a locus as the sequence span covered by the furthest LD proxies of a
    genome-wide significant variant. These loci are then linked to annotations by overlapping the
    locus spans with the annotation spans. If a locus does not overlap any annotations, it will be
    expanded by :math:`10`kb on either side and overlap will be attempted again. Loci which overlap
    the same annotations are combined into a single locus.

    These annotation-locus links can be represented in a bipartite adjacency matrix
    :math:`\mathbf{L} \in \{0, 1\}^{|\mathcal{G}| \times M}`, where :math:`M` specifies the number
    of loci in the study. Using this and :math:`\bar{\mathbf{X}}`, we can compute the
    partition-locus specificity score matrix :math:`\mathbf{K}`, where :math:`K_{il}` between
    partition :math:`i` and locus :math:`l` is defines as:

    .. math::
        K_{il} = 1 - \left(1 - \langle \bar{\mathbf{X}}_i, \mathbf{L}_l \rangle \right) ^ {c_l}

    where :math:`\bar{\mathbf{X}}_i` is the normalised activity profile for partition :math:`i`,
    :math:`\mathbf{L}_l` is the adjacency vector for locus :math:`l` and

    .. math::
        c_l = \sum_j^{|\mathcal{G}|} L_{jl}

    is the total number of annotations in locus :math:`l`. Finally, the inner product between
    :math:`\mathbf{X}_i` and :math:`\mathbf{L}_l` is defined as:

    .. math::
        \langle \bar{\mathbf{X}}_{i}, \mathbf{L}_{l} \rangle := \min_j \bar{X}_{ij} ^ {L_{jl}}

    Since :math:`L_{kl} \in \{0, 1\}` and :math:`\bar{X}_{ik} \in [\frac{1}{D}, 1]`, the above
    inner product will always return the highest specificity value in a locus. Remember that a low
    value in :math:`\bar{\mathbf{X}}` (percentile), denotes high specificity.

    The enrichment :math:`e_i` for partition :math:`i` is defined as:

    .. math::
        e_i = - \sum_l^{M} \log (K_{il})

    In the case when :math:`\mathbf{X} \in \{0, 1\}^{|\mathcal{G}| \times M}`, the specificity score
    matrix :math:`\mathbf{K}` is computed in the following way:

    .. math::
        K_{il} =
        \begin{cases}
            1 - \frac{{|\mathcal{G}| - n_i \choose c_l}}{{|\mathcal{G}| \choose c_l}} & \text{if } \sum_j^{|\mathcal{G}|} X_{ij} L_{jl} \, > 0 \\
            1 & \text{otherwise}
        \end{cases}

    where :math:`n_i = \sum_j^{|\mathcal{G}|} X_{ij}`.

    The significance of :math:`e_i` is then computed via a permutation-based test. First, a set of
    "null" loci are generated from a list of LD pruned, non-associated variants. Then, a set of null
    loci are sampled, such that each true locus has a null counterpart, which contains (roughly) the
    same number of annotations (i.e. :math:`c_l \approx c_{\tilde{l}}`). From these null loci, a
    null enrichment :math:`\tilde{e}_{ik}` is computed. These steps are repeated :math:`N` times,
    giving us the :math:`p`-value for enrichment :math:`e_i`, like so:

    .. math::
        p_i = \frac{1}{N} \sum_k^N \mathbb{1} \left[ e_i \leq \tilde{e}_{ik} \right]

    References:
        .. [1] Hu, X., Kim, H., Stahl, E., Plenge, R., Daly, M. and Raychaudhuri, S., 2011.
            Integrating autoimmune risk loci with gene-expression data identifies specific
            pathogenic immune cell subsets. The American Journal of Human Genetics, 89(4),
            pp.496-506.
        .. [2] Slowikowski, K., Hu, X. and Raychaudhuri, S., 2014. SNPsea: an algorithm to identify
            cell types, tissues and pathways affected by risk loci. Bioinformatics, 30(17),
            pp.2496-2497.
        .. [3] Amaratunga, D. and Cabrera, J., 2001. Analysis of data from viral DNA microchips.
            Journal of the American Statistical Association, 96(456), pp.1161-1170.
        .. [4] Bolstad, B.M., Irizarry, R.A., Åstrand, M. and Speed, T.P., 2003. A comparison of
            normalization methods for high density oligonucleotide array data based on variance and
            bias. Bioinformatics, 19(2), pp.185-193.
    """

    @classmethod
    def default(
        cls, annotation: data.AnnotationComponent | None = None, params: Params = Params()
    ) -> Self:
        """Construct a default instance of this workflow.

        Args:
            annotation: The annotation component to use. If not specified, a default annotation
                component will be used (GTEx).
            params: The parameters to use for this workflow.
        """
        if annotation is None:
            gtex_cols = data.GTEx.columns
            # mypy check
            assert gtex_cols.values is not None
            annotation = data.GTEx(
                normalisation=norm.Compose(
                    norm.QuantileNorm(
                        values_col=gtex_cols.values,
                        output_col=gtex_cols.values,
                        partition_by=(gtex_cols.partition_id,),
                    ),
                    norm.EuclidNorm(
                        values_col=gtex_cols.values,
                        output_col=gtex_cols.values,
                        partition_by=(gtex_cols.annotation_id,),
                    ),
                    PercentileAssignment(
                        values_col=gtex_cols.values,
                        output_col=gtex_cols.values,
                        partition_by=(gtex_cols.partition_id,),
                    ),
                )
            )
        # TODO: how and when to check that the specified column is in the actual dataframe?
        if not annotation.hard_partitions and annotation.columns.values is None:
            raise ValueError(
                "Using soft-partitioned annotation data requires a `values` column to be specified "
                "and to be present in the annotations dataframe."
            )
        extension = (params.expansion_window, params.expansion_window)
        return cls(
            pipeline=Pipeline(
                locus_definition=locus_def.Compose(
                    locus_def.Expand(
                        ld_component=data.LDComponent(
                            r2_threshold=params.expansion_r2_threshold,
                            return_self_ld=True,
                        ),
                        extension=extension,
                    ),
                    locus_def.Overlap(),
                    CollectExpandedIfEmpty(
                        extension=extension,
                    ),
                    locus_def.Merge(),
                    annotation=annotation,
                ),
                variant_selector=variant_selection.Compose(
                    variant_selection.AssociationThreshold(threshold=params.association_threshold),
                    variant_selection.DropHLA(),
                    variant_selection.DropIndel(),
                ),
                n_permutations=params.n_permutations,
            ),
            method=Method(
                partition_id_col=annotation.columns.partition_id,
                annotation_id_col=annotation.columns.annotation_id,
                partition_type=annotation.partition_type,
                n_permutations=params.n_permutations,
            ),
        )
