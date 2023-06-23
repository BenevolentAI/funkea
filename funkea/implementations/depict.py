import functools
import random
import warnings
from typing import cast

import pandas as pd
import pydantic
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame, Window
from scipy import stats
from typing_extensions import Self

from funkea.components import filtering
from funkea.components import locus_definition as locus_def
from funkea.components import normalisation as norm
from funkea.components import variant_selection
from funkea.core import data, method
from funkea.core import params as params_
from funkea.core import pipeline, workflow
from funkea.core.utils import files, functions, partition, types


class OverlapAndNearestAnnotation(locus_def.Overlap):
    """Overlaps annotations, or nearest annotation if not overlaps were found.

    This is a DEPICT-specific locus definition step, which says that if a given locus has no
    overlapping annotation, we should include the nearest annotation.
    """

    def _get_nearest_annotation(self, dataset: DataFrame) -> DataFrame:
        if self.annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure to set this during initialisation as it is "
                "necessary for finding the nearest annotation."
            )

        ann_cols = self.annotation.columns
        partition_cols = self.get_partition_cols(self.locus_columns.id)
        return (
            dataset.drop(
                ann_cols.annotation_id,
                ann_cols.partition_id,
                ann_cols.start,
                ann_cols.end,
                # column dropping has no effect if a column does not exist
                ann_cols.values or "",
            )
            .join(
                self.annotation.load().dropDuplicates([ann_cols.annotation_id]),
                on=ann_cols.chromosome,
                how="inner",
            )
            .withColumn(
                "distance",
                F.least(
                    F.abs(F.col(ann_cols.start) - F.col(data.POSITION_COL)),
                    F.abs(F.col(ann_cols.end) - F.col(data.POSITION_COL)),
                ),
            )
            .withColumn(
                "rank",
                F.dense_rank().over(Window.partitionBy(*partition_cols).orderBy("distance")),
            )
            .filter(F.col("rank") == 1)
            .dropDuplicates([*partition_cols])
            .drop("distance", "rank")
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        nearest_annotation = self._get_nearest_annotation(dataset)
        annotation = super(OverlapAndNearestAnnotation, self)._transform(dataset)
        return annotation.unionByName(
            nearest_annotation.join(
                annotation,
                on=list(self.get_partition_cols(self.locus_columns.id)),
                how="anti",
            )
        )


class NullLociColumns(pydantic.BaseModel):
    """Column names for the null loci dataframe."""

    id: str = ""
    annotations: str = "annotations"
    size: str = "size"
    n_loci: str = "n_loci"


class NullLoci(data.DataComponent):
    """Loads the precomputed DEPICT null-loci.

    The default data were taken from the original implementation. As a consequence, they use a
    specific annotation type: genes. Hence, the default null loci are not applicable to annotations
    which are not genes (with Ensembl IDs).
    Please refer to the original paper for details on how to generate these loci for other
    annotation types.

    Args:
        dataset: Path to the dataframe containing the null loci, or the dataframe itself.
    """

    columns = NullLociColumns()

    def __init__(self, dataset: data.Dataset | None = files.default.depict_null_loci):
        super(NullLoci, self).__init__(dataset=dataset)

    def _load(self) -> DataFrame:
        return self._get_dataset().withColumn(self.columns.size, F.size(self.columns.annotations))


def _create_permutation(
    matched_loci: pd.DataFrame,
    locus_id_col: str,
    null_locus_col: str,
    partition_cols: partition.PartitionByType = partition.DEFAULT_PARTITION_COLS,
) -> pd.DataFrame:
    r"""Creates a non-overlapping set of null loci.

    Given a real locus, we sample randomly from the set of "matched" null loci until we get a locus
    which does not overlap (in annotation) any of the previously sampled null loci. I.e. the
    intersect between all the null loci should be :math:`\emptyset`.
    In the case of DEPICT, null loci are matched to the real loci by number of annotations in the
    locus.

    Args:
        matched_loci: A dataframe containing the (real) locus IDs, the partition columns, and the
            "matched" null loci. The matched null loci should for each real locus be a list of lists
            of strings. I.e. for each real locus, we have the set of matched null loci, each of
            which defines the set of overlapped annotations.
        locus_id_col: The column name of the locus ID.
        null_locus_col: The column name of the matched null loci.
        partition_cols: The columns used to group by in the function application (see Notes).

    Returns:
        Dataframe containing the non-overlapping null-loci.

    Notes:
        This function assumes to be used in a ``groupBy + applyInPandas`` operation, i.e.
        ``matched_loci.groupby(*partition_cols).ngroups == 1``.
    """
    seen: set[str] = set()
    perm: list[list[str]] = []

    size: int
    locus_set: list[list[str]]
    for locus_set in matched_loci[null_locus_col]:
        # random shuffling and for loops are much safer than while loops and random choice
        # this is an in-place operation, which is not ideal, but in our use-case, it does not matter
        random.shuffle(locus_set)

        locus: list[str]
        for locus in locus_set:
            if not seen.intersection(locus):
                perm.append(locus)
                seen.update(locus)
                break
        else:
            # if we did not have a single "independent" locus, just add the last one
            # this is an arbitrary decision to ensure we do not end up in an infinite loop (using
            # while) or end up not choosing a null locus for a given real locus
            perm.append(locus_set[-1])
            seen.update(locus_set[-1])
    return functions.cross_join(
        matched_loci[list(partition_cols)].drop_duplicates(),
        pd.DataFrame({locus_id_col: matched_loci[locus_id_col].to_list(), null_locus_col: perm}),
    )


class Pipeline(pipeline.DataPipeline):
    """Data pipeline for running DEPICT.

    Args:
        locus_definition: The locus definition.
        variant_selector: The variant selector.
        null_loci: The null loci dataset.
        n_permutations: The number of permutations to generate for the significance test.
    """

    id_col: str = "locus_id"
    score_col: str = "score"
    locus_type: str = "locus_type"

    def __init__(
        self,
        locus_definition: locus_def.LocusDefinition,
        variant_selector: variant_selection.VariantSelection | None = None,
        null_loci: NullLoci = NullLoci(),
        n_permutations: pydantic.NonNegativeInt = 500,
    ):
        self.null_loci = null_loci
        self.n_permutations = n_permutations
        super(Pipeline, self).__init__(
            locus_definition=locus_definition,
            variant_selector=variant_selector,
        )

    def get_matched_null_loci(self, loci: DataFrame) -> DataFrame:
        """Gets the matched null loci for the given real loci.

        Matching here is done by number of annotations in the locus. I.e. we find the null loci
        which have the closest number of annotations to the real loci.

        Args:
            loci: The real loci.

        Returns:
            The matched null loci.
        """
        cols = self.locus_definition.locus_columns
        null_cols = self.null_loci.columns

        null_partition_cols = self.get_partition_cols(cols.id)
        # Unfortunately, cases often arise where we do not have a set of null loci which came from
        # a GWAS which produced exactly the same number of loci as the real one.
        # Hence, we have to do a "fuzzy join", i.e. cross join and then take the "closest match".
        # This means in simple terms we want to find the null GWAS which produced the most similar
        # number of loci.
        # Similarly, when we try to find similar null loci for each real locus (similarity is
        # defined by number of annotations in the locus) we also have to do a "fuzzy join". But
        # instead of just taking the most similar, we get a set of at least 10 most similar null
        # loci per real locus. These sets of loci can then be used for random sampling. See
        # `_create_permutations` for details.
        n_loci: DataFrame = loci.groupBy(*self.partition_cols).count()
        null_loci = self.null_loci.load().repartition(null_cols.n_loci)
        exact_match = null_loci.join(
            n_loci.withColumnRenamed("count", null_cols.n_loci),
            on=null_cols.n_loci,
            how="inner",
        )
        return (
            exact_match.unionByName(
                n_loci.join(
                    exact_match,
                    on=list(self.partition_cols),
                    how="anti",
                )
                .crossJoin(null_loci)
                .withColumn(
                    # we compute the `distance` between the number of loci in the real GWAS and the
                    # number of loci in the null set
                    # we then use this distance to get the `closest` one
                    "distance",
                    F.abs(F.col(null_cols.n_loci) - F.col("count")),
                )
                .withColumn("rank", F.dense_rank().over(self.get_window().orderBy("distance")))
                .filter(F.col("rank") == 1)
                .drop("distance", "count", "rank")
            )
            .join(
                loci.select(
                    *null_partition_cols, F.size(locus_def.Collect.collected_col).alias("real_size")
                ),
                on=list(self.partition_cols),
                how="inner",
            )
            .withColumn("distance", F.abs(F.col("real_size") - F.col(null_cols.size)))
            # below we can use the sparse (min) rank, as we will get the same (min) number for all
            # the tied positions, and then a jump to the total count
            .withColumn("rank", F.rank().over(self.get_window(cols.id).orderBy("distance")))
            .filter(F.col("rank") <= 10)
            # now we collect the annotations for each null locus into a list for each `true` locus
            # that way, we can easily sample in `_create_permutations` for each true locus a null
            # locus with the appropriate number of annotations.
            .groupBy(*null_partition_cols)
            .agg(F.collect_list(null_cols.annotations).alias(locus_def.Collect.collected_col))
            .crossJoin(
                # N null permutations
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
                loci.select(*null_partition_cols, locus_def.Collect.collected_col).schema.add(
                    T.StructField(self.locus_type, T.StringType())
                ),
            )
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        annotation = self.locus_definition.annotation
        if annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure to set this during initialisation in the "
                "`locus_definition`, as it is needed for Pipeline execution."
            )

        dataset = self.get_variant_selector().transform(dataset)
        loci = self.locus_definition.transform(dataset)

        if locus_def.Collect.collected_col not in loci.columns:
            raise RuntimeError(
                f"{locus_def.Collect.collected_col!r} column is missing from the loci dataframe, "
                "suggesting that the `locus_definition` component is missing a `Collect` step."
            )

        cols = self.locus_definition.locus_columns
        ann_cols = annotation.columns
        # mypy checks
        assert ann_cols.values is not None

        partition_cols = self.get_partition_cols(cols.id)
        return (
            loci.select(
                *partition_cols,
                locus_def.Collect.collected_col,
                F.lit("true").alias(self.locus_type),
            )
            .unionByName(self.get_matched_null_loci(loci))
            .select(
                *partition_cols,
                F.explode(locus_def.Collect.collected_col).alias(ann_cols.annotation_id),
                self.locus_type,
            )
            .join(
                annotation.load().select(
                    ann_cols.partition_id, ann_cols.annotation_id, ann_cols.values
                ),
                on=ann_cols.annotation_id,
                how="inner",
            )
            .select(
                *self.partition_cols,
                F.col(cols.id).alias(self.id_col),
                ann_cols.partition_id,
                self.locus_type,
                F.col(ann_cols.values).alias(self.score_col),
            )
        )


class Method(method.EnrichmentMethod):
    """The method used to compute the enrichment score for DEPICT.

    This method is based on the one described in the DEPICT paper [1]_.

    Args:
        partition_id_col: The name of the column containing the partition ID.

    References:
        .. [1] Pers, T.H., Karjalainen, J.M., Chan, Y., Westra, H.J., Wood, A.R., Yang, J.,
            Lui, J.C., Vedantam, S., Gustafsson, S., Esko, T. and Frayling, T., 2015. Biological
            interpretation of genome-wide association studies using predicted gene functions.
            Nature communications, 6(1), p.5890.
    """

    def __init__(self, partition_id_col: str):
        super(Method, self).__init__()
        self.partition_id_col = partition_id_col

    def _transform(self, dataset: DataFrame) -> DataFrame:
        partition_locus_scores = (
            dataset.groupBy(
                *self.get_partition_cols(
                    Pipeline.id_col, Pipeline.locus_type, self.partition_id_col
                )
            )
            .agg(F.mean(Pipeline.score_col).alias("score"))
            .groupBy(*self.get_partition_cols(self.partition_id_col, Pipeline.locus_type))
            .agg((F.mean("score") / F.stddev_samp("score")).alias("score"))
            .fillna(value=0.0, subset="score")
        )
        parts = self.get_partition_cols(self.partition_id_col)

        @F.pandas_udf("double")  # type: ignore[call-overload]
        def gauss_sf(e: pd.Series) -> pd.Series:
            return pd.Series(stats.norm.sf(e))

        return (
            partition_locus_scores.filter(F.col(Pipeline.locus_type) == "true")
            .drop(Pipeline.locus_type)
            .join(
                partition_locus_scores.filter(F.col(Pipeline.locus_type).rlike(r"^null\d+$"))
                .groupBy(*parts)
                .agg(F.mean("score").alias("mean"), F.stddev_samp("score").alias("std")),
                on=list(parts),
                how="inner",
            )
            .select(
                *parts,
                ((F.col("score") - F.col("mean")) / F.col("std")).alias("enrichment"),
            )
            .withColumn("p_value", gauss_sf("enrichment"))
        )


class Params(params_.Params):
    """Parameters for the default configuration of ``DEPICT``.

    Args:
        association_threshold: The :math:`p`-value threshold used to select genome-wide significant
            variants. (``default=5e-8``)
        expansion_r2_threshold: The minimum :math:`r^2` value for a variant-LD proxy pair.
            (``default=0.5``)
        ld_pruning_r2_threshold: The maximum :math:`r^2` between any pair of variants in the
            provided sumstats. (``default=0.1``)
    """

    association_threshold: types.UnitFloat = cast(
        types.UnitFloat, variant_selection.DEFAULT_ASSOCIATION_THRESHOLD
    )
    expansion_r2_threshold: types.UnitFloat = cast(types.UnitFloat, 0.5)
    ld_pruning_r2_threshold: types.UnitFloat = cast(types.UnitFloat, 0.1)


class DEPICT(workflow.Workflow[Pipeline, Method, Params]):
    r"""Computes functional enrichment using ``DEPICT``.

    ``DEPICT`` is a popular method for gene prioritisation, but also has proven itself useful for
    functional enrichment analysis [1]_. Similarly to :class:`.SNPsea`, it uses activity matrix
    :math:`\mathbf{X} \in \mathbb{R}^{K \times |\mathcal{G}|}` to assign the annotations
    :math:`s_{ij}` to the (soft) partitions :math:`i \in \{1, \dots, K\}`, by normalising the
    activities over the :math:`K` partitions. However, in this case, :math:`\bar{\mathbf{X}}` is
    produced from :math:`\mathbf{X}` via :math:`z`-scoring, rather than normalising to unit length.
    Also, much like :class:`.SNPsea`, ``DEPICT`` defines bipartite adjacency matrix
    :math:`\mathbf{L} \in \{0, 1\}^{|\mathcal{G}| \times M}`, which defines whether an annotation
    and a locus overlap. It then computes the scores between each partition :math:`i` and locus as
    follows:

    .. math::
        \mathbf{K} = \bar{\mathbf{X}} \mathbf{L} \cdot \mathrm{diag}(\mathbf{L}^T \mathbf{1})^{-1}

    The loci are defined as the LD pruned genome-wide significant variants and their LD proxies.
    Each locus is overlapped with the genomic annotations and loci which do not overlap any
    annotations are associated with their nearest annotation. Loci which share annotations are
    fused.
    From :math:`\mathbf{K}`, we then get the biased estimate of the enrichment :math:`\hat{e}_i` of
    partition :math:`i`:

    .. math::
        \hat{e}_i = \frac{\mu_i}{\sigma_i}

    where :math:`\mu_i` and :math:`\sigma_i` are the sample mean and standard deviation of the
    partition-locus scores :math:`K_{ij}`. The final bias corrected enrichment :math:`e_i` is then
    defined as

    .. math::
        e_i = \frac{\hat{e}_i - \mathbb{E}[\hat{e}_i]}{SE(\hat{e}_i)}

    where :math:`\mathbb{E}[\hat{e}_i]` and :math:`SE(\hat{e}_i)` are the expectation and standard
    error of :math:`\hat{e}_i`, which are approximated empirically via Monte Carlo sampling. Since
    we assume a normal distribution of :math:`e_i`, we get :math:`p_i = 1 - F(e_i)`, where
    :math:`F(e_i)` is the cumulative distribution function of the unit Gaussian.

    Finally, ``DEPICT`` estimates the false discovery rate (FDR) for each :math:`p_i` by running
    :math:`N` repetitions of the above steps using :math:`M` randomly generated loci. Each generated
    locus will have the same number of annotations as their "real" counterpart. The FDR is computed
    as follows:

    .. math::
        FDR(p_i) = \frac{1}{R_i N} \sum_k^N \mathbb{1} \left[ \tilde{p}_{ik} \leq p_i \right]

    where :math:`R_i` is the ordinal rank of :math:`p_i` and :math:`\tilde{p}_{ik}` is the
    :math:`p`-value of the :math:`k`-th null enrichment for partition :math:`i`. For the sake of
    consistency with the other methods, we did not include this step.

    References:
        .. [1] Pers, T.H., Karjalainen, J.M., Chan, Y., Westra, H.J., Wood, A.R., Yang, J.,
            Lui, J.C., Vedantam, S., Gustafsson, S., Esko, T. and Frayling, T., 2015. Biological
            interpretation of genome-wide association studies using predicted gene functions.
            Nature communications, 6(1), p.5890.
    """

    @classmethod
    def default(
        cls, annotation: data.AnnotationComponent | None = None, params: Params = Params()
    ) -> Self:
        """Returns the default configuration of ``DEPICT``.

        Args:
            annotation: The annotation component to use. If ``None``, the default GTEx annotation
                component will be used. (``default=None``)
            params: The parameters to use. (``default=DEPICT.Params()``)

        Returns:
            The default configuration of ``DEPICT``.
        """
        if annotation is None:
            gtex_cols = data.GTEx.columns
            # mypy check
            assert gtex_cols.values is not None
            annotation = data.GTEx(
                normalisation=norm.StandardScaler(
                    values_col=gtex_cols.values,
                    output_col=gtex_cols.values,
                    partition_by=(gtex_cols.annotation_id,),
                )
            )
        if not isinstance(annotation.normalisation, norm.StandardScaler):
            warnings.warn(
                "DEPICT assumes all annotation-partition values to be z-scored. Hence, it is "
                "recommended to specify the StandardScaler normalisation component in the "
                "annotation component.",
                UserWarning,
            )

        if annotation.partition_type == data.PartitionType.HARD:
            filter_op = annotation.filter_operation
            has_make_full = filter_op is not None and (
                isinstance(filter_op, filtering.MakeFull)
                or (
                    isinstance(filter_op, filtering.Compose)
                    and any(isinstance(op, filtering.MakeFull) for op in filter_op.steps)
                )
            )
            if not has_make_full:
                warnings.warn(
                    "The annotation component specified has hard partitions, but does not specify "
                    "a `MakeFull` filter operation. Due to the z-scoring, DEPICT assumes all "
                    "annotation-partition pairs to be in the dataset. Make sure this assumption is "
                    "not violated to avoid unexpected behaviour.",
                    UserWarning,
                )

        return cls(
            pipeline=Pipeline(
                locus_def.Compose(
                    locus_def.Expand(
                        ld_component=data.LDComponent(
                            # return self LD, such that we do not lose variants
                            r2_threshold=params.expansion_r2_threshold,
                            return_self_ld=True,
                        )
                    ),
                    OverlapAndNearestAnnotation(),
                    locus_def.Collect(what="annotation"),
                    locus_def.Merge(),
                    annotation=annotation,
                ),
                variant_selection.Compose(
                    variant_selection.AssociationThreshold(
                        threshold=params.association_threshold,
                    ),
                    variant_selection.DropHLA(),
                    variant_selection.DropIndel(),
                    variant_selection.DropComplement(),
                    variant_selection.LDPrune(
                        ld_component=data.LDComponent(
                            r2_threshold=params.ld_pruning_r2_threshold,
                        )
                    ),
                ),
            ),
            method=Method(partition_id_col=annotation.columns.partition_id),
        )
