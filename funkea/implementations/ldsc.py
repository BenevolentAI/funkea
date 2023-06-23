import functools
import warnings
from typing import Final, Literal, cast

import numpy as np
import pandas as pd
import pydantic
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame
from scipy import stats
from typing_extensions import Self

import ldsc
from funkea.components import filtering
from funkea.components import locus_definition as locus_def
from funkea.components import variant_selection
from funkea.core import data, method
from funkea.core import params as params_
from funkea.core import pipeline, workflow
from funkea.core.utils import files, functions, partition, types

BETA_COL: Final[str] = "beta"
SE_COL: Final[str] = "se"
CHI2_COL: Final[str] = "chi2"
WEIGHT_COL: Final[str] = "weight"
CONTROL_COL: Final[str] = "control"
SAMPLE_SIZE_COL: Final[str] = "sample_size"


class LDScoreColumns(pydantic.BaseModel):
    """Column name mapping for the LD scores."""

    id: str = "rsid"
    partition_id: str = "partition_id"
    ld_score: str = "ld_score"
    n_variants: str = "n_variants"
    ancestry: str = data.ANCESTRY_COL


class LDScores(data.DataComponent):
    """Loads LD scores from disk or memory.

    This is a data component which can be used to load LD scores. These will be precomputed LD
    scores, either used for enrichment or as controlling covariates.

    Args:
        dataset: Path to the LD scores, or the dataframe itself. If None, an error will be
            raised when trying to load the data. (``default=None``)
        columns: The column name mapping for the dataframe. (``default=LDScoreColumns()``)
    """

    def __init__(
        self,
        dataset: data.Dataset | None = None,
        columns: LDScoreColumns = LDScoreColumns(),
    ):
        self.columns = columns
        super(LDScores, self).__init__(dataset=dataset)

    def load_pivot_table(self, values: Literal["ld_score", "n_variants"] = "ld_score") -> DataFrame:
        return (
            self.load()
            .groupBy(self.columns.id, self.columns.ancestry)
            .pivot(self.columns.partition_id)
            .agg(
                F.first(self.columns.ld_score if values == "ld_score" else self.columns.n_variants)
            )
        )

    def get_partitions(self) -> list[str]:
        col = self.columns.partition_id
        # still typing issues inside Spark
        return cast(pd.DataFrame, self.load().select(col).distinct().toPandas())[col].to_list()


def compute_chi2(sumstats: DataFrame) -> DataFrame:
    r"""Computes the :math:`\chi^2` association statistics from summary statistics.

    The statistic is either computed from :math:`\beta` and :math:`SE(\beta)` or from
    :math:`p`-values depending on which columns are available.

    Args:
        sumstats: The GWAS summary statistics containing either coefficients and standard error, or
            the :math:`p`-values.

    Returns:
        The original dataframe with an additional :math:`\chi^2` column.
    """
    # should we even be allowing options here?
    if {BETA_COL, SE_COL}.intersection(sumstats.columns) != {BETA_COL, SE_COL}:
        if data.ASSOCIATION_COL not in sumstats.columns:
            raise ValueError(
                f"If either {BETA_COL!r} or {SE_COL!r} are missing from the sumstats, "
                f"{data.ASSOCIATION_COL!r} is required to compute the `chi2` association statistics"
            )

        @F.pandas_udf("double")  # type: ignore[call-overload]
        def chi2_isf(p_value: pd.Series) -> pd.Series:
            return pd.Series(stats.chi2.isf(p_value, 1))

        return sumstats.withColumn(CHI2_COL, chi2_isf(data.ASSOCIATION_COL))
    return sumstats.withColumn(CHI2_COL, (F.col(BETA_COL) / F.col(SE_COL)) ** 2)


def extract_control_ld(
    partitioned_ld_scores: DataFrame, columns: LDScoreColumns, control_value: str = CONTROL_COL
) -> DataFrame:
    is_control = F.col(columns.partition_id) == control_value
    partitions = partitioned_ld_scores.filter(~is_control)
    control = partitioned_ld_scores.filter(is_control)
    return partitions.join(
        control.select(
            columns.id,
            columns.ancestry,
            F.col(columns.ld_score).alias(control_value),
            F.col(columns.n_variants).alias(control_value + "_n_variants"),
        ),
        on=[columns.id, columns.ancestry],
        how="inner",
    )


class Pipeline(pipeline.DataPipeline):
    """Data pipeline for LD score regression.

    This pipeline prepares the data for LD score regression. It requires a locus definition and
    variant selection method to be provided. If not available, it will compute the partitioned
    LD scores for the annotation dataset provided.

    Args:
        locus_definition: The locus definition to use for the analysis.
        variant_selector: The variant selection method to use for the analysis.
        weighting_ld_scores: The LD scores to used for computing the regression weights. These
            should be precomputed. Make sure to use the correct ancestry.
        precomputed_ld_scores: The precomputed LD scores to use for the analysis. If not provided,
            the pipeline will compute the LD scores from the annotation dataset.
        controlling_ld_scores: The LD scores to use as controlling covariates. These should be
            precomputed. Make sure to use the correct ancestry.
        ld_component: The LD component to use for the analysis. If no precomputed LD scores are]
            provided, this will be used to compute the LD scores.
        n_partitions_ld_computation: The number of partitions to use for computing the LD scores.
            ``partitions`` here refers to Spark partitions. It can be important to control this
            number to avoid memory issues. For example, if the number of partitions or annotations
            is very high, it can be that each Spark partition will be too large to fit in memory.
            Hence, if you get memory errors, try to increase this number. However, decrease this
            if garbage collection is taking too long. (``default=10_000``)
    """

    target_col: str = CHI2_COL

    def __init__(
        self,
        locus_definition: locus_def.LocusDefinition,
        variant_selector: variant_selection.VariantSelection | None = None,
        weighting_ld_scores: LDScores = LDScores(dataset=files.default.ldsc_weighting_ld_scores),
        precomputed_ld_scores: LDScores | None = None,
        controlling_ld_scores: LDScores
        | None = LDScores(
            dataset=files.default.ldsc_controlling_ld_scores,
        ),
        ld_component: data.LDComponent | None = None,
        n_partitions_ld_computation: int = 10_000,
    ):
        self.weighting_ld_scores = weighting_ld_scores
        self.precomputed_ld_scores = precomputed_ld_scores
        self.controlling_ld_scores = controlling_ld_scores
        self.ld_component = ld_component
        self.n_partitions_ld_computation = n_partitions_ld_computation
        super(Pipeline, self).__init__(
            locus_definition=locus_definition,
            variant_selector=variant_selector,
        )

    def get_annotation(self) -> data.AnnotationComponent:
        annotation = self.locus_definition.annotation
        if annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure to set this during initialisation of "
                "`locus_definition`"
            )
        return annotation

    def get_ld_component(self) -> data.LDComponent:
        if self.ld_component is None:
            raise ValueError(
                "`ld_component` is None. When computing the partitioned LD scores, make sure to "
                "set the LD component."
            )
        return self.ld_component

    def annotate_variants(self, dataset: DataFrame) -> DataFrame:
        annotation = self.get_annotation()
        loci = self.locus_definition.transform(dataset)
        if annotation.columns.partition_id not in loci.columns:
            raise RuntimeError(
                f"{annotation.columns.partition_id!r} is missing from the loci dataframe. Make "
                "sure to add an `Overlap` step to the `locus_definition` to ensure association "
                "between variants and annotation partitions."
            )
        return loci.select(data.ID_COL, annotation.columns.partition_id).distinct()

    def load_annotated_reference_variants(self) -> DataFrame:
        # we need to add a dummy p-value column, such that it works with the locus definition
        return self.annotate_variants(
            self.get_ld_component().load_variants().withColumn(data.ASSOCIATION_COL, F.lit(1.0))
        )

    def load_annotation_reference_variants_combinations(
        self, annotated_variants: DataFrame
    ) -> DataFrame:
        ld_component = self.get_ld_component()
        cols = ld_component.columns
        ann_cols = self.get_annotation().columns
        return (
            ld_component.load_variants()
            .crossJoin(annotated_variants.select(ann_cols.partition_id).distinct())
            .crossJoin(
                ld_component.load().select(F.col(cols.ancestry).alias(data.ANCESTRY_COL)).distinct()
            )
        )

    def _compute_ld_scores(self, annotated_variants: DataFrame) -> DataFrame:
        ld_component = self.get_ld_component()
        cols = ld_component.columns
        ld_cols = self.weighting_ld_scores.columns

        # the repartitioning here is necessary to not break the join operation happening in
        # partitioned LD score computation. The number of partitions should be set quite high (from
        # limited experimentation it seems that having < 100 Mb per partition works). This is
        # because when we have a low number of partitions, each partition holds a large amount of
        # data at once, which inevitably leads to spills. Spills slow down Spark, but can be
        # managed as long as a PVC is provided. However, if shuffle partitions are exceptionally
        # large (which can happen through data skews), this can seemingly either kill the executor
        # or cause network IO freezes. A common error message may be something along the lines of
        # ``FetchFailed`` etc..
        repartition: tuple[int, str] = (self.n_partitions_ld_computation, data.ID_COL)
        return (
            annotated_variants.repartition(*repartition)
            .join(
                ld_component.load_full_symmetric()
                .select(
                    cols.source_id,
                    F.col(cols.target_id).alias(data.ID_COL),
                    cols.correlation,
                    cols.ancestry,
                )
                .repartition(*repartition),
                on=data.ID_COL,
                how="inner",
            )
            .groupBy(
                F.col(cols.source_id).alias(ld_cols.id),
                F.col(self.get_annotation().columns.partition_id).alias(ld_cols.partition_id),
                F.col(cols.ancestry).alias(ld_cols.ancestry),
            )
            .agg(
                F.sum(cols.correlation).alias(self.weighting_ld_scores.columns.ld_score),
            )
        )

    def _compute_n_variants(self, annotated_variants: DataFrame) -> DataFrame:
        cols = self.weighting_ld_scores.columns
        return (
            annotated_variants.groupBy(
                # Here we assume that all variants are shared across ancestries.
                # While this is incorrect, this simplifies things a bit.
                F.col(self.get_annotation().columns.partition_id).alias(cols.partition_id)
            )
            .count()
            .withColumnRenamed("count", cols.n_variants)
        )

    def compute_ld_scores_and_n_variants(self, annotated_variants: DataFrame) -> DataFrame:
        ld_component = self.get_ld_component()

        if ld_component.r2_threshold is not None:
            warnings.warn(
                "`ld_component.r2_threshold` is not None. This is not recommended, as it standard "
                "to compute LD scores with all available correlations.",
                UserWarning,
            )

        if not ld_component.return_unbiased_r2:
            warnings.warn(
                "`ld_component.return_unbiased_r2` is set to False. It is recommended to set this "
                "to True, following the official implementation.",
                UserWarning,
            )

        if ld_component.return_self_ld:
            warnings.warn(
                "`ld_component.return_self_ld` is set to True. It is recommended to set this to "
                "False, to avoid inflating the LD scores by 1 (as per original implementation).",
                UserWarning,
            )

        ld_cols = self.weighting_ld_scores.columns
        return (
            self._compute_ld_scores(annotated_variants)
            .join(
                # we join back all the possible variant-annotation combinations, such that we get
                # the LD-scores for variants which are not in LD with any relevant annotations.
                # These LD scores should be 0.
                self.load_annotation_reference_variants_combinations(annotated_variants).select(
                    F.col(data.ID_COL).alias(ld_cols.id),
                    F.col(self.get_annotation().columns.partition_id).alias(ld_cols.partition_id),
                    F.col(data.ANCESTRY_COL).alias(ld_cols.ancestry),
                ),
                on=[ld_cols.id, ld_cols.partition_id, ld_cols.ancestry],
                how="right",
            )
            .fillna(value=0.0, subset=ld_cols.ld_score)
            # we also have to compute the number of variants in each partition
            .join(
                self._compute_n_variants(annotated_variants),
                on=ld_cols.partition_id,
                how="inner",
            )
        )

    def compute_control_ld_scores_and_n_variants(self, annotated_variants: DataFrame) -> DataFrame:
        return self.compute_ld_scores_and_n_variants(
            annotated_variants.withColumn(
                self.get_annotation().columns.partition_id, F.lit(CONTROL_COL)
            ).distinct()
        )

    def compute_partitioned_ld_scores_and_control(self) -> DataFrame:
        annotated_variants = self.load_annotated_reference_variants()
        return self.compute_ld_scores_and_n_variants(annotated_variants).unionByName(
            self.compute_control_ld_scores_and_n_variants(annotated_variants)
        )

    def compute_partitioned_ld_scores_and_control_batched(
        self, batch_size: int = 1024
    ) -> list[DataFrame]:
        annotated_variants = self.load_annotated_reference_variants().cache()

        partition_col = self.get_annotation().columns.partition_id
        parts = annotated_variants.select(partition_col).distinct()
        subset_schema = parts.schema

        ld_scores: list[DataFrame] = []
        subset: list[str]
        for subset in functions.chunk(
            cast(
                # mypy issues with toPandas calls
                pd.DataFrame,
                parts.toPandas(),
            )[partition_col].to_list(),
            batch_size,
        ):
            variants = annotated_variants.join(
                functions.get_session().createDataFrame([*zip(subset)], subset_schema),
                on=partition_col,
                how="leftsemi",
            )
            ld_scores.append(self.compute_ld_scores_and_n_variants(variants))
        ld_scores.append(self.compute_control_ld_scores_and_n_variants(annotated_variants))
        return ld_scores

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if SAMPLE_SIZE_COL not in dataset.columns:
            raise ValueError(
                f"{SAMPLE_SIZE_COL!r} column not in input dataframe. LD score regression requires "
                "the sample size used in the GWAS."
            )

        annotation = self.get_annotation()
        dataset = self.get_variant_selector().transform(dataset)

        ld_scores: DataFrame
        if self.precomputed_ld_scores is None:
            warnings.warn(
                "No precomputed LD scores set. This means they will be computed on the fly. "
                "As this is a very expensive operation, it is recommended to precompute these by "
                "using the `LDSC.compute_partitioned_ld_scores` function and then saving them to "
                "disk. Please refer to the aforementioned function's documentation for details.",
                UserWarning,
            )
            ld_scores = self.compute_partitioned_ld_scores_and_control()
        else:
            ld_scores = self.precomputed_ld_scores.load()

        w_cols = self.weighting_ld_scores.columns
        ld_scores = (
            extract_control_ld(ld_scores, self.weighting_ld_scores.columns)
            .withColumnRenamed(w_cols.id, data.ID_COL)
            .withColumnRenamed(w_cols.ancestry, data.ANCESTRY_COL)
        )
        if self.controlling_ld_scores is not None:
            controls = self.controlling_ld_scores.load_pivot_table("ld_score")
            cont_cols = self.controlling_ld_scores.columns
            cols = tuple(set(controls.columns).difference([cont_cols.id, cont_cols.ancestry]))

            controls = controls.join(
                self.controlling_ld_scores.load_pivot_table("n_variants").select(
                    cont_cols.id,
                    cont_cols.ancestry,
                    *(F.col(column).alias(column + "_n_variants") for column in cols),
                ),
                on=[cont_cols.id, cont_cols.ancestry],
                how="inner",
            )

            ld_scores = ld_scores.join(
                controls.withColumnRenamed(cont_cols.id, data.ID_COL).withColumnRenamed(
                    cont_cols.ancestry, data.ANCESTRY_COL
                ),
                on=[data.ID_COL, data.ANCESTRY_COL],
                how="inner",
            )
        ann_cols = annotation.columns
        return (
            ld_scores.withColumnRenamed(w_cols.partition_id, ann_cols.partition_id)
            .join(
                self.weighting_ld_scores.load().select(
                    F.col(w_cols.id).alias(data.ID_COL),
                    F.col(w_cols.ld_score).alias(WEIGHT_COL),
                ),
                on=data.ID_COL,
                how="inner",
            )
            .join(
                dataset,
                on=[data.ID_COL, data.ANCESTRY_COL],
                how="inner",
            )
            .transform(compute_chi2)
            .filter(
                # this filter condition is taken from the original implementation of LDSC
                F.col(self.target_col)
                < F.greatest(F.col(SAMPLE_SIZE_COL) * 0.001, F.lit(80.0))
            )
        )


class _LDSCColumns(pydantic.BaseModel):
    """A pydantic model containing the column names of the DataFrame used to fit an LDSC model."""

    ld_score: str = "ld_score"
    partition_id: str = "partition_id"
    controlling_covariates: tuple[str, ...] = ()
    target: str = CHI2_COL
    position: tuple[str, ...] = (data.CHROMOSOME_COL, data.POSITION_COL)
    sample_size: str = SAMPLE_SIZE_COL
    n_variants: str = "n_variants"
    weight: str = WEIGHT_COL
    partition_cols: partition.PartitionByType = partition.DEFAULT_PARTITION_COLS


def fit_ldsc_model(
    sumstats: pd.DataFrame,
    columns: _LDSCColumns,
) -> pd.DataFrame:
    """Fit a heritability model.

    Args:
        sumstats: A pandas DataFrame containing the summary statistics and LD scores.
        columns: A pydantic model containing the column names of the sumstats DataFrame.

    Returns:
        A pandas DataFrame containing the enrichment estimate of the target.
    """
    # We need to sort the variants by genome-wide position, as this will affect the jackknife
    # standard-error estimates. If we do not sort by position, we will systematically underestimate
    # the standard-error.
    sumstats = sumstats.sort_values([*columns.position])

    X = sumstats[[columns.ld_score, *columns.controlling_covariates]].to_numpy(np.float32)
    y = sumstats[columns.target].to_numpy(np.float32)
    N = sumstats[columns.sample_size].to_numpy(np.int32)
    M = (
        sumstats[[columns.n_variants, *map("{}_n_variants".format, columns.controlling_covariates)]]
        .iloc[0]
        .to_numpy(np.int32)
    )
    w = sumstats[columns.weight].to_numpy(np.float32)

    try:
        model: ldsc.HeritabilityRegression = ldsc.HeritabilityRegression(
            n_jack_knife_blocks=min(len(X), 200),
            max_irls_iter=0,
            use_intercept=True,
        ).fit(X, y, n_samples=N, n_variants=M, ld_weighting=w)
        coef = model.coefficients
        result = pd.DataFrame(
            {
                "enrichment": [coef.coefficients[0]],
                "p_value": [stats.norm.sf(coef.coefficients[0] / coef.standard_error[0])],
            }
        )
    except np.linalg.LinAlgError:
        # There are two cases when we could encounter a singular matrix:
        #  1. The LD score partition is indeed 0 across the board.
        #  2. One of the blocks in the jackknife has all 0s for the LD score matrix.
        result = pd.DataFrame([(np.nan, np.nan)], columns=["enrichment", "p_value"])
    return functions.cross_join(
        sumstats[[*columns.partition_cols, columns.partition_id]].drop_duplicates(),
        result,
    )


class Method(method.EnrichmentMethod):
    """Estimate enrichment using LD score regression.

    This method is based on the LD score regression method described in Bulik-Sullivan et al. (2015).
    The method is implemented in the ``ldsc`` Python package (shipped together with ``funkea``). The
    method is described in detail in :class:`.LDSC`.

    ``ldsc`` provides GPU support for the LD score regression. To enable GPU support, set the
    ``LDSC_BACKEND`` environment variable to ``JAX``. Note that GPU support is experimental and
    might not work for all datasets. If GPU support is enabled, the method will automatically
    use the GPU if available (you will need to provide GPUs to the Spark executors). This can be
    very helpful for annotation datasets with many partitions (each partition has to fit its own
    LD score regression model).

    Args:
        partition_id_col: The name of the column containing the partition ID.
        control_ld: The names of the columns containing the LD scores of the controlling
            covariates. The LD scores of the controlling covariates will be used to control for
            the effects of the covariates in the LD score regression model.
    """

    def __init__(
        self,
        partition_id_col: str,
        control_ld: tuple[str, ...] = (CONTROL_COL,),
    ):
        super(Method, self).__init__()
        self.partition_id_col = partition_id_col
        self.control_ld = control_ld

    def _transform(self, dataset: DataFrame) -> DataFrame:
        parts = self.get_partition_cols(self.partition_id_col)
        return dataset.groupBy(*parts).applyInPandas(
            functools.partial(
                fit_ldsc_model,
                columns=_LDSCColumns(
                    partition_id=self.partition_id_col,
                    controlling_covariates=self.control_ld,
                    target=Pipeline.target_col,
                    partition_cols=self.partition_cols,
                ),
            ),
            dataset.select(*parts)
            .schema.add(T.StructField("enrichment", T.DoubleType()))
            .add(T.StructField("p_value", T.DoubleType())),
        )


class Params(params_.Params):
    """Parameters for the default configuration of ``LDSC``.

    Args:
        expansion_window: The size of the window within which a variant and an annotation can be
            overlapped. (``default=100_000``)
        maf_threshold: The minimum MAF of the variants to be used in the regression.
            (``default=0.01``)
        precomputed_ld_scores: Optional precomputed LD scores. (``default=None``)
        controlling_ld_scores: Optional controlling LD scores. (``default=None``)
    """

    expansion_window: pydantic.NonNegativeInt = cast(pydantic.NonNegativeInt, 100_000)
    maf_threshold: types.HalfUnitFloat = cast(
        types.HalfUnitFloat, variant_selection.DEFAULT_MAF_THRESHOLD
    )
    precomputed_ld_scores: LDScores | None = None
    controlling_ld_scores: LDScores | None = LDScores(
        dataset=files.default.ldsc_controlling_ld_scores,
    )


class LDSC(workflow.Workflow[Pipeline, Method, Params]):
    r"""Computes functional enrichment using ``LDSC``.

    Stratified LD score regression has been used for functional and tissue enrichment analysis [1]_
    [2]_. It is defined as follows:

    .. math::
        \mathbb{E} \left[ \boldsymbol{\chi}^2 | \mathbf{C}, \mathbf{K} \right] &= \mathbf{R} \mathbf{z} + b \\
        \mathbf{z} &= \mathbf{C}^T \boldsymbol{\alpha} + \mathbf{K}^T \boldsymbol{\beta}

    where :math:`\mathbf{R}` is the square LD correlation matrix. Similarly to
    :class:`.GARFIELD`, the enrichment of partition :math:`i` is defined as:

    .. math::
        e_i = \frac{1}{P} \beta_i

    where :math:`P` is the sample size (i.e. number of patients) used in the GWAS. From LD score
    regression, we can see that this is the total heritability of partition :math:`i` [1]_. From
    this enrichment we get the :math:`p`-value in the usual way:

    .. math::
        p_i = 1 - F\left(  \frac{e_i P}{SE(e_i P)} \right)

    The adjacency matrix :math:`\mathbf{L}` is again constructed in a slightly different way. Here,
    every variant is considered individually, and considered "linked" to a particular annotation if
    it is within 100kb of that annotation. Moreover, :math:`\mathbf{C}` is usually another
    :math:`\mathbf{K}`, derived from a different :math:`\mathbf{X}` and :math:`\mathbf{L}` --- i.e.
    different annotations and corresponding partitions.

    References:
        .. [1] Finucane, H.K., Bulik-Sullivan, B., Gusev, A., Trynka, G., Reshef, Y., Loh, P.R.,
            Anttila, V., Xu, H., Zang, C., Farh, K. and Ripke, S., 2015. Partitioning heritability
            by functional annotation using genome-wide association summary statistics.
            Nature genetics, 47(11), pp.1228-1235.
        .. [2] Finucane, H.K., Reshef, Y.A., Anttila, V., Slowikowski, K., Gusev, A., Byrnes, A.,
            Gazal, S., Loh, P.R., Lareau, C., Shoresh, N. and Genovese, G., 2018. Heritability
            enrichment of specifically expressed genes identifies disease-relevant tissues and cell
            types. Nature genetics, 50(4), pp.621-629.
    """

    @classmethod
    def default(
        cls, annotation: data.AnnotationComponent | None = None, params: Params = Params()
    ) -> Self:
        """Returns the default configuration of ``LDSC``.

        Args:
            annotation: The annotation to use. If ``None``, the PEM annotation from GTEx will be
                used. (``default=None``)
            params: The parameters to use. (``default=LDSC.Params()``)
        """
        if annotation is None:
            annotation = data.GTEx(
                filter_operation=filtering.QuantileCutoff(
                    value_col="pem",
                    threshold=0.9,
                    partition_by=(data.GTEx.columns.partition_id,),
                )
            )
            annotation.columns.values = "pem"
        control_ld = (
            tuple(params.controlling_ld_scores.get_partitions())
            if params.controlling_ld_scores is not None
            else ()
        )
        return cls(
            pipeline=Pipeline(
                locus_definition=locus_def.Compose(
                    locus_def.Expand(extension=(params.expansion_window, params.expansion_window)),
                    locus_def.Overlap(),
                    annotation=annotation,
                ),
                variant_selector=variant_selection.Compose(
                    variant_selection.FilterMAF(threshold=params.maf_threshold),
                    variant_selection.DropInvalidPValues(),
                ),
                precomputed_ld_scores=params.precomputed_ld_scores,
                controlling_ld_scores=params.controlling_ld_scores,
                ld_component=data.LDComponent(
                    r2_threshold=None,
                    return_self_ld=False,
                    return_unbiased_r2=True,
                    columns=data.LDReferenceColumns(
                        # sample size obligatory in LD score regression for computing the LD scores
                        # (require the bias corrected R2)
                        sample_size="sample_size",
                    ),
                ),
            ),
            method=Method(
                partition_id_col=annotation.columns.partition_id,
                control_ld=control_ld + (CONTROL_COL,),
            ),
        )

    def compute_partitioned_ld_scores(
        self, batch_size: int | None = None
    ) -> DataFrame | list[DataFrame]:
        """Computes the partitioned LD scores for the provided annotation data.

        It is generally recommended to precompute the partitioned LD scores before running any
        analysis, as this is a very compute intensive operation. For this purpose, invoke this
        method and then write the resulting dataframe to disk. The stored LD scores can then be
        loaded by using the ``with_precomputed_ld_scores`` method.

        Args:
            batch_size: An optional batch size argument, specifying how many unique partitions will
                be processed at once. This is useful if the dataset has many partitions and is
                difficult to process all at once. If None, all partitions will be processed at once.
                (``default=None``)

        Returns:
            Dataframe containing the precomputed LD scores or a list of dataframes, one dataframe
            per batch (+ 1 for the control LD scores).

        Examples:

            The following example would compute the partitioned LD scores for the GTEx dataset:

            >>> model = LDSC.default()
            >>> ld_scores = model.compute_partitioned_ld_scores()
            >>> ld_scores.columns
            ['partition_id', 'rsid', 'ancestry', 'ld_score', 'n_variants']
            >>> ld_scores.write.parquet("some/file/path")

            Another example, where we set the batch size (useful for *very* large datasets):

            >>> from pyspark.sql import SparkSession
            >>> spark: SparkSession
            >>>
            >>> for ix, ld_scores in enumerate(model.compute_partitioned_ld_scores(batch_size=10)):
            ...     (
            ...         ld_scores.write.mode("overwrite" if ix == 0 else "append")
            ...         .parquet("some/file/path")
            ...     )
            >>> # clear the cache to avoid memory build-up (caching done internally in LD
            ... # computation)
            >>> spark.catalog.clearCache()

        Notes:
            If ``batch_size`` is specified, make sure to clear the cache after saving the dataframes
            in the returned list. This is because we use caching internally to speed up some
            (otherwise repeated) computation. The easiest way to do this is by using
            ``SparkSession.catalog.clearCache`` (see Examples).
        """
        if batch_size is not None:
            return self.pipeline.compute_partitioned_ld_scores_and_control_batched(batch_size)
        return self.pipeline.compute_partitioned_ld_scores_and_control()

    def with_precomputed_ld_scores(
        self, dataset: data.Dataset, columns: LDScoreColumns = LDScoreColumns()
    ) -> "LDSC":
        """Sets the precomputed LD scores.

        Args:
            dataset: Path to the precomputed LD scores, or the dataframe itself.
            columns: The column names. (``default=LDScoreColumns()``)

        Returns:
            The ``LDSC`` instance with the precomputed LD scores set.

        Examples:
            >>> model = LDSC.default()
            >>> model.with_precomputed_ld_scores("some/file/path")
        """
        self.pipeline.precomputed_ld_scores = LDScores(dataset=dataset, columns=columns)
        return self

    def with_controlling_ld_scores(
        self, dataset: data.Dataset, columns: LDScoreColumns = LDScoreColumns()
    ) -> "LDSC":
        """Sets the controlling LD scores.

        Args:
            dataset: Path to the controlling LD scores, or the dataframe itself.
            columns: The column names. (``default=LDScoreColumns()``)

        Returns:
            The ``LDSC`` instance with the controlling LD scores set.

        Examples:
            >>> model = LDSC.default()
            >>> model.with_controlling_ld_scores("some/file/path")
        """
        self.pipeline.controlling_ld_scores = LDScores(
            dataset=dataset,
            columns=columns,
        )
        return self
