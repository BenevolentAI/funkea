import functools
import logging
from typing import Sequence, cast

import numpy as np
import pandas as pd
import patsy
import pydantic
import pyspark.sql.functions as F
import pyspark.sql.types as T
import quinn
import statsmodels.api as sm
import statsmodels.tools.sm_exceptions
from pyspark.sql import Column, DataFrame
from statsmodels.genmod import generalized_linear_model as glm
from typing_extensions import Self

from funkea.components import filtering
from funkea.components import locus_definition as locus_def
from funkea.components import variant_selection
from funkea.core import data, method
from funkea.core import params as params_
from funkea.core import pipeline, workflow
from funkea.core.utils import files, functions, partition, types


class WithinSpan(locus_def.LocusDefinition):
    """Checks whether a lead variant is within a particular distance of an annotation.

    This is an additional requirement in the original implementation of :class:`.GARFIELD`. It is
    applied after the :class:`.Overlap` step in the locus definition.

    Args:
        span: The upstream and downstream distances, within which variant-annotation pairs are
            retained.
        annotation: The annotation component. Held optional, such that it can be passed from
            :class:`~funkea.components.locus_definition.Compose` after initialisation.
            (``default=None``)
    """

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        span: tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt],
        annotation: data.AnnotationComponent | None = None,
    ):
        super(WithinSpan, self).__init__(annotation=annotation)
        self.span = span

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure to set this during initialisation of `WithinSpan` "
                "or passed down from a `Compose` object."
            )

        ann_cols = self.annotation.columns
        if {ann_cols.start, ann_cols.end}.difference(dataset.columns):
            raise RuntimeError(
                "Annotation coordinate columns are missing from the input dataframe. `WithinSpan` "
                "assumes as input a set of sumstats which are already overlapped with annotations. "
                "Make sure to apply `Overlap` before applying `WithinSpan`."
            )

        from_, to = self.span

        def within_distance(col: Column, bp: int) -> Column:
            pos = F.col(data.POSITION_COL)
            return F.abs(F.when(col.isNull(), pos).otherwise(col) - pos) <= bp

        return dataset.filter(
            within_distance(F.col(ann_cols.start), to) | within_distance(F.col(ann_cols.end), from_)
        )


class ControllingCovariates(data.DataComponent):
    """Loads variant-level controlling covariates to include in the regression.

    Args:
        continuous_covariates: A sequence of column names, which define controlling covariates with
            numerical values. (``default=()``)
        categorical_covariates: A sequence of column names, which define the controlling covariates
            with categorical values. (``default=("binned_tss_distance", "binned_n_proxies")``)
        dataset: Path to the controlling covariates, or the dataframe itself.
    """

    id_col: str = data.ID_COL

    def __init__(
        self,
        continuous_covariates: tuple[str, ...] = (),
        categorical_covariates: tuple[str, ...] = ("binned_tss_distance", "binned_n_proxies"),
        dataset: data.Dataset | None = files.default.garfield_control_covariates,
    ):
        self.continuous_covariates = continuous_covariates
        self.categorical_covariates = categorical_covariates
        super(ControllingCovariates, self).__init__(dataset=dataset)

    def _load(self) -> DataFrame:
        out = self._get_dataset()
        quinn.validate_presence_of_columns(
            out, self.continuous_covariates + self.categorical_covariates + (self.id_col,)
        )

        return out


class Pipeline(pipeline.DataPipeline):
    """The GARFIELD data pipeline.

    Args:
        locus_definition: The locus definition component.
        variant_selector: The variant selection component. (``default=None``)
        controlling_covariates: The controlling covariates component. These are the variant-level
            controlling covariates used in the regression. (``default=None``)
    """

    def __init__(
        self,
        locus_definition: locus_def.LocusDefinition,
        variant_selector: variant_selection.VariantSelection | None = None,
        controlling_covariates: ControllingCovariates | None = ControllingCovariates(),
    ):
        self.controlling_covariates = controlling_covariates
        super(Pipeline, self).__init__(
            locus_definition=locus_definition,
            variant_selector=variant_selector,
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.locus_definition.annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure to set this during initialisation of "
                "`locus_definition`."
            )

        cols = self.locus_definition.locus_columns
        ann_cols = self.locus_definition.annotation.columns

        dataset = self.get_variant_selector().transform(dataset)
        loci = self.locus_definition.transform(dataset)

        if locus_def.Collect.collected_col not in loci.columns:
            raise RuntimeError(
                f"{locus_def.Collect.collected_col!r} column is missing from the loci dataframe. "
                "Make sure to include `Overlap` and `Collect` in the `locus_definition`."
            )

        cont_col: tuple[str, ...] = ()
        if self.controlling_covariates is not None:
            covariates = self.controlling_covariates.load()
            loci = loci.join(
                covariates.withColumnRenamed(
                    self.controlling_covariates.id_col, self.locus_definition.locus_columns.id
                ),
                on=self.locus_definition.locus_columns.id,
                how="inner",
            )
            cont_col = (
                self.controlling_covariates.continuous_covariates
                + self.controlling_covariates.categorical_covariates
            )

        return loci.select(
            *self.get_partition_cols(cols.id),
            F.col(locus_def.Collect.collected_col).alias(ann_cols.partition_id),
            F.col(self.locus_definition.locus_columns.association).alias(data.ASSOCIATION_COL),
            *cont_col,
        )


def create_design_matrix(
    df: pd.DataFrame,
    categorical_covariates: list[str],
    continuous_covariates: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # the annotation map is a workaround for an issue with the Patsy formula parser. If there are
    # variables in the formula which do not conform to the python variable standard, there will be
    # an error. One can circumvent this issue by using the `Q()` builtin
    # (https://patsy.readthedocs.io/en/latest/builtins-reference.html#patsy.builtins.Q) but this
    # then causes the issue of having each variable be named `Q(...)` in the final model output.
    # Thus, we map the column names of the design matrix back into the original names, after patsy
    # has generated the matrix.
    def quote(column: str) -> str:
        return f"Q({column!r})"

    column_name_map = {quote(v): v for v in continuous_covariates}
    column_name_map.update({f"C({quote(v)})": v for v in categorical_covariates})
    formula = " + ".join(column_name_map.keys())

    quoted_target = f"Q({target!r})"
    y, X = patsy.dmatrices(
        f"{quoted_target} ~ {formula}",
        data=df,
        return_type="dataframe",
    )
    # does not quite rename all the columns appropriately -- categorical columns will not map back
    # because of their suffixes (and should not map back to avoid collision)
    # this does not really matter for us
    X = X.rename(columns=column_name_map)
    y = y.rename(columns={quoted_target: target})
    return y, X


def fit_garfield_model(
    dataframe: pd.DataFrame,
    annotation_col: str,
    categorical_covariates: list[str],
    continuous_covariates: list[str],
    target_col: str,
    partition_cols: partition.PartitionByType,
) -> pd.DataFrame:
    """Fit a GARFIELD model.

    Args:
        dataframe: The dataframe to fit the model on.
        annotation_col: The name of the annotation column.
        categorical_covariates: The names of the categorical covariates.
        continuous_covariates: The names of the continuous covariates.
        target_col: The name of the target column.
        partition_cols: The names of the partition columns.

    Returns:
        Enrichment results produced by the model.
    """
    parts: list[str] = list(partition_cols)

    def add_if_missing(columns: Sequence[str], column: str) -> list[str]:
        return sorted(set(columns).union([column]))

    # we should only run one annotation at a time
    assert dataframe.groupby(parts).ngroups == 1

    y, X = create_design_matrix(
        dataframe,
        categorical_covariates=categorical_covariates,
        # annotations should either already be binary, i.e. in {0, 1}, or a continuous
        # annotation
        continuous_covariates=add_if_missing(continuous_covariates, annotation_col),
        target=target_col,
    )
    result: pd.DataFrame
    try:
        # Perfect separation errors can occur in some situations, and hence we catch them so as to
        # not crash all the other tasks
        model: glm.GLMResults = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        result = pd.DataFrame(
            [
                {
                    "enrichment": np.exp(model.params[annotation_col]),
                    "p_value": model.pvalues[annotation_col],
                }
            ]
        )
    except statsmodels.tools.sm_exceptions.PerfectSeparationError as exc:
        logging.exception(f"{type(exc)} encountered")
        result = pd.DataFrame([(np.nan, np.nan)], columns=["enrichment", "p_value"])

    return functions.cross_join(
        dataframe[parts].drop_duplicates(),
        result,
    )


class Method(method.EnrichmentMethod):
    """Enrichment method using GARFIELD.

    Args:
        partition_id_col: The name of the column containing the partition IDs.
        continuous_covariates: The names of the continuous covariates.
        categorical_covariates: The names of the categorical covariates.
        association_threshold: The threshold for the association p-value. Variants with a p-value
            below this threshold will be assigned a positive label in the logistic regression.
    """

    def __init__(
        self,
        partition_id_col: str,
        continuous_covariates: tuple[str, ...] = (),
        categorical_covariates: tuple[str, ...] = ("binned_tss_distance", "binned_n_proxies"),
        association_threshold: float = 1e-8,
    ):
        super(Method, self).__init__()
        self.partition_id_col = partition_id_col
        self.continuous_covariates = continuous_covariates
        self.categorical_covariates = categorical_covariates
        self.association_threshold = association_threshold

    def _transform(self, dataset: DataFrame) -> DataFrame:
        out = (
            dataset.crossJoin(
                dataset.select(F.explode(self.partition_id_col).alias("indicate")).distinct()
            )
            .withColumn(
                "target", (F.col(data.ASSOCIATION_COL) < self.association_threshold).cast("byte")
            )
            .withColumn(
                "indicator", F.array_contains(self.partition_id_col, F.col("indicate")).cast("byte")
            )
            .drop(self.partition_id_col)
        )
        parts = self.partition_cols + ("indicate",)
        return (
            out.groupBy(*parts)
            .applyInPandas(
                functools.partial(
                    fit_garfield_model,
                    annotation_col="indicator",
                    categorical_covariates=list(self.categorical_covariates),
                    continuous_covariates=list(self.continuous_covariates),
                    target_col="target",
                    partition_cols=parts,
                ),
                out.select(*parts)
                .schema.add(T.StructField("enrichment", T.DoubleType()))
                .add(T.StructField("p_value", T.DoubleType())),
            )
            .withColumnRenamed("indicate", self.partition_id_col)
        )


class Params(params_.Params):
    """Parameters for the default configuration of :class:`.GARFIELD`.

    Args:
        expansion_r2_threshold: The minimum :math:`r^2` of a variant's LD proxies.
            (``default=0.8``)
        ld_pruning_r2_threshold: The maximum :math:`r^2` between two variants in the GWAS
            sumstats. (``default=0.01``)
        within_span: The maximum distance between a given lead variant and an overlapped
            annotation. (``default=500_000``)
        association_threshold: A :math:`p`-value threshold, where variants below this value
            will be assigned the positive label in the logistic regression. (``default=1e-8``)
        controlling_covariates: The controlling covariates in the regression.
            (``default=ControllingCovariates()``)
    """

    expansion_r2_threshold: types.UnitFloat = cast(types.UnitFloat, 0.8)
    ld_pruning_r2_threshold: types.UnitFloat = cast(types.UnitFloat, 0.01)
    within_span: pydantic.NonNegativeInt = cast(pydantic.NonNegativeInt, 500_000)
    association_threshold: types.UnitFloat = cast(types.UnitFloat, 1e-8)
    controlling_covariates: ControllingCovariates | None = ControllingCovariates()


class GARFIELD(workflow.Workflow[Pipeline, Method, Params]):
    r"""Computes functional enrichment using ``GARFIELD``.

    ``GARFIELD`` is a general method for functional enrichment analysis [1]_. It regresses locus
    association to a given trait on whether the lead variant --- or one of its LD proxies ---
    overlaps a given partition :math:`i`, as well as a set of controlling covariates. Formally:

    .. math::
         \mathbb{E} \left[ \mathbb{1} \left[ \boldsymbol{\chi}^2 > \theta \right] | \mathbf{C}, \mathbf{K} \right] &= \left(1 + \exp \left(- \mathbf{z}  - b \right) \right)^{\odot -1} \\
         \mathbf{z} &= \mathbf{C}^T \boldsymbol{\alpha} + \mathbf{K}^T \boldsymbol{\beta}

    where :math:`\boldsymbol{\chi}^2 \in \mathbb{R}^M` are the association statistics for the LD
    pruned lead variants, :math:`\theta` is some threshold,
    :math:`\mathbf{C} \in \{0, 1\}^{D \times M}` are the controlling covariates and
    :math:`\mathbf{K}` is:

    .. math::
        \mathbf{K} &= t \left( \mathbf{Q} \right) \\
        \mathbf{Q} &= \mathbf{X} \mathbf{L} \\
        t(Q_{il}) &= \mathbb{1} \left[ Q_{il} > 0 \right]

    an indicator matrix showing whether a given locus overlaps an annotation partition. The
    definitions of :math:`\mathbf{X}` and :math:`\mathbf{L}` are the same here as for
    :class:`.SNPsea`; however, :math:`\mathbf{L}` is constructed in a slightly different way. As
    mentioned above, an annotation :math:`j` and a locus :math:`l` are considered "linked", if the
    lead variant or one of its LD proxies overlaps it. If the overlap is with an LD proxy, the
    annotation as to also be withing 500kb of the lead variant.
    Here, we define the controlling covariates :math:`\mathbf{C}` as being binary; however, any type
    of variant-level covariate is valid. In the original formulation of ``GARFIELD``,
    :math:`\mathbf{C}` is divided into two feature type subsets: (1) binned distance from
    transcription start site (TSS), and (2) binned number of LD proxies. Formulated differently,
    :math:`\mathbf{C}` is an indicator matrix, telling us how far from the nearest gene a given lead
    variant is, and how much LD it is experiencing. The enrichment :math:`e_i` for given partition
    :math:`i` is then defined as follows:

    .. math::
        e_i = \exp (\beta_i)

    The significance of the enrichment is then computed in the standard way for linear models:

    .. math::
        p_i = 2 F \left(- \left|\frac{\log e_i}{SE(\log e_i)} \right| \right)

    where :math:`F` is the cumulative density function of a unit Gaussian.

    References:
        .. [1] Iotchkova, V., Ritchie, G.R., Geihs, M., Morganella, S., Min, J.L., Walter, K.,
            Timpson, N.J., UK10K Consortium, Dunham, I., Birney, E. and Soranzo, N., 2019. GARFIELD
            classifies disease-relevant genomic features through integration of functional
            annotations with association signals. Nature genetics, 51(2), pp.343-353.
    """

    @classmethod
    def default(
        cls, annotation: data.AnnotationComponent | None = None, params: Params = Params()
    ) -> Self:
        """Returns the default workflow for ``GARFIELD``.

        Args:
            annotation: The annotation to use for the enrichment. If ``None``, the default
                annotation will be used (GTEx).
            params: The parameters to use for the workflow.

        Returns:
            The default ``GARFIELD`` workflow.
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

        control_var = params.controlling_covariates or ControllingCovariates((), ())
        return cls(
            pipeline=Pipeline(
                locus_definition=locus_def.Compose(
                    locus_def.AddLDProxies(
                        ld_component=data.LDComponent(
                            r2_threshold=params.expansion_r2_threshold,
                            return_self_ld=True,
                        ),
                    ),
                    locus_def.Overlap(keep_variants=True),
                    WithinSpan(span=(params.within_span, params.within_span)),
                    locus_def.Collect(what="partition"),
                    annotation=annotation,
                ),
                variant_selector=variant_selection.Compose(
                    variant_selection.DropComplement(),
                    variant_selection.DropIndel(),
                    variant_selection.DropHLA(),
                    variant_selection.LDPrune(
                        ld_component=data.LDComponent(
                            r2_threshold=params.ld_pruning_r2_threshold,
                        )
                    ),
                ),
                controlling_covariates=params.controlling_covariates,
            ),
            method=Method(
                partition_id_col=annotation.columns.partition_id,
                continuous_covariates=control_var.continuous_covariates,
                categorical_covariates=control_var.categorical_covariates,
                association_threshold=params.association_threshold,
            ),
        )
