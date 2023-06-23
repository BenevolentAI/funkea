import abc
from typing import Final, cast

import pydantic
import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import Column, DataFrame

from funkea.core import data
from funkea.core.utils import files, functions, jvm, partition, types

__all__ = [
    "VariantSelection",
    "Identity",
    "AssociationThreshold",
    "DropHLA",
    "DropIndel",
    "DropComplement",
    "LDPrune",
    "FilterMAF",
    "DropInvalidPValues",
    "Compose",
]

DEFAULT_ASSOCIATION_THRESHOLD: Final[types.UnitFloat] = cast(types.UnitFloat, 5e-8)
DEFAULT_MAF_THRESHOLD: Final[types.HalfUnitFloat] = cast(types.HalfUnitFloat, 0.01)
ALLOWED_BASES: Final[list[str]] = ["A", "C", "G", "T"]


class VariantSelection(Transformer, metaclass=abc.ABCMeta):
    r"""Base class for variant selection."""
    pass


class Identity(VariantSelection):
    r"""The identity transform."""

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset


class AssociationThreshold(VariantSelection):
    r"""Filters out variants which fall above the specified association threshold.

    This transform removes variants which are less significant than a given threshold. The
    (significance of the) association is assumed to be a :math:`p`-value.

    Args:
        threshold: The association threshold. A float in the range :math:`[0, 1]`.
            (``default=5e-8``)
        association_column: The column name which holds the association scores (:math:`p`-values).
    """

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        threshold: types.UnitFloat = DEFAULT_ASSOCIATION_THRESHOLD,
        association_column: str = data.ASSOCIATION_COL,
    ):
        super(AssociationThreshold, self).__init__()
        self.threshold = threshold
        self.association_columns = association_column

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.filter(F.col(self.association_columns) < self.threshold)


class DropHLA(VariantSelection):
    r"""Filters out the Human Leukocyte Antigen (HLA) locus.

    This is a highly problematic locus in GWAS, and is commonly excluded. Some studies, however, do
    not exclude this locus, hence making it good practice to add this safeguard.

    Args:
        start_pos: The base-pair position of the start of the HLA locus. (``default=28477797``)
        end_pos: The base-pair position of the end of the HLA locus. (``default=33448354``)
        chromosome: The chromosome of the HLA locus. (``default=6``)

    Notes:
        The locus coordinates used as the default here are from build 37 of the human genome.
    """

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        # these coordinates are for build 37
        start_pos: pydantic.NonNegativeInt = 28477797,
        end_pos: pydantic.NonNegativeInt = 33448354,
        chromosome: pydantic.NonNegativeInt = 6,
    ):
        super(DropHLA, self).__init__()
        self.start = start_pos
        self.end = end_pos
        self.chr = chromosome

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.filter(
            ~(
                (F.col(data.CHROMOSOME_COL) == self.chr)
                & functions.overlap(
                    F.lit(self.start),
                    F.lit(self.end),
                    F.col(data.POSITION_COL),
                )
            )
        )


class DropIndel(VariantSelection):
    r"""Drops the indel variants from the sumstats.

    This transform ensures that all variants in the sumstats are just single nucleotide
    polymorphisms (SNP).

    Args:
        effect_allele_col: The column name of the effect allele (i.e. "alt").
        other_allele_col: The column name of the other allele (i.e. "ref").
    """

    def __init__(
        self, effect_allele_col: str = "effect_allele", other_allele_col: str = "other_allele"
    ):
        super(DropIndel, self).__init__()
        self.effect_allele_col = effect_allele_col
        self.other_allele_col = other_allele_col

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.filter(
            F.col(self.effect_allele_col).isin(ALLOWED_BASES)
            & F.col(self.other_allele_col).isin(ALLOWED_BASES)
        )


class DropComplement(VariantSelection):
    r"""Removes ambiguous variants.

    Ambiguous variants are cases when the effect and other (alt and ref) are each other's
    complement. In certain cases, one may want to remove these.

    Args:
        effect_allele_col: Name of the column holding the effect (alt) allele.
        other_allele_col: Name of the column holding the other (ref) allele.

    """

    def __init__(
        self, effect_allele_col: str = "effect_allele", other_allele_col: str = "other_allele"
    ):
        super(DropComplement, self).__init__()
        self.effect_allele_col = effect_allele_col
        self.other_allele_col = other_allele_col

    def _transform(self, dataset: DataFrame) -> DataFrame:
        condition: Column = F.lit(False)
        effect, other = F.col(self.effect_allele_col), F.col(self.other_allele_col)
        for e, o in [
            # list of all complement substitutions
            ("A", "T"),
            ("T", "A"),
            ("G", "C"),
            ("C", "G"),
        ]:
            condition = condition | ((effect == e) & (other == o))
        return dataset.filter(~condition)


class LDPrune(VariantSelection, partition.PartitionByMixin):
    r"""Prunes the variants in the sumstats based on linkage disequilibrium.

    LD pruning is done by sorting variants by :math:`p`-value and then excluding each variant if it
    was the LD proxy to any previously retained variant. It is a way of reducing the number of
    correlated variants, and will sometimes be referred to as a set of "independent" variants.

    Args:
        ld_component: The LD matrix definition. Only variants with an :math:`r^2` below the provided
            threshold will be retained. For example, if :math:`r^2 > 0.2`, then variants in the
            pruned sumstats will have pairwise LD of :math:`r^2 \leq 0.2`.
        order_by: Sequence of columns to order by when doing the pruning. Order is ascending. If
            there are ties (e.g. two variants have the same :math:`p`-value), then the tie is broken
            randomly.

    Notes:
          This transform requires the Scala UDF, which is provided as a precompiled ``jar``. Make
          sure to provide the jar when creating ``SparkSession`` using the ``spark.jars`` option.

    Raises:
        RuntimeError: if the Scala UDF jar file is not specified in the active ``SparkSession``.
    """

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        ld_component: data.LDComponent = data.LDComponent(),
        order_by: tuple[str, ...] = (data.ASSOCIATION_COL, data.POSITION_COL),
    ):
        super(LDPrune, self).__init__()
        self.ld_component = ld_component
        self.order_by = order_by

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if not functions.jar_in_session(files.SCALA_UDF.name):
            raise RuntimeError(
                f"{files.SCALA_UDF.name!r} required for LD pruning. Add it during Spark "
                "initialisation using the `spark.jars` option."
            )

        jvm_f: jvm.JVMFunction = jvm.get_jvm_udf("ai.benevolent.VariantIsIndependent")
        return (
            dataset.join(
                self.ld_component.load_tagged_variants(),
                on=[data.ANCESTRY_COL, data.ID_COL],
                how="left",
            )
            .withColumn(
                "independent",
                jvm_f(data.ID_COL, self.ld_component.tags_col).over(
                    # monotonically increasing ID breaks ties randomly
                    self.get_window("chr").orderBy(*self.order_by, F.monotonically_increasing_id())
                ),
            )
            .filter(F.col("independent"))
            .drop(self.ld_component.tags_col, "independent")
        )


class FilterMAF(VariantSelection):
    r"""Filters the sumstats based on Minor Allele Frequency (MAF).

    Args:
        threshold: The minimum MAF. Should be a float in range :math:`[0, 1]`. (``default=0.01``)
        maf_col: The name of the column containing the MAF. (``default="maf"``)
    """

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self, threshold: types.HalfUnitFloat = DEFAULT_MAF_THRESHOLD, maf_col: str = "maf"
    ):
        super(FilterMAF, self).__init__()
        self.threshold = threshold
        self.maf_col = maf_col

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.filter(F.col(self.maf_col) > self.threshold)


class DropInvalidPValues(VariantSelection):
    r"""Drops variants with :math:`p`-values outside the range :math:`(0, 1]`."""

    def _transform(self, dataset: DataFrame) -> DataFrame:
        p = F.col(data.ASSOCIATION_COL)
        return dataset.filter((0.0 < p) & (p <= 1.0))


class Compose(VariantSelection):
    r"""Compose a sequence of variant selection steps."""

    def __init__(self, *steps: VariantSelection):
        super(Compose, self).__init__()
        self.steps = steps

    def _transform(self, dataset: DataFrame) -> DataFrame:
        for step in self.steps:
            dataset = step.transform(dataset)
        return dataset
