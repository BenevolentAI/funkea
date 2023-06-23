r"""Set of transforms which take selected variants and create loci from them.

This module contains a set of transforms which take a set of variants and create loci from them. The
definitions can be chained together to create more complex definitions. For example, we can define
a locus as the lead variant of a GWAS study, and then expand the locus to include all variants in
LD with the lead variant. We can then overlap the locus with a set of regulatory elements.

The locus definition transforms are designed to be used in a :class:`funkea.core.pipeline.Pipeline`
object. The pipeline object takes a set of sumstats and a set of locus definition transforms, and
applies the transforms to the sumstats in order to create a set of loci. The pipeline object also
defines a set of transforms which operate on the loci, and applies them to the loci in order to
create a set of features.
"""
import abc
import functools
import warnings
from typing import Final, Literal

import pydantic
import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from typing_extensions import Self

from funkea.core import data
from funkea.core.utils import functions, partition, types

__all__ = [
    "LocusColumns",
    "LocusDefinition",
    "Identity",
    "AddLDProxies",
    "Expand",
    "Overlap",
    "Collect",
    "Merge",
    "Compose",
]


class LocusColumns(pydantic.BaseModel):
    r"""Column names of the locus dataframe.

    Attributes:
        id: Unique identifier of the locus.
        start: The chromosomal start position of the locus.
        end: The chromosomal end position of the locus.
        chr: The chromosome identifier for the locus.
        association: The column containing the disease association statistic of the locus (e.g. the
            :math:`p`-value of lead variant in the locus).
    """
    id: str = "locus_id"
    start: str = "locus_start"
    end: str = "locus_end"
    chr: str = data.CHROMOSOME_COL
    association: str = data.ASSOCIATION_COL


class LocusDefinition(Transformer, partition.PartitionByMixin, metaclass=abc.ABCMeta):
    r"""Base class for locus definition transforms.

    Contains some callbacks for smooth operation (column name renames etc.) and propagates the
    annotation data component to all sub-definitions.

    Args:
        annotation: The annotation data component.
    """
    locus_columns: LocusColumns = LocusColumns()
    annotation: data.AnnotationComponent | None

    def __init__(self, annotation: data.AnnotationComponent | None = None):
        super(LocusDefinition, self).__init__()
        self.annotation = annotation
        self.propagate_annotations()

    def propagate_annotations(self) -> Self:
        sub_definition: LocusDefinition
        for sub_definition in (
            attribute
            for attribute in vars(self).values()
            if issubclass(type(attribute), LocusDefinition)
        ):
            sub_definition.annotation = self.annotation
        return self

    def init_locus(self, dataset: DataFrame) -> DataFrame:
        cols = self.locus_columns
        for expected, observed in [
            (cols.id, data.ID_COL),
            (cols.start, data.POSITION_COL),
            (cols.end, data.POSITION_COL),
            (cols.association, data.ASSOCIATION_COL),
        ]:
            if expected not in dataset.columns:
                dataset = dataset.withColumn(expected, F.col(observed))
        return dataset

    def transform(self, dataset: DataFrame, params: types.ParamMap | None = None) -> DataFrame:
        return super(LocusDefinition, self).transform(self.init_locus(dataset), params)


class Identity(LocusDefinition):
    r"""The identity transform.

    Returns sumstats dataframe as is. Useful for when no locus definition is required (``Pipeline``
    objects require a locus definition at initialisation).
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset


class AddLDProxies(LocusDefinition):
    r"""Adds LD proxies to the sumstats variants.

    LD proxies are any variant in LD with a variant in the sumstats, above a given :math:`r^2`
    threshold. For example, if we had a variant ``rs1`` which had an :math:`r^2 > \theta` with
    variants ``rs2`` and ``rs3``, but not with ``rs4``, then ``[rs2, rs3]`` would be its LD proxies
    (:math:`\theta \in [0, 1]` is a user-define threshold).

    Args:
        ld_component: The component loading the :math:`r^2` matrix.
    """

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(self, ld_component: data.LDComponent | None = None):
        self.ld_component = ld_component
        super(AddLDProxies, self).__init__()

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.ld_component is None:
            raise ValueError(
                "`ld_component` can not be None. Make sure to set it at initialisation."
            )

        if not self.ld_component.return_self_ld:
            warnings.warn(
                "It is recommended to set `return_self_ld` in `ld_component` to True, to avoid "
                "losing variants in the join operation happening internally in `AddLDProxies`. "
                "If set to False, some variants could be lost.",
                UserWarning,
            )

        cols = self.locus_columns
        ld_cols = self.ld_component.columns
        return (
            dataset.join(
                self.ld_component.load_full_symmetric().select(
                    F.col(ld_cols.source_id).alias(cols.id),
                    ld_cols.target_pos,
                    F.col(ld_cols.ancestry).alias(data.ANCESTRY_COL),
                ),
                on=[data.ANCESTRY_COL, cols.id],
                how="left",
            )
            .withColumn(
                ld_cols.target_pos,
                F.when(F.col(ld_cols.target_pos).isNull(), F.col(data.POSITION_COL)).otherwise(
                    F.col(ld_cols.target_pos)
                ),
            )
            .drop(
                cols.start,
                cols.end,
            )
            .withColumn(cols.start, F.col(ld_cols.target_pos))
            .withColumnRenamed(ld_cols.target_pos, cols.end)
        )


class Expand(AddLDProxies):
    r"""Expands the range of the loci.

    By defining a set of LD proxies, and / or a fixed range, a locus will be expanded. That is, we
    change the starting and ending positions of the loci. In the case of LD proxies, the new
    start and end positions will be the furthest LD proxies from the lead variant. When using
    ``extension = (x, y)``, the start and end are shifted by ``-x`` and ``+y`` respectively.

    This can produce negative start positions or end positions which extend beyond the length of the
    chromosome. However, for our use-cases this does not matter (purely cosmetic). If it is
    important the constraints are respected, the ``chromosomes`` component can be passed, which will
    ensure that all start positions are non-negative and that all end positions are less than the
    length of the respective chromosome.

    However, it is essential to avoid this correction if downstream transforms need to shrink the
    loci back down using some fixed length.
    (e.g. ``funkea.implementations.snpsea.CollectExpandedIfEmpty``)

    Args:
        ld_component: Optional LD matrix definition. Used to get the LD proxies. If None, no LD
            proxy-based expansion will happen. (``default=None``)
        extension: A tuple of non-negative integers, specifying the length of the extension in
            either direction. (``default=(0, 0)``)
        chromosomes: An optional chromosome data component, specifying the maximum length of each
            of the chromosomes. If specified, will ensure that all locus start positions are
            non-negative and that all end positions do not extend beyond the chromosome lengths.
    """

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        ld_component: data.LDComponent | None = None,
        extension: tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt] = (0, 0),
        chromosomes: data.ChromosomeComponent | None = None,
    ):
        super(Expand, self).__init__(ld_component=ld_component)
        self.extension = extension
        self.chromosomes = chromosomes

    def _transform(self, dataset: DataFrame) -> DataFrame:
        cols = self.locus_columns
        if self.ld_component is not None:
            parts = self.get_partition_cols(cols.id)
            dataset = dataset.drop(cols.start, cols.end).join(
                super(Expand, self)
                ._transform(dataset)
                .groupBy(*parts)
                .agg(
                    F.min(cols.start).alias(cols.start),
                    F.max(cols.end).alias(cols.end),
                ),
                on=list(parts),
                how="inner",
            )

        start, end = self.extension
        dataset = dataset.withColumn(cols.start, F.col(cols.start) - start).withColumn(
            cols.end, F.col(cols.end) + end
        )
        if self.chromosomes is not None:
            # ensuring that start is non-negative and end is less that chr length is purely
            # cosmetic
            dataset = (
                dataset.join(
                    self.chromosomes.load().select(
                        F.col(self.chromosomes.chr_col).alias(cols.chr), self.chromosomes.length_col
                    ),
                    on=cols.chr,
                    how="inner",
                )
                .withColumn(cols.start, F.greatest(cols.start, F.lit(0)))
                .withColumn(cols.end, F.least(cols.end, self.chromosomes.length_col))
                .drop(self.chromosomes.length_col)
            )
        return dataset


class Overlap(LocusDefinition):
    r"""Overlaps the variants with the annotation data.

    Produces all the locus-annotation combinations, which overlap with respect to their genomic
    coordinates. By default, overlap is established by checking each locus-annotation combination
    which share a chromosome. While this may seem excessive, due to Spark's optimisations this
    usually works well. However, it is important to keep this in mind if either sumstats carry many
    variants, or there are many annotations.

    In cases when this join becomes too expensive, consider assigning the variants and annotations
    to LD blocks, and then joining on those instead.

    Args:
        annotation: The annotation data component. These are the genome-wide annotations, which
            will be overlapped with the variants in the sumstats. If None, an error will be raised
        join_columns: A column name or list of column names specifying on which columns to join the
            sumstats and the annotation data on. (``default="chr"``)
        keep_variants: Whether to keep all variants in the sumstats. Since only the loci which
            overlap at least one annotation are kept by default, setting this to true makes sure
            that loci are not lost. (``default=False``)
    """

    def __init__(
        self,
        # `annotation` is None by default for consistency with the base class
        annotation: data.AnnotationComponent | None = None,
        join_columns: list[str] | str = "chr",
        keep_variants: bool = False,
    ):
        super(Overlap, self).__init__(annotation=annotation)
        self.join_columns = join_columns
        self.keep_variants = keep_variants

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure this is set during initialisation such that "
                "overlap can be performed."
            )

        cols = self.locus_columns
        ann_cols = self.annotation.columns
        out = dataset.join(self.annotation.load(), on=self.join_columns, how="inner").filter(
            functions.overlap(
                F.col(ann_cols.start),
                F.col(ann_cols.end),
                F.col(cols.start),
                F.col(cols.end),
            )
        )
        if self.keep_variants:
            parts = self.get_partition_cols(cols.id)
            out = out.unionByName(
                dataset.join(
                    out.select(*parts),
                    on=list(parts),
                    how="anti",
                ),
                allowMissingColumns=True,
            )
        return out


class Collect(LocusDefinition):
    r"""Collects set of overlapped annotations for each locus.

    Expects a dataframe of loci overlapped with annotation data (output from ``Collect``) and
    aggregates the annotations into sets.

    Args:
        what: One of ``{"annotation", "partition"}``, defining which part of the overlapped
            annotation to collect -- i.e. whether to collect the unique annotations (e.g.
            ``gene_id``) or unique partitions (e.g. ``tissue``).
    """
    collected_col: Final[str] = "locus_collection"

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        what: Literal["annotation", "partition"] = "annotation",
        annotation: data.AnnotationComponent | None = None,
    ):
        super(Collect, self).__init__(annotation=annotation)
        self.what = what

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.annotation is None:
            raise ValueError(
                "`annotation` is None. Make sure this is set during initialisation, as it is "
                "required for the collection process."
            )

        ann_cols = self.annotation.columns
        if not {ann_cols.annotation_id, ann_cols.partition_id}.intersection(dataset.columns):
            raise RuntimeError(
                f"Input dataframe has neither {ann_cols.annotation_id!r} nor "
                f"{ann_cols.partition_id!r} columns. Make sure to apply the `Overlap` transform to "
                f"the sumstats before applying `Collect`."
            )

        cols = self.locus_columns
        agg_col = {"annotation": ann_cols.annotation_id, "partition": ann_cols.partition_id}[
            self.what
        ]
        return dataset.groupBy(*self.get_partition_cols(cols.id)).agg(
            F.first(cols.start).alias(cols.start),
            F.first(cols.end).alias(cols.end),
            F.first(cols.chr).alias(cols.chr),
            F.first(cols.association).alias(cols.association),
            F.collect_set(agg_col).alias(self.collected_col),
        )


class Merge(LocusDefinition):
    r"""Merges loci if their annotation sets overlap.

    Suppose we have loci ``a`` and ``b``, then these would be merged if their annotation sets (see
    ``Collect``) overlapped. For example, if ``a`` and ``b`` both have gene ``z`` in them, they
    would be merged into a single locus ``a union b``.
    """

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if Collect.collected_col not in dataset.columns:
            raise RuntimeError(
                f"{Collect.collected_col!r} is missing from the input dataframe columns. Make sure "
                f"to apply `Collect` before applying `Merge`."
            )

        partition_cols = self.get_partition_cols(self.locus_columns.chr)
        return dataset.groupBy(*partition_cols).applyInPandas(
            functools.partial(
                functions.merge_collections,
                id_column=self.locus_columns.id,
                collections_column=Collect.collected_col,
                partition_cols=partition_cols,
            ),
            dataset.select(*partition_cols, self.locus_columns.id, Collect.collected_col).schema,
        )


class Compose(LocusDefinition):
    r"""Compose a sequence of locus definition objects.

    Args:
        steps: a sequence of locus definition steps.
        annotation: the annotation data component.
    """

    def __init__(self, *steps: LocusDefinition, annotation: data.AnnotationComponent):
        self.steps = steps
        for ix, step in enumerate(steps):
            # setting these attributes so `propagate_annotations` works properly
            setattr(self, f"{step.__class__.__name__.lower()}_{ix}", step)

        # additionally, we want to call the super class init _after_ the setting of the attributes
        # such that `propagate_annotations` works as intended
        super(Compose, self).__init__(annotation=annotation)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        for step in self.steps:
            dataset = step.transform(dataset)
        return dataset
