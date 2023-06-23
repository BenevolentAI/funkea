"""Data and annotation components."""
import abc
import enum
import warnings
from typing import Final, cast

import pydantic
import pydantic_spark.base
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Column, DataFrame

from funkea.components import filtering
from funkea.components import normalisation as norm
from funkea.core.utils import files, functions, types

Dataset = DataFrame | str

ID_COL: Final[str] = "rsid"
ASSOCIATION_COL: Final[str] = "p"
POSITION_COL: Final[str] = "pos"
CHROMOSOME_COL: Final[str] = "chr"
ANCESTRY_COL: Final[str] = "ancestry"


class Sumstats(pydantic_spark.base.SparkBase):
    """Sumstats schema for validation purposes."""

    rsid: str | None = pydantic.Field(
        default=None,
        title="Unique variant identifier.",
        description="An identifier which should ideally be unique for each variant.",
    )
    chr: int | None = pydantic.Field(
        default=None,
        title="Chromosome identifier.",
        description="The identifier (number) of the chromosome. Assumes numbers, which means that "
        "X will be 23 and Y will be 24.",
    )
    pos: int | None = pydantic.Field(
        default=None,
        title="Variant position.",
        description="The (base-pair) position of the variant on its chromosome.",
    )
    p: float | None = pydantic.Field(
        default=None,
        title="Significance of association.",
        description="The p-value of the association of a variant with a given trait.",
    )

    @classmethod
    def spark_schema(cls) -> T.StructType:
        schema = T.StructType.fromJson(super(Sumstats, cls).spark_schema())
        # the parent class adds a metadata field which causes unnecessary mismatches
        for field in schema:
            field.metadata = {}
        return schema


class DataComponent(abc.ABC):
    """Loads tabular data either from disk or memory.

    A data component is an object which provides programs access to a particular data source in
    tabular format. These sources can be provided either as paths to static files, or dataframes
    loaded into the current Spark session. Unlike most other objects in ``funkea``, data components
    do not inherit from ``pyspark.ml.Transformer``, as this would constrain their instantiation to
    only once a SparkSession is active. But since data components can load from disk, initialisation
    should not need to depend on the SparkSession being active.
    """

    dataset: Dataset | None

    def __init__(self, dataset: Dataset | None = None):
        self.dataset = dataset

    def _read(self) -> DataFrame:
        # internal check for mypy
        assert isinstance(self.dataset, str)
        return functions.get_session().read.parquet(self.dataset)

    def _get_dataset(self) -> DataFrame:
        # check for mypy
        assert self.dataset is not None
        return self._read() if isinstance(self.dataset, str) else self.dataset

    def _load(self) -> DataFrame:
        return self._get_dataset()

    def load(self) -> DataFrame:
        """Loads the dataset into a Spark dataframe."""
        if self.dataset is None:
            raise ValueError(
                "`dataset` attribute is None. Set this attribute on the data component (either a "
                "file path or a dataframe). If None, no data can be loaded."
            )
        return self._load()


class LDReferenceColumns(pydantic.BaseModel):
    source_id: str = "rsid_source"
    target_id: str = "rsid_target"
    source_pos: str = "pos_source"
    target_pos: str = "pos_target"
    chromosome: str = "chr"
    correlation: str = "correlation"
    sample_size: str | None = "sample_size"
    ancestry: str = ANCESTRY_COL


def unbiased_r2_estimate(r2: Column, n: Column) -> Column:
    r"""Computes the unbiased estimate of the squared Pearson correlation.

    The bias corrected :math:`r^2` is defined as:

    .. math::

        \hat{r}^2 = r^2 - \frac{1 - r^2}{n - 2}

    where :math:`n` is the sample size used to compute the correlation coefficient.

    Args:
        r2: Column containing the squared correlation coefficients.
        n: Column containing the sample sizes used to compute the correlation.

    Returns:
        A column containing the bias corrected squared correlation coefficients.

    """
    return r2 - (1 - r2) / (n - F.when(n > 2, F.lit(2)).otherwise(F.lit(0)))


class LDComponent(DataComponent):
    """Loads linkage-disequilibrium reference data.

    Data component which handles LD reference data. It is used for getting tagging variants and
    loading full symmetric matrices (in long table format). It also can be used to load the bias
    corrected :math:`r^2`.
    It assumes the LD data to be in the format produced by ``PLINK``; that is, as a long table
    representing the condensed (triangular) correlation matrix. The default column names differ from
    ``PLINK`` however, for consistency with column names we use in other places. Moreover,
    additional (optional) columns can also be included (``sample_size``, ``ancestry``), which
    may be needed for specific analyses (LD score regression, LD pruning).

    Args:
        r2_threshold: Optional minimum pairwise correlation between two variants returned by the
            object. This should be a float in the range :math:`[0, 1]`. If None, no thresholding
            is applied. (``default=0.20``)
        return_unbiased_r2: Whether to apply bias correction to :math:`r^2`. If this is true, the
            LD table has to have a column containing the sample size used for computing the
            correlation. (``default=False``)
        return_self_ld: ``PLINK`` returns a condensed correlation matrix, which omits the
            correlation of a variant with itself (as it is trivial). However, in some cases it is
            useful to retain this relationship, as variants may be lost otherwise
            (see ``funkea.components.locus_defnition.AddLDProxies``). (``default=False``)
        dataset: Dataframe or filepath to the correlation table.
            (``default=funkea.core.utils.files.default.ld_reference_data``)
        columns: Mapping of column names. (``default=LDReferenceColumns()``)
    """

    tags_col: Final[str] = "tags"

    @pydantic.validate_arguments(config=types.PYDANTIC_VALIDATION_CONFIG)
    def __init__(
        self,
        r2_threshold: types.UnitFloat | None = cast(types.UnitFloat, 0.20),
        return_unbiased_r2: bool = False,
        return_self_ld: bool = False,
        # we ought to make this not None and force users to set this
        # the optional adds a lot of risk here and annoying checks
        dataset: Dataset | None = files.default.ld_reference_data,
        columns: LDReferenceColumns = LDReferenceColumns(),
    ):
        self.r2_threshold = r2_threshold
        self.return_unbiased_r2 = return_unbiased_r2
        self.return_self_ld = return_self_ld
        self.columns = columns
        super(LDComponent, self).__init__(dataset=dataset)

    def _load_self_ld(self, dataset: DataFrame) -> DataFrame:
        cols = self.columns

        constants = [cols.ancestry]
        if cols.sample_size is not None:
            constants.append(cols.sample_size)
        metadata = dataset.select(*constants).distinct()

        return dataset.unionByName(
            self.load_variants()
            .withColumnRenamed(ID_COL, cols.source_id)
            .withColumnRenamed(POSITION_COL, cols.source_pos)
            .withColumn(cols.target_id, F.col(cols.source_id))
            .withColumn(cols.target_pos, F.col(cols.source_pos))
            .withColumn(cols.correlation, F.lit(1.0))
            # make sure that all variants also have the relevant metadata
            .crossJoin(metadata)
        )

    def _load(self) -> DataFrame:
        cols = self.columns
        dataset: DataFrame = self._get_dataset()
        if cols.ancestry not in dataset.columns:
            # create a dummy ancestry to make this easier downstream
            dataset = dataset.withColumn(cols.ancestry, F.lit("unknown"))

        if self.return_self_ld:
            dataset = self._load_self_ld(dataset)

        if self.return_unbiased_r2:
            assert cols.sample_size is not None and cols.sample_size in dataset.columns
            dataset = dataset.withColumn(
                cols.correlation,
                unbiased_r2_estimate(F.col(cols.correlation), F.col(cols.sample_size)),
            )

        if self.r2_threshold is not None:
            dataset = dataset.filter(F.col(cols.correlation) > self.r2_threshold)

        return dataset

    def load_tagged_variants(self) -> DataFrame:
        """Loads the variants and their tags.

        Tags here are the variants which are in LD with a given variant, above a certain threshold.
        Unlike in the standard ``load`` function, this function returns each variant-ancestry pair
        and aggregates all the tagging variants into a set.
        This is used primarily in LD pruning, where we need to check that each variant we encounter
        has not been already observed in one of the tags.

        Returns:
            A dataframe with columns for variant ID, ancestry and a set of tagging variants.
        """
        if self.return_self_ld:
            warnings.warn(
                "Loading tagged variants with `return_self_ld=True` is discouraged, as it will "
                "define the variant as tag to itself.",
                UserWarning,
            )

        cols = self.columns
        return (
            self.load_full_symmetric()
            .groupBy(F.col(cols.source_id).alias(ID_COL), cols.ancestry)
            .agg(F.collect_set(cols.target_id).alias(self.tags_col))
        )

    def load_full_symmetric(self) -> DataFrame:
        """Loads the full, symmetric matrix."""
        # we need to deactivate self LD, as it will be duplicated otherwise
        return_self_ld = self.return_self_ld

        self.return_self_ld = False
        dataset = self.load()
        self.return_self_ld = return_self_ld

        cols = self.columns
        constants = [cols.chromosome, cols.correlation, cols.ancestry]
        if cols.sample_size is not None:
            constants.append(cols.sample_size)
        dataset = dataset.unionByName(
            dataset.select(
                *(
                    F.col(column).alias(alias)
                    for column, alias in [
                        (cols.source_id, cols.target_id),
                        (cols.target_id, cols.source_id),
                        (cols.source_pos, cols.target_pos),
                        (cols.target_pos, cols.source_pos),
                    ]
                ),
                *constants,
            )
        )
        if return_self_ld:
            dataset = self._load_self_ld(dataset)
        return dataset

    def load_variants(self) -> DataFrame:
        """Loads all the unique variants in the LD data."""
        dataset = self._get_dataset()
        cols = self.columns
        return (
            dataset.select(cols.source_id, cols.source_pos, cols.chromosome)
            .unionByName(
                dataset.select(
                    F.col(cols.target_id).alias(cols.source_id),
                    F.col(cols.target_pos).alias(cols.source_pos),
                    cols.chromosome,
                )
            )
            .withColumnRenamed(cols.source_id, ID_COL)
            .withColumnRenamed(cols.source_pos, POSITION_COL)
            .distinct()
        )


class ChromosomeComponent(DataComponent):
    """Loads information on the human chromosomes.

    This component is mainly used for providing upper bounds on locus end positions. Since this is
    cosmetic adjustment, it is optional.
    """

    chr_col: Final[str] = "chr"
    length_col: Final[str] = "chromosome_length"

    def __init__(self, dataset: Dataset | None = files.default.chromosomes):
        super(ChromosomeComponent, self).__init__(dataset=dataset)


class AnnotationColumns(pydantic.BaseModel):
    annotation_id: str = "annotation_id"
    partition_id: str = "partition_id"
    start: str = "start"
    end: str = "end"
    chromosome: str = "chr"
    values: str | None = None


class PartitionType(str, enum.Enum):
    """The type of the annotation partitions.

    Partitions can either be "hard" (binary indicator of annotation membership in a partition) or
    "soft" (annotation has distribution over partitions). For example, a hard partition may be an
    indicator of membership of a particular gene (annotation) in a given pathway (partition). A soft
    partition may be the gene expression levels of a given gene across a set of tissues.
    """

    HARD = "hard"
    SOFT = "soft"


class AnnotationComponent(DataComponent, metaclass=abc.ABCMeta):
    """Loads the annotation data.

    Annotations are sequence spans with genome-wide coordinates; that is, they have a start, end
    and chromosome position. It is assumed that ``start <= end``, i.e. the annotation should be
    strand corrected.
    Annotations are furthermore assumed to be partitioned into :math:`K` partitions. For example,
    genes can be partitioned into tissues by activity. Partitioning can also be either "soft" or
    "hard" (see ``PartitionType``).

    Args:
        columns: The columns in the loaded dataframe.
        partition_type: The type of the partitioning.
        filter_operation: An optional filter operation. This can be used to subset the annotations
            at runtime.
        normalisation: An optional normalisation step (this is performed *after* the filter
            operation). Can be used to normalise the ``values`` column at runtime.
    """

    columns: AnnotationColumns
    partition_type: PartitionType
    filter_operation: filtering.FilterOperation | None
    normalisation: norm.Normalisation | None

    def __init__(
        self,
        columns: AnnotationColumns,
        partition_type: PartitionType,
        filter_operation: filtering.FilterOperation | None = None,
        normalisation: norm.Normalisation | None = None,
        dataset: Dataset | None = None,
    ):
        self.columns = columns
        self.partition_type = partition_type
        self.filter_operation = filter_operation
        self.normalisation = normalisation
        super(AnnotationComponent, self).__init__(dataset=dataset)

    def get_filter_operation(self) -> filtering.FilterOperation:
        """Gets the filter operation."""
        return self.filter_operation or filtering.Identity()

    def get_normalisation(self) -> norm.Normalisation:
        """Gets the normalisation operation."""
        return self.normalisation or norm.Identity()

    def _get_unique_col(self, col: str) -> DataFrame:
        return self.load().select(col).distinct()

    def get_annotations_count(self) -> int:
        """Gets the total number of unique annotations.

        Notes:
            This is the count *after* the filter operation.
        """
        return self._get_unique_col(self.columns.annotation_id).count()

    def get_partitions_count(self) -> int:
        """Gets the total number of unique partitions.

        Notes:
            This is the count *after* the filter operation.
        """
        return self._get_unique_col(self.columns.partition_id).count()

    @property
    def hard_partitions(self) -> bool:
        """Tells whether the partition type is hard."""
        return self.partition_type == PartitionType.HARD

    def _load(self) -> DataFrame:
        # TODO: how to handle binary case etc.?
        return (
            self._get_dataset()
            .transform(self.get_filter_operation().transform)
            .transform(self.get_normalisation().transform)
        )


DEFAULT_GTEX_COLUMNS = AnnotationColumns(
    annotation_id="gene_id", partition_id="tissue", values="nTPM"
)


class GTEx(AnnotationComponent):
    """Loads the GTEx data.

    Loads the GTEx annotation dataset, which partitions the genes into 40 tissues. The partitions
    are "soft", as we get an expression value (non-normalised probability) for each gene-tissue
    pair. Hence, for methods which require data with hard partitions, one should use a filter
    operation to just keep certain annotation for each partition (e.g. QuantileCutoff).

    Args:
        filter_operation: An optional filter operation. Recommended if method requires annotations to
            have hard partitions.
        normalisation: Optional normalisation applied to filtered data. The normalisation is applied
            to the ``values`` column. In the case of hard partitions, a dummy ``values`` column will
            be used (if it does not exist).
        dataset: Path to long table GTEx data, or the Spark dataframe containing said data.
    """

    columns = DEFAULT_GTEX_COLUMNS

    def __init__(
        self,
        filter_operation: filtering.FilterOperation | None = None,
        normalisation: norm.Normalisation | None = None,
        dataset: Dataset | None = files.default.gtex,
    ):
        super(GTEx, self).__init__(
            columns=DEFAULT_GTEX_COLUMNS,
            partition_type=PartitionType.SOFT,
            filter_operation=filter_operation,
            normalisation=normalisation,
            dataset=dataset,
        )
