import abc

from pyspark.ml import Transformer

from funkea.components import locus_definition as locus_def
from funkea.components import variant_selection
from funkea.core.utils import partition


class DataPipeline(Transformer, partition.PartitionByMixin, metaclass=abc.ABCMeta):
    """Base class for data pipeline."""

    locus_definition: locus_def.LocusDefinition
    variant_selector: variant_selection.VariantSelection | None

    def __init__(
        self,
        locus_definition: locus_def.LocusDefinition,
        variant_selector: variant_selection.VariantSelection | None = None,
    ):
        super(DataPipeline, self).__init__()
        self.locus_definition = locus_definition
        self.variant_selector = variant_selector

    def get_variant_selector(self) -> variant_selection.VariantSelection:
        """Gets variant selector transform."""
        return self.variant_selector or variant_selection.Identity()
