import abc

from pyspark.ml import Transformer

from funkea.core.utils import partition


class EnrichmentMethod(Transformer, partition.PartitionByMixin, metaclass=abc.ABCMeta):
    """Base class for enrichment methods."""

    pass
