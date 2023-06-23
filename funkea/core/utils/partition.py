"""Utilities for implicit, global partitioning."""
import collections
from typing import Final

from pyspark.sql import Window, WindowSpec
from typing_extensions import Self

PartitionByType = tuple[str, ...]
DEFAULT_PARTITION_COLS: Final[PartitionByType] = ("id",)


class PartitionByMixin:
    """Provides functionalities and global source of truth for partitioning.

    Taking an idea from deep learning, the batch dimension allows for seamless parallel processing
    of multiple examples. Here we do the same, defining our own batch dimension as a concatenation
    of multiple dataframe columns.

    The primary motivation was to be able to process multiple GWA studies at once, treating each
    study as an independent example. Since many methods have any number of aggregation operations,
    it was important to have a way to ensure that these did not spill across studies.
    """

    _partition_by_history: collections.deque[PartitionByType] = collections.deque()
    partition_cols: PartitionByType = DEFAULT_PARTITION_COLS

    def get_partition_cols(self, *additional_cols: str) -> PartitionByType:
        unique_additional_cols: tuple[str, ...] = tuple(
            set(additional_cols).difference(self.partition_cols)
        )
        return self.partition_cols + unique_additional_cols

    def partition_by(self, *cols: str) -> Self:
        PartitionByMixin._partition_by_history.append(PartitionByMixin.partition_cols)
        PartitionByMixin.partition_cols = cols
        return self

    def reset(self) -> Self:
        if len(PartitionByMixin._partition_by_history) == 0:
            return self
        PartitionByMixin.partition_cols = PartitionByMixin._partition_by_history.pop()
        return self

    def get_window(self, *additional_cols: str) -> WindowSpec:
        return Window.partitionBy(*self.get_partition_cols(*additional_cols))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()
