from funkea.core.utils import partition


class TestPartitionByMixin:
    def test_get_partition_cols(self):
        part = partition.PartitionByMixin()
        assert part.get_partition_cols() == partition.DEFAULT_PARTITION_COLS
        assert part.get_partition_cols("hello") == partition.DEFAULT_PARTITION_COLS + ("hello",)
        assert (
            part.get_partition_cols(*partition.DEFAULT_PARTITION_COLS)
            == partition.DEFAULT_PARTITION_COLS
        )

    def test_partition_by(self):
        part = partition.PartitionByMixin()
        part.partition_by("hello")
        assert part.get_partition_cols() == ("hello",)

        # resetting for other tests (this is global)
        part.reset()

    def test_reset(self):
        part = partition.PartitionByMixin()
        part.partition_by("hello")
        part.reset()
        assert part.get_partition_cols() == partition.DEFAULT_PARTITION_COLS

    def test_context(self):
        part = partition.PartitionByMixin()

        assert part.get_partition_cols() == partition.DEFAULT_PARTITION_COLS
        with part.partition_by("hello"):
            assert part.get_partition_cols() == ("hello",)

        assert part.get_partition_cols() == partition.DEFAULT_PARTITION_COLS

        with part:
            assert part.get_partition_cols() == partition.DEFAULT_PARTITION_COLS
