import chispa
import pytest

from funkea.components import filtering


class TestIdentity:
    def test_transform(self, annotation_data, precision):
        res = filtering.Identity().transform(annotation_data)
        chispa.assert_approx_df_equality(res, annotation_data, precision=precision)


class TestQuantileCutoff:
    def test_transform(self, annotation_data):
        res = filtering.QuantileCutoff(
            value_col="score", threshold=0.9, partition_by=("partition_id",)
        ).transform(annotation_data)
        assert (
            res.select("partition_id").distinct().count()
            == annotation_data.select("partition_id").distinct().count()
        )
        assert res.select("annotation_id").distinct().count() == 1
        assert set(annotation_data.columns) == set(res.columns)


@pytest.mark.parametrize(
    ["example", "length"],
    [
        (("hello", ["world"]), 2),
        ((["one", "two"], "three", ["four"]), 4),
        (("one", ["two", "three", "four"], "five"), 5),
    ],
)
def test__flatten(example, length):
    res = filtering._flatten(example)
    assert all(isinstance(elem, str) for elem in res)
    assert len(res) == length


class TestMakeFull:
    def test_transform(self, annotation_data):
        res = filtering.MakeFull(
            combination_cols=("partition_id", ["annotation_id", "chr", "start", "end"]),
            fill_value=0.0,
        ).transform(annotation_data)
        assert res.count() == 4

    def test_raises_value_error(self, annotation_data):
        component = filtering.MakeFull(
            combination_cols=("partition_id",),
            fill_value=0.0,
        )
        with pytest.raises(
            ValueError, match="`combination_cols` has to have at least two elements"
        ):
            component.transform(annotation_data)


class TestCompose:
    def test_transform(self, spark, annotation_data, precision):
        component = filtering.Compose()
        chispa.assert_approx_df_equality(
            component.transform(annotation_data), annotation_data, precision=precision
        )

        steps = (
            filtering.QuantileCutoff(
                value_col="score", threshold=0.9, partition_by=("partition_id",)
            ),
            filtering.MakeFull(
                combination_cols=("partition_id", ["annotation_id", "chr", "start", "end"]),
                fill_value=0.0,
            ),
        )
        res = filtering.Compose(*steps).transform(annotation_data)
        assert res.count() == 2
        chispa.assert_df_equality(
            res.select("annotation_id").distinct(),
            spark.createDataFrame([("z",)], "annotation_id string"),
        )

        res = filtering.Compose(*steps[::-1]).transform(annotation_data)
        assert res.count() == 2
        chispa.assert_df_equality(
            res.select("annotation_id").distinct(),
            spark.createDataFrame([("x",), ("z",)], "annotation_id string"),
        )
