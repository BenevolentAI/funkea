import chispa
import numpy as np
import pyspark.sql.functions as F
import pytest
from scipy import stats

from funkea.components import locus_definition
from funkea.implementations import fisher


@pytest.fixture(scope="function")
def pipeline_component(annotation_component):
    return fisher.Pipeline(locus_definition.Overlap(annotation_component))


class TestPipeline:
    def test_transform(self, pipeline_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        res = pipeline_component.transform(sumstats)
        assert res.count() == 2

    def test_raises_value_error(self, pipeline_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        pipeline_component.locus_definition.annotation = None
        with pytest.raises(ValueError, match="`annotation` is None."):
            pipeline_component.transform(sumstats)


@pytest.mark.parametrize("ix", range(5))
def test_hyper_geom_test(spark, precision, ix):
    table = np.random.randint(0, 101, size=(2, 2), dtype=int)
    p_value = stats.fisher_exact(table, alternative="greater").pvalue

    res = fisher.hyper_geom_test(
        spark.createDataFrame(
            [tuple(table.ravel().tolist())], "a int, b int, c int, d int"
        ).withColumn("entry_id", F.lit("entry1"))
    ).withColumn("expected_p_value", F.lit(float(p_value)))
    chispa.assert_approx_column_equality(res, "expected_p_value", "p_value", precision=precision)


class TestMethod:
    def test_transform(self, spark, precision):
        res = fisher.Method().transform(
            spark.createDataFrame(
                [("entry1", ["x"], ["x", "z"], 20, 0.1), ("entry2", ["x"], ["z"], 20, 1.0)],
                "entry_id string, query array<string>, library array<string>, "
                "space_size int, expected_p_value double",
            )
        )
        chispa.assert_approx_column_equality(
            res, "expected_p_value", "p_value", precision=precision
        )


class TestFisher:
    def test_default(self):
        fisher.Fisher.default()
