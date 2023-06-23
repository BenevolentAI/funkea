from typing import Callable

import chispa
import pyspark.sql.functions as F
import pytest
from pyspark.sql import DataFrame

from funkea.components import locus_definition
from funkea.core import data
from funkea.implementations import snpsea


@pytest.fixture(scope="function")
def snpsea_pipeline(
    ancestry_sumstats,
) -> Callable[[data.AnnotationComponent | None], snpsea.Pipeline]:
    def pipeline(annotation_component: data.AnnotationComponent | None) -> snpsea.Pipeline:
        return snpsea.Pipeline(
            locus_definition.Compose(
                locus_definition.Overlap(),
                locus_definition.Collect(),
                locus_definition.Merge(),
                annotation=annotation_component,
            ),
            background_variants=snpsea.BackgroundVariants(dataset=ancestry_sumstats),
            n_permutations=2,
        )

    return pipeline


@pytest.fixture(scope="function")
def locus_dataframe(spark) -> DataFrame:
    return spark.createDataFrame(
        [
            ('z', 'dummy', 'rs117898110;rs34673795;rs72844114;rs73728002', 'true', 'a', 2, 0.5, 2),
            ('z', 'dummy', 'rs117898110;rs34673795;rs72844114;rs73728002', 'true', 'b', 1, 1.0, 2),
            ('z', 'dummy', 'rs117898110;rs34673795;rs72844114;rs73728002', 'null0', 'a', 2, 0.5, 2),
            ('z', 'dummy', 'rs117898110;rs34673795;rs72844114;rs73728002', 'null0', 'b', 1, 1.0, 2),
            ('z', 'dummy', 'rs117898110;rs34673795;rs72844114;rs73728002', 'null1', 'a', 2, 0.5, 2),
            ('z', 'dummy', 'rs117898110;rs34673795;rs72844114;rs73728002', 'null1', 'b', 1, 1.0, 2),
        ],
        [
            "annotation_id",
            "id",
            "locus_id",
            "locus_type",
            "partition_id",
            "discrete_score",
            "continuous_score",
            "space_size",
        ],
    )


class TestCollectExpandedIfEmpty:
    def test_transform(self, annotation_component, sumstats):
        extension = (100_000, 100_000)
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        res = locus_definition.Compose(
            locus_definition.Expand(extension=extension),
            locus_definition.Overlap(),
            snpsea.CollectExpandedIfEmpty(extension),
            annotation=annotation_component,
        ).transform(
            sumstats.unionByName(
                sumstats.withColumn("pos", F.col("pos") - 200_000).withColumn(
                    "rsid", F.concat(F.col("rsid"), F.lit("__x"))
                )
            )
        )
        chispa.assert_column_equality(
            res.filter(F.col("locus_id").endswith("__x")).withColumn(
                "expected_locus_collection", F.array(F.lit("x"))
            ),
            "expected_locus_collection",
            "locus_collection",
        )

    def test_raises_value_error(self, sumstats):
        with pytest.raises(ValueError, match="`annotation` is None"):
            snpsea.CollectExpandedIfEmpty((10, 10)).transform(sumstats)

    def test_raises_runtime_error(self, annotation_component, sumstats):
        component = snpsea.CollectExpandedIfEmpty((10, 10), annotation_component)
        with pytest.raises(
            RuntimeError, match="Either .+ or .+ is missing from the input dataframe"
        ):
            component.transform(sumstats)


class TestPercentileAssignment:
    def test_transform(self, spark, annotation_data, precision):
        res = snpsea.PercentileAssignment(
            values_col="score", output_col="score", partition_by=("partition_id",)
        ).transform(annotation_data)
        chispa.assert_approx_column_equality(
            res.join(
                spark.createDataFrame(
                    [
                        ("a", "x", 1.0),
                        ("a", "z", 0.5),
                        ("b", "z", 1.0),
                    ],
                    ["partition_id", "annotation_id", "expected_score"],
                ),
                on=["partition_id", "annotation_id"],
                how="inner",
            ),
            "expected_score",
            "score",
            precision=precision,
        )


class TestBackgroundVariants:
    def test_load(self, ancestry_sumstats):
        assert "id" not in ancestry_sumstats.columns
        res = snpsea.BackgroundVariants(dataset=ancestry_sumstats).load()
        assert "id" in res.columns
        chispa.assert_column_equality(
            res.withColumn("expected_id", F.lit("dummy")), "expected_id", "id"
        )


class TestPipeline:
    def test_get_matched_null_loci(self, spark, snpsea_pipeline, annotation_component):
        pipe = snpsea_pipeline(annotation_component)
        loci = spark.createDataFrame([("rs1", ["x"])], ["locus_id", "locus_collection"]).withColumn(
            "id", F.lit("dummy")
        )

        res = pipe.get_matched_null_loci(loci)
        chispa.assert_column_equality(
            res.withColumn("expected_size", F.lit(1)).withColumn(
                "size", F.size("locus_collection")
            ),
            "expected_size",
            "size",
        )
        assert res.select("locus_type").distinct().count() == pipe.n_permutations
        assert res.count() == loci.count() * pipe.n_permutations

    def test_transform(self, spark, snpsea_pipeline, annotation_component, ancestry_sumstats):
        ancestry_sumstats = ancestry_sumstats.withColumn("id", F.lit("dummy"))
        pipe = snpsea_pipeline(annotation_component)
        res = pipe.transform(ancestry_sumstats)
        assert set(res.columns) == {
            "id",
            "locus_id",
            "locus_type",
            "annotation_id",
            "partition_id",
            "score",
            "space_size",
        }
        chispa.assert_column_equality(
            res.join(
                spark.createDataFrame(
                    [
                        ("a", 2),
                        ("b", 1),
                    ],
                    ["partition_id", "expected_score"],
                ),
                on="partition_id",
                how="inner",
            ),
            "expected_score",
            "score",
        )
        chispa.assert_column_equality(
            res.withColumn("expected_space_size", F.lit(2)), "expected_space_size", "space_size"
        )
        assert res.select("locus_type").distinct().count() == pipe.n_permutations + 1

    def test_transform_raises_value_error(self, snpsea_pipeline, ancestry_sumstats):
        pipe = snpsea_pipeline(None)
        with pytest.raises(ValueError, match="`annotation` is None"):
            pipe.transform(ancestry_sumstats)

    def test_transform_raises_runtime_error(
        self, snpsea_pipeline, annotation_component, ancestry_sumstats
    ):
        pipe = snpsea_pipeline(None)
        pipe.locus_definition = locus_definition.Identity(annotation_component)
        with pytest.raises(RuntimeError, match=".+ not in the loci dataframe"):
            pipe.transform(ancestry_sumstats)


class TestSpecificityScore:
    def test_init(self, mock_abstract_methods):
        mock_abstract_methods(snpsea.SpecificityScore)
        comp = snpsea.SpecificityScore(
            locus_id_col="c1", partition_id_col="c2", annotation_id_col="c3", values_col="c4"
        )
        assert comp.locus_id_col == "c1"
        assert comp.partition_id_col == "c2"
        assert comp.annotation_id_col == "c3"
        assert comp.values_col == "c4"

    @pytest.mark.parametrize(
        ["subclass", "expected_data", "score_col"],
        [
            (snpsea.ContinuousSpecificity, [("a", 0.5), ("b", 1.0)], "continuous_score"),
            (snpsea.BinarySpecificity, [("a", 1.0), ("b", 1.0)], "discrete_score"),
        ],
    )
    def test_subclass(self, spark, locus_dataframe, precision, subclass, expected_data, score_col):
        component = subclass(
            locus_id_col="locus_id",
            partition_id_col="partition_id",
            annotation_id_col="annotation_id",
            values_col=score_col,
        )
        with component.partition_by(*component.partition_cols, "locus_type"):
            res = component.transform(locus_dataframe)
        assert set(res.columns) == {"id", "locus_type", "id", "partition_id", "locus_id", "score"}
        chispa.assert_approx_column_equality(
            res.join(
                spark.createDataFrame(expected_data, ["partition_id", "expected_score"]),
                on="partition_id",
            ),
            "expected_score",
            "score",
            precision=precision,
        )
        assert res.count() == locus_dataframe.count()


class TestMethod:
    @pytest.mark.parametrize(
        ["partition_type", "expected_data", "score_col"],
        [
            (data.PartitionType.HARD, [("a", 0.0, 1.0), ("b", 0.0, 1.0)], "discrete_score"),
            (
                data.PartitionType.SOFT,
                [("a", 0.6931471805599453, 1.0), ("b", 0.0, 1.0)],
                "continuous_score",
            ),
        ],
    )
    def test_transform(
        self, spark, locus_dataframe, precision, partition_type, expected_data, score_col
    ):
        method = snpsea.Method(
            partition_id_col="partition_id",
            annotation_id_col="annotation_id",
            partition_type=partition_type,
            score_col=score_col,
            n_permutations=2,
        )
        res = method.transform(locus_dataframe)
        assert set(res.columns) == {"id", "partition_id", "enrichment", "p_value"}
        assert res.count() == 2
        res = res.join(
            spark.createDataFrame(
                expected_data, ["partition_id", "expected_enrichment", "expected_p_value"]
            ),
            on="partition_id",
        )
        chispa.assert_approx_column_equality(
            res, "expected_enrichment", "enrichment", precision=precision
        )
        chispa.assert_approx_column_equality(
            res, "expected_p_value", "p_value", precision=precision
        )


class TestSNPsea:
    def test_default(self):
        snpsea.SNPsea.default()

    def test_default_raises_value_error(self, annotation_component):
        annotation_component.partition_type = data.PartitionType.SOFT
        annotation_component.columns.values = None
        with pytest.raises(
            ValueError, match="Using soft-partitioned annotation data requires a `values` column"
        ):
            snpsea.SNPsea.default(annotation_component)
