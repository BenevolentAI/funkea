import chispa
import pandas as pd
import pyspark.sql.functions as F
import pytest

from funkea.components import filtering, locus_definition, normalisation
from funkea.implementations import depict


@pytest.fixture(scope="function")
def overlap_and_nearest_annotation_component(annotation_component):
    return depict.OverlapAndNearestAnnotation(annotation=annotation_component)


@pytest.fixture(scope="function")
def null_loci_data(spark):
    return spark.createDataFrame(
        [(["x"], 2), (["z"], 2), (["x", "z"], 1)], ["annotations", "n_loci"]
    )


@pytest.fixture(scope="function")
def null_loci_component(null_loci_data):
    return depict.NullLoci(dataset=null_loci_data)


@pytest.fixture(scope="function")
def pipeline_component(null_loci_component):
    return depict.Pipeline(
        locus_definition.Identity(),
        null_loci=null_loci_component,
        n_permutations=2,
    )


class TestOverlapAndNearestAnnotation:
    def test__get_nearest_annotation(self, overlap_and_nearest_annotation_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("eur"))
        res = overlap_and_nearest_annotation_component._get_nearest_annotation(
            overlap_and_nearest_annotation_component.init_locus(sumstats)
        )
        chispa.assert_column_equality(
            res.withColumn("expected_nearest_annotation", F.lit("z")),
            "expected_nearest_annotation",
            "annotation_id",
        )

    def test_raises_value_error(self, overlap_and_nearest_annotation_component, sumstats):
        overlap_and_nearest_annotation_component.annotation = None
        with pytest.raises(ValueError, match="`annotation` is None"):
            overlap_and_nearest_annotation_component.transform(sumstats)

    def test_transform(self, overlap_and_nearest_annotation_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("eur"))
        res = overlap_and_nearest_annotation_component.transform(sumstats)
        chispa.assert_column_equality(
            res.withColumn("expected_annotation", F.lit("z")),
            "expected_annotation",
            "annotation_id",
        )


class TestNullLoci:
    def test_load(self, null_loci_component):
        res = null_loci_component.load()
        assert null_loci_component.columns.size in res.columns


def test__create_permutation():
    perm = depict._create_permutation(
        pd.DataFrame(
            {
                "locus_id": ["rs1", "rs2"],
                "null_locus": [[["x"], ["z"]], [["x"], ["z"]]],
                "id": ["dummy"] * 2,
            }
        ),
        locus_id_col="locus_id",
        null_locus_col="null_locus",
        partition_cols=("id",),
    )
    assert len(perm) == 2
    assert set(perm["null_locus"].explode()) == {"x", "z"}


class TestPipeline:
    def test_get_matched_null_loci(self, spark, pipeline_component):
        loci = spark.createDataFrame(
            [
                ("rs1", ["i"]),
                ("rs2", ["j"]),
            ],
            ["locus_id", "locus_collection"],
        ).withColumn("id", F.lit("dummy"))

        res = pipeline_component.get_matched_null_loci(loci)
        assert res.select("locus_type").distinct().count() == 2
        chispa.assert_column_equality(
            res.groupBy("locus_id")
            .count()
            .join(
                spark.createDataFrame(
                    [
                        ("rs1", 2),
                        ("rs2", 2),
                    ],
                    ["locus_id", "expected_count"],
                )
            ),
            "expected_count",
            "count",
        )
        chispa.assert_column_equality(
            res.withColumn("size", F.size("locus_collection")).withColumn(
                "expected_size", F.lit(1)
            ),
            "expected_size",
            "size",
        )

    def test_transform(self, pipeline_component, annotation_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        pipeline_component.locus_definition = locus_definition.Compose(
            depict.OverlapAndNearestAnnotation(),
            locus_definition.Collect(what="annotation"),
            locus_definition.Merge(),
            annotation=annotation_component,
        )
        res = pipeline_component.transform(sumstats)
        # not sure how to best test this?
        assert res.select("partition_id").distinct().count() == 2
        # N null loci + 1 true set
        assert res.select("locus_type").distinct().count() == pipeline_component.n_permutations + 1

    def test_raises_value_error(self, pipeline_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        pipeline_component.locus_definition.annotation = None

        with pytest.raises(ValueError, match="`annotation` is None"):
            pipeline_component.transform(sumstats)

    def test_raises_runtime_error(self, pipeline_component, annotation_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("eur"))
        pipeline_component.locus_definition.annotation = annotation_component
        with pytest.raises(RuntimeError, match=".+ column is missing from the loci dataframe"):
            pipeline_component.transform(sumstats)


class TestMethod:
    def test_transform(self, spark, precision):
        method = depict.Method(partition_id_col="partition_id")
        res = method.transform(
            spark.createDataFrame(
                [
                    ("rs1", "a", "true", 1.96),
                    ("rs2", "a", "true", 1.89),
                    ("rs3", "a", "true", 2.01),
                    ("rs4", "b", "true", -0.01),
                    ("rs5", "b", "true", -0.012),
                    ("rs1", "a", "null0", 0.1),
                    ("rs2", "a", "null0", 0.24),
                    ("rs3", "a", "null0", -0.01),
                    ("rs4", "b", "null0", -0.05),
                    ("rs5", "b", "null0", 0.112),
                    ("rs1", "a", "null1", 0.90),
                    ("rs2", "a", "null1", 0.17),
                    ("rs3", "a", "null1", -0.3),
                    ("rs4", "b", "null1", -0.051),
                    ("rs5", "b", "null1", 0.04),
                ],
                ["locus_id", "partition_id", "locus_type", "score"],
            ).withColumn("id", F.lit("dummy"))
        )
        assert set(res.columns) == {"id", "partition_id", "enrichment", "p_value"}
        res = res.join(
            spark.createDataFrame(
                [
                    ("a", 99.041335999017, 0.0),
                    ("b", -31.258234358422122, 1.0),
                ],
                ["partition_id", "expected_enrichment", "expected_p_value"],
            ),
            on="partition_id",
            how="inner",
        )
        chispa.assert_approx_column_equality(
            res, "expected_enrichment", "enrichment", precision=precision
        )
        chispa.assert_approx_column_equality(
            res, "expected_p_value", "p_value", precision=precision
        )


class TestDEPICT:
    def test_default(self):
        depict.DEPICT.default()

    def test_issues_user_warning_0(self, annotation_component):
        annotation_component.filter_operation = filtering.MakeFull(
            ("partition_id", ["annotation_id", "start", "end", "chr"]), fill_value=0.0
        )
        with pytest.warns(
            UserWarning, match="DEPICT assumes all annotation-partition values to be z-scored"
        ):
            depict.DEPICT.default(annotation=annotation_component)

    def test_issues_user_warning_1(self, annotation_component):
        annotation_component.normalisation = normalisation.StandardScaler(
            "score", "score", partition_by=("annotation_id",)
        )
        with pytest.warns(
            UserWarning, match="The annotation component specified has hard partitions"
        ):
            depict.DEPICT.default(annotation=annotation_component)
