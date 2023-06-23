import functools

import chispa
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pytest
from pyspark.sql import DataFrame, Window
from scipy import stats

from funkea.components import locus_definition
from funkea.implementations import ldsc


@pytest.fixture(scope="function")
def ld_score_data(spark):
    return spark.createDataFrame(
        [
            ("rs117898110", "a", 1.09, 100, "eur"),
            ("rs117898110", "b", 10.09, 370, "eur"),
            ("rs34673795", "a", 0.02, 100, "eur"),
            ("rs34673795", "b", 5.9, 370, "eur"),
            ("rs73728002", "a", -0.003, 100, "eur"),
            ("rs73728002", "b", 6.7, 370, "eur"),
            ("rs72844114", "a", 1.2, 100, "eur"),
            ("rs72844114", "b", 8.9, 370, "eur"),
        ],
        ["rsid", "partition_id", "ld_score", "n_variants", "ancestry"],
    )


@pytest.fixture(scope="function")
def ld_score_component(ld_score_data):
    return ldsc.LDScores(dataset=ld_score_data)


@pytest.fixture(scope="function")
def ldsc_pipeline(annotation_component, ld_score_data):
    return ldsc.Pipeline(
        locus_definition.Overlap(annotation=annotation_component),
        precomputed_ld_scores=ldsc.LDScores(
            dataset=ld_score_data.unionByName(
                ld_score_data.withColumn("partition_id", F.lit("control")).dropDuplicates(
                    ["rsid", "partition_id"]
                )
            )
        ),
        weighting_ld_scores=ldsc.LDScores(
            dataset=ld_score_data.withColumn("partition_id", F.lit("weight")).dropDuplicates(
                ["rsid", "partition_id"]
            )
        ),
        controlling_ld_scores=ldsc.LDScores(dataset=ld_score_data),
    )


@pytest.fixture()
def ldsc_input():
    return pd.DataFrame(
        np.random.randn(1000, 3) ** 2, columns=["ld_score", "weight", "chi2"]
    ).assign(
        partition_id="a",
        chr=6,
        pos=list(range(1000)),
        n_variants=1000,
        sample_size=1000,
        id="dummy",
    )


class TestLDScores:
    def test_load_pivot_table(self, ld_score_component):
        res = ld_score_component.load_pivot_table("ld_score")
        assert set(res.columns) == {"rsid", "ancestry", "a", "b"}
        assert res.count() == 4
        assert (
            res.schema["a"].dataType == T.DoubleType()
            and res.schema["b"].dataType == T.DoubleType()
        )

        res = ld_score_component.load_pivot_table("n_variants")
        assert set(res.columns) == {"rsid", "ancestry", "a", "b"}
        assert res.count() == 4
        assert res.schema["a"].dataType == T.LongType() and res.schema["b"].dataType == T.LongType()

    def test_get_partitions(self, ld_score_component):
        parts = ld_score_component.get_partitions()
        assert sorted(parts) == ["a", "b"]


@pytest.fixture(scope="function")
def beta_se_sumstats(spark):
    def example(beta: float, se: float) -> tuple[float, float, float, float]:
        p = 2 * float(stats.norm.cdf(-abs(beta / se)))
        return (beta, se, p, float(stats.chi2.isf(p, 1)))

    return spark.createDataFrame(
        [
            example(1.2, 0.9),
            example(-0.3, 0.2),
            example(-1.1, 0.54),
            example(0.4, 0.1),
        ],
        ["beta", "se", "p", "expected_chi2"],
    )


def test_compute_chi2(beta_se_sumstats, precision):
    res = ldsc.compute_chi2(beta_se_sumstats)
    chispa.assert_approx_column_equality(res, "expected_chi2", "chi2", precision=precision)
    chispa.assert_approx_column_equality(
        ldsc.compute_chi2(beta_se_sumstats.drop("beta", "se")),
        "expected_chi2",
        "chi2",
        precision=precision,
    )


def test_compute_chi2_raises_value_error(beta_se_sumstats):
    with pytest.raises(ValueError, match=".+ or .+ are missing from the sumstats"):
        ldsc.compute_chi2(beta_se_sumstats.drop("beta", "se", "p"))


def test_extract_control_ld(ld_score_component):
    ld = ld_score_component.load()
    res = ldsc.extract_control_ld(ld, ld_score_component.columns, "b")
    assert set(res.columns) == {
        "rsid",
        "partition_id",
        "ld_score",
        "n_variants",
        "ancestry",
        "b",
        "b_n_variants",
    }
    assert res.count() == ld.count() // 2


class TestPipeline:
    def test_get_annotation(self, annotation_component):
        pipe = ldsc.Pipeline(locus_definition.Identity(annotation_component))
        assert pipe.get_annotation() is annotation_component

    def test_get_annotation_raises_value_error(self):
        with pytest.raises(ValueError, match="`annotation` is None."):
            ldsc.Pipeline(locus_definition.Identity()).get_annotation()

    def test_get_ld_component(self, annotation_component, ld_component):
        ld = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component), ld_component=ld_component
        ).get_ld_component()
        assert ld is ld_component

    def test_get_ld_component_raises_value_error(self, annotation_component):
        with pytest.raises(ValueError, match="`ld_component` is None"):
            ldsc.Pipeline(
                locus_definition.Overlap(annotation=annotation_component),
            ).get_ld_component()

    def test_annotate_variants(self, annotation_component, sumstats):
        pipe = ldsc.Pipeline(locus_definition.Overlap(annotation=annotation_component))
        res = pipe.annotate_variants(sumstats)
        assert res.count() == 8
        assert set(res.columns) == {"rsid", "partition_id"}

    def test_annotate_variants_raises_runtime_error(self, annotation_component, sumstats):
        with pytest.raises(RuntimeError, match=".+ is missing from the loci dataframe"):
            ldsc.Pipeline(
                locus_definition.Identity(annotation=annotation_component)
            ).annotate_variants(sumstats)

    def test_load_annotated_reference_variants(self, annotation_component, ld_component):
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component), ld_component=ld_component
        )
        res = pipe.load_annotated_reference_variants()
        assert res.count() == 8
        assert set(res.columns) == {"rsid", "partition_id"}

    def test_load_annotated_reference_variants_combinations(
        self, annotation_component, ld_component
    ):
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component),
            ld_component=ld_component,
        )
        res = pipe.load_annotation_reference_variants_combinations(
            pipe.load_annotated_reference_variants()
        )
        assert res.count() == 8

    def test__compute_ld_scores(self, spark, annotation_component, ld_component, precision):
        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component),
            ld_component=ld_component,
            n_partitions_ld_computation=2,
        )
        res = pipe._compute_ld_scores(pipe.load_annotated_reference_variants())
        assert set(res.columns) == {"rsid", "partition_id", "ancestry", "ld_score"}
        assert res.count() == 8
        chispa.assert_approx_column_equality(
            res.join(
                spark.createDataFrame(
                    [
                        ("rs117898110", "a", 0.7266024),
                        ("rs117898110", "b", 0.7266024),
                        ("rs34673795", "a", 5.13697947e-01),
                        ("rs34673795", "b", 5.13697947e-01),
                        ("rs73728002", "a", -2.9370422e-05),
                        ("rs73728002", "b", -2.9370422e-05),
                        ("rs72844114", "a", 2.1293384e-01),
                        (
                            "rs72844114",
                            "b",
                            2.1293384e-01,
                        ),
                    ],
                    ["rsid", "partition_id", "expected_ld_score"],
                ),
                on=["rsid", "partition_id"],
                how="inner",
            ),
            "expected_ld_score",
            "ld_score",
            precision=precision,
        )

    def test__compute_n_variants(self, annotation_component, ld_component):
        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component),
            ld_component=ld_component,
            n_partitions_ld_computation=2,
        )
        res = pipe._compute_n_variants(pipe.load_annotated_reference_variants())
        assert set(res.columns) == {"partition_id", "n_variants"}
        assert res.count() == 2
        assert res.filter(F.col("n_variants") == 4).count() == 2

    def test_compute_ld_scores_and_n_variants(self, annotation_component, ld_component):
        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component), ld_component=ld_component
        )

        res = pipe.compute_ld_scores_and_n_variants(pipe.load_annotated_reference_variants())
        assert set(res.columns) == {"rsid", "partition_id", "ancestry", "ld_score", "n_variants"}
        assert res.count() == 8

    def test_compute_ld_scores_and_n_variants_issues_user_warning(
        self, annotation_component, ld_component
    ):
        def _test_warning(self_ld=False, unbiased=True, threshold=None, msg=""):
            ld_component.return_self_ld = self_ld
            ld_component.return_unbiased_r2 = unbiased
            ld_component.r2_threshold = threshold
            pipe = ldsc.Pipeline(
                locus_definition.Overlap(annotation=annotation_component),
                ld_component=ld_component,
                n_partitions_ld_computation=2,
            )
            with pytest.warns(UserWarning, match=msg):
                pipe.compute_ld_scores_and_n_variants(pipe.load_annotated_reference_variants())

        _test_warning(threshold=0.2, msg="`ld_component.r2_threshold` is not None")
        _test_warning(unbiased=False, msg="`ld_component.return_unbiased_r2` is set to False")
        _test_warning(self_ld=True, msg="`ld_component.return_self_ld` is set to True")

    def test_compute_control_ld_scores_and_n_variants(self, annotation_component, ld_component):
        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component),
            ld_component=ld_component,
            n_partitions_ld_computation=2,
        )
        res = pipe.compute_control_ld_scores_and_n_variants(
            pipe.load_annotated_reference_variants()
        )
        assert res.count() == res.filter(F.col("partition_id") == "control").count()
        assert res.count() == 4

    def test_compute_partitioned_ld_scores_and_control(self, annotation_component, ld_component):
        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component),
            ld_component=ld_component,
            n_partitions_ld_computation=2,
        )
        res = pipe.compute_partitioned_ld_scores_and_control()
        assert res.select("partition_id").distinct().count() == 3
        assert res.groupBy("partition_id").count().filter(F.col("count") == 4).count() == 3

    def test_compute_partitioned_ld_scores_and_control_batched(
        self, annotation_component, ld_component
    ):
        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        pipe = ldsc.Pipeline(
            locus_definition.Overlap(annotation=annotation_component),
            ld_component=ld_component,
            n_partitions_ld_computation=2,
        )
        res = pipe.compute_partitioned_ld_scores_and_control_batched(
            batch_size=1,
        )
        assert len(res) == 3
        assert functools.reduce(DataFrame.unionByName, res).count() == 12

    def test_transform(self, ldsc_pipeline, ancestry_sumstats):
        ancestry_sumstats = ancestry_sumstats.withColumn("sample_size", F.lit(1000))

        def _test_transform(control_set):
            res1 = ldsc_pipeline.transform(ancestry_sumstats)
            assert "weight" in res1.columns
            assert "control" in res1.columns
            assert {"a", "b"}.intersection(res1.columns) == control_set
            assert res1.count() == 8

        _test_transform({"a", "b"})

        ldsc_pipeline.controlling_ld_scores = None
        _test_transform(set())

    def test_transform_raises_value_error(self, ancestry_sumstats):
        with pytest.raises(ValueError, match=".+ column not in input dataframe"):
            ldsc.Pipeline(locus_definition.Identity()).transform(ancestry_sumstats)

    def test_transform_issues_user_warning(self, ldsc_pipeline, ld_component, ancestry_sumstats):
        ancestry_sumstats = ancestry_sumstats.withColumn("sample_size", F.lit(1000))
        ldsc_pipeline.precomputed_ld_scores = None

        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        ldsc_pipeline.ld_component = ld_component
        with pytest.warns(UserWarning, match="No precomputed LD scores set"):
            ldsc_pipeline.transform(ancestry_sumstats)


def test_fit_ldsc_model(ldsc_input):
    res = ldsc.fit_ldsc_model(ldsc_input, ldsc._LDSCColumns())
    assert set(res.columns) == {"id", "partition_id", "enrichment", "p_value"}
    assert not res[["enrichment", "p_value"]].isna().any(axis=None)


def test_fit_ldsc_model_linalg_error(ldsc_input):
    res = ldsc.fit_ldsc_model(
        ldsc_input.assign(
            ld_score=ldsc_input["ld_score"] * 0,
            c1=np.random.randn(1000) ** 2,
            c1_n_variants=1000,
        ),
        ldsc._LDSCColumns(controlling_covariates=("c1",)),
    )
    assert set(res.columns) == {"id", "partition_id", "enrichment", "p_value"}
    assert res[["enrichment", "p_value"]].isna().all(axis=None)


class TestMethod:
    def test_transform(self, ld_score_data):
        method = ldsc.Method(
            partition_id_col="partition_id",
            control_ld=(),
        )
        res = method.transform(
            ld_score_data.withColumn("chr", F.lit(6))
            .withColumn("pos", F.row_number().over(Window.orderBy("ld_score")))
            .withColumn("chi2", F.randn() ** 2)
            .withColumn("sample_size", F.lit(1000))
            .withColumn("weight", F.randn() ** 2)
            .withColumn("id", F.lit("dummy"))
        )
        assert set(res.columns) == {"id", "partition_id", "enrichment", "p_value"}
        assert res.count() == 2


class TestLDSC:
    def test_default(self, ld_score_component):
        ldsc.LDSC.default(params=ldsc.Params(controlling_ld_scores=ld_score_component))

    def test_compute_partitioned_ld_scores(
        self, ld_score_component, ld_component, annotation_component
    ):
        model = ldsc.LDSC.default(params=ldsc.Params(controlling_ld_scores=ld_score_component))
        ld_component.return_self_ld = False
        ld_component.return_unbiased_r2 = True
        ld_component.r2_threshold = None
        model.pipeline.ld_component = ld_component
        model.pipeline.locus_definition.annotation = annotation_component
        model.pipeline.locus_definition.propagate_annotations()

        assert isinstance(model.compute_partitioned_ld_scores(), DataFrame)
        assert isinstance(model.compute_partitioned_ld_scores(batch_size=1), list)

    def test_with_precomputed_ld_scores(self, ld_score_data, ld_score_component):
        model = ldsc.LDSC.default(params=ldsc.Params(controlling_ld_scores=ld_score_component))
        assert model.pipeline.precomputed_ld_scores is None
        model.with_precomputed_ld_scores(ld_score_data)
        assert isinstance(model.pipeline.precomputed_ld_scores, ldsc.LDScores)

    def test_with_controlling_ld_scores(self, ld_score_data, ld_score_component):
        model = ldsc.LDSC.default(params=ldsc.Params(controlling_ld_scores=ld_score_component))
        model.with_controlling_ld_scores(ld_score_data)
        assert model.pipeline.controlling_ld_scores is not ld_score_component
