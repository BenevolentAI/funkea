import random
import string

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pytest

from funkea.components import locus_definition
from funkea.implementations import garfield


@pytest.fixture(scope="function")
def controlling_covariates_data(spark):
    return spark.createDataFrame(
        [
            ("rs117898110", 1.0, 2.0),
            ("rs34673795", 3.0, 1.0),
            ("rs73728002", 4.0, 3.0),
            ("rs72844114", 1.0, 1.0),
        ],
        ["rsid", "binned_tss_distance", "binned_n_proxies"],
    )


@pytest.fixture(scope="function")
def controlling_covariates_component(controlling_covariates_data):
    return garfield.ControllingCovariates(dataset=controlling_covariates_data)


class TestWithinSpan:
    def test_transform(self, spark, annotation_component):
        component = garfield.WithinSpan((10, 10), annotation_component)
        res = component.transform(
            spark.createDataFrame(
                [
                    ("rs1", 100, 110, 120, 0.0),
                    ("rs2", 300, 250, 295, 0.0),
                    ("rs3", 400, 450, 470, 0.0),
                ],
                ["rsid", "pos", "start", "end", "p"],
            )
        )
        assert res.count() == 2
        assert set(res.toPandas()["rsid"]) == {"rs1", "rs2"}

    def test_raises_value_error(self, sumstats):
        component = garfield.WithinSpan((10, 10))
        with pytest.raises(ValueError, match="`annotation` is None"):
            component.transform(sumstats)

    def test_raises_runtime_error(self, annotation_component, sumstats):
        component = garfield.WithinSpan((10, 10), annotation=annotation_component)
        with pytest.raises(
            RuntimeError, match="Annotation coordinate columns are missing from the input dataframe"
        ):
            component.transform(sumstats)


class TestControllingCovariates:
    def test_load(self, controlling_covariates_component):
        controlling_covariates_component.load()


class TestPipeline:
    def test_transform(self, annotation_component, controlling_covariates_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        pipe = garfield.Pipeline(
            locus_definition.Compose(
                locus_definition.Overlap(keep_variants=True),
                locus_definition.Collect(),
                annotation=annotation_component,
            ),
            controlling_covariates=None,
        )
        res1 = pipe.transform(sumstats)
        base_cols = {"id", "locus_id", "partition_id", "p"}
        assert set(res1.columns) == base_cols

        pipe.controlling_covariates = controlling_covariates_component
        res2 = pipe.transform(sumstats)
        assert set(res2.columns) == base_cols.union({"binned_tss_distance", "binned_n_proxies"})

    def test_raises_value_error(self, sumstats):
        with pytest.raises(ValueError, match="`annotation` is None."):
            garfield.Pipeline(locus_definition.Identity()).transform(sumstats)

    def test_raises_runtime_error(self, annotation_component, sumstats):
        with pytest.raises(RuntimeError, match=".+ column is missing from the loci dataframe"):
            garfield.Pipeline(locus_definition.Identity(annotation=annotation_component)).transform(
                sumstats
            )


def test_create_design_matrix():
    y, X = garfield.create_design_matrix(
        pd.DataFrame(
            [
                (1, 0.9, 0),
                (2, 1.2, 0),
                (2, 0.22, 1),
            ],
            columns=["cat", "cont", "tgt"],
        ),
        categorical_covariates=["cat"],
        continuous_covariates=["cont"],
        target="tgt",
    )
    pd.testing.assert_frame_equal(pd.DataFrame({"tgt": [0.0, 0.0, 1.0]}), y)
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            [
                (1.0, 0.0, 0.90),
                (1.0, 1.0, 1.2),
                (1.0, 1.0, 0.22),
            ],
            columns=["Intercept", "C(Q('cat'))[T.2]", "cont"],
        ),
        X,
    )


def test_fit_garfield_model():
    res = garfield.fit_garfield_model(
        pd.DataFrame(
            np.random.randint([0, 0, 0, 0], [5, 15, 2, 2], size=(1000, 4)),
            columns=["c1", "c2", "a", "t"],
        ).assign(id="dummy"),
        annotation_col="a",
        categorical_covariates=["c1", "c2"],
        continuous_covariates=[],
        target_col="t",
        partition_cols=("id",),
    )
    assert set(res.columns) == {"id", "enrichment", "p_value"}
    assert (~res[["enrichment", "p_value"]].isna()).all().all()


def test_fit_garfield_model_perfect_separation():
    res = garfield.fit_garfield_model(
        pd.DataFrame(
            [
                (4, 3, 0, 1),
                (1, 2, 0, 0),
                (3, 2, 0, 0),
                (1, 5, 1, 1),
                (2, 3, 0, 1),
                (3, 4, 1, 1),
            ],
            columns=["c1", "c2", "a", "t"],
        ).assign(id="dummy"),
        annotation_col="a",
        categorical_covariates=["c1", "c2"],
        continuous_covariates=[],
        target_col="t",
        partition_cols=("id",),
    )
    assert res[["enrichment", "p_value"]].isna().all().all()


class TestMethod:
    def test_transform(self, spark):
        method = garfield.Method(
            partition_id_col="partition_id",
            continuous_covariates=(),
            categorical_covariates=("c1", "c2"),
            association_threshold=0.05,
        )
        annotations = list(string.ascii_lowercase)
        data = (
            pd.DataFrame(
                np.random.randint([0, 0, 0], [5, 15, 11], size=(1000, 3)), columns=["c1", "c2", "k"]
            )
            .assign(
                partition_id=lambda df: df.apply(
                    lambda row: random.sample(annotations, row["k"]), axis=1
                ),
                p=np.random.RandomState(42).rand(1000, 1).ravel(),
            )
            .drop(columns="k")
        )
        res = method.transform(spark.createDataFrame(data).withColumn("id", F.lit("dummy")))
        assert set(res.columns) == {"id", "partition_id", "enrichment", "p_value"}
        assert res.count() == data["partition_id"].explode().nunique()


class TestGARFIELD:
    def test_default(self):
        garfield.GARFIELD.default()
