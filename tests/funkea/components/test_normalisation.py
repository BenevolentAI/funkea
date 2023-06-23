import chispa

from funkea.components import normalisation


class TestNormalisation:
    def test_init(self, mock_abstract_methods):
        mock_abstract_methods(normalisation.Normalisation)
        normalisation.Normalisation(values_col="", output_col="")


class TestIdentity:
    def test_transform(self, annotation_data, precision):
        chispa.assert_approx_df_equality(
            normalisation.Identity().transform(annotation_data),
            annotation_data,
            precision=precision,
        )


class TestStandardScaler:
    def test_transform(self, spark, annotation_data, precision):
        component = normalisation.StandardScaler(
            values_col="score", output_col="score", partition_by=("annotation_id",)
        )
        res = component.transform(annotation_data)
        chispa.assert_approx_column_equality(
            res.join(
                spark.createDataFrame(
                    [
                        ("a", "x", 0.0),
                        ("a", "z", 0.70710678),
                        ("b", "z", -0.70710678),
                    ],
                    "partition_id string, annotation_id string, expected_score double",
                ),
                on=["partition_id", "annotation_id"],
            ),
            "expected_score",
            "score",
            precision=precision,
        )


class TestQuantileNorm:
    def test_transform(self, spark, annotation_data, precision):
        component = normalisation.QuantileNorm(
            values_col="score", output_col="score", partition_by=("partition_id",)
        )
        chispa.assert_approx_column_equality(
            component.transform(annotation_data).join(
                spark.createDataFrame(
                    [("a", "x", -0.165), ("a", "z", 0.96), ("b", "z", -0.165)],
                    "partition_id string, annotation_id string, expected_score double",
                ),
                on=["partition_id", "annotation_id"],
                how="inner",
            ),
            "expected_score",
            "score",
            precision=precision,
        )


class TestEuclidNorm:
    def test_transform(self, spark, annotation_data, precision):
        component = normalisation.EuclidNorm(
            values_col="score", output_col="score", partition_by=("annotation_id",)
        )
        chispa.assert_approx_column_equality(
            component.transform(annotation_data).join(
                spark.createDataFrame(
                    [("a", "x", 1.0), ("a", "z", 0.90545893595886), ("b", "z", -0.42443387623071)],
                    "partition_id string, annotation_id string, expected_score double",
                ),
                on=["partition_id", "annotation_id"],
                how="inner",
            ),
            "expected_score",
            "score",
            precision=precision,
        )


class TestCompose:
    def test_transform(self, spark, annotation_data, precision):
        col = "score"
        component = normalisation.Compose(
            normalisation.QuantileNorm(
                values_col=col, output_col=col, partition_by=("partition_id",)
            ),
            normalisation.EuclidNorm(
                values_col=col, output_col=col, partition_by=("annotation_id",)
            ),
        )
        res = component.transform(annotation_data)
        assert set(res.columns) == set(annotation_data.columns)

        chispa.assert_approx_column_equality(
            res.join(
                spark.createDataFrame(
                    [
                        ("a", "x", -1.0),
                        ("a", "z", 0.9855488907597534),
                        ("b", "z", -0.16939121559933262),
                    ],
                    "partition_id string, annotation_id string, expected_score double",
                ),
                on=["partition_id", "annotation_id"],
                how="inner",
            ),
            "expected_score",
            "score",
            precision=precision,
        )
