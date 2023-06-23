import chispa
import pyspark.sql.functions as F
import pytest

from funkea.components import filtering, normalisation
from funkea.core import data


@pytest.fixture(scope="function")
def mock_read_parquet(monkeypatch):
    def _mock(df):
        class _Mock:
            @staticmethod
            def parquet(*args, **kwargs):
                return df

        monkeypatch.setattr(
            "pyspark.sql.SparkSession.read",
            _Mock(),
        )

    return _mock


class TestDataComponent:
    def test_load(self, spark, mock_read_parquet):
        component = data.DataComponent()
        with pytest.raises(ValueError, match="`dataset` attribute is None"):
            component.load()

        df = spark.createDataFrame([(1, 2)], "a string, b string")
        component = data.DataComponent(df)

        chispa.assert_df_equality(component.load(), df)

        mock_read_parquet(df)
        component = data.DataComponent("some/path.parquet")
        chispa.assert_df_equality(component.load(), df)


@pytest.mark.parametrize(
    "example",
    [
        [
            (0.1, 503, 0.09820359),
            (0.25, 503, 0.24850299),
            (1.0, 503, 1.0),
        ],
        [
            (0.1, 2, -0.35),
            (0.25, 2, -0.125),
            (1.0, 2, 1.0),
        ],
    ],
)
def test_unbiased_r2_estimate(spark, example, precision):
    df = spark.createDataFrame(example, "r2 double, sample_size int, expected double").withColumn(
        "observed", data.unbiased_r2_estimate(F.col("r2"), F.col("sample_size"))
    )
    chispa.assert_approx_column_equality(df, "expected", "observed", precision=precision)


class TestLDComponent:
    def test_load(self, spark, ld_reference_data, precision):
        component = data.LDComponent(dataset=ld_reference_data)

        filtered = ld_reference_data.filter(F.col("correlation") > 0.2)
        chispa.assert_approx_df_equality(component.load(), filtered, precision=precision)
        component = data.LDComponent(return_self_ld=True, dataset=ld_reference_data)

        chispa.assert_df_equality(
            component.load().agg(F.sum((F.col("correlation") == 1).cast("byte")).alias("n")),
            spark.createDataFrame([(4,)], "n long"),
        )

    def test_load_full_symmetric(self, ld_reference_data):
        component = data.LDComponent(dataset=ld_reference_data)
        assert component.load_full_symmetric().count() == 4
        component.return_self_ld = True
        assert component.load_full_symmetric().count() == 8

    def test_load_tagged_variants(self, ld_reference_data):
        component = data.LDComponent(r2_threshold=None, dataset=ld_reference_data)
        assert component.load_tagged_variants().count() == 4
        component.return_self_ld = True
        with pytest.warns(
            UserWarning, match="Loading tagged variants with `return_self_ld=True` is discouraged"
        ):
            component.load_tagged_variants()

    def test_load_variants(self, ld_reference_data):
        assert data.LDComponent(dataset=ld_reference_data).load_variants().count() == 4


class TestChromosomeComponent:
    def test_init(self):
        data.ChromosomeComponent()


class TestAnnotationComponent:
    def test_get_filter_operation(self, annotation_component):
        assert isinstance(annotation_component.get_filter_operation(), filtering.Identity)
        annotation_component.filter_operation = filtering.QuantileCutoff(
            value_col="score", threshold=0.9, partition_by=("chr",)
        )
        assert isinstance(annotation_component.get_filter_operation(), filtering.QuantileCutoff)

    def test_get_normalisation(self, annotation_component):
        assert isinstance(annotation_component.get_normalisation(), normalisation.Identity)
        annotation_component.normalisation = normalisation.StandardScaler(
            values_col="score", output_col="score", partition_by=("chr",)
        )
        assert isinstance(annotation_component.get_normalisation(), normalisation.StandardScaler)

    def test_get_annotations_count(self, annotation_component):
        assert annotation_component.get_annotations_count() == 2

    def test_get_partitions_count(self, annotation_component):
        assert annotation_component.get_partitions_count() == 2

    def test_hard_partitions(self, annotation_component):
        assert annotation_component.hard_partitions

    def test_load(self, annotation_component, annotation_data, precision):
        chispa.assert_approx_df_equality(
            annotation_component.load(), annotation_data, precision=precision
        )


class TestGTEx:
    def test_init(self):
        data.GTEx()
