from typing import Callable, Type, TypeVar

import pyspark.sql.functions as F
import pytest
from pyspark.ml import Transformer
from pyspark.sql import DataFrame, SparkSession

T = TypeVar("T", bound=Transformer)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    from funkea.core.utils import files

    session = (
        SparkSession.builder.master("local[2]")
        .config("spark.jars", files.SCALA_UDF.as_posix())
        .getOrCreate()
    )
    yield session
    session.stop()


def class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


@pytest.fixture(scope="function")
def mock_abstract_methods(monkeypatch) -> Callable[[type], None]:
    def mock(cls: type) -> None:
        monkeypatch.setattr(f"{class_path(cls)}.__abstractmethods__", set())

    return mock


@pytest.fixture(scope="function")
def mock_transform(monkeypatch) -> Callable[[Type[T], Callable[[T, DataFrame], DataFrame]], None]:
    def mock(
        cls: Type[T], transform: Callable[[T, DataFrame], DataFrame] = lambda _, df: df
    ) -> None:
        monkeypatch.setattr(f"{class_path(cls)}._transform", transform)

    return mock


@pytest.fixture(scope="function")
def annotation_data(spark):
    return spark.createDataFrame(
        [
            ("a", "x", 32503750 - 100_000, 32514030 - 100_000, 6, 0.12),
            ("a", "z", 32503750 - 100, 32503750 + 11_000, 6, 0.96),
            ("b", "z", 32503750 - 100, 32503750 + 11_000, 6, -0.45),
        ],
        "partition_id string, annotation_id string, start int, end int, chr byte, score double",
    )


@pytest.fixture(scope="function")
def annotation_component(annotation_data):
    from funkea.core import data

    return data.AnnotationComponent(
        columns=data.AnnotationColumns(values="score"),
        partition_type=data.PartitionType.HARD,
        dataset=annotation_data,
    )


@pytest.fixture(scope="function")
def sumstats(spark):
    return spark.createDataFrame(
        [
            ("rs117898110", 6, 32503750, 0.1, "A", "T", 0.0001),
            ("rs34673795", 6, 32513988, 0.05, "AA", "G", 0.05),
            ("rs73728002", 6, 32513993, 5e-8, "G", "A", 0.25),
            ("rs72844114", 6, 32514030, 1e-13, "T", "C", 0.45),
        ],
        "rsid string, chr long, pos long, p double, "
        "effect_allele string, other_allele string, maf double",
    )


@pytest.fixture(scope="function")
def ancestry_sumstats(sumstats):
    return sumstats.withColumn("ancestry", F.lit("eur"))


@pytest.fixture(scope="module")
def ld_reference_data(spark):
    return spark.createDataFrame(
        [
            (6, 32503750, 'rs117898110', 32513988, 'rs34673795', 0.514666699804365635, 503, 'eur'),
            (6, 32503750, 'rs117898110', 32513993, 'rs73728002', 0.001962719950824976, 503, 'eur'),
            (6, 32503750, 'rs117898110', 32514030, 'rs72844114', 0.214501700177788734, 503, 'eur'),
        ],
        "chr byte, pos_source long, rsid_source string, "
        "pos_target long, rsid_target string, correlation float, "
        "sample_size int, ancestry string",
    )


@pytest.fixture(scope="function")
def ld_component(ld_reference_data):
    from funkea.core import data

    return data.LDComponent(
        return_self_ld=True,
        dataset=ld_reference_data,
    )


@pytest.fixture(scope="function")
def chromosome_data(spark) -> DataFrame:
    from funkea.core import data

    return spark.createDataFrame(
        [(6, 171115067)], [data.ChromosomeComponent.chr_col, data.ChromosomeComponent.length_col]
    )


@pytest.fixture(scope="session")
def precision() -> float:
    return 1e-8
