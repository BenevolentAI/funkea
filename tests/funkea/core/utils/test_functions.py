import chispa
import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from funkea.core.utils import functions


def test_overlap(spark):
    example = (
        spark.createDataFrame(
            [
                (10, 20, 15, 35, True, True),
                (33, 41, 32, 35, False, True),
                (9, 11, 12, 64, False, False),
            ],
            "start_a byte, end_a byte, start_b byte, end_b byte, "
            "point_overlap boolean, range_overlap boolean",
        )
        .withColumn(
            "test_point_overlap",
            functions.overlap(F.col("start_a"), F.col("end_a"), F.col("start_b")),
        )
        .withColumn(
            "test_range_overlap",
            functions.overlap(F.col("start_a"), F.col("end_a"), F.col("start_b"), F.col("end_b")),
        )
    )
    chispa.assert_column_equality(example, "test_point_overlap", "point_overlap")
    chispa.assert_column_equality(example, "test_range_overlap", "range_overlap")


@pytest.mark.parametrize(
    ["inputs", "result"],
    [
        (
            (pd.DataFrame([(1,)], columns=["a"]), pd.DataFrame([(2,), (3,)], columns=["b"])),
            pd.DataFrame([(1, 2), (1, 3)], columns=["a", "b"]),
        ),
        (
            (
                pd.DataFrame([(1, 2)], columns=["a", "b"]),
                pd.DataFrame([(2, 3), (4, 5)], columns=["c", "d"]),
            ),
            pd.DataFrame([(1, 2, 2, 3), (1, 2, 4, 5)], columns=["a", "b", "c", "d"]),
        ),
    ],
)
def test_cross_join(inputs, result):
    pd.testing.assert_frame_equal(
        functions.cross_join(*inputs).reset_index(drop=True),
        result,
    )


@pytest.mark.parametrize(
    ["example", "result"],
    [
        (
            pd.DataFrame(
                {
                    "id": ["id"] * 3,
                    "rsid": ["a", "b", "c"],
                    "locus": [["x", "y"], ["y"], ["y", "z"]],
                }
            ),
            pd.DataFrame({"id": ["id"], "rsid": ["a;b;c"], "locus": [["x", "y", "z"]]}),
        )
    ],
)
def test_merge_collections(example, result):
    pd.testing.assert_frame_equal(
        functions.merge_collections(
            example, id_column="rsid", collections_column="locus", partition_cols=("id",)
        )[result.columns].assign(locus=lambda df: df["locus"].apply(sorted)),
        result.assign(locus=lambda df: df["locus"].apply(sorted)),
    )


def test_get_session(spark):
    assert isinstance(functions.get_session(), SparkSession)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_chunk(batch_size):
    sequence = [*range(103)]
    assert all(len(batch) <= batch_size for batch in functions.chunk(sequence, batch_size))
    assert sum(len(batch) for batch in functions.chunk(sequence, batch_size)) == len(sequence)


def test_jar_in_session(spark):
    from funkea.core.utils import files

    assert functions.jar_in_session(files.SCALA_UDF.name)
    assert not functions.jar_in_session("some-random_scala.jar")
