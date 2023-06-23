import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="package")
def spark(request):
    SparkSession.Builder._options = {}

    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    def _spark_stop():
        spark.stop()
        SparkSession.Builder._options = {}

    request.addfinalizer(_spark_stop)
    return spark
