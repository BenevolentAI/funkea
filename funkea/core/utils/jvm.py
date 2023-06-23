"""Interface for Spark UDFs written in Scala."""
from pathlib import Path
from typing import Callable, Iterable

from py4j.java_gateway import JavaClass, JavaObject, JavaPackage, JVMView
from pyspark import SparkContext
from pyspark.sql import Column, column

from funkea.core.utils import functions

PathLike = str | Path
ColumnOrName = str | Column

# converter type -- converts a set of python columns (or column names) into a java object
Converter = Callable[[ColumnOrName], JavaObject]

# For some reason, importing these directly makes the static analysers complain; but these
# definitely exist. Important to keep an eye on these, however, as these are private members and
# hence may be changed without warning
to_seq: Callable[[SparkContext, Iterable[ColumnOrName], Converter], JavaObject] = getattr(
    column, "_to_seq"
)
to_java_column: Converter = getattr(column, "_to_java_column")


class JVMFunction:
    """Wrapper class around a JVM UDF.

    Abstracts away the internals of interacting with the JVM and makes the Scala-Spark UDF a simple
    callable. The JVM UDF can then be executed like any other Spark function (see
    ``pyspark.sql.functions``).
    """

    def __init__(self, java_obj: JavaClass, class_path: str | None = None):
        """Initialise a JVMFunction.

        Args:
            java_obj: The ``UDFWrapper`` instance (see ``scala/spark-udf/README.md``).
            class_path: The class path of ``UDFWrapper`` instance.
        """
        sc: SparkContext | None = getattr(SparkContext, "_active_spark_context")
        assert sc is not None
        self.sc = sc

        if not hasattr(java_obj, "getUDF"):
            raise TypeError(
                f"The Java object at {class_path!r} does not have a `getUDF` method. Please make "
                "sure that the UDF extends the `UDFWrapper` trait."
            )

        # getUDF will return the UserDefinedFunction java object
        self.fn: JavaObject = java_obj.getUDF()
        if not hasattr(self.fn, "apply"):
            raise TypeError(
                "Returned object from `getUDF` does not have an apply method. Is it of "
                "`UserDefinedFunction` type?"
            )

    def __call__(self, *cols: ColumnOrName) -> Column:
        return Column(self.fn.apply(to_seq(self.sc, cols, to_java_column)))


def _get_jvm_udf(jvm: JVMView, class_path: str) -> JVMFunction:
    java_obj: JavaClass | JavaPackage = getattr(jvm, class_path)
    if isinstance(java_obj, JavaPackage):
        raise ValueError(
            f"No object at {class_path!r} in the current Java runtime. Ensure that the classPath "
            "is correct and that the relevant jar is provided at session initialisation."
        )

    _udf = JVMFunction(java_obj, class_path)
    return _udf


def get_jvm_udf(class_path: str) -> JVMFunction:
    """Get a UDF from the JVM.

    Allows users to access UDFs defined in Scala.

    Args:
        class_path: The path to the Scala class (e.g.
            ``ai.benevolent.VariantIsIndependent``).

    Returns:
        The JVMFunction instance.

    Examples:

        The following example gets the ``VariantIsIndependent`` UDAF from JVM and applies it to an
        example dataframe:

        >>> from pyspark.sql import Window
        >>> spark: SparkSession
        >>>
        >>> variant_is_independent = get_jvm_udf("ai.benevolent.VariantIsIndependent")
        >>> df = spark.createDataFrame(
        ...     [("rs1", ["rs3"], 0.02, 1), ("rs3", [], 0.5, 1)],
        ...     "rsid string, tags array<string>, p float, chr byte"
        ... )
        >>> window = Window.partitionBy("chr").orderBy("p", "rsid")
        >>> (
        ...     df.withColumn(
        ...         "independent",
        ...         variant_is_independent("rsid", "tags")
        ...         .over(window)
        ...     )
        ... )

    Notes:
        Only JVM UDFs which are currently available in the jars specified in ``spark.jars`` can be
        loaded in.
    """
    active_session = functions.get_session()

    jvm: JVMView = getattr(active_session, "_jvm")
    return _get_jvm_udf(jvm, class_path)
