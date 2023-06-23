# Spark UDFs in Scala

In some scenarios, the Python / Pandas UDF interface will not be satisfactory, or runtime / resource constraints will
become too costly. In these cases, we can create Scala versions of these UDFs.

## Build

To build the `ai.beno.pmpt.genetics` package, assuming you already have `scala` installed, run the following:

```shell
sbt clean package
```

This generates a new `spark-udf_2.12-<version-tag>.jar` file containing the compiled package contents in the
`targets/scala-2.12` directory. The `jar` can then be provided to the `SparkSession` at runtime:

```python
from py4j.java_gateway import JVMView
from pyspark.sql import SparkSession

package_jar = "path/to/spark-udf_2.12-0.1.0.jar"
package_name = "ai.beno.pmpt.genetics"
udf_name = "VariantIsIndependent"
class_path = package_name + "." + udf_name

spark = (
    SparkSession.appName("some_name")
    .master("local[1]")
    .config("spark.jars", package_jar)
    .getOrCreate()
)

# The running Java virtual machine -- this is what Spark runs on.
# This is where our UDF object can be found!
jvm: JVMView = getattr(spark, "_jvm")

# now we can get our UDF object
udf = getattr(jvm, class_path).getUDF()
```

If this seems like a lot, don't worry, `funkea` has the function `funkea.core.utils.jvm.get_jvm_udf` which abstracts
all of this away.

## Development

### Project Structure

The project structure is as follows:

```shell
project/
  build.properties  # defines the scala build tool version
  plugins.sbt  # defines the build-time dependencies
src/
  main/  # defines all the source code of the project
  test/  # mirrors `src/main` and holds all the tests for the source code
build.sbt  # defines the build process for the project -- read by `sbt` (scala build tool)
```

Thus, all new functionality should be added somewhere under `src/main`. Moreover, it is recommended for new source code
to be added under `src/main/scala/ai/beno/pmpt/genetics` (deep folder structure is a convention of Java-derived
languages).

### Dependencies

Scala dependencies are handled by the `sbt` and need to be defined in the `build.sbt` by appending to the
`libraryDependencies` object. Also, the Scala version is defined in this file and should be kept at `2.12` for
compatability with Spark.

### Examples

A simple UDF example may look like this:

```scala
// define the package this UDF belongs to
// note: in scala this gives you access to all items under the package (e.g. UDFWrapper)
package ai.beno.pmpt.genetics

// import the required items
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

// create a UDF object which inherits from UDFWrapper
object PlusOne extends UDFWrapper {
  // define the UDF getter method and the UDF
  def getUDF: UserDefinedFunction = udf((x: Int) => x + 1)
}
```

Here, we wrap the actual UDF definition (`udf((x: Int) => x + 1)`) in an `object` which extends (~=inheritance) the
`UDFWrapper`. This is to ensure that the user defines a `getUDF` method, which makes it easier for us down the line
when we want to expose this function in `PySpark`.

## Useful Links

* [Scala Official Webpage](https://www.scala-lang.org/)
* [Scalar User Defined Functions (UDFs)](https://spark.apache.org/docs/latest/sql-ref-functions-udf-scalar.html)
* [User Defined Aggregate Functions (UDAFs)](https://spark.apache.org/docs/latest/sql-ref-functions-udf-aggregate.html)
* [Using Scala UDFs in PySpark](https://medium.com/wbaa/using-scala-udfs-in-pyspark-b70033dd69b9)
* [Spark custom aggregator behavior on ordered window with duplicates](https://vincent.doba.fr/posts/20201206_spark-custom-aggregator-behavior-on-ordered-window-with-duplicates/)
