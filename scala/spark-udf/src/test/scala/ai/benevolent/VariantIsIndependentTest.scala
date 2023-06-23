package ai.benevolent

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.expressions.Window
import org.scalatest.flatspec.AnyFlatSpec

class VariantIsIndependentTest extends AnyFlatSpec {
  "VariantIsIndependent" should "only keep the variants not previously observed in a set of tags" in {
    val spark = SparkSession.builder()
      .appName("test")
      .master("local[2]")
      .getOrCreate()

    import spark.implicits._

    val df = Seq(
      ("rs6", Array("rs3"), 1, 0.01),
      ("rs18", Array("rs15", "rs9"), 1, 0.2),
      ("rs3", Array("rs1", "rs2"), 1, 0.001),
      ("rs5", Array("rs10"), 1, 0.005),
      ("rs1", Array("rs3"), 2, 0.01),
      ("rs2", Array("rs15", "rs9"), 2, 0.2),
    ).toDF("rsid", "tags", "chr", "p")

    val fn = VariantIsIndependent.getUDF
    val res = df.select(
      col("rsid"),
      col("chr"),
      fn
       .apply(col("rsid"), col("tags"))
       .over(
          Window
            .partitionBy("chr")
            .orderBy("p")
        )
       .alias("independent")
    )
      .filter(col("independent"))
      .select("rsid")
      .collect().map(_.getString(0))

    assert(Set("rs3", "rs5", "rs1", "rs2").intersect(res.toSet).size == 4)
  }
}
