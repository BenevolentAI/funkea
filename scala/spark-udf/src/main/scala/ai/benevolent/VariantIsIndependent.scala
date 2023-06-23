package ai.benevolent

import scala.collection.mutable
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.{Aggregator, UserDefinedFunction}
import org.apache.spark.sql.functions.udaf

/** Checks whether a variant is independent, given previously observed tags.
 *
 * This is a component part of the LD pruning algorithm. It should be applied over a Window, sorted by p-value (and
 * other arbitrary columns; see `finish` below). It is advised to use this via its Python wrapper in `funkea`, rather
 * than directly.
 *
 * Example usage:
 * {{{
 *   val df = Seq(
 *        ("rs6", Array("rs3"), 1, 0.01),
 *        ("rs18", Array("rs15", "rs9"), 1, 0.2),
 *        ("rs3", Array("rs1", "rs2"), 1, 0.001),
 *        ("rs5", Array("rs10"), 1, 0.005),
 *        ("rs1", Array("rs3"), 2, 0.01),
 *        ("rs2", Array("rs15", "rs9"), 2, 0.2),
 *   ).toDF("rsid", "tags", "chr", "p")
 *
 *   val fn = VariantIsIndependent.getUDF
 *   val res = df.select(
 *        col("rsid"),
 *        col("chr"),
 *        fn
 *          .apply(col("rsid"), col("tags"))
 *          .over(
 *            Window
 *            .partitionBy("chr")
 *            .orderBy("p", "rsid")
 *          )
 *          .alias("independent")
 *        )
 *        .show()
 * }}}
 *
 */
object VariantIsIndependent extends Aggregator[TaggedVariant, BufferType, Boolean] with UDFWrapper {
  // for details on the overrides, see https://spark.apache.org/docs/latest/sql-ref-functions-udf-aggregate.html

  override def zero: BufferType = BufferType(mutable.HashSet(), lastIndependent = false)

  override def reduce(b: BufferType, a: TaggedVariant): BufferType = {
    if (a != null && !b.seen.contains(a.rsid)) {
      /*
       if the observed rsid is not in the previously seen tags, mark it as independent
       the `.lastIndependent` is used in the `finish` method, which gets called at every unique value in an ordered
       window.
       See for gotchas:
       https://vincent.doba.fr/posts/20201206_spark-custom-aggregator-behavior-on-ordered-window-with-duplicates/
       */
      b.seen ++= a.tags.getOrElse[Array[String]](Array())
      b.lastIndependent = true
      b
    } else {
      b.lastIndependent = false
      b
    }
  }

  override def merge(b1: BufferType, b2: BufferType): BufferType = {
    // merge is not really relevant for our use-case, but is necessary for objects extending `Aggregator`
    b1.seen ++= b2.seen
    b1.lastIndependent = false
    b1
  }

  override def finish(reduction: BufferType): Boolean = {
    // This will be called for every row in the Window. Refer to the link in `VariantIsIndependent.reduce` for strange
    // behaviour in Spark
    reduction.lastIndependent
  }

  override def bufferEncoder: Encoder[BufferType] = Encoders.kryo[BufferType]

  override def outputEncoder: Encoder[Boolean] = Encoders.scalaBoolean

  def getUDF: UserDefinedFunction = udaf(this)
}
