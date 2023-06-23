package ai.benevolent

import org.apache.spark.sql.expressions.UserDefinedFunction

/** Trait to enforce the implementation of a getUDF method.
 *
 * When we import Spark UDFs from Scala into Python, the function wrappers in `mendel` will look for a `getUDF` method.
 * This wrapper trait enforces this convention. Hence, each UDF should extend this.
 *
 */
trait UDFWrapper {
  // traits vs abstract classes https://www.baeldung.com/scala/traits-vs-abstract-classes
  // in short: traits can be used in multiple inheritance

  /**
   * Get the UDF callable.
   *
   * This is a getter method for convenience when we import them into Python.
   *
   */
  def getUDF: UserDefinedFunction
}
