package ai.benevolent

import scala.collection.mutable

/** Container case class for the VariantIsIndependent aggregator buffer.
 */
case class BufferType(seen: mutable.HashSet[String], var lastIndependent: Boolean)
