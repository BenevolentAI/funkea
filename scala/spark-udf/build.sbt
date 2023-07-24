ThisBuild / version := sys.env.get("VERSION").getOrElse("0.1.0")

ThisBuild / scalaVersion := "2.12.16"

ThisBuild / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

scalacOptions ++= Seq("-unchecked", "-feature", "-deprecation")

lazy val root = (project in file("."))
  .settings(
    name := "spark-udf"
  )

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-sql_2.12" % "3.4.1",
  "org.scalatest" %% "scalatest" % "3.2.15" % Test,
  // below are pinned version to fix vulnerabilities detected by Snyk
  "com.google.protobuf" % "protobuf-java" % "3.22.2",
  "com.google.code.gson" % "gson" % "2.10.1",
  "com.google.guava" % "guava" % "30.0-jre",
  "io.netty" % "netty-common" % "4.1.77.Final"
)
