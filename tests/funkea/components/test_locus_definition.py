from typing import Literal

import chispa
import pyspark.sql.functions as F
import pytest
from pyspark.sql import Column

from funkea.components import locus_definition
from funkea.core import data


@pytest.fixture(scope="function")
def mock_abc(mock_abstract_methods):
    mock_abstract_methods(locus_definition.LocusDefinition)


@pytest.mark.usefixtures("mock_abc")
class TestLocusDefinition:
    def test_propagate_annotations(self, annotation_component):
        component = locus_definition.LocusDefinition()
        component.annotation = annotation_component
        component.some_step = locus_definition.LocusDefinition()
        assert component.some_step.annotation is None

        component.propagate_annotations()
        assert component.some_step.annotation is annotation_component

    def test_init_locus(self, sumstats):
        component = locus_definition.LocusDefinition()
        cols = component.locus_columns

        res = component.init_locus(sumstats)
        assert set(res.columns).difference(sumstats.columns) == {cols.id, cols.start, cols.end}
        chispa.assert_column_equality(res, data.ID_COL, cols.id)
        chispa.assert_column_equality(res, data.POSITION_COL, cols.start)
        chispa.assert_column_equality(res, data.POSITION_COL, cols.end)


class TestIdentity:
    def test_transform(self, sumstats, precision):
        component = locus_definition.Identity()
        cols = component.locus_columns
        chispa.assert_approx_df_equality(
            component.transform(sumstats).drop(cols.id, cols.start, cols.end),
            sumstats,
            precision=precision,
        )


class TestAddLDProxies:
    def test_transform(self, spark, ld_component, ancestry_sumstats):
        component = locus_definition.AddLDProxies(ld_component)
        res = component.transform(ancestry_sumstats)
        cols = component.locus_columns

        counts = res.groupBy(cols.id).count()
        chispa.assert_column_equality(
            spark.createDataFrame(
                [("rs117898110", 3), ("rs34673795", 2), ("rs73728002", 1), ("rs72844114", 2)],
                counts.withColumnRenamed("count", "count_expected").schema,
            ).join(counts.withColumnRenamed("count", "count_observed"), on=cols.id),
            "count_expected",
            "count_observed",
        )

    def test_raises_value_error(self, ancestry_sumstats):
        with pytest.raises(ValueError, match="`ld_component` can not be None"):
            locus_definition.AddLDProxies().transform(ancestry_sumstats)

    def test_issues_user_warning(self, ld_component, ancestry_sumstats):
        ld_component.return_self_ld = False
        with pytest.warns(
            UserWarning, match="It is recommended to set `return_self_ld` in `ld_component` to True"
        ):
            locus_definition.AddLDProxies(ld_component=ld_component).transform(ancestry_sumstats)


class TestExpand:
    def test_transform(self, spark, ancestry_sumstats, ld_component, chromosome_data):
        ancestry_sumstats = ancestry_sumstats.withColumn("id", F.lit("dummy"))
        component = locus_definition.Expand(extension=(10, 10))
        cols = component.locus_columns
        res = component.transform(ancestry_sumstats)
        chispa.assert_column_equality(
            res.withColumn("diff_observed", F.col(cols.start) - F.col("pos")).withColumn(
                "diff_expected", F.lit(-10)
            ),
            "diff_expected",
            "diff_observed",
        )
        chispa.assert_column_equality(
            res.withColumn("diff_observed", F.col(cols.end) - F.col("pos")).withColumn(
                "diff_expected", F.lit(10)
            ),
            "diff_expected",
            "diff_observed",
        )

        component.ld_component = ld_component
        res = (
            # here we need to add a dummy partition column as there is an internal aggregation
            component.transform(ancestry_sumstats).join(
                spark.createDataFrame(
                    [
                        ("rs117898110", 32503750 - 10, 32514030 + 10),
                        ("rs34673795", 32503750 - 10, 32513988 + 10),
                        ("rs73728002", 32513993 - 10, 32513993 + 10),
                        ("rs72844114", 32503750 - 10, 32514030 + 10),
                    ],
                    f"{cols.id} string, expected_locus_start long, expected_locus_end long",
                ),
                on=cols.id,
                how="inner",
            )
        )
        chispa.assert_column_equality(res, "expected_locus_start", cols.start)
        chispa.assert_column_equality(res, "expected_locus_end", cols.end)

        component.chromosomes = data.ChromosomeComponent(dataset=chromosome_data)

        res = res.join(
            component.transform(ancestry_sumstats).select(
                cols.id,
                F.col(cols.start).alias("chr_locus_start"),
                F.col(cols.end).alias("chr_locus_end"),
            ),
            on=cols.id,
            how="inner",
        )
        chispa.assert_column_equality(res, cols.start, "chr_locus_start")
        chispa.assert_column_equality(res, cols.end, "chr_locus_end")

        component.extension = (171115067, 171115067)
        chispa.assert_column_equality(
            component.transform(ancestry_sumstats).withColumn("chr_len", F.lit(171115067)),
            "chr_len",
            cols.end,
        )
        chispa.assert_column_equality(
            component.transform(ancestry_sumstats).withColumn("zero", F.lit(0)), "zero", cols.start
        )

    def test_output_columns_same(self, ancestry_sumstats, ld_component, chromosome_data):
        ancestry_sumstats = ancestry_sumstats.withColumn("id", F.lit("dummy"))
        component = locus_definition.Expand(extension=(10, 10))
        cols1 = component.transform(ancestry_sumstats).columns
        component.ld_component = ld_component
        cols2 = component.transform(ancestry_sumstats).columns
        component.chromosomes = data.ChromosomeComponent(dataset=chromosome_data)
        cols3 = component.transform(ancestry_sumstats).columns
        assert set(cols1) == set(cols2) == set(cols3)


class TestOverlap:
    def test_transform(self, annotation_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        component = locus_definition.Overlap(annotation_component)
        res = component.transform(sumstats)
        chispa.assert_column_equality(
            res.withColumn("expected_annotation", F.lit("z")),
            "expected_annotation",
            "annotation_id",
        )
        assert res.count() == 8

        component.keep_variants = True
        assert component.transform(sumstats).count() == 8

        annotation_component.dataset = annotation_component.dataset.filter(
            F.col("annotation_id") == "x"
        )
        assert component.transform(sumstats).count() == 4

    def test_raises_value_error(self, sumstats):
        with pytest.raises(ValueError, match="`annotation` is None"):
            locus_definition.Overlap().transform(sumstats)


class TestCollect:
    def test_transform(self, annotation_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))

        def _test_collection(
            what: Literal["annotation", "partition"], expected_col: Column
        ) -> None:
            res = locus_definition.Collect(what=what, annotation=annotation_component).transform(
                locus_definition.Overlap(annotation=annotation_component).transform(sumstats)
            )
            chispa.assert_column_equality(
                res.withColumn("expected_locus_collection", expected_col),
                "expected_locus_collection",
                locus_definition.Collect.collected_col,
            )
            assert res.count() == 4

        _test_collection("annotation", F.array(F.lit("z")))
        _test_collection("partition", F.array(F.lit("b"), F.lit("a")))

    def test_raises_value_error(self, sumstats):
        with pytest.raises(ValueError, match="`annotation` is None"):
            locus_definition.Collect().transform(sumstats)

    def test_raises_runtime_error(self, annotation_component, sumstats):
        with pytest.raises(RuntimeError, match="Input dataframe has neither .+ nor .+"):
            locus_definition.Collect(annotation=annotation_component).transform(sumstats)


class TestMerge:
    def test_transform(self, annotation_component, sumstats):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        res = locus_definition.Merge().transform(
            locus_definition.Collect(annotation=annotation_component).transform(
                locus_definition.Overlap(annotation=annotation_component).transform(sumstats)
            )
        )
        chispa.assert_column_equality(
            res.withColumn("expected_merge", F.array(F.lit("z"))),
            "expected_merge",
            locus_definition.Collect.collected_col,
        )
        assert res.count() == 1

    def test_raises_runtime_error(self, sumstats):
        with pytest.raises(RuntimeError, match=".+ is missing from the input dataframe columns"):
            locus_definition.Merge().transform(sumstats)


class TestCompose:
    def test_transform(self, annotation_component, sumstats, precision):
        sumstats = sumstats.withColumn("id", F.lit("dummy"))
        cols = locus_definition.LocusDefinition.locus_columns
        chispa.assert_approx_df_equality(
            locus_definition.Compose(annotation=annotation_component)
            .transform(sumstats)
            .drop(cols.id, cols.start, cols.end),
            sumstats,
            precision=precision,
        )

        res = locus_definition.Compose(
            locus_definition.Overlap(),
            locus_definition.Collect(),
            locus_definition.Merge(),
            annotation=annotation_component,
        ).transform(sumstats)
        chispa.assert_column_equality(
            res.withColumn("expected_merge", F.array(F.lit("z"))),
            "expected_merge",
            locus_definition.Collect.collected_col,
        )
        assert res.count() == 1
