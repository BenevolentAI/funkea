import chispa
import pyspark.sql.functions as F
import pytest

from funkea.components import variant_selection


@pytest.fixture(scope="function")
def mock_jar_in_session(monkeypatch):
    monkeypatch.setattr("funkea.core.utils.functions.jar_in_session", lambda *args, **kwargs: False)


@pytest.fixture(scope="function")
def mock_jvm_function_call(monkeypatch):
    # we need to have a mock object which holds an `over` method, as it is used inside LDPrune
    class _MockWindowFunction:
        @staticmethod
        def over(*args, **kwargs):
            return F.lit(True)

    # we mock the VariantIsIndependent UDF, as it is tested separately in Scala
    monkeypatch.setattr(
        "funkea.core.utils.jvm.JVMFunction.__call__", lambda *args, **kwargs: _MockWindowFunction()
    )


class TestVariantSelection:
    def test_init(self, mock_abstract_methods):
        mock_abstract_methods(variant_selection.VariantSelection)
        variant_selection.VariantSelection()


class TestIdentity:
    def test_transform(self, sumstats, precision):
        chispa.assert_approx_df_equality(
            variant_selection.Identity().transform(sumstats), sumstats, precision=precision
        )


class TestAssociationThreshold:
    def test_transform(self, sumstats):
        assert variant_selection.AssociationThreshold().transform(sumstats).count() == 1
        assert (
            variant_selection.AssociationThreshold(threshold=0.06).transform(sumstats).count() == 3
        )


class TestDropHLA:
    def test_transform(self, sumstats):
        assert variant_selection.DropHLA().transform(sumstats).count() == 0
        assert variant_selection.DropHLA(chromosome=10).transform(sumstats).count() == 4


class TestDropIndel:
    def test_transform(self, sumstats):
        assert variant_selection.DropIndel().transform(sumstats).count() == 3


class TestDropComplement:
    def test_transform(self, sumstats):
        assert variant_selection.DropComplement().transform(sumstats).count() == 3


class TestLDPrune:
    def test_transform(self, mock_jvm_function_call, ld_component, ancestry_sumstats):
        ld_component.return_self_ld = False
        ancestry_sumstats = ancestry_sumstats.withColumn("id", F.lit("dummy"))
        assert variant_selection.LDPrune(ld_component).transform(ancestry_sumstats).count() == 4

    def test_raises_runtime_error(self, mock_jar_in_session, ld_component, ancestry_sumstats):
        with pytest.raises(RuntimeError, match=".+ required for LD pruning."):
            variant_selection.LDPrune(ld_component).transform(ancestry_sumstats)


class TestFilterMAF:
    def test_transform(self, sumstats):
        assert variant_selection.FilterMAF().transform(sumstats).count() == 3


class TestDropInvalidPValues:
    def test_transform(self, sumstats):
        assert (
            variant_selection.DropInvalidPValues()
            .transform(sumstats.withColumn("p", F.col("p") - 2e-13))
            .count()
            == 3
        )


class TestCompose:
    def test_transform(self, sumstats, precision):
        chispa.assert_approx_df_equality(
            variant_selection.Compose().transform(sumstats), sumstats, precision=precision
        )
        res = variant_selection.Compose(
            variant_selection.AssociationThreshold(threshold=0.06),
            variant_selection.DropIndel(),
        ).transform(sumstats)
        assert set(res.columns) == set(sumstats.columns)
        assert res.count() == 2
