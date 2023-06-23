import inspect

import numpy as np
import pytest

import ldsc.base


class Cases:
    X = [np.random.randn(10, 1), np.random.randn(10, 1)]
    Y = [np.random.randn(10), np.random.randn(10)]
    N_SAMPLES = [np.ones(10) * 100_000, np.ones(10) * 100_000]
    N_VARIANTS = [np.array([11_000_000]), np.array([11_000_000])]

    @classmethod
    def get_flattened_y(cls):
        return [np.prod(y, axis=1) if len(y.shape) > 1 else y for y in cls.Y]


@pytest.mark.parametrize(
    ["X", "y", "n_samples", "n_variants", "ld_weighting"],
    list(zip(Cases.X, Cases.Y, Cases.N_SAMPLES, Cases.N_VARIANTS, Cases.X)),
)
def test_cast_to_kw_fn_scope(X, y, n_samples, n_variants, ld_weighting):
    class _Obj:
        @ldsc.base.cast_kw_to_fn_scope
        def method(self, a):
            return a

    o = _Obj().method(**locals())
    assert isinstance(o, ldsc.base._FitFunctionScope)
    assert all(
        isinstance(getattr(o, name, None), np.ndarray)
        for name in ("X", "y", "n_samples", "n_variants", "ld_weighting")
    )


class TestCheckFitParamMixin:
    @pytest.mark.parametrize("n_samples", Cases.N_SAMPLES)
    def test_check_n_samples(self, n_samples):
        checker = ldsc.base.CheckFitParamMixin()
        assert np.all(checker.check_n_samples(n_samples, len(n_samples)) == n_samples)
        assert len(n_samples.shape) == 2 or np.all(
            checker.check_n_samples(int(n_samples[0]), len(n_samples)) == n_samples
        )

    @pytest.mark.parametrize("n_variants", Cases.N_VARIANTS)
    def test_check_n_variants(self, n_variants):
        checker = ldsc.base.CheckFitParamMixin()
        assert np.all(checker.check_n_variants(n_variants, len(n_variants)) == n_variants)
        assert np.all(checker.check_n_variants(int(n_variants[0]), len(n_variants)) == n_variants)

    @pytest.mark.parametrize(["ld_weighting", "ld_scores"], list(zip(Cases.X, Cases.X)))
    def test_check_ld_weighting(self, ld_weighting, ld_scores):
        checker = ldsc.base.CheckFitParamMixin()
        assert np.min(checker.check_ld_weighting(ld_weighting, ld_scores)) == 1.0
        assert len(checker.check_ld_weighting(ld_weighting, ld_scores).shape) == 2
        assert np.all(
            checker.check_ld_weighting(None, ld_scores)
            == np.fmax(np.sum(ld_scores, axis=1, keepdims=True), 1.0)
        )
        assert np.all(
            checker.check_ld_weighting(float(ld_weighting.reshape(-1)[0]), ld_scores)
            == np.fmax(np.ones(len(ld_scores)) * ld_weighting.reshape(-1)[0], 1.0)
        )
        with pytest.raises(ValueError):
            checker.check_ld_weighting(ld_weighting[:5], ld_scores)


@pytest.fixture()
def mock_ldsc_abc(monkeypatch):
    cls_path = "ldsc.base.LDScoreRegression"
    monkeypatch.setattr(f"{cls_path}.__abstractmethods__", set())
    monkeypatch.setattr(f"{cls_path}._update_fn", lambda *args, **kwargs: lambda x: np.ones(len(x)))
    monkeypatch.setattr(f"{cls_path}._init_weights", lambda _, a: np.ones(len(a.X)))


@pytest.mark.usefixtures("mock_ldsc_abc")
class TestLDScoreRegression:
    def test_init(self):
        model = ldsc.base.LDScoreRegression(n_jack_knife_blocks=2)
        assert model.n_jack_knife_blocks == 2
        assert set(model.config).symmetric_difference(
            inspect.signature(ldsc.base.LDScoreRegression.__init__).parameters
        ) == {"self"}

    @pytest.mark.parametrize(
        ["X", "y", "n_samples", "n_variants"],
        list(zip(Cases.X, Cases.get_flattened_y(), Cases.N_SAMPLES, Cases.N_VARIANTS)),
    )
    def test_fit(self, X, y, n_samples, n_variants):
        model = ldsc.base.LDScoreRegression(n_jack_knife_blocks=2)
        model = model.fit(X, y, n_samples=n_samples, n_variants=n_variants)

        # TODO: refactor fit function such that the units are easier to test
        # right now, the fit function is a much too large method and mixes responsibilities
        assert isinstance(model, ldsc.base.LDScoreRegression)
        assert all(
            hasattr(model, name)
            for name in (
                "_biased_estimate",
                "_jack_knife",
                "_jack_knife_summary",
                "_n_bar",
                "_n_vars",
            )
        )

    @pytest.mark.parametrize(
        ["X", "y", "n_samples", "n_variants"],
        list(zip(Cases.X, Cases.get_flattened_y(), Cases.N_SAMPLES, Cases.N_VARIANTS)),
    )
    def test_predict(self, X, y, n_samples, n_variants):
        model = ldsc.base.LDScoreRegression(n_jack_knife_blocks=2).fit(
            X, y, n_samples=n_samples, n_variants=n_variants
        )
        assert model.predict(X).shape == y.shape


@pytest.mark.usefixtures("mock_ldsc_abc")
class TestTwoStepFitter:
    def test_init(self):
        model = ldsc.base.LDScoreRegression(n_jack_knife_blocks=2)
        fitter = ldsc.base.TwoStepFitter(model)
        assert isinstance(fitter._model_step_1, ldsc.base.LDScoreRegression) and isinstance(
            fitter._model_step_2, ldsc.base.LDScoreRegression
        )
        assert (fitter._model_step_1 is not model) and (fitter._model_step_2 is not model)
        assert fitter._model_step_1.use_intercept
        assert not fitter._model_step_2.use_intercept
        assert callable(fitter.two_step_condition)

    @pytest.mark.parametrize(
        ["X", "y", "n_samples", "n_variants"],
        list(zip(Cases.X, Cases.get_flattened_y(), Cases.N_SAMPLES, Cases.N_VARIANTS)),
    )
    def test_fit(self, X, y, n_samples, n_variants):
        fitter = ldsc.base.TwoStepFitter(ldsc.base.LDScoreRegression(n_jack_knife_blocks=2)).fit(
            X, y, n_samples=n_samples, n_variants=n_variants
        )
        assert all(
            hasattr(fitter, name)
            for name in (
                "_biased_estimate",
                "_jack_knife",
                "_jack_knife_summary",
                "_n_samples",
                "_n_vars",
            )
        )

        model = fitter.model
        assert fitter._biased_estimate is model._biased_estimate
