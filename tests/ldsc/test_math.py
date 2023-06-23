import numpy as np
import pytest

import ldsc.math


def test_mask_infinite():
    assert np.all(ldsc.math.mask_infinite(np.array([1.0, float("inf")])) == np.ones(2))


def test_ld_correction():
    assert np.all(ldsc.math.ld_correction(np.random.randn(10, 1)) >= 1.0)
    assert ldsc.math.ld_correction(np.random.randn(10, 4)).shape == (10, 1)


def test_heteroskedasticity_correction():
    assert isinstance(ldsc.math.heteroskedasticity_correction(1, 1, 1, 1, 1), np.ndarray)
    assert ldsc.math.heteroskedasticity_correction(1, 1, 1, 1, 1).shape == (1, 1)


def test_extract_informative_measure_from_coefficient():
    assert np.all(0 <= ldsc.math.extract_informative_measure_from_coefficient(1, 1, 1) <= 1.0)
    assert ldsc.math.extract_informative_measure_from_coefficient(1, 1, 1).shape == (1,)


def test_compute_naive_informative_measure():
    assert np.all(0 <= ldsc.math.compute_naive_informative_measure(1, 1, 1, 1, 1) <= 1.0)
    assert ldsc.math.compute_naive_informative_measure(1, 1, 1, 1, 1).shape == (1,)


def test_solve_weighted_least_squares():
    X = np.random.randn(10, 1)
    b = np.random.randn(1)
    y = X @ b
    assert np.allclose(ldsc.math.solve_weighted_least_squares(X, y, np.ones(10)), b)


def test_iteratively_reweighted_least_squares():
    X = np.random.randn(10, 1)
    b = np.random.randn(1)
    y = X @ b

    assert np.allclose(ldsc.math.iteratively_reweighted_least_squares(X, y), b)
    assert np.allclose(
        ldsc.math.iteratively_reweighted_least_squares(X, y, return_weights=True).weights,
        np.ones(10),
    )


def test_reduce_condition_number():
    assert isinstance(ldsc.math.reduce_condition_number(1, 2), np.ndarray)
    assert ldsc.math.reduce_condition_number(1, 2).item() == 1.0


@pytest.fixture()
def mock_resampling_method_abc(monkeypatch):
    cls_path = "ldsc.math.ResamplingMethod"
    monkeypatch.setattr(f"{cls_path}.__abstractmethods__", set())


@pytest.mark.usefixtures("mock_resampling_method_abc")
class TestResamplingMethod:
    def test_init(self):
        m = ldsc.math.ResamplingMethod(10000, 200)

        assert len(m.sep) == 201

        with pytest.raises(ValueError, match="`n_observations` should be more than `n_blocks`"):
            ldsc.math.ResamplingMethod(10, 200)


class TestJackKnife:
    # TODO: create tests
    pass
