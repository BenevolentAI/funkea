import numpy as np
import pandas as pd
import pytest

import ldsc.heritability


@pytest.fixture()
def data():
    return (
        np.random.randn(1000, 1),
        np.random.randn(1000),
        np.ones(1000) * 100_000,
        np.array([11_000_000]),
    )


@pytest.fixture()
def data_df(data):
    X, y, n_samples, n_variants = data
    df = pd.DataFrame(
        {"ld_score": X.flatten(), "chi2": y, "pos": [*range(1, len(X) + 1)], "n_samples": n_samples}
    )
    df["n_variants"] = n_variants[0]
    df["chr"] = 1
    df["ncase"] = 1000
    df["population_prevalence"] = 0.01
    return df


class TestHeritabilityRegression:
    def test_fit(self, data):
        X, y, n_samples, n_variants = data
        model = ldsc.heritability.HeritabilityRegression(n_jack_knife_blocks=2).fit(
            X, y, n_samples=n_samples, n_variants=n_variants
        )

        assert isinstance(model.h2_estimate, ldsc.heritability.LDSCh2)


def test_compute_heritability_estimate(data_df):
    df = ldsc.heritability.compute_heritability_estimate(data_df)
    assert isinstance(df, pd.DataFrame)
    assert not {
        "h2_observed",
        "standard_error",
        "covariance",
        "h2_liability",
        "intercept",
        "intercept_standard_error",
    }.symmetric_difference(df.columns)
