"""Module for estimating trait heritability."""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Union

import pandas as pd
import quinn
from scipy import stats

from ldsc.base import (
    LDScoreRegression,
    ModelNotFitException,
    ToPandasMixin,
    TwoStepFitter,
    _FitFunctionScope,
)
from ldsc.math import (
    compute_naive_informative_measure,
    extract_informative_measure_from_coefficient,
    heteroskedasticity_correction,
    ld_correction,
    mask_infinite,
)
from ldsc.util import backend, enforce_array_types, flatten_output_array, instance_has_attr

np = backend()


@dataclass
class LDSCh2(ToPandasMixin):
    h2: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray


@enforce_array_types
def compute_naive_heritability(
    x: np.ndarray,
    y: np.ndarray,
    intercept: np.ndarray,
    n_samples: np.ndarray,
    total_n_variants: float,
) -> np.ndarray:
    h2 = compute_naive_informative_measure(x, y, intercept, n_samples, total_n_variants)
    heritability = np.sqrt(h2)
    return heritability


@enforce_array_types
@flatten_output_array
def compute_h2_weights(
    x: np.ndarray,
    intercept: np.ndarray,
    heritability: np.ndarray,
    ld_weighting: np.ndarray,
    n_samples: np.ndarray,
    total_n_variants: float,
):
    corrected_ld = ld_correction(x)

    het_correction = heteroskedasticity_correction(
        intercept=intercept,
        ld_scores=corrected_ld,
        heritability=heritability,
        n_samples=n_samples,
        total_n_variants=total_n_variants,
    )
    weights = 1 / (2 * het_correction * ld_weighting)
    return weights


def compute_initial_h2_weights(
    x: np.ndarray,
    y: np.ndarray,
    ld_weighting: np.ndarray,
    n_samples: np.ndarray,
    total_n_variants: float,
    constrained_intercept: np.ndarray = None,
) -> np.ndarray:
    intercept = constrained_intercept if constrained_intercept is not None else np.ones(1)
    naive_heritability = compute_naive_heritability(
        x, y, intercept=intercept, n_samples=n_samples, total_n_variants=total_n_variants
    )
    initial_weights = compute_h2_weights(
        x=x,
        intercept=intercept,
        heritability=naive_heritability,
        ld_weighting=ld_weighting,
        n_samples=n_samples,
        total_n_variants=total_n_variants,
    )
    return initial_weights


@enforce_array_types
def extract_heritability_from_coefficient(
    coefficient: np.ndarray, total_n_variants: float, avg_n_samples: float
) -> np.ndarray:
    # decided to extract heritability instead of h2 for readability (mistakenly, perhaps?)
    h2 = extract_informative_measure_from_coefficient(coefficient, total_n_variants, avg_n_samples)
    heritability = np.sqrt(h2)
    return heritability


def h2_weight_update(
    x: np.ndarray,
    coefficients: np.ndarray,
    n_samples: np.ndarray,
    ld_weighting: np.ndarray,
    total_n_variants: float,
    use_intercept: bool,
    constrained_intercept: np.ndarray = None,
) -> np.ndarray:
    # coefficients have first element as intercept, hence we exclude it
    heritability = extract_heritability_from_coefficient(
        coefficients[int(use_intercept) :], total_n_variants, n_samples.mean()
    )

    intercept = coefficients[:1] ** int(
        use_intercept
    )  # if we don't use an intercept, default to 1.0
    if constrained_intercept is not None:
        intercept = constrained_intercept

    weights = compute_h2_weights(
        x=x[..., int(use_intercept) :],
        intercept=intercept,
        heritability=heritability[:1],
        ld_weighting=ld_weighting,
        n_samples=n_samples,
        total_n_variants=total_n_variants,
    )
    return weights


@enforce_array_types
def convert_h2_to_liability(
    h2: np.ndarray,
    phenotype_sample_prevalence: np.ndarray,
    phenotype_population_prevalence: np.ndarray,
) -> np.ndarray:
    threshold = stats.norm.isf(phenotype_population_prevalence)

    # alias variables to make the below expression more terse
    p = phenotype_sample_prevalence
    k = phenotype_population_prevalence

    conversion_factor = k**2 * (1 - k) ** 2 / (p * (1 - p) * stats.norm.pdf(threshold) ** 2)
    return h2 * conversion_factor


class HeritabilityRegression(LDScoreRegression):
    _ex: ModelNotFitException = ModelNotFitException(
        "method not viable as model has not yet been fit"
    )

    def _update_fn(self, fit_fn_scope: _FitFunctionScope) -> Callable:
        total_n_variants = fit_fn_scope.n_variants.sum()
        fn = partial(
            h2_weight_update,
            ld_weighting=fit_fn_scope.ld_weighting,
            n_samples=fit_fn_scope.n_samples,
            total_n_variants=total_n_variants,
            use_intercept=self.use_intercept,
            constrained_intercept=self.constrained_intercept,
        )
        return fn

    def _init_weights(self, fit_fn_scope: _FitFunctionScope) -> np.ndarray:
        total_n_variants = fit_fn_scope.n_variants.sum()
        weights = compute_initial_h2_weights(
            fit_fn_scope.X,
            fit_fn_scope.y,
            n_samples=fit_fn_scope.n_samples,
            ld_weighting=fit_fn_scope.ld_weighting,
            total_n_variants=total_n_variants,
            constrained_intercept=self.constrained_intercept,
        )
        return weights

    @property
    @instance_has_attr("_jack_knife_summary", _ex)
    def h2_estimate(self) -> LDSCh2:
        measure = self._measure_estimate
        out = LDSCh2(
            h2=measure.measure, covariance=measure.covariance, standard_error=measure.standard_error
        )
        return out

    @instance_has_attr("_jack_knife_summary", _ex)
    def liability(
        self,
        phenotype_sample_prevalence: Union[float, np.ndarray],
        phenotype_population_prevalence: Union[float, np.ndarray],
    ) -> np.ndarray:
        observed_heritability = self.h2_estimate
        out = convert_h2_to_liability(
            observed_heritability.h2, phenotype_sample_prevalence, phenotype_population_prevalence
        )
        return out


LD_SCORE_COLUMNS = ["chr", "pos", "ld_score", "chi2", "n_samples", "n_variants"]
LIABILITY_COLUMNS = ["ncase", "population_prevalence"]


def compute_heritability_estimate(df: pd.DataFrame) -> pd.DataFrame:
    r"""Compute the heritability estimate for a given study. It will compute the observed and
    liability scale :math:`h^2` for the study provided in ``df``. If the trait being checked is
    quantitative, then the :math:`h^2` on the liability scale will simply be ``nan``. For
    quantitative traits, one can either set the ``population_prevalence`` as ``1.0`` and
    ``ncase == n_samples``, or one can set ``population_prevalence`` to ``nan``. In either scenario,
    the result is the same, as the liability scale :math:`h^2` does not apply. However, it is
    computed here regardless of trait type for easier scaling when used in Spark (see
    `compute_heritability_estimate_udf`).

    Args:
        df: dataframe containing data required for computing the heritability estimate. The columns
            required are: ``chr``, ``pos``, ``ld_score``, ``chi2``, ``n_samples``, ``n_variants``,
            ``ncase``, and ``population_prevalence``.

    Returns:
        The heritability estimate and associated statistics.
    """
    quinn.validate_presence_of_columns(df, LD_SCORE_COLUMNS + LIABILITY_COLUMNS)
    df.sort_values(["chr", "pos"], inplace=True)
    X = df["ld_score"].values.reshape((-1, 1))
    y = mask_infinite(df["chi2"].values.reshape(-1))
    n_samples = int(df["n_samples"].iloc[0])
    model: HeritabilityRegression = (  # type: ignore
        TwoStepFitter(
            HeritabilityRegression(n_jack_knife_blocks=min(len(X), 200), use_intercept=True)
        )
        .fit(X, y, n_samples=n_samples, n_variants=int(df["n_variants"].iloc[0]))
        .model
    )

    return (
        model.h2_estimate.to_pandas(array_to_item=True)
        .rename(columns={"h2": "h2_observed"})
        .assign(
            h2_liability=model.liability(
                df["ncase"].iloc[0] / n_samples, df["population_prevalence"].iloc[0]
            ).item()
        )
        .join(
            model.intercept.to_pandas(array_to_item=True).rename(
                columns={"standard_error": "intercept_standard_error"}
            )
        )
    )


def compute_heritability_estimate_udf(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience UDF computing the study heritability estimate with Spark. Please see `Examples`
    for recommended usage.

    Args:
        df (pd.DataFrame): dataframe containing data required for heritability estimation. See
            ``compute_heritability_estimate`` for required columns. Additionally, this function also
            requires a ``study_id`` column.

    Returns:
        The heritability estimate and associated statistics.

    Examples:

        >>> from pyspark.sql import DataFrame
        >>>
        >>> sumstats: DataFrame
        >>> res = (
        ...     sumstats.groupBy("study_id")
        ...     .applyInPandas(compute_heritability_estimate_udf, udf_schema())
        ...     .toPandas()
        ... )
    """
    quinn.validate_presence_of_columns(df, LD_SCORE_COLUMNS + LIABILITY_COLUMNS + ["study_id"])
    study_id: str = df["study_id"].iat[0]
    try:
        return compute_heritability_estimate(df).assign(
            study_id=study_id, succeeded=True, failure_msg=""
        )
    except Exception as exc:
        columns = [
            "h2_observed",
            "standard_error",
            "covariance",
            "h2_liability",
            "intercept",
            "intercept_standard_error",
        ]
        return pd.DataFrame([(np.nan,) * len(columns)], columns=columns, dtype=float).assign(
            study_id=study_id, succeeded=False, failure_msg="; ".join(map(str, exc.args))
        )


def udf_schema() -> str:
    """The output schema for the ``compute_heritability_estimate_udf``.

    Returns:
        The Spark schema.
    """
    return (
        "h2_observed double, standard_error double, covariance double, h2_liability double, "
        "intercept double, intercept_standard_error double, study_id string, succeeded boolean, "
        "failure_msg string"
    )
