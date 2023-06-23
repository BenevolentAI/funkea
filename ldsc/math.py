"""Mathematical utilities and functions, shared between other modules in ldsc."""

import abc
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

from ldsc.util import (
    ToPandasMixin,
    backend,
    enforce_array_types,
    enforce_column_vector_if_1d,
    instance_has_attr,
    populate_arguments_from_locals,
)

np = backend()


def mask_infinite(y: np.ndarray) -> np.ndarray:
    is_finite = np.isfinite(y)
    return np.where(~is_finite, np.max(y[is_finite]), y)


@enforce_array_types
def ld_correction(x: np.ndarray) -> np.ndarray:
    ld = np.sum(x, axis=1, keepdims=True)
    corrected_ld = np.fmax(ld, 1.0)
    return corrected_ld


@enforce_array_types
@enforce_column_vector_if_1d(argnum=-2)
def heteroskedasticity_correction(
    intercept: np.ndarray,
    ld_scores: np.ndarray,
    heritability: np.ndarray,
    n_samples: np.ndarray,
    total_n_variants: float,
) -> np.ndarray:
    # see Online Methods under "regression weights"
    weighting = (intercept + n_samples * (heritability**2) * ld_scores / total_n_variants) ** 2
    return weighting


@enforce_array_types
def extract_informative_measure_from_coefficient(
    coefficient: np.ndarray,
    total_n_variants: float,
    avg_n_samples: float,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> np.ndarray:
    # workhorse function for getting either h^2 or rho_g from regression coefficients
    measure = np.clip(coefficient * total_n_variants / avg_n_samples, lower_bound, upper_bound)
    return measure


@enforce_array_types
@enforce_column_vector_if_1d(argnum=3)
def compute_naive_informative_measure(
    x: np.ndarray,
    y: np.ndarray,
    intercept: np.ndarray,
    n_samples: np.ndarray,
    total_n_variants: float,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> np.ndarray:
    # this needs thought -- it should be used to compute the initial weights used in the IRLS
    measure = total_n_variants * (np.mean(y) - intercept) / np.mean(x * n_samples)
    measure = np.clip(measure, lower_bound, upper_bound)
    return measure


@enforce_array_types
def solve_weighted_least_squares(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # solves the weighted least squares equation
    # https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
    w = np.sqrt(w)
    x = x * w.reshape((-1, 1))
    y = y * w
    b = np.linalg.solve(x.T @ x, x.T @ y)
    return b


IRLSResults = namedtuple("IRLSResults", ["coefficients", "weights"])


def iteratively_reweighted_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    p: float = 2.0,
    delta: float = np.finfo(float).eps,
    maxiter: int = 10_000,
    tol: float = 1e-8,
    initial_weights: np.ndarray | None = None,
    weight_update: Callable[..., np.ndarray] | None = None,
    return_weights: bool = False,
) -> np.ndarray | IRLSResults:
    r"""
    Compute the solution to a linear system of equations via generalised least squares with
    iterative reweighting. It is defined as:

    .. math::

        \mathbf{\beta}_{t} = (\mathbf{X}^T \mathbf{W}_t \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}_t \mathbf{y}

    where :math:`\mathbf{W}_t` are the regression weights at iteration :math:`t`. If
    ``weight_update`` is left unspecified, :math:`\mathbf{W}_t` is defined as follows:

    .. math::

        \mathbf{W}_t = diag(|\mathbf{y} - \mathbf{X} \mathbf{\beta}_{t-1}|^{p - 2})

    Args:
        x: Array of shape :math:`(N, D)`, containing the independent variables in the regression.
        y: Array of shape :math:`(N,)`, containing the dependent variables in the regression.
        p: The order of the :math:`p`-norm in when computing the residuals. (``default=2.0``)
        delta: The minimum value for the residuals when using the default weight update.
            (``default=np.finfo(float).eps``)
        maxiter: The maximum number of iterations in the weight updating step. (``default=10_000``)
        tol: The early stopping tolerance. When the sum of the p-norm residuals is equal to or
            below this threshold, the reweighting is stopped early. (``default=1e-8``)
        initial_weights: Optional initial regression weights. If None, it will be set to a 1-d
            array of ones of length :math:`N`. (``default=None``)
        weight_update: An optional callable defining the regression weight update. The arguments can
            be any of the local variables available in this function. If None, the default weight
            update function is used (see above definition). (``default=None``)
        return_weights: Whether to return the final regression weights. If True, the function
            returns a namedtuple of coefficients and regression weights. Otherwise, only the
            coefficients array is returned. (``default=False``)

    Returns:
        Either the model coefficients, or a tuple of the model coefficients and the regression
        weights.

    References:
        [1] https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares

    """
    (n, d) = x.shape
    if initial_weights is None:
        initial_weights = np.ones(1)
    weights = initial_weights.reshape(-1) * np.ones(n)
    coefficients = solve_weighted_least_squares(x, y, weights)

    if weight_update is None:

        def weight_update(residuals: np.ndarray) -> np.ndarray:
            return np.fmax(residuals, delta) ** (p - 2)

    # we wrap the weight update function, such that we can allow for maximum flexibility when
    # changing the update function -- the user can access all local variables in this function scope
    weight_update = populate_arguments_from_locals(weight_update)
    for _ in range(maxiter):
        residuals = abs(y - x @ coefficients) ** p
        weights = np.ones(n) * weight_update(locals())
        coefficients = solve_weighted_least_squares(x, y, weights)
        if np.sum(residuals) <= tol:
            break

    if return_weights:
        return IRLSResults(coefficients, weights)

    return coefficients


@enforce_array_types
@enforce_column_vector_if_1d(argnum=1)
def reduce_condition_number(x: np.ndarray, n_samples: np.ndarray) -> np.ndarray:
    # keep condition number low
    out = x * n_samples / np.mean(n_samples)
    return out


Estimator = Callable[..., np.ndarray]


@dataclass
class ResamplingSummary(ToPandasMixin):
    estimate: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    covariance: np.ndarray


class ResamplingMethod(abc.ABC):
    """Abstract base class for error estimation, in case we would like to implement a Bootstrap
    option in the future."""

    def __init__(
        self,
        n_observations: int,
        n_blocks: int,
        sep: np.ndarray = None,
        estimator: Estimator = solve_weighted_least_squares,
    ):
        if sep is None:
            if n_observations < n_blocks:
                raise ValueError("`n_observations` should be more than `n_blocks`")
            sep = np.linspace(0, n_observations, n_blocks + 1).astype(int)
        self.sep = sep
        self.n_blocks = n_blocks
        self.estimator = estimator

    @abc.abstractmethod
    def fit(self, *arrays: np.ndarray) -> "ResamplingMethod":
        pass

    @abc.abstractmethod
    def transform(self, biased_estimate: np.ndarray) -> ResamplingSummary:
        pass

    @abc.abstractmethod
    def fit_transform(self, *arrays: np.ndarray, **kwargs) -> ResamplingSummary:
        pass


class JackKnifeSummary(ResamplingSummary):
    pass


def compute_pseudo_values(
    biased_estimate: np.ndarray, partial_estimates: np.ndarray, n_blocks: int
) -> np.ndarray:
    return n_blocks * biased_estimate - (n_blocks - 1) * partial_estimates


def summarise_pseudo_values(pseudo_values: np.ndarray, n_blocks: int) -> JackKnifeSummary:
    covariance = np.atleast_2d(np.cov(pseudo_values.T, ddof=1) / n_blocks)
    variance = np.diag(covariance)
    std_err = np.sqrt(variance)
    estimate = np.mean(pseudo_values, axis=0)

    out = JackKnifeSummary(
        estimate=estimate, variance=variance, standard_error=std_err, covariance=covariance
    )
    return out


def block_jack_knife(
    *arrays: np.ndarray, sep: list, estimator: Estimator = solve_weighted_least_squares
) -> np.ndarray:
    def _drop_observations(arr: np.ndarray, bounds: tuple) -> np.ndarray:
        out = np.delete(arr, slice(*bounds), axis=0)
        return out

    partial_estimates = []
    for i, j in zip(sep[:-1], sep[1:]):
        # jack knife computes the partial estimates for a given statistic by leaving out the j-th
        # observation out of n observations
        # in the LD regression case, instead of considering all observations individually, we group
        # them into `blocks` and treat these blocks as observations with which to estimate the
        # statistic -- regression coefficients in this case
        arrays_ = (_drop_observations(a, (i, j)) for a in arrays)
        partial_estimate = estimator(*arrays_)
        partial_estimates.append(partial_estimate)
    partial_estimates = np.stack(partial_estimates)
    return partial_estimates


class JackKnifeNotFitException(Exception):
    pass


class JackKnife(ResamplingMethod):
    """Block jackknife estimation [1]_[2]_ of the least-squares regression coefficients.

    The pseudo- values are used to compute the standard errors of the coefficients.

    References:
        [1] https://influentialpoints.com/Training/jackknifing.htm
        [2] https://reich.hms.harvard.edu/sites/reich.hms.harvard.edu/files/inline-files/lecture2.pdf
    """

    partial_estimates: np.ndarray
    _ex = JackKnifeNotFitException("{name} cannot be called without first fitting the JackKnife.")

    def compute_partial_estimates(self, *arrays: np.ndarray) -> np.ndarray:
        return block_jack_knife(*arrays, sep=self.sep, estimator=self.estimator)

    def _summarise(self, pseudo_values: np.ndarray) -> JackKnifeSummary:
        return summarise_pseudo_values(pseudo_values, self.n_blocks)

    def fit(self, *arrays: np.ndarray) -> "JackKnife":
        self.partial_estimates = self.compute_partial_estimates(*arrays)
        return self

    @instance_has_attr("partial_estimates", _ex)
    def transform(self, biased_estimate: np.ndarray) -> JackKnifeSummary:
        pseudo_values = compute_pseudo_values(
            biased_estimate, self.partial_estimates, self.n_blocks
        )
        out = self._summarise(pseudo_values)
        return out

    def fit_transform(
        self, *arrays: np.ndarray, biased_estimate: np.ndarray = None
    ) -> JackKnifeSummary:
        if biased_estimate is None:
            biased_estimate = self.estimator(*arrays)

        self.fit(*arrays)
        out = self.transform(biased_estimate)
        return out
