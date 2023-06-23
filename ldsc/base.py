"""Module containing the package LDScoreRegression abstract base class and the TwoStepFitter class.

Only the TwoStepFitter class ought to be used directly.
"""

import abc
import inspect
import warnings
from dataclasses import dataclass
from typing import Callable

from ldsc.math import (
    JackKnife,
    JackKnifeSummary,
    iteratively_reweighted_least_squares,
    reduce_condition_number,
)
from ldsc.util import (
    ToPandasMixin,
    add_docstring,
    backend,
    check_fit_fn_arrays,
    enforce_data_dim,
    harmonise_inputs_to_backend,
    instance_has_attr,
    remove_kw,
    remove_self_kw,
)

np = backend()


@dataclass
class LDSCCoefficients(ToPandasMixin):
    # dataclass holding the LDSC model coefficients and standard error
    coefficients: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray


@dataclass
class LDSCIntercept(ToPandasMixin):
    # dataclass for the intercept and error
    intercept: np.ndarray
    standard_error: np.ndarray


@dataclass
class _LDSCMeasureEstimate:
    # base dataclass for measure estimate; "measure" is one of heritability / genetic covariance;
    # i.e. the model beta
    measure: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray


@dataclass
class _FitFunctionScope:
    # this object gives us easy access to the most important variables in the fit function
    # the weighting functions (at least in part) depend on these, and hence we need to provide them
    # all
    X: np.ndarray
    y: np.ndarray
    n_samples: np.ndarray
    n_variants: np.ndarray
    ld_weighting: np.ndarray


def cast_kw_to_fn_scope(method):
    # cast the keywords to a function scope object to make access easier (autocompletion) and
    # options clearer to the reader
    params = inspect.signature(_FitFunctionScope.__init__).parameters

    def _method(self, **kwargs):
        kw = {k: v for k, v in kwargs.items() if k in params}
        scope = _FitFunctionScope(**kw)
        out = method(self, scope)
        return out

    return _method


class ModelNotFitException(Exception):
    # custom exception for the case when methods / properties are being called before the model has
    # been fit
    pass


class CheckFitParamMixin:
    """A mixin to do some standard checks whether a subset of the fit function parameters fulfill
    certain conditions.

    Provided as a mixin to allow re-use between ``LDScoreRegression`` and ``TwoStepFitter``. For
    internal use only.
    """

    @staticmethod
    def check_n_samples(n_samples: int | float | np.ndarray, n_observations: int):
        # `n_samples` always needs to be a 1-d / 2-d array of numerics (int or float; float in the
        # case of `GeneticCovarianceRegression`), where the first dimension is of length
        # `n_observations` (number of SNPs in regression)
        if isinstance(n_samples, (int, float)) or len(n_samples) == 1:
            n_samples = np.ones(n_observations) * n_samples
        return n_samples

    @staticmethod
    def check_n_variants(n_variants: int | float | np.ndarray, n_dimensions: int):
        # TODO: remove float annotation for n_variants
        # `n_samples` always needs to be a 1-d array of integers of length `n_dimensions`. So in
        # the univariate case, it would be a 1-d array of length 1 (i.e. a scalar kept as an array
        # for generality).
        if isinstance(n_variants, (int, float)) or len(n_variants) == 1:
            n_variants = np.ones(n_dimensions) * n_variants
        return n_variants

    @staticmethod
    def check_ld_weighting(
        ld_weighting: float | np.ndarray | None, ld_scores: np.ndarray
    ) -> np.ndarray:
        # `ld_weighting` must be of length `n_snp` in the first dimension (i.e. matching the first
        # dimension of `ld_scores`) and must be a 1-d array of floats.
        (n_observations, _) = ld_scores.shape
        if ld_weighting is None:
            # the default case removes the partitioning of the LD scores. In the univariate case,
            # we will just get back the LD scores in a flattened array (as per tutorials this should
            # be most of the cases).
            out = np.sum(ld_scores, axis=1, keepdims=True)
        elif isinstance(ld_weighting, float) or len(ld_weighting) == 1:
            # TODO: remove this condition?
            out = np.ones((n_observations, 1)) * ld_weighting
        elif isinstance(ld_weighting, np.ndarray) and len(ld_weighting) == len(ld_scores):
            # this is the case when `ld_weighting` has a valid specification
            out = ld_weighting
        else:
            raise ValueError("`ld_weighting` is not valid")

        # `ld_weighting` is always at least 1, because LD scores should always be at least 1 (a
        # variant always has an r^2 of 1 with itself)
        out = np.fmax(out, 1.0)
        out = out.reshape((-1, 1))
        return out


class LDScoreRegression(abc.ABC, CheckFitParamMixin):
    r"""
    Implementation of LD-score regression adapted from [1]_ and [2]_. Briefly, LD score regression
    uses iteratively reweighted least-squares with a custom weighting function to compute the linear
    model coefficients. The model maps index variant LD score(s) to :math:`\chi^2` statistics.

    The model is defined as:

    .. math::

        \mathbb{E}[\chi^2|l_j] = 1 + Na + \frac{Nh^2l_j}{M}

    where :math:`N` is the number of samples, :math:`M` is the number of variants, :math:`a` is a
    measure of confounding, :math:`h^2` is the heritability and :math:`l_j` is the LD score of
    variant :math:`j`. The LD score :math:`l_j` of variant `j` is defined as:

    .. math::

        l_j = \sum_{k \in C} r_{jk}^2

    where :math:`C` is the set of indices of the SNPs in LD with variant :math:`j`, :math:`r_{jk}^2`
    is the squared Pearson correlation of variant :math:`j` and variant :math:`k`. The original
    paper [1]_ used an unbiased form of the classical Pearson correlation, which is given by:

    .. math::

        \hat{r^2} = r^2 - \frac{1 - r^2}{N - 2}

    The coefficient standard errors are then computed via block jackknifing [3]_[4]_.

    Args:
        n_jack_knife_blocks: The number of blocks for which to compute partial estimates
            in the jackknife standard error estimation. (``default=200``)
        use_intercept: Whether to use an intercept in the regression. (``default=True``)
        sep: an optional list of integers defining the separators for the
            blocks in the block jackknife. If None, evenly spaced separators are created for
            `n_jack_knife_blocks` blocks. (``default=None``)
        use_jack_knife_estimate: Whether to use the jackknifed estimate for measure
            estimate (heritability). This is recommended to be left as the default.
            (``default=False``)
        weight_update_fn: An optional callable which computes the weight update used during the
            IRLS model fitting procedure. Can take any argument as defined in
            `ldsc.math.iteratively_reweighted_least_squares`. If None, the default depends on
            the subclass of LDScoreRegression. (``default=None``)
        max_irls_iter: The number of iterations used for reweighting in IRLS. (``default=2``)
        constrained_intercept: An optional 1-d array (of length 1), predefining an intercept.
            (``default=None``)

    Examples:
        Fitting a simple model:

        >>> X, y, n_samples, n_variants = ...  # load necessary data
        >>>
        >>> model = LDScoreRegression(n_jack_knife_blocks=5, use_intercept=True)
        >>> model = model.fit(X, y, n_samples=n_samples, n_variants=n_variants)

    References:
        .. [1] Bulik-Sullivan, B.K., Loh, P.R., Finucane, H.K., Ripke, S., Yang, J., Patterson, N.,
            Daly, M.J., Price, A.L. and Neale, B.M., 2015. LD Score regression distinguishes
            confounding from polygenicity in genome-wide association studies. Nature genetics,
            47(3), pp.291-295.
        .. [2] GitHub repo: https://github.com/bulik/ldsc
        .. [3] https://influentialpoints.com/Training/jackknifing.htm
        .. [4] https://reich.hms.harvard.edu/sites/reich.hms.harvard.edu/files/inline-files/lecture2.pdf

    """
    _biased_estimate: np.ndarray
    _jack_knife: JackKnife
    _jack_knife_summary: JackKnifeSummary
    _n_bar: float
    _n_vars: np.ndarray
    _ex: ModelNotFitException = ModelNotFitException(
        "method not viable as model has not yet been fit"
    )

    def __init__(
        self,
        n_jack_knife_blocks: int = 200,
        use_intercept: bool = True,
        sep: list[int] | None = None,
        use_jack_knife_estimate: bool = False,
        weight_update_fn: Callable[..., np.ndarray] | None = None,
        max_irls_iter: int = 2,
        constrained_intercept: np.ndarray | None = None,
    ):
        self.n_jack_knife_blocks = n_jack_knife_blocks
        self.use_intercept = use_intercept
        self.sep = sep
        self.use_jack_knife_estimate = use_jack_knife_estimate
        self.weight_update_fn = weight_update_fn
        self.max_irls_iter = max_irls_iter
        self.constrained_intercept = constrained_intercept

        # keeping the configuration here to make instantiation of new instances easy
        self.config = remove_self_kw(locals())

    @harmonise_inputs_to_backend
    @check_fit_fn_arrays(allow_multiple_outputs=False)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_samples: int | float | np.ndarray,
        n_variants: int | float | np.ndarray,
        ld_weighting: float | np.ndarray | None = None,
        **kwargs,
    ) -> "LDScoreRegression":
        r"""Fit an LD score regression model to a set of observed chi-square statistics.

        Args:
            X: The data matrix. It contains the LD-scores of the variants and is of shape
                :math:`(N, D)`, where :math:`N` is the number of variants and :math:`D` is
                the dimensionality of the data. Usually, :math:`D = 1`, as we are considering just
                the LD score for a given variant. However, :math:`D > 1` is also possible and can
                be used to compute the partitioned heritability [1]_.
            y: The dependent variable. This is a 1-d array of length :math:`N` containing the
                :math:`\chi^2` statistics for the variants under consideration.
            n_samples: The number of samples for a given variant in a study. This is not necessarily
                the same across variants, as multiple studies can be concatenated for this analysis.
                In which case a 1-d array of length :math:`N` is expected. Otherwise, an integer can
                be provided which is then expanded into an array of length :math:`N` where each
                entry is the number provided. As it stands, it is not recommended to concatenate
                multiple studies.
            n_variants: The number of variants in each partition used to compute the LD scores.
                Each dimension may specify its own number of relevant variants, in which case a 1-d
                array of  length :math:`D` should be provided. An integer can also be provided, in
                which case it is   expanded into a 1-d array of length :math:`D` with a constant
                value.
            ld_weighting: An optional weighting parameter used for computing the IRLS weights. If
                None, it is defined as the row-sums of the data matrix `X`. (``default=None``)

        Returns:
            The model object with estimated coefficients.

        References:
            [1] Finucane, H.K., Bulik-Sullivan, B., Gusev, A., Trynka, G., Reshef, Y., Loh, P.R.,
                Anttila, V., Xu, H., Zang, C., Farh, K. and Ripke, S., 2015. Partitioning
                heritability by functional annotation using genome-wide association summary
                statistics. Nature genetics, 47(11), pp.1228-1235.
        """
        (n, d) = X.shape
        n_samples = self.check_n_samples(n_samples, n_observations=n)
        n_variants = self.check_n_variants(n_variants, n_dimensions=d)
        ld_weighting = self.check_ld_weighting(ld_weighting, X)

        # we need to keep the condition number low
        X = reduce_condition_number(X, n_samples)

        if self.use_intercept:
            X = np.hstack((np.ones((n, 1)), X))

        # the weight update function can be provided arbitrarily -- if None, the default will be
        # used, which can be set when initialising the LDScoreRegression object
        update_fn = self.default_update_fn(**remove_self_kw(locals()))
        initial_weights = self.compute_initial_weights(**remove_self_kw(locals()))

        # the implementation of IRLS in the original repo hard-codes 2 iterations of re-weighting
        # and notes a cryptic comment of 'update this later' -- we can assume that the writers were
        # hoping to make the # iterations a parameter; we allow arbitrary setting of maxiter, but
        # will default to 2 (see __init__)
        biased_coefficient_estimate, weights = iteratively_reweighted_least_squares(
            X,
            y,
            maxiter=self.max_irls_iter,
            initial_weights=initial_weights,
            weight_update=update_fn,
            return_weights=True,
        )
        jack_knife = JackKnife(n_blocks=self.n_jack_knife_blocks, n_observations=n, sep=self.sep)
        try:
            jack_knife_summary = jack_knife.fit_transform(
                X, y, weights, biased_estimate=biased_coefficient_estimate
            )
        except np.linalg.LinAlgError:
            # we may encounter singular matrices because we have very sparse LD scores in one of
            # the columns
            # so we will estimate the errors in the classical way, which is an underestimate, but
            # the best we can do...
            w = np.sqrt(weights)
            X_ = X * w.reshape(-1, 1)
            y_ = y * w
            Q, R = np.linalg.qr(X_)
            C = np.linalg.inv(R.T @ R)
            b = C @ R.T @ Q.T @ y_
            # assuming full rank
            cov = C * np.sum((y_ - X_ @ b) ** 2) / (n - d)
            v = np.diag(cov)
            se = np.sqrt(v)
            jack_knife_summary = JackKnifeSummary(
                estimate=b.ravel(), variance=v, standard_error=se, covariance=cov
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.apply_fit_fn_side_effects(
                biased_coefficient_estimate, jack_knife, jack_knife_summary, n_samples, n_variants
            )
        return self

    def apply_fit_fn_side_effects(
        self,
        biased_estimate: np.ndarray,
        jack_knife: JackKnife,
        jack_knife_summary: JackKnifeSummary,
        n_samples: np.ndarray,
        n_variants: np.ndarray,
    ) -> None:
        warnings.warn("manually applying fit function side effects is unexpected", UserWarning)
        self._biased_estimate = biased_estimate
        self._jack_knife = jack_knife
        self._jack_knife_summary = jack_knife_summary
        self._n_bar = np.mean(n_samples)
        self._n_vars = n_variants

    @instance_has_attr("_biased_estimate", _ex)
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Once the model is fit, it can be used to predict association statistics from LD scores.

        Args:
            X: The (partitioned) LD scores. An array of shape :math:`(N, D)`, where :math:`N` is
                the number of SNPs and :math:`D` is the number of partitions.

        Returns:
            The association statistics. A 1-d array of floats of length :math:`N`.
        """
        (n, _) = X.shape
        if self.use_intercept:
            X = np.hstack((np.ones((n, 1)), X))

        out = X @ self._biased_estimate
        return out

    @abc.abstractmethod
    def _update_fn(self, fit_fn_scope: _FitFunctionScope) -> Callable:
        # we can maximize flexibility by exposing the entire fit function scope to these methods,
        # however this feels like bad practice
        pass

    @abc.abstractmethod
    def _init_weights(self, fit_fn_scope: _FitFunctionScope) -> np.ndarray:
        # abstract method allows us to define different initial weighting schemes for subclasses
        pass

    @cast_kw_to_fn_scope
    def compute_initial_weights(self, fit_fn_scope: _FitFunctionScope) -> np.ndarray:
        # this is a thin wrapper method for `_init_weights`, using the decorator to create a
        # _FitFunctionScope object and slices the `X` in order to account for the intercept. This
        # is common in all subclasses, hence this wrapper is kept in the base.
        fit_fn_scope.X = fit_fn_scope.X[..., int(self.use_intercept) :]
        out = self._init_weights(fit_fn_scope)
        return out

    @cast_kw_to_fn_scope
    def default_update_fn(self, fit_fn_scope: _FitFunctionScope) -> Callable:
        # this condition gives us the option to use the weight update fn provided at init time
        # should we allow this?
        if callable(self.weight_update_fn):
            return self.weight_update_fn

        fn = self._update_fn(fit_fn_scope)
        return fn

    @instance_has_attr("_biased_estimate", _ex)
    def get_estimate(self):
        # convenience method to get the relevant estimate
        # we allow for the use of the jackknife estimate of the coefficients, so we need to account
        # for this
        if self.use_jack_knife_estimate:
            return self._jack_knife_summary.estimate
        return self._biased_estimate

    @property
    @instance_has_attr("_jack_knife_summary", _ex)
    def coefficients(self) -> LDSCCoefficients:
        estimate = self.get_estimate()

        # when we use the intercept for regression, we need to exclude it here (remember that the
        # first element in the coefficient vector is the intercept)
        slc = slice(int(self.use_intercept), None)
        coefficients = estimate[slc, ...] / self._n_bar
        covariance = self._jack_knife_summary.covariance[slc, slc] / self._n_bar**2
        std_err = np.sqrt(np.diag(covariance))
        out = LDSCCoefficients(
            coefficients=coefficients, covariance=covariance, standard_error=std_err
        )
        return out

    @property
    @instance_has_attr("_jack_knife_summary", _ex)
    def _measure_estimate(self) -> _LDSCMeasureEstimate:
        # a "measure" here is one of heritability; referred to as `cat` in the original repo.
        coef = self.coefficients

        measure = coef.coefficients * self._n_vars
        covariance = coef.covariance * np.sum(self._n_vars**2)
        std_err = np.sqrt(np.diag(covariance))
        out = _LDSCMeasureEstimate(measure=measure, covariance=covariance, standard_error=std_err)
        return out

    @property
    @instance_has_attr("_jack_knife_summary", _ex)
    def intercept(self) -> LDSCIntercept:
        if not self.use_intercept:
            raise ValueError("no intercept used")
        estimate = self.get_estimate()
        out = LDSCIntercept(
            intercept=estimate[:1], standard_error=self._jack_knife_summary.standard_error[:1]
        )
        return out

    @classmethod
    def from_instance(cls, model: "LDScoreRegression") -> "LDScoreRegression":
        # create fresh instances using the configuration of an existing instance -- avoid copying of
        # instance specific data (e.g. coefficients)
        out = cls(**model.config)
        return out

    @property
    @instance_has_attr("_jack_knife", _ex)
    def partial_estimates(self):
        # in certain scenarios, partial estimates need to be accessed from outside
        return self._jack_knife.partial_estimates


class TwoStepFitter(CheckFitParamMixin):
    """Fits univariate LDSC in two steps.

    This approach first fits an LD score regression to find the intercept and then fits another LD
    score regression to find the slope.

    Args:
        model: An instance of a subclass of ``LDScoreRegression``.
        two_step_cutoff: The threshold above which the variants will not be considered for the
            intercept estimation. (``default=30.0``)
        two_step_condition: An optional callable taking an array and returning a boolean array
            of the same shape. If None, a default is chosen based on the model instance type.
            (``default=None``)
    """

    _biased_estimate: np.ndarray
    _jack_knife: JackKnife
    _jack_knife_summary: JackKnifeSummary
    _n_samples: np.ndarray
    _n_vars: np.ndarray
    _ex: ModelNotFitException = ModelNotFitException(
        "method not viable as model has not yet been fit"
    )

    def __init__(
        self,
        model: LDScoreRegression,
        two_step_cutoff: float = 30.0,
        two_step_condition: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        model_type = type(model)
        if not issubclass(model_type, LDScoreRegression):
            raise TypeError(
                f"`model` must be a subclass of `LDScoreRegression`; "
                f"{repr(model_type.__name__)} has no such base"
            )

        # instantiate new instances to avoid having models that have already been fit
        self._model_step_1 = model_type.from_instance(model)
        self._model_step_2 = model_type.from_instance(model)
        self._model_step_2.use_intercept = False
        self.model_type = model_type

        def default_two_step(y: np.ndarray):
            return y < two_step_cutoff

        self.two_step_condition = two_step_condition or default_two_step

    @cast_kw_to_fn_scope
    def _filter_dataset_on_cutoff(self, fit_fn_scope: _FitFunctionScope) -> _FitFunctionScope:
        # all variables within a _FitFunctionScope object (except n_variants) have length n_snp in
        # the first dimension and hence need to be subset according to the two_step_condition.
        idx = self.two_step_condition(fit_fn_scope.y)
        for name, value in remove_kw(fit_fn_scope.__dict__, remove=("n_variants",)).items():
            fit_fn_scope.__setattr__(name, value[idx])
        return fit_fn_scope

    @harmonise_inputs_to_backend
    @check_fit_fn_arrays(allow_multiple_outputs=False)
    @enforce_data_dim(n_dim=1)  # we only allow non-partitioned LDSC
    @add_docstring(LDScoreRegression.fit.__doc__)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_samples: int | float | np.ndarray,
        n_variants: int | float | np.ndarray,
        **kwargs,
    ) -> "TwoStepFitter":
        (n, d) = X.shape
        n_samples = self.check_n_samples(n_samples, n_observations=n)
        n_variants = self.check_n_variants(n_variants, n_dimensions=d)
        ld_weighting = self.check_ld_weighting(ld_weighting=None, ld_scores=X)

        step_1_data = self._filter_dataset_on_cutoff(**remove_self_kw(locals()))
        step_1 = self._model_step_1.fit(
            step_1_data.X,
            step_1_data.y,
            n_samples=step_1_data.n_samples,
            n_variants=step_1_data.n_variants,
            ld_weighting=step_1_data.ld_weighting,
            **kwargs,
        )

        itc = step_1.intercept
        y_ = y - itc.intercept

        self._model_step_2.constrained_intercept = itc.intercept
        self._model_step_2.sep = self._update_separators(self._model_step_1._jack_knife.sep, y)
        step_2 = self._model_step_2.fit(
            X, y_, n_samples=n_samples, n_variants=n_variants, ld_weighting=ld_weighting, **kwargs
        )
        factor_weights = self._compute_factor_weights(remove_self_kw(locals()))
        factor_weights = factor_weights.reshape(-1, 1)
        factor = np.sum(factor_weights * X) / np.sum(factor_weights * X**2)

        biased_estimate = np.concatenate((itc.intercept, self._model_step_2.get_estimate()))
        jack_knife = JackKnife(n_observations=n, n_blocks=self._model_step_1.n_jack_knife_blocks)
        jack_knife.partial_estimates = self._compute_partial_estimates(
            intercept=itc.intercept,
            step_1_estimates=self._model_step_1.partial_estimates,
            step_2_estimates=self._model_step_2.partial_estimates,
            factor=factor,
        )
        jack_knife_summary = jack_knife.transform(biased_estimate)
        self.apply_fit_fn_side_effects(
            biased_estimate=biased_estimate,
            jack_knife=jack_knife,
            jack_knife_summary=jack_knife_summary,
            n_samples=n_samples,
            n_variants=n_variants,
        )
        return self

    def apply_fit_fn_side_effects(
        self,
        biased_estimate: np.ndarray,
        jack_knife: JackKnife,
        jack_knife_summary: JackKnifeSummary,
        n_samples: np.ndarray,
        n_variants: np.ndarray,
    ) -> None:
        self._biased_estimate = biased_estimate
        self._jack_knife = jack_knife
        self._jack_knife_summary = jack_knife_summary
        self._n_samples = n_samples
        self._n_vars = n_variants

    @staticmethod
    def _compute_partial_estimates(
        intercept: np.ndarray,
        step_1_estimates: np.ndarray,
        step_2_estimates: np.ndarray,
        factor: float,
    ) -> np.ndarray:
        intercept_estimates = step_1_estimates[..., :1]
        slope_estimates = step_2_estimates - factor * (step_1_estimates[..., :1] - intercept)
        out = np.hstack((intercept_estimates, slope_estimates))
        return out

    def _update_separators(self, separators: np.ndarray, y: np.ndarray) -> np.ndarray:
        # not sure why we are doing this
        idx = np.arange(len(y), dtype=int)[self.two_step_condition(y)]
        out = np.array([0, *idx[separators[1:-1]], len(y)])
        return out

    def _compute_factor_weights(self, lcs: dict) -> np.ndarray:
        self._model_step_1.use_intercept = False
        weights = self._model_step_1.compute_initial_weights(**lcs)
        self._model_step_1.use_intercept = True
        return weights

    @property
    @instance_has_attr("_biased_estimate", _ex)
    def model(self) -> LDScoreRegression:
        # convenience property for accessing the final model instance with the fit intercept and
        # coefficient
        model = self.model_type.from_instance(self._model_step_1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.apply_fit_fn_side_effects(
                biased_estimate=self._biased_estimate,
                jack_knife=self._jack_knife,
                jack_knife_summary=self._jack_knife_summary,
                n_samples=self._n_samples,
                n_variants=self._n_vars,
            )
        return model
