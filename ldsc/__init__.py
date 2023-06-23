"""Python 3 implementation of LD score regression.

It implements the heritability version of LDSC [1]_[2]_. It also offers optional GPU accelaration
via JAX.

Examples:
    A simple example using the heritability regression on simulated data:

    >>> import numpy as np
    >>>
    >>> # simulate data
    >>> h2_1, h2_2 = 0.2, 0.7
    >>> N, M = 1e5, 1e7 / 2
    >>> X = np.abs(np.random.randn(100, 1) + 1)
    >>> y = 1 + 1e5 * (X[:, 0] * h2_1 / M + X[:, 1] * h2_2 / M)
    >>>
    >>> # fit model
    >>> model = HeritabilityRegression(n_jack_knife_blocks=5, use_intercept=True)
    >>> model = model.fit(X, y, n_samples=N, n_variants=M)

    The package provides some convenience interfaces for the model classes. For example, we can
    estimate the heritability of a given trait simply by doing the following:

    >>> import pandas as pd
    >>> import ldsc
    >>>
    >>> # load the study in question into a pandas dataframe, with its association Z-score and LD
    ... # scores
    >>> study: pd.DataFrame = ...
    >>>
    >>> h2 = ldsc.compute_heritability_estimate(study)

    Below, we can use the Pandas UDF for easy parallel heritability estimation of multiple GWAS
    studies:

    >>> from pyspark.sql import DataFrame
    >>> import ldsc
    >>>
    >>> # load the studies with the Z association statistic
    >>> sumstats: DataFrame = ...
    >>>
    >>> # compute h2 for each study
    >>> h2 = (
    ...     # we need the chi-squared association statistic + a population prevalence column.
    ...     sumstats.groupBy("study_id")
    ...     .applyInPandas(compute_heritability_estimate_udf, ldsc.heritability.udf_schema())
    ... )

References:
    [1] Bulik-Sullivan, B.K., Loh, P.R., Finucane, H.K., Ripke, S., Yang, J., Patterson, N., Daly,
        M.J., Price, A.L. and Neale, B.M., 2015. LD Score regression distinguishes confounding from
        polygenicity in genome-wide association studies. Nature genetics, 47(3), pp.291-295.
    [2] GitHub repo: https://github.com/bulik/ldsc
"""

from ldsc import heritability
from ldsc.base import TwoStepFitter
from ldsc.heritability import (
    HeritabilityRegression,
    compute_heritability_estimate,
    compute_heritability_estimate_udf,
)

__all__ = [
    "HeritabilityRegression",
    "TwoStepFitter",
    "compute_heritability_estimate",
    "compute_heritability_estimate_udf",
    "heritability",
]
