"""Perform functional enrichment analysis at scale.

Examples:
    >>> from pyspark.sql import DataFrame
    >>> from funkea.implementations import Fisher
    >>>
    >>> # your GWAS sumstats dataframe
    >>> sumstats: DataFrame
    >>>
    >>> # takes the default configuration -- this will perform tissue enrichment on GTEx data
    >>> model = Fisher.default()
    >>> enrichment = model.transform(sumstats)
"""

from funkea.core.utils.version import __version__

__all__ = ["__version__"]
