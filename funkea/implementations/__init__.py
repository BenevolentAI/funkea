"""A set of popular functional enrichment method implementations.

This sub-package contains implementations of popular functional enrichment methods. Each
implementation is a class that inherits from the abstract base class :class:`.Workflow`. It is
recommended to use the implementations through the :class:`.Workflow` class, which provides a
consistent interface to all implementations. Moreover, it is recommended to use the default
configurations of these methods, which can be accessed via the :func:`.Workflow.default` method.

The following implementations are currently available:

* :class:`.DEPICT`: DEPICT (Data-driven Expression-Prioritized Integration for Complex Traits)
  is a method that integrates GWAS summary statistics with gene expression data to identify
  gene sets that are enriched for trait-associated variants.

* :class:`.Fisher`: Fisher's exact test is a statistical test used to determine if there are
  non-random associations between two categorical variables.

* :class:`.GARFIELD`: GARFIELD (GWAS Analysis of Regulatory or Functional Information Enrichment
  with LD correction) is a method that integrates GWAS summary statistics with functional
  annotations to identify gene sets that are enriched for trait-associated variants.

* :class:`.LDSC`: LDSC (Linkage Disequilibrium Score Regression) is a method that uses GWAS
  summary statistics to estimate heritability and genetic correlation.

* :class:`.SNPsea`: SNPsea is a method that integrates GWAS summary statistics with functional
  annotations to identify gene sets that are enriched for trait-associated variants.
"""

from funkea.implementations.depict import DEPICT
from funkea.implementations.fisher import Fisher
from funkea.implementations.garfield import GARFIELD
from funkea.implementations.ldsc import LDSC
from funkea.implementations.snpsea import SNPsea

__all__ = [
    "DEPICT",
    "Fisher",
    "GARFIELD",
    "LDSC",
    "SNPsea",
]
