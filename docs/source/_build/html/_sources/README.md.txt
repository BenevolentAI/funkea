# funkea

Perform functional enrichment analysis at scale.

## Install

```shell
pip install funkea
```

## Quickstart

`funkea` was built with composability in mind, but also ships 5 popular enrichment methods out
of the box. The simplest and fastest method is a Fisher's exact test on the overlapped annotations:

```python
from funkea.implementations import Fisher
from pyspark.sql import DataFrame

sumstats: DataFrame = ...  # your GWAS sumstats

# perform tissue enrichment using GTEx data
model = Fisher.default()
enrichment = model.transform(sumstats)
```

This assumes that the default filepaths are set in the file-registry.

## Introduction

`funkea` is a Python library for large-scale functional enrichment analysis. It provides 5 popular
enrichment methods, and also allows for experimentation by composing different components. It is
written in Spark, and allows users to run an arbitrary number of GWAS studies concurrently, given
the resources are available.

### Concepts

`funkea` has a few concepts used for abstraction, such that all methods could be unified. A view of
the schematic is outlined below

![schematic](_static/schematic.png)

## Setting default filepaths

For ease of use, `funkea` uses a file-registry for its source of truth of various data sources. These
need to be set by a user, which can be set easily by using the `funkea-cli`. For example, if the
user has acquired the data provided by BAI, the registry can be set like so:

```shell
funkea-cli init --from-stem <PATH_TO_DATASET> 
```
