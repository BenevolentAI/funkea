"""Collection of utility functions used widely across the codebase."""
import functools
import itertools
from pathlib import Path
from typing import Iterator, TypeVar

import networkx as nx
import pandas as pd
from pyspark.sql import Column, SparkSession

from funkea.core.utils import files, partition

T = TypeVar("T")


def _overlap_point(start: Column, end: Column, point: Column) -> Column:
    return (start <= point) & (point <= end)


def _overlap_range(start_a: Column, end_a: Column, start_b: Column, end_b: Column) -> Column:
    return (
        _overlap_point(start_a, end_a, start_b)
        | _overlap_point(start_a, end_a, end_b)
        | _overlap_point(start_b, end_b, start_a)
        | _overlap_point(start_b, end_b, end_a)
    )


def overlap(start: Column, end: Column, point: Column, point_end: Column | None = None) -> Column:
    r"""Returns a boolean column indicating whether there is an overlap.

    This function can be used for both finding overlap between a range and a point, and between two
    ranges.

    Args:
        start: The start position of the range.
        end: The end position of the range.
        point: Either a point, or the start position of the second range.
        point_end: Optional end position for the second range. If None, ``point`` will be
            interpreted as a point.

    Returns:
        Boolean column specifying whether there is an overlap.

    """
    if point_end is None:
        return _overlap_point(start, end, point)
    return _overlap_range(start, end, point, point_end)


def _collapse_graph(
    graph: nx.Graph, collections: pd.DataFrame, id_column: str, collections_column: str
) -> pd.DataFrame:
    rows = []
    for node in collections.index:
        if node not in graph.nodes:
            continue
        adj = list(nx.dfs_preorder_nodes(graph, source=node))
        graph.remove_nodes_from(adj)

        elems = collections.loc[adj, collections_column]
        elems = functools.reduce(lambda a, b: set(a).union(b), elems)
        rows.append({collections_column: list(elems), id_column: ";".join(map(str, adj))})
    return pd.DataFrame(rows)


def cross_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    r"""Cross joins two pandas dataframes.

    This is a hack to get the Spark ``crossJoin`` feature in pandas. Useful for ``applyInPandas``
    functionality, where a set of columns defines the group and then the output is produced in the
    UDF. A cross join will then associate all the group variables to the function output.

    Args:
        left: The left dataframe.
        right: the right dataframe.

    Returns:
        The cross joined dataframes.

    """
    return (
        left.assign(key=0)
        .merge(
            # hack for a cross join in pandas
            # see: https://stackoverflow.com/a/46895905
            right.assign(key=0),
            on="key",
            how="outer",
        )
        .drop(columns="key")
    )


def merge_collections(
    collections: pd.DataFrame,
    id_column: str,
    collections_column: str,
    partition_cols: partition.PartitionByType,
) -> pd.DataFrame:
    r"""Merges a set of collections if there is any overlap.

    This function is used to merge loci which overlap the same annotations
    (see ``funkea.components.locus_definition.Merge``). The idea is that if we have sets :math:`A`,
    :math:`B` and :math:`C`, then we could have the following:

    .. math::

        A \cap B &\neq \emptyset \\
        B \cap C &\neq \emptyset \\
        A \cap C &= \emptyset

    In which case, non-recursive merging might either merge :math:`A` and :math:`B`, or :math:`B`
    and :math:`C`, but not all three. Hence, the recursion is necessary to produce the full
    overlapping set.

    Args:
        collections: Dataframe containing the group columns and a column of iterables (collections).
        id_column: The column name of the unique identifiers for each collection.
        collections_column: The column name of the column containing the collections.
        partition_cols: A sequence of columns which define the group.

    Returns:
        A dataframe containing the merged collections and their concatenated IDs.

    Notes:
        This function expects to be used in a groupby-apply operation, such that
        ``collections.groupby(list(partition_cols)).ngroups == 1``. Usually, this would be grouping
        by study ID and chromosome.

    """
    graph = nx.Graph()
    collections = collections.set_index(id_column)

    graph.add_nodes_from(collections.index)
    for (i, ids_i), (j, ids_j) in itertools.combinations(
        collections[collections_column].items(), r=2
    ):
        gene_ids = set(ids_i)
        if gene_ids.intersection(ids_j):
            graph.add_edge(i, j)

    # essentially, we should assign the unique partition to all the loci
    # NOTE: this function should always be used in a groupby apply, such that the dataframe after
    # `drop_duplicates` has len == 1
    return cross_join(
        # if the sequence is not a list (e.g. tuple), it will be interpreted as a multi-index
        collections[list(partition_cols)].drop_duplicates(),
        _collapse_graph(graph, collections, id_column, collections_column).assign(key=0),
    )


def get_session() -> SparkSession:
    """Gets the SparkSession and checks it is not None."""
    session: SparkSession | None = SparkSession.getActiveSession()
    assert session is not None
    return session


def chunk(sequence: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Chunks a sequence."""
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i : (i + chunk_size)]


def jar_in_session(jar: str = files.SCALA_UDF.name) -> bool:
    """Checks whether given jar is in the active session."""
    session_jars = get_session().conf.get("spark.jars", "").split(",")
    return jar in [Path(path).name for path in session_jars]
