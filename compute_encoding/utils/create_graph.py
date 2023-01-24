import networkx as nx
import pandas as pd
from datetime import date
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from .read_log import read_log


def create_graph(log) -> nx.Graph:
    """
    Creates a graph using the pm4py library and converts to a networkx DiGraph

    Parameters
    -----------------------
    import_path: str,
        Path and file name to be imported
    Returns
    -----------------------
    graph: nx.DiGraph()
        A graph generated from the event log (includes edge weights based on transition occurrences)
    """
    graph = nx.Graph()

    log.insert(3, "time:timestamp", date.today())
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    dfg = dfg_discovery.apply(log)
    for edge in dfg:
        graph.add_weighted_edges_from([(edge[0], edge[1], dfg[edge])])

    return graph
