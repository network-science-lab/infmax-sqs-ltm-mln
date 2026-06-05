"""Loader for the network of fMRI scans of 74 healthy women in resting-state."""

from pathlib import Path

import network_diffusion as nd
import networkx as nx
import pandas as pd
from tqdm import tqdm

from sqs.utils import logger


def _parse_adj_mats(
    network_dir: Path, binary: bool, thresh: float | None
) -> dict[str, pd.DataFrame]:
    """
    Convert a directory of adjacency-matrix CSV files into a dict of edge-list DataFrames.

    :param network_dir: directory containing one CSV adjacency matrix per subject.
    :param binary: if True, replace all non-zero weights with 1.
    :param thresh: drop edges whose weight is <= this value; ignored when None.

    :returns: mapping from filename to edge-list DataFrame with columns source, target, weight.
    """
    layers: dict[str, pd.DataFrame] = {}
    for network_file in tqdm(network_dir.glob("*.csv")):
        try:
            layer = pd.read_csv(network_file, index_col=0)
            if layer.shape[0] != layer.shape[1]:
                raise ValueError(
                    "Expecting square matrix with index as source and column as target."
                )
            if thresh is not None:
                layer[layer <= thresh] = 0
            if binary:
                layer[layer != 0] = 1
            if pd.api.types.is_numeric_dtype(layer.index):
                layer.index = layer.index.map(str)
            layer = layer.stack(future_stack=True).reset_index()
            layer.columns = pd.Index(["source", "target", "weight"])
            layer = layer[~(layer["weight"] == 0.0)].reset_index(drop=True)
            layers[network_file.name] = layer
        except Exception as exc:
            logger.warning("Could not read file '%s': %s", network_file, exc)
    return layers


def _parse_edgelists(
    edgelists: dict[str, pd.DataFrame],
) -> tuple[list[str], list[nx.Graph]]:
    """
    Convert a dict of edge-list DataFrames into parallel lists of layer names and graphs.

    :param edgelists: mapping from layer name to edge-list DataFrame with a weight column.

    :returns: tuple of (layer_names, layer_graphs) ready for MultilayerNetwork.from_nx_layers.
    """
    l_names: list[str] = []
    l_graphs: list[nx.Graph] = []
    for l_name, l_edge_list in edgelists.items():
        l_names.append(l_name)
        l_graphs.append(nx.convert_matrix.from_pandas_edgelist(l_edge_list, edge_attr="weight"))
    return l_names, l_graphs


def read_fmri74(network_dir: Path, binary: bool, thresh: float | None) -> nd.MultilayerNetwork:
    """
    Read the fMRI resting-state network of 74 healthy women.

    The dataset was posted to the 1000 Functional Connectomes Project and used in
    "Analysis of Population Functional Connectivity Data via Multilayer Network Embeddings",
    Network Science, DOI: 10.1017/nws.2020.39.

    :param network_dir: directory of per-subject adjacency-matrix CSV files.
    :param binary: if True, binarise edge weights (0 or 1).
    :param thresh: discard edges with weight <= this threshold; pass None to keep all.

    :returns: MultilayerNetwork with one layer per subject CSV file.
    """
    edge_lists = _parse_adj_mats(network_dir, binary, thresh)
    l_names, l_graphs = _parse_edgelists(edge_lists)
    return nd.MultilayerNetwork.from_nx_layers(network_list=l_graphs, layer_names=l_names)
