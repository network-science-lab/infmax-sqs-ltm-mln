"""Loaders for multilayer networks stored in the dataset."""

from functools import wraps
from pathlib import Path
from typing import Callable

import network_diffusion as nd
import networkx as nx
import pandas as pd

from sqs.loaders.constants import (
    ARXIV_NETSCIENCE_COAUTHORSHIP,
    ARXIV_NETSCIENCE_COAUTHORSHIP_MATH,
    AUCS,
    CANNES,
    CKM_PHYSICIANS,
    ER1,
    ER2,
    ER3,
    ER5,
    EU_TRANSPORT_KLM,
    EU_TRANSPORTATION,
    FMRI74,
    L2_COURSE_NET_1,
    L2_COURSE_NET_2,
    L2_COURSE_NET_3,
    LAZEGA,
    MLN_2ND_STAGE_PREFIX,
    MLN_RAW_DATA_PATH,
    SF1,
    SF2,
    SF3,
    SF5,
    TIMIK1Q2009,
    TOY_NETWORK_CIM,
    TOY_NETWORK_PIOTR,
)
from sqs.loaders.fmri74 import read_fmri74


def _network_from_pandas(path: Path) -> nd.MultilayerNetwork:
    """
    Build a MultilayerNetwork from a CSV edge list with columns node_1, node_2, layer.

    :param path: path to the CSV file.

    :returns: MultilayerNetwork with one layer per unique value in the layer column.
    """
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {
        l_name: nx.from_pandas_edgelist(group, source="node_1", target="node_2")
        for l_name, group in df.groupby("layer")
    }
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_dict.keys()), network_list=list(net_dict.values())
    )


def return_some_layers(get_network_func: Callable) -> Callable:
    """Wrap a network loader to optionally filter its result to a subset of layers."""

    @wraps(get_network_func)
    def wrapper(layer_slice: list[str] | None = None) -> nd.MultilayerNetwork:
        net = get_network_func()
        if not layer_slice:
            return net
        l_graphs = [net.layers[layer] for layer in layer_slice]
        return nd.MultilayerNetwork.from_nx_layers(l_graphs, layer_slice)

    return wrapper


def get_aucs_network() -> nd.MultilayerNetwork:
    """Return the AUCS multilayer network."""
    return nd.MultilayerNetwork.from_mpx(
        file_path=str(MLN_RAW_DATA_PATH / "small_real" / "aucs.mpx")
    )


def get_ckm_physicians_network() -> nd.MultilayerNetwork:
    """Return the CKM Physicians Innovation multilayer network."""
    return _network_from_pandas(
        MLN_RAW_DATA_PATH / "small_real" / "CKM-Physicians-Innovation_4NoNature.edges"
    )


@return_some_layers
def get_eu_transportation_network() -> nd.MultilayerNetwork:
    """Return the EU Air Transportation multilayer network."""
    return _network_from_pandas(
        MLN_RAW_DATA_PATH / "small_real" / "EUAirTransportation_multiplex_4NoNature.edges"
    )


def get_lazega_network() -> nd.MultilayerNetwork:
    """Return the Lazega Law Firm multilayer network."""
    return _network_from_pandas(
        MLN_RAW_DATA_PATH / "small_real" / "Lazega-Law-Firm_4NoNatureNoLoops.edges"
    )


def get_er2_network() -> nd.MultilayerNetwork:
    """Return a 2-layer Erdos-Renyi synthetic multilayer network."""
    return nd.MultilayerNetwork.from_mpx(
        file_path=str(MLN_RAW_DATA_PATH / "small_artificial" / "er_2.mpx")
    )


def get_er3_network() -> nd.MultilayerNetwork:
    """Return a 3-layer Erdos-Renyi synthetic multilayer network."""
    return nd.MultilayerNetwork.from_mpx(
        file_path=str(MLN_RAW_DATA_PATH / "small_artificial" / "er_3.mpx")
    )


@return_some_layers
def get_er5_network() -> nd.MultilayerNetwork:
    """Return a 5-layer Erdos-Renyi synthetic multilayer network."""
    return nd.MultilayerNetwork.from_mpx(
        file_path=str(MLN_RAW_DATA_PATH / "small_artificial" / "er_5.mpx")
    )


def get_sf2_network() -> nd.MultilayerNetwork:
    """Return a 2-layer scale-free synthetic multilayer network."""
    return nd.MultilayerNetwork.from_mpx(
        file_path=str(MLN_RAW_DATA_PATH / "small_artificial" / "sf_2.mpx")
    )


def get_sf3_network() -> nd.MultilayerNetwork:
    """Return a 3-layer scale-free synthetic multilayer network."""
    return nd.MultilayerNetwork.from_mpx(
        file_path=str(MLN_RAW_DATA_PATH / "small_artificial" / "sf_3.mpx")
    )


@return_some_layers
def get_sf5_network() -> nd.MultilayerNetwork:
    """Return a 5-layer scale-free synthetic multilayer network."""
    return nd.MultilayerNetwork.from_mpx(
        file_path=str(MLN_RAW_DATA_PATH / "small_artificial" / "sf_5.mpx")
    )


def get_ddm_network(
    layernames_path: Path,
    edgelist_path: Path,
    weighted: bool,
    digraph: bool,
) -> nd.MultilayerNetwork:
    """
    Build a MultilayerNetwork from a DDM-format layer-name file and multiplex edge list.

    :param layernames_path: text file mapping integer layer IDs to layer names (one per line,
        space-separated: ``<id> <name>``).
    :param edgelist_path: space-separated edge list with columns layer_id, node_1, node_2, weight.
    :param weighted: if True, copy the weight column onto each edge as an attribute.
    :param digraph: if True, create directed graphs per layer; otherwise undirected.

    :returns: MultilayerNetwork constructed from the supplied edge list.
    """
    with layernames_path.open(encoding="utf-8") as fh:
        layer_names: dict[str, str] = {
            parts[0]: parts[1] for line in fh for parts in [line.rstrip("\n").split(" ")]
        }

    df = pd.read_csv(
        edgelist_path,
        names=["layer_id", "node_1", "node_2", "weight"],
        sep=" ",
        dtype={"node_1": str, "node_2": str},
    )
    net_ids_dict: dict[int, nx.Graph] = {
        l_id: nx.DiGraph() if digraph else nx.Graph() for l_id in df["layer_id"].unique()
    }
    for row in df.itertuples(index=False):
        attrs = {"weight": row.weight} if weighted else {}
        net_ids_dict[row.layer_id].add_edge(row.node_1, row.node_2, **attrs)

    net_names_dict = {
        layer_names[str(layer_id)]: layer_graph for layer_id, layer_graph in net_ids_dict.items()
    }
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=list(net_names_dict.keys()), network_list=list(net_names_dict.values())
    )


@return_some_layers
def get_arxiv_network() -> nd.MultilayerNetwork:
    """Return the arXiv Network Science coauthorship multilayer network."""
    root_path = MLN_RAW_DATA_PATH / "arxiv_netscience_coauthorship" / "Dataset"
    return get_ddm_network(
        layernames_path=root_path / "arxiv_netscience_layers.txt",
        edgelist_path=root_path / "arxiv_netscience_multiplex.edges",
        weighted=False,
        digraph=False,
    )


def get_cannes_network() -> nd.MultilayerNetwork:
    """Return the Cannes 2013 social multilayer network."""
    root_path = MLN_RAW_DATA_PATH / "cannes_2013_social" / "Dataset"
    return get_ddm_network(
        layernames_path=root_path / "Cannes2013_layers.txt",
        edgelist_path=root_path / "Cannes2013_multiplex.edges",
        weighted=False,
        digraph=False,
    )


def get_timik1q2009_network() -> nd.MultilayerNetwork:
    """Return the TIMIK 1Q2009 multilayer network."""
    layer_names, layer_graphs = [], []
    for csv_file in (MLN_RAW_DATA_PATH / "timik1q2009").glob("*.csv"):
        layer_names.append(csv_file.stem)
        layer_graphs.append(nx.from_pandas_edgelist(pd.read_csv(csv_file)))
    return nd.MultilayerNetwork.from_nx_layers(layer_graphs, layer_names)


def load_network(net_name: str) -> nd.MultilayerNetwork:
    """
    Return the multilayer network identified by net_name.

    :param net_name: name of the network; must match one of the constants in
        ``sqs.loaders.constants`` or start with ``MLN_2ND_STAGE_PREFIX``.

    :returns: loaded MultilayerNetwork ready for simulation.
    """
    if net_name == FMRI74:
        return read_fmri74(network_dir=MLN_RAW_DATA_PATH / "CONTROL_fmt", binary=True, thresh=0.5)
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP:
        return get_arxiv_network()
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP_MATH:
        return get_arxiv_network(["math.OC"])
    elif net_name == AUCS:
        return get_aucs_network()
    elif net_name == CANNES:
        return get_cannes_network()
    elif net_name == CKM_PHYSICIANS:
        return get_ckm_physicians_network()
    elif net_name == EU_TRANSPORTATION:
        return get_eu_transportation_network()
    elif net_name == EU_TRANSPORT_KLM:
        return get_eu_transportation_network(["KLM"])
    elif net_name == L2_COURSE_NET_1:
        return nd.nets.get_l2_course_net(
            node_features=True, edge_features=True, directed=False
        ).snaps[0]
    elif net_name == L2_COURSE_NET_2:
        return nd.nets.get_l2_course_net(
            node_features=True, edge_features=True, directed=False
        ).snaps[1]
    elif net_name == L2_COURSE_NET_3:
        return nd.nets.get_l2_course_net(
            node_features=True, edge_features=True, directed=False
        ).snaps[2]
    elif net_name == LAZEGA:
        return get_lazega_network()
    elif net_name == ER1:
        return get_er5_network(["l2"])
    elif net_name == ER2:
        return get_er2_network()
    elif net_name == ER3:
        return get_er3_network()
    elif net_name == ER5:
        return get_er5_network()
    elif net_name == SF1:
        return get_sf5_network(["l3"])
    elif net_name == SF2:
        return get_sf2_network()
    elif net_name == SF3:
        return get_sf3_network()
    elif net_name == SF5:
        return get_sf5_network()
    elif net_name == TIMIK1Q2009:
        return get_timik1q2009_network()
    elif net_name == TOY_NETWORK_CIM:
        return nd.nets.get_toy_network_cim()
    elif net_name == TOY_NETWORK_PIOTR:
        return nd.nets.get_toy_network_piotr()
    elif net_name.startswith(MLN_2ND_STAGE_PREFIX):
        mpx_path = Path(net_name.replace("-", "/") + ".mpx")
        return nd.MultilayerNetwork.from_mpx(str(mpx_path))
    raise AttributeError(f"Unknown network: {net_name}")
