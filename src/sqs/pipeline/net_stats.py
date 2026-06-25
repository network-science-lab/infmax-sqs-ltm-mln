"""Pipeline stage: load all configured networks and report basic statistics."""

from pathlib import Path
from typing import Any

from tqdm import tqdm

from sqs.misc import utils
from sqs.loaders.net_loader import load_network

logger = utils.logger

_MD_TABLE_FILE = "net_stats.md"


def _summarise(net_name: str, net: Any) -> str:
    """
    Format a one-line summary of a loaded MultilayerNetwork.

    :param net_name: key used to identify the network in the config.
    :param net: loaded MultilayerNetwork instance.

    :returns: human-readable summary string.
    """
    layer_names = net.get_layer_names()
    actors = net.get_actors_num()
    edge_counts = " | ".join(f"{ln}: {net.layers[ln].number_of_edges()}" for ln in layer_names)
    return f"{net_name}  layers={len(layer_names)}  actors={actors}  [{edge_counts}]"


def _write_md_table(rows: list[tuple[str, Any]], path: Path) -> None:
    """
    Write a Markdown table of network statistics to disk.

    Columns: network name, number of layers, number of actors, total edges,
    and a per-layer edge breakdown.

    :param rows: list of ``(net_name, net)`` pairs in display order.
    :param path: destination file path.
    """
    header = "| Network | Layers | Actors | Total edges | Edges per layer |"
    separator = "|---------|--------|--------|-------------|-----------------|"
    lines = [header, separator]
    for net_name, net in rows:
        layer_names = net.get_layer_names()
        actors = net.get_actors_num()
        layer_edges = {ln: net.layers[ln].number_of_edges() for ln in layer_names}
        total = sum(layer_edges.values())
        breakdown = ", ".join(f"{ln}: {e}" for ln, e in layer_edges.items())
        lines.append(f"| {net_name} | {len(layer_names)} | {actors} | {total} | {breakdown} |")
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run(config: dict[str, Any]) -> None:
    """
    Load networks and persist a Markdown stats table to disk.

    Config snapshot and git SHA are saved by Hydra to ``<out_dir>/.hydra/``.
    The stats table is written to ``<out_dir>/net_stats.md``.

    :param config: pipeline configuration dict (see ``configs/``).
    """
    out_dir = Path(config["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pipeline started")

    network_names: list[str] = config["networks"]
    loaded: list[tuple[str, Any]] = []

    for net_name in tqdm(network_names, desc="Loading networks", colour="green"):
        net = load_network(net_name)
        loaded.append((net_name, net))
        tqdm.write(f"  {_summarise(net_name, net)}")

    _write_md_table(loaded, out_dir / _MD_TABLE_FILE)

    logger.info("Loaded %d/%d networks successfully.", len(loaded), len(network_names))
    logger.info("Stats table saved to %s", out_dir / _MD_TABLE_FILE)
    logger.info("Pipeline finished")
