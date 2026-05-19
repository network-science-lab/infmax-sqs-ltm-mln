"""Pipeline stage: load all configured networks and report basic statistics."""

from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from sqs import utils
from sqs.loaders.net_loader import load_network

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
    Load networks, print per-network stats, and persist a Markdown stats table to disk.

    The config is written to ``<out_dir>/config.yaml`` with an additional
    ``git_sha`` field recording the HEAD commit at run time. The stats table
    is written to ``<out_dir>/net_stats.md``.

    :param config: pipeline configuration dict (see ``scripts/configs/``).
    """
    out_dir = Path(config["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    config["git_sha"] = utils.get_recent_git_sha()
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False, allow_unicode=True)

    start_time = utils.get_current_time()
    print(f"\nPipeline started at {start_time}")
    print(f"Config saved to {out_dir / 'config.yaml'}\n")

    network_names: list[str] = config["networks"]
    loaded: list[tuple[str, Any]] = []

    for net_name in tqdm(network_names, desc="Loading networks", colour="green"):
        net = load_network(net_name)
        loaded.append((net_name, net))
        tqdm.write(f"  {_summarise(net_name, net)}")

    _write_md_table(loaded, out_dir / _MD_TABLE_FILE)

    finish_time = utils.get_current_time()
    print(f"\nLoaded {len(loaded)}/{len(network_names)} networks successfully.")
    print(f"Stats table saved to {out_dir / _MD_TABLE_FILE}")
    print(f"Pipeline finished at {finish_time}")
