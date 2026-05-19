"""Pipeline stage: load all configured networks and report basic statistics."""

from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from sqs.loaders.net_loader import load_network
from sqs import utils


def _summarise(net_name: str, net: Any) -> str:
    """
    Format a one-line summary of a loaded MultilayerNetwork.

    :param net_name: key used to identify the network in the config.
    :param net: loaded MultilayerNetwork instance.

    :returns: human-readable summary string.
    """
    layer_names = net.get_layer_names()
    actors = net.get_actors_num()
    edge_counts = " | ".join(
        f"{ln}: {net.layers[ln].number_of_edges()} edges" for ln in layer_names
    )
    return f"{net_name}  layers={len(layer_names)}  actors={actors}  [{edge_counts}]"


def run(config: dict[str, Any]) -> None:
    """
    Load every network listed in config and persist the stamped config to disk.

    The config is written to ``<out_dir>/config.yaml`` with an additional
    ``git_sha`` field recording the HEAD commit of the repository at run time.

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
    loaded = []

    for net_name in tqdm(network_names, desc="Loading networks", colour="green"):
        net = load_network(net_name)
        loaded.append((net_name, net))
        tqdm.write(f"  {_summarise(net_name, net)}")

    finish_time = utils.get_current_time()
    print(f"\nLoaded {len(loaded)}/{len(network_names)} networks successfully.")
    print(f"Pipeline finished at {finish_time}")
