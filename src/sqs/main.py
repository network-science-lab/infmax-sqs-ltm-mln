"""CLI entrypoint for the sqs simulation pipeline."""

import argparse

import yaml

from sqs.pipeline import net_stats


def parse_args(*args: str) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param args: argument strings (defaults to sys.argv when empty).

    :returns: parsed namespace with a ``config`` path attribute.
    """
    parser = argparse.ArgumentParser(description="Run a simulation pipeline stage.")
    parser.add_argument(
        "config",
        nargs="?",
        default="scripts/configs/example_net_stats.yaml",
        help="Path to the YAML pipeline config.",
    )
    return parser.parse_args(*args)


_RUNNERS = {
    "net_stats": net_stats.run,
}


def main() -> None:
    """Read a YAML config and dispatch to the appropriate pipeline stage."""
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print(f"Loaded config: {config}")

    experiment_type = config["run"]["experiment_type"]
    runner = _RUNNERS.get(experiment_type)
    if runner is None:
        raise ValueError(
            f"Unknown experiment_type {experiment_type!r}. "
            f"Available: {list(_RUNNERS)}"
        )

    runner(config)
