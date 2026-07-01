"""CLI entrypoint for the sqs simulation pipeline."""

from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from sqs import utils
from sqs.pipeline import net_stats, spreading

_RUNNERS = {
    "net_stats": net_stats.run,
    "spreading": spreading.run,
}

_CONFIGS_DIR = str(Path(__file__).parent.parent.parent / "configs")


@hydra.main(version_base=None, config_path=_CONFIGS_DIR, config_name="example_net_stats")
def main(cfg: DictConfig) -> None:
    """Read a Hydra config and dispatch to the appropriate pipeline stage."""
    utils.logger.info("git SHA: %s", utils.get_recent_git_sha())

    config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    utils.logger.info("Loaded config: %s", config)

    experiment_type = config["run"]["experiment_type"]
    runner = _RUNNERS.get(experiment_type)
    if runner is None:
        raise ValueError(
            f"Unknown experiment_type {experiment_type!r}. Available: {list(_RUNNERS)}"
        )

    runner(config)
