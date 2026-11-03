"""Selector registry and network-state helpers for seeding strategies."""

import logging
from collections.abc import Callable
from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.seeding import RandomSeedSelector

from sqs.runners.selectors.deterministic import (
    MdsSelector,
    VoteRankSelector,
    degree_centrality,
    degree_discount,
    neighbourhood_size_centrality,
    neighbourhood_size_discount,
)

_logger = logging.getLogger(__name__)

# Selector factories (classes or functions); rankings break ties by actor id
SELECTOR_CLASSES: dict[str, Callable[..., Any]] = {
    "deg-c": degree_centrality,
    "deg-c-d": degree_discount,
    "nghb-s": neighbourhood_size_centrality,
    "nghb-sd": neighbourhood_size_discount,
    "random": RandomSeedSelector,
    "voterank": VoteRankSelector,
    "mds": MdsSelector,
    # MDS with degree centrality as secondary ranking criterion
    "mds+deg": lambda **kw: MdsSelector(base=degree_centrality(), **kw),
}


try:
    from sqs.runners.selectors.ts_net import TsNetSeedSelector

    SELECTOR_CLASSES["ts-net"] = TsNetSeedSelector
except ImportError:
    _logger.debug(
        "torch or torch-geometric not installed -- 'ts-net' selector unavailable. "
        "Install with: uv add torch torch-geometric"
    )


def is_already_active(actor: MLNetworkActor, net: MultilayerNetwork) -> bool:
    """Return True if the actor is active in any of its layers in the current network state.

    :param actor: actor to check.
    :param net: live multilayer network whose node attributes are queried.

    :returns: True when the actor has ``ACTIVE_STATE`` status in at least one layer.
    """
    return any(
        net.layers[layer].nodes[actor.actor_id].get("status") == MLTModel.ACTIVE_STATE
        for layer in actor.layers
    )
