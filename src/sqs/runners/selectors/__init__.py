"""Selector registry and network-state helpers for seeding strategies."""

import logging
from functools import partial

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.seeding import (
    DegreeCentralityDiscountSelector,
    DegreeCentralitySelector,
    DriverActorSelector,
    NeighbourhoodSizeDiscountSelector,
    NeighbourhoodSizeSelector,
    RandomSeedSelector,
    VoteRankMLNSeedSelector,
)

_logger = logging.getLogger(__name__)

SELECTOR_CLASSES: dict[str, type] = {
    "deg-c": DegreeCentralitySelector,
    "deg-c-d": DegreeCentralityDiscountSelector,
    "nghb-s": NeighbourhoodSizeSelector,
    "nghb-sd": NeighbourhoodSizeDiscountSelector,
    "random": RandomSeedSelector,
    "voterank": VoteRankMLNSeedSelector,
    "mds": partial(DriverActorSelector, method=None),  # method required; None = pure MDS
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
