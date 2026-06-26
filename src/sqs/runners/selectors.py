"""Selector registry and network-state helpers for seeding strategies."""

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.seeding import DegreeCentralitySelector

SELECTOR_CLASSES: dict[str, type] = {
    "deg-c": DegreeCentralitySelector,
}


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
