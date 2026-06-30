"""Adaptive seeding strategy based on structural activation-frontier scoring."""

from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.mlt_model import MLTModel

from sqs.runners.selectors import is_already_active
from sqs.runners.strategies.base import BaseSeedingStrategy


class AdaptiveSeedingStrategy(BaseSeedingStrategy):
    """
    Greedy strategy that seeds actors on the activation frontier.

    At every seeding event the strategy proceeds in two phases:

    **Phase 1 -- structural filter (frontier candidates).**
    Actors in ``remaining`` that are not yet organically active and have at
    least one active neighbour in any layer constitute the *activation
    frontier*.  These are the actors closest to organic activation, so seeding
    them is most likely to amplify the ongoing cascade.

    **Phase 2 -- activation-potential ranking.**
    Each frontier candidate is scored by its *activation potential*::

        score(a) = active_neighbours(a) / total_degree(a)

    across all layers.  A higher score indicates that a larger fraction of the
    actor's social environment is already spreading the information, so seeding
    that actor would both activate it immediately and put its inactive neighbours
    closer to their own threshold.  Candidates are returned in descending score
    order.

    **Fallback -- selector on inactive subnet.**
    When no frontier candidates exist (typically at epoch 0, before any seeds
    have been planted, or when the frontier is exhausted), the strategy falls
    back to running ``selector.actorwise`` on the subnet of inactive actors --
    identical to ``SubnetSeedingStrategy``.

    Timing modes (SS, SQ, SQ_R) and the ``selector`` / ``batch_size`` /
    ``selector_params`` parameters are inherited from ``BaseSeedingStrategy``
    and behave identically to the other strategies.
    """

    def __init__(
        self,
        selector: str,
        mode: str,
        batch_size: int = 1,
        selector_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the strategy.

        :param selector: key from ``SELECTOR_CLASSES`` used as a fallback
            ranker when no frontier candidates are available.
        :param mode: timing mode -- ``"SS"``, ``"SQ"``, or ``"SQ_R"``.
        :param batch_size: number of actors to seed per epoch (``SQ`` and
            ``SQ_R`` only).
        :param selector_params: extra keyword arguments forwarded verbatim to
            the selector constructor.  ``None`` is equivalent to ``{}``.
        """
        super().__init__(selector, mode, batch_size, selector_params)

    def _activation_potential(self, actor: MLNetworkActor, net: MultilayerNetwork) -> float:
        """
        Compute the activation potential of an actor as active-neighbour ratio.

        :param actor: candidate actor to score.
        :param net: current state of the multilayer network.

        :returns: fraction of the actor's neighbours (across all layers) that
            are currently active; 0.0 when the actor has no neighbours.
        """
        active = 0
        total = 0
        for layer in actor.layers:
            graph = net.layers[layer]
            neighbours = list(graph.neighbors(actor.actor_id))
            total += len(neighbours)
            active += sum(
                1 for nb in neighbours if graph.nodes[nb].get("status") == MLTModel.ACTIVE_STATE
            )
        return active / total if total > 0 else 0.0

    def _frontier_candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Return remaining, inactive actors that border at least one active node.

        Actors are taken directly from ``remaining`` so the objects are
        compatible with the simulator's ``remaining.remove()`` call.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: frontier actors (at least one active neighbour), in their
            original order inside ``remaining``.
        """
        result = []
        for actor in remaining:
            if is_already_active(actor, net):
                continue
            on_frontier = any(
                any(
                    net.layers[layer].nodes[nb].get("status") == MLTModel.ACTIVE_STATE
                    for nb in net.layers[layer].neighbors(actor.actor_id)
                )
                for layer in actor.layers
            )
            if on_frontier:
                result.append(actor)
        return result

    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Return actors to consider for seeding, ordered by preference.

        Tries the frontier filter first; falls back to selector-based subnet
        ranking when no frontier candidates exist.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: eligible actors in descending preference order.
        """
        frontier = self._frontier_candidates(net, remaining)
        if frontier:
            return sorted(
                frontier,
                key=lambda a: self._activation_potential(a, net),
                reverse=True,
            )

        # Fallback: subnet re-ranking via selector (same as SubnetSeedingStrategy).
        subnet = self._build_subnet(net, remaining)
        if subnet.get_actors_num() == 0:
            return []
        ranking = self._selector.actorwise(subnet)
        remaining_by_id = {a.actor_id: a for a in remaining}
        return [remaining_by_id[a.actor_id] for a in ranking if a.actor_id in remaining_by_id]
