"""Subnet seeding strategy with per-call re-ranking on inactive actors."""

from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.strategies.base import BaseSeedingStrategy


class SubnetSeedingStrategy(BaseSeedingStrategy):
    """
    Seed deployment strategy that re-ranks actors at every seeding event.

    Unlike ``StaticSeedingStrategy``, the ranking is **not** cached.  Before
    each seeding event the strategy builds a subnet consisting only of actors
    that are both available for seeding (present in ``remaining``) and not yet
    organically activated by LTM.  The selector then ranks that shrinking
    subnet, so the choice of whom to seed next reflects the current diffusion
    state of the network.

    Timing modes are inherited from ``BaseSeedingStrategy`` (SS, SQ, SQ_R).
    ``SS`` is equivalent to ``StaticSeedingStrategy`` with ``SS`` because at
    epoch 0 all actors are still inactive.

    ``batch_size`` is ignored for ``SS``.
    ``selector_params`` forwards extra keyword arguments to the selector
    constructor (same contract as in ``StaticSeedingStrategy``).
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

        :param selector: key from ``SELECTOR_CLASSES`` identifying the ranking
            method.
        :param mode: timing mode -- ``"SS"``, ``"SQ"``, or ``"SQ_R"``.
        :param batch_size: number of actors to seed per epoch (``SQ`` and
            ``SQ_R`` only).
        :param selector_params: extra keyword arguments forwarded verbatim to
            the selector constructor.  ``None`` is equivalent to ``{}``.
        """
        super().__init__(selector, mode, batch_size, selector_params)

    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Re-rank the inactive subnet and return candidates in ranking order.

        The returned actors are objects from ``remaining`` (not from the
        subnet), so the simulator's ``remaining.remove()`` call works
        correctly.  An empty list is returned when no eligible actors remain.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: actors eligible for seeding, ordered by the fresh subnet
            ranking.
        """
        subnet = self._build_subnet(net, remaining)
        if subnet.get_actors_num() == 0:
            return []
        ranking = self._selector.actorwise(subnet)
        remaining_by_id = {a.actor_id: a for a in remaining}
        return [remaining_by_id[a.actor_id] for a in ranking if a.actor_id in remaining_by_id]
