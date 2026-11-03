"""Subnet seeding strategy: rank the actors still eligible for seeding during the run."""

from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.strategies.base import ActorPool, BaseSeedingStrategy


class SubnetSeedingStrategy(BaseSeedingStrategy):
    """
    Seed by the selector's ranking of the subnet of actors neither seeded nor active.

    The subnet is re-ranked each time ``rerank_percent`` % of the budget has been seeded
    (0: before every seed), so the choice follows the diffusion state. SS equals Static SS.
    """

    def __init__(
        self,
        selector: str,
        mode: str,
        batch_size: int = 1,
        selector_params: dict[str, Any] | None = None,
        rerank_percent: float = 1.0,
    ) -> None:
        """Initialise as ``BaseSeedingStrategy``."""
        super().__init__(selector, mode, batch_size, selector_params)
        self._rerank_percent = rerank_percent

    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """Return the eligible actors at the head of the last subnet ranking."""
        pool = remaining if isinstance(remaining, ActorPool) else ActorPool(remaining)
        if not pool:
            return []
        if self._rerank_due(self._rerank_percent):
            self._set_ranking(self._selector.actorwise(self._build_subnet(net, remaining)))
        return self._ranked(pool, len(pool) if self._mode == "SS" else self._batch_size)
