"""Subnet seeding strategy with per-call re-ranking on inactive actors."""

import heapq
from collections import defaultdict
from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.strategies.base import ActorPool, BaseSeedingStrategy

# Selectors whose first choice on the subnet is the actor with the highest neighbour score
# (discount and VoteRank only change later positions), so SQ / SQ_R with batch_size 1 can track
# the scores incrementally instead of re-ranking the subnet at every seed. Ties: lowest id.
_FIRST_PICK_SELECTORS = {"deg-c", "deg-c-d", "nghb-s", "nghb-sd", "voterank"}


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
        self._selector_key = selector
        self.first_pick = mode != "SS" and batch_size == 1 and selector in _FIRST_PICK_SELECTORS
        self._weights: dict[Any, dict[Any, int]] | None = None
        self._scores: dict[Any, int] = {}
        self._heap: list[tuple[int, Any]] = []
        self._seen_removed = 0

    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Re-rank the inactive subnet and return candidates in ranking order.

        The returned actors are objects from ``remaining`` (not from the
        subnet).  An empty list is returned when no eligible actors remain.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: actors eligible for seeding, ordered by the fresh subnet
            ranking.
        """
        if self.first_pick:
            return self._first_candidate(net, remaining)
        subnet = self._build_subnet(net, remaining)
        if subnet.get_actors_num() == 0:
            return []
        ranking = self._selector.actorwise(subnet)
        return self._stable_candidates(ranking, remaining)

    def _neighbour_weights(self, net: MultilayerNetwork) -> dict[Any, dict[Any, int]]:
        """Per actor: neighbour id -> its contribution to the selector's first-round score."""
        weights: dict[Any, dict[Any, int]] = defaultdict(dict)
        if self._selector_key == "voterank":  # same link multiset as VoteRankSelector
            pairs = [(a.actor_id, b.actor_id) for a, b in net.get_links()]
        else:
            pairs = [(u, v) for graph in net.layers.values() for u, v in graph.edges()]
        unit = self._selector_key in ("nghb-s", "nghb-sd")
        for u, v in pairs:
            for a, b in ((u, v), (v, u)):
                weights[a][b] = 1 if unit else weights[a].get(b, 0) + 1
        return weights

    def _first_candidate(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """Return the subnet's top actor, keeping neighbour scores up to date incrementally."""
        pool = remaining if isinstance(remaining, ActorPool) else ActorPool(remaining)
        if self._weights is None or pool is not remaining:  # first call, or no removal log
            self._weights = self._weights or self._neighbour_weights(net)
            self._scores = {
                aid: sum(w for nb, w in self._weights.get(aid, {}).items() if nb in pool.ids)
                for aid in pool.ids
            }
            self._heap = [(-score, aid) for aid, score in self._scores.items()]
            heapq.heapify(self._heap)
            self._seen_removed = len(pool.removed)
        for gone in pool.removed[self._seen_removed :]:
            self._scores.pop(gone, None)
            for nb, weight in self._weights.get(gone, {}).items():
                if nb in self._scores:
                    self._scores[nb] -= weight
                    heapq.heappush(self._heap, (-self._scores[nb], nb))
        self._seen_removed = len(pool.removed)
        while self._heap:
            neg, aid = self._heap[0]
            if aid in self._scores and -neg == self._scores[aid]:
                return [pool[aid]]
            heapq.heappop(self._heap)  # stale entry
        return []
