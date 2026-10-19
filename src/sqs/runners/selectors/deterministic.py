"""
Selectors with explicit tie-breaking (score desc, then actor id).

network_diffusion breaks ties in hash / memory-address dependent order, so its rankings differ
between processes. These versions are reproducible, except ``MdsSelector``: its minimum
dominating set comes from a search with a wall-clock timeout, so only the ranking cache
(``spreading.shared_ranking``) makes it consistent.
"""

import heapq
from collections.abc import Callable
from typing import Any

import torch
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.centralities import degree, neighbourhood_size
from network_diffusion.mln.mds import get_mds_locimpr
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding import VoteRankMLNSeedSelector

Scores = Callable[[MultilayerNetwork], dict[MLNetworkActor, int]]
Scored = list[tuple[MLNetworkActor, Any]]  # ranking with the score each actor got


def _neighbours(net: MultilayerNetwork, actor: MLNetworkActor) -> set[Any]:
    """Return ids of actors adjacent to ``actor`` on any layer."""
    return {nb for layer in actor.layers for nb in net.layers[layer].neighbors(actor.actor_id)}


class CentralitySelector:
    """Rank actors by a centrality score, ties broken by actor id."""

    def __init__(self, scores: Scores, **_: Any) -> None:
        """Store the score function (extra kwargs are ignored, as in network_diffusion)."""
        self._scores = scores

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Return the actors in ranking order."""
        return [a for a, _ in self.scored(net)]

    def scored(self, net: MultilayerNetwork) -> Scored:
        """Return (actor, centrality) ordered by (centrality desc, actor id)."""
        scores = self._scores(net)
        return [(a, scores[a]) for a in sorted(scores, key=lambda a: (-scores[a], a.actor_id))]


class DiscountSelector(CentralitySelector):
    """
    Centrality discount (Chen et al. 2009), ties broken by actor id.

    Same rule as network_diffusion: after an actor is picked, each not-yet-picked neighbour with a
    positive score loses one point.
    """

    def scored(self, net: MultilayerNetwork) -> Scored:
        """Return (actor, discounted score when picked) in discount order."""
        scores = {a.actor_id: s for a, s in self._scores(net).items()}
        actors = {a.actor_id: a for a in net.get_actors()}
        heap = [(-s, aid) for aid, s in scores.items()]
        heapq.heapify(heap)
        ranking: Scored = []
        while heap:
            neg, aid = heapq.heappop(heap)
            if aid not in scores or -neg != scores[aid]:
                continue  # stale entry
            ranking.append((actors[aid], scores.pop(aid)))
            for nb in _neighbours(net, actors[aid]):
                if scores.get(nb):
                    scores[nb] -= 1
                    heapq.heappush(heap, (-scores[nb], nb))
        return ranking


class VoteRankSelector(VoteRankMLNSeedSelector):
    """
    Actorwise VoteRank from network_diffusion with actors indexed by id.

    The library indexes actors and edges in hash-dependent order, so ``argmax`` ties, float
    summation order and the order of unelected actors changed between processes. Unelected actors
    are appended by id. On CUDA ``scatter_add_`` is still not bitwise deterministic.
    """

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Return the actors in ranking order."""
        return [a for a, _ in self.scored(net)]

    def scored(self, net: MultilayerNetwork) -> Scored:
        """Return (actor, votes when elected) in voting order, then the rest (0 votes) by id."""
        actors = sorted(net.get_actors(), key=lambda a: a.actor_id)
        n = len(actors)
        total_neighbours = sum(neighbourhood_size(net=net).values())
        if total_neighbours == 0:  # no links (e.g. a Subnet of isolated actors): nobody gets votes
            return [(a, 0.0) for a in actors]
        src, dst = _edges(net, {a.actor_id: i for i, a in enumerate(actors)}, self.device)
        inv_avg_nbs = torch.tensor(n / total_neighbours, dtype=torch.float32, device=src.device)
        scores = torch.zeros(n, dtype=torch.float32, device=src.device)
        ability = torch.ones(n, dtype=torch.float32, device=src.device)
        elected = torch.zeros(n, dtype=torch.bool, device=src.device)
        ranking: Scored = []
        for _ in range(n):
            scores.zero_()
            scores.scatter_add_(0, dst, ability[src])
            scores[elected] = 0.0
            top = int(scores.argmax().item())
            if scores[top].item() == 0.0:
                break
            ranking.append((actors[top], scores[top].item()))
            elected[top] = True
            ability[top] = 0.0
            ability[dst[src == top]] -= inv_avg_nbs
            ability.clamp_(min=0.0)
        return ranking + [(a, 0.0) for i, a in enumerate(actors) if not elected[i]]


def _edges(
    net: MultilayerNetwork, index: dict[Any, int], device: str | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both directions of every link (as network_diffusion does) in a fixed order."""
    links = {(index[a.actor_id], index[b.actor_id]) for a, b in net.get_links()}
    # (dst, src) order fixes the summation order of scatter_add_
    pairs = sorted([*links, *((d, s) for s, d in links)], key=lambda e: (e[1], e[0]))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    src = torch.tensor([s for s, _ in pairs], dtype=torch.long, device=device)
    dst = torch.tensor([d for _, d in pairs], dtype=torch.long, device=device)
    return src, dst


class MdsSelector:
    """Driver actors (minimum dominating set) first, then the rest; each group in base order."""

    def __init__(self, base: CentralitySelector | None = None, **_: Any) -> None:
        """Store the base ranking (``None`` orders by actor id)."""
        self._base = base

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Return the actors in ranking order."""
        return [a for a, _ in self.scored(net)]

    def scored(self, net: MultilayerNetwork) -> Scored:
        """Return (actor, {driver, degree}) with drivers first; ``degree`` only with a base."""
        drivers = {a.actor_id for a in get_mds_locimpr(net)}
        if self._base is None:
            order = [(a, None) for a in sorted(net.get_actors(), key=lambda a: a.actor_id)]
        else:
            order = self._base.scored(net)

        def score(actor: MLNetworkActor, base: Any) -> dict[str, Any]:
            driver = int(actor.actor_id in drivers)
            return {"driver": driver} if base is None else {"driver": driver, "degree": base}

        return [(a, score(a, b)) for a, b in order if a.actor_id in drivers] + [
            (a, score(a, b)) for a, b in order if a.actor_id not in drivers
        ]


def degree_centrality(**kwargs: Any) -> CentralitySelector:
    """Build the deterministic ``deg-c`` selector."""
    return CentralitySelector(degree, **kwargs)


def neighbourhood_size_centrality(**kwargs: Any) -> CentralitySelector:
    """Build the deterministic ``nghb-s`` selector."""
    return CentralitySelector(neighbourhood_size, **kwargs)


def degree_discount(**kwargs: Any) -> DiscountSelector:
    """Build the deterministic ``deg-c-d`` selector."""
    return DiscountSelector(degree, **kwargs)


def neighbourhood_size_discount(**kwargs: Any) -> DiscountSelector:
    """Build the deterministic ``nghb-sd`` selector."""
    return DiscountSelector(neighbourhood_size, **kwargs)
