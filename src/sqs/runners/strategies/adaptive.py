"""Adaptive seeding: selector ranking steered by the activation rule learnt online."""

from collections import Counter
from fractions import Fraction
from functools import cache
from math import ceil
from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.strategies.base import ActorPool
from sqs.runners.strategies.static import StaticSeedingStrategy

MU_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95


@cache
def _need(mu: float, degree: int) -> int:
    """Fewest active neighbours whose impulse exceeds ``mu`` (summed as MLTModel does)."""
    impulse = 0.0
    for count in range(1, degree + 1):
        impulse += 1 / degree
        if impulse > mu:
            return count
    return degree + 1  # never


class AdaptiveSeedingStrategy(StaticSeedingStrategy):
    """
    Seed the actor that pushes most inactive neighbours over the learnt rule; otherwise by rank.

    The rule is the multilayer LTM one with unknown ``mu`` (0.05 grid) and ``delta`` (fractions
    k / layers). After every step the actors whose neighbourhood changed show whether they
    activated, which narrows the feasible ``delta`` per ``mu``. The estimate is the middle
    feasible ``mu`` with its lowest feasible ``delta``.

    Ties and the order without push follow the selector ranking: the Static one at the start,
    then a ranking of the subnet of eligible actors each time ``rerank_percent`` % of the budget
    has been seeded (0: before every seed, > 100: never).
    """

    def __init__(
        self,
        selector: str,
        mode: str,
        batch_size: int = 1,
        selector_params: dict[str, Any] | None = None,
        rerank_percent: float = 1.0,
    ) -> None:
        """Initialise as ``StaticSeedingStrategy``."""
        super().__init__(selector, mode, batch_size, selector_params)
        self._rerank_percent = rerank_percent
        self._lo = dict.fromkeys(MU_GRID, Fraction(0))  # feasible: lo < delta <= hi
        self._hi = dict.fromkeys(MU_GRID, Fraction(1))
        self._rule: tuple[float, Fraction] | None = None
        self._counts: Counter = Counter()  # (layer, actor id) -> active neighbours
        self._changed: set[Any] = set()  # actors with a new active neighbour before the step
        self._last_batch: set[Any] = set()
        self._pushers: dict[Any, set[Any]] = {}  # inactive actor -> actors that push it over
        self._push: Counter = Counter()
        self._self_active: set[Any] = set()  # inactive actors the rule activates next step
        self._rank: dict[Any, int] = {}

    def next_batch(
        self,
        net: MultilayerNetwork,
        remaining: ActorPool,
        epoch: int,
        stagnant: bool,
    ) -> list[MLNetworkActor]:
        """Learn from the last step, then apply the timing mode."""
        if self._rule is None:
            self._start(net, remaining)
        else:
            self._observe(remaining)
        batch = super().next_batch(net, remaining, epoch, stagnant)
        self._last_batch = {i.actor_id for i in batch}
        return batch

    def _candidates(self, net: MultilayerNetwork, remaining: ActorPool) -> list[MLNetworkActor]:
        """Best pushers by (push, rank, id), then the ranking; self-active actors are skipped."""
        if not remaining:
            return []
        if self._rerank_due(self._rerank_percent):
            self._set_ranking(self._selector.actorwise(self._build_subnet(net, remaining)))
        n = len(remaining) if self._mode == "SS" else self._batch_size
        pushers = [i for i in self._push if i not in self._self_active]
        pushers = sorted(
            pushers, key=lambda i: (-self._push[i], self._rank.get(i, len(self._rank)), i)
        )[:n]
        rest = self._ranked(remaining, n - len(pushers), self._self_active, set(pushers))
        return [remaining[i] for i in pushers] + rest

    def _set_ranking(self, ranking: list[MLNetworkActor]) -> None:
        """Rank by ``ranking`` (also for ties between pushers)."""
        super()._set_ranking(ranking)
        self._rank = {a.actor_id: r for r, a in enumerate(ranking)}

    def _start(self, net: MultilayerNetwork, remaining: ActorPool) -> None:
        """Read the initial state of ``net``."""
        self._net = net
        self._layers = {i.actor_id: i.layers for i in net.get_actors()}
        self._deltas = sorted(
            {Fraction(k, len(ls)) for ls in self._layers.values() for k in range(1, len(ls) + 1)}
        )
        self._increment([i for i in self._layers if i not in remaining.ids])
        self._seen = len(remaining.removed)
        self._set_ranking(self.get_or_compute_ranking(net))
        self._rule = self._estimate()
        self._refresh(remaining, set(remaining.ids))

    def _observe(self, remaining: ActorPool) -> None:
        """Learn from the last step and update the pushes."""
        new = remaining.removed[self._seen :]
        self._seen = len(remaining.removed)
        seeds = [i for i in new if i in self._last_batch]
        activated = {i for i in new if i not in self._last_batch}
        # the step saw our seeds; self-active actors that stayed inactive refute the estimate
        judged = self._changed | self._self_active | self._increment(seeds)
        for i in judged:
            if i in remaining.ids or i in activated:
                self._adapt(i, i in activated)
        self._changed = self._increment(activated)
        rule = self._estimate()
        if rule != self._rule:
            self._rule, self._pushers, self._push, self._self_active = rule, {}, Counter(), set()
            self._refresh(remaining, set(remaining.ids))
        else:
            self._refresh(remaining, judged | self._changed | set(new))

    def _increment(self, actor_ids: Any) -> set[Any]:
        """Count newly active actors at their neighbours; return those neighbours."""
        touched = set()
        for actor_id in actor_ids:
            for layer in self._layers[actor_id]:
                for nb in self._net.layers[layer].neighbors(actor_id):
                    self._counts[(layer, nb)] += 1
                    touched.add(nb)
        return touched

    def _passing(self, mu: float, actor_id: Any) -> int:
        """Layers on which the actor gets a positive input under ``mu``."""
        return sum(
            self._counts[(layer, actor_id)] >= _need(mu, self._net.layers[layer].degree(actor_id))
            for layer in self._layers[actor_id]
        )

    def _adapt(self, actor_id: Any, activated: bool) -> None:
        """Narrow the feasible delta of every mu with one observed (non-)activation."""
        for mu in MU_GRID:
            share = Fraction(self._passing(mu, actor_id), len(self._layers[actor_id]))
            if activated:
                self._hi[mu] = min(self._hi[mu], share)
            else:
                self._lo[mu] = max(self._lo[mu], share)

    def _estimate(self) -> tuple[float, Fraction]:
        """Middle feasible mu with its lowest feasible delta."""
        feasible = {
            mu: [d for d in self._deltas if self._lo[mu] < d <= self._hi[mu]] for mu in MU_GRID
        }
        mus = [mu for mu in MU_GRID if feasible[mu]]
        if not mus:  # outside the model family: keep the last estimate
            return self._rule or (0.5, self._deltas[0])
        mu = mus[(len(mus) - 1) // 2]
        return mu, feasible[mu][0]

    def _refresh(self, remaining: ActorPool, actor_ids: set[Any]) -> None:
        """Recompute who pushes ``actor_ids`` over the estimated rule."""
        mu, delta = self._rule
        for actor_id in actor_ids:
            for pusher in self._pushers.pop(actor_id, ()):
                self._push[pusher] -= 1
                if not self._push[pusher]:
                    del self._push[pusher]
            self._self_active.discard(actor_id)
            if actor_id not in remaining.ids:
                continue
            layers = self._layers[actor_id]
            passing = self._passing(mu, actor_id)
            required = ceil(delta * len(layers))
            if passing >= required:
                self._self_active.add(actor_id)
                continue
            hits = Counter(  # neighbour -> layers where it is the one missing
                nb
                for layer in layers
                if self._counts[(layer, actor_id)]
                == _need(mu, self._net.layers[layer].degree(actor_id)) - 1
                for nb in self._net.layers[layer].neighbors(actor_id)
                if nb in remaining.ids
            )
            self._pushers[actor_id] = {nb for nb, n in hits.items() if passing + n >= required}
            self._push.update(self._pushers[actor_id])
