"""Abstract base class for SS/SQ/SQ_R timing-mode seeding strategies."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import islice
from math import ceil
from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.selectors import SELECTOR_CLASSES


class ActorPool:
    """Actors still eligible for seeding: original order, O(1) lookup and removal."""

    def __init__(self, actors: Iterable[MLNetworkActor]) -> None:
        """Keep ``actors`` keyed by id."""
        self._by_id = {a.actor_id: a for a in actors}
        self.removed: list[Any] = []  # ids in removal order, for incremental consumers

    def __iter__(self) -> Iterator[MLNetworkActor]:
        """Iterate over the actors in their original order."""
        return iter(self._by_id.values())

    def __len__(self) -> int:
        """Return the number of actors."""
        return len(self._by_id)

    def __getitem__(self, actor_id: Any) -> MLNetworkActor:
        """Return the actor with ``actor_id``."""
        return self._by_id[actor_id]

    @property
    def ids(self) -> Any:
        """Return a live set-like view of the actor ids."""
        return self._by_id.keys()

    def discard(self, actor_ids: Iterable[Any]) -> None:
        """Remove the given actors (missing ones are ignored)."""
        for actor_id in actor_ids:
            if self._by_id.pop(actor_id, None) is not None:
                self.removed.append(actor_id)


class BaseSeedingStrategy(ABC):
    """
    Abstract base for seeding strategies that share SS/SQ/SQ_R timing logic.

    Subclasses implement ``_candidates`` to control *which* actors are eligible
    and in what order.  ``next_batch`` applies the chosen timing mode on top of
    that candidate list and is not intended to be overridden.

    Timing modes:

    - ``SS``   -- single-shot: seed all candidates at epoch 0.
    - ``SQ``   -- sequential: ``batch_size`` candidates seeded every epoch.
    - ``SQ_R`` -- sequential with revival: seed only at epoch 0 or when
      spreading has stagnated (``stagnant=True``).

    ``batch_size`` is ignored for ``SS``.
    ``selector_params`` forwards extra keyword arguments to the selector
    constructor (e.g. ``TsNetSeedSelector`` needs ``model_dir``).
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
        if selector not in SELECTOR_CLASSES:
            raise ValueError(
                f"Unknown selector {selector!r}. Available: {sorted(SELECTOR_CLASSES)}"
            )
        if mode not in {"SS", "SQ", "SQ_R"}:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of: SS, SQ, SQ_R")
        self._selector = SELECTOR_CLASSES[selector](**(selector_params or {}))
        self._mode = mode
        self._batch_size = batch_size
        self._subnet_cache: MultilayerNetwork | None = None
        self._subnet_actor_ids: frozenset[str] = frozenset()
        self.budget: int | None = None  # seeds of the run, set by the simulator
        self._seeded = 0
        self._ranked_at: int | None = None  # seeds planted at the last ranking
        self._ranking: list[MLNetworkActor] | None = None
        self._cursor = 0  # ranking position of the first actor still eligible

    def _set_ranking(self, ranking: list[MLNetworkActor]) -> None:
        """Rank by ``ranking`` from now on."""
        self._ranking, self._cursor, self._ranked_at = ranking, 0, self._seeded

    def _ranked(self, pool: ActorPool, n: int, *skip: Any) -> list[MLNetworkActor]:
        """First ``n`` actors of the ranking still in ``pool`` and in none of ``skip``."""
        ranking = self._ranking or []
        while self._cursor < len(ranking) and ranking[self._cursor].actor_id not in pool.ids:
            self._cursor += 1  # actors only leave the pool
        ids = (a.actor_id for a in islice(ranking, self._cursor, None))
        eligible = (i for i in ids if i in pool.ids and not any(i in s for s in skip))
        return [pool[i] for i in islice(eligible, n)]

    def _rerank_due(self, percent: float) -> bool:
        """Whether ``percent`` % of the budget was seeded since the last ranking."""
        if self._ranked_at is None:
            return True
        return self._seeded - self._ranked_at >= ceil(percent / 100 * (self.budget or 0))

    def _build_subnet(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> MultilayerNetwork:
        """
        Return a subnet restricted to actors still eligible for seeding.

        The simulator prunes ``remaining`` after every LTM step, so every actor
        in ``remaining`` is guaranteed to be neither seeded nor organically
        activated.  The subnet is built once and updated incrementally: when
        actors leave ``remaining`` (seeded or organically activated), their
        nodes are removed directly from the cached subnet's layer graphs, which
        is far cheaper than rebuilding via ``net.subgraph()`` on every call.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: subnet of ``net`` restricted to eligible actors.
        """
        remaining_ids = frozenset(a.actor_id for a in remaining)
        if self._subnet_cache is None:
            eligible = [a for a in net.get_actors() if a.actor_id in remaining_ids]
            self._subnet_cache = net.subgraph(eligible)
            self._subnet_actor_ids = remaining_ids
        else:
            gone = self._subnet_actor_ids - remaining_ids
            if gone:
                for layer_graph in self._subnet_cache.layers.values():
                    for actor_id in gone:
                        if layer_graph.has_node(actor_id):
                            layer_graph.remove_node(actor_id)
                self._subnet_actor_ids = remaining_ids
        return self._subnet_cache

    @staticmethod
    def _stable_candidates(
        ranking: list[MLNetworkActor],
        remaining: list[MLNetworkActor],
    ) -> list[MLNetworkActor]:
        """
        Filter and sort candidates by selector rank (primary) then actor_id (secondary).

        Actors are taken from ``remaining`` (the simulator removes them by id) and ordered by
        their position
        in ``ranking`` first, with ``actor_id`` breaking any ties to ensure
        deterministic output when the selector assigns equal scores.

        :param ranking: full selector output in descending preference order.
        :param remaining: actors still eligible for seeding.

        :returns: eligible actors sorted by ``(rank, actor_id)``.
        """
        rank_of = {a.actor_id: i for i, a in enumerate(ranking)}
        return sorted(
            (a for a in remaining if a.actor_id in rank_of),
            key=lambda a: (rank_of[a.actor_id], a.actor_id),
        )

    @abstractmethod
    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Return actors eligible for seeding at this moment, in preference order.

        Implementations decide how to rank actors (static pre-computed ranking,
        live subnet re-ranking, etc.).  The caller (``next_batch``) applies
        the timing-mode slice on the result.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: eligible actors in descending preference order.
        """

    def next_batch(
        self,
        net: MultilayerNetwork,
        remaining: list[MLNetworkActor],
        epoch: int,
        stagnant: bool,
    ) -> list[MLNetworkActor]:
        """
        Return actors to seed at this epoch according to the timing mode.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.
        :param epoch: current epoch index.
        :param stagnant: ``True`` when spreading has not produced new
            activations for ``patience`` consecutive epochs.

        :returns: actors to seed now; empty list means no seeding this epoch.
        """
        if self._mode == "SS":
            batch = self._candidates(net, remaining) if epoch == 0 else []
        elif self._mode == "SQ" or (self._mode == "SQ_R" and (epoch == 0 or stagnant)):
            batch = self._candidates(net, remaining)[: self._batch_size]
        elif self._mode == "SQ_R":
            batch = []
        else:
            raise ValueError(f"Unknown mode: {self._mode!r}")
        self._seeded += len(batch)
        return batch
