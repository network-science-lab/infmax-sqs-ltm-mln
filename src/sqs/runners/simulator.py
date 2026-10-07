"""Sequential seeding simulator and its result container."""

import logging
from dataclasses import dataclass, field
from typing import Any

from network_diffusion.logger import Logger
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBuff
from network_diffusion.simulator import Simulator

from sqs.runners.strategies import SeedingStrategy
from sqs.runners.strategies.base import ActorPool

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequentialRunLog:
    """
    Result container for a single sequential seeding simulation run.

    Bundles the ``network_diffusion`` Logger with the seeding timeline so that
    the result handler can reconstruct both the ``seedings`` and ``expositions``
    columns without re-running the simulation.

    ``seeding_events`` maps each epoch index at which a seeding event occurred to
    the list of actor IDs seeded at that epoch.  Epochs with no seeding are absent
    from the dict, not stored as empty lists.

    ``active_counts`` maps each LTM epoch to the cumulative count of distinct active
    actors after that epoch.  This is the only per-epoch data ``extract_simulation_result``
    needs; keeping it separate from ``logger`` means result metrics stay cheap to compute
    (O(1) per epoch) even when ``track_full_logs=False`` drops the full per-actor snapshots.
    """

    logger: Logger
    seeding_events: dict[int, list[str]] = field(default_factory=dict)
    active_counts: dict[int, int] = field(default_factory=dict)


class SequentialSimulator(Simulator):
    """
    Simulator that plants seeds during the LTM run loop according to a strategy.

    Overrides ``network_diffusion.Simulator.perform_propagation`` with a strategy-driven
    signature.
    """

    def perform_propagation(  # type: ignore[override]
        self,
        n_seeds: int,
        strategy: SeedingStrategy,
        max_epochs: int | None,
        patience: int | None = None,
        track_full_logs: bool = True,
    ) -> SequentialRunLog:
        """
        Run a seeding simulation on the network held by this instance.

        Every epoch is ``seed (if the strategy returns any) -> one LTM step``, so diffusion
        always follows each seeding event within the same epoch. The LTM step evaluates only
        inactive neighbours of newly active actors; the result equals a full MLTModel step.

        The loop terminates when all seeds have been planted and stagnation is
        detected, or after ``max_epochs`` epochs regardless.  Pass
        ``max_epochs=None`` to remove the hard cap; ``patience`` must then be
        set or the loop has no termination signal.  ``SQ_R`` schedules with
        large seed budgets should use ``max_epochs=None`` to avoid being cut off
        before all seeds are planted -- LTM activation is monotone, so a run
        with ``patience`` set is guaranteed to terminate at a real fixed point.

        :param n_seeds: total number of seeds to plant.
        :param strategy: decides which actors to seed and when.
        :param max_epochs: hard upper bound on the total number of epochs, or
            ``None`` for an unbounded run (requires ``patience`` to be set).
        :param patience: stop the simulation once all seeds are planted and this
            many consecutive LTM epochs produce no new activations; ``None``
            disables early stopping.
        :param track_full_logs: whether to retain the full per-actor state snapshot for
            every epoch in ``logger`` (needed only to write detailed logs to disk).  Result
            metrics rely solely on ``active_counts``, which is always populated, so callers
            that do not persist detailed logs should pass ``False`` -- on large networks with
            ``SQ_R`` schedules the epoch count scales with ``n_seeds * patience``, and a full
            per-epoch, per-actor snapshot at that scale can grow into gigabytes per run.

        :returns: ``SequentialRunLog`` containing the Logger and seeding timeline.
        """
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be None or a positive integer.")
        if max_epochs is None and patience is None:
            raise ValueError(
                "At least one of max_epochs or patience must be set; "
                "an unbounded run with no stagnation check would loop forever."
            )
        if patience is not None and patience > 1:
            # In SQ_R mode each revival seed needs its own epoch plus patience stagnant ones
            sqr_min_epochs = n_seeds * (patience + 1)
            if max_epochs is not None and sqr_min_epochs > max_epochs:
                _log.warning(
                    "patience=%d with n_seeds=%d needs approximately %d epochs in SQ_R mode "
                    "(max_epochs=%d); not all seeds may be planted before the epoch limit.",
                    patience,
                    n_seeds,
                    sqr_min_epochs,
                    max_epochs,
                )
        if not isinstance(self._network, MultilayerNetwork):
            raise TypeError(
                "SequentialSimulator requires a MultilayerNetwork, not a TemporalNetwork."
            )

        net = self._network
        # str(net) computes network statistics (e.g. clustering): costly, only for detailed logs
        logger = Logger(str(self._model), str(net) if track_full_logs else "")
        seeding_events: dict[int, list[str]] = {}
        active_counts: dict[int, int] = {}
        self._model.update_network(net, self._model.determine_initial_states(net))
        remaining = ActorPool(net.get_actors())
        actors = {a.actor_id: a for a in remaining}
        self._active_neighbours: dict[tuple[str, Any], int] = {}  # (layer, actor id) -> count
        seeds_planted = active = 0
        touched: set[Any] = set()  # actors with a new active neighbour since the last LTM step
        stagnant = False
        self.stopping_counter = 0

        epoch = 0
        while max_epochs is None or epoch < max_epochs:
            # --- seeding: the strategy sees the state after the previous diffusion step ---
            seeds: list[MLNetworkActor] = []
            if seeds_planted < n_seeds:
                seeds = strategy.next_batch(net, remaining, epoch, stagnant)
                seeds = seeds[: n_seeds - seeds_planted]
            if seeds:
                remaining.discard(a.actor_id for a in seeds)
                touched |= self._activate(seeds)
                seeding_events[epoch] = [str(a.actor_id) for a in seeds]
                seeds_planted += len(seeds)

            # --- diffusion: one synchronous LTM step ---
            # Only inactive actors with a new active neighbour can cross the threshold, so the
            # result equals MLTModel.network_evaluation_step on the whole network.
            candidates = [actors[i] for i in touched if i in remaining.ids]
            activated = self._model.activations(net, candidates, self._active_neighbours)
            remaining.discard(a.actor_id for a in activated)
            touched = self._activate(activated)
            active += len(seeds) + len(activated)
            active_counts[epoch] = active
            if track_full_logs:
                logger.add_global_stat(self._model.get_states_num(net))
                logger.add_local_stat(epoch, self._snapshot())

            if not remaining:
                # Every actor is seeded or active: the fixed point is reached even if part of the
                # budget is left (otherwise the loop would never end with max_epochs=None).
                break

            if patience is not None:
                # Stagnation: `patience` consecutive epochs without any new active actor
                self.stopping_counter = 0 if seeds or activated else self.stopping_counter + 1
                stagnant = self.stopping_counter >= patience
                if seeds_planted >= n_seeds and stagnant:
                    break

            epoch += 1

        logger.convert_logs(self._model.get_allowed_states(self._network))
        return SequentialRunLog(
            logger=logger, seeding_events=seeding_events, active_counts=active_counts
        )

    def _activate(self, actors: list[MLNetworkActor]) -> set[Any]:
        """Set ``actors`` active on all their layers; return ids of their neighbours."""
        self._model.update_network(
            self._network,
            [
                NUBuff(node_name=a.actor_id, layer_name=layer, new_state=MLTModel.ACTIVE_STATE)
                for a in actors
                for layer in a.layers
            ],
        )
        neighbours = set()
        for actor in actors:
            for layer in actor.layers:
                for nb in self._network.layers[layer].neighbors(actor.actor_id):
                    key = (layer, nb)
                    self._active_neighbours[key] = self._active_neighbours.get(key, 0) + 1
                    neighbours.add(nb)
        return neighbours

    def _snapshot(self) -> list[dict[str, str]]:
        """Return the state of every node (for detailed logs)."""
        return [
            NUBuff(node_name=node, layer_name=layer, new_state=status).to_json()
            for layer, graph in self._network.layers.items()
            for node, status in graph.nodes(data="status")
        ]
