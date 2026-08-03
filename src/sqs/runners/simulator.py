"""Sequential seeding simulator and its result container."""

import logging
from dataclasses import dataclass, field

from network_diffusion.logger import Logger
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBuff
from network_diffusion.simulator import Simulator

from sqs.runners.strategies import SeedingStrategy

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

    Inherits utility methods (``_update_counter``) from ``network_diffusion.Simulator``
    and overrides ``perform_propagation`` with a strategy-driven signature.
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

        At every epoch the strategy is asked for a batch of actors to seed.  If
        it returns a non-empty batch, those actors are planted and the epoch is
        logged as a seeding event (no LTM step runs that epoch).  Otherwise one
        LTM step executes.  The epoch counter is continuous across the whole
        simulation.

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
            # In SQ_R mode each revival seed requires patience + 1 LTM epochs to confirm
            # stagnation (1 epoch where stopping_counter resets because old_states == seed_buffs,
            # then patience identical epochs).  The rough lower bound on total epochs needed is
            # n_seeds * (patience + 2).  SS and SQ modes are unaffected.
            sqr_min_epochs = n_seeds * (patience + 2)
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

        logger = Logger(str(self._model), str(self._network))
        seeding_events: dict[int, list[str]] = {}
        active_counts: dict[int, int] = {}
        seeds_planted = 0
        stagnant = False
        old_states: list[NUBuff] = []
        self.stopping_counter = 0

        initial_states = self._model.determine_initial_states(self._network)
        self._model.update_network(self._network, initial_states)

        remaining: list[MLNetworkActor] = list(self._network.get_actors())

        epoch = 0
        while max_epochs is None or epoch < max_epochs:
            # --- seeding epoch ---
            # Strategy decides whether to plant seeds this epoch and which actors to use.
            # Seeding and LTM steps are mutually exclusive: a seeding epoch skips LTM so
            # that the network state seen by the strategy is always post-LTM, not mid-step.
            if seeds_planted < n_seeds:
                batch = strategy.next_batch(self._network, remaining, epoch, stagnant)
                batch = batch[: n_seeds - seeds_planted]  # cap to remaining budget
                if batch:
                    for actor in batch:
                        remaining.remove(actor)  # remove before seeding so strategy never re-seeds
                    seed_buffs: list[NUBuff] = [
                        NUBuff(
                            node_name=actor.actor_id,
                            layer_name=layer,
                            new_state=MLTModel.ACTIVE_STATE,
                        )
                        for actor in batch
                        for layer in actor.layers
                    ]
                    seed_json = self._model.update_network(self._network, seed_buffs)
                    seeding_events[epoch] = [str(a.actor_id) for a in batch]
                    logger.add_global_stat(self._model.get_states_num(self._network))
                    if track_full_logs:
                        # partial snapshot: seeded actors only
                        logger.add_local_stat(epoch, seed_json)
                    seeds_planted += len(batch)
                    stagnant = False
                    self.stopping_counter = 0
                    old_states = seed_buffs
                    epoch += 1
                    continue

            # --- LTM epoch ---
            # network_evaluation_step returns the proposed next state for every node;
            # update_network applies it and returns the full state snapshot for logging.
            new_states = self._model.network_evaluation_step(self._network)
            epoch_json = self._model.update_network(self._network, new_states)
            logger.add_global_stat(self._model.get_states_num(self._network))
            # new_states is a full snapshot (every actor, every layer), so the count of
            # distinct active node_names is the cumulative active-actor count after this
            # epoch.  Tracked unconditionally (O(1) to store) so result metrics never need
            # the full per-actor logger snapshot below, which is what makes track_full_logs
            # safe to disable on large networks with long-running SQ_R schedules.
            active_counts[epoch] = len(
                {buff.node_name for buff in new_states if buff.new_state == MLTModel.ACTIVE_STATE}
            )
            if track_full_logs:
                logger.add_local_stat(epoch, epoch_json)  # full snapshot: all actors

            # Prune actors from remaining that organically activated this epoch.
            # Keeping remaining accurate eliminates the need for is_already_active()
            # checks inside every strategy on every call.
            remaining_ids = {a.actor_id for a in remaining}
            newly_active = {
                buff.node_name
                for buff in new_states
                if buff.new_state == MLTModel.ACTIVE_STATE and buff.node_name in remaining_ids
            }
            if newly_active:
                remaining = [a for a in remaining if a.actor_id not in newly_active]

            if not remaining:
                # Every actor is either seeded or organically active. LTM activation is
                # monotone, so the network has reached its final fixed point regardless of
                # whether the full seed budget was planted -- there are no candidates left
                # for the strategy to seed and no further state changes are possible.
                # Waiting for seeds_planted to reach n_seeds here would loop forever when
                # max_epochs is None.
                break

            if patience is not None:
                self._update_counter(new_states, old_states)
                stagnant = self.stopping_counter >= patience
                if not stagnant:
                    old_states = new_states
                if seeds_planted >= n_seeds and stagnant:
                    break

            epoch += 1

        logger.convert_logs(self._model.get_allowed_states(self._network))
        return SequentialRunLog(
            logger=logger, seeding_events=seeding_events, active_counts=active_counts
        )
