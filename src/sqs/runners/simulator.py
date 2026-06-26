"""Sequential seeding simulator and its result container."""

from dataclasses import dataclass, field

from network_diffusion.logger import Logger
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBuff
from network_diffusion.simulator import Simulator

from sqs.runners.strategies import SeedingStrategy


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
    """

    logger: Logger
    seeding_events: dict[int, list[str]] = field(default_factory=dict)


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
        max_epochs: int,
        patience: int | None = None,
    ) -> SequentialRunLog:
        """
        Run a seeding simulation on the network held by this instance.

        At every epoch the strategy is asked for a batch of actors to seed.  If
        it returns a non-empty batch, those actors are planted and the epoch is
        logged as a seeding event (no LTM step runs that epoch).  Otherwise one
        LTM step executes.  The epoch counter is continuous across the whole
        simulation.  The loop terminates when all seeds have been planted and
        stagnation is detected, or after ``max_epochs`` epochs regardless.

        :param n_seeds: total number of seeds to plant.
        :param strategy: decides which actors to seed and when.
        :param max_epochs: hard upper bound on the total number of epochs.
        :param patience: stop the simulation once all seeds are planted and this
            many consecutive LTM epochs produce no new activations; ``None``
            disables early stopping (loop runs to ``max_epochs``).

        :returns: ``SequentialRunLog`` containing the Logger and seeding timeline.
        """
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be None or a positive integer.")
        if not isinstance(self._network, MultilayerNetwork):
            raise TypeError(
                "SequentialSimulator requires a MultilayerNetwork, not a TemporalNetwork."
            )

        logger = Logger(str(self._model), str(self._network))
        seeding_events: dict[int, list[str]] = {}
        seeds_planted = 0
        stagnant = False
        old_states: list[NUBuff] = []
        self.stopping_counter = 0

        initial_states = self._model.determine_initial_states(self._network)
        self._model.update_network(self._network, initial_states)

        remaining: list[MLNetworkActor] = list(self._network.get_actors())

        for epoch in range(max_epochs):
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
                    logger.add_local_stat(epoch, seed_json)  # partial snapshot: seeded actors only
                    seeds_planted += len(batch)
                    stagnant = False
                    self.stopping_counter = 0
                    old_states = seed_buffs
                    continue

            # --- LTM epoch ---
            # network_evaluation_step returns the proposed next state for every node;
            # update_network applies it and returns the full state snapshot for logging.
            new_states = self._model.network_evaluation_step(self._network)
            epoch_json = self._model.update_network(self._network, new_states)
            logger.add_global_stat(self._model.get_states_num(self._network))
            logger.add_local_stat(epoch, epoch_json)  # full snapshot: all actors

            if patience is not None:
                self._update_counter(new_states, old_states)
                stagnant = self.stopping_counter >= patience
                if not stagnant:
                    old_states = new_states
                if seeds_planted >= n_seeds and stagnant:
                    break

        logger.convert_logs(self._model.get_allowed_states(self._network))
        return SequentialRunLog(logger=logger, seeding_events=seeding_events)
