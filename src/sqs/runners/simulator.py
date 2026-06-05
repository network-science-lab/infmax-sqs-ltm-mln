"""Sequential seeding simulator and its result container."""

from dataclasses import dataclass, field

from network_diffusion.logger import Logger
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBuff
from network_diffusion.simulator import Simulator

from sqs.runners.seed_schedule import SeedSchedule


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
    Simulator that introduces seeds one at a time during the LTM run loop.

    Inherits utility methods (``_update_counter``) from ``network_diffusion.Simulator``
    and overrides ``perform_propagation`` with a sequential-seeding signature.
    """

    def perform_propagation(  # type: ignore[override]
        self,
        n_seeds: int,
        seed_schedule: SeedSchedule,
        spreading_epochs: int,
        patience: int | None = None,
    ) -> SequentialRunLog:
        """
        Run a sequential seeding simulation on the network held by this instance.

        Seeds are introduced one at a time.  After each seeding event, LTM
        propagation runs for up to ``spreading_epochs`` epochs (or until ``patience``
        consecutive epochs pass with no new activations).  The epoch counter is
        continuous across all seeding phases.

        :param n_seeds: total number of seeds to plant.
        :param seed_schedule: strategy that selects the next actor to seed.
        :param spreading_epochs: maximum number of LTM epochs between consecutive
            seeding events.
        :param patience: stop a spreading phase early when this many consecutive
            epochs produce no new activations; ``None`` disables early stopping.

        :returns: ``SequentialRunLog`` containing the Logger and the seeding
            timeline (epoch -> actor IDs).
        """
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be None or a positive integer.")

        logger = Logger(str(self._model), str(self._network))
        seeding_events: dict[int, list[str]] = {}
        epoch = 0

        # Set every actor to INACTIVE before the first seeding event.
        initial_states = self._model.determine_initial_states(self._network)
        self._model.update_network(self._network, initial_states)

        remaining: list[MLNetworkActor] = list(self._network.get_actors())

        for _ in range(n_seeds):
            # --- seeding event ------------------------------------------------
            actor = seed_schedule.next_seed(self._network, remaining)
            remaining.remove(actor)

            seed_buffs: list[NUBuff] = [
                NUBuff(
                    node_name=actor.actor_id,
                    layer_name=layer,
                    new_state=MLTModel.ACTIVE_STATE,
                )
                for layer in actor.layers
            ]
            seed_json = self._model.update_network(self._network, seed_buffs)

            seeding_events[epoch] = [str(actor.actor_id)]
            logger.add_global_stat(self._model.get_states_num(self._network))
            logger.add_local_stat(epoch, seed_json)

            # --- spreading phase ----------------------------------------------
            self.stopping_counter = 0
            old_states = seed_buffs

            for _ in range(spreading_epochs):
                epoch += 1
                new_states = self._model.network_evaluation_step(self._network)
                epoch_json = self._model.update_network(self._network, new_states)

                logger.add_global_stat(self._model.get_states_num(self._network))
                logger.add_local_stat(epoch, epoch_json)

                if patience is not None:
                    self._update_counter(new_states, old_states)
                    if self.stopping_counter >= patience:
                        break
                    old_states = new_states

            epoch += 1  # advance past this spreading phase before next seed

        logger.convert_logs(self._model.get_allowed_states(self._network))
        return SequentialRunLog(logger=logger, seeding_events=seeding_events)
