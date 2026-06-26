"""Unit tests for SequentialSimulator."""

import pytest
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.models.mltm import SQSMLTModel
from sqs.runners.strategies import MockSeedingStrategy
from sqs.runners.simulator import SequentialRunLog, SequentialSimulator


class TestSQSMLTModel:
    def test_all_actors_inactive_after_init(
        self, toy_net: MultilayerNetwork, or_model: SQSMLTModel
    ) -> None:
        """determine_initial_states must mark every node INACTIVE regardless of network size."""
        states = or_model.determine_initial_states(toy_net)
        active = [s for s in states if s.new_state == or_model.ACTIVE_STATE]
        assert len(active) == 0
        assert len(states) > 0


class TestSequentialSimulator:
    def test_returns_sequential_run_log(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """perform_propagation must return a SequentialRunLog."""
        result = simulator.perform_propagation(
            n_seeds=2, strategy=mock_strategy, max_epochs=20
        )
        assert isinstance(result, SequentialRunLog)

    def test_correct_number_of_seeding_events(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """seeding_events must contain exactly n_seeds entries."""
        n_seeds = 3
        result = simulator.perform_propagation(
            n_seeds=n_seeds, strategy=mock_strategy, max_epochs=20
        )
        assert len(result.seeding_events) == n_seeds

    def test_seeding_event_actor_ids_are_strings(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """Actor IDs stored in seeding_events must be strings."""
        result = simulator.perform_propagation(
            n_seeds=2, strategy=mock_strategy, max_epochs=20
        )
        for actor_ids in result.seeding_events.values():
            for actor_id in actor_ids:
                assert isinstance(actor_id, str)

    def test_seeding_epochs_are_non_decreasing(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """Each successive seeding event must occur at a strictly later epoch."""
        result = simulator.perform_propagation(
            n_seeds=3, strategy=mock_strategy, max_epochs=20
        )
        epochs = sorted(result.seeding_events.keys())
        assert epochs == sorted(set(epochs))
        assert epochs[0] == 0
        for a, b in zip(epochs, epochs[1:]):
            assert b > a

    def test_epoch_counter_is_continuous(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """Local stats must contain a gapless sequence of epoch indices from 0 upward."""
        result = simulator.perform_propagation(
            n_seeds=2, strategy=mock_strategy, max_epochs=10
        )
        epochs = sorted(result.logger.get_detailed_logs().keys())
        assert epochs == list(range(len(epochs)))

    def test_patience_stops_spreading_early(self, toy_net: MultilayerNetwork) -> None:
        """With patience=1, simulation stops shortly after the last seed is planted."""
        # mi=1.0 means a node only activates when ALL neighbours are active -- effectively
        # no organic spread in a sparse toy network.
        # MockSeedingStrategy plants one seed per epoch, so seeds land at epochs 0..n_seeds-1.
        # After all seeds are planted, two LTM epochs occur before patience triggers:
        #   first LTM:  new_states=[] != old_states=seed_buffs -> counter stays 0
        #   second LTM: new_states=[] == old_states=[]         -> counter=1 >= patience -> stop
        # Total logged = n_seeds + patience + 1
        model = SQSMLTModel(protocol="OR", mi_value=1.0)
        sim = SequentialSimulator(model, toy_net.copy())
        n_seeds = 2
        patience = 1
        result = sim.perform_propagation(
            n_seeds=n_seeds,
            strategy=MockSeedingStrategy(),
            max_epochs=50,
            patience=patience,
        )
        assert len(result.logger.get_detailed_logs()) == n_seeds + patience + 1

    def test_invalid_patience_raises(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """patience=0 must raise ValueError."""
        with pytest.raises(ValueError, match="[Pp]atience"):
            simulator.perform_propagation(
                n_seeds=1, strategy=mock_strategy, max_epochs=10, patience=0
            )

    def test_reproducibility(self, toy_net: MultilayerNetwork) -> None:
        """Two runs with identical config must seed the same actors and reach the same coverage."""
        from sqs.misc.result_handler import extract_simulation_result

        def _run(net: MultilayerNetwork) -> SequentialRunLog:
            model = SQSMLTModel(protocol="OR", mi_value=0.15)
            sim = SequentialSimulator(model, net.copy())
            return sim.perform_propagation(
                n_seeds=3, strategy=MockSeedingStrategy(), max_epochs=20, patience=3
            )

        total_actors = toy_net.get_actors_num()
        run_a = _run(toy_net)
        run_b = _run(toy_net)

        # MockSeedingStrategy is deterministic so the same actors are seeded in the same order.
        assert run_a.seeding_events == run_b.seeding_events
        # Final coverage must match even though per-epoch actor evaluation order is random.
        assert (
            extract_simulation_result(run_a, total_actors, n_seeds=3).exposed_nb
            == extract_simulation_result(run_b, total_actors, n_seeds=3).exposed_nb
        )
