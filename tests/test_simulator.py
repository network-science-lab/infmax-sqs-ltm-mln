"""Unit tests for SequentialSimulator."""

import pytest
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.models.propagation.mltm import SQSMLTModel
from sqs.runners.strategies import MockSeedingStrategy, StaticSeedingStrategy
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
        # Seeds land at epochs 0..n_seeds-1 (each followed by LTM in the same epoch), then
        # patience stagnant epochs stop the run: total logged = n_seeds + patience
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
        assert len(result.logger.get_detailed_logs()) == n_seeds + patience

    def test_invalid_patience_raises(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """patience=0 must raise ValueError."""
        with pytest.raises(ValueError, match="[Pp]atience"):
            simulator.perform_propagation(
                n_seeds=1, strategy=mock_strategy, max_epochs=10, patience=0
            )

    def test_stops_when_organic_spread_exhausts_remaining_before_budget(
        self, toy_net: MultilayerNetwork
    ) -> None:
        """
        Must terminate even when the network fully saturates before all seeds land.

        SQ_R only re-seeds once spreading stagnates, giving organic activation time
        to reach every actor from a single seed on this densely-connected toy network.
        Once no actor remains to seed, ``seeds_planted`` can never reach ``n_seeds``,
        so with ``max_epochs=None`` the loop must exit via the empty-``remaining``
        check rather than looping forever.
        """
        model = SQSMLTModel(protocol="OR", mi_value=0.1)
        sim = SequentialSimulator(model, toy_net.copy())
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        result = sim.perform_propagation(
            n_seeds=3, strategy=strategy, max_epochs=None, patience=2
        )
        planted = sum(len(actors) for actors in result.seeding_events.values())
        assert planted == 1

    def test_track_full_logs_false_skips_logger_but_keeps_active_counts(
        self, simulator: SequentialSimulator, mock_strategy: MockSeedingStrategy
    ) -> None:
        """track_full_logs=False must leave the logger empty while still populating active_counts."""
        result = simulator.perform_propagation(
            n_seeds=2, strategy=mock_strategy, max_epochs=20, track_full_logs=False
        )
        assert result.logger.get_detailed_logs() == {}
        assert len(result.active_counts) > 0

    def test_track_full_logs_false_does_not_change_metrics(
        self, toy_net: MultilayerNetwork, or_model: SQSMLTModel
    ) -> None:
        """extract_simulation_result must be identical regardless of track_full_logs."""
        from sqs.misc.result_handler import extract_simulation_result

        def _run(track_full_logs: bool) -> SequentialRunLog:
            sim = SequentialSimulator(or_model, toy_net.copy())
            return sim.perform_propagation(
                n_seeds=3,
                strategy=MockSeedingStrategy(),
                max_epochs=20,
                patience=3,
                track_full_logs=track_full_logs,
            )

        total_actors = toy_net.get_actors_num()
        with_logs = extract_simulation_result(_run(True), total_actors, n_seeds=3)
        without_logs = extract_simulation_result(_run(False), total_actors, n_seeds=3)
        assert with_logs == without_logs

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


@pytest.mark.parametrize("selector", ["deg-c-d", "voterank"])
@pytest.mark.parametrize("protocol", ["OR", "AND"])
def test_sequential_modes_never_below_ss(
    toy_net: MultilayerNetwork, selector: str, protocol: str
) -> None:
    """
    With a shared ranking, SQ and SQ_R reach at least the gain of SS.

    Each sequential seed is the best-ranked inactive actor, so once the budget is spent all top-k
    actors are active (seeded or organic); LTM is monotone, so the final state contains SS's.
    """
    from sqs.misc.result_handler import extract_simulation_result

    params = {"device": "cpu"} if selector == "voterank" else {}
    ranking = StaticSeedingStrategy(selector, "SS", selector_params=params).get_or_compute_ranking(
        toy_net
    )
    total = toy_net.get_actors_num()
    for mi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for n_seeds in range(1, 7):
            gain = {}
            for mode in ["SS", "SQ", "SQ_R"]:
                strategy = StaticSeedingStrategy(selector, mode, selector_params=params)
                strategy.set_precomputed_ranking(ranking)
                sim = SequentialSimulator(SQSMLTModel(protocol, mi), toy_net.copy())
                log = sim.perform_propagation(n_seeds, strategy, max_epochs=None, patience=2)
                gain[mode] = extract_simulation_result(log, total, n_seeds).gain
            assert gain["SQ"] >= gain["SS"] - 1e-9, (mi, n_seeds, gain)
            assert gain["SQ_R"] >= gain["SS"] - 1e-9, (mi, n_seeds, gain)
