"""Unit tests for sqs.result_handler."""

from sqs.misc.result_handler import (
    SimulationFullResult,
    _build_expositions,
    _build_seedings,
    _compute_auc,
    extract_simulation_result,
)
from sqs.runners.strategies import MockSeedingStrategy
from sqs.runners.simulator import SequentialSimulator


class TestBuildSeedings:
    def test_single_seed_at_epoch_zero(self) -> None:
        seedings = _build_seedings({0: ["A"]}, total_epochs=3)
        parts = seedings.split(";")
        assert parts[0] == "[A]"
        assert all(p == "" for p in parts[1:])

    def test_seeds_at_non_consecutive_epochs(self) -> None:
        seedings = _build_seedings({0: ["A", "B"], 3: ["C"]}, total_epochs=3)
        parts = seedings.split(";")
        assert parts[0] == "[A,B]"
        assert parts[1] == ""
        assert parts[2] == ""
        assert parts[3] == "[C]"

    def test_no_seedings(self) -> None:
        seedings = _build_seedings({}, total_epochs=2)
        assert seedings == ";;"


class TestBuildExpositions:
    def test_counts_active_nodes_per_epoch(self) -> None:
        # epoch 0: full LTM snapshot -- A active (2 layers, deduplicated), B inactive
        # epoch 1: seeding epoch -- C seeded (partial log, handled via seeding_events)
        local_stats = {
            0: [
                {"node_name": "A", "layer_name": "l1", "new_state": "1"},
                {"node_name": "A", "layer_name": "l2", "new_state": "1"},
                {"node_name": "B", "layer_name": "l1", "new_state": "0"},
            ],
            1: [
                {"node_name": "C", "layer_name": "l1", "new_state": "1"},
            ],
        }
        expositions = _build_expositions(
            local_stats, total_epochs=1, seeding_events={1: ["C"]}
        )
        parts = expositions.split(";")
        assert parts[0] == "1"  # epoch 0 (LTM): A newly active
        assert parts[1] == "1"  # epoch 1 (seeding): C seeded

    def test_empty_epoch_yields_zero(self) -> None:
        expositions = _build_expositions({}, total_epochs=2, seeding_events={})
        assert expositions == "0;0;0"


class TestComputeAuc:
    def test_single_point_returns_zero(self) -> None:
        # single epoch -> no area under curve
        assert _compute_auc("5", norm=5) == 0.0

    def test_uniform_spread_auc(self) -> None:
        # 1 actor per epoch for 4 epochs, norm=4 -> y=[0.25,0.5,0.75,1.0]
        # x=[0,1/3,2/3,1] -> trapezoid ~0.625
        result = _compute_auc("1;1;1;1", norm=4)
        assert 0.0 < result < 1.0

    def test_zero_norm_returns_zero(self) -> None:
        assert _compute_auc("0;0;0", norm=0) == 0.0

    def test_auc_within_bounds(self) -> None:
        assert 0.0 <= _compute_auc("2;3;1;0", norm=6) <= 1.0

    def test_auc_total_le_auc(self) -> None:
        # auc_total uses larger denominator -> y values are smaller -> area is smaller
        auc = _compute_auc("1;2;1", norm=4)       # exposed_nb=4
        auc_total = _compute_auc("1;2;1", norm=10)  # total_actors=10
        assert auc_total <= auc


class TestExtractSimulationResult:
    def test_basic_metrics(self, toy_net, or_model) -> None:
        """gain, seed_nb, exposed_nb, unexposed_nb must be internally consistent."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=2, strategy=MockSeedingStrategy(), max_epochs=30, patience=3
        )
        total_actors = toy_net.get_actors_num()
        result = extract_simulation_result(run_log, total_actors, n_seeds=2)

        assert result.seed_nb == 2
        assert result.saved_seeds == 0  # MockSeedingStrategy always uses the full budget
        assert result.exposed_nb + result.unexposed_nb == total_actors
        assert 0.0 <= result.gain <= 100.0
        assert 0.0 <= result.auc <= 1.0
        assert 0.0 <= result.auc_total <= result.auc

    def test_saved_seeds(self, toy_net, or_model) -> None:
        """saved_seeds must equal n_seeds minus the seeds actually planted."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=2, strategy=MockSeedingStrategy(), max_epochs=20
        )
        result = extract_simulation_result(run_log, toy_net.get_actors_num(), n_seeds=10)
        assert result.saved_seeds == 10 - result.seed_nb

    def test_seedings_format(self, toy_net, or_model) -> None:
        """Seedings must start with a bracket and contain the correct number of segments."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=2, strategy=MockSeedingStrategy(), max_epochs=20
        )
        result = extract_simulation_result(run_log, toy_net.get_actors_num(), n_seeds=2)
        parts = result.seedings.split(";")
        # First segment must be a seeding event.
        assert parts[0].startswith("[") and parts[0].endswith("]")

    def test_expositions_segment_count_matches_seedings(self, toy_net, or_model) -> None:
        """Expositions and seedings must have the same number of ';'-separated segments."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=2, strategy=MockSeedingStrategy(), max_epochs=20
        )
        result = extract_simulation_result(run_log, toy_net.get_actors_num(), n_seeds=2)
        assert len(result.seedings.split(";")) == len(result.expositions.split(";"))

    def test_full_result_from_partial(self, toy_net, or_model) -> None:
        """SimulationFullResult.from_partial must propagate all partial fields."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=1, strategy=MockSeedingStrategy(), max_epochs=10
        )
        partial = extract_simulation_result(run_log, toy_net.get_actors_num(), n_seeds=1)
        full = SimulationFullResult.from_partial(
            partial=partial,
            network="toy_network_piotr",
            protocol="OR",
            seed_budget=0.1,
            mi_value=0.15,
            ss_method="mock_run",
            ss_class="MockSeedingStrategy",
            ss_selector="",
            ss_mode="",
            rep=7,
            wall_seconds=1.23,
            rss_delta_mb=42.0,
            peak_gpu_mb=None,
        )
        assert full.gain == partial.gain
        assert full.auc == partial.auc
        assert full.auc_total == partial.auc_total
        assert full.saved_seeds == partial.saved_seeds
        assert full.seedings == partial.seedings
        assert full.network == "toy_network_piotr"
        assert full.ss_class == "MockSeedingStrategy"
        assert full.ss_selector == ""
        assert full.ss_mode == ""
        assert full.rep == 7
        assert full.wall_seconds == 1.23
        assert full.rss_delta_mb == 42.0
        assert full.peak_gpu_mb is None
