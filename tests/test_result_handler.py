"""Unit tests for sqs.result_handler."""

from sqs.result_handler import (
    SimulationFullResult,
    _build_expositions,
    _build_seedings,
    extract_simulation_result,
)
from sqs.runners.seeding_strategy import MockSeedingStrategy
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
        local_stats = {
            0: [
                {"node_name": "A", "layer_name": "l1", "new_state": "1"},
                {
                    "node_name": "A",
                    "layer_name": "l2",
                    "new_state": "1",
                },  # same actor, different layer
                {"node_name": "B", "layer_name": "l1", "new_state": "0"},
            ],
            1: [
                {"node_name": "C", "layer_name": "l1", "new_state": "1"},
            ],
        }
        expositions = _build_expositions(local_stats, total_epochs=1)
        parts = expositions.split(";")
        # epoch 0: nodes A (deduplicated across layers) => 1 unique active node
        assert parts[0] == "1"
        # epoch 1: node C => 1 unique active node
        assert parts[1] == "1"

    def test_empty_epoch_yields_zero(self) -> None:
        expositions = _build_expositions({}, total_epochs=2)
        assert expositions == "0;0;0"


class TestExtractSimulationResult:
    def test_basic_metrics(self, toy_net, or_model) -> None:
        """gain, seed_nb, exposed_nb, unexposed_nb must be internally consistent."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=2, strategy=MockSeedingStrategy(), max_epochs=30, patience=3
        )
        total_actors = toy_net.get_actors_num()
        result = extract_simulation_result(run_log, total_actors)

        assert result.seed_nb == 2
        assert result.exposed_nb + result.unexposed_nb == total_actors
        assert 0.0 <= result.gain <= 100.0

    def test_seedings_format(self, toy_net, or_model) -> None:
        """Seedings must start with a bracket and contain the correct number of segments."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=2, strategy=MockSeedingStrategy(), max_epochs=20
        )
        result = extract_simulation_result(run_log, toy_net.get_actors_num())
        parts = result.seedings.split(";")
        # First segment must be a seeding event.
        assert parts[0].startswith("[") and parts[0].endswith("]")

    def test_expositions_segment_count_matches_seedings(self, toy_net, or_model) -> None:
        """Expositions and seedings must have the same number of ';'-separated segments."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=2, strategy=MockSeedingStrategy(), max_epochs=20
        )
        result = extract_simulation_result(run_log, toy_net.get_actors_num())
        assert len(result.seedings.split(";")) == len(result.expositions.split(";"))

    def test_full_result_from_partial(self, toy_net, or_model) -> None:
        """SimulationFullResult.from_partial must propagate all partial fields."""
        sim = SequentialSimulator(or_model, toy_net.copy())
        run_log = sim.perform_propagation(
            n_seeds=1, strategy=MockSeedingStrategy(), max_epochs=10
        )
        partial = extract_simulation_result(run_log, toy_net.get_actors_num())
        full = SimulationFullResult.from_partial(
            partial=partial,
            network="toy_network_piotr",
            protocol="OR",
            seed_budget=0.1,
            mi_value=0.15,
            ss_method="MockSeedingStrategy",
        )
        assert full.gain == partial.gain
        assert full.seedings == partial.seedings
        assert full.network == "toy_network_piotr"
