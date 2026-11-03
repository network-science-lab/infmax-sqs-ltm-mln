"""Unit tests for sqs.result_handler."""

from pathlib import Path

from sqs.misc.result_handler import (
    SimulationFullResult,
    _build_expositions,
    _build_seedings,
    _compute_auc,
    append_result,
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
    def test_counts_newly_active_per_epoch(self) -> None:
        # cumulative active after each epoch: 1 -> 2 -> 2 -> 5
        assert _build_expositions([1, 2, 2, 5]) == "1;1;0;3"

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
            cpu_seconds=1.2,
            peak_rss_mb=420.0,
            peak_rss_delta_mb=42.0,
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
        assert full.cpu_seconds == 1.2
        assert full.peak_rss_mb == 420.0
        assert full.peak_rss_delta_mb == 42.0
        assert full.peak_gpu_mb is None


class TestAppendResult:
    def _make_result(self, rep: int) -> SimulationFullResult:
        partial_fields = dict(
            gain=1.0,
            auc=0.5,
            auc_total=0.4,
            simulation_length=3,
            seed_nb=1,
            saved_seeds=0,
            exposed_nb=2,
            unexposed_nb=1,
            expositions="1;1;0",
            seedings="[A];;",
        )
        return SimulationFullResult(
            **partial_fields,
            network="toy_network_piotr",
            protocol="OR",
            seed_budget=0.1,
            mi_value=0.15,
            ss_method="mock_run",
            ss_class="MockSeedingStrategy",
            ss_selector="",
            ss_mode="",
            rep=rep,
            wall_seconds=1.0,
            cpu_seconds=1.0,
            peak_rss_mb=1.0,
            peak_rss_delta_mb=1.0,
            peak_gpu_mb=None,
        )

    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        append_result(self._make_result(rep=0), path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("gain,")
        assert len(lines) == 2

    def test_appends_without_duplicating_header(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        append_result(self._make_result(rep=0), path)
        append_result(self._make_result(rep=1), path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines.count(lines[0]) == 1
        assert len(lines) == 3


def test_trailing_empty_epochs_are_cut() -> None:
    """Epochs after the last new activation (patience) do not enter expositions or AUC."""
    from sqs.runners.simulator import SequentialRunLog

    def log(extra_empty: int) -> SequentialRunLog:
        counts = {0: 2, 1: 5, 2: 6, **{3 + i: 6 for i in range(extra_empty)}}
        return SequentialRunLog(logger=None, seeding_events={0: ["a", "b"]}, active_counts=counts)

    short, long = (extract_simulation_result(log(n), total_actors=10, n_seeds=2) for n in (1, 5))
    assert short == long
    assert short.simulation_length == 2
    assert short.expositions == "2;3;1"
    assert short.seedings == "[a,b];;"
