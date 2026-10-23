"""Integration test: full spreading pipeline on toy networks."""

import csv
from pathlib import Path

import pytest
import threadpoolctl

from sqs.pipeline import spreading


@pytest.fixture()
def spreading_config(tmp_path: Path) -> dict:
    """Minimal spreading config using toy networks, writing output to tmp_path."""
    return {
        "networks": ["toy_network_piotr"],
        "run": {"experiment_type": "spreading", "out_dir": str(tmp_path)},
        "spreading": {
            "protocols": ["OR"],
            "mi_values": [0.15],
            "seed_budgets": [0.1],
            "repetitions": 2,
            "max_epochs": 30,
            "patience": 3,
        },
        "seeding_schedules": [
            {
                "name": "mock",
                "class": "MockSeedingStrategy",
                "params": {},
            }
        ],
    }


def test_spreading_pipeline_creates_csv(spreading_config: dict, tmp_path: Path) -> None:
    """Pipeline must write results.csv to out_dir with the expected columns."""
    spreading.run(spreading_config)

    csv_path = tmp_path / "results.csv"
    assert csv_path.exists(), "results.csv was not created"

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        rows = list(reader)

    expected_columns = {
        "gain",
        "auc",
        "auc_total",
        "simulation_length",
        "seed_nb",
        "saved_seeds",
        "exposed_nb",
        "unexposed_nb",
        "expositions",
        "seedings",
        "network",
        "protocol",
        "seed_budget",
        "mi_value",
        "ss_method",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_mb",
        "peak_rss_delta_mb",
        "peak_gpu_mb",
    }
    assert expected_columns.issubset(set(columns)), (
        f"Missing columns: {expected_columns - set(columns)}"
    )
    assert len(rows) == 2, f"Expected 2 rows (repetitions=2), got {len(rows)}"


def test_spreading_pipeline_row_values(spreading_config: dict, tmp_path: Path) -> None:
    """Each result row must have valid numeric fields and parseable seedings/expositions."""
    spreading.run(spreading_config)

    with (tmp_path / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        assert float(row["gain"]) >= 0.0
        assert 0.0 <= float(row["auc"]) <= 1.0
        assert 0.0 <= float(row["auc_total"]) <= 1.0
        assert int(row["seed_nb"]) >= 1
        assert int(row["exposed_nb"]) >= 0
        assert int(row["unexposed_nb"]) >= 0
        assert float(row["wall_seconds"]) >= 0.0
        assert float(row["cpu_seconds"]) >= 0.0
        assert float(row["peak_rss_mb"]) > 0.0
        assert float(row["peak_rss_delta_mb"]) >= 0.0
        # peak_gpu_mb is empty string on CPU-only runs
        assert row["peak_gpu_mb"] == "" or float(row["peak_gpu_mb"]) >= 0.0
        # seedings: first segment must be a non-empty bracket
        first_seg = row["seedings"].split(";")[0]
        assert first_seg.startswith("[") and first_seg.endswith("]")
        # expositions: every segment must be a non-negative integer
        for seg in row["expositions"].split(";"):
            assert int(seg) >= 0


def test_spreading_pipeline_applies_max_threads(
    spreading_config: dict, tmp_path: Path
) -> None:
    """run.max_threads must cap BLAS thread pools in-process, not just document intent."""
    spreading_config["run"]["max_threads"] = 1

    spreading.run(spreading_config)

    assert all(info["num_threads"] == 1 for info in threadpoolctl.threadpool_info())


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read all data rows of a results CSV."""
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_spreading_pipeline_resumes_missing_reps(spreading_config: dict, tmp_path: Path) -> None:
    """Resume computes only missing reps, identical to an uninterrupted run."""
    spreading.run(spreading_config)
    csv_path = tmp_path / "results.csv"
    full_rows = _read_rows(csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines(keepends=True)
    csv_path.write_text("".join(lines[:2]), encoding="utf-8")

    spreading.run(spreading_config)

    resumed_rows = _read_rows(csv_path)
    assert [r["rep"] for r in resumed_rows] == ["0", "1"]
    assert resumed_rows[1]["seedings"] == full_rows[1]["seedings"]
    assert resumed_rows[1]["gain"] == full_rows[1]["gain"]


def test_spreading_pipeline_repairs_truncated_trailing_row(
    spreading_config: dict, tmp_path: Path
) -> None:
    """A truncated last row is dropped and recomputed."""
    spreading.run(spreading_config)
    csv_path = tmp_path / "results.csv"
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text[: len(text) - 20], encoding="utf-8")

    spreading.run(spreading_config)

    rows = _read_rows(csv_path)
    assert [r["rep"] for r in rows] == ["0", "1"]
    assert all(r["peak_gpu_mb"] is not None for r in rows), "row has missing trailing fields"


def test_monitor_resources_measures_peak_inside_block() -> None:
    """A temporary allocation freed before the block ends must still show up in the peak."""
    from sqs.misc.resources import monitor_resources

    with monitor_resources() as usage:
        block = bytearray(200 * 1024 * 1024)
        del block
    assert usage.peak_rss_delta_mb > 150
    assert usage.peak_rss_mb >= usage.peak_rss_delta_mb
    assert usage.peak_gpu_mb is None


def test_rankings_cached_per_rep_for_mds(spreading_config: dict, tmp_path: Path, monkeypatch) -> None:
    """mds: one cached ranking per rep shared by all modes; deg-c: one ranking for all reps."""
    monkeypatch.setattr(spreading, "_RANKINGS_DIR", tmp_path / "rankings")
    spreading_config["seeding_schedules"] = [
        {"name": f"{mode}_{sel}", "class": "StaticSeedingStrategy", "params": {"selector": sel, "mode": mode}}
        for sel in ["mds", "deg-c"]
        for mode in ["SS", "SQ"]
    ]
    spreading.run(spreading_config)

    files = sorted(p.name for p in (tmp_path / "rankings" / "toy_network_piotr").iterdir())
    assert files == ["deg-c.json", "mds__rep0.json", "mds__rep1.json"]
    assert len(_read_rows(tmp_path / "results.csv")) == 4 * 2


def test_static_uses_ranking_replica_per_rep(spreading_config: dict, tmp_path: Path, monkeypatch) -> None:
    """With ``ranking_replicas`` rep 0 seeds from the v2 ranking, rep r from replica r."""
    import json

    rankings = tmp_path / "v2" / "toy_network_piotr"
    rankings.mkdir(parents=True)
    monkeypatch.setattr(spreading, "_REPLICAS_DIR", tmp_path / "v2")
    actors = list(range(1, 12))
    for name, order in [("deg-c-d.json", actors), ("deg-c-d__rep1.json", actors[::-1])]:
        (rankings / name).write_text(json.dumps([{"actor": a, "score": 0} for a in order]), encoding="utf-8")
    spreading_config["spreading"]["ranking_replicas"] = True
    spreading_config["seeding_schedules"] = [
        {"name": f"{mode}_deg_c_d", "class": "StaticSeedingStrategy", "params": {"selector": "deg-c-d", "mode": mode}}
        for mode in ["SS", "SQ"]
    ]
    spreading.run(spreading_config)

    first = {(r["ss_mode"], r["rep"]): r["seedings"].split(";")[0] for r in _read_rows(tmp_path / "results.csv")}
    assert first == {("SS", "0"): "[1]", ("SQ", "0"): "[1]", ("SS", "1"): "[11]", ("SQ", "1"): "[11]"}


def test_missing_replica_raises(spreading_config: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(spreading, "_REPLICAS_DIR", tmp_path / "none")
    spreading_config["spreading"]["ranking_replicas"] = True
    spreading_config["seeding_schedules"] = [
        {"name": "ss_deg_c_d", "class": "StaticSeedingStrategy", "params": {"selector": "deg-c-d", "mode": "SS"}}
    ]
    with pytest.raises(FileNotFoundError, match="replicate_rankings"):
        spreading.run(spreading_config)
