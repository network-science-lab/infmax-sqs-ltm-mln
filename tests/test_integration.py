"""Integration test: full spreading pipeline on toy networks."""

import csv
from pathlib import Path

import pytest

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
        "peak_cpu_mb",
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
        assert float(row["peak_cpu_mb"]) >= 0.0  # may be negative on tiny runs, but >=0 in practice
        # peak_gpu_mb is empty string on CPU-only runs
        assert row["peak_gpu_mb"] == "" or float(row["peak_gpu_mb"]) >= 0.0
        # seedings: first segment must be a non-empty bracket
        first_seg = row["seedings"].split(";")[0]
        assert first_seg.startswith("[") and first_seg.endswith("]")
        # expositions: every segment must be a non-negative integer
        for seg in row["expositions"].split(";"):
            assert int(seg) >= 0
