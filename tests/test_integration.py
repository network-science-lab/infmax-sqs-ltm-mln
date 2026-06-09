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
            "seeding_schedule": {
                "class": "MockSeedingStrategy",
                "max_epochs": 30,
                "patience": 3,
                "params": {},
            },
        },
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
        "simulation_length",
        "seed_nb",
        "exposed_nb",
        "unexposed_nb",
        "expositions",
        "seedings",
        "network",
        "protocol",
        "seed_budget",
        "mi_value",
        "ss_method",
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
        assert int(row["seed_nb"]) >= 1
        assert int(row["exposed_nb"]) >= 0
        assert int(row["unexposed_nb"]) >= 0
        # seedings: first segment must be a non-empty bracket
        first_seg = row["seedings"].split(";")[0]
        assert first_seg.startswith("[") and first_seg.endswith("]")
        # expositions: every segment must be a non-negative integer
        for seg in row["expositions"].split(";"):
            assert int(seg) >= 0
