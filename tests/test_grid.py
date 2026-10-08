"""Tests for sqs.grid: spec expansion, slicing and merging."""

import csv
import sys
from pathlib import Path

import pytest

from sqs.grid import jobs as grid_jobs
from sqs.grid.jobs import Job, merge_slices
from sqs.grid.spec import expand, load_spec, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # verify_grid_results

STATIC_SPEC = Path(__file__).resolve().parents[1] / "configs" / "grids" / "static.yaml"


def _job(out_dir: str, **kwargs) -> Job:
    """Build a small Static job writing to ``out_dir``."""
    fields = {
        "name": "static/full/or_er2_sq_deg_c_d",
        "config_name": "full/or_er2_sq_deg_c_d",
        "out_dir": out_dir,
        "schedule": {"name": "x", "class": "StaticSeedingStrategy", "params": {"mode": "SQ"}},
        "network": "er2",
        "mi_values": [0.1, 0.2],
        "seed_budgets": [0.05, 0.1],
        "repetitions": 1,
        "logs_dir": Path("logs"),
    }
    return Job(**{**fields, **kwargs})


def _write(root: Path, out_dir: str, header: list[str], rows: list[list[str]]) -> None:
    path = root / out_dir / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])


def test_static_spec_expansion() -> None:
    spec = load_spec(STATIC_SPEC)
    jobs = expand(spec)
    by_source = {s: [j for j in jobs if j.name.split("/")[1] == s] for s in ["full", "aucs", "arxiv", "timik"]}
    timik_modes = 3 - len(spec.exclude_modes.get("timik", []))  # SQ_R may be excluded
    assert {s: len(js) for s, js in by_source.items()} == {
        "full": 270,
        "aucs": 30,
        "arxiv": 30,
        "timik": 10 * timik_modes,
    }
    aucs = next(j for j in by_source["aucs"] if j.name == "static/aucs/and_sq_r_deg_c_d")
    assert aucs.config_name == "full_experiment_and_aucs"
    assert aucs.out_dir == "data/_static_results/aucs/and_sq_r_deg_c_d"
    assert aucs.seed_budgets == [0.15, 0.2, 0.25, 0.3, 0.35]
    assert aucs.expected_rows == 9 * 5 * 30
    assert "spreading.mi_values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]" in aucs.overrides


def test_exclude_modes(tmp_path: Path) -> None:
    spec = tmp_path / "static.yaml"
    spec.write_text(STATIC_SPEC.read_text() + "\nexclude:\n  timik:\n    modes:\n      - SQ_R\n")
    timik = [j for j in expand(load_spec(spec)) if j.name.startswith("static/timik/")]
    assert len(timik) == 20 and all(j.schedule["params"]["mode"] != "SQ_R" for j in timik)


@pytest.mark.parametrize(("by_budget", "n"), [(False, 2), (True, 4)])
def test_split(by_budget: bool, n: int) -> None:
    job = _job("data/_static_results/full/or_er2_sq_deg_c_d")
    slices = job.split(by_budget)
    assert len(slices) == n
    assert sum(s.expected_rows for s in slices) == job.expected_rows
    assert all(s.base_out_dir == job.out_dir for s in slices)
    assert all(s.out_dir.startswith("data/_static_results/_chunks/full/or_er2_sq_deg_c_d__mi") for s in slices)
    assert len({s.out_dir for s in slices}) == n


def test_select_does_not_split_jobs_with_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(grid_jobs, "ROOT", tmp_path)
    started = "data/_static_results/full/or_er2_sq_deg_c_d"
    _write(tmp_path, started, ["a"], [["1"]])
    jobs = select(STATIC_SPEC, ["static/full/or_er2_sq_*"], split="mi")
    assert [j.out_dir for j in jobs if j.base_out_dir is None] == [started]
    assert len(jobs) == 1 + 9 * 9  # started job + 9 other SQ / SQ_R jobs split per mi


def test_merge_slices(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(grid_jobs, "ROOT", tmp_path)
    base = _job("out/job")
    slices = base.split(by_budget=False)
    for i, piece in enumerate(slices):
        _write(tmp_path, piece.out_dir, ["mi", "x"], [[str(piece.mi_values[0]), str(i)]] * 2)

    merge_slices(slices)
    with (tmp_path / "out/job/results.csv").open(newline="", encoding="utf-8") as fh:
        assert list(csv.reader(fh)) == [["mi", "x"], ["0.1", "0"], ["0.1", "0"], ["0.2", "1"], ["0.2", "1"]]

    merge_slices(slices)  # target exists: warn, never overwrite
    assert "already exists" in capsys.readouterr().out


def test_rows_done_with_huge_fields(tmp_path: Path, monkeypatch) -> None:
    """Seedings of large networks exceed the default csv field limit."""
    monkeypatch.setattr(grid_jobs, "ROOT", tmp_path)
    job = _job("out/job")
    _write(tmp_path, "out/job", ["seedings"], [["1;" * 100_000]] * job.expected_rows)
    assert job.rows_done() == job.expected_rows


def test_merge_skips_incomplete_and_mismatched_headers(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(grid_jobs, "ROOT", tmp_path)
    slices = _job("out/job").split(by_budget=False)
    _write(tmp_path, slices[0].out_dir, ["mi", "x"], [["0.1", "0"]] * 2)
    merge_slices(slices)  # second slice missing
    assert not (tmp_path / "out/job/results.csv").exists()

    _write(tmp_path, slices[1].out_dir, ["mi", "y"], [["0.2", "0"]] * 2)
    merge_slices(slices)
    assert not (tmp_path / "out/job/results.csv").exists()
    assert "headers differ" in capsys.readouterr().out


def test_mode_violations(tmp_path: Path, monkeypatch) -> None:
    from verify_grid_results import mode_violations

    monkeypatch.setattr(grid_jobs, "ROOT", tmp_path)
    import verify_grid_results

    monkeypatch.setattr(verify_grid_results, "ROOT", tmp_path)
    header = ["network", "protocol", "mi_value", "seed_budget", "ss_selector", "rep", "ss_mode", "gain"]
    jobs = []
    for mode, gain in [("SS", "50"), ("SQ", "60"), ("SQ_R", "40")]:
        job = _job(f"out/{mode}", schedule={"name": mode, "class": "StaticSeedingStrategy", "params": {"mode": mode}})
        _write(tmp_path, job.out_dir, header, [["er2", "OR", "0.1", "0.05", "deg-c-d", "0", mode, gain]])
        jobs.append(job)
    bad = mode_violations(jobs)
    assert len(bad) == 1 and bad.iloc[0]["SQ_R"] == 40


def test_strategies_spec_selectors() -> None:
    """Only the listed selectors are expanded, with per-protocol mi values."""
    jobs = expand(load_spec(STATIC_SPEC.parent / "strategies.yaml"))
    assert {j.schedule["params"]["selector"] for j in jobs} == {"deg-c-d", "nghb-sd", "ts-net"}
    or_job = next(j for j in jobs if j.name == "subnet/full/or_er2_sq_deg_c_d")
    assert or_job.mi_values == [0.7, 0.8, 0.9] and or_job.seed_budgets == [0.1, 0.15, 0.2, 0.25]
