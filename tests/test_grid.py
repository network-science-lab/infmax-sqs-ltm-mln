"""Tests for sqs.grid: spec expansion, slicing and merging."""

import csv
import sys
from pathlib import Path

import pytest
import yaml

from sqs.grid import jobs as grid_jobs
from sqs.grid import spec as grid_spec
from sqs.grid.jobs import Job, merge_slices
from sqs.grid.spec import expand, load_spec, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # verify_grid_results

SCHEDULES = [(sel, mode) for sel in ["deg-c-d", "nghb-sd"] for mode in ["SS", "SQ", "SQ_R"]]


def _schedule(selector: str, mode: str) -> dict:
    name = f"{mode.lower()}_{selector.replace('-', '_')}"
    return {"name": name, "class": "StaticSeedingStrategy", "params": {"selector": selector, "mode": mode}}


def _dump(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture()
def configs(tmp_path: Path, monkeypatch) -> Path:
    """Base configs: ``full/<proto>_er2_<schedule>`` (one schedule each), ``multi_<proto>`` (all)."""
    root = tmp_path / "configs"
    for proto in ("OR", "AND"):
        base = {"networks": ["er2"], "spreading": {"protocols": [proto], "repetitions": 3}}
        for selector, mode in SCHEDULES:
            schedule = _schedule(selector, mode)
            name = f"{proto.lower()}_er2_{schedule['name']}"
            _dump(root / "full" / f"{name}.yaml", {**base, "seeding_schedules": [schedule]})
        multi = [_schedule(*s) for s in SCHEDULES]
        _dump(root / f"multi_{proto.lower()}.yaml", {**base, "seeding_schedules": multi})
    monkeypatch.setattr(grid_spec, "CONFIGS_DIR", root)
    return root


def _spec(tmp_path: Path, **overrides) -> Path:
    """Static spec over the ``configs`` fixture."""
    raw = {
        "sources": {"full": "full/*", "multi": "multi_*"},
        "strategies": {"StaticSeedingStrategy": {"prefix": "static", "results_root": "res"}},
        "modes": ["SS", "SQ", "SQ_R"],
        "mi_values": [0.1, 0.2],
        "seed_budgets": {"OR": [0.05, 0.1], "AND": [0.2, 0.3, 0.4]},
    }
    return _dump(tmp_path / "spec.yaml", {**raw, **overrides})


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


def test_spec_expansion(configs: Path, tmp_path: Path) -> None:
    jobs = expand(load_spec(_spec(tmp_path)))
    by_source = {s: [j for j in jobs if j.name.split("/")[1] == s] for s in ["full", "multi"]}
    assert {s: len(js) for s, js in by_source.items()} == {"full": 12, "multi": 12}
    multi = next(j for j in by_source["multi"] if j.name == "static/multi/and_sq_r_deg_c_d")
    assert multi.config_name == "multi_and"
    assert multi.out_dir == "res/multi/and_sq_r_deg_c_d"
    assert multi.schedule["name"] == "sq_r_deg_c_d"
    assert multi.seed_budgets == [0.2, 0.3, 0.4]
    assert multi.expected_rows == 2 * 3 * 3
    assert "spreading.mi_values=[0.1,0.2]" in multi.overrides


def test_spec_overrides_reach_every_job(configs: Path, tmp_path: Path) -> None:
    jobs = expand(load_spec(_spec(tmp_path, overrides=["+spreading.ranking_replicas=true"])))
    assert all(j.overrides[-1] == "+spreading.ranking_replicas=true" for j in jobs)
    assert all(s.overrides[-1] == "+spreading.ranking_replicas=true" for s in jobs[0].split(True))


def test_exclude_modes(configs: Path, tmp_path: Path) -> None:
    jobs = expand(load_spec(_spec(tmp_path, exclude={"multi": {"modes": ["SQ_R"]}})))
    multi = [j for j in jobs if j.name.startswith("static/multi/")]
    assert len(multi) == 8 and all(j.schedule["params"]["mode"] != "SQ_R" for j in multi)
    assert len(jobs) - len(multi) == 12


def test_selectors_and_mi_per_protocol(configs: Path, tmp_path: Path) -> None:
    """Only the listed selectors are expanded; other strategies get prefixed names."""
    spec = _spec(
        tmp_path,
        strategies={"AdaptiveSeedingStrategy": {"prefix": "adaptive", "results_root": "ad"}},
        modes=["SQ", "SQ_R"],
        selectors=["nghb-sd"],
        mi_values={"OR": [0.7], "AND": [0.5, 0.6]},
    )
    jobs = expand(load_spec(spec))
    assert {j.schedule["params"]["selector"] for j in jobs} == {"nghb-sd"}
    or_job = next(j for j in jobs if j.name == "adaptive/full/or_er2_sq_nghb_sd")
    assert or_job.schedule["name"] == "adaptive_sq_nghb_sd"
    assert or_job.mi_values == [0.7] and or_job.seed_budgets == [0.05, 0.1]


@pytest.mark.parametrize(("by_budget", "n"), [(False, 2), (True, 4)])
def test_split(by_budget: bool, n: int) -> None:
    job = _job("data/_static_results/full/or_er2_sq_deg_c_d")
    slices = job.split(by_budget)
    assert len(slices) == n
    assert sum(s.expected_rows for s in slices) == job.expected_rows
    assert all(s.base_out_dir == job.out_dir for s in slices)
    assert all(s.out_dir.startswith("data/_static_results/_chunks/full/or_er2_sq_deg_c_d__mi") for s in slices)
    assert len({s.out_dir for s in slices}) == n


def test_select_does_not_split_jobs_with_results(configs: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(grid_jobs, "ROOT", tmp_path)
    started = "res/full/or_er2_sq_deg_c_d"
    _write(tmp_path, started, ["a"], [["1"]])
    jobs = select(_spec(tmp_path), ["static/full/or_er2_sq_*"], split="mi")
    assert [j.out_dir for j in jobs if j.base_out_dir is None] == [started]
    assert len(jobs) == 1 + 3 * 2  # started job + 3 other SQ / SQ_R jobs split per mi


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
