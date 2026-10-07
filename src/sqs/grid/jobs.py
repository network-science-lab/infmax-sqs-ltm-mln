"""Grid jobs: cost estimates, slicing, merging and progress on disk."""

import csv
import json
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS_FILE = "results.csv"
MODES = {"SS", "SQ", "SQ_R"}

# Seconds per rep of Static by (network, mode), estimated by test simulations simulator
SECONDS_PER_REP = {
    "aucs": {"SS": 0.01, "SQ": 0.01, "SQ_R": 0.01},
    "ckm_physicians": {"SS": 0.01, "SQ": 0.02, "SQ_R": 0.02},
    "er2": {"SS": 0.05, "SQ": 0.08, "SQ_R": 0.06},
    "er3": {"SS": 0.1, "SQ": 0.1, "SQ_R": 0.09},
    "er5": {"SS": 0.1, "SQ": 0.16, "SQ_R": 0.17},
    "l2_course_net_1": {"SS": 0.01, "SQ": 0.02, "SQ_R": 0.01},
    "lazega": {"SS": 0.01, "SQ": 0.02, "SQ_R": 0.02},
    "sf2": {"SS": 0.05, "SQ": 0.09, "SQ_R": 0.07},
    "sf3": {"SS": 0.054, "SQ": 0.09, "SQ_R": 0.09},
    "sf5": {"SS": 0.12, "SQ": 0.15, "SQ_R": 0.15},
    "arxiv_netscience_coauthorship": {"SS": 0.73, "SQ": 0.92, "SQ_R": 0.95},
    "timik1q2009": {"SS": 7.2, "SQ": 12.62, "SQ_R": 12.62},  # SQ_R not run: as SQ
}
# Cost per (strategy, mode) relative to Static, from er2/ckm/sf3 smoke runs
STRATEGY_WEIGHT = {
    ("StaticSeedingStrategy", "SS"): 1.0,
    ("StaticSeedingStrategy", "SQ"): 1.0,
    ("StaticSeedingStrategy", "SQ_R"): 1.0,
    ("SubnetSeedingStrategy", "SS"): 1.0,
    ("SubnetSeedingStrategy", "SQ"): 20.0,
    ("SubnetSeedingStrategy", "SQ_R"): 1.6,
    ("AdaptiveSeedingStrategy", "SS"): 1.0,
    ("AdaptiveSeedingStrategy", "SQ"): 3.5,
    ("AdaptiveSeedingStrategy", "SQ_R"): 1.2,
}


@dataclass
class Job:
    """One ``sqs`` run: a Static base config with one schedule and the grid's mi / budgets."""

    name: str
    config_name: str
    out_dir: str
    schedule: dict[str, Any]
    network: str
    mi_values: list[float]
    seed_budgets: list[float]
    repetitions: int
    logs_dir: Path
    base_out_dir: str | None = None  # slices only: where merged results end up

    @property
    def overrides(self) -> list[str]:
        """Hydra overrides applied on top of ``config_name``."""
        return [
            f"seeding_schedules=[{to_override_value(self.schedule)}]",
            f"spreading.mi_values={to_override_value(self.mi_values)}",
            f"spreading.seed_budgets={to_override_value(self.seed_budgets)}",
            f"run.out_dir={self.out_dir}",
        ]

    @property
    def expected_rows(self) -> int:
        """Rows of a complete ``results.csv``."""
        return len(self.mi_values) * len(self.seed_budgets) * self.repetitions

    @property
    def weight_seconds(self) -> float:
        """Estimated single-CPU run time."""
        mode = self.schedule["params"]["mode"]
        if self.network not in SECONDS_PER_REP:
            raise KeyError(f"No timing for network {self.network!r} in SECONDS_PER_REP")
        per_rep = SECONDS_PER_REP[self.network][mode]
        return self.expected_rows * per_rep * STRATEGY_WEIGHT[(self.schedule["class"], mode)]

    def rows_done(self) -> int:
        """Rows already in ``results.csv`` (0 if missing or unreadable)."""
        path = ROOT / self.out_dir / RESULTS_FILE
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                return max(0, sum(1 for _ in csv.reader(fh)) - 1)
        except (OSError, csv.Error):
            return 0

    def _chunks_dir(self) -> Path:
        """Directory holding this job's slices (``<results_root>/_chunks/<source>``)."""
        base = Path(self.out_dir)
        return base.parent.parent / "_chunks" / base.parent.name

    def has_slices(self) -> bool:
        """Whether slices of this job were started (e.g. by an earlier ``--split`` run)."""
        return any((ROOT / self._chunks_dir()).glob(f"{Path(self.out_dir).name}__mi*"))

    def split(self, by_budget: bool) -> list["Job"]:
        """Slices per mi value (and per seed budget) under ``<results_root>/_chunks/``."""
        base = Path(self.out_dir)
        chunks = self._chunks_dir()
        budgets: list[Any] = self.seed_budgets if by_budget else [None]
        slices = []
        for mi in self.mi_values:
            for budget in budgets:
                suffix = f"mi{mi}" + (f"__b{budget}" if budget is not None else "")
                slices.append(
                    replace(
                        self,
                        name=f"{self.name}@{suffix}",
                        out_dir=str(chunks / f"{base.name}__{suffix}"),
                        mi_values=[mi],
                        seed_budgets=self.seed_budgets if budget is None else [budget],
                        base_out_dir=self.out_dir,
                    )
                )
        return slices


def to_override_value(value: Any) -> str:
    """Render a value as a Hydra override value (also valid YAML)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {to_override_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(to_override_value(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or isinstance(value, (int, float)):
        return "null" if value is None else repr(value)
    return json.dumps(str(value))


def merge_slices(jobs: list[Job]) -> None:
    """Concatenate complete slices into their job's ``results.csv`` (never overwrites one)."""
    by_base: dict[str, list[Job]] = {}
    for job in jobs:
        if job.base_out_dir is not None:
            by_base.setdefault(job.base_out_dir, []).append(job)
    for base_out_dir, slices in sorted(by_base.items()):
        if any(s.rows_done() < s.expected_rows for s in slices):
            continue
        target = ROOT / base_out_dir / RESULTS_FILE
        if target.exists():
            print(f"WARNING: not merging {len(slices)} slices, {target} already exists")
            continue
        rows: list[list[str]] = []
        headers = set()
        for piece in slices:
            with (ROOT / piece.out_dir / RESULTS_FILE).open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                headers.add(tuple(next(reader)))
                rows.extend(reader)
        if len(headers) != 1:
            print(f"WARNING: not merging slices of {base_out_dir}, their headers differ")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(headers.pop())
            writer.writerows(rows)
        print(f"merged {len(slices)} slices -> {target}")


def running_out_dirs() -> set[str]:
    """Return ``run.out_dir`` values of running ``sqs`` processes."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return {m.group(1) for m in re.finditer(r"run\.out_dir=(\S+)", out)}
