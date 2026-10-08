"""Grid spec (``configs/grids/*.yaml``): validation and expansion into jobs."""

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sqs.grid.jobs import MODES, ROOT, STRATEGY_WEIGHT, Job

CONFIGS_DIR = ROOT / "configs"


@dataclass(frozen=True)
class GridSpec:
    """Validated grid spec."""

    sources: dict[str, str]  # source name -> glob over config names
    strategies: dict[str, dict[str, str]]  # class -> {prefix, results_root}
    modes: list[str]
    mi_values: dict[str, list[float]]  # per protocol
    seed_budgets: dict[str, list[float]]  # per protocol
    exclude_modes: dict[str, list[str]]  # per source
    selectors: list[str] | None  # selectors to run; None: all of the base configs


def load_spec(path: Path) -> GridSpec:
    """Load and validate a grid spec file."""
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    known = {"sources", "strategies", "modes", "mi_values", "seed_budgets", "exclude", "selectors"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    exclude = {src: list(rule.get("modes", [])) for src, rule in (raw.get("exclude") or {}).items()}
    spec = GridSpec(
        sources={str(k): str(v) for k, v in raw["sources"].items()},
        strategies={cls: dict(v) for cls, v in raw["strategies"].items()},
        modes=list(raw["modes"]),
        mi_values=_per_protocol(raw["mi_values"]),
        seed_budgets={p: [float(b) for b in v] for p, v in raw["seed_budgets"].items()},
        exclude_modes=exclude,
        selectors=raw.get("selectors"),
    )
    known_classes = {cls for cls, _ in STRATEGY_WEIGHT}
    if set(spec.strategies) - known_classes:
        raise ValueError(f"{path}: unknown strategies, known: {sorted(known_classes)}")
    if set(spec.modes + [m for ms in exclude.values() for m in ms]) - MODES:
        raise ValueError(f"{path}: unknown modes, known: {sorted(MODES)}")
    if set(exclude) - set(spec.sources):
        raise ValueError(f"{path}: exclude refers to unknown sources")
    for pattern in spec.sources.values():
        config_names(pattern)  # raises if nothing matches
    return spec


def _per_protocol(values: Any) -> dict[str, list[float]]:
    """Read a list shared by both protocols, or a mapping protocol -> list."""
    if isinstance(values, dict):
        return {proto: [float(v) for v in vs] for proto, vs in values.items()}
    return {proto: [float(v) for v in values] for proto in ("OR", "AND")}


def config_names(pattern: str) -> list[str]:
    """Return config names (``configs/`` relative, no suffix) matching a glob."""
    names = sorted(
        p.relative_to(CONFIGS_DIR).with_suffix("").as_posix() for p in CONFIGS_DIR.rglob("*.yaml")
    )
    hits = [n for n in names if fnmatch.fnmatch(n, pattern)]
    if not hits:
        raise ValueError(f"Pattern {pattern!r} matched no config in {CONFIGS_DIR}")
    return hits


def expand(spec: GridSpec, device: str | None = None) -> list[Job]:
    """
    Expand a spec into one job per (strategy, Static base schedule).

    A base config with several schedules (e.g. aucs) yields one job per schedule, named
    ``<protocol>_<schedule>``; a single-schedule config keeps its file stem.
    ``device`` replaces ``selector_params.device`` (e.g. ``cpu`` without a GPU).
    """
    jobs = []
    for source, pattern in spec.sources.items():
        for config_name in config_names(pattern):
            with (CONFIGS_DIR / f"{config_name}.yaml").open(encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            schedules = cfg["seeding_schedules"]
            (protocol,) = cfg["spreading"]["protocols"]
            if any(s["class"] != "StaticSeedingStrategy" for s in schedules):
                raise ValueError(f"{config_name}: grid base configs must hold Static schedules")
            for base in schedules:
                mode = base["params"]["mode"]
                if mode not in spec.modes or mode in spec.exclude_modes.get(source, []):
                    continue
                if spec.selectors is not None and base["params"]["selector"] not in spec.selectors:
                    continue
                stem = (
                    config_name.rsplit("/", 1)[-1]
                    if len(schedules) == 1
                    else f"{protocol.lower()}_{base['name']}"
                )
                params = _with_device(base["params"], device)
                for cls, strategy in spec.strategies.items():
                    root = strategy["results_root"]
                    jobs.append(
                        Job(
                            name=f"{strategy['prefix']}/{source}/{stem}",
                            config_name=config_name,
                            out_dir=f"{root}/{source}/{stem}",
                            schedule={
                                # Static keeps the original names (ss_deg_c_d, ...)
                                "name": base["name"]
                                if cls == "StaticSeedingStrategy"
                                else f"{strategy['prefix']}_{base['name']}",
                                "class": cls,
                                "params": params,
                            },
                            network=cfg["networks"][0],
                            mi_values=spec.mi_values[protocol],
                            seed_budgets=spec.seed_budgets[protocol],
                            repetitions=cfg["spreading"]["repetitions"],
                            logs_dir=ROOT / root / "_launcher_logs",
                        )
                    )
    return jobs


def _with_device(params: dict[str, Any], device: str | None) -> dict[str, Any]:
    """Return ``params`` with ``selector_params.device`` replaced, if both are set."""
    selector_params = params.get("selector_params") or {}
    if device is None or "device" not in selector_params:
        return params
    return {**params, "selector_params": {**selector_params, "device": device}}


def select(
    spec_path: Path, patterns: list[str], device: str | None = None, split: str = "none"
) -> list[Job]:
    """Return the spec's jobs whose names match any glob, optionally split into slices."""
    jobs = expand(load_spec(spec_path), device)
    picked = []
    for pattern in patterns:
        hits = [j for j in jobs if fnmatch.fnmatch(j.name, pattern)]
        if not hits:
            raise ValueError(f"Pattern {pattern!r} matched no job in {spec_path}")
        picked.extend(j for j in hits if j not in picked)
    if split == "none":
        for job in picked:
            if not job.rows_done() and job.has_slices():
                print(f"WARNING: {job.name} has slices from --split; unsplit it starts over")
    else:
        # Jobs that already have results (complete, or started without --split) run unsplit
        picked = [
            s
            for j in picked
            for s in ([j] if j.rows_done() else j.split(by_budget=split == "mi-budget"))
        ]
    return sorted(picked, key=lambda j: j.name)
