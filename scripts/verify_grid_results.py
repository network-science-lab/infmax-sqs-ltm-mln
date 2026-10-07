#!/usr/bin/env python3
"""
Verify that every matched grid job has actually finished computing its results.

Cross-checks each job's expected row count (mi_values x seed_budgets x repetitions) against its
``results.csv``. Rows are keyed by (network, protocol, mi_value, seed_budget, ss_method, rep);
duplicate keys (e.g. from two overlapping launches writing the same file) are removed in place by
default, keeping the first occurrence.

Usage:
    uv run python scripts/verify_grid_results.py configs/grids/static.yaml "static/*"
    uv run python scripts/verify_grid_results.py configs/grids/static.yaml "*" --check-only

Exits with status 1 if any matched job is pending, incomplete, unreadable, or has more rows
than expected once duplicates are resolved; exits 0 only when every matched job has exactly
the expected number of rows.
"""

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sqs.grid.jobs import RESULTS_FILE, ROOT, Job
from sqs.grid.spec import select

_KEY_COLUMNS = ["network", "protocol", "mi_value", "seed_budget", "ss_method", "rep"]
_PROBLEM_STATUSES = ("PENDING", "INCOMPLETE", "TOO_MANY_ROWS", "DUPLICATES", "UNREADABLE")


@dataclass
class CheckResult:
    """Completeness and integrity outcome for one job's ``results.csv``."""

    rows: int
    duplicates_removed: int
    status: str


def check_job(job: Job, dedupe: bool) -> CheckResult:
    """
    Check one job's output for completeness, removing duplicate rows by default.

    :param job: job whose ``results.csv`` should be checked.
    :param dedupe: if True, rewrite ``results.csv`` with duplicate keys collapsed to their
        first occurrence; if False, only report duplicates without modifying the file.

    :returns: rows remaining, number of duplicate rows found/removed, and an overall status.
    """
    results_path = ROOT / job.out_dir / RESULTS_FILE
    if not results_path.exists():
        return CheckResult(rows=0, duplicates_removed=0, status="PENDING")

    try:
        df = pd.read_csv(results_path)
    except (OSError, pd.errors.ParserError):
        return CheckResult(rows=0, duplicates_removed=0, status="UNREADABLE")

    is_duplicate = df.duplicated(subset=_KEY_COLUMNS)
    duplicate_count = int(is_duplicate.sum())
    if duplicate_count and dedupe:
        df = df.loc[~is_duplicate]
        df.to_csv(results_path, index=False)

    rows = len(df)
    if duplicate_count and not dedupe:
        status = "DUPLICATES"
    elif rows < job.expected_rows:
        status = "INCOMPLETE"
    elif rows > job.expected_rows:
        status = "TOO_MANY_ROWS"
    else:
        status = "OK"
    return CheckResult(rows=rows, duplicates_removed=duplicate_count, status=status)


def mode_violations(jobs: list[Job]) -> pd.DataFrame:
    """
    Return Static cells where SQ or SQ_R gain is below SS.

    :param jobs: jobs to check (non-Static and missing results are ignored).

    :returns: offending rows with the gain of each mode.
    """
    key = ["network", "protocol", "mi_value", "seed_budget", "ss_selector", "rep"]
    frames = []
    for job in jobs:
        path = ROOT / job.out_dir / RESULTS_FILE
        if job.schedule["class"] == "StaticSeedingStrategy" and path.exists():
            frames.append(pd.read_csv(path, usecols=[*key, "ss_mode", "gain"]))
    if not frames:
        return pd.DataFrame()
    gains = pd.concat(frames).pivot_table(index=key, columns="ss_mode", values="gain")
    bad = pd.Series(False, index=gains.index)
    for mode in ("SQ", "SQ_R"):
        if {"SS", mode} <= set(gains.columns):
            bad |= gains[mode] < gains["SS"] - 1e-9
    return gains[bad].reset_index()


def main() -> None:
    """Parse CLI arguments and report completeness/integrity of matched grid jobs."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="grid spec, e.g. configs/grids/static.yaml")
    parser.add_argument("patterns", nargs="+", help="globs over job names, e.g. 'static/full/*'")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report duplicate rows without rewriting results.csv (default: dedupe in place)",
    )
    args = parser.parse_args()

    jobs = select(args.spec, args.patterns)
    results = {job.name: check_job(job, dedupe=not args.check_only) for job in jobs}

    counts = {status: 0 for status in ("OK", *_PROBLEM_STATUSES)}
    for result in results.values():
        counts[result.status] += 1

    print(f"Checked {len(jobs)} config(s):")
    for status, n in counts.items():
        if n:
            print(f"  {status:<14} {n}")

    deduped = [(job, results[job.name]) for job in jobs if results[job.name].duplicates_removed]
    if deduped:
        verb = "found" if args.check_only else "removed"
        print()
        for job, result in deduped:
            print(f"  {verb} {result.duplicates_removed} duplicate row(s): {job.name}")

    problems = [(job, results[job.name]) for job in jobs if results[job.name].status != "OK"]
    if problems:
        print()
        for job, result in problems:
            print(f"  {result.status:<14} {job.name:<45} {result.rows}/{job.expected_rows} rows")

    violations = mode_violations(jobs)
    if len(violations):
        print(f"\n  SQ / SQ_R below SS in {len(violations)} (cell, rep) pairs (rankings differ?):")
        print(violations.groupby(["network", "ss_selector"]).size().to_string())

    if any(counts[status] for status in _PROBLEM_STATUSES) or len(violations):
        sys.exit(1)
    print("\nAll matched jobs complete and clean; SQ / SQ_R >= SS everywhere.")


if __name__ == "__main__":
    main()
