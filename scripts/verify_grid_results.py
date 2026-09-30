#!/usr/bin/env python3
"""
Verify that every matched grid job has actually finished computing its results.

Cross-checks each config's expected row count (protocols x mi_values x seed_budgets x
seeding_schedules x repetitions) against what is actually in its ``results.csv``. Rows are
uniquely keyed by (network, protocol, mi_value, seed_budget, ss_method, rep); duplicates of
that key -- the signature left behind by two overlapping ``run_grid.py`` launches writing to
the same file (see ``already_running`` in ``run_grid.py``) -- are removed in place by
default, keeping the first occurrence of each key.

Usage:
    uv run python scripts/verify_grid_results.py "full/*" "arxiv/*" "timik/*"
    uv run python scripts/verify_grid_results.py "full/*" --check-only

Exits with status 1 if any matched job is pending, incomplete, unreadable, or has more rows
than expected once duplicates are resolved; exits 0 only when every matched job has exactly
the expected number of rows.
"""

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import _RESULTS_FILE, _ROOT, Job, build_job, resolve_config_names  # noqa: E402

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
    results_path = _ROOT / job.out_dir / _RESULTS_FILE
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


def main() -> None:
    """Parse CLI arguments and report completeness/integrity of matched grid jobs."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs", nargs="+", help="Config names or fnmatch globs to check, e.g. 'full/*'"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report duplicate rows without rewriting results.csv (default: dedupe in place)",
    )
    args = parser.parse_args()

    jobs = [build_job(name) for name in resolve_config_names(args.configs)]
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
            print(f"  {result.status:<14} {job.name:<35} {result.rows}/{job.expected_rows} rows")

    if any(counts[status] for status in _PROBLEM_STATUSES):
        sys.exit(1)
    print("\nAll matched jobs complete and clean.")


if __name__ == "__main__":
    main()
