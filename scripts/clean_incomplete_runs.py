#!/usr/bin/env python3
"""
Remove partial output directories left by ``run_grid.py`` jobs that did not finish.

``spreading.run`` appends one CSV row per repetition and never deduplicates, so simply
re-running an interrupted job (e.g. after the machine ran out of disk space) would append a
second copy of every case it had already completed. This script identifies jobs whose
``results.csv`` has fewer rows than the config expects and deletes their whole output
directory, so a subsequent ``run_grid.py`` invocation regenerates them cleanly from scratch.

Jobs that already have the expected number of rows are left untouched -- ``run_grid.py``
already skips those on its own (see ``count_result_rows``).

Usage:
    uv run python scripts/clean_incomplete_runs.py "full/*"
    uv run python scripts/clean_incomplete_runs.py "full/*" --apply
"""

import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import _ROOT, build_job, count_result_rows, resolve_config_names  # noqa: E402


def main() -> None:
    """Parse CLI arguments and report or delete incomplete job output directories."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs", nargs="+", help="Config names or fnmatch globs to check, e.g. 'full/*'"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete incomplete output directories (default: dry run, report only)",
    )
    args = parser.parse_args()

    jobs = [build_job(name) for name in resolve_config_names(args.configs)]

    complete, incomplete, pending = [], [], []
    for job in jobs:
        rows = count_result_rows(job.out_dir)
        if rows == 0:
            pending.append(job)
        elif rows >= job.expected_rows:
            complete.append(job)
        else:
            incomplete.append((job, rows))

    print(f"{len(complete)} complete, {len(incomplete)} incomplete, {len(pending)} never started")
    if not incomplete:
        print("Nothing to clean.")
        return

    print()
    for job, rows in incomplete:
        out_dir = _ROOT / job.out_dir
        action = "removing" if args.apply else "would remove"
        print(f"{action}: {job.name:<35} {rows}/{job.expected_rows} rows  ({out_dir})")
        if args.apply:
            shutil.rmtree(out_dir)

    if not args.apply:
        print("\nDry run only -- re-run with --apply to delete these directories.")


if __name__ == "__main__":
    main()
