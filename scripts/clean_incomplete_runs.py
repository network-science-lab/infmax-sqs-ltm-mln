#!/usr/bin/env python3
"""
Remove partial output directories left by ``run_grid.py`` jobs that did not finish.

``spreading.run`` resumes interrupted jobs, so this is only needed to force a clean rerun: it
deletes output directories whose ``results.csv`` has fewer rows than expected.

Jobs that already have the expected number of rows are left untouched (``run_grid.py`` skips
them too).

Usage:
    uv run python scripts/clean_incomplete_runs.py configs/grids/static.yaml "static/full/*"
    uv run python scripts/clean_incomplete_runs.py configs/grids/static.yaml "static/*" --apply
"""

import shutil
from argparse import ArgumentParser
from pathlib import Path

from sqs.grid.jobs import ROOT
from sqs.grid.spec import select


def main() -> None:
    """Parse CLI arguments and report or delete incomplete job output directories."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="grid spec, e.g. configs/grids/static.yaml")
    parser.add_argument("patterns", nargs="+", help="globs over job names, e.g. 'static/full/*'")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete incomplete output directories (default: dry run, report only)",
    )
    args = parser.parse_args()

    jobs = select(args.spec, args.patterns)

    complete, incomplete, pending = [], [], []
    for job in jobs:
        rows = job.rows_done()
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
        out_dir = ROOT / job.out_dir
        action = "removing" if args.apply else "would remove"
        print(f"{action}: {job.name:<45} {rows}/{job.expected_rows} rows  ({out_dir})")
        if args.apply:
            shutil.rmtree(out_dir)

    if not args.apply:
        print("\nDry run only -- re-run with --apply to delete these directories.")


if __name__ == "__main__":
    main()
