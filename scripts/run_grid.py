#!/usr/bin/env python3
r"""
Run a grid of ``sqs`` jobs defined by a spec in ``configs/grids/``.

Each job re-runs a Static base config (``configs/full``, ``arxiv``, ``timik``, aucs) with one
strategy, mode and selector through Hydra overrides; job names are ``<prefix>/<source>/<stem>``.
Finished jobs are skipped and interrupted ones resume (see ``spreading.run``).

Usage:
    uv run python scripts/run_grid.py configs/grids/static.yaml "static/full/*" --status
    uv run python scripts/run_grid.py configs/grids/static.yaml "static/arxiv/*" \\
        --split mi-budget --device cpu --workers 16 --longest-first
"""

from argparse import ArgumentParser
from pathlib import Path

from sqs.grid.launcher import print_status, run
from sqs.grid.spec import select


def main() -> None:
    """Parse CLI arguments and run, list or report the selected jobs."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="grid spec, e.g. configs/grids/static.yaml")
    parser.add_argument("patterns", nargs="+", help="globs over job names, e.g. 'static/full/*'")
    parser.add_argument("--workers", type=int, default=2, help="max concurrent jobs")
    parser.add_argument(
        "--split",
        choices=["none", "mi", "mi-budget"],
        default="none",
        help="run jobs as slices per mi (or mi x budget), merged when all slices are done",
    )
    parser.add_argument("--device", help="replace selector_params.device, e.g. 'cpu'")
    parser.add_argument("--cuda-devices", help="GPU ids to round-robin, e.g. '0,1'")
    parser.add_argument("--longest-first", action="store_true", help="heaviest jobs first")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    parser.add_argument("--dry-run", action="store_true", help="list jobs and exit")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    jobs = select(args.spec, args.patterns, args.device, args.split)
    jobs.sort(key=lambda j: j.weight_seconds, reverse=args.longest_first)
    if args.status:
        print_status(jobs, args.workers)
    elif args.dry_run:
        for job in jobs:
            print(f"{job.name:<55} ~{job.weight_seconds / 3600:7.1f} h {job.expected_rows:>6} rows")
        print(f"\n{len(jobs)} jobs; overrides of {jobs[0].name}:", *jobs[0].overrides, sep="\n  ")
    else:
        cuda = args.cuda_devices.split(",") if args.cuda_devices else None
        run(jobs, args.workers, cuda)


if __name__ == "__main__":
    main()
