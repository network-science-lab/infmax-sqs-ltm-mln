#!/usr/bin/env python3
r"""Merge per-job ``results.csv`` files back into one combined file per network.

``configs/full/``, ``configs/arxiv/``, and ``configs/timik/`` split each (protocol, network,
mode, selector) combination into its own job (see ``scripts/run_grid.py``), so each job directory
under ``data/raw_results/full/`` etc. only holds a thin slice of the full picture. This script
regroups those slices by the ``network`` column so that, for each network, all of its protocols,
modes, and selectors end up in a single ``results.csv`` -- the same shape that
``data/raw_results/full_aucs/results.csv`` already has for the (unsplit) ``aucs`` network.

Job directories without a ``results.csv`` (not yet run, or still running) are skipped and reported.

Usage:
    uv run python scripts/merge_results.py data/raw_results/full
    uv run python scripts/merge_results.py data/raw_results/full data/raw_results/arxiv \\
        data/raw_results/timik --out-root data/raw_results
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def find_job_results(input_dirs: list[Path]) -> list[Path]:
    """
    Locate every job's ``results.csv`` under the given input directories.

    :param input_dirs: directories holding one subdirectory per job (e.g.
        ``data/raw_results/full``).

    :returns: sorted list of ``results.csv`` paths found one level below the input directories.
    """
    results = []
    missing = []
    for input_dir in input_dirs:
        for job_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
            results_csv = job_dir / "results.csv"
            if results_csv.exists():
                results.append(results_csv)
            else:
                missing.append(job_dir)
    if missing:
        print(f"Skipping {len(missing)} job(s) with no results.csv yet:")
        for job_dir in missing:
            print(f"  {job_dir}")
    return results


def merge_by_network(results_paths: list[Path], out_root: Path) -> None:
    """
    Concatenate job results and write one combined ``results.csv`` per network.

    :param results_paths: paths to per-job ``results.csv`` files, each already tagged with a
        ``network`` column.
    :param out_root: directory under which ``full_<network>/results.csv`` is written for each
        network, mirroring the ``full_aucs`` layout.
    """
    df = pd.concat((pd.read_csv(p) for p in results_paths), ignore_index=True)
    for network, group in df.groupby("network"):
        out_dir = out_root / f"full_{network}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"
        group.to_csv(out_path, index=False)
        print(f"  {network:<35} {len(group):>8,} rows -> {out_path}")


def main() -> None:
    """Parse CLI arguments and merge job results into per-network combined files."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="Directories holding one subdirectory per job, e.g. data/raw_results/full",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_ROOT / "data" / "raw_results",
        help="Directory under which full_<network>/results.csv is written (default: "
        "data/raw_results)",
    )
    args = parser.parse_args()

    for input_dir in args.input_dirs:
        if not input_dir.is_dir():
            raise ValueError(f"Not a directory: {input_dir}")

    results_paths = find_job_results(args.input_dirs)
    if not results_paths:
        print("No results.csv files found -- nothing to merge.")
        return

    print(f"Merging {len(results_paths)} job result(s):")
    merge_by_network(results_paths, args.out_root)


if __name__ == "__main__":
    main()
