#!/usr/bin/env python3
r"""Merge per-job ``results.csv`` files into one ``<out-root>/<network>/results.csv`` per network.

Usage:
    uv run python scripts/merge_results.py data/_static_results/{full,aucs,arxiv,timik} \
        --out-root data/static_results
"""

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

csv.field_size_limit(sys.maxsize)  # seedings of large networks exceed the default


def merge(input_dirs: list[Path], out_root: Path) -> None:
    """Stream the rows of every job under ``input_dirs`` into per-network files."""
    jobs = sorted(p / "results.csv" for d in input_dirs for p in d.iterdir() if p.is_dir())
    missing = [p.parent for p in jobs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} job(s) without results.csv, e.g. {missing[0]}")
    writers, files, rows, header = {}, [], {}, None
    try:
        for path in jobs:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                job_header = next(reader)
                if header is None:
                    header = job_header
                elif job_header != header:
                    raise ValueError(f"{path}: header differs from the first job's")
                column = header.index("network")
                for row in reader:
                    network = row[column]
                    if network not in writers:
                        out = out_root / network / "results.csv"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        files.append(out.open("w", newline="", encoding="utf-8"))
                        writers[network] = csv.writer(files[-1])
                        writers[network].writerow(header)
                        rows[network] = 0
                    writers[network].writerow(row)
                    rows[network] += 1
    finally:
        for fh in files:
            fh.close()
    for network, n in sorted(rows.items()):
        print(f"  {network:<35} {n:>8,} rows -> {out_root / network / 'results.csv'}")


def main() -> None:
    """Parse arguments and merge."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", nargs="+", type=Path, help="Dirs with one subdir per job")
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    merge(args.input_dirs, args.out_root)


if __name__ == "__main__":
    main()
