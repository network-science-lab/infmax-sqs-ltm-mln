#!/usr/bin/env python3
r"""
Add replicas with random tie-breaks to the scored rankings (``data/rankings/v2``).

A deterministic selector's ranking ``<tag>.json`` stays as repetition 0; repetitions 1..n-1 get
``<tag>__rep<r>.json``: the ranking shuffled with seed r, then stably sorted by score as its
method orders actors, so ties no longer follow actor ids. Rankings that already exist per
repetition (mds) are left as they are, as are replicas written before.

Usage:
    uv run python scripts/replicate_rankings.py --replicas 30
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from sqs.misc.ranking_replicas import is_sorted, replicate

_ROOT = Path(__file__).resolve().parent.parent


def replicate_dir(rankings: Path, replicas: int) -> tuple[int, int]:
    """Write the missing replicas 1..``replicas``-1 next to each ranking; return (new, kept)."""
    written = kept = 0
    for path in sorted(rankings.glob("*/*.json")):
        if "__rep" in path.stem:
            continue  # per-rep ranking (mds) or a replica
        selector = path.stem.split("__")[0]
        entries = None
        for rep in range(1, replicas):
            out = path.with_name(f"{path.stem}__rep{rep}.json")
            if out.exists():
                kept += 1
                continue
            if entries is None:
                entries = json.loads(path.read_text(encoding="utf-8"))
                if not is_sorted(entries, selector):
                    print(f"WARNING: {path} is not in score order")
            tmp = out.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(replicate(entries, selector, rep), indent=2), encoding="utf-8"
            )
            tmp.replace(out)
            written += 1
    return written, kept


def main() -> None:
    """Parse arguments and write the replicas."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, default=_ROOT / "data" / "rankings" / "v2")
    parser.add_argument("--replicas", type=int, default=30, help="repetitions, original included")
    args = parser.parse_args()
    written, kept = replicate_dir(args.rankings, args.replicas)
    print(f"{written} replicas written, {kept} already there -> {args.rankings}")


if __name__ == "__main__":
    main()
