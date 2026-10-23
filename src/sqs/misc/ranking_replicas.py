"""Ranking replicas: a scored ranking shuffled, then re-sorted by score as its method orders it."""

import random
from collections.abc import Callable
from typing import Any

# Sort key of a score per selector (ascending = better first), as each method orders actors
SORT_KEYS: dict[str, Callable[[Any], Any]] = {
    "deg-c": lambda s: -s,
    "deg-c-d": lambda s: -s,  # discounted score when picked
    "nghb-s": lambda s: -s,
    "nghb-sd": lambda s: -s,
    "voterank": lambda s: -round(s, 4),  # votes when elected (float32: equal up to rounding)
    "mds": lambda s: -s["driver"],
    "mds+deg": lambda s: (-s["driver"], -s["degree"]),
    "ts-net": lambda s: (
        -s["exposed"],
        s["simulation_length"],
        s["peak_infected"],
        -s["peak_iteration"],
    ),
}


def _key(selector: str) -> Callable[[dict[str, Any]], Any]:
    """Key of a ranking entry; entries without a score go last."""
    score_key = SORT_KEYS[selector]
    return lambda e: (1,) if e["score"] is None else (0, score_key(e["score"]))


def replicate(entries: list[dict[str, Any]], selector: str, seed: int) -> list[dict[str, Any]]:
    """Return ``entries`` (``{"actor", "score"}``) shuffled with ``seed``, then sorted by score."""
    replica = list(entries)
    random.Random(seed).shuffle(replica)
    replica.sort(key=_key(selector))  # stable: ties keep the shuffled order
    return replica


def is_sorted(entries: list[dict[str, Any]], selector: str) -> bool:
    """Whether ``entries`` are already in score order (holds for every v2 ranking)."""
    keys = [_key(selector)(e) for e in entries]
    return all(a <= b for a, b in zip(keys, keys[1:]))
