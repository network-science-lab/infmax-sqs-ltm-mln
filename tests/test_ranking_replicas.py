"""Ranking replicas: random tie-breaks, the order of each method's score otherwise."""

import json
import sys
from pathlib import Path

import pytest

from sqs.misc.ranking_replicas import SORT_KEYS, is_sorted, replicate

# deg-c-d style ranking with ties: score desc, ties by id (as v2 stores it)
ENTRIES = [{"actor": f"a{i}", "score": s} for i, s in enumerate([5, 5, 5, 3, 3, 1, 1, 1, 1])]


def test_replica_keeps_the_actors_and_the_score_order() -> None:
    for seed in range(5):
        replica = replicate(ENTRIES, "deg-c-d", seed)
        assert sorted(e["actor"] for e in replica) == sorted(e["actor"] for e in ENTRIES)
        assert [e["score"] for e in replica] == [e["score"] for e in ENTRIES]


def test_ties_are_shuffled_reproducibly() -> None:
    orders = {tuple(e["actor"] for e in replicate(ENTRIES, "deg-c-d", seed)) for seed in range(10)}
    assert len(orders) > 1  # ties no longer follow the actor id
    assert replicate(ENTRIES, "deg-c-d", 3) == replicate(ENTRIES, "deg-c-d", 3)


@pytest.mark.parametrize(
    ("selector", "scores"),
    [
        ("mds+deg", [{"driver": 1, "degree": 4}, {"driver": 1, "degree": 2}, {"driver": 0, "degree": 9}]),
        ("mds", [{"driver": 1}, {"driver": 0}]),
        ("voterank", [3.0, 2.5, 0.0]),
        (
            "ts-net",
            [
                {"exposed": 0.9, "simulation_length": 0.1, "peak_infected": 0.2, "peak_iteration": 0.3},
                {"exposed": 0.9, "simulation_length": 0.2, "peak_infected": 0.2, "peak_iteration": 0.3},
                {"exposed": 0.1, "simulation_length": 0.0, "peak_infected": 0.0, "peak_iteration": 0.9},
            ],
        ),
    ],
)
def test_sort_key_follows_the_method(selector: str, scores: list) -> None:
    entries = [{"actor": f"a{i}", "score": s} for i, s in enumerate(scores)]
    assert is_sorted(entries, selector)
    assert replicate(entries, selector, 0) == entries  # no ties: the order is kept
    assert set(SORT_KEYS) >= {"deg-c-d", "nghb-sd", "voterank", "mds+deg", "ts-net"}


def test_missing_scores_go_last() -> None:
    entries = [{"actor": "a", "score": None}, {"actor": "b", "score": 2}]
    assert [e["actor"] for e in replicate(entries, "deg-c", 0)] == ["b", "a"]


def test_voterank_float_noise_is_a_tie() -> None:
    entries = [{"actor": "a", "score": 5.4202776}, {"actor": "b", "score": 5.4202781}]
    assert is_sorted(entries, "voterank")
    assert len({tuple(e["actor"] for e in replicate(entries, "voterank", s)) for s in range(10)}) == 2


def test_replicas_written_next_to_v2_rankings(tmp_path: Path) -> None:
    """Original kept as rep 0, replicas 1..n-1 added, per-rep (mds) rankings untouched."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from replicate_rankings import replicate_dir

    net = tmp_path / "toy"
    net.mkdir()
    (net / "deg-c-d.json").write_text(json.dumps(ENTRIES), encoding="utf-8")
    mds = [{"actor": "a0", "score": {"driver": 1, "degree": 2}}]
    (net / "mds+deg__rep0.json").write_text(json.dumps(mds), encoding="utf-8")

    assert replicate_dir(tmp_path, 3) == (2, 0)
    assert sorted(p.name for p in net.iterdir()) == [
        "deg-c-d.json",
        "deg-c-d__rep1.json",
        "deg-c-d__rep2.json",
        "mds+deg__rep0.json",
    ]
    assert json.loads((net / "deg-c-d.json").read_text(encoding="utf-8")) == ENTRIES
    assert json.loads((net / "deg-c-d__rep2.json").read_text(encoding="utf-8")) == replicate(ENTRIES, "deg-c-d", 2)
    assert replicate_dir(tmp_path, 3) == (0, 2)
