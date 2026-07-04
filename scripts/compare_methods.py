#!/usr/bin/env python3
"""Pairwise Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction.

Loads a results.csv produced by the spreading pipeline, groups rows by the
dimensions specified in the config, and runs pairwise Wilcoxon signed-rank tests
for each (group, metric) combination.  Observations are paired by ``rep`` —
repetition i of method A is matched to repetition i of method B, which ran on
the same network instance under identical conditions.

FDR correction (BH) is applied per metric across all groups and pairs.

Two output files are written to ``out_dir``:
- ``pairwise_tests.csv``  — one row per (group, metric, pair) with p-values,
  adjusted p-values, effect size (rank-biserial r), and a ``winner`` column.
- ``wins_summary.csv``    — for each (metric, method_a, method_b), total
  significant wins per direction across all groups.

Usage:
    uv run python scripts/compare_methods.py --config configs/compare_methods.yaml
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import false_discovery_control, wilcoxon


def _rank_biserial_wilcoxon(statistic: float, n: int, median_a: float, median_b: float) -> float:
    """
    Signed rank-biserial correlation for Wilcoxon signed-rank test.

    ``scipy.stats.wilcoxon`` with ``alternative='two-sided'`` returns the
    minimum of the two rank sums, which is symmetric in ``a`` and ``b``.  The
    magnitude of ``r`` is therefore always in ``[0, 1]`` and carries no
    directional information on its own.  The sign is recovered from the
    difference of the sample medians so that a positive ``r`` means ``a``
    tends to exceed ``b``.

    :param statistic: Wilcoxon test statistic.
    :param n: number of paired observations.
    :param median_a: median of the first sample.
    :param median_b: median of the second sample.

    :returns: signed rank-biserial correlation in ``[-1, 1]``.
    """
    max_stat = n * (n + 1) / 2
    magnitude = 1.0 - (2.0 * statistic) / max_stat
    return magnitude if median_a >= median_b else -magnitude


def _collect_raw_tests(df: pd.DataFrame, cfg: dict) -> list[dict]:
    group_by: list[str] = cfg["group_by"]
    metrics: list[str] = cfg["metrics"]
    comp_mode: str = cfg["comparisons"]["mode"]
    baseline: str | None = cfg["comparisons"].get("baseline")
    alternative: str = cfg.get("alternative", "two-sided")

    records = []

    for group_vals, grp in df.groupby(group_by, sort=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_info = dict(zip(group_by, group_vals))

        methods = sorted(grp["ss_method"].unique().tolist())

        if comp_mode == "all_pairs":
            pairs = list(combinations(methods, 2))
        else:
            if baseline is None:
                raise ValueError("comparisons.baseline must be set when mode='vs_baseline'")
            pairs = [(baseline, m) for m in methods if m != baseline]

        for metric in metrics:
            for method_a, method_b in pairs:
                sub_a = grp.loc[grp["ss_method"] == method_a, ["rep", metric]].dropna()
                sub_b = grp.loc[grp["ss_method"] == method_b, ["rep", metric]].dropna()
                # align by rep so pairing is exact
                merged = sub_a.merge(sub_b, on="rep", suffixes=("_a", "_b"))
                if len(merged) < 2:
                    continue
                a_vals = merged[f"{metric}_a"].to_numpy()
                b_vals = merged[f"{metric}_b"].to_numpy()
                # wilcoxon requires non-zero differences
                diffs = a_vals - b_vals
                if np.all(diffs == 0):
                    continue
                stat, p = wilcoxon(a_vals, b_vals, alternative=alternative)
                n_pairs = len(merged)
                median_a = float(np.median(a_vals))
                median_b = float(np.median(b_vals))
                records.append(
                    {
                        **group_info,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_pairs": n_pairs,
                        "median_a": median_a,
                        "median_b": median_b,
                        "statistic": stat,
                        "p_value": p,
                        "effect_r": _rank_biserial_wilcoxon(stat, n_pairs, median_a, median_b),
                    }
                )

    return records


def _apply_fdr(out: pd.DataFrame, metrics: list[str], alpha: float) -> pd.DataFrame:
    out["p_adjusted"] = np.nan
    for metric in metrics:
        mask = out["metric"] == metric
        if mask.sum() == 0:
            continue
        p_adj = false_discovery_control(out.loc[mask, "p_value"].to_numpy(), method="bh")
        out.loc[mask, "p_adjusted"] = p_adj
    out["significant"] = out["p_adjusted"] < alpha
    out["winner"] = out.apply(
        lambda r: (
            r["method_a"]
            if r["significant"] and r["median_a"] > r["median_b"]
            else (r["method_b"] if r["significant"] and r["median_b"] > r["median_a"] else None)
        ),
        axis=1,
    )
    return out


def _wins_summary(out: pd.DataFrame) -> pd.DataFrame:
    sig = out[out["significant"]].copy()
    rows = []
    for (metric, method_a, method_b), grp in sig.groupby(["metric", "method_a", "method_b"]):
        rows.append(
            {
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "wins_a": (grp["winner"] == method_a).sum(),
                "wins_b": (grp["winner"] == method_b).sum(),
                "total_significant": len(grp),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point: parse CLI args, load results CSV, run tests, and write output files."""
    parser = argparse.ArgumentParser(description="Pairwise statistical tests on spreading results.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["results_csv"])
    alpha: float = cfg.get("alpha", 0.05)

    print(f"Loaded {len(df):,} rows from {cfg['results_csv']}")

    records = _collect_raw_tests(df, cfg)
    if not records:
        print("No tests produced — check group_by columns and results_csv content.")
        return

    out = pd.DataFrame(records)
    out = _apply_fdr(out, cfg["metrics"], alpha)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tests_path = out_dir / "pairwise_tests.csv"
    out.to_csv(tests_path, index=False)

    wins = _wins_summary(out)
    wins_path = out_dir / "wins_summary.csv"
    wins.to_csv(wins_path, index=False)

    n_sig = int(out["significant"].sum())
    print(f"Tests: {len(out):,} | Significant (q<{alpha}): {n_sig:,}")
    print(f"  {tests_path}")
    print(f"  {wins_path}")


if __name__ == "__main__":
    main()
