# Sequential Seeding in Multilayer Networks with LTM

A research codebase investigating **sequential seeding strategies** for influence maximisation in
multilayer networks under the Linear Threshold Model (LTM).

## Repository structure

```bash
.
├── configs/                ->  Hydra configs for experiments and analysis scripts
│   └── hydra/              ->  Hydra internal overrides (logging, etc.)
├── data/                   ->  DVC-tracked datasets
│   └── networks/           ->  raw multilayer network files (.mpx)
├── examples/               ->  dry-run outputs for smoke-testing
├── scripts/                ->  standalone analysis scripts
│   └── compare_methods.py  ->  pairwise Wilcoxon tests with BH FDR correction
└── src/sqs/
    ├── loaders/            ->  multilayer network loaders
    ├── misc/               ->  result dataclasses and extraction helpers
    ├── models/             ->  MLTModel factory
    ├── pipeline/           ->  pipeline stages (net_stats, spreading)
    ├── runners/            ->  simulator, seeding strategies, selector registry
    ├── main.py             ->  CLI entrypoint
    └── utils.py            ->  shared utilities (logger, timing, git SHA)
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Sync the environment:
   ```bash
   uv sync
   ```
3. Pull network data:
   ```bash
   dvc pull
   ```

## Running experiments

```bash
uv run sqs --config-name=full_experiment_aucs        # aucs, both protocols, all modes -- cheap, unsplit
uv run sqs --config-name=full/and_er2_ss_mds_deg      # one (protocol, network, mode, selector) slice
```

`configs/full/`, `configs/arxiv/`, and `configs/timik/` each hold one config per (protocol, network,
mode, selector) combination (`arxiv`/`timik` fix the network, since those two are orders of
magnitude slower than the rest and cannot be handed to a single process as a whole).
`full_experiment_aucs.yaml` stays a single bundled config since aucs is cheap enough that splitting
it further would not help.

Each config writes its `results.csv` to `data/raw_results/<out_dir>/`, appending one row per
repetition as soon as it completes (rather than only at the end of the run), so a killed or crashed
job keeps whatever it has already finished.

To run many of these concurrently (weight-balanced by network, default 2 workers since this machine
is shared), use `scripts/run_grid.py` with an `fnmatch` glob:

```bash
uv run python scripts/run_grid.py "full/and_*" --workers 2
uv run python scripts/run_grid.py "timik/*" "arxiv/*" --workers 2 --cuda-devices 0,1
```

See the script's module docstring for details (job ordering, GPU pinning via `--cuda-devices`,
duplicate-launch protection).

Since each job only covers one (protocol, network, mode, selector) slice, use
`scripts/merge_results.py` to regroup finished jobs by network into one combined `results.csv`
per network -- the same shape `data/raw_results/full_aucs/results.csv` already has:

```bash
uv run python scripts/merge_results.py data/raw_results/full
uv run python scripts/merge_results.py data/raw_results/full data/raw_results/arxiv data/raw_results/timik
```

To test the pipeline on a small toy network first:

```bash
uv run sqs --config-name=example_static_strategy
```

The pipeline is configured via [Hydra](https://hydra.cc). Config files live in `configs/`; add a new
YAML there for each experiment variant.

### Seeding schedules

Each config declares a list of `seeding_schedules`, where every entry specifies a strategy class,
selector, and timing mode:

| Mode   | Behaviour                                                  |
| ------ | ---------------------------------------------------------- |
| `SS`   | Full budget seeded at epoch 0 (single-stage baseline)      |
| `SQ`   | One seed per epoch in pre-ranked order                     |
| `SQ_R` | One seed per epoch; new seed only when spreading stagnates |

- **`outputs/<date>/<time>/<job>.log`** — full log of the run (console output is mirrored here)
- **`outputs/<date>/<time>/.hydra/`** — Hydra metadata and the composed config snapshot

Stage-specific artefacts are written to the directory specified by `logging.out_dir` in the config.

## Development

Install pre-commit hooks (one-off):

```bash
uv run pre-commit install
```

Run tests:

```bash
uv run pytest
```
