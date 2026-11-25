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

Each run writes `results.csv` to its `run.out_dir`, appending one row per repetition as soon as it
completes, so a killed or crashed job keeps whatever it has already finished.

Grids of jobs are defined by specs in `configs/grids/` (each job re-runs a Static base config with
one strategy, mode and selector via Hydra overrides) and run with `scripts/run_grid.py`:

- `static.yaml`: Static baseline, jobs in `data/_static_results/`
- `strategies.yaml`: Subnet and Adaptive, jobs in `data/_subnet_results/`, `data/_adaptive_results/`

```bash
uv run python scripts/precompute_rankings.py configs/grids/static.yaml "static/*" --device cpu --workers 16
uv run python scripts/run_grid.py configs/grids/static.yaml "static/full/*" --status --workers 16
uv run python scripts/run_grid.py configs/grids/static.yaml "static/full/*" "static/aucs/*" \
  --split mi --device cpu --workers 16 --longest-first
uv run python scripts/verify_grid_results.py configs/grids/static.yaml "static/*"
```

Interrupted jobs resume where they stopped (`+run.resume=false` to disable); `--split` runs slices
that are merged once all are complete. Rankings are cached in `data/rankings/v<N>/`: one per
(network, selector), and one per repetition for mds (randomised), so SS / SQ / SQ_R always share a
ranking. `precompute_rankings.py` fills the cache up front.

Each job covers one (protocol, network, mode, selector) slice; `scripts/merge_results.py` regroups
finished jobs into one `<out-root>/<network>/results.csv` per network (DVC-tracked):

```bash
uv run python scripts/merge_results.py data/_static_results/{full,aucs,arxiv,timik} \
  --out-root data/static_results
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

## Acknowledgement

This research was partially supported by: Horizon Europe, EU, grant no. 10108632; National Science Centre, Poland, grant no. 2022/45/B/ST6/04145; Polish National Agency for Academic Exchange, Strategic Partnerships, grant no. BPI/PST/2024/1/00129/U/00001; Wrocław University of Science and Technology, Academia Professorum Iuniorum and  Minigrants projects. Views and opinions expressed here are those of the authors and do not necessarily reflect those of the funding agencies.
