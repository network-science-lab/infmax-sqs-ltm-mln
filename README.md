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
uv run sqs --config-name=full_experiment_or    # OR protocol, budgets 5–25 %
uv run sqs --config-name=full_experiment_and   # AND protocol, budgets 15–35 %
```

Both configs write `results.csv` to `data/raw_results/full_or/` and `full_and/` respectively.
To test the pipeline on a small toy network first:

```bash
uv run sqs --config-name=example_static_strategy
```

The pipeline is configured via [Hydra](https://hydra.cc). Config files live in `configs/`; add a new
YAML there for each experiment variant.

### Seeding schedules

Each config declares a list of `seeding_schedules`, where every entry specifies a strategy class,
selector, and timing mode:

| Mode   | Behaviour |
|--------|-----------|
| `SS`   | Full budget seeded at epoch 0 (single-stage baseline) |
| `SQ`   | One seed per epoch in pre-ranked order |
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
