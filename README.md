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
uv run sqs --config-name=full_experiment_or_sq    # OR protocol, SQ mode, all networks
uv run sqs --config-name=full_experiment_and_ss   # AND protocol, SS mode, all networks
```

Each config writes its `results.csv` to `data/raw_results/<config-name>/`, appending one row per
repetition as soon as it completes (rather than only at the end of the run), so a killed or crashed
job keeps whatever it has already finished.

A single config runs its whole network list sequentially in one process, so a large network
(`timik1q2009`, `arxiv_netscience_coauthorship`) can dominate the wall-clock time. To spread a
config's networks across several workers instead -- splitting the slow networks further by
`mi_value` so they do not become a single-worker bottleneck -- use `scripts/run_grid.py`:

```bash
uv run python scripts/run_grid.py full_experiment_and_ss --workers 8
```

See the script's module docstring for details (shard sizing, GPU pinning via `--cuda-devices`,
duplicate-launch protection).

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
