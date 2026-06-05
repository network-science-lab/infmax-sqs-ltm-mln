# Sequential Seeding in Multilayer Networks with LTM

A research codebase investigating **sequential seeding strategies** for influence maximisation in
multilayer networks under the Linear Threshold Model (LTM).

## Repository structure

```bash
.
├── configs/                ->  Hydra configs for each pipeline stage
│   └── hydra/              ->  Hydra internal overrides (logging, etc.)
├── data/                   ->  DVC-tracked network datasets
│   └── networks/           ->  raw multilayer network files
└── src/sqs/
    ├── loaders/            ->  multilayer network loaders
    ├── pipeline/           ->  pipeline stages (one module per stage)
    ├── main.py             ->  CLI entrypoint
    └── utils.py            ->  shared utilities (logger, time, git SHA)
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

## Usage

```bash
uv run sqs                                          # run with default config (example_net_stats)
uv run sqs --config-name=<config>                   # pick a different config from configs/
uv run sqs logging.out_dir=./my_output              # override a single value
```

The pipeline is configured via [Hydra](https://hydra.cc). Config files live in `configs/`; add a new
YAML there for each experiment. Every run writes:

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
