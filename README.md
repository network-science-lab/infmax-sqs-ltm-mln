# Sequential Seeding in Multilayer Networks with LTM

A research codebase investigating **sequential seeding strategies** for influence maximisation in
multilayer networks under the Linear Threshold Model (LTM).

## Repository structure

```bash
.
├── data/                   ->  DVC-tracked network datasets
│   └── networks/           ->  raw multilayer network files
├── scripts/
│   └── configs/            ->  example YAML configs for each pipeline stage
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
uv run sqs                                         # run with the default example config
uv run sqs scripts/configs/example_net_stats.yaml  # run with an explicit config
```

Each run saves a `config.yaml` (stamped with the HEAD git SHA) and stage-specific outputs to the
directory specified by `logging.out_dir` in the config.

## Development

Install pre-commit hooks (one-off):

```bash
uv run pre-commit install
```

Run tests:

```bash
uv run pytest
```
