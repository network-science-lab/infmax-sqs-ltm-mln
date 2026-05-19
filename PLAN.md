# Add networks data and loaders

## Steps

### 0. Bump version

Per the branch workflow, bump `version` in `pyproject.toml` before any other changes.

The version was already bumped to `0.2.0` when the branch was created, so this step is
already done — verify with `grep version pyproject.toml`.

### 1. Initialise DVC and copy network data

DVC has not yet been initialised in this repo. Initialise it, then copy all six network
directories from the sibling repo into `data/networks/`.

```bash
dvc init
cp -r ../infmax-mds-ltm-mln/data/networks/CONTROL_fmt          data/networks/
cp -r ../infmax-mds-ltm-mln/data/networks/arxiv_netscience_coauthorship data/networks/
cp -r ../infmax-mds-ltm-mln/data/networks/cannes_2013_social   data/networks/
cp -r ../infmax-mds-ltm-mln/data/networks/small_artificial     data/networks/
cp -r ../infmax-mds-ltm-mln/data/networks/small_real           data/networks/
cp -r ../infmax-mds-ltm-mln/data/networks/timik1q2009          data/networks/
```

Track the directory with DVC (one `.dvc` file per subdirectory keeps future partial pulls
feasible, matching the pattern in the source repo):

```bash
dvc add data/networks/CONTROL_fmt
dvc add data/networks/arxiv_netscience_coauthorship
dvc add data/networks/cannes_2013_social
dvc add data/networks/small_artificial
dvc add data/networks/small_real
dvc add data/networks/timik1q2009
```

DVC creates `.gitignore` entries for the data automatically. Verify with `git status` — you
should see `.dvc/` metadata files and `.dvc` stubs staged, but none of the raw CSV/MPX files.

### 2. Add runtime dependencies

The loader code depends on `network-diffusion`, `networkx`, `pandas`, and `tqdm`. Add them
with uv so they appear in `pyproject.toml` and `uv.lock`:

```bash
uv add network-diffusion networkx pandas tqdm
```

Verify: `uv run python -c "import network_diffusion, networkx, pandas, tqdm; print('ok')"`.

### 3. Copy loaders from mds repo (as-is, paths adjusted)

Create `src/sqs/loaders/` and copy the three source files verbatim, adjusting only the
import prefix from `src.loaders` to `sqs.loaders`:

Files to create:
- `src/sqs/loaders/__init__.py` — stub (mirrors the mds original)
- `src/sqs/loaders/constants.py` — copy from `../infmax-mds-ltm-mln/src/loaders/constants.py`;
  update `MLN_RAW_DATA_PATH` so it resolves relative to this file's location in `src/sqs/loaders/`
  (the path depth changes by one level)
- `src/sqs/loaders/fmri74.py` — copy from `../infmax-mds-ltm-mln/src/loaders/fmri74.py`; no
  import changes needed (no intra-package imports)
- `src/sqs/loaders/net_loader.py` — copy from `../infmax-mds-ltm-mln/src/loaders/net_loader.py`;
  change `from src.loaders.` → `from sqs.loaders.`; drop the unused `import torch`

Verify: `uv run python -c "from sqs.loaders.net_loader import load_network; print('ok')"`.

### 4. Harden loaders to meet repo standards

Rewrite the three loader files in-place to meet this repo's code conventions. Do one file at a
time so each can be reviewed independently.

Key changes per file:

**`constants.py`** — no logic changes; add module-level docstring.

**`fmri74.py`**:
- Add type hints to all function signatures
- Replace `str` path arguments with `pathlib.Path`
- Replace `print(...)` error reporting with a proper `logging.warning(...)`
- Add reST docstrings to `_parse_adj_mats` and `_parse_edgelists`

**`net_loader.py`**:
- Add type hints to all functions (return type `nd.MultilayerNetwork`)
- Replace bare string path concatenation with `Path` arithmetic
- Add reST docstrings to `_network_from_pandas`, `return_some_layers`, `load_network`,
  and the per-network getter functions
- Replace inline `# TODO` comments with either a fix or removal
- Replace `str | None` literals with `Optional[str]` if Python < 3.10 syntax is needed
  (check `.python-version`; 3.14 is fine with `X | Y`)

Verify after each file: `uv run python -c "from sqs.loaders.net_loader import load_network"` and
`uv run mypy src/sqs/loaders/` should report no errors.

### 5. Commit

Stage and commit in three logical commits:

1. Data + DVC metadata (after step 1):
   ```bash
   git add .dvc/ data/networks/*.dvc data/networks/.gitignore
   git commit -m "add networks data under DVC"
   ```

2. Loaders copied as-is + dependencies (after step 3):
   ```bash
   git add src/sqs/loaders/ pyproject.toml uv.lock
   git commit -m "copy sqs.loaders from infmax-mds-ltm-mln"
   ```

3. Hardened loaders (after step 4):
   ```bash
   git add src/sqs/loaders/
   git rm PLAN.md
   git commit -m "harden sqs.loaders to meet repo code standards"
   ```
