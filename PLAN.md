# Spreading Functionality — Design Plan

## Status

PR 1 complete (`feature--mltm-simulator`). PR 2 in progress.

---

## Research context

Sequential seeding: seeds are introduced during the simulation (one or more per epoch), not all at
once at epoch 0. Between seeding events the LTM spreading runs normally. The goal is to compare
different seed-deployment strategies (static pre-rankings vs. adaptive/greedy) to determine whether
the timing and order of seeding affects final coverage.

---

## Guiding constraint

Reuse `network_diffusion` (v0.19.1) as-is. Only override the minimum necessary surface area.

---

## What is reused unchanged

| Component                          | From `network_diffusion`             | Notes                                                                    |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `MLTModel`                         | `network_diffusion.models.mlt_model` | Instantiated with `seeding_budget=(100, 0)` so all actors start INACTIVE |
| `MLTModel.network_evaluation_step` | —                                    | One epoch of LTM propagation, called per epoch                           |
| `BaseModel.update_network`         | —                                    | Static method; used to force-activate seeds mid-loop                     |
| `BaseModel.get_states_num`         | —                                    | Used for per-epoch logging                                               |
| `Logger`                           | `network_diffusion.logger`           | Accumulates global/local stats; `.convert_logs()` produces DataFrames    |
| All `BaseSeedSelector` subclasses  | `network_diffusion.seeding.*`        | Used to produce pre-ranked actor lists (PR 2)                            |
| `Simulator._update_counter`        | —                                    | Patience logic, reused by `SequentialSimulator`                          |

---

## Work plan

### PR 1 — `feature--mltm-simulator` — thin but fully working pipeline

Every commit leaves the codebase in a runnable state. `MockSeedingStrategy` is a placeholder (seeds
one actor per epoch in list order) — enough to drive the simulator end-to-end. After this PR,
`uv run sqs --config-name=example_spreading` produces a real CSV and detailed logs on disk.

| #   | Commit                                                                     | Content                                                                                                                         |
| --- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `add sqs.models.mltm`                                                      | MLTModel factory with `seeding_budget=(100, 0)`                                                                                 |
| 2   | `add sqs.runners`                                                          | `SeedingStrategy` protocol + `MockSeedingStrategy` + `SequentialSimulator` + `SequentialRunLog`                                 |
| 3   | `add sqs.result_handler`                                                   | `SimulationPartialResult` / `SimulationFullResult` dataclasses, `extract_simulation_result`, `seedings` / `expositions` helpers |
| 4   | `add sqs.pipeline.spreading`                                               | pipeline stage + `configs/example_spreading.yaml` + wire `"spreading"` into `main._RUNNERS`                                     |
| 5   | `add tests`                                                                | unit tests for `SequentialSimulator` and `result_handler`; one integration test running the full pipeline on a toy network      |
| 6   | `refactor SeedingStrategy interface and flatten simulator loop`            | flat single loop; `next_batch` signature; `SequentialRunLog`; patience wired through strategy                                   |
| 7   | `add configurable detailed logs with optional zip archiving`               | `_log_sink` context manager; `detailed_logs` config block                                                                       |
| 8   | `support multiple seeding schedules per experiment with unique name guard` | `seeding_schedules` as top-level list; duplicate name guard; log filename includes `ss_name`                                    |
| 9   | `move max_epochs and patience to global spreading config`                  | `max_epochs`/`patience` in `spreading` block (not per-schedule)                                                                 |

### PR 2 — `feature--seed-selectors` — replace mock, expand strategy space

Replace `MockSeedingStrategy` with real implementations backed by `BaseSeedSelector.actorwise()` and
add further strategy variants. Exact commit breakdown TBD.

Planned content:

- Real `StaticSeedingStrategy` backed by `BaseSeedSelector.actorwise()` (degree, VoteRank, etc.)
- Random-order baseline strategy
- `AdaptiveSeedingStrategy` stub (Protocol satisfied, logic deferred)
- Any fixes to result_handler / pipeline uncovered during PR 1

---

## New modules

### `sqs/models/mltm.py`

Thin factory / wrapper. Instantiates `MLTModel` with `seeding_budget=(100, 0)` so that
`determine_initial_states` returns all-INACTIVE. No model subclassing required.

---

### `sqs/runners/seeding_strategy.py`

Strategy pattern for deciding which actors to seed each epoch.

```
SeedingStrategy        (Protocol, runtime_checkable)
  .next_batch(net: MultilayerNetwork,
              remaining: list[MLNetworkActor],
              epoch: int,
              stagnant: bool) -> list[MLNetworkActor]

MockSeedingStrategy    (implements SeedingStrategy)
  — ignores net state; returns [remaining[0]] each epoch
  — placeholder for PR 1; replaced by real strategies in PR 2
```

`remaining` contains actors not yet **seeded** (not not-yet-activated). Organic LTM activation does
NOT remove an actor from `remaining`. Adaptive strategies that want to skip already-active actors
must query `net` directly.

Returning `[]` means "no seeding this epoch" — the simulator runs one LTM step instead. Returning a
list > 1 plants multiple seeds in a single epoch (no LTM step runs that epoch).

---

### `sqs/runners/simulator.py`

```
SequentialRunLog       (frozen dataclass)
  .logger: Logger
  .seeding_events: dict[int, list[str]]   # epoch -> [actor_id, ...]

SequentialSimulator(network_diffusion.Simulator)
  — overrides perform_propagation(n_seeds, strategy, max_epochs, patience) -> SequentialRunLog

Flat loop (single range(max_epochs)):
  if seeds_planted < n_seeds:
      batch = strategy.next_batch(net, remaining, epoch, stagnant)
      if batch: plant seeds via update_network, log, continue    # no LTM this epoch
  run network_evaluation_step; log; check patience; maybe break
```

The epoch counter is continuous across the whole simulation (seeding and LTM epochs share the same
index) so the Logger output is a single coherent time series.

---

### `sqs/pipeline/spreading.py`

Pipeline stage wired into the existing YAML-dispatch runner in `sqs/main.py`.

Responsibilities:

- Parse spreading config (networks, protocols, mi_values, seed_budgets, seeding_schedules)
- Guard against duplicate schedule names
- Build cartesian product of all parameters x repetitions
- For each combination: load network, instantiate strategy + model + simulator, call
  `perform_propagation`, extract metrics, write detailed log via `_log_sink`
- Write single `results.csv` to `out_dir`

Strategy class resolution is done by `_resolve_strategy` via the `_STRATEGY_CLASSES` dict (add new
strategy classes there when implementing PR 2).

---

## Module layout

```
src/sqs/
  models/
    __init__.py
    mltm.py               <- new in PR 1
  runners/
    __init__.py
    seeding_strategy.py   <- new in PR 1
    simulator.py          <- new in PR 1
  pipeline/
    __init__.py
    net_stats.py          <- existing
    spreading.py          <- new in PR 1
  main.py                 <- extended with "spreading" runner
  utils.py
```

---

## Config YAML schema

```yaml
defaults:
  - override hydra/job_logging: custom
  - _self_

networks:
  - "aucs"
  - "ckm_physicians"

run:
  experiment_type: "spreading"
  out_dir: "./data/raw_results"

spreading:
  protocols: ["AND", "OR"]
  mi_values: [0.1, 0.15]
  seed_budgets: [0.05, 0.10]
  repetitions: 10
  max_epochs: 200 # hard upper bound on total simulation epochs
  patience: 10 # stop once all seeds planted and this many no-change LTM epochs pass

  detailed_logs:
    save: true # write per-run activation logs alongside results.csv
    zip: true # archive logs into detailed_logs.zip instead of separate JSON files

seeding_schedules:
  - name: "degree_static"
    class: "StaticSeedingStrategy" # available in PR 2
    params:
      selector: "degree"
  - name: "voterank_static"
    class: "StaticSeedingStrategy"
    params:
      selector: "voterank"
```

`seeding_schedules` is a top-level list (not nested in `spreading`) so multiple strategy instances
can run in one experiment suite. Each entry's `name` must be unique — it appears in log filenames
and the `ss_method` CSV column. `max_epochs` and `patience` are global so all strategies run under
identical conditions.

---

## Results CSV schema

| Column              | Type    | Description                                                                           |
| ------------------- | ------- | ------------------------------------------------------------------------------------- |
| `gain`              | `float` | `(exposed - seeds) / (total - seeds) * 100`                                           |
| `simulation_length` | `int`   | last epoch where any activation occurred                                              |
| `seed_nb`           | `int`   | total number of seeds used                                                            |
| `exposed_nb`        | `int`   | actors active at end of simulation (seeds + organically activated)                    |
| `unexposed_nb`      | `int`   | actors inactive at end of simulation                                                  |
| `expositions`       | `str`   | `;`-joined per-epoch new-activation counts (continuous across seeding and LTM epochs) |
| `seedings`          | `str`   | `[id,id,...]` per epoch, separated by `;`; empty segment = no seeding that epoch      |
| `network`           | `str`   | network name constant                                                                 |
| `protocol`          | `str`   | `AND` or `OR`                                                                         |
| `seed_budget`       | `float` | fraction of actors, e.g. `0.05`                                                       |
| `mi_value`          | `float` | LTM activation threshold                                                              |
| `ss_method`         | `str`   | seeding schedule name (from `seeding_schedules[*].name`)                              |

Detailed per-run logs are JSON files named:
`{net}__{protocol}__mi{mi}__{b{budget}__{ss_name}__r{rep:03d}.json`
