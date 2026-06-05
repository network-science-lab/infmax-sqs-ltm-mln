# Spreading Functionality — Design Plan

## Status

Work in progress — architecture agreed, config/results schema TBD.

---

## Research context

Sequential seeding: seeds are introduced **one at a time during the simulation**, not all at once at
epoch 0. Between each seeding event the LTM spreading runs for a configurable number of epochs (or
until patience). The goal is to compare different seed-ordering strategies (static pre-rankings vs.
adaptive/greedy) to determine whether the timing and order of seeding affects final coverage.

---

## Guiding constraint

Reuse `network_diffusion` (v0.19.1) as-is. Only override the minimum necessary surface area.

---

## What is reused unchanged

| Component                          | From `network_diffusion`             | Notes                                                                    |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `MLTModel`                         | `network_diffusion.models.mlt_model` | Instantiated with `seeding_budget=(100, 0)` so all actors start INACTIVE |
| `MLTModel.network_evaluation_step` | —                                    | One epoch of LTM propagation, called per epoch                           |
| `BaseModel.update_network`         | —                                    | Static method; used to force-activate a seed mid-loop                    |
| `BaseModel.get_states_num`         | —                                    | Used for per-epoch logging                                               |
| `Logger`                           | `network_diffusion.logger`           | Accumulates global/local stats; `.convert_logs()` produces DataFrames    |
| All `BaseSeedSelector` subclasses  | `network_diffusion.seeding.*`        | Used to produce pre-ranked actor lists for `StaticSeedSchedule`          |
| `Simulator._update_counter`        | —                                    | Patience logic, reused by `SequentialSimulator`                          |

---

## Work plan

### PR 1 — `feature--spreading` — thin but fully working pipeline

Every commit leaves the codebase in a runnable state. `StaticSeedSchedule` is a **mock** (hardcoded
/ random order) — enough to drive the simulator end-to-end. After this PR merges,
`uv run sqs configs/example_spreading.yaml` produces a real CSV on disk.

| #   | Commit                       | Content                                                                                                                                                           |
| --- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `add sqs.models.mltm`        | MLTModel factory with `seeding_budget=(100, 0)`                                                                                                                   |
| 2   | `add sqs.runners`            | `SeedSchedule` protocol + `StaticSeedSchedule` (mock) + `SequentialSimulator` — all three in one commit; `SequentialSimulator` is not testable without a schedule |
| 3   | `add sqs.result_handler`     | `SimulationPartialResult` / `SimulationFullResult` dataclasses, `extract_simulation_result`, `seedings` / `expositions` serialisation helpers                     |
| 4   | `add sqs.pipeline.spreading` | pipeline stage + `configs/example_spreading.yaml` + wire `"spreading"` into `main._RUNNERS`                                                                       |
| 5   | `add tests`                  | unit tests for `SequentialSimulator` and `result_handler`; one integration test running the full pipeline on a toy network                                        |

### PR 2 — `feature--seed-selectors` — replace mock, expand strategy space

Replace the mock `StaticSeedSchedule` with the real implementation (accepts any
`network_diffusion.BaseSeedSelector` for ranking) and add further schedule variants. Exact commit
breakdown TBD after PR 1 surfaces any gaps.

Planned content:

- Real `StaticSeedSchedule` backed by `BaseSeedSelector.actorwise()`
- Random-order baseline schedule
- `AdaptiveSeedSchedule` stub (Protocol satisfied, logic deferred)
- Any fixes to result_handler / pipeline uncovered during PR 1

---

## New modules

### `sqs/models/mltm.py`

Thin factory / wrapper. Instantiates `MLTModel` with `seeding_budget=(100, 0)` so that
`determine_initial_states` returns all-INACTIVE — the `SequentialSimulator` then controls when each
seed is activated. No model subclassing required unless extra behaviour is needed.

---

### `sqs/runners/seed_schedule.py`

Strategy pattern for deciding which actor to seed next.

```
SeedSchedule          (Protocol)
  .next_seed(net: MultilayerNetwork, remaining: list[ActorId]) -> ActorId

StaticSeedSchedule    (implements SeedSchedule)
  — ignores net state; pops next actor from a pre-sorted list
  — list produced by any network_diffusion BaseSeedSelector

AdaptiveSeedSchedule  (implements SeedSchedule)   ← DEFERRED
  — inspects current net state
  — runs sub-simulations over remaining candidates
  — returns actor with max marginal gain
```

`StaticSeedSchedule` is the first implementation target.

---

### `sqs/runners/simulator.py`

```
SequentialSimulator(network_diffusion.Simulator)
  — overrides perform_propagation(...)
  — signature (sketch):

    def perform_propagation(
        self,
        seed_schedule: SeedSchedule,
        n_seeds: int,
        spreading_epochs: int,    # T — epochs of LTM between each seeding event
        patience: int | None,     # early-stop within each spreading phase
    ) -> Logger

  Loop outline:
    1. determine_initial_states → all INACTIVE; update_network
    2. for each seeding step i in range(n_seeds):
         a. next_actor = seed_schedule.next_seed(net, remaining)
         b. force-activate next_actor via BaseModel.update_network + NUBuff(state="1")
         c. log seeding event
         d. run spreading_epochs epochs of network_evaluation_step
            — check patience after each epoch (reuse _update_counter)
            — log global/local stats each epoch
    3. return Logger with continuous epoch numbering across all seeding phases
```

The epoch counter runs continuously (0, 1, 2, … across all phases) so the `Logger` output is a
single coherent time series.

---

### `sqs/pipeline/spreading.py`

Pipeline stage wired into the existing YAML-dispatch runner in `sqs/main.py`.

Responsibilities:

- Parse spreading config
- Build parameter space (cartesian product of networks × seed methods × budgets × thresholds ×
  spreading_epochs)
- For each parameter combination × repetition:
  - Load network, instantiate `MLTModel` + `SeedSchedule` + `SequentialSimulator`
  - Call `perform_propagation`
  - Extract metrics from `Logger`
  - Append row to results
- Write CSV + save config with git SHA

---

## Module layout

```
src/sqs/
  models/
    __init__.py
    mltm.py               ← new
  runners/
    __init__.py
    seed_schedule.py      ← new
    simulator.py          ← new
  pipeline/
    __init__.py
    net_stats.py          ← existing
    spreading.py          ← new
  main.py                 ← extend _RUNNERS dict
  utils.py
```

---

## Config YAML schema

The project uses **Hydra** (`hydra-core`) for config management. Hydra handles output directory
creation, config snapshotting (written to `<out_dir>/.hydra/`), and logging (via
`configs/hydra/job_logging/custom.yaml`). Manual config saving is not needed.

```yaml
defaults:
  - override hydra/job_logging: custom
  - _self_

networks:
  - "aucs"
  - "ckm_physicians"
  # ...

run:
  experiment_type: "spreading"
  out_dir: "./data/raw_results"

spreading:
  protocols: ["AND", "OR"]
  mi_values: [0.1, 0.15, 0.2]
  seed_budgets: [0.05, 0.10, 0.15] # fraction of actors (e.g. 0.05 = 5 %)
  repetitions: 10
  ss_methods:
    - "deg_c"
    - "betweenness"
    - "pagerank"

  seeding_schedule:
    class: "StaticSeedSchedule" # maps to a SeedSchedule implementation
    params:
      spreading_epochs: 50 # fixed T epochs between seeding events
      patience: 100 # early-stop within each spreading phase

hydra:
  run:
    dir: ${run.out_dir}
```

`seeding_schedule.class` is the name of the `SeedSchedule` implementation to instantiate; `params`
are forwarded as kwargs to its constructor. Swapping in `AdaptiveSeedSchedule` later requires only
changing `class` and providing its own `params` keys.

---

## Results CSV schema

Filename pattern (matching MDS repo): `results--ver-{seed}_{rep}.csv`

Repetition index and random seed are encoded in the filename; they are not columns.
`spreading_epochs` and `patience` are fixed per run and live in the saved `config.yaml`.

| Column              | Type    | Description                                                                                                                                                                             |
| ------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gain`              | `float` | `(exposed - seeds) / (total - seeds) * 100`                                                                                                                                             |
| `simulation_length` | `int`   | last epoch where any activation occurred                                                                                                                                                |
| `seed_nb`           | `int`   | total number of seeds used                                                                                                                                                              |
| `exposed_nb`        | `int`   | actors active at end of simulation                                                                                                                                                      |
| `unexposed_nb`      | `int`   | actors inactive at end of simulation                                                                                                                                                    |
| `expositions`       | `str`   | `;`-joined new activation counts per epoch (continuous across all seeding phases)                                                                                                       |
| `seedings`          | `str`   | seeding timeline: `[actor,actor,...]` per epoch, epochs separated by `;`, empty bracket = no seeding. Example: `[ID4,ID5];;;[ID1]` = {ID4,ID5} seeded at epoch 0, ID1 seeded at epoch 3 |
| `network`           | `str`   | network name constant                                                                                                                                                                   |
| `protocol`          | `str`   | `AND` or `OR`                                                                                                                                                                           |
| `seed_budget`       | `float` | fraction, e.g. `0.05`                                                                                                                                                                   |
| `mi_value`          | `float` | LTM activation threshold                                                                                                                                                                |
| `ss_method`         | `str`   | seed selector name                                                                                                                                                                      |
