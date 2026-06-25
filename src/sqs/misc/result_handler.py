"""Dataclasses and helpers for extracting and persisting simulation results."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import auc as sklearn_auc

from sqs.runners.simulator import SequentialRunLog


@dataclass(frozen=True)
class SimulationPartialResult:
    """Metrics extracted from a single simulation run, without experiment metadata."""

    gain: float
    auc: float
    auc_total: float
    simulation_length: int
    seed_nb: int
    saved_seeds: int
    exposed_nb: int
    unexposed_nb: int
    expositions: str
    seedings: str


@dataclass(frozen=True)
class SimulationFullResult(SimulationPartialResult):
    """Simulation metrics combined with the experiment parameters that produced them."""

    network: str
    protocol: str
    seed_budget: float
    mi_value: float
    ss_method: str
    wall_seconds: float
    peak_cpu_mb: float
    peak_gpu_mb: float | None

    @classmethod
    def from_partial(
        cls,
        partial: SimulationPartialResult,
        network: str,
        protocol: str,
        seed_budget: float,
        mi_value: float,
        ss_method: str,
        wall_seconds: float = 0.0,
        peak_cpu_mb: float = 0.0,
        peak_gpu_mb: float | None = None,
    ) -> "SimulationFullResult":
        """
        Combine a ``SimulationPartialResult`` with experiment-level parameters.

        :param partial: metrics extracted from the simulation run.
        :param network: name of the network used.
        :param protocol: activation protocol (``"AND"`` or ``"OR"``).
        :param seed_budget: seed budget as a fraction of all actors (e.g. ``0.05``).
        :param mi_value: LTM activation threshold.
        :param ss_method: name of the seed selection method.
        :param wall_seconds: wall-clock time of the simulation run in seconds.
        :param peak_cpu_mb: peak RSS increase of the process during the run in MB.
        :param peak_gpu_mb: peak GPU memory allocated during the run in MB; ``None``
            when running on CPU.

        :returns: fully populated ``SimulationFullResult``.
        """
        return cls(
            gain=partial.gain,
            auc=partial.auc,
            auc_total=partial.auc_total,
            simulation_length=partial.simulation_length,
            seed_nb=partial.seed_nb,
            saved_seeds=partial.saved_seeds,
            exposed_nb=partial.exposed_nb,
            unexposed_nb=partial.unexposed_nb,
            expositions=partial.expositions,
            seedings=partial.seedings,
            network=network,
            protocol=protocol,
            seed_budget=seed_budget,
            mi_value=mi_value,
            ss_method=ss_method,
            wall_seconds=wall_seconds,
            peak_cpu_mb=peak_cpu_mb,
            peak_gpu_mb=peak_gpu_mb,
        )


def _build_seedings(seeding_events: dict[int, list[str]], total_epochs: int) -> str:
    """
    Serialise the seeding timeline into the ``seedings`` column format.

    Each epoch from 0 to ``total_epochs`` (inclusive) occupies one segment,
    separated by ``;``.  Epochs with seeds produce ``[id,id,...]``; epochs
    without seeding produce an empty segment.

    Example: seeds ``{0: ["A", "B"], 3: ["C"]}`` with ``total_epochs=4``
    yields ``"[A,B];;;[C];"`` -- five segments for epochs 0..4.

    :param seeding_events: mapping of epoch index to actor IDs seeded there.
    :param total_epochs: index of the last logged epoch.

    :returns: the ``seedings`` column string.
    """
    segments: list[str] = []
    for epoch in range(total_epochs + 1):
        ids = seeding_events.get(epoch)
        segments.append(f"[{','.join(ids)}]" if ids else "")
    return ";".join(segments)


def _build_expositions(
    local_stats: dict[int, list[dict[str, str]]],
    total_epochs: int,
    seeding_events: dict[int, list[str]],
) -> str:
    """
    Serialise per-epoch new-activation counts into the ``expositions`` column format.

    Each segment (separated by ``;``) is the count of actors that first became ACTIVE
    in that epoch.  For seeding epochs the count equals the number of seeds planted;
    for LTM epochs it is derived from the cumulative active count in the full-state
    snapshot.  Epochs with no new activations produce ``"0"``.

    LTM epoch logs are full-state snapshots: every actor has exactly one entry with
    either ``new_state="0"`` or ``new_state="1"``, so
    ``len({active node_names})`` is a reliable cumulative active-actor count.
    Seeding epoch logs are partial (only seeded actors), so their count comes from
    ``seeding_events`` directly.

    :param local_stats: ``Logger._local_stats`` -- mapping of epoch to the list of
        state-change dicts (``{"node_name": ..., "layer_name": ..., "new_state": ...}``).
    :param total_epochs: index of the last logged epoch.
    :param seeding_events: mapping of seeding epoch index to seeded actor IDs, used
        to distinguish seeding epochs from LTM epochs.

    :returns: the ``expositions`` column string.
    """
    prev_active = 0
    segments: list[str] = []
    for epoch in range(total_epochs + 1):
        if epoch in seeding_events:
            n = len(seeding_events[epoch])
            prev_active += n
        else:
            epoch_log = local_stats.get(epoch, [])
            active_count = len({e["node_name"] for e in epoch_log if e["new_state"] == "1"})
            n = max(0, active_count - prev_active)
            prev_active = max(prev_active, active_count)
        segments.append(str(n))
    return ";".join(segments)


def _compute_auc(expositions_str: str, norm: int) -> float:
    """
    Compute the area under the normalized spreading curve (Lambda from proposal.tex).

    The curve maps normalized epoch index (x in [0, 1]) to normalized cumulative
    activation (y in [0, 1]).  Area is integrated with the trapezoidal rule via
    ``sklearn.metrics.auc``.

    :param expositions_str: the ``expositions`` column string (``";"``-separated
        per-epoch new-activation counts).
    :param norm: denominator used to normalize the y axis.  Pass ``exposed_nb`` to
        measure spreading speed independently of coverage, or ``total_actors`` to
        obtain a combined speed-and-coverage score.  Returns ``0.0`` when ``norm``
        is zero.

    :returns: AUC value in [0, 1].
    """
    if norm == 0:
        return 0.0
    counts = [int(v) for v in expositions_str.split(";")]
    cumsum: list[float] = []
    total = 0
    for c in counts:
        total += c
        cumsum.append(total)
    n = len(cumsum)
    if n < 2:
        return 0.0
    x = [i / (n - 1) for i in range(n)]
    y = [c / norm for c in cumsum]
    return float(sklearn_auc(x, y))


def extract_simulation_result(
    run_log: SequentialRunLog, total_actors: int, n_seeds: int
) -> SimulationPartialResult:
    """
    Derive ``SimulationPartialResult`` metrics from a completed simulation run.

    :param run_log: result of ``SequentialSimulator.perform_propagation``.
    :param total_actors: total number of actors in the network.
    :param n_seeds: planned seed budget passed to ``perform_propagation``; used
        to compute ``saved_seeds`` (budget unused when spreading terminates early).

    :returns: extracted metrics ready to be combined with experiment parameters.
    """
    local_stats = run_log.logger.get_detailed_logs()
    total_epochs = max(local_stats.keys()) if local_stats else 0

    prev_active = 0
    simulation_length = 0
    for epoch in range(total_epochs + 1):
        if epoch in run_log.seeding_events:
            prev_active += len(run_log.seeding_events[epoch])
            simulation_length = epoch
        else:
            epoch_log = local_stats.get(epoch, [])
            active_count = len({e["node_name"] for e in epoch_log if e["new_state"] == "1"})
            if active_count > prev_active:
                simulation_length = epoch
                prev_active = active_count
    exposed_nb = prev_active
    seed_nb = sum(len(ids) for ids in run_log.seeding_events.values())
    saved_seeds = max(0, n_seeds - seed_nb)
    unexposed_nb = total_actors - exposed_nb
    max_gain = total_actors - seed_nb
    gain = 100.0 * (exposed_nb - seed_nb) / max_gain if max_gain > 0 else 0.0

    expositions = _build_expositions(local_stats, total_epochs, run_log.seeding_events)
    return SimulationPartialResult(
        gain=gain,
        auc=_compute_auc(expositions, exposed_nb),
        auc_total=_compute_auc(expositions, total_actors),
        simulation_length=simulation_length,
        seed_nb=seed_nb,
        saved_seeds=saved_seeds,
        exposed_nb=exposed_nb,
        unexposed_nb=unexposed_nb,
        expositions=expositions,
        seedings=_build_seedings(run_log.seeding_events, total_epochs),
    )


def save_results(results: list[SimulationFullResult], path: Path) -> None:
    """
    Write a list of results to a CSV file.

    :param results: simulation results to persist.
    :param path: destination file path; parent directories must already exist.
    """
    pd.DataFrame([asdict(r) for r in results]).to_csv(path, index=False, encoding="utf-8")
