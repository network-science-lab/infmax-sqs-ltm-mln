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
    ss_class: str
    ss_selector: str
    ss_mode: str
    rep: int
    wall_seconds: float
    cpu_seconds: float
    peak_rss_mb: float
    peak_rss_delta_mb: float
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
        ss_class: str,
        ss_selector: str = "",
        ss_mode: str = "",
        rep: int = 0,
        wall_seconds: float = 0.0,
        cpu_seconds: float = 0.0,
        peak_rss_mb: float = 0.0,
        peak_rss_delta_mb: float = 0.0,
        peak_gpu_mb: float | None = None,
    ) -> "SimulationFullResult":
        """
        Combine a ``SimulationPartialResult`` with experiment-level parameters.

        :param partial: metrics extracted from the simulation run.
        :param network: name of the network used.
        :param protocol: activation protocol (``"AND"`` or ``"OR"``).
        :param seed_budget: seed budget as a fraction of all actors (e.g. ``0.05``).
        :param mi_value: LTM activation threshold.
        :param ss_method: schedule name (unique identifier from ``seeding_schedules[*].name``).
        :param ss_class: strategy class name (e.g. ``"StaticSeedingStrategy"``).
        :param ss_selector: selector key (e.g. ``"deg-c"``); empty string when the strategy
            has no selector parameter.
        :param ss_mode: timing mode (e.g. ``"SS"``, ``"SQ"``, ``"SQ_R"``); empty string when
            the strategy has no mode parameter.
        :param wall_seconds: wall-clock time of the simulation run in seconds.
        :param cpu_seconds: CPU time of the process during the run in seconds.
        :param peak_rss_mb: peak resident memory of the process during the run in MB.
        :param peak_rss_delta_mb: ``peak_rss_mb`` minus the resident memory before the run.
        :param peak_gpu_mb: peak GPU memory allocated during the run in MB; ``None``
            without CUDA.

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
            ss_class=ss_class,
            ss_selector=ss_selector,
            ss_mode=ss_mode,
            rep=rep,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_mb=peak_rss_mb,
            peak_rss_delta_mb=peak_rss_delta_mb,
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


def _build_expositions(counts: list[int]) -> str:
    """
    Serialise per-epoch new-activation counts into the ``expositions`` column format.

    :param counts: cumulative active actors after each epoch (seeds included).

    :returns: ``;``-separated numbers of actors that became active in each epoch.
    """
    return ";".join(str(c - p) for c, p in zip(counts, [0, *counts[:-1]]))


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
    counts = [run_log.active_counts[e] for e in sorted(run_log.active_counts)] or [0]
    exposed_nb = counts[-1]
    simulation_length = max(
        (e for e in range(1, len(counts)) if counts[e] > counts[e - 1]), default=0
    )
    counts = counts[: simulation_length + 1]
    seed_nb = sum(len(ids) for ids in run_log.seeding_events.values())
    saved_seeds = max(0, n_seeds - seed_nb)
    unexposed_nb = total_actors - exposed_nb
    max_gain = total_actors - seed_nb
    gain = 100.0 * (exposed_nb - seed_nb) / max_gain if max_gain > 0 else 0.0

    expositions = _build_expositions(counts)
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
        seedings=_build_seedings(run_log.seeding_events, simulation_length),
    )


def append_result(result: SimulationFullResult, path: Path) -> None:
    """
    Append a single result as a row to a CSV file, writing the header on first write.

    Appending after every repetition -- instead of accumulating results in memory and writing
    them once at the end -- means a killed or crashed run keeps whatever it has already
    completed, which matters for runs spanning hours or days on large networks.

    :param result: simulation result to persist.
    :param path: destination file path; parent directories must already exist.
    """
    df = pd.DataFrame([asdict(result)])
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
