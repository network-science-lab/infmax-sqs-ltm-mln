"""Dataclasses and helpers for extracting and persisting simulation results."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from sqs.runners.simulator import SequentialRunLog


@dataclass(frozen=True)
class SimulationPartialResult:
    """Metrics extracted from a single simulation run, without experiment metadata."""

    gain: float
    simulation_length: int
    seed_nb: int
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

    @classmethod
    def from_partial(
        cls,
        partial: SimulationPartialResult,
        network: str,
        protocol: str,
        seed_budget: float,
        mi_value: float,
        ss_method: str,
    ) -> "SimulationFullResult":
        """
        Combine a ``SimulationPartialResult`` with experiment-level parameters.

        :param partial: metrics extracted from the simulation run.
        :param network: name of the network used.
        :param protocol: activation protocol (``"AND"`` or ``"OR"``).
        :param seed_budget: seed budget as a fraction of all actors (e.g. ``0.05``).
        :param mi_value: LTM activation threshold.
        :param ss_method: name of the seed selection method.

        :returns: fully populated ``SimulationFullResult``.
        """
        return cls(
            gain=partial.gain,
            simulation_length=partial.simulation_length,
            seed_nb=partial.seed_nb,
            exposed_nb=partial.exposed_nb,
            unexposed_nb=partial.unexposed_nb,
            expositions=partial.expositions,
            seedings=partial.seedings,
            network=network,
            protocol=protocol,
            seed_budget=seed_budget,
            mi_value=mi_value,
            ss_method=ss_method,
        )


def _build_seedings(seeding_events: dict[int, list[str]], total_epochs: int) -> str:
    """
    Serialise the seeding timeline into the ``seedings`` column format.

    Each epoch from 0 to ``total_epochs`` (inclusive) occupies one segment,
    separated by ``;``.  Epochs with seeds produce ``[id,id,...]``; epochs
    without seeding produce an empty segment.

    Example: seeds ``{0: ["A", "B"], 3: ["C"]}`` with ``total_epochs=4``
    yields ``"[A,B];;;[C];"`` — five segments for epochs 0..4.

    :param seeding_events: mapping of epoch index to actor IDs seeded there.
    :param total_epochs: index of the last logged epoch.

    :returns: the ``seedings`` column string.
    """
    segments: list[str] = []
    for epoch in range(total_epochs + 1):
        ids = seeding_events.get(epoch)
        segments.append(f"[{','.join(ids)}]" if ids else "")
    return ";".join(segments)


def _build_expositions(local_stats: dict[int, list[dict[str, str]]], total_epochs: int) -> str:
    """
    Serialise per-epoch new-activation counts into the ``expositions`` column format.

    Each segment (separated by ``;``) is the count of nodes that transitioned to
    ACTIVE in that epoch.  Epochs with no activations produce ``"0"``.

    :param local_stats: ``Logger._local_stats`` — mapping of epoch to the list of
        state-change dicts (``{"node_name": ..., "layer_name": ..., "new_state": ...}``).
    :param total_epochs: index of the last logged epoch.

    :returns: the ``expositions`` column string.
    """
    segments: list[str] = []
    for epoch in range(total_epochs + 1):
        epoch_log = local_stats.get(epoch, [])
        active_nodes = {e["node_name"] for e in epoch_log if e["new_state"] == "1"}
        segments.append(str(len(active_nodes)))
    return ";".join(segments)


def extract_simulation_result(
    run_log: SequentialRunLog, total_actors: int
) -> SimulationPartialResult:
    """
    Derive ``SimulationPartialResult`` metrics from a completed simulation run.

    :param run_log: result of ``SequentialSimulator.perform_propagation``.
    :param total_actors: total number of actors in the network.

    :returns: extracted metrics ready to be combined with experiment parameters.
    """
    local_stats = run_log.logger.get_detailed_logs()
    total_epochs = max(local_stats.keys()) if local_stats else 0

    # Count cumulative active actors per epoch to find simulation length and
    # final exposed count.
    prev_active = 0
    simulation_length = 0
    for epoch in range(total_epochs + 1):
        epoch_log = local_stats.get(epoch, [])
        active_nodes = {e["node_name"] for e in epoch_log if e["new_state"] == "1"}
        if len(active_nodes) > prev_active:
            simulation_length = epoch
            prev_active = len(active_nodes)

    exposed_nb = prev_active
    seed_nb = sum(len(ids) for ids in run_log.seeding_events.values())
    unexposed_nb = total_actors - exposed_nb
    max_gain = total_actors - seed_nb
    gain = 100.0 * (exposed_nb - seed_nb) / max_gain if max_gain > 0 else 0.0

    return SimulationPartialResult(
        gain=gain,
        simulation_length=simulation_length,
        seed_nb=seed_nb,
        exposed_nb=exposed_nb,
        unexposed_nb=unexposed_nb,
        expositions=_build_expositions(local_stats, total_epochs),
        seedings=_build_seedings(run_log.seeding_events, total_epochs),
    )


def save_results(results: list[SimulationFullResult], path: Path) -> None:
    """
    Write a list of results to a CSV file.

    :param results: simulation results to persist.
    :param path: destination file path; parent directories must already exist.
    """
    pd.DataFrame([asdict(r) for r in results]).to_csv(path, index=False, encoding="utf-8")
