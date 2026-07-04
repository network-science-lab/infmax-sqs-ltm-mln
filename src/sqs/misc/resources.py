"""Resource-usage monitoring context manager for simulation runs."""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import psutil


@dataclass
class ResourceUsage:
    """Wall time and memory consumed by a single simulation run."""

    device: str
    wall_seconds: float
    # Net RSS change (after - before) in MB; large intermediate allocations freed
    # before the context manager exits are not captured.
    rss_delta_mb: float
    # torch.cuda.max_memory_allocated / 1e6; None on CPU-only runs
    peak_gpu_mb: float | None


@contextmanager
def monitor_resources(device: str = "cpu") -> Generator[ResourceUsage, None, None]:
    """
    Context manager that records wall time and peak memory for a simulation run.

    :param device: device string (``"cpu"`` or ``"cuda:N"``).  For CPU-only LTM
        spreading pass ``"cpu"`` (the default); GPU tracking activates only when
        ``device`` starts with ``"cuda"``.

    :returns: Generator yielding a mutable ``ResourceUsage`` object populated in
        the ``finally`` block after the wrapped code completes.
    """
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss

    if device.startswith("cuda"):
        import torch  # noqa: PLC0415

        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    usage = ResourceUsage(device=device, wall_seconds=0.0, rss_delta_mb=0.0, peak_gpu_mb=None)

    try:
        yield usage
    finally:
        usage.wall_seconds = time.perf_counter() - t0
        usage.rss_delta_mb = (proc.memory_info().rss - mem_before) / 1e6
        if device.startswith("cuda"):
            import torch  # noqa: PLC0415

            usage.peak_gpu_mb = torch.cuda.max_memory_allocated(device) / 1e6
