"""Resource-usage monitoring context manager for simulation runs."""

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psutil

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


@dataclass
class ResourceUsage:
    """Time and memory used by a single simulation run."""

    device: str
    wall_seconds: float
    cpu_seconds: float  # CPU time of the process (all threads), independent of machine load
    peak_rss_mb: float  # peak resident memory of the process during the run
    peak_rss_delta_mb: float  # peak_rss_mb minus the resident memory before the run
    peak_gpu_mb: float | None  # torch.cuda.max_memory_allocated; None without CUDA


def _reset_peak_rss() -> bool:
    """Reset the kernel's peak RSS counter (VmHWM); return False where unsupported."""
    try:
        _CLEAR_REFS.write_text("5", encoding="utf-8")
        return True
    except OSError:
        return False


def _peak_rss() -> int | None:
    """Return the process peak RSS (VmHWM) in bytes, or None where unavailable."""
    try:
        for line in _STATUS.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


@contextmanager
def monitor_resources(device: str = "cpu") -> Generator[ResourceUsage, None, None]:
    """
    Record wall / CPU time and peak memory of the wrapped code.

    Measures only the wrapped code (e.g. not the network copy made before it). The peak RSS is
    exact on Linux (VmHWM reset before the run); elsewhere it falls back to the larger of the RSS
    before and after, which misses peaks freed before the end.

    :param device: selector device, e.g. ``"cpu"`` or ``"cuda:0"``; GPU memory is tracked only
        for a CUDA device that is available.

    :returns: Generator yielding a ``ResourceUsage`` filled in after the wrapped code completes.
    """
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    exact_peak = _reset_peak_rss()
    if exact_peak:
        rss_before = _peak_rss() or rss_before  # right after the reset VmHWM is the current RSS
    cuda = False
    if device.startswith("cuda"):
        import torch  # noqa: PLC0415

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)

    usage = ResourceUsage(device, 0.0, 0.0, 0.0, 0.0, None)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        yield usage
    finally:
        usage.wall_seconds = time.perf_counter() - wall0
        usage.cpu_seconds = time.process_time() - cpu0
        peak = _peak_rss() if exact_peak else None
        if peak is None:
            peak = max(rss_before, proc.memory_info().rss)
        usage.peak_rss_mb = peak / 1e6
        usage.peak_rss_delta_mb = (peak - rss_before) / 1e6
        if cuda:
            import torch  # noqa: PLC0415

            usage.peak_gpu_mb = torch.cuda.max_memory_allocated(device) / 1e6
