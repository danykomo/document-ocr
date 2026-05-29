"""Resource measurement helpers.

Latency is wall-clock per op (recorded by the provider). Here we add process
peak RSS and (when an NVIDIA GPU is visible) GPU memory via nvidia-smi.
For remote HTTP or CPU-only providers, peak RSS reflects only the harness process, so GPU
memory is the meaningful signal and is sampled from the serving host.
"""

from __future__ import annotations

import resource
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import Optional


def _maxrss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


@contextmanager
def track_peak_memory():
    """Yield a callable returning the peak RSS delta (MB) since block entry.

    ``ru_maxrss`` is a monotonic high-water mark, so calling the returned
    function during or after the block reports the peak growth.
    """
    before = _maxrss_mb()

    def delta() -> float:
        return round(max(0.0, _maxrss_mb() - before), 2)

    yield delta


def gpu_memory_used_mb() -> Optional[float]:
    """Total used GPU memory across visible devices, or None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    values = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
    return round(sum(values), 1) if values else None
