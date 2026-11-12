# coding = utf-8
"""
Timing utilities for measuring computation-only time.

- CPU-only code: uses time.process_time() to exclude I/O wait, sleep, and time
  spent in other processes. Only counts CPU time of the current process.
- GPU/PyTorch code: uses time.perf_counter() with torch.cuda.synchronize()
  before and after to ensure we measure actual GPU execution, not async
  kernel launch overhead or CPU wait time.
"""
import time
from typing import Optional
from contextlib import contextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _is_cuda_device(device) -> bool:
    """Check if device is CUDA."""
    if not TORCH_AVAILABLE:
        return False
    if device is None:
        return torch.cuda.is_available()
    if isinstance(device, str):
        return device.lower().startswith('cuda')
    if hasattr(device, 'type'):
        return device.type == 'cuda'
    return False


@contextmanager
def compute_timer(device=None):
    """
    Context manager for measuring computation-only time.

    - For CPU: uses process_time() (excludes I/O, sleep, other processes).
    - For GPU: uses perf_counter() with cuda.synchronize() before/after
      (measures actual GPU execution, not kernel launch overhead).

    Yields a timer object with .elapsed attribute set after exiting the block.

    Example:
        with compute_timer(device='cuda') as t:
            model(x)
        print(f"Compute time: {t.elapsed:.3f}s")
    """
    class _Timer:
        elapsed: float = 0.0

    t = _Timer()
    use_cuda = _is_cuda_device(device)

    if use_cuda and TORCH_AVAILABLE:
        torch.cuda.synchronize()
    start = time.perf_counter() if use_cuda else time.process_time()

    try:
        yield t
    finally:
        if use_cuda and TORCH_AVAILABLE:
            torch.cuda.synchronize()
        end = time.perf_counter() if use_cuda else time.process_time()
        t.elapsed = end - start


def compute_time_start(device=None) -> float:
    """
    Start a computation timer. Call compute_time_elapsed() with the returned
    value and same device to get elapsed time.

    Returns the start timestamp (internal format depends on device).
    """
    if _is_cuda_device(device) and TORCH_AVAILABLE:
        torch.cuda.synchronize()
    return time.perf_counter() if _is_cuda_device(device) else time.process_time()


def compute_time_elapsed(start: float, device=None) -> float:
    """
    Return elapsed computation time since compute_time_start().

    Args:
        start: Value returned by compute_time_start()
        device: Same device passed to compute_time_start()
    """
    if _is_cuda_device(device) and TORCH_AVAILABLE:
        torch.cuda.synchronize()
    end = time.perf_counter() if _is_cuda_device(device) else time.process_time()
    return end - start
