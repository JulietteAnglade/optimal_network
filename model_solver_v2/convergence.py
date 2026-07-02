"""Shared helper for reporting *which* named quantity (and which index
within it) is responsible for a max-abs-diff residual, used by the inner,
intermediate, and outer loops' convergence checks (inner_loop.py,
outer_loop.py) so residuals can be monitored step-by-step rather than
only as an opaque scalar."""
import numpy as np


def argmax_residual(named_diffs):
    """named_diffs: iterable of (name, diff_array_or_scalar) pairs.
    Returns {'value': float, 'variable': name, 'index': tuple} identifying
    the single largest |diff| across all of them, and where in that
    variable's array it occurred (empty tuple for a scalar)."""
    best_name, best_idx, best_val = None, (), -1.0
    for name, diff in named_diffs:
        arr = np.abs(np.asarray(diff, dtype=float))
        if arr.size == 0:
            continue
        idx = tuple(int(i) for i in np.unravel_index(np.argmax(arr), arr.shape))
        val = float(arr[idx])
        if val > best_val:
            best_name, best_idx, best_val = name, idx, val
    return {'value': best_val, 'variable': best_name, 'index': best_idx}
