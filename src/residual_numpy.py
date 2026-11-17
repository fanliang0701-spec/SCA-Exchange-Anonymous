"""Recorded NumPy residual centering and whole-vector sampling."""
import numpy as np

def _matrix(values, name):
    raw = np.asarray(values)
    if raw.ndim != 2 or not all(raw.shape) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty real n x d matrix")
    result = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def center_residuals(x, mean_rows):
    """Subtract per-row means and center the bank in float64; inputs are unchanged."""
    x, means = _matrix(x, "x"), _matrix(mean_rows, "mean_rows")
    if x.shape != means.shape:
        raise ValueError("x and mean_rows must have the same shape")
    with np.errstate(over="raise", invalid="raise"):
        try:
            bank = x - means
            bank -= bank.mean(axis=0)
        except FloatingPointError as error:
            raise ValueError("residual centering overflowed float64") from error
    return bank


def transplant(target_mean_rows, bank, donor_indices):
    """Add selected residual rows in float64, then cast once to finite float32."""
    target, bank = _matrix(target_mean_rows, "target_mean_rows"), _matrix(bank, "bank")
    indices = np.asarray(donor_indices)
    if (indices.ndim != 1 or len(indices) != len(target) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(bank)) or target.shape[1] != bank.shape[1]):
        raise ValueError("donor indices and target/bank dimensions do not match")
    with np.errstate(over="raise", invalid="raise"):
        try:
            return (target + bank[indices]).astype(np.float32)
        except FloatingPointError as error:
            raise ValueError("transplant exceeds finite float32 range") from error

