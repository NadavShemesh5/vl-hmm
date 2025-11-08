import numpy as np
from scipy import special


def normalize_by_indexes(a, indexes, axis=None):
    for idx in indexes:
        a_sum = a[..., idx].sum(axis)
        if axis and a[..., idx].ndim > 1:
            # Make sure we don't divide by zero.
            a_sum[a_sum == 0] = 1
            shape = list(a[..., idx].shape)
            shape[axis] = 1
            a_sum.shape = shape

        a[..., idx] /= a_sum


def normalize(a, axis=None):
    """
    Normalize the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalize(a, axis=None):
    """
    Normalize the input array so that ``sum(exp(a)) == 1``.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = special.logsumexp(a, axis, keepdims=True)
        a -= a_lse
