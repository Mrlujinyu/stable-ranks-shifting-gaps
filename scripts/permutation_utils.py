"""Finite-sample inclusive upper-tail counting for paired permutations."""
import math

def permutation_upper_tail(obs, perm, B=None):
    if B is None:
        B = len(perm)
    if B <= 0 or len(perm) != B:
        raise ValueError('draw count must match nonempty permutation values')
    if not math.isfinite(obs) or any(not math.isfinite(x) for x in perm):
        raise ValueError('observed and permutation statistics must be finite')
    return (1 + sum(x >= obs for x in perm)) / (B + 1)
