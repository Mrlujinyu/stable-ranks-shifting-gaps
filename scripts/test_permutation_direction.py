"""Direction and counting regression test for the one-sided permutation p-value.

This is a software test over small synthetic inputs. It verifies the exact rule
used by the influence permutation (controls.py):

    p_upper = (1 + count(T_perm >= T_obs)) / (B + 1)

The test does not run the 10,000-draw analysis and is not part of the paper's
Results. It pins the `>=` (not strict `>`) upper tail and the plus-one correction.
"""
import math


from permutation_utils import permutation_upper_tail


def close(a, b):
    return abs(a - b) < 1e-12


def main():
    # Case 1: all permutation values strictly below the observed value -> count 0.
    assert close(permutation_upper_tail(5.0, [0, 1, 2, 3]), 1 / 5), "all-below should give 1/(B+1)"

    # Case 2: all permutation values reach or exceed the observed value -> count B -> p = 1.
    assert close(permutation_upper_tail(-1.0, [0, 1, 2, 3]), 1.0), "all-at-or-above should give p = 1"

    # Case 3: mixed with a tie at the observed value. B=4, T_obs=2, perm=[0,1,2,3].
    # `>=` counts {2,3} -> 2 -> (1+2)/5 = 3/5. A strict `>` would count {3} -> 2/5.
    assert close(permutation_upper_tail(2.0, [0, 1, 2, 3]), 3 / 5), ">= must count ties, not strict >"

    # Case 4: a strongly negative observed value under a "greater" (upper-tail)
    # alternative lies below every draw, so p = 1 (no support for "greater").
    perm = [-0.002, 0.001, -0.003, 0.000]
    assert close(permutation_upper_tail(-0.006, perm), 1.0), "negative observed should give p = 1 in the upper tail"

    # Case 5: error handling for non-finite and invalid inputs.
    for bad_obs in (float("nan"), float("inf")):
        try:
            permutation_upper_tail(bad_obs, [0, 1, 2, 3])
            raise AssertionError("non-finite observed must raise")
        except ValueError:
            pass
    try:
        permutation_upper_tail(0.0, [0, 1, 2, 3], B=5)
        raise AssertionError("draw-count mismatch must raise")
    except ValueError:
        pass
    try:
        permutation_upper_tail(0.0, [])
        raise AssertionError("empty draws must raise")
    except ValueError:
        pass

    for values in ([0, float("nan")], [0, float("inf")], [float("-inf"), 0]):
        try:
            permutation_upper_tail(0, values)
            raise AssertionError("non-finite draws must raise")
        except ValueError:
            pass

    print("test_permutation_direction.py: all direction/counting cases PASS")


if __name__ == "__main__":
    main()
