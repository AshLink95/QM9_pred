"""Wannier matching metric: exact on identical sets, order-free, counts misses, uses first k."""

import numpy as np

from training.metrics import match_wannier, wannier_summary

RNG = np.random.default_rng(0)
TRUE_C = RNG.standard_normal((5, 3))
TRUE_R = RNG.uniform(0.5, 1.0, 5)


def test_identical_sets_are_exact():
    m = match_wannier(TRUE_C, TRUE_R, 5, TRUE_C, TRUE_R)
    s = wannier_summary([m])
    assert s["center_mae"] < 1e-12 and s["radius_mae"] < 1e-12
    assert s["count_accuracy"] == 1.0 and s["n_matched"] == 5


def test_order_does_not_matter():
    perm = RNG.permutation(5)
    m = match_wannier(TRUE_C[perm], TRUE_R[perm], 5, TRUE_C, TRUE_R)
    assert wannier_summary([m])["center_mae"] < 1e-12


def test_missing_center_is_a_count_error_not_a_distance_error():
    m = match_wannier(TRUE_C[:4], TRUE_R[:4], 4, TRUE_C, TRUE_R)
    s = wannier_summary([m])
    assert s["count_error_breakdown"] == {-1: 1} and s["count_accuracy"] == 0.0
    assert s["center_mae"] < 1e-12 and s["n_matched"] == 4


def test_only_first_k_slots_are_predictions():
    pred = np.vstack([TRUE_C[:4], np.full((2, 3), 99.0)])      # slots 4,5 are unused junk
    m = match_wannier(pred, np.ones(6), 4, TRUE_C[:4], TRUE_R[:4])
    assert m["n_pred"] == 4 and m["kept"] == [0, 1, 2, 3]
    assert wannier_summary([m])["center_mae"] < 1e-12


def test_known_offset():
    m = match_wannier(TRUE_C + [0.1, 0, 0], TRUE_R, 5, TRUE_C, TRUE_R)
    assert abs(wannier_summary([m])["center_mae"] - 0.1) < 1e-9
