"""TOPSIS-Bausteine: Idealpunkte, Abstände, Nähe, Erfüllungsskalarisierung, wirksame Gewichte (je Einheit gegen Handwerte der Mini-Instanzen)."""

import math

import numpy as np
import pytest

import tps_topsis as tp


def test_ideal_points(N3):
    hi, lo = tp.ideal_points(N3)
    assert hi.tolist() == [1.0, 1.0, 1.0] and lo.tolist() == [0.0, 0.0, 0.0]


def test_p_norm_values_and_rejects_p_below_one():
    D = np.array([[3.0, -4.0], [0.0, 0.0]])
    assert tp.p_norm(D, 1.0).tolist() == [7.0, 0.0]
    assert tp.p_norm(D, 2.0).tolist() == [5.0, 0.0]
    assert tp.p_norm(D, math.inf).tolist() == [4.0, 0.0]
    assert tp.p_norm(D, 3.0)[0] == pytest.approx((27 + 64) ** (1 / 3))
    with pytest.raises(ValueError):
        tp.p_norm(D, 0.5)


def test_manhattan_closeness_equals_the_weighted_sum(N3, w3):
    dp, dm, c = tp.closeness(N3, w3, 1.0)
    assert dp == pytest.approx([0.30, 0.58, 0.60]) and dm == pytest.approx([0.70, 0.42, 0.40])
    assert c == pytest.approx(N3 @ w3) and c == pytest.approx([0.70, 0.42, 0.40])


def test_euclidean_closeness_hand_values_and_order(N3, w3):
    dp, dm, c = tp.closeness(N3, w3, 2.0)
    assert dp == pytest.approx([0.3000, 0.5064, 0.3536], abs=5e-5)
    assert dm == pytest.approx([0.5385, 0.3231, 0.2915], abs=5e-5)
    assert c == pytest.approx([0.6422, 0.3895, 0.4519], abs=5e-5)
    assert np.argsort(-c).tolist() == [0, 2, 1]            # 1 > 3 > 2, die Summe ordnet 1 > 2 > 3


def test_chebyshev_closeness_hand_values(N3, w3):
    dp, dm, c = tp.closeness(N3, w3, math.inf)
    assert dp == pytest.approx([0.30, 0.50, 0.25]) and dm == pytest.approx([0.50, 0.30, 0.25])
    assert c == pytest.approx([0.625, 0.375, 0.50]) and np.argsort(-c).tolist() == [0, 2, 1]


def test_closeness_with_explicit_reference_points(N3, w3):
    dp, dm, c = tp.closeness(N3, w3, 1.0, ideal=np.array([0.5, 0.5, 0.5]), anti=np.zeros(3))
    assert dp == pytest.approx([0.5 * 0.5 + 0.3 * 0.5 + 0.2 * 0.5, 0.25 + 0.15 + 0.02, 0.0 + 0.0 + 0.1])
    assert dm == pytest.approx(N3 @ w3)
    assert c == pytest.approx(dm / (dp + dm))


def test_closeness_is_one_half_when_both_distances_vanish():
    N = np.array([[0.5, 0.5], [0.5, 0.5]])
    dp, dm, c = tp.closeness(N, np.array([0.5, 0.5]), 2.0)
    assert dp.tolist() == [0.0, 0.0] and c.tolist() == [0.5, 0.5]


def test_asf_distance_hand_values(N3, w3):
    s = tp.asf_distance(N3, w3, None, 0.01)
    assert s == pytest.approx([0.3030, 0.5058, 0.2560], abs=5e-5)
    assert np.argsort(s).tolist() == [2, 0, 1]            # kleiner ist besser: 3 > 1 > 2


def test_asf_is_one_sided_over_achievement_counts_negative(N3, w3):
    s = tp.asf_distance(N3, w3, np.array([0.5, 0.5, 0.5]), 0.0)
    assert s == pytest.approx([max(-0.25, 0.15, -0.1), max(0.25, -0.15, -0.02), max(0.0, 0.0, 0.1)])


def test_hull_point_is_lost_by_sum_and_topsis_but_won_by_the_reference_point(hull):
    w = np.array([0.5, 0.5])
    assert (hull @ w).tolist() == [0.5, 0.5, 0.4]
    for p in (1.0, 2.0, math.inf):
        assert tp.closeness(hull, w, p)[2].tolist() == pytest.approx([0.5, 0.5, 0.4])
    s = tp.asf_distance(hull, w, None, 0.01)
    assert s == pytest.approx([0.505, 0.505, 0.306]) and int(np.argmin(s)) == 2


def test_hull_point_over_a_grid_of_weights(hull):
    grid = [np.array([a, 1 - a]) for a in np.linspace(0.001, 0.999, 999)]
    assert {int(np.argmax(hull @ w)) for w in grid} == {0, 1}
    for p in (1.0, 2.0, math.inf):
        assert {int(np.argmax(tp.closeness(hull, w, p)[2])) for w in grid} == {0, 1}
    assert 2 in {int(np.argmin(tp.asf_distance(hull, w, None, 0.01))) for w in grid}


def test_asf_reachable_finds_the_hull_point_and_every_extreme(hull):
    assert tp.asf_reachable(hull, 2, 0.01) and tp.asf_reachable(hull, 0, 0.01) and tp.asf_reachable(hull, 1, 0.01)


def test_asf_reachable_is_false_for_a_dominated_alternative():
    N = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.3, 0.3, 0.4], [0.2, 0.2, 0.3]])     # Alternative 4 wird von 3 dominiert
    assert tp.asf_reachable(N, 2, 0.01) and not tp.asf_reachable(N, 3, 0.01)


def test_effective_and_equivalent_weights(w3):
    e = tp.effective_weights(w3, 2.0)
    assert e == pytest.approx(np.array([0.25, 0.09, 0.04]) / 0.38) and e.sum() == pytest.approx(1.0)
    assert tp.effective_weights(w3, 1.0) == pytest.approx(w3)
    q = tp.equivalent_weights(w3, 2.0)
    assert q == pytest.approx(np.sqrt(w3) / np.sqrt(w3).sum())
    assert tp.effective_weights(q, 2.0) == pytest.approx(w3)           # in der Quadratsumme wirken die gleichwertigen Gewichte wie w
    assert tp.effective_weights(w3, math.inf) == pytest.approx(w3) and tp.equivalent_weights(w3, math.inf) == pytest.approx(w3)


def test_reference_raw_modes():
    S = np.array([[10.0, -100.0], [30.0, -300.0], [20.0, -200.0]])
    lo, hi = np.array([0.0, -500.0]), np.array([100.0, -50.0])
    best, worst = tp.reference_raw(S, "stichprobe", lo, hi)
    assert best.tolist() == [30.0, -100.0] and worst.tolist() == [10.0, -300.0]
    best, worst = tp.reference_raw(S, "nenn", lo, hi)
    assert best.tolist() == [100.0, -50.0] and worst.tolist() == [0.0, -500.0]
    best, worst = tp.reference_raw(S, "anspruch", lo, hi, 0.5)
    assert best.tolist() == [20.0, -200.0] and worst.tolist() == [10.0, -300.0]
    assert tp.reference_raw(S, "anspruch", lo, hi, 1.0)[0].tolist() == [30.0, -100.0]
    with pytest.raises(ValueError):
        tp.reference_raw(S, "irgendwo", lo, hi)
