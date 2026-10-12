"""Orakel (DEMO-PLAYBOOK §12): unabhängige Rechenwege für die Nähe, die Identität p = 1 = Summe, Invarianzen, die Reichweite und die wirksamen Gewichte.

Jede Eigenschaft hat eine Gegenprobe, die zeigt, dass der Test etwas trennen kann (p = 2 ist nicht die Summe; Normierung „roh“ ist nicht skaleninvariant; ein Punkt unter
der Hülle gewinnt nur beim Referenzpunkt-Verfahren)."""

import math

import numpy as np
import pytest
from scipy.spatial import distance

import tps_score as sc
import tps_topsis as tp


def _random_case(rng, n=12, m=4):
    S = rng.uniform(1.0, 100.0, size=(n, m)) * rng.uniform(0.1, 50.0, size=m)
    w = rng.dirichlet(np.ones(m))
    return S, w


def _closeness_loop(N, w, p):
    """Definition als Doppelschleife mit scipy.spatial.distance."""
    n, m = N.shape
    V = N * w
    vp = np.array([max(V[i, j] for i in range(n)) for j in range(m)])
    vm = np.array([min(V[i, j] for i in range(n)) for j in range(m)])
    out = np.empty(n)
    for i in range(n):
        if math.isinf(p):
            dp, dm = distance.chebyshev(V[i], vp), distance.chebyshev(V[i], vm)
        else:
            dp, dm = distance.minkowski(V[i], vp, p), distance.minkowski(V[i], vm, p)
        out[i] = dm / (dp + dm)
    return out


# ----------------------------------------------------------------------------- 1. Nähe gegen scipy

@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, math.inf])
def test_closeness_equals_the_definition_loop_with_scipy_distances(p):
    rng = np.random.default_rng(1)
    for _ in range(25):
        S, w = _random_case(rng)
        N = sc.normalize(S, "vektor")
        assert tp.closeness(N, w, p)[2] == pytest.approx(_closeness_loop(N, w, p), abs=1e-12)


# ----------------------------------------------------------------------------- 2. p = 1 ist die Summe

def test_manhattan_topsis_is_the_weighted_sum_for_every_affine_normalization():
    rng = np.random.default_rng(2)
    for _ in range(500):
        S, w = _random_case(rng, n=int(rng.integers(4, 25)), m=int(rng.integers(2, 8)))
        for how in ("minmax", "vektor", "z"):
            N = sc.normalize(S, how)
            c = tp.closeness(N, w, 1.0)[2]
            ws = N @ w
            assert np.array_equal(np.argsort(-c, kind="stable"), np.argsort(-ws, kind="stable"))
        N = sc.normalize(S, "minmax")
        assert tp.closeness(N, w, 1.0)[2] == pytest.approx(N @ w, abs=1e-12)


def test_euclidean_topsis_is_not_the_weighted_sum():
    """Gegenprobe zur Identität: Bei p = 2 stimmt die Rangfolge nur selten mit der Summe überein."""
    rng = np.random.default_rng(3)
    same, top_same = 0, 0
    total = 300
    for _ in range(total):
        S, w = _random_case(rng, n=15, m=5)
        N = sc.normalize(S, "minmax")
        c = tp.closeness(N, w, 2.0)[2]
        ws = N @ w
        same += np.array_equal(np.argsort(-c, kind="stable"), np.argsort(-ws, kind="stable"))
        top_same += np.argmax(c) == np.argmax(ws)
    assert same / total < 0.3 and 0.5 < top_same / total < 0.98


# ----------------------------------------------------------------------------- 3. Invarianzen

@pytest.mark.parametrize("how", ["vektor", "minmax", "z"])
def test_rescaling_a_criterion_leaves_the_ranking_unchanged(how):
    rng = np.random.default_rng(4)
    for _ in range(30):
        S, w = _random_case(rng)
        T = S.copy()
        T[:, 1] *= 1000.0
        for p in (1.0, 2.0, math.inf):
            a = tp.closeness(sc.normalize(S, how), w, p)[2]
            b = tp.closeness(sc.normalize(T, how), w, p)[2]
            assert b == pytest.approx(a, abs=1e-9)


def test_raw_values_are_not_scale_invariant():
    """Gegenprobe: Ohne Normierung kippt die Rangfolge, wenn eine Einheit (Euro gegen Cent) gewechselt wird."""
    rng = np.random.default_rng(5)
    changed = 0
    for _ in range(40):
        S, w = _random_case(rng)
        T = S.copy()
        T[:, 1] *= 1000.0
        a = tp.closeness(sc.normalize(S, "roh"), w, 2.0)[2]
        b = tp.closeness(sc.normalize(T, "roh"), w, 2.0)[2]
        changed += int(np.argmax(a) != np.argmax(b))
    assert changed >= 5


def test_scaling_all_weights_by_a_factor_leaves_the_closeness_unchanged():
    rng = np.random.default_rng(6)
    S, w = _random_case(rng)
    N = sc.normalize(S, "minmax")
    for p in (1.0, 2.0, math.inf):
        assert tp.closeness(N, 7.3 * w, p)[2] == pytest.approx(tp.closeness(N, w, p)[2], abs=1e-12)


# ----------------------------------------------------------------------------- 4. Erfüllungsskalarisierung und Reichweite

def test_asf_equals_the_definition_loop():
    rng = np.random.default_rng(7)
    for _ in range(20):
        S, w = _random_case(rng, n=9, m=4)
        N = sc.normalize(S, "minmax")
        ref = N.max(axis=0)
        loop = [max(w[j] * (ref[j] - N[i, j]) for j in range(4)) + 0.01 * sum(w[j] * (ref[j] - N[i, j]) for j in range(4)) for i in range(9)]
        assert tp.asf_distance(N, w, None, 0.01) == pytest.approx(loop, abs=1e-12)


def test_asf_reach_is_not_smaller_than_the_reach_of_the_sum_on_random_matrices():
    """Über 30 Zufallsmatrizen: Der Referenzpunkt (Stichprobe und Konstruktion) erreicht mindestens so viele nichtdominierte Alternativen wie die Summe (LP)."""
    import tps_evaluation as ev
    from entscheidungsmatrix_generator import SplitMix64

    W = np.vstack([ev.draw_weights(4, 3000, SplitMix64(11)), np.eye(4)])
    lp = asf = front_total = 0
    for s in range(200000, 200030):
        c = ev.make_case(15, 4, s)
        nm = sc.normalize(c.S, "minmax")
        front = set(np.flatnonzero(sc.front_mask(c.goodness)).tolist())
        front_total += len(front)
        lp += sum(sc.is_possible_winner(nm, i) for i in front)
        reached = ev.possible_winners_by_sampling(nm, W, "asf") | {i for i in front if tp.asf_reachable(nm, i)}
        asf += len(front & reached)
    assert asf >= lp and asf / front_total > 0.9 and lp / front_total < 0.9


def test_a_point_under_the_hull_is_never_a_sum_winner_but_is_found_by_the_reference_point(hull):
    assert not sc.is_possible_winner(hull, 2)
    assert tp.asf_reachable(hull, 2)
    for p in (1.0, 2.0, math.inf):
        wins = {int(np.argmax(tp.closeness(hull, np.array([a, 1 - a]), p)[2])) for a in np.linspace(0.01, 0.99, 99)}
        assert 2 not in wins


def test_dominated_alternatives_never_win_any_distance_rule():
    rng = np.random.default_rng(8)
    for _ in range(20):
        S, w = _random_case(rng, n=10, m=3)
        N = sc.normalize(S, "minmax")
        front = sc.front_mask(N)
        for p in (1.0, 2.0, math.inf):
            assert front[int(np.argmax(tp.closeness(N, w, p)[2]))]
        assert front[int(np.argmin(tp.asf_distance(N, w, None, 0.01)))]


# ----------------------------------------------------------------------------- 5. Wirksame Gewichte

def test_squared_distance_with_equivalent_weights_is_the_weighted_sum_of_squares():
    rng = np.random.default_rng(9)
    for _ in range(20):
        S, w = _random_case(rng, n=8, m=5)
        N = sc.normalize(S, "minmax")
        ideal = N.max(axis=0)
        q = tp.equivalent_weights(w, 2.0)
        d = tp.distances(N, q, ideal, 2.0)
        expected = ((w * (ideal - N) ** 2).sum(axis=1)) / np.sqrt(w).sum() ** 2
        assert d ** 2 == pytest.approx(expected, abs=1e-12)


def test_effective_weights_describe_the_distance_ranking_for_p_two():
    """Der Rang von -d+ (Euklid, Gewichte w) ist der Rang der negativen Summe von (w_j (r_j - n_ij))^2; die wirksamen Gewichte w^2 sind genau die der quadrierten Abweichungen."""
    rng = np.random.default_rng(10)
    S, w = _random_case(rng, n=10, m=4)
    N = sc.normalize(S, "minmax")
    d = tp.distances(N, w, N.max(axis=0), 2.0)
    eff = tp.effective_weights(w, 2.0)
    expected = (eff * (N.max(axis=0) - N) ** 2).sum(axis=1)
    assert np.array_equal(np.argsort(d), np.argsort(expected))
    assert d ** 2 == pytest.approx(expected * (w ** 2).sum(), abs=1e-12)
