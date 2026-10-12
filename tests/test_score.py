"""Score-Helfer: Normierungen (auch für beliebige Punkte), Summe, Wertskalen, Verschmälerung, Dominanz, LP der Reichweite."""

import numpy as np
import pytest

import tps_score as sc


def test_normalize_hand_values():
    S = np.array([[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]])
    assert sc.normalize(S, "minmax").tolist() == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    assert sc.normalize(S, "roh").tolist() == S.tolist()
    assert sc.normalize(S, "vektor")[:, 0] == pytest.approx(np.array([1, 3, 5]) / np.sqrt(35))
    z = sc.normalize(S, "z")
    assert z[:, 0] == pytest.approx([-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    assert sc.normalize(S, "perzentil").tolist() == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]


def test_normalize_constant_column_becomes_zero_or_equal():
    S = np.array([[2.0, 5.0], [2.0, 7.0], [2.0, 9.0]])
    assert sc.normalize(S, "minmax")[:, 0].tolist() == [0.0, 0.0, 0.0]
    assert sc.normalize(S, "z")[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_normalize_points_matches_normalize_on_the_sample_itself():
    rng = np.random.default_rng(2)
    S = rng.normal(size=(7, 4))
    for how in ("minmax", "z", "vektor", "roh", "perzentil"):
        assert sc.normalize_points(S, S, how) == pytest.approx(sc.normalize(S, how))


def test_normalize_points_maps_outside_values_with_the_sample_parameters():
    S = np.array([[0.0], [10.0]])
    X = np.array([[-10.0], [5.0], [20.0]])
    assert sc.normalize_points(S, X, "minmax").ravel().tolist() == [-1.0, 0.5, 2.0]
    assert sc.normalize_points(S, X, "perzentil").ravel().tolist() == [0.0, 1.0, 1.0]      # Anteil der Stichprobenwerte darunter, begrenzt auf [0, 1]


def test_normalize_rejects_unknown_method():
    with pytest.raises(ValueError):
        sc.normalize(np.ones((2, 2)), "tanh")


def test_weighted_score_and_top1_tie_goes_to_the_first():
    N = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    s = sc.weighted_score(N, np.array([0.5, 0.5]))
    assert s.tolist() == [0.5, 0.5, 0.5] and sc.top1(s) == 0


def test_narrow_criterion_shrinks_around_one_half_and_leaves_the_rest():
    g = np.array([[0.0, 0.3], [1.0, 0.7], [0.5, 0.9]])
    h = sc.narrow_criterion(g, 0.2)
    assert h[:, 0].tolist() == pytest.approx([0.4, 0.6, 0.5]) and (h[:, 1] == g[:, 1]).all()
    assert (sc.narrow_criterion(g, 1.0) == g).all()


def test_decider_utility_on_every_form():
    g = np.array([[0.25, 0.81], [1.0, 0.04]])
    w = np.array([0.5, 0.5])
    S = np.array([[20.0, -1000.0], [80.0, -4000.0]])
    assert sc.decider_utility("guete", "linear", g, S, w, 2).tolist() == pytest.approx([0.53, 0.52])
    assert sc.decider_utility("guete", "konkav", g, S, w, 2).tolist() == pytest.approx([0.5 * 0.5 + 0.5 * 0.9, 0.5 * 1.0 + 0.5 * 0.2])
    assert sc.decider_utility("guete", "cobb", g, S, w, 2).tolist() == pytest.approx(
        [0.5 * np.log(0.25) + 0.5 * np.log(0.81), 0.5 * np.log(1.0) + 0.5 * np.log(0.04)])
    u = sc.decider_utility("roh", "linear", g, S, w, 2)       # Kriterium 2: Euro, Nenn [800, 5000], Vorzeichen: [-5000, -800]
    assert u.tolist() == pytest.approx([0.5 * 0.2 + 0.5 * (4000 / 4200), 0.5 * 0.8 + 0.5 * (1000 / 4200)])
    with pytest.raises(ValueError):
        sc.decider_utility("roh", "cobb", g, S, w, 2)
    with pytest.raises(ValueError):
        sc.decider_utility("zehn", "linear", g, S, w, 2)


def test_kendall_tau_b_hand_values():
    assert sc.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert sc.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert sc.kendall_tau_b([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)
    assert sc.kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0


def test_utility_loss():
    assert sc.utility_loss(np.array([0.2, 0.9, 0.5]), 2) == pytest.approx(0.4)
    assert sc.utility_loss(np.array([0.2, 0.9, 0.5]), 1) == 0.0


def test_dominance_and_front():
    G = np.array([[1.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    assert sc.front_mask(G).tolist() == [True, False, False, False]      # (1,1) dominiert alle anderen
    G2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4], [0.3, 0.3]])
    assert sc.front_mask(G2).tolist() == [True, True, True, False]
    assert sc.front_mask(np.array([[0.5, 0.5], [0.5, 0.5]])).tolist() == [True, True]      # identische Punkte dominieren sich nicht


def test_possible_winner_lp_on_the_hull_instance(hull):
    assert sc.is_possible_winner(hull, 0) and sc.is_possible_winner(hull, 1)
    assert not sc.is_possible_winner(hull, 2)                           # C unter der Hülle
    assert sc.is_possible_winner(np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]]), 2)      # C über der Hülle gewinnt bei gleichen Gewichten


def test_possible_winner_is_false_for_a_duplicate():
    assert not sc.is_possible_winner(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 0)


def test_percentile_with_two_alternatives_and_one_alternative():
    """Mutationstest: Der Zweig n < 2 darf bei genau zwei Alternativen nicht greifen (Ränge 0 und 1), bei einer Alternative liefert er Nullen."""
    assert sc.normalize(np.array([[1.0, 9.0], [5.0, 3.0]]), "perzentil").tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert sc.normalize(np.array([[1.0, 9.0]]), "perzentil").tolist() == [[0.0, 0.0]]
