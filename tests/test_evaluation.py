"""Auswertung: Fall der App (Invarianten), Referenzpunkte, Experimente im Kleinen, Verdikte."""

import math

import numpy as np
import pytest

import tps_constants as C
import tps_evaluation as ev
import tps_score as sc
import tps_topsis as tp
from entscheidungsmatrix_generator import SplitMix64


def test_make_case_is_deterministic_and_matches_the_decider_utility():
    a, b = ev.make_case(20, 5, 7), ev.make_case(20, 5, 7)
    assert (a.S == b.S).all() and (a.w == b.w).all()
    assert a.w.sum() == pytest.approx(1.0) and a.utility == pytest.approx(a.goodness @ a.w)
    c = ev.make_case(20, 5, 7, 1.0, "guete_konkav")
    assert c.utility == pytest.approx(np.sqrt(c.goodness) @ c.w)
    r = ev.make_case(20, 5, 7, 1.0, "roh_linear")
    lo, hi = sc.nominal_bounds(5)
    assert r.utility == pytest.approx(((r.S - lo) / (hi - lo)) @ r.w)


def test_narrowing_changes_only_the_first_column():
    full, narrow = ev.make_case(20, 5, 7, 1.0), ev.make_case(20, 5, 7, 0.05)
    assert (full.goodness[:, 1:] == narrow.goodness[:, 1:]).all()
    assert np.ptp(narrow.goodness[:, 0]) == pytest.approx(0.05 * np.ptp(full.goodness[:, 0]))


def test_input_weights_modes():
    w = np.array([0.6, 0.3, 0.1])
    assert (ev.input_weights(w, "wahr", 2.0) == w).all()
    assert ev.input_weights(w, "gleichwertig", 2.0) == pytest.approx(np.sqrt(w) / np.sqrt(w).sum())
    with pytest.raises(ValueError):
        ev.input_weights(w, "irgendwie", 2.0)


def test_reference_points_in_the_normalized_space():
    case = ev.make_case(20, 4, 3)
    ideal, anti, ideal_raw, anti_raw = ev.reference_points(case.S, "minmax", "stichprobe", 1.0)
    assert ideal.tolist() == pytest.approx([1.0] * 4) and anti.tolist() == pytest.approx([0.0] * 4)
    assert ideal_raw.tolist() == case.S.max(axis=0).tolist()
    ideal, anti, _, _ = ev.reference_points(case.S, "minmax", "anspruch", 0.5)
    assert ideal.tolist() == pytest.approx([0.5] * 4) and anti.tolist() == pytest.approx([0.0] * 4)
    ideal, anti, _, _ = ev.reference_points(case.S, "minmax", "nenn", 1.0)
    assert (ideal >= 1.0).all() and (anti <= 0.0).all()       # die Nenn-Spannweite umfasst die Stichprobe


def test_analyse_basic_invariants():
    a = ev.analyse(ev.Settings())
    assert a.w_in.sum() == pytest.approx(1.0) and a.w_eff.sum() == pytest.approx(1.0)
    assert a.closeness == pytest.approx(a.d_minus / (a.d_plus + a.d_minus))
    assert set(a.results) == {"wahr", "summe", "summe_wirksam", "topsis", "asf"} and a.best == int(np.argmax(a.case.utility))
    for k, r in a.results.items():
        assert r.hit == (r.top1 == a.best) and r.loss >= 0 and -1 <= r.tau <= 1
        assert a.sum_reach[k] == bool(a.front[r.top1] and sc.is_possible_winner(a.N, r.top1))
    assert a.results["topsis"].top1 == int(np.argmax(a.closeness))


def test_manhattan_case_has_the_ranking_of_the_sum():
    for seed in range(1, 15):
        a = ev.analyse(ev.Settings(norm="minmax", p=1.0, seed=seed))
        assert np.array_equal(np.argsort(-a.closeness, kind="stable"), np.argsort(-a.results["summe"].score, kind="stable"))
        assert a.closeness == pytest.approx(a.results["summe"].score)


def test_equivalent_weights_make_the_squared_distance_a_weighted_sum_of_squares():
    a = ev.analyse(ev.Settings(norm="minmax", p=2.0, weights="gleichwertig", seed=5))
    assert a.w_in == pytest.approx(np.sqrt(a.case.w) / np.sqrt(a.case.w).sum())
    assert a.w_eff == pytest.approx(a.case.w)               # in der Quadratsumme wirken sie wie die wahren Gewichte


def test_chebyshev_case_uses_the_reference_point_of_the_setting():
    a = ev.analyse(ev.Settings(norm="minmax", p=math.inf, reference="anspruch", aspiration=0.6, seed=2))
    assert a.ideal.tolist() == pytest.approx([0.6] * 5)
    expected = -tp.asf_distance(a.N, a.w_in, a.ideal, C.RHO)
    assert a.results["asf"].score == pytest.approx(expected)


def test_p_label():
    assert ev.p_label(1.0) == "1" and ev.p_label(2.0) == "2" and ev.p_label(1.5) == "1,5" and ev.p_label(math.inf) == "∞"


def test_draw_weights_are_on_the_simplex_and_deterministic():
    W = ev.draw_weights(5, 200, SplitMix64(1))
    assert W.shape == (200, 5) and W.sum(axis=1) == pytest.approx(np.ones(200)) and (W > 0).all()
    assert (ev.draw_weights(5, 200, SplitMix64(1)) == W).all()


def test_experiment_identity_small_run():
    rows = ev.experiment_identity(ms=(3,), norms=("minmax", "vektor"), ps=(1.0, 2.0), seeds=tuple(range(200000, 200015)))
    p1 = [r for r in rows if r["p"] == 1.0]
    assert all(r["same_ranking"] == 1.0 and r["same_top1"] == 1.0 for r in p1)
    assert next(r for r in p1 if r["norm"] == "minmax")["max_diff"] < 1e-12
    assert all(r["same_ranking"] < 1.0 for r in rows if r["p"] == 2.0)


def test_cost_scores_cover_every_method_and_agree_with_the_building_blocks():
    c = ev.make_case(20, 5, 3)
    v = ev.cost_scores(c.S, c.w)
    assert set(v) == set(ev.COST_METHODS)
    nm = sc.normalize(c.S, "minmax")
    assert v["t_minmax_2"] == pytest.approx(tp.closeness(nm, c.w, 2.0)[2]) and v["t_minmax_1"] == pytest.approx(nm @ c.w)
    assert v["asf_minmax"] == pytest.approx(-tp.asf_distance(nm, c.w, None, C.RHO))


def test_experiment_cost_small_run():
    rows = ev.experiment_cost(ms=(3,), deciders=("roh_linear",), seeds=tuple(range(200000, 200010)))
    assert len(rows) == len(ev.COST_METHODS) and all(0 <= r["hit"] <= 1 and r["se"] >= 0 for r in rows)


def test_experiment_reference_small_run_and_verdict():
    rows = ev.experiment_reference(deciders=("roh_linear",), seeds=tuple(range(200000, 200010)))
    assert len(rows) == 6 and {r["reference"] for r in rows} == {"stichprobe", "nenn", "anspruch"}
    full = [{"decider": "guete_linear", "m": 5, "reference": ref, "method": k, "hit": h}
            for ref, ht, ha in (("stichprobe", 0.625, 0.56), ("nenn", 0.61, 0.575), ("anspruch", 0.185, 0.445)) for k, h in (("topsis", ht), ("asf", ha))]
    kind, text = ev.verdict_reference(full)
    assert kind == "warning" and "18 bis 18 %" in text and "62 bis 62 %" in text and "44 bis 44 %" in text and "80 %" in text


def test_possible_winners_by_sampling_on_the_hull_instance(hull):
    W = np.vstack([ev.draw_weights(2, 500, SplitMix64(3)), np.eye(2)])
    assert ev.possible_winners_by_sampling(hull, W, "summe") == {0, 1}
    assert ev.possible_winners_by_sampling(hull, W, "topsis", 2.0) == {0, 1}
    assert 2 in ev.possible_winners_by_sampling(hull, W, "asf")
    with pytest.raises(ValueError):
        ev.possible_winners_by_sampling(hull, W, "orakel")


def test_experiment_reach_small_run_orders_the_methods():
    r = ev.experiment_reach(ms=(3,), seeds=tuple(range(200000, 200008)), weights=500)[0]
    assert r["asf"] >= r["summe_lp"] and r["topsis_minmax"] >= r["summe"] - 0.05 and 0 < r["summe_lp"] <= 1


def test_experiment_weights_and_stability_small_runs():
    rows = ev.experiment_weights(ms=(5,), deciders=("guete_cobb",), seeds=tuple(range(200000, 200010)))
    assert {r["method"] for r in rows} == {"summe", "summe_wirksam", "topsis_w", "topsis_gleichwertig"}
    st = ev.experiment_stability(sigmas=(0.0, 0.4), seeds=tuple(range(200000, 200010)), draws=3)
    assert all(r["stay"] == 1.0 for r in st if r["sigma"] == 0.0) and all(0 <= r["stay"] <= 1 for r in st)


def test_experiments_are_deterministic():
    assert ev.experiment_stability(sigmas=(0.2,), seeds=(200000, 200001), draws=3) == ev.experiment_stability(sigmas=(0.2,), seeds=(200000, 200001), draws=3)


def test_verdicts_read_their_numbers_from_the_rows():
    ident = [{"m": 5, "norm": nm, "p": p, "same_ranking": 1.0 if p == 1.0 else 0.0, "same_top1": 1.0 if p == 1.0 else s, "tau": 1.0, "max_diff": 0.0, "cases": 200}
             for nm, s in (("minmax", 0.84), ("vektor", 0.835)) for p in (1.0, 2.0)]
    kind, text = ev.verdict_identity(ident)
    assert kind == "info" and "100 %" in text and "400 Fällen" in text and "84 %" in text and "5 Kriterien" in text
    cost = [{"decider": d, "m": m, "method": k, "hit": h} for d in ("guete_linear", "roh_linear") for m in (3, 5, 8)
            for k, h in (("summe_minmax", 0.96), ("t_minmax_2", 0.85), ("asf_minmax", 0.74))]
    kind, text = ev.verdict_cost(cost)
    assert kind == "warning" and "96 %" in text and "85 %" in text and "74 %" in text and "11,0 bis 11,0" in text
    kind, text = ev.verdict_reach([{"m": 5, "front": 14.0, "matrices": 60, "weights": 100, "summe_lp": 0.79, "summe": 0.7, "topsis_minmax": 0.83, "topsis_vektor": 0.84, "asf": 0.996}])
    assert kind == "info" and "79 %" in text and "83 %" in text and "100 %" in text
    wrows = [{"decider": "guete_cobb", "m": 8, "method": "topsis_gleichwertig", "hit": 0.73}, {"decider": "guete_cobb", "m": 8, "method": "topsis_w", "hit": 0.585}]
    srows = [{"sigma": 0.4, "method": k, "stay": v} for k, v in (("summe", 0.81), ("topsis_minmax", 0.77), ("topsis_gleichwertig", 0.85))]
    kind, text = ev.verdict_weights(wrows, srows)
    assert kind == "info" and "81 %" in text and "77 %" in text and "85 %" in text and "73,0 gegen 58,5 %" in text
