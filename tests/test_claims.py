"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Experimente 1, 2 und 4 laufen über SWEEP_SEEDS (200 feste Matrizen), die Reichweite über die ersten 60. Prozentwerte mit Toleranz 0,5 Punkte, Abstände und tau mit 0,01, damit
Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler einer Trefferquote über 200 Matrizen ist 1,3 bis 3,5 Punkte: Unterschiede unter
etwa 6 Punkten sind NICHT belastbar und stehen nirgends als Aussage.
"""

import math

import pytest

import tps_constants as C
import tps_evaluation as ev


@pytest.fixture(scope="module")
def identity():
    return ev.experiment_identity()


@pytest.fixture(scope="module")
def cost():
    return ev.experiment_cost()


@pytest.fixture(scope="module")
def agree():
    return ev.agreement()


@pytest.fixture(scope="module")
def reference():
    return ev.experiment_reference()


@pytest.fixture(scope="module")
def reach():
    return ev.experiment_reach()


@pytest.fixture(scope="module")
def weights():
    return ev.experiment_weights()


@pytest.fixture(scope="module")
def stability():
    return ev.experiment_stability()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


# ----------------------------------------------------------------------------- Experiment 1: p = 1 ist die Summe

def test_manhattan_topsis_is_the_sum_in_every_case(identity):
    p1 = [r for r in identity if r["p"] == 1.0]
    assert len(p1) == 9 and sum(r["cases"] for r in p1) == 1800
    assert all(r["same_ranking"] == 1.0 and r["same_top1"] == 1.0 and r["tau"] == pytest.approx(1.0) for r in p1)
    assert all(r["max_diff"] < 1e-12 for r in p1 if r["norm"] == "minmax")


@pytest.mark.parametrize("m,norm,p,top1", [
    (3, "minmax", 2.0, 88.5), (3, "vektor", 2.0, 88.0), (3, "z", 2.0, 88.5), (3, "minmax", math.inf, 77.5),
    (5, "minmax", 2.0, 84.0), (5, "vektor", 2.0, 83.5), (5, "z", 2.0, 84.0), (5, "minmax", math.inf, 75.5), (5, "vektor", math.inf, 73.5), (5, "z", math.inf, 77.0),
    (8, "minmax", 2.0, 79.5), (8, "vektor", 2.0, 76.5), (8, "z", 2.0, 79.0), (8, "minmax", math.inf, 61.0),
])
def test_euclid_and_chebyshev_agree_with_the_sum_only_partly(identity, m, norm, p, top1):
    r = cell(identity, m=m, norm=norm, p=p)
    assert pct(r["same_top1"]) == pytest.approx(top1, abs=0.5) and r["same_ranking"] <= 0.015


# ----------------------------------------------------------------------------- Experiment 2: was der Abstand kostet

COST = {
    ("guete_linear", 3): {"summe_minmax": 85.0, "summe_vektor": 81.5, "t_vektor_1": 81.5, "t_vektor_2": 77.0, "t_vektor_inf": 68.0, "t_minmax_1": 85.0, "t_minmax_2": 80.0,
                          "t_minmax_inf": 72.5, "t_z_2": 83.0, "t_perzentil_2": 74.0, "t_roh_2": 39.0, "asf_minmax": 67.5},
    ("guete_linear", 5): {"summe_minmax": 67.0, "summe_vektor": 70.5, "t_vektor_1": 70.5, "t_vektor_2": 65.0, "t_vektor_inf": 60.5, "t_minmax_1": 67.0, "t_minmax_2": 62.5,
                          "t_minmax_inf": 59.5, "t_z_2": 62.0, "t_perzentil_2": 72.5, "t_roh_2": 36.5, "asf_minmax": 56.0},
    ("guete_linear", 8): {"summe_minmax": 78.5, "summe_vektor": 81.0, "t_vektor_1": 81.0, "t_vektor_2": 66.5, "t_vektor_inf": 51.5, "t_minmax_1": 78.5, "t_minmax_2": 69.0,
                          "t_minmax_inf": 53.5, "t_z_2": 67.0, "t_perzentil_2": 73.5, "t_roh_2": 33.5, "asf_minmax": 53.5},
    ("guete_konkav", 5): {"summe_minmax": 72.5, "summe_vektor": 79.0, "t_vektor_1": 79.0, "t_vektor_2": 71.5, "t_vektor_inf": 66.5, "t_minmax_1": 72.5, "t_minmax_2": 65.5,
                          "t_minmax_inf": 62.5, "t_z_2": 66.5, "t_perzentil_2": 68.5, "t_roh_2": 35.0, "asf_minmax": 59.5},
    ("guete_konkav", 8): {"summe_minmax": 78.0, "summe_vektor": 79.5, "t_vektor_1": 79.5, "t_vektor_2": 69.5, "t_vektor_inf": 54.0, "t_minmax_1": 78.0, "t_minmax_2": 70.0,
                          "t_minmax_inf": 56.0, "t_z_2": 68.5, "t_perzentil_2": 71.0, "t_roh_2": 34.5, "asf_minmax": 57.5},
    ("guete_cobb", 5): {"summe_minmax": 69.0, "summe_vektor": 73.5, "t_vektor_1": 73.5, "t_vektor_2": 71.0, "t_vektor_inf": 62.0, "t_minmax_1": 69.0, "t_minmax_2": 64.0,
                        "t_minmax_inf": 59.5, "t_z_2": 64.0, "t_perzentil_2": 64.0, "t_roh_2": 34.5, "asf_minmax": 59.5},
    ("guete_cobb", 8): {"summe_minmax": 67.5, "summe_vektor": 69.5, "t_vektor_1": 69.5, "t_vektor_2": 58.5, "t_vektor_inf": 47.5, "t_minmax_1": 67.5, "t_minmax_2": 58.5,
                        "t_minmax_inf": 51.5, "t_z_2": 57.0, "t_perzentil_2": 60.0, "t_roh_2": 34.0, "asf_minmax": 53.0},
    ("roh_linear", 3): {"summe_minmax": 94.0, "summe_vektor": 92.5, "t_vektor_1": 92.5, "t_vektor_2": 85.5, "t_vektor_inf": 75.0, "t_minmax_1": 94.0, "t_minmax_2": 87.0,
                        "t_minmax_inf": 76.5, "t_z_2": 87.5, "t_perzentil_2": 72.0, "t_roh_2": 31.0, "asf_minmax": 73.5},
    ("roh_linear", 5): {"summe_minmax": 96.5, "summe_vektor": 92.0, "t_vektor_1": 92.0, "t_vektor_2": 79.5, "t_vektor_inf": 72.5, "t_minmax_1": 96.5, "t_minmax_2": 85.0,
                        "t_minmax_inf": 77.5, "t_z_2": 82.5, "t_perzentil_2": 62.0, "t_roh_2": 30.5, "asf_minmax": 74.0},
    ("roh_linear", 8): {"summe_minmax": 93.0, "summe_vektor": 89.0, "t_vektor_1": 89.0, "t_vektor_2": 74.5, "t_vektor_inf": 59.5, "t_minmax_1": 93.0, "t_minmax_2": 80.5,
                        "t_minmax_inf": 62.0, "t_z_2": 78.0, "t_perzentil_2": 63.5, "t_roh_2": 28.0, "asf_minmax": 61.5},
}


@pytest.mark.parametrize("decider,m", sorted(COST))
def test_first_place_for_every_method(cost, decider, m):
    for method, hit in COST[(decider, m)].items():
        assert pct(cell(cost, decider=decider, m=m, method=method)["hit"]) == pytest.approx(hit, abs=0.5), method


def test_the_sum_is_at_least_as_good_as_euclid_topsis_in_every_cell_with_the_same_normalization(cost):
    for dec in C.DECIDERS:
        for m in C.EXP_M:
            for s_key, t_key in (("summe_minmax", "t_minmax_2"), ("summe_vektor", "t_vektor_2")):
                assert cell(cost, decider=dec, m=m, method=s_key)["hit"] >= cell(cost, decider=dec, m=m, method=t_key)["hit"], (dec, m, s_key)


def test_the_gap_between_sum_and_euclid_on_linear_deciders_is_four_to_fifteen_points(cost):
    gaps = []
    for dec in ("guete_linear", "roh_linear"):
        for m in C.EXP_M:
            for s_key, t_key in (("summe_minmax", "t_minmax_2"), ("summe_vektor", "t_vektor_2")):
                gaps.append(pct(cell(cost, decider=dec, m=m, method=s_key)["hit"] - cell(cost, decider=dec, m=m, method=t_key)["hit"]))
    assert min(gaps) == pytest.approx(4.5, abs=0.5) and max(gaps) == pytest.approx(14.5, abs=0.5)
    raw_world = [pct(cell(cost, decider="roh_linear", m=m, method="summe_minmax")["hit"] - cell(cost, decider="roh_linear", m=m, method="t_minmax_2")["hit"]) for m in C.EXP_M]
    assert min(raw_world) == pytest.approx(7.0, abs=0.5) and max(raw_world) == pytest.approx(12.5, abs=0.5)


def test_higher_p_loses_in_every_cell_for_min_max(cost):
    for dec in C.DECIDERS:
        for m in C.EXP_M:
            h = {k: cell(cost, decider=dec, m=m, method=k)["hit"] for k in ("t_minmax_1", "t_minmax_2", "t_minmax_inf")}
            assert h["t_minmax_1"] >= h["t_minmax_2"] >= h["t_minmax_inf"], (dec, m)


def test_topsis_without_normalization_is_decided_by_the_units(cost):
    for dec in C.DECIDERS:
        for m in C.EXP_M:
            assert 28.0 <= pct(cell(cost, decider=dec, m=m, method="t_roh_2")["hit"]) <= 41.5


@pytest.mark.parametrize("a,b,top1,tau", [
    ("t_vektor_2", "summe_minmax", 80.5, 0.84), ("t_vektor_2", "t_minmax_2", 82.5, 0.88), ("t_minmax_2", "t_minmax_1", 84.0, 0.88),
    ("t_minmax_2", "t_minmax_inf", 88.0, 0.87), ("t_perzentil_2", "t_minmax_2", 65.0, 0.82), ("t_roh_2", "t_minmax_2", 28.5, 0.35),
])
def test_agreement_between_variants(agree, a, b, top1, tau):
    r = cell(agree, a=a, b=b)
    assert pct(r["same_top1"]) == pytest.approx(top1, abs=0.5) and r["tau"] == pytest.approx(tau, abs=0.01)


REFERENCE = {
    "guete_linear": {"stichprobe": (62.5, 56.0), "nenn": (61.0, 57.5), "anspruch": (18.5, 44.5)},
    "guete_konkav": {"stichprobe": (65.5, 59.5), "nenn": (63.0, 59.5), "anspruch": (23.5, 49.5)},
    "guete_cobb": {"stichprobe": (64.0, 59.5), "nenn": (60.5, 57.5), "anspruch": (23.5, 52.5)},
    "roh_linear": {"stichprobe": (85.0, 74.0), "nenn": (84.0, 75.0), "anspruch": (27.0, 59.0)},
}


@pytest.mark.parametrize("decider", sorted(REFERENCE))
def test_reference_point_source(reference, decider):
    for mode, (t, a) in REFERENCE[decider].items():
        assert pct(cell(reference, decider=decider, reference=mode, method="topsis")["hit"]) == pytest.approx(t, abs=0.5), (mode, "topsis")
        assert pct(cell(reference, decider=decider, reference=mode, method="asf")["hit"]) == pytest.approx(a, abs=0.5), (mode, "asf")


def test_aspiration_below_the_best_punishes_over_achievement_in_topsis(reference):
    for dec in C.DECIDERS:
        t = {m: cell(reference, decider=dec, reference=m, method="topsis")["hit"] for m in ("stichprobe", "nenn", "anspruch")}
        a = cell(reference, decider=dec, reference="anspruch", method="asf")["hit"]
        assert t["anspruch"] < t["stichprobe"] - 0.3 and a > t["anspruch"] + 0.2
        assert abs(t["stichprobe"] - t["nenn"]) < 0.04          # Ideal der Stichprobe und Nenn-Ideal: im Rauschen


# ----------------------------------------------------------------------------- Experiment 3: Reichweite

@pytest.mark.parametrize("m,front,lp,topsis_mm,topsis_vec,asf", [(3, 7.55, 74.8, 85.4, 85.2, 99.1), (5, 14.0, 79.2, 83.0, 84.2, 99.6)])
def test_reach(reach, m, front, lp, topsis_mm, topsis_vec, asf):
    r = cell(reach, m=m)
    assert r["front"] == pytest.approx(front, abs=0.05) and r["matrices"] == 60 and r["weights"] == 10000 + m
    assert pct(r["summe_lp"]) == pytest.approx(lp, abs=0.5) and pct(r["topsis_minmax"]) == pytest.approx(topsis_mm, abs=0.5)
    assert pct(r["topsis_vektor"]) == pytest.approx(topsis_vec, abs=0.5) and pct(r["asf"]) == pytest.approx(asf, abs=0.5)


def test_reach_ordering_and_the_sample_underestimates_the_sum(reach):
    for r in reach:
        assert r["summe_lp"] > r["summe"] and r["topsis_minmax"] > r["summe_lp"] and r["asf"] > r["topsis_minmax"]
    assert pct(cell(reach, m=5)["summe_lp"] - cell(reach, m=5)["summe"]) == pytest.approx(9.4, abs=0.6)


# ----------------------------------------------------------------------------- Experiment 4: Gewichte und Stabilität

WEIGHT_CLAIMS = {
    ("guete_linear", 5): (67.0, 67.5, 62.5, 58.5), ("guete_linear", 8): (78.5, 69.5, 69.0, 65.0),
    ("guete_konkav", 5): (72.5, 67.0, 65.5, 68.5), ("guete_konkav", 8): (78.0, 67.0, 70.0, 71.0),
    ("guete_cobb", 5): (69.0, 60.5, 64.0, 70.0), ("guete_cobb", 8): (67.5, 55.0, 58.5, 73.0),
    ("roh_linear", 5): (96.5, 75.0, 85.0, 79.5),
}


@pytest.mark.parametrize("decider,m", sorted(WEIGHT_CLAIMS))
def test_equivalent_weights_and_effective_weights(weights, decider, m):
    s, e, t, g = WEIGHT_CLAIMS[(decider, m)]
    got = {k: pct(cell(weights, decider=decider, m=m, method=k)["hit"]) for k in ("summe", "summe_wirksam", "topsis_w", "topsis_gleichwertig")}
    assert got["summe"] == pytest.approx(s, abs=0.5) and got["summe_wirksam"] == pytest.approx(e, abs=0.5)
    assert got["topsis_w"] == pytest.approx(t, abs=0.5) and got["topsis_gleichwertig"] == pytest.approx(g, abs=0.5)


def test_equivalent_weights_help_only_for_the_concave_deciders(weights):
    for m in (5, 8):
        for dec in ("guete_konkav", "guete_cobb"):
            assert cell(weights, decider=dec, m=m, method="topsis_gleichwertig")["hit"] > cell(weights, decider=dec, m=m, method="topsis_w")["hit"]
        for dec in ("guete_linear", "roh_linear"):
            if (dec, m) in WEIGHT_CLAIMS:
                assert cell(weights, decider=dec, m=m, method="topsis_gleichwertig")["hit"] <= cell(weights, decider=dec, m=m, method="topsis_w")["hit"]


STAY = {
    0.1: {"summe": 95.1, "topsis_vektor": 92.3, "topsis_minmax": 94.9, "topsis_gleichwertig": 95.8, "asf": 93.0},
    0.2: {"summe": 90.3, "topsis_vektor": 86.7, "topsis_minmax": 88.7, "topsis_gleichwertig": 92.5, "asf": 84.5},
    0.4: {"summe": 80.7, "topsis_vektor": 77.3, "topsis_minmax": 77.0, "topsis_gleichwertig": 85.3, "asf": 70.8},
}


@pytest.mark.parametrize("sigma", sorted(STAY))
def test_stability_of_first_place(stability, sigma):
    for method, v in STAY[sigma].items():
        assert pct(cell(stability, sigma=sigma, method=method)["stay"]) == pytest.approx(v, abs=0.6), method


def test_stability_ordering(stability):
    for sg in C.EXP_STABILITY_SIGMAS:
        s = {r["method"]: r["stay"] for r in stability if r["sigma"] == sg}
        assert s["topsis_gleichwertig"] > s["summe"] > s["topsis_minmax"], sg
        if sg >= 0.2:
            assert s["asf"] == min(s.values()), sg


def test_experiment_sizes_are_the_documented_ones():
    assert len(C.SWEEP_SEEDS) == 200 and C.EXP_N == 20 and len(C.REACH_SEEDS) == 60 and C.EXP_REACH_WEIGHTS == 10000 and C.RHO == 0.01
