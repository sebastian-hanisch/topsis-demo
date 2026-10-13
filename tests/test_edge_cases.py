"""Randfälle und Verdikt-Schwellen aus dem Mutationstest: kleine, von Hand gerechnete Matrizen (n = 1, n = 2, identische Alternativen, Gleichstand der letzten
zwei Werte, beide Abstände 0) und synthetische Zeilen für die Verdikte."""

import math

import numpy as np
import pytest

import tps_constants as C
import tps_evaluation as ev
import tps_score as sc
import tps_topsis as tp


# ----------------------------------------------------------------------------- Ränge

def test_average_ranks_ties_reach_the_last_position():
    """Ein Gleichstand an der letzten Stelle muss zusammengefasst werden: [1, 2, 2] hat die Ränge 1; 2,5; 2,5 (nicht 1; 2; 3)."""
    assert sc.average_ranks(np.array([1.0, 2.0, 2.0])).tolist() == [1.0, 2.5, 2.5]
    assert sc.average_ranks(np.array([3.0, 3.0])).tolist() == [1.5, 1.5]
    assert sc.average_ranks(np.array([5.0, 5.0, 5.0])).tolist() == [2.0, 2.0, 2.0]
    assert sc.average_ranks(np.array([2.0, 1.0, 2.0])).tolist() == [2.5, 1.0, 2.5]           # sortiert 1, 2, 2: die beiden Zweien teilen sich Rang 2 und 3
    assert sc.average_ranks(np.array([4.0, 4.0, 1.0, 9.0, 9.0])).tolist() == [2.5, 2.5, 1.0, 4.5, 4.5]


def test_average_ranks_agree_with_scipy_on_coarse_values():
    from scipy.stats import rankdata

    rng = np.random.default_rng(11)
    for _ in range(50):
        col = rng.integers(0, 4, size=int(rng.integers(1, 9))).astype(float)         # grobe Werte: viele Gleichstände, auch am Ende
        assert sc.average_ranks(col).tolist() == rankdata(col, method="average").tolist()


# ----------------------------------------------------------------------------- Reichweite der Summe (LP)

def test_possible_winner_with_a_single_alternative_wins_trivially():
    assert sc.is_possible_winner(np.array([[0.3, 0.7]]), 0) is True


def test_possible_winner_with_exactly_two_alternatives():
    """n = 2: ein Gegner. Eine dominierte Alternative kann nicht gewinnen, eine identische nicht strikt, unvergleichbare gewinnen beide."""
    dominated = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert sc.is_possible_winner(dominated, 0) is False and sc.is_possible_winner(dominated, 1) is True
    identical = np.array([[0.4, 0.6], [0.4, 0.6]])
    assert sc.is_possible_winner(identical, 0) is False and sc.is_possible_winner(identical, 1) is False
    incomparable = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert sc.is_possible_winner(incomparable, 0) is True and sc.is_possible_winner(incomparable, 1) is True


# ----------------------------------------------------------------------------- Reichweite der Erfüllungsskalarisierung

def test_asf_reachable_with_a_single_alternative_is_true():
    assert tp.asf_reachable(np.array([[0.3, 0.7]]), 0) is True


def test_asf_reachable_with_exactly_two_alternatives():
    """n = 2: Gegen genau einen anderen Punkt entscheidet der Vergleich selbst. Dominierte Alternative: 0,51 gegen 0 (Referenz (1, 1), w = (0,5; 0,5), rho = 0,01)."""
    dominated = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert tp.asf_reachable(dominated, 0) is False and tp.asf_reachable(dominated, 1) is True
    incomparable = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert tp.asf_reachable(incomparable, 0) is True and tp.asf_reachable(incomparable, 1) is True


def test_asf_reachable_is_false_for_an_exact_tie():
    """Identische Alternativen haben bei jeder Gewichtung exakt denselben Wert: ein strikter Sieg ist unmöglich (der Abstand 1e-12 darf einen Gleichstand nicht zum Sieg machen)."""
    same = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert tp.asf_reachable(same, 0) is False and tp.asf_reachable(same, 1) is False
    dup = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert tp.asf_reachable(dup, 0) is False and tp.asf_reachable(dup, 1) is False and tp.asf_reachable(dup, 2) is True


# ----------------------------------------------------------------------------- TOPSIS: beide Abstände 0

def test_closeness_without_any_spread_is_one_half_and_never_divides_by_zero():
    """Lauter gleiche Zeilen (oder eine einzige Alternative): Ideal = Anti-Ideal = jede Zeile, d+ = d- = 0, C = 0,5. Die Division 0/0 darf gar nicht erst ausgeführt werden
    (die Ersatzteiler 1 ersetzt den Teiler 0): unter errstate('raise') bricht jede 0/0-Division mit FloatingPointError ab."""
    w = np.array([0.4, 0.6])
    with np.errstate(all="raise"):
        for N in (np.array([[0.2, 0.9], [0.2, 0.9], [0.2, 0.9]]), np.array([[0.2, 0.9]])):
            for p in (1.0, 2.0, math.inf):
                dp, dm, c = tp.closeness(N, w, p)
                assert dp.tolist() == [0.0] * len(N) and dm.tolist() == [0.0] * len(N) and c.tolist() == [0.5] * len(N)
        dp, dm, c = tp.closeness(np.array([[0.5, 0.5], [1.0, 0.5]]), np.array([0.5, 0.5]), 1.0, ideal=np.array([0.5, 0.5]), anti=np.array([0.5, 0.5]))
        assert dp.tolist() == dm.tolist() == [0.0, 0.25] and c.tolist() == [0.5, 0.5]       # gleiche Referenzpunkte: Zeile 2 hat d+ = d- = 0,25


# ----------------------------------------------------------------------------- Auswertung: Reichweite per Stichprobe (Tschebyschow)

def test_possible_winners_by_sampling_asf_uses_the_shortfall_to_the_ideal_point(hull):
    """hull: A = (1, 0), B = (0, 1), C = (0,4; 0,4), Ideal (1, 1), rho = 0,01. Je eine Gewichtung, Wert = max_j w_j (1 - N_j) + rho Summe (kleiner ist besser):
    w = (0,9; 0,1): A 0,1 + 0,001 = 0,101, B 0,9 + 0,009, C 0,54 + 0,006 -> A. w = (0,1; 0,9) spiegelbildlich -> B. w = (0,5; 0,5): A = B = 0,505, C = 0,3 + 0,006 -> C."""
    assert ev.possible_winners_by_sampling(hull, np.array([[0.9, 0.1]]), "asf") == {0}
    assert ev.possible_winners_by_sampling(hull, np.array([[0.1, 0.9]]), "asf") == {1}
    assert ev.possible_winners_by_sampling(hull, np.array([[0.5, 0.5]]), "asf") == {2}
    assert ev.possible_winners_by_sampling(hull, np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]), "asf") == {0, 1, 2}


# ----------------------------------------------------------------------------- Auswertung: Experimente

def test_experiment_identity_max_diff_is_the_largest_deviation_over_all_cases():
    seeds = tuple(range(200000, 200012))
    rows = ev.experiment_identity(ms=(3,), norms=("minmax",), ps=(2.0,), seeds=seeds)
    diffs = []
    for s in seeds:
        c = ev.make_case(C.EXP_N, 3, s)
        N = sc.normalize(c.S, "minmax")
        diffs.append(float(np.abs(tp.closeness(N, c.w, 2.0)[2] - sc.weighted_score(N, c.w)).max()))
    assert min(diffs) < max(diffs) and max(diffs) > 0.01                  # mit p = 2 weichen Nähe und Summe wirklich ab, und die Fälle unterscheiden sich
    assert len(rows) == 1 and rows[0]["max_diff"] == pytest.approx(max(diffs), abs=1e-15) and rows[0]["cases"] == len(seeds)


def test_experiment_cost_standard_error_is_the_binomial_one():
    """se = sqrt(h (1 - h) / k) mit k Matrizen: bei h = 0 oder 1 ist er 0, bei h = 0,5 am größten."""
    seeds = tuple(range(200000, 200010))
    rows = ev.experiment_cost(ms=(3,), deciders=("roh_linear",), seeds=seeds)
    assert any(0.0 < r["hit"] < 1.0 for r in rows)                        # sonst prüft der Test nur die Nullstellen
    for r in rows:
        assert r["se"] == pytest.approx(math.sqrt(r["hit"] * (1.0 - r["hit"]) / len(seeds)), abs=1e-15)
    assert all(r["se"] == 0.0 for r in rows if r["hit"] in (0.0, 1.0))


def test_experiment_stability_draws_exactly_the_requested_number_of_perturbations(monkeypatch):
    """Die Zahl der Störungen ist beobachtbar: je Matrix und Sigma ein Basisaufruf plus `draws` gestörte. Ein Ersatz liefert für den Basisaufruf und die erste Störung
    Platz 0, danach Platz 1; bei drei Störungen bleibt also genau eine von drei gleich (1/3), bei zwei wären es 1/2."""
    calls = []

    def fake(S, w):
        calls.append(1)
        return {k: (0 if len(calls) <= 2 else 1) for k in ev.STABILITY_METHODS}

    monkeypatch.setattr(ev, "_stability_tops", fake)
    rows = ev.experiment_stability(sigmas=(0.3,), seeds=(200000,), draws=3)
    assert len(calls) == 1 + 3
    assert [r["stay"] for r in rows] == [pytest.approx(1 / 3)] * len(ev.STABILITY_METHODS)

    calls.clear()
    rows = ev.experiment_stability(sigmas=(0.1, 0.3), seeds=(200000, 200001), draws=4)
    assert len(calls) == 2 * 2 * (1 + 4) and len(rows) == 2 * len(ev.STABILITY_METHODS)


# ----------------------------------------------------------------------------- Verdikte: Schwellen und Auswahl der Zeilen

def test_verdict_identity_takes_the_minimum_over_p1_and_the_range_over_p2_at_the_stability_m():
    """Drei Kriterienzahlen, drei Normierungen. p = 1: eine Zelle hat nur 90 % identische Rangfolgen, 9 x 200 = 1800 Fälle. p = 2 bei m = 5: Platz 1 stimmt in
    70, 90 und 80 % überein -> 70 bis 90 %. Die Zeilen bei m = 3 und 8 (10 und 99 %) gehören nicht in die Spanne."""
    rows = []
    for m in (3, 5, 8):
        for i, norm in enumerate(("minmax", "vektor", "z")):
            rows.append({"m": m, "norm": norm, "p": 1.0, "same_ranking": 0.9 if (m, norm) == (8, "z") else 1.0, "same_top1": 1.0, "tau": 1.0, "max_diff": 0.0, "cases": 200})
            top = (0.7, 0.9, 0.8)[i] if m == C.EXP_STABILITY_M else (0.1 if m == 3 else 0.99)
            rows.append({"m": m, "norm": norm, "p": 2.0, "same_ranking": 0.0, "same_top1": top, "tau": 0.5, "max_diff": 0.1, "cases": 200})
    kind, text = ev.verdict_identity(rows)
    assert kind == "info"
    assert "in 90 % von 1800 Fällen" in text and "nur noch in 70 bis 90 % der Fälle" in text and "(5 Kriterien)" in text


def test_verdict_cost_uses_the_linear_deciders_and_the_whole_range_of_gaps():
    """Lücke Summe - TOPSIS je Zelle (Güte/linear und Rohwerte/linear, m = 3, 5, 8): 2, 10, 6 und 4, 11, 12 Punkte -> kleinste 2,0, größte 12,0. Der konkave Entscheider
    (Lücke 89 Punkte) zählt nicht mit; die drei Trefferquoten stammen aus der Zelle Rohwerte/linear bei m = 5 (96, 85, 74 %)."""
    cells = {("guete_linear", 3): (0.90, 0.88), ("guete_linear", 5): (0.95, 0.85), ("guete_linear", 8): (0.80, 0.74),
             ("roh_linear", 3): (0.92, 0.88), ("roh_linear", 5): (0.96, 0.85), ("roh_linear", 8): (0.97, 0.85)}
    rows = []
    for (d, m), (s, t) in cells.items():
        rows += [{"decider": d, "m": m, "method": "summe_minmax", "hit": s}, {"decider": d, "m": m, "method": "t_minmax_2", "hit": t},
                 {"decider": d, "m": m, "method": "asf_minmax", "hit": 0.74 if (d, m) == ("roh_linear", 5) else 0.5}]
    rows += [{"decider": "guete_konkav", "m": m, "method": k, "hit": h} for m in (3, 5, 8) for k, h in (("summe_minmax", 0.99), ("t_minmax_2", 0.10), ("asf_minmax", 0.2))]
    kind, text = ev.verdict_cost(rows)
    assert kind == "warning"
    assert "96 %" in text and "85 %" in text and "74 %" in text
    assert "2,0 bis 12,0 Punkte" in text


def test_verdict_weights_reads_the_largest_perturbation():
    """Die Verdikt-Zahlen gehören zur größten Streuung (sigma = 0,4), nicht zur kleinsten (0,1) oder mittleren (0,2); das Sigma steht im Text."""
    srows = []
    for sigma, vals in ((0.4, (0.81, 0.77, 0.85)), (0.1, (0.99, 0.97, 0.95)), (0.2, (0.90, 0.88, 0.92))):
        srows += [{"sigma": sigma, "method": k, "stay": v} for k, v in zip(("summe", "topsis_minmax", "topsis_gleichwertig"), vals)]
    wrows = [{"decider": "guete_cobb", "m": 8, "method": "topsis_gleichwertig", "hit": 0.73}, {"decider": "guete_cobb", "m": 8, "method": "topsis_w", "hit": 0.585}]
    kind, text = ev.verdict_weights(wrows, srows)
    assert kind == "info" and "σ = 0,4" in text
    assert "bei der Summe in 81 %" in text and "bei TOPSIS (p = 2) in 77 %" in text and "gleichwertigen Gewichten in 85 %" in text
    for fremd in ("99 %", "97 %", "95 %", "90 %", "88 %", "92 %"):
        assert fremd not in text
