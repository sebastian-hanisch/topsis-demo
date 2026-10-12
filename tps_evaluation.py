"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy (scipy über tps_score), kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tps_constants as C
import tps_score as sc
import tps_topsis as tp
from entscheidungsmatrix_generator import SplitMix64, make_matrix

NOISE_OFFSET = 20011        # Strom der Gewichtsstörungen: SplitMix64(seed + NOISE_OFFSET), unabhängig vom Strom der Matrix
REACH_OFFSET = 777          # Strom der Gewichtsstichprobe der Reichweite


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def p_label(p: float) -> str:
    """Norm p für die Anzeige: ∞, ganze Zahlen ohne Nachkommastelle, sonst mit Komma."""
    if math.isinf(p):
        return "∞"
    return str(int(p)) if float(p).is_integer() else de(p, 1)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    norm: str = C.DEFAULT_NORM
    p: float = C.DEFAULT_P
    reference: str = C.DEFAULT_REFERENCE
    aspiration: float = C.DEFAULT_ASPIRATION
    weights: str = C.DEFAULT_WEIGHTS
    decider: str = C.DEFAULT_DECIDER
    delta: float = C.DEFAULT_DELTA
    seed: int = C.DEFAULT_SEED


@dataclass
class Case:
    """Eine Matrix mit simuliertem Entscheider: vorzeichenbehaftete Rohwerte nach Verschmälerung, wahre Gewichte, Nutzen."""
    matrix: object
    goodness: np.ndarray
    S: np.ndarray
    w: np.ndarray
    utility: np.ndarray


def make_case(n: int, m: int, seed: int, delta: float = 1.0, decider: str = "guete_linear") -> Case:
    scale, form = C.DECIDER_PARTS[decider]
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    g, _, S = sc.build_case(x.goodness, x.directions, m, delta)
    return Case(x, g, S, x.weights, sc.decider_utility(scale, form, g, S, x.weights, m))


def input_weights(w: np.ndarray, mode: str, p: float) -> np.ndarray:
    """Gewichte, die in den Abstand eingehen: die des Entscheiders oder die gleichwertigen w^(1/p) (so wirken sie im Abstand wie w in der Summe)."""
    if mode == "wahr":
        return np.asarray(w, dtype=float)
    if mode == "gleichwertig":
        return tp.equivalent_weights(w, p)
    raise ValueError(f"unbekannte Gewichtsvorgabe: {mode}")


def reference_points(S: np.ndarray, norm: str, mode: str, aspiration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ideal- und Anti-Idealpunkt: (normiert Ideal, normiert Anti, roh Ideal, roh Anti)."""
    lo, hi = sc.nominal_bounds(S.shape[1])
    best, worst = tp.reference_raw(S, mode, lo, hi, aspiration)
    pts = sc.normalize_points(S, np.vstack([best, worst]), norm)
    return pts[0], pts[1], best, worst


@dataclass
class Result:
    score: np.ndarray          # höher ist besser
    top1: int
    hit: bool
    tau: float
    loss: float


def _result(case: Case, score: np.ndarray, best: int) -> Result:
    t = sc.top1(score)
    return Result(score, t, t == best, sc.kendall_tau_b(score, case.utility), sc.utility_loss(case.utility, t))


@dataclass
class Analysis:
    settings: Settings
    case: Case
    N: np.ndarray
    w_in: np.ndarray           # in den Abstand eingehende Gewichte
    w_eff: np.ndarray          # wirksame Gewichte (w^p normiert)
    ideal: np.ndarray          # Referenzpunkte im normierten Raum
    anti: np.ndarray
    ideal_raw: np.ndarray      # dieselben Punkte im vorzeichenbehafteten Rohraum
    anti_raw: np.ndarray
    d_plus: np.ndarray
    d_minus: np.ndarray
    closeness: np.ndarray
    best: int
    results: dict              # "wahr", "summe", "summe_wirksam", "topsis", "asf"
    front: np.ndarray          # Maske der nichtdominierten Alternativen
    sum_reach: dict            # je Verfahren: kann die Summe (bei irgendeiner Gewichtung) die gewählte Alternative zum Sieger machen? (LP)


def analyse(settings: Settings) -> Analysis:
    s = settings
    case = make_case(s.n, s.m, s.seed, s.delta, s.decider)
    N = sc.normalize(case.S, s.norm)
    w_in = input_weights(case.w, s.weights, s.p)
    ideal, anti, ideal_raw, anti_raw = reference_points(case.S, s.norm, s.reference, s.aspiration)
    dp, dm, c = tp.closeness(N, w_in, s.p, ideal, anti)
    best = sc.top1(case.utility)
    w_eff = tp.effective_weights(w_in, s.p)
    asf = -tp.asf_distance(N, w_in, ideal, C.RHO)
    results = {
        "wahr": _result(case, sc.weighted_score(sc.normalize(case.S, "minmax"), case.w), best),
        "summe": _result(case, sc.weighted_score(N, w_in), best),
        "summe_wirksam": _result(case, sc.weighted_score(N, w_eff), best),
        "topsis": _result(case, c, best),
        "asf": _result(case, asf, best),
    }
    front = sc.front_mask(case.goodness)
    reach = {k: bool(front[r.top1] and sc.is_possible_winner(N, r.top1)) for k, r in results.items()}
    return Analysis(s, case, N, w_in, w_eff, ideal, anti, ideal_raw, anti_raw, dp, dm, c, best, results, front, reach)


def _rank_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")))


def draw_weights(m: int, count: int, rng: SplitMix64) -> np.ndarray:
    """count Gewichtsvektoren, gleichverteilt über dem Simplex (normierte Exponentialverteilte)."""
    e = -np.log1p(-rng.uniform(count * m)).reshape(count, m)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- Experiment 1: p = 1 ist die gewichtete Summe

def experiment_identity(ms=C.EXP_M, norms=C.EXP_IDENTITY_NORMS, ps=(1.0, 2.0, math.inf), n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """TOPSIS gegen die gewichtete Summe mit derselben Normierung und denselben (wahren) Gewichten: identische Vollrangfolgen, gleicher Platz 1, mittleres tau,
    größte Abweichung C - Summe."""
    rows = []
    for m in ms:
        cases = [make_case(n, m, s) for s in seeds]
        for norm in norms:
            for p in ps:
                same_rank, same_top, taus, dmax = [], [], [], 0.0
                for c in cases:
                    N = sc.normalize(c.S, norm)
                    ws = sc.weighted_score(N, c.w)
                    cl = tp.closeness(N, c.w, p)[2]
                    same_rank.append(_rank_equal(cl, ws))
                    same_top.append(sc.top1(cl) == sc.top1(ws))
                    taus.append(sc.kendall_tau_b(cl, ws))
                    dmax = max(dmax, float(np.abs(cl - ws).max()))
                rows.append({"m": m, "norm": norm, "p": p, "same_ranking": float(np.mean(same_rank)), "same_top1": float(np.mean(same_top)),
                             "tau": float(np.mean(taus)), "max_diff": dmax, "cases": len(cases)})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: was der Abstand kostet

COST_METHODS = ("summe_minmax", "summe_vektor", "t_vektor_1", "t_vektor_2", "t_vektor_inf", "t_minmax_1", "t_minmax_2", "t_minmax_inf", "t_z_2", "t_perzentil_2",
                "t_roh_2", "asf_minmax")
COST_LABELS = {
    "summe_minmax": "Summe, Min-Max", "summe_vektor": "Summe, Vektor", "t_vektor_1": "TOPSIS Vektor p = 1", "t_vektor_2": "TOPSIS Vektor p = 2",
    "t_vektor_inf": "TOPSIS Vektor p = ∞", "t_minmax_1": "TOPSIS Min-Max p = 1", "t_minmax_2": "TOPSIS Min-Max p = 2", "t_minmax_inf": "TOPSIS Min-Max p = ∞",
    "t_z_2": "TOPSIS z-Wert p = 2", "t_perzentil_2": "TOPSIS Perzentil p = 2", "t_roh_2": "TOPSIS Rohwerte p = 2", "asf_minmax": "Tschebyschow-Referenzpunkt, Min-Max",
}


def cost_scores(S: np.ndarray, w: np.ndarray) -> dict:
    """Alle Verfahren der Kostenmessung auf derselben Matrix (höher ist besser)."""
    nm, nv = sc.normalize(S, "minmax"), sc.normalize(S, "vektor")
    out = {"summe_minmax": nm @ w, "summe_vektor": nv @ w}
    for key, N, p in (("t_vektor_1", nv, 1.0), ("t_vektor_2", nv, 2.0), ("t_vektor_inf", nv, math.inf), ("t_minmax_1", nm, 1.0), ("t_minmax_2", nm, 2.0),
                      ("t_minmax_inf", nm, math.inf)):
        out[key] = tp.closeness(N, w, p)[2]
    out["t_z_2"] = tp.closeness(sc.normalize(S, "z"), w, 2.0)[2]
    out["t_perzentil_2"] = tp.closeness(sc.normalize(S, "perzentil"), w, 2.0)[2]
    out["t_roh_2"] = tp.closeness(sc.normalize(S, "roh"), w, 2.0)[2]
    out["asf_minmax"] = -tp.asf_distance(nm, w, None, C.RHO)
    return out


def experiment_cost(ms=C.EXP_M, deciders=C.DECIDERS, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for dec in deciders:
        for m in ms:
            hits = {k: [] for k in COST_METHODS}
            taus = {k: [] for k in COST_METHODS}
            for s in seeds:
                c = make_case(n, m, s, 1.0, dec)
                best = sc.top1(c.utility)
                v = cost_scores(c.S, c.w)
                for k in COST_METHODS:
                    hits[k].append(sc.top1(v[k]) == best)
                    taus[k].append(sc.kendall_tau_b(v[k], c.utility))
            for k in COST_METHODS:
                h = float(np.mean(hits[k]))
                rows.append({"decider": dec, "m": m, "method": k, "hit": h, "tau": float(np.mean(taus[k])), "se": float(np.sqrt(h * (1 - h) / len(seeds)))})
    return rows


def experiment_reference(m=C.EXP_STABILITY_M, deciders=C.DECIDERS, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Woher der Referenzpunkt kommt: Idealpunkt der Stichprobe, Nenn-Idealpunkt oder Anspruchsniveau (Standard 0,8). Platz 1 getroffen für TOPSIS (p = 2) und
    den Tschebyschow-Referenzpunkt, Min-Max-Normierung, wahre Gewichte."""
    rows = []
    modes = (("stichprobe", 1.0), ("nenn", 1.0), ("anspruch", C.DEFAULT_ASPIRATION))
    for dec in deciders:
        hits = {(mode, k): [] for mode, _ in modes for k in ("topsis", "asf")}
        for s in seeds:
            c = make_case(n, m, s, 1.0, dec)
            best = sc.top1(c.utility)
            N = sc.normalize(c.S, "minmax")
            for mode, a in modes:
                ideal, anti, _, _ = reference_points(c.S, "minmax", mode, a)
                hits[(mode, "topsis")].append(sc.top1(tp.closeness(N, c.w, 2.0, ideal, anti)[2]) == best)
                hits[(mode, "asf")].append(sc.top1(-tp.asf_distance(N, c.w, ideal, C.RHO)) == best)
        for (mode, k), v in hits.items():
            rows.append({"decider": dec, "m": m, "reference": mode, "method": k, "hit": float(np.mean(v))})
    return rows


def agreement(m=C.EXP_STABILITY_M, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Übereinstimmung von Platz 1 (Güte/linear): Norm und Normierung ändern die Entscheidung."""
    pairs = (("t_vektor_2", "summe_minmax"), ("t_vektor_2", "t_minmax_2"), ("t_minmax_2", "t_minmax_1"), ("t_minmax_2", "t_minmax_inf"), ("t_perzentil_2", "t_minmax_2"),
             ("t_roh_2", "t_minmax_2"))
    acc = {pr: ([], []) for pr in pairs}
    for s in seeds:
        c = make_case(n, m, s)
        v = cost_scores(c.S, c.w)
        for a, b in pairs:
            acc[(a, b)][0].append(sc.top1(v[a]) == sc.top1(v[b]))
            acc[(a, b)][1].append(sc.kendall_tau_b(v[a], v[b]))
    return [{"a": a, "b": b, "same_top1": float(np.mean(x)), "tau": float(np.mean(y))} for (a, b), (x, y) in acc.items()]


# ----------------------------------------------------------------------------- Experiment 3: Reichweite

def possible_winners_by_sampling(N: np.ndarray, W: np.ndarray, kind: str, p: float = 2.0) -> set:
    """Alternativen, die für mindestens eine der Gewichtungen W (K x m) Platz 1 werden: Summe, topsis (Nähe in der Norm p) oder asf (Tschebyschow zum Idealpunkt)."""
    N = np.asarray(N, dtype=float)
    if kind == "summe":
        return {int(i) for i in np.argmax(N @ W.T, axis=0)}
    if kind == "asf":
        D = W[None, :, :] * (N.max(axis=0) - N)[:, None, :]
        return {int(i) for i in np.argmin(D.max(axis=2) + C.RHO * D.sum(axis=2), axis=0)}
    if kind == "topsis":
        V = N[:, None, :] * W[None, :, :]
        hi, lo = V.max(axis=0), V.min(axis=0)
        dp, dm = tp.p_norm(V - hi, p), tp.p_norm(V - lo, p)
        return {int(i) for i in np.argmax(dm / (dp + dm), axis=0)}
    raise ValueError(f"unbekanntes Verfahren: {kind}")


def experiment_reach(ms=C.EXP_REACH_M, n=C.EXP_N, seeds=C.REACH_SEEDS, weights=C.EXP_REACH_WEIGHTS) -> list[dict]:
    """Anteil der nichtdominierten Alternativen, die bei IRGENDEINER Gewichtung Platz 1 werden: die Summe exakt (LP) und per Stichprobe, TOPSIS (p = 2, Min-Max und
    Vektor) per Stichprobe, der Tschebyschow-Referenzpunkt per Stichprobe und zusätzlich über die Konstruktion w_j ~ 1 / (Ideal_j - Wert_j). Stichprobe und Konstruktion
    sind untere Schranken: Sie unterschätzen die Reichweite; verglichen werden die Verfahren auf denselben Gewichten."""
    rows = []
    for m in ms:
        W = np.vstack([draw_weights(m, weights, SplitMix64(REACH_OFFSET + m)), np.eye(m)])
        tot = {k: 0 for k in ("front", "summe_lp", "summe", "topsis_minmax", "topsis_vektor", "asf")}
        for s in seeds:
            c = make_case(n, m, s)
            nm, nv = sc.normalize(c.S, "minmax"), sc.normalize(c.S, "vektor")
            front = set(np.flatnonzero(sc.front_mask(c.goodness)).tolist())
            tot["front"] += len(front)
            tot["summe_lp"] += sum(sc.is_possible_winner(nm, i) for i in front)
            tot["summe"] += len(front & possible_winners_by_sampling(nm, W, "summe"))
            tot["topsis_minmax"] += len(front & possible_winners_by_sampling(nm, W, "topsis", 2.0))
            tot["topsis_vektor"] += len(front & possible_winners_by_sampling(nv, W, "topsis", 2.0))
            asf_set = possible_winners_by_sampling(nm, W, "asf") | {i for i in front if tp.asf_reachable(nm, i, C.RHO)}
            tot["asf"] += len(front & asf_set)
        f = tot["front"]
        rows.append({"m": m, "front": f / len(seeds), "matrices": len(seeds), "weights": len(W),
                     **{k: tot[k] / f for k in ("summe_lp", "summe", "topsis_minmax", "topsis_vektor", "asf")}})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Gewichte und Stabilität

def experiment_weights(ms=(5, 8), deciders=C.DECIDERS, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Euklid-TOPSIS (Min-Max) mit den Gewichten w und mit den gleichwertigen w^(1/p), daneben die Summe mit w und mit den wirksamen Gewichten w^p."""
    rows = []
    for dec in deciders:
        for m in ms:
            hits = {k: [] for k in ("summe", "summe_wirksam", "topsis_w", "topsis_gleichwertig")}
            for s in seeds:
                c = make_case(n, m, s, 1.0, dec)
                best = sc.top1(c.utility)
                N = sc.normalize(c.S, "minmax")
                hits["summe"].append(sc.top1(sc.weighted_score(N, c.w)) == best)
                hits["summe_wirksam"].append(sc.top1(sc.weighted_score(N, tp.effective_weights(c.w, 2.0))) == best)
                hits["topsis_w"].append(sc.top1(tp.closeness(N, c.w, 2.0)[2]) == best)
                hits["topsis_gleichwertig"].append(sc.top1(tp.closeness(N, tp.equivalent_weights(c.w, 2.0), 2.0)[2]) == best)
            for k, v in hits.items():
                rows.append({"decider": dec, "m": m, "method": k, "hit": float(np.mean(v))})
    return rows


STABILITY_METHODS = ("summe", "topsis_vektor", "topsis_minmax", "topsis_gleichwertig", "asf")


def _stability_tops(S: np.ndarray, w: np.ndarray) -> dict:
    nm, nv = sc.normalize(S, "minmax"), sc.normalize(S, "vektor")
    return {"summe": sc.top1(nm @ w), "topsis_vektor": sc.top1(tp.closeness(nv, w, 2.0)[2]), "topsis_minmax": sc.top1(tp.closeness(nm, w, 2.0)[2]),
            "topsis_gleichwertig": sc.top1(tp.closeness(nm, tp.equivalent_weights(w, 2.0), 2.0)[2]), "asf": sc.top1(-tp.asf_distance(nm, w, None, C.RHO))}


def experiment_stability(sigmas=C.EXP_STABILITY_SIGMAS, m=C.EXP_STABILITY_M, n=C.EXP_N, seeds=C.SWEEP_SEEDS, draws=C.EXP_STABILITY_DRAWS) -> list[dict]:
    """Bleibt Platz 1, wenn die Gewichte lognormal gestört werden (w' proportional w exp(sigma eps))? Entscheider Güte/linear; Anteil über Matrizen und Störungen."""
    rows = []
    for sg in sigmas:
        stay = {k: [] for k in STABILITY_METHODS}
        for s in seeds:
            c = make_case(n, m, s)
            base = _stability_tops(c.S, c.w)
            rng = SplitMix64(s + NOISE_OFFSET)
            for _ in range(draws):
                wp = c.w * np.exp(sg * rng.normal(m))
                wp = wp / wp.sum()
                t = _stability_tops(c.S, wp)
                for k in STABILITY_METHODS:
                    stay[k].append(t[k] == base[k])
        for k in STABILITY_METHODS:
            rows.append({"sigma": sg, "method": k, "stay": float(np.mean(stay[k]))})
    return rows


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_identity(rows: list[dict]) -> tuple[str, str]:
    p1 = [r for r in rows if r["p"] == 1.0]
    p2 = [r for r in rows if r["p"] == 2.0 and r["m"] == C.EXP_STABILITY_M]
    total = sum(r["cases"] for r in p1)
    lo, hi = min(r["same_top1"] for r in p2), max(r["same_top1"] for r in p2)
    return "info", (f"Mit p = 1 ist die Nähe die gewichtete Summe: identische Rangfolgen in {de(100 * min(r['same_ranking'] for r in p1), 0)} % von {total} Fällen "
                    f"(Normierungen Min-Max, Vektor, z-Wert). Mit p = 2 stimmt Platz 1 nur noch in {de(100 * lo, 0)} bis {de(100 * hi, 0)} % der Fälle mit der Summe überein "
                    f"({C.EXP_STABILITY_M} Kriterien).")


def verdict_cost(rows: list[dict]) -> tuple[str, str]:
    m = C.EXP_STABILITY_M
    s = _row(rows, decider="roh_linear", m=m, method="summe_minmax")["hit"]
    t = _row(rows, decider="roh_linear", m=m, method="t_minmax_2")["hit"]
    a = _row(rows, decider="roh_linear", m=m, method="asf_minmax")["hit"]
    gaps = [_row(rows, decider=d, m=mm, method="summe_minmax")["hit"] - _row(rows, decider=d, m=mm, method="t_minmax_2")["hit"]
            for d in ("guete_linear", "roh_linear") for mm in sorted({r["m"] for r in rows})]
    return "warning", (f"Bei einem Entscheider, der linear in den Rohwerten bewertet ({m} Kriterien, Min-Max), trifft die Summe Platz 1 in {de(100 * s, 0)} % der Fälle, "
                       f"TOPSIS mit p = 2 in {de(100 * t, 0)} % und der Tschebyschow-Referenzpunkt in {de(100 * a, 0)} %. Über alle linearen Zellen liegt TOPSIS "
                       f"{de(100 * min(gaps), 1)} bis {de(100 * max(gaps), 1)} Punkte hinter der Summe.")


def verdict_reference(rows: list[dict]) -> tuple[str, str]:
    t = [r["hit"] for r in rows if r["reference"] == "anspruch" and r["method"] == "topsis"]
    a = [r["hit"] for r in rows if r["reference"] == "anspruch" and r["method"] == "asf"]
    s_ = [r["hit"] for r in rows if r["reference"] == "stichprobe" and r["method"] == "topsis"]
    return "warning", (f"Liegt das Anspruchsniveau bei {de(100 * C.DEFAULT_ASPIRATION, 0)} % der Strecke vom schlechtesten zum besten Wert, trifft TOPSIS Platz 1 nur in "
                       f"{de(100 * min(t), 0)} bis {de(100 * max(t), 0)} % der Fälle (mit dem Idealpunkt der Stichprobe {de(100 * min(s_), 0)} bis {de(100 * max(s_), 0)} %): "
                       f"Der zweiseitige Abstand bestraft Alternativen, die das Ziel übertreffen. Der einseitige Tschebyschow-Referenzpunkt verliert weniger "
                       f"({de(100 * min(a), 0)} bis {de(100 * max(a), 0)} %). Idealpunkt der Stichprobe und Nenn-Idealpunkt unterscheiden sich bei {C.EXP_N} Alternativen nicht belastbar.")


def verdict_reach(rows: list[dict]) -> tuple[str, str]:
    r = _row(rows, m=C.EXP_STABILITY_M)
    return "info", (f"Bei {r['m']} Kriterien kann die Summe {de(100 * r['summe_lp'], 0)} % der nichtdominierten Alternativen zum Sieger machen (LP; auf der Gewichtsstichprobe "
                    f"{de(100 * r['summe'], 0)} %), TOPSIS mit p = 2 {de(100 * r['topsis_minmax'], 0)} % (Stichprobe, eine untere Schranke) und der Tschebyschow-Referenzpunkt "
                    f"{de(100 * r['asf'], 0)} % (ebenfalls untere Schranke aus Stichprobe und Konstruktion). "
                    "Mehr Reichweite ist nicht bessere Treffer: siehe Experiment 2.")


def verdict_weights(weights_rows: list[dict], stab_rows: list[dict]) -> tuple[str, str]:
    top = max(r["sigma"] for r in stab_rows)
    s = _row(stab_rows, sigma=top, method="summe")["stay"]
    t = _row(stab_rows, sigma=top, method="topsis_minmax")["stay"]
    g = _row(stab_rows, sigma=top, method="topsis_gleichwertig")["stay"]
    c = _row(weights_rows, decider="guete_cobb", m=8, method="topsis_gleichwertig")["hit"]
    c0 = _row(weights_rows, decider="guete_cobb", m=8, method="topsis_w")["hit"]
    return "info", (f"Bei einer Gewichtsstörung von σ = {de(top, 1)} bleibt Platz 1 bei der Summe in {de(100 * s, 0)} % der Fälle, bei TOPSIS (p = 2) in {de(100 * t, 0)} %, "
                    f"mit gleichwertigen Gewichten in {de(100 * g, 0)} %: Die Empfindlichkeit liegt an der Gewichtssemantik. Gleichwertige Gewichte helfen nur bei konkavem Nutzen "
                    f"(Cobb-Douglas, 8 Kriterien: {de(100 * c, 1)} gegen {de(100 * c0, 1)} %).")
