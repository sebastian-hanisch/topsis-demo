"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import tps_constants as C
import tps_score as sc
from tps_evaluation import COST_LABELS, Analysis, de, p_label

ARROW = {1: "↑", -1: "↓"}
SHORT = {"wahr": "Summe, wahre Gewichte (Min-Max)", "summe": "Summe, gewählte Normierung", "summe_wirksam": "Summe mit wirksamen Gewichten w^p",
         "topsis": "TOPSIS", "asf": "Tschebyschow-Referenzpunkt"}


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def criterion_label(a: Analysis, k: int) -> str:
    inst = a.case.matrix
    return f"K{k + 1} {ARROW[int(inst.directions[k])]} {inst.units[k]}"


def build_scatter(a: Analysis, cx: int, cy: int) -> go.Figure:
    """Alternativen im Min-Max-Raum zweier Kriterien (1 = bester Wert der Stichprobe); Referenzpunkte, Front und die Wahl jedes Verfahrens."""
    mm = sc.normalize(a.case.S, "minmax")
    ref = sc.normalize_points(a.case.S, np.vstack([a.ideal_raw, a.anti_raw]), "minmax")
    names = alt_names(a.case.matrix.n)
    fig = go.Figure()
    for flag, label, color in ((~a.front, "dominiert", C.COLOR_FRONT), (a.front, "nichtdominiert", C.COLOR_LINE)):
        idx = np.flatnonzero(flag)
        fig.add_trace(go.Scatter(x=mm[idx, cx], y=mm[idx, cy], mode="markers+text", text=[names[i] for i in idx], textposition="top center", name=label,
                                 marker=dict(color=color, size=8), textfont=dict(size=9)))
    marks = (("best", "bester nach Nutzen", "star", C.COLOR_TRUE, a.best), ("topsis", "TOPSIS", "diamond", C.COLOR_TOPSIS, a.results["topsis"].top1),
             ("summe", "Summe", "square", C.COLOR_SUM, a.results["summe"].top1), ("asf", "Tschebyschow", "x", C.COLOR_ASF, a.results["asf"].top1))
    for _, label, symbol, color, i in marks:
        fig.add_trace(go.Scatter(x=[mm[i, cx]], y=[mm[i, cy]], mode="markers", name=f"Platz 1: {label}",
                                 marker=dict(symbol=symbol, size=15, color="rgba(0,0,0,0)", line=dict(color=color, width=2.5))))
    fig.add_trace(go.Scatter(x=[ref[0, cx]], y=[ref[0, cy]], mode="markers", name="Idealpunkt", marker=dict(symbol="cross", size=12, color=C.COLOR_TOPSIS)))
    fig.add_trace(go.Scatter(x=[ref[1, cx]], y=[ref[1, cy]], mode="markers", name="Anti-Idealpunkt", marker=dict(symbol="cross", size=12, color=C.COLOR_WARN)))
    pad = 0.08
    xs, ys = np.r_[mm[:, cx], ref[:, cx]], np.r_[mm[:, cy], ref[:, cy]]
    fig.update_xaxes(title=criterion_label(a, cx) + " (Min-Max)", range=[float(xs.min()) - pad, float(xs.max()) + pad])
    fig.update_yaxes(title=criterion_label(a, cy) + " (Min-Max)", range=[float(ys.min()) - pad, float(ys.max()) + pad])
    return _fixed(fig, 420)


def build_weights(a: Analysis) -> go.Figure:
    x = [f"K{k + 1}" for k in range(a.settings.m)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=a.case.w, name="wahre Gewichte des Entscheiders", marker_color=C.COLOR_TRUE))
    if not np.allclose(a.w_in, a.case.w / a.case.w.sum()):
        fig.add_trace(go.Bar(x=x, y=a.w_in, name="in den Abstand eingegeben", marker_color=C.COLOR_SUM))
    fig.add_trace(go.Bar(x=x, y=a.w_eff, name=f"wirksam (w^p, p = {p_label(a.settings.p)})", marker_color=C.COLOR_TOPSIS))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Gewicht", range=[0, 1])
    return _fixed(fig, 320)


def build_table(a: Analysis) -> pd.DataFrame:
    """Alle Alternativen nach dem TOPSIS-Rang."""
    n = a.settings.n
    names = alt_names(n)
    rank = lambda v: np.argsort(np.argsort(-np.asarray(v), kind="stable"), kind="stable") + 1  # noqa: E731
    df = pd.DataFrame({
        "Alternative": names, "d⁺ (zum Ideal)": a.d_plus, "d⁻ (zum Anti-Ideal)": a.d_minus, "Nähe C": a.closeness, "Rang TOPSIS": rank(a.closeness),
        "Summe": a.results["summe"].score, "Rang Summe": rank(a.results["summe"].score), "Rang Entscheider": rank(a.case.utility), "nichtdominiert": np.where(a.front, "ja", "nein"),
    })
    return df.sort_values("Rang TOPSIS").reset_index(drop=True)


def style_table(df: pd.DataFrame):
    return df.style.format({"d⁺ (zum Ideal)": lambda v: de(v, 3), "d⁻ (zum Anti-Ideal)": lambda v: de(v, 3), "Nähe C": lambda v: de(v, 3), "Summe": lambda v: de(v, 3)})


def build_method_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    rows = []
    for k, r in a.results.items():
        rows.append({"Verfahren": SHORT[k], "Platz 1": names[r.top1], "Trifft den Besten": "ja" if r.hit else "nein", "τ gegen Nutzen": de(r.tau, 2),
                     "Nutzenverlust": de(r.loss, 3), "für die Summe erreichbar": "ja" if a.sum_reach[k] else "nein"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Experimente

def build_identity(rows: list[dict], m: int = 5) -> go.Figure:
    """Anteil der Fälle, in denen TOPSIS und die Summe denselben Platz 1 wählen, je Norm p und Normierung."""
    fig = go.Figure()
    norms = list(dict.fromkeys(r["norm"] for r in rows))
    for norm in norms:
        rs = [r for r in rows if r["m"] == m and r["norm"] == norm]
        fig.add_trace(go.Bar(x=[f"p = {p_label(r['p'])}" for r in rs], y=[100 * r["same_top1"] for r in rs], name=C.NORM_SHORT[norm]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="gleicher Platz 1 wie die Summe (%)", range=[0, 102])
    return _fixed(fig, 320)


def identity_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append({"Kriterien m": r["m"], "Normierung": C.NORM_SHORT[r["norm"]], "p": p_label(r["p"]), "gleiche Rangfolge": f"{de(100 * r['same_ranking'], 1)} %",
                    "gleicher Platz 1": f"{de(100 * r['same_top1'], 1)} %", "mittleres τ": de(r["tau"], 2),
                    "größte Abweichung C − Summe": (f"{r['max_diff']:.1e}".replace(".", ",") if r["norm"] == "minmax" else "-")})
    return pd.DataFrame(out)


COST_PLOT = ("summe_minmax", "t_minmax_1", "t_minmax_2", "t_minmax_inf", "t_vektor_2", "asf_minmax")
COST_COLORS = {"summe_minmax": C.COLOR_SUM, "t_minmax_1": "#bbbbbb", "t_minmax_2": C.COLOR_TOPSIS, "t_minmax_inf": "#5b7f95", "t_vektor_2": "#7fb3a8", "asf_minmax": C.COLOR_ASF}


def build_cost(rows: list[dict], m: int = 5) -> go.Figure:
    fig = go.Figure()
    for k in COST_PLOT:
        rs = [r for r in rows if r["m"] == m and r["method"] == k]
        fig.add_trace(go.Bar(x=[C.DECIDER_SHORT[r["decider"]] for r in rs], y=[100 * r["hit"] for r in rs], name=COST_LABELS[k], marker_color=COST_COLORS[k]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    return _fixed(fig, 360)


def cost_table(rows: list[dict]) -> pd.DataFrame:
    deciders = list(dict.fromkeys(r["decider"] for r in rows))
    ms = sorted({r["m"] for r in rows})
    out = []
    for k in COST_LABELS:
        row = {"Verfahren": COST_LABELS[k]}
        for dec in deciders:
            for m in ms:
                r = next(x for x in rows if x["decider"] == dec and x["m"] == m and x["method"] == k)
                row[f"{C.DECIDER_SHORT[dec]}, m = {m}"] = f"{de(100 * r['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def agreement_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Verfahren A": COST_LABELS[r["a"]], "Verfahren B": COST_LABELS[r["b"]], "gleicher Platz 1": f"{de(100 * r['same_top1'], 1)} %",
                          "mittleres τ": de(r["tau"], 2)} for r in rows])


REACH_LABELS = {"summe_lp": "Summe (exakt, LP)", "summe": "Summe (Gewichtsstichprobe)", "topsis_minmax": "TOPSIS p = 2, Min-Max (Stichprobe)",
                "topsis_vektor": "TOPSIS p = 2, Vektor (Stichprobe)", "asf": "Tschebyschow-Referenzpunkt (Stichprobe und Konstruktion)"}


def build_reach(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for k, label in REACH_LABELS.items():
        fig.add_trace(go.Bar(x=[f"m = {r['m']}" for r in rows], y=[100 * r[k] for r in rows], name=label))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="nichtdominierte Alternativen, die bei einer Gewichtung gewinnen (%)", range=[0, 102])
    return _fixed(fig, 360)


def build_stability(rows: list[dict]) -> go.Figure:
    labels = {"summe": "Summe (Min-Max)", "topsis_vektor": "TOPSIS Vektor p = 2", "topsis_minmax": "TOPSIS Min-Max p = 2",
              "topsis_gleichwertig": "TOPSIS Min-Max p = 2, gleichwertige Gewichte", "asf": "Tschebyschow-Referenzpunkt"}
    colors = {"summe": C.COLOR_SUM, "topsis_vektor": "#7fb3a8", "topsis_minmax": C.COLOR_TOPSIS, "topsis_gleichwertig": "#9b6b9b", "asf": C.COLOR_ASF}
    fig = go.Figure()
    for k, label in labels.items():
        rs = [r for r in rows if r["method"] == k]
        fig.add_trace(go.Scatter(x=[r["sigma"] for r in rs], y=[100 * r["stay"] for r in rs], mode="lines+markers", name=label, line=dict(color=colors[k])))
    fig.update_xaxes(title="Streuung σ der Gewichtsstörung (lognormal)", tickvals=sorted({r["sigma"] for r in rows}))
    fig.update_yaxes(title="Platz 1 bleibt (%)", range=[0, 102])
    return _fixed(fig, 340)


def weights_table(rows: list[dict]) -> pd.DataFrame:
    labels = {"summe": "Summe, w", "summe_wirksam": "Summe, w² (normiert)", "topsis_w": "TOPSIS p = 2, w", "topsis_gleichwertig": "TOPSIS p = 2, √w (gleichwertig)"}
    deciders = list(dict.fromkeys(r["decider"] for r in rows))
    ms = sorted({r["m"] for r in rows})
    out = []
    for k, label in labels.items():
        row = {"Gewichte": label}
        for dec in deciders:
            for m in ms:
                r = next(x for x in rows if x["decider"] == dec and x["m"] == m and x["method"] == k)
                row[f"{C.DECIDER_SHORT[dec]}, m = {m}"] = f"{de(100 * r['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def reference_table(rows: list[dict]) -> pd.DataFrame:
    modes = {"stichprobe": "Ideal der Stichprobe", "nenn": "Nenn-Ideal", "anspruch": f"Anspruchsniveau {de(100 * C.DEFAULT_ASPIRATION, 0)} %"}
    out = []
    for dec in dict.fromkeys(r["decider"] for r in rows):
        row = {"Entscheider": C.DECIDER_SHORT[dec]}
        for method, label in (("topsis", "TOPSIS p = 2"), ("asf", "Tschebyschow")):
            for mode, mlabel in modes.items():
                r = next(x for x in rows if x["decider"] == dec and x["reference"] == mode and x["method"] == method)
                row[f"{label}, {mlabel}"] = f"{de(100 * r['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)
