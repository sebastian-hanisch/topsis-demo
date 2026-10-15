"""TOPSIS und Referenzpunkt: der Abstand zum Idealpunkt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Linie Mehrkriterien-Ranking (Kind der gewichteten Summe). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seine Gewichte und seinen
Nutzen. TOPSIS (Abstand zum Ideal- und Anti-Idealpunkt in der Norm p) und die Erfüllungsskalarisierung (Tschebyschow-Abstand zu einem Referenzpunkt) werden an der
Entscheidung gemessen. Kern: tps_topsis.py, tps_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import tps_constants as C
from tps_evaluation import (
    Settings,
    agreement,
    analyse,
    de,
    experiment_cost,
    experiment_identity,
    experiment_reach,
    experiment_reference,
    experiment_stability,
    experiment_weights,
    p_label,
    verdict_cost,
    verdict_identity,
    verdict_reach,
    verdict_reference,
    verdict_weights,
)
from tps_pdf_export import build_pdf
from tps_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from tps_visualization import (
    SHORT,
    agreement_table,
    alt_names,
    build_cost,
    build_identity,
    build_method_table,
    build_reach,
    build_scatter,
    build_stability,
    build_table,
    build_weights,
    cost_table,
    criterion_label,
    identity_table,
    reference_table,
    style_table,
    weights_table,
)

st.set_page_config(page_title="TOPSIS und Referenzpunkt – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    weights = experiment_weights()
    stability = experiment_stability()
    return {
        "identity": experiment_identity(),
        "cost": experiment_cost(),
        "agreement": agreement(),
        "reference": experiment_reference(),
        "reach": experiment_reach(),
        "weights": weights,
        "stability": stability,
    }


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎯 TOPSIS und Referenzpunkt: der Abstand zum Idealpunkt")
st.markdown(
    """
**TOPSIS ist eine gewichtete Summe, deren Abstandsnorm die Gewichte umdeutet.** Dies ist das fünfte Stück der Linie **Mehrkriterien-Ranking**: Statt Punkte zu addieren, misst
TOPSIS für jede Alternative den **Abstand zum Idealpunkt** (der beste Wert je Kriterium) und zum **Anti-Idealpunkt** und ordnet nach der Nähe. Die Demo stellt dem die
**Summe** und den **Tschebyschow-Referenzpunkt** gegenüber und misst jedes Verfahren an der **Entscheidung** eines simulierten Entscheiders, der seinen Nutzen und seine Gewichte
kennt. Die Norm p des Abstands ist der Regler, an dem sich alles entscheidet.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert TOPSIS", expanded=True):
    st.markdown(
        """
1. **Normieren und gewichten:** Jede Spalte wird normiert (Hwang und Yoon: durch die euklidische Länge der Spalte) und mit dem Gewicht des Kriteriums multipliziert.
2. **Referenzpunkte:** Der **Idealpunkt** hat den besten, der **Anti-Idealpunkt** den schlechtesten Wert je Kriterium (aus der Stichprobe, aus der festen Nenn-Spannweite oder als
   Anspruchsniveau dazwischen).
3. **Abstand in der Norm p:** d⁺ zum Ideal, d⁻ zum Anti-Ideal; die **Nähe** C = d⁻ / (d⁺ + d⁻) ordnet. p = 1 (Manhattan) ist eine Summe, p = 2 (Euklid) der Standard,
   p = ∞ (Tschebyschow) bewertet nur die größte gewichtete Abweichung.
4. **Tschebyschow-Referenzpunkt (Wierzbicki):** nur der Abstand zum Idealpunkt, in der Norm ∞ und mit einem kleinen Aufschlag ρ auf die Summe der Abweichungen.
5. **Gemessen wird an der Entscheidung:** Platz 1 gegen den Besten des Entscheiders, Kendalls τ der Rangfolge, Nutzenverlust. Der Entscheider ist ein Orakel mit drei
   Nutzenformen (linear, konkav, Cobb-Douglas in der Güte) oder linear in den Rohwerten.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen zur Wahl stehen.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    delta = st.selectbox("Streuung des ersten Kriteriums", C.DELTA_CHOICES, key="delta_select", format_func=lambda d: "unverändert" if d == 1.0 else f"{de(100 * d, 0)} % der Streuung",
                         help="Wie stark sich die Alternativen im ersten Kriterium unterscheiden; der Entscheider selbst ändert sich dadurch nicht.")
    decider = st.selectbox("Entscheider (simuliert)", C.DECIDERS, key="decider_select", format_func=lambda k: C.DECIDER_LABELS[k],
                           help="Welchen verborgenen Nutzen der Entscheider hat. Konkav und Cobb-Douglas bevorzugen ausgewogene Alternativen.")
    st.markdown("**Abstand**")
    norm = st.selectbox("Normierung", C.NORMS, key="norm_select", format_func=lambda k: C.NORM_LABELS[k],
                        help="Vektor ist die Normierung von Hwang und Yoon. Ohne Normierung entscheidet die Einheit des Kriteriums (Euro gegen Punkte).")
    p = st.selectbox("Norm p des Abstands", C.P_CHOICES, key="p_select", format_func=lambda v: "∞ (Tschebyschow)" if v == float("inf") else p_label(v),
                     help="p = 1: Summe der Abweichungen, die Nähe C ist dann die gewichtete Summe. p = 2: euklidisch. p = ∞: größte gewichtete Abweichung.")
    reference = st.selectbox("Referenzpunkt", C.REFERENCES, key="reference_select", format_func=lambda k: C.REFERENCE_LABELS[k],
                             help="Woher Ideal- und Anti-Idealpunkt kommen. Der Idealpunkt der Stichprobe ändert sich mit jeder hinzugefügten Alternative.")
    aspiration = st.selectbox("Anspruchsniveau", C.ASPIRATION_CHOICES, key="aspiration_select", format_func=lambda v: f"{de(100 * v, 0)} % der Strecke",
                              disabled=reference != "anspruch", help="Nur beim Referenzpunkt „Anspruchsniveau“: Anteil der Strecke vom Anti-Ideal zum Ideal der Stichprobe. TOPSIS misst den Abstand zweiseitig und bestraft damit "
                                   "Alternativen oberhalb des Anspruchs; der Tschebyschow-Referenzpunkt ist einseitig.")
    weights = st.selectbox("Gewichte im Abstand", C.WEIGHT_MODES, key="weights_select", format_func=lambda k: C.WEIGHT_LABELS[k],
                           help="Im Abstand der Norm p geht das Gewicht mit w^p ein. Gleichwertige Gewichte w^(1/p) gleichen das aus, damit sie wie in der Summe wirken.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "norm_select": norm, "p_select": float(p), "reference_select": reference, "aspiration_select": float(aspiration),
                   "weights_select": weights, "decider_select": decider, "delta_select": float(delta), "seed_input": int(seed)})

settings = Settings(int(n), int(m), norm, float(p), reference, float(aspiration), weights, decider, float(delta), int(seed))
a = _analysis(settings)
names = alt_names(a.case.matrix.n)
R = a.results

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Platz 1: TOPSIS", names[R["topsis"].top1], delta=f"bester nach Nutzen: {names[a.best]}", delta_color="off",
          help="Platz 1 nach der Nähe C mit der gewählten Normierung, Norm p und dem gewählten Referenzpunkt.")
c2.metric("Platz 1: Summe", names[R["summe"].top1], delta=f"τ gegen Nutzen: {de(R['summe'].tau, 2)} (TOPSIS {de(R['topsis'].tau, 2)})", delta_color="off",
          help="Gewichtete Summe mit derselben Normierung und denselben Gewichten; τ ist Kendalls τ-b der ganzen Rangfolge gegen den verborgenen Nutzen.")
c3.metric("Platz 1: Tschebyschow", names[R["asf"].top1], delta=f"für die Summe erreichbar: {'ja' if a.sum_reach['asf'] else 'nein'}", delta_color="off",
          help="Erfüllungsskalarisierung zum Referenzpunkt (nur Abstand zum Ideal, Norm ∞, Aufschlag ρ = 0,01). „Erreichbar“: Es gibt Gewichte, bei denen die Summe diese "
               "Alternative auf Platz 1 setzt (LP).")
c4.metric("Wirksame Gewichte", f"w^{p_label(float(p))}" if p != float("inf") else "w", delta=f"größtes Gewicht: {de(float(a.case.w.max()), 2)} → {de(float(a.w_eff.max()), 2)}",
          delta_color="off", help="In der Summe der p-ten Potenzen geht jedes Gewicht mit w^p ein (auf Summe 1 gebracht). Bei p = ∞ gibt es kein Summen-Äquivalent.")

left, right = st.columns([3, 2])
with left:
    kx, ky = st.columns(2)
    cx = kx.selectbox("Achse x", list(range(settings.m)), index=0, format_func=lambda k: criterion_label(a, k), key="x_select")
    cy = ky.selectbox("Achse y", list(range(settings.m)), index=1, format_func=lambda k: criterion_label(a, k), key="y_select")
    st.plotly_chart(build_scatter(a, int(cx), int(cy)), width="stretch", key="scatter_chart")
with right:
    st.plotly_chart(build_weights(a), width="stretch", key="weights_chart")

tab_t, tab_m = st.tabs(["Alle Alternativen nach TOPSIS-Rang", "Verfahren im Vergleich"])
with tab_t:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch")
    st.caption("d⁺ und d⁻ sind gewichtete Abstände im normierten Raum; „Summe“ ist die gewichtete Summe mit derselben Normierung und denselben Gewichten.")
with tab_m:
    st.dataframe(build_method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, derselbe Entscheider. „Summe, wahre Gewichte (Min-Max)“ ist die Vergleichsgröße der Vorgänger; „wirksame Gewichte“ ist die Summe mit w^p.")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="topsis.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist der Abstand zum Idealpunkt etwas anderes als eine Summe?")
same_rank = bool((a.closeness.argsort(kind="stable") == R["summe"].score.argsort(kind="stable")).all())
if float(p) == 1.0 and reference == "stichprobe":
    if same_rank:
        st.success("Mit p = 1 ist die Nähe die gewichtete Summe: TOPSIS und Summe ordnen alle Alternativen gleich (bei Min-Max sogar mit demselben Wert).")
    else:
        st.warning("Mit p = 1 und diesem Referenzpunkt ordnen TOPSIS und Summe verschieden; die Gleichheit gilt für Ideal- und Anti-Idealpunkt der Stichprobe.")
elif R["topsis"].top1 == R["summe"].top1:
    st.info(f"TOPSIS und Summe wählen dieselbe Alternative ({names[R['topsis'].top1]}); die Rangfolge " + ("ist gleich." if same_rank else "unterscheidet sich weiter hinten."))
else:
    st.warning(f"TOPSIS wählt {names[R['topsis'].top1]}, die Summe {names[R['summe'].top1]}; der Beste des Entscheiders ist {names[a.best]} "
               f"({'TOPSIS trifft' if R['topsis'].hit else 'Summe trifft' if R['summe'].hit else 'keines trifft'}).")
st.caption(f"Norm p = {p_label(float(p))}: In der Summe der p-ten Potenzen geht jedes Gewicht mit w^p ein; das größte Gewicht steigt von {de(float(a.case.w.max()), 2)} auf "
           f"{de(float(a.w_eff.max()), 2)}, wenn die Gewichte unverändert eingegeben werden.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), n = 20, wahre Gewichte, unabhängig von den Reglern (Experiment 3: 60 Matrizen). Die Experimente rechnen zusammen rund 20 Sekunden "
           "und starten erst auf Klick.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: p = 1 ist die gewichtete Summe")
    st.caption("TOPSIS mit denselben Gewichten und derselben Normierung wie die Summe; Entscheider Güte/linear. Anteil gleicher Platz 1 für 5 Kriterien, darunter die vollständige Tabelle.")
    st.plotly_chart(build_identity(exp["identity"], 5), width="stretch", key="identity_chart")
    verdict(verdict_identity(exp["identity"]))
    st.dataframe(identity_table(exp["identity"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 2: Was der Abstand kostet")
    st.caption("Platz 1 getroffen, 5 Kriterien, wahre Gewichte, vier Entscheider. Standardfehler 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind nicht belastbar.")
    st.plotly_chart(build_cost(exp["cost"], 5), width="stretch", key="cost_chart")
    verdict(verdict_cost(exp["cost"]))
    st.dataframe(cost_table(exp["cost"]), hide_index=True, width="stretch")
    st.markdown("**Wie viel ändern Norm und Normierung an Platz 1?** (5 Kriterien, Entscheider Güte/linear)")
    st.dataframe(agreement_table(exp["agreement"]), hide_index=True, width="stretch")
    st.markdown("**Woher kommt der Referenzpunkt?** (5 Kriterien, Min-Max, p = 2, wahre Gewichte)")
    st.dataframe(reference_table(exp["reference"]), hide_index=True, width="stretch")
    verdict(verdict_reference(exp["reference"]))

    st.markdown("### Experiment 3: Reichweite")
    st.caption("Anteil der nichtdominierten Alternativen, die bei IRGENDEINER Gewichtung auf Platz 1 kommen können; 60 Matrizen, 10 000 Gewichte je Matrix. Stichprobe und Konstruktion "
               "sind untere Schranken.")
    st.plotly_chart(build_reach(exp["reach"]), width="stretch", key="reach_chart")
    verdict(verdict_reach(exp["reach"]))

    st.markdown("### Experiment 4: Gewichte und Stabilität")
    st.caption("Links: Euklid-TOPSIS (Min-Max) mit den Gewichten w und mit den gleichwertigen √w, daneben die Summe mit w und mit w². Rechts: Bleibt Platz 1, wenn die Gewichte "
               "lognormal gestört werden (5 Kriterien, 10 Störungen je Matrix)?")
    wl, wr = st.columns(2)
    with wl:
        st.dataframe(weights_table(exp["weights"]), hide_index=True, width="stretch")
    with wr:
        st.plotly_chart(build_stability(exp["stability"]), width="stretch", key="stability_chart")
    verdict(verdict_weights(exp["weights"], exp["stability"]))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Entscheider ist ein Orakel** (linear, konkav oder Cobb-Douglas in der Güte, oder linear in den Rohwerten) | Wie Menschen Abstände zum Ideal wahrnehmen, ist nicht gemessen. Wer Ausgewogenheit als Wert an sich will, hat ein anderes Orakel; die Cobb-Douglas-Zelle ist dafür nur ein Anhaltspunkt. | – |
| **Der Idealpunkt kommt aus der Stichprobe** | Er ändert sich mit jeder hinzugefügten oder entfernten Alternative. Bei 20 Alternativen ändert das die Trefferquote nicht messbar; die Rangumkehr selbst ist hier nicht gemessen. | Rangumkehr |
| **Die Gewichte sind bekannt** | In der Demo sind es die wahren Gewichte des Entscheiders; wie man sie erhebt, steht in den Vorgängern. | Gewichte erheben, AHP |
| **Ein Gewichtsvektor, ein Punkt** | Wie sich die Wahl über alle verträglichen Gewichte verteilt, zeigt die Reichweite nur grob (Ja oder Nein je Alternative). | Unsichere Gewichte |
| **Abstand zum Referenzpunkt als Verfahren** | Mehrheiten, Veto und paarweise Präferenzflüsse sind andere Verfahren und hier nicht enthalten. | ELECTRE, PROMETHEE |
| **Vier Normierungen, drei Normen, ein Aufschlag** | Der Aufschlag ρ = 0,01 ist fest; weitere Varianten (Fuzzy-Zahlen, andere Abstandsmaße, Summen- und Max-Normierung) sind nicht gemessen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Normierung und Gewichtung.** Mit der normierten Matrix $r_{ij}$ (Vektor: $r_{ij} = x_{ij} / \sqrt{\sum_i x_{ij}^2}$) ist $v_{ij} = w_j\, r_{ij}$, $w_j \ge 0$, $\sum_j w_j = 1$.

**Referenzpunkte.** $v_j^{+} = \max_i v_{ij}$, $v_j^{-} = \min_i v_{ij}$ (Stichprobe), oder die Nenn-Spannweite, oder $v_j^{-} + a\,(v_j^{+} - v_j^{-})$ mit dem Anspruchsniveau $a$.

**Abstand und Nähe.** $d_i^{\pm} = \left(\sum_j |v_{ij} - v_j^{\pm}|^p\right)^{1/p}$ für $p \ge 1$, $d_i^{\pm} = \max_j |v_{ij} - v_j^{\pm}|$ für $p = \infty$;
$C_i = d_i^{-} / (d_i^{+} + d_i^{-})$.

**p = 1 ist die Summe.** Für $p = 1$ gilt $d_i^{+} = \sum_j w_j (\max_k r_{kj} - r_{ij})$ und $d_i^{-} = \sum_j w_j (r_{ij} - \min_k r_{kj})$, also
$d_i^{+} + d_i^{-} = \sum_j w_j (\max_k r_{kj} - \min_k r_{kj})$ für alle $i$ gleich; $C_i$ ist eine affine Funktion der gewichteten Summe $\sum_j w_j r_{ij}$ und ordnet gleich. Bei
Min-Max ist $\max - \min = 1$, und $C_i = \sum_j w_j n_{ij}$ ist die Summe selbst.

**Wirksame Gewichte.** $\sum_j (w_j |x_j|)^p = \sum_j w_j^p |x_j|^p$: In der Summe der $p$-ten Potenzen wirkt $w_j^p$, normiert $w_j^p / \sum_k w_k^p$. Gleichwertige Gewichte
$\tilde w_j \propto w_j^{1/p}$ wirken wie $w$.

**Tschebyschow-Referenzpunkt (Erfüllungsskalarisierung).** $s_i(w, r) = \max_j w_j (r_j - n_{ij}) + \rho \sum_j w_j (r_j - n_{ij})$ mit dem Referenzpunkt $r$ (hier der
Idealpunkt) und $\rho > 0$; kleiner ist besser. Mit $w_j \propto 1/(r_j - n_{ij})$ sind alle gewichteten Abweichungen von $i$ gleich; jede nichtdominierte Alternative ist damit
bis auf den Einfluss von $\rho$ erreichbar.

**Reichweite der Summe.** Alternative $i$ gewinnt für ein $w \ge 0$, $\sum w = 1$, genau dann, wenn das LP $\max t$ unter $w^{\top}(n_i - n_k) \ge t$ für alle $k \ne i$ eine
Lösung mit $t > 0$ hat.

**Literatur** (nur aus Zusammenfassungen, nicht im Original geprüft): C. L. Hwang, K. Yoon, *Multiple Attribute Decision Making: Methods and Applications*, Springer, 1981
(TOPSIS; die Reihenfolge der Autoren ist in Zitaten uneinheitlich); A. P. Wierzbicki, Arbeiten zu Referenzpunkt-Verfahren und Erfüllungsskalarisierung (1980er Jahre; Titel und
Fundstelle nicht geprüft); K. Miettinen, *Nonlinear Multiobjective Optimization*, 1999 (Lehrbuch); A. Nikulin, K. Miettinen, M. M. Mäkelä, 2012 (Tschebyschow-Erfüllungsskalarisierung
und ihre Reichweite); A. Kokaraki, C. Hopfe, E. Robinson, E. Nikolaidou, *Testing the reliability of deterministic multi-criteria decision-making methods using building performance
simulation*, Renewable and Sustainable Energy Reviews, 2019 (TOPSIS am empfindlichsten gegenüber geänderten Präferenzen, ein Fallbeispiel).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
