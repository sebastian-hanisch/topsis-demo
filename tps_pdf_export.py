"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import tps_constants as C
from tps_evaluation import Analysis, de, p_label

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "δ": "delta", "∞": "inf", "⁺": "+", "⁻": "-", "≥": ">=", "≤": "<=", "−": "-",
            "€": "EUR", "·": "*", "²": "^2", "√": "sqrt"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    names = [f"A{i + 1}" for i in range(s.n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("TOPSIS und Referenzpunkt: der Abstand zum Idealpunkt"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Alternativen n = {s.n}, Kriterien m = {s.m}, Normierung: {C.NORM_SHORT[s.norm]}, Norm p = {p_label(s.p)}, Referenzpunkt: "
                            f"{C.REFERENCE_LABELS[s.reference]}, Gewichte: {C.WEIGHT_LABELS[s.weights]}, Entscheider: {C.DECIDER_LABELS[s.decider]}, "
                            f"Streuung des ersten Kriteriums delta = {de(s.delta, 2)}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for k, label in (("topsis", "TOPSIS"), ("summe", "Summe"), ("asf", "Tschebyschow-Referenzpunkt")):
        r = a.results[k]
        pdf.cell(0, 5, _s(f"{label}: Platz 1 {names[r.top1]} ({'getroffen' if r.hit else 'nicht getroffen'}), tau {de(r.tau, 2)}, Verlust {de(r.loss, 3)}, "
                          f"von der Summe erreichbar: {'ja' if a.sum_reach[k] else 'nein'}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, _s(f"Bester nach Nutzen: {names[a.best]}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head = ["Kriterium", "wahres Gewicht", "eingegeben", "wirksam (w^p)"]
    widths = [40, 36, 36, 36]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    inst = a.case.matrix
    for k in range(s.m):
        row = [f"K{k + 1} ({'+' if inst.directions[k] > 0 else '-'}) {inst.units[k]}", de(a.case.w[k], 3), de(a.w_in[k], 3), de(a.w_eff[k], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Die ersten zehn Alternativen nach TOPSIS-Rang"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    order = sorted(range(s.n), key=lambda i: -a.closeness[i])[:10]
    for rank, i in enumerate(order, start=1):
        pdf.cell(0, 5, _s(f"{rank}. {names[i]}: d+ {de(a.d_plus[i], 3)}, d- {de(a.d_minus[i], 3)}, C {de(a.closeness[i], 3)}, Summe {de(a.results['summe'].score[i], 3)}"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Entscheider ist ein simuliertes Orakel, kein Mensch. Quelle: topsis-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
