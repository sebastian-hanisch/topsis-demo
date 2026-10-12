"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

N_MIN, N_MAX, DEFAULT_N = 5, 40, 20           # Score und Normierung: O(n m); die LP der Reichweite laufen nur in Experiment 3 (n = 20)
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
SEED_MAX, DEFAULT_SEED = 999999, 45          # Seed 45: alle vier Presets zeigen ihren Effekt (je Preset geprüft)

NORMS = ("vektor", "minmax", "z", "perzentil", "roh")
NORM_LABELS = {
    "vektor": "Vektor (Euklidische Norm; die Normierung von Hwang und Yoon)",
    "minmax": "Min-Max (Spannweite der Stichprobe)",
    "z": "z-Wert (Standardabweichung der Stichprobe)",
    "perzentil": "Perzentil (Rang in der Stichprobe)",
    "roh": "Rohwerte (ohne Normierung)",
}
NORM_SHORT = {"vektor": "Vektor", "minmax": "Min-Max", "z": "z-Wert", "perzentil": "Perzentil", "roh": "Rohwerte"}
DEFAULT_NORM = "vektor"

P_CHOICES = (1.0, 1.5, 2.0, 3.0, math.inf)    # Norm des Abstands; inf = Tschebyschow (größte gewichtete Abweichung)
DEFAULT_P = 2.0

REFERENCES = ("stichprobe", "nenn", "anspruch")
REFERENCE_LABELS = {
    "stichprobe": "Idealpunkt der Stichprobe (bester und schlechtester Wert der Alternativen)",
    "nenn": "Nenn-Idealpunkt (feste Nenn-Spannweite je Kriterium)",
    "anspruch": "Anspruchsniveau (Anteil der Strecke Anti-Ideal bis Ideal der Stichprobe)",
}
DEFAULT_REFERENCE = "stichprobe"
ASPIRATION_CHOICES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_ASPIRATION = 0.8

WEIGHT_MODES = ("wahr", "gleichwertig")
WEIGHT_LABELS = {
    "wahr": "Die des Entscheiders (verborgen), als eingegeben",
    "gleichwertig": "Gleichwertig: w hoch 1/p, damit der Abstand w als Austauschrate nutzt",
}
DEFAULT_WEIGHTS = "wahr"

DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear")
DECIDER_LABELS = {
    "guete_linear": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "guete_konkav": "Konkav in der Güte (Wurzel, gewichtet)",
    "guete_cobb": "Cobb-Douglas in der Güte (Logarithmus, gewichtet)",
    "roh_linear": "Linear in den Rohwerten (feste Nenn-Spannweite)",
}
DECIDER_SHORT = {"guete_linear": "Güte/linear", "guete_konkav": "Güte/konkav", "guete_cobb": "Güte/Cobb-Douglas", "roh_linear": "Rohwerte/linear"}
DECIDER_PARTS = {"guete_linear": ("guete", "linear"), "guete_konkav": ("guete", "konkav"), "guete_cobb": ("guete", "cobb"), "roh_linear": ("roh", "linear")}
DEFAULT_DECIDER = "guete_linear"

DELTA_CHOICES = (1.0, 0.5, 0.2, 0.05)         # Anteil der Streuung des ersten Kriteriums (1 = unverändert)
DEFAULT_DELTA = 1.0

RHO = 0.01                                     # Aufschlag der Erfüllungsskalarisierung (Wierzbicki)
CRITERIA_PAIR = (0, 1)                         # Standardachsen der Punktwolke (Kriterium 1 und 2)

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
REACH_SEEDS = SWEEP_SEEDS[:60]                 # Experiment 3: je Matrix bis zu 25 LP
EXP_N = 20                                     # Experimente: feste Instanz mit 20 Alternativen, unabhängig vom Regler
EXP_M = (3, 5, 8)
EXP_IDENTITY_NORMS = ("minmax", "vektor", "z")
EXP_REACH_M = (3, 5)
EXP_REACH_WEIGHTS = 10000                      # Gewichte je Matrix in der Reichweite (Stichprobe unterschätzt; die Summe läuft zusätzlich über die LP)
EXP_STABILITY_M = 5
EXP_STABILITY_SIGMAS = (0.1, 0.2, 0.4)         # Streuung der lognormalen Gewichtsstörung
EXP_STABILITY_DRAWS = 10                       # Störungen je Matrix

COLOR_SUM = "#8A96A6"
COLOR_TOPSIS = "#2F6B65"
COLOR_ASF = "#c0392b"
COLOR_WARN = "#c0392b"
COLOR_TRUE = "#D68A2E"
COLOR_LINE = "#4c78a8"
COLOR_FRONT = "rgba(150,150,150,0.55)"
METHOD_COLORS = {"summe": COLOR_SUM, "topsis": COLOR_TOPSIS, "asf": COLOR_ASF, "wahr": COLOR_TRUE, "summe_wirksam": "#9b6b9b", "gleich": "#bbbbbb"}

PRESETS = {
    "Standardfall (Euklid, Vektor)": {"n": DEFAULT_N, "m": DEFAULT_M, "norm": DEFAULT_NORM, "p": DEFAULT_P, "reference": DEFAULT_REFERENCE,
                                      "aspiration": DEFAULT_ASPIRATION, "weights": DEFAULT_WEIGHTS, "decider": DEFAULT_DECIDER, "delta": 1.0,
                                      "seed": DEFAULT_SEED},
    "Manhattan = Summe": {"n": 20, "m": 5, "norm": "minmax", "p": 1.0, "reference": "stichprobe", "aspiration": 0.8, "weights": "wahr", "decider": "guete_linear",
                          "delta": 1.0, "seed": 45},
    "Ein Punkt, den keine Summe erreicht": {"n": 20, "m": 3, "norm": "minmax", "p": 2.0, "reference": "stichprobe", "aspiration": 0.8, "weights": "wahr",
                                            "decider": "guete_linear", "delta": 1.0, "seed": 18},
    "Gleichwertige Gewichte": {"n": 20, "m": 8, "norm": "minmax", "p": 2.0, "reference": "stichprobe", "aspiration": 0.8, "weights": "gleichwertig",
                               "decider": "guete_cobb", "delta": 1.0, "seed": 52},
}
PRESET_HELP = {
    "Standardfall (Euklid, Vektor)": "20 Alternativen, 5 Kriterien, Vektor-Normierung, euklidischer Abstand zum Idealpunkt der Stichprobe: Die Summe trifft den Besten, TOPSIS nicht.",
    "Manhattan = Summe": "Mit p = 1 ist die Nähe die gewichtete Summe (dieselbe Rangfolge, bei Min-Max sogar derselbe Wert); mit p = 2 wählt TOPSIS bei diesem Seed eine andere Alternative.",
    "Ein Punkt, den keine Summe erreicht": "Der Tschebyschow-Referenzpunkt wählt eine Alternative, die für KEINE Gewichtung die Summe gewinnt (Punkt unter der Hülle). Sie ist hier nicht der Beste: "
                                           "Reichweite ist nicht Güte.",
    "Gleichwertige Gewichte": "Cobb-Douglas-Entscheider, 8 Kriterien: Mit den gleichwertigen Gewichten w^(1/p) trifft TOPSIS den Besten, mit w nicht, und die Summe verfehlt ihn.",
}
