"""TOPSIS und Referenzpunkt-Verfahren: Idealpunkt, Abstände in der Norm p, Nähe, Erfüllungsskalarisierung, wirksame Gewichte. Nur numpy.

Alle Funktionen arbeiten auf einer normierten Matrix N (n x m, höher besser) und Gewichten w. Der Abstand der Alternative i zu einem Punkt r ist
||w * (N_i - r)||_p; p = 1, 2, beliebiges p >= 1 oder inf (größte gewichtete Abweichung, Tschebyschow). Das Gewicht geht damit mit w^p in die Summe der p-ten Potenzen ein.
"""

from __future__ import annotations

import math

import numpy as np


def ideal_points(N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Idealpunkt (bester Wert je Kriterium der Stichprobe) und Anti-Idealpunkt (schlechtester) der normierten Matrix."""
    N = np.asarray(N, dtype=float)
    return N.max(axis=0), N.min(axis=0)


def p_norm(D: np.ndarray, p: float) -> np.ndarray:
    """Zeilenweise p-Norm einer Matrix von Abweichungen; p = inf: größte Abweichung. Für p < 1 keine Norm, daher abgelehnt."""
    if p < 1:
        raise ValueError("die Norm braucht p >= 1")
    A = np.abs(np.asarray(D, dtype=float))
    if math.isinf(p):
        return A.max(axis=-1)
    return (A ** p).sum(axis=-1) ** (1.0 / p)


def distances(N: np.ndarray, w: np.ndarray, point: np.ndarray, p: float) -> np.ndarray:
    """Gewichteter Abstand jeder Alternative zu einem Punkt (im Raum der normierten Werte) in der Norm p."""
    return p_norm((np.asarray(N, dtype=float) - np.asarray(point, dtype=float)) * np.asarray(w, dtype=float), p)


def closeness(N: np.ndarray, w: np.ndarray, p: float, ideal: np.ndarray | None = None, anti: np.ndarray | None = None):
    """TOPSIS: (d+, d-, C) mit C = d- / (d+ + d-). Ideal und Anti-Ideal aus der Stichprobe, wenn nicht vorgegeben. Fallen beide Abstände auf 0, ist C = 0,5."""
    N = np.asarray(N, dtype=float)
    hi, lo = ideal_points(N)
    d_plus = distances(N, w, hi if ideal is None else ideal, p)
    d_minus = distances(N, w, lo if anti is None else anti, p)
    total = d_plus + d_minus
    c = np.where(total == 0, 0.5, d_minus / np.where(total == 0, 1.0, total))
    return d_plus, d_minus, c


def asf_distance(N: np.ndarray, w: np.ndarray, ref: np.ndarray | None = None, rho: float = 0.01) -> np.ndarray:
    """Erfüllungsskalarisierung (Wierzbicki): max_j w_j (ref_j - N_ij) + rho * Summe_j w_j (ref_j - N_ij). Einseitig: Übererfüllung zählt negativ.
    KLEINER ist besser. Ohne Referenzpunkt: der Idealpunkt der Stichprobe."""
    N = np.asarray(N, dtype=float)
    ref = N.max(axis=0) if ref is None else np.asarray(ref, dtype=float)
    D = np.asarray(w, dtype=float) * (ref - N)
    return D.max(axis=1) + rho * D.sum(axis=1)


def asf_reachable(N: np.ndarray, i: int, rho: float = 0.01, eps: float = 1e-9) -> bool:
    """Gibt es Gewichte, bei denen Alternative i die Erfüllungsskalarisierung (Referenz: Idealpunkt der Stichprobe) STRIKT gewinnt? Konstruktion: w_j proportional zu
    1 / (Ideal_j - N_ij), dann sind alle gewichteten Abweichungen von i gleich und jede andere Alternative hat im Maximum-Term einen größeren Wert (sonst dominiert sie i).
    Das ist eine HINREICHENDE Bedingung: Der Aufschlag rho kann einen Punkt trotzdem verlieren lassen, dann ist die Antwort False, obwohl andere Gewichte ihn finden
    könnten. Abstände unter eps zählen als eps (i ist in diesem Kriterium bester)."""
    N = np.asarray(N, dtype=float)
    d = np.maximum(N.max(axis=0) - N[i], eps)
    w = 1.0 / d
    w = w / w.sum()
    s = asf_distance(N, w, None, rho)
    others = np.delete(s, i)
    return bool(len(others) == 0 or s[i] < others.min() - 1e-12)


def effective_weights(w: np.ndarray, p: float) -> np.ndarray:
    """Wirksame Gewichte in der Summe der p-ten Potenzen: w^p, auf Summe 1. Bei p = inf gibt es kein Summen-Äquivalent: w bleibt unverändert."""
    w = np.asarray(w, dtype=float)
    if math.isinf(p):
        return w / w.sum()
    e = w ** p
    return e / e.sum()


def equivalent_weights(w: np.ndarray, p: float) -> np.ndarray:
    """Gewichte, die im Abstand der Norm p wie w in der Summe wirken: w^(1/p), auf Summe 1 (p = inf: w unverändert)."""
    w = np.asarray(w, dtype=float)
    if math.isinf(p):
        return w / w.sum()
    e = w ** (1.0 / p)
    return e / e.sum()


def reference_raw(S: np.ndarray, mode: str, lo: np.ndarray, hi: np.ndarray, aspiration: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Ideal- und Anti-Idealpunkt im vorzeichenbehafteten Rohraum: stichprobe (bester/schlechtester Wert der Alternativen), nenn (feste Nenn-Spannweite lo, hi) oder
    anspruch (Anteil `aspiration` der Strecke vom Anti-Ideal der Stichprobe zum Ideal der Stichprobe)."""
    S = np.asarray(S, dtype=float)
    best, worst = S.max(axis=0), S.min(axis=0)
    if mode == "stichprobe":
        return best, worst
    if mode == "nenn":
        return np.asarray(hi, dtype=float), np.asarray(lo, dtype=float)
    if mode == "anspruch":
        return worst + aspiration * (best - worst), worst
    raise ValueError(f"unbekannter Referenzpunkt: {mode}")
