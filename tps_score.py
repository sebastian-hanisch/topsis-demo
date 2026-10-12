"""Normierungen (auch für beliebige Punkte), gewichtete Summe, Nutzen des Entscheiders, Verschmälerung, Dominanz und Reichweite. Nur numpy (scipy nur in der LP);
eigene Kopie der kleinen Helfer aus gewichtete-summe-demo (kein Import über Repos).

Alle Funktionen arbeiten auf S (n x m) im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt (Kriterien mit Richtung ↓ vorher mit -1 multipliziert).
"""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import CRITERIA, to_raw, utility


def average_ranks(col: np.ndarray) -> np.ndarray:
    """Ränge 1..n mit Mittelwert bei Gleichständen (wie scipy.stats.rankdata, method='average')."""
    col = np.asarray(col, dtype=float)
    order = np.argsort(col, kind="mergesort")
    sorted_col = col[order]
    ranks = np.empty(len(col), dtype=float)
    i = 0
    n = len(col)
    while i < n:
        j = i
        while j + 1 < n and sorted_col[j + 1] == sorted_col[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _safe(divisor: np.ndarray) -> np.ndarray:
    """Teiler 0 (konstante Spalte) durch 1 ersetzen: die Spalte wird dann zu lauter gleichen Werten."""
    return np.where(divisor == 0, 1.0, divisor)


def normalize(S: np.ndarray, how: str) -> np.ndarray:
    """Normierte Matrix; `how` aus NORMS. roh: unverändert."""
    S = np.asarray(S, dtype=float)
    if how == "perzentil":
        n = len(S)
        if n < 2:
            return np.zeros_like(S)
        return np.column_stack([(average_ranks(S[:, k]) - 1.0) / (n - 1.0) for k in range(S.shape[1])])
    return normalize_points(S, S, how)


def normalize_points(S: np.ndarray, X: np.ndarray, how: str) -> np.ndarray:
    """Beliebige Punkte X (k x m, im vorzeichenbehafteten Raum) mit den Parametern der Stichprobe S normieren (Min-Max, z, Vektor, roh: dieselbe affine Abbildung;
    Perzentil: Anteil der Stichprobenwerte unter dem Punkt, durch n - 1, auf [0, 1] begrenzt)."""
    S = np.asarray(S, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if how == "roh":
        return X.copy()
    if how == "minmax":
        return (X - S.min(axis=0)) / _safe(S.max(axis=0) - S.min(axis=0))
    if how == "z":
        return (X - S.mean(axis=0)) / _safe(S.std(axis=0))
    if how == "vektor":
        return X / _safe(np.linalg.norm(S, axis=0))
    if how == "perzentil":
        n = len(S)
        below = (S[None, :, :] < X[:, None, :]).sum(axis=1)
        return np.clip(below / max(n - 1, 1), 0.0, 1.0)
    raise ValueError(f"unbekannte Normierung: {how}")


def weighted_score(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(N, dtype=float) @ np.asarray(w, dtype=float)


def top1(score: np.ndarray) -> int:
    """Index des höchsten Scores; bei Gleichstand die kleinste Nummer."""
    return int(np.argmax(score))


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def narrow_criterion(goodness: np.ndarray, delta: float, column: int = 0) -> np.ndarray:
    """Staucht die Güte eines Kriteriums um 0,5 auf den Anteil `delta` der Streuung: g' = 0,5 + delta (g - 0,5). delta = 1 ändert nichts."""
    g = np.array(goodness, dtype=float, copy=True)
    g[:, column] = 0.5 + delta * (g[:, column] - 0.5)
    return g


def decider_utility(scale: str, form: str, goodness: np.ndarray, S: np.ndarray, w: np.ndarray, m: int) -> np.ndarray:
    """Nutzen des Entscheiders: in der Güte (Form linear, konkav, cobb aus dem Generator) oder linear in den Rohwerten bezogen auf die feste Nenn-Spannweite."""
    if scale == "guete":
        return utility(goodness, w, form)
    if scale == "roh":
        if form != "linear":
            raise ValueError("in den Rohwerten ist nur die lineare Form definiert")
        lo, hi = nominal_bounds(m)
        return ((np.asarray(S, dtype=float) - lo) / (hi - lo)) @ np.asarray(w, dtype=float)
    raise ValueError(f"unbekannte Wertskala: {scale}")


def build_case(goodness: np.ndarray, directions: np.ndarray, m: int, delta: float):
    """Aus Güte und Richtungen: (Güte nach Verschmälerung, Rohwerte, vorzeichenbehaftete Rohwerte)."""
    g = narrow_criterion(goodness, delta)
    raw = to_raw(g, m)
    return g, raw, raw * directions


def utility_loss(utility_values: np.ndarray, chosen: int) -> float:
    return float(np.max(utility_values) - utility_values[chosen])


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)


def dominance_matrix(G: np.ndarray) -> np.ndarray:
    """D[i, j] = True, wenn i die Alternative j schwach dominiert (in jedem Kriterium mindestens so gut, in einem besser)."""
    G = np.asarray(G, dtype=float)
    return (G[:, None, :] >= G[None, :, :]).all(axis=2) & (G[:, None, :] > G[None, :, :]).any(axis=2)


def front_mask(G: np.ndarray) -> np.ndarray:
    """Alternativen, die von keiner anderen dominiert werden."""
    return ~dominance_matrix(G).any(axis=0)


def is_possible_winner(N: np.ndarray, i: int) -> bool:
    """Gibt es Gewichte w >= 0 mit sum w = 1, bei denen i die Summe STRIKT vor allen anderen hat? (LP: maximiere den kleinsten Abstand t, t > 0 nötig.)
    Eine identische Alternative verhindert einen strikten Sieg."""
    from scipy.optimize import linprog

    N = np.asarray(N, dtype=float)
    n, m = N.shape
    diffs = np.delete(N[i] - N, i, axis=0)
    if len(diffs) == 0:
        return True
    if (np.abs(diffs).sum(axis=1) == 0).any():
        return False
    res = linprog(np.r_[np.zeros(m), -1.0], A_ub=np.hstack([-diffs, np.ones((len(diffs), 1))]), b_ub=np.zeros(len(diffs)),
                  A_eq=[[1.0] * m + [0.0]], b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)], method="highs")
    return bool(res.status == 0 and -res.fun > 1e-9)
