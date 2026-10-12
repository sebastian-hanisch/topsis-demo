"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def N3():
    """Drei Alternativen, drei Kriterien (alle höher besser, Min-Max-normiert: jede Spalte läuft von 0 bis 1), w = (0,5; 0,3; 0,2).
    Summe 0,70 / 0,42 / 0,40 (Ordnung 1 > 2 > 3).
    p = 1: d+ = 0,30 / 0,58 / 0,60, d- = 0,70 / 0,42 / 0,40, C = 0,70 / 0,42 / 0,40 (gleich der Summe).
    p = 2: d+ = 0,3000 / 0,5064 / 0,3536, d- = 0,5385 / 0,3231 / 0,2915, C = 0,6422 / 0,3895 / 0,4519 (Ordnung 1 > 3 > 2).
    p = inf: d+ = 0,30 / 0,50 / 0,25, d- = 0,50 / 0,30 / 0,25, C = 0,625 / 0,375 / 0,50 (Ordnung 1 > 3 > 2).
    Erfüllungsskalarisierung (Referenz (1, 1, 1), rho = 0,01, kleiner ist besser): 0,3030 / 0,5058 / 0,2560 (Ordnung 3 > 1 > 2)."""
    return np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.6], [0.5, 0.5, 0.0]])


@pytest.fixture
def w3():
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture
def hull():
    """A = (1, 0), B = (0, 1), C = (0,4; 0,4), w = (0,5; 0,5): Summe 0,5 / 0,5 / 0,4; C liegt unter der Strecke AB, keine Gewichtung macht C zum Sieger.
    TOPSIS (p = 1, 2, inf): C = 0,5 / 0,5 / 0,4 (C verliert); Erfüllungsskalarisierung: 0,505 / 0,505 / 0,306 (C gewinnt)."""
    return np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]])
