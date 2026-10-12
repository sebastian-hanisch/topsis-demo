"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import math

import numpy as np

import tps_constants as C
import tps_evaluation as ev
import tps_presets as P
import tps_score as sc


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["norm"] in C.NORMS and p["p"] in C.P_CHOICES and p["reference"] in C.REFERENCES and p["weights"] in C.WEIGHT_MODES, name
        assert p["decider"] in C.DECIDERS and p["delta"] in C.DELTA_CHOICES and p["aspiration"] in C.ASPIRATION_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Euklid, Vektor)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.NORM_LABELS) == set(C.NORMS) == set(C.NORM_SHORT)
    assert set(C.REFERENCE_LABELS) == set(C.REFERENCES) and set(C.WEIGHT_LABELS) == set(C.WEIGHT_MODES)
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT) == set(C.DECIDER_PARTS)
    assert C.DEFAULT_P in C.P_CHOICES and math.inf in C.P_CHOICES and 1.0 in C.P_CHOICES


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000
    assert C.REACH_SEEDS == C.SWEEP_SEEDS[:60]


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Standardfall (Euklid, Vektor)")
    assert a.results["summe"].hit and a.results["wahr"].hit and not a.results["topsis"].hit

    a = _analysis("Manhattan = Summe")
    assert a.settings.p == 1.0
    assert np.array_equal(np.argsort(-a.closeness, kind="stable"), np.argsort(-a.results["summe"].score, kind="stable"))
    b = _analysis("Manhattan = Summe", p=2.0)
    assert b.results["topsis"].top1 != b.results["summe"].top1 and b.results["summe"].hit and not b.results["topsis"].hit

    a = _analysis("Ein Punkt, den keine Summe erreicht")
    t = a.results["asf"].top1
    assert a.front[t] and not sc.is_possible_winner(a.N, t) and t != a.results["summe"].top1
    assert not a.sum_reach["asf"] and a.results["summe"].hit and not a.results["asf"].hit

    a = _analysis("Gleichwertige Gewichte")
    b = _analysis("Gleichwertige Gewichte", weights="wahr")
    assert a.results["topsis"].hit and not b.results["topsis"].hit and not b.results["summe"].hit
    assert a.settings.decider == "guete_cobb" and a.settings.m == 8
