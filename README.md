# TOPSIS und Referenzpunkt: der Abstand zum Idealpunkt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-topsis-demo.streamlit.app/)**

Fünftes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind der
[gewichteten Summe](https://github.com/sebastian-hanisch/gewichtete-summe-demo). Statt gewichtete Werte zu addieren, misst TOPSIS für jede Alternative den **Abstand zum Idealpunkt**
(bester Wert je Kriterium) und zum **Anti-Idealpunkt** und ordnet nach der Nähe; das Referenzpunkt-Verfahren (Wierzbicki) nutzt nur den Abstand zu einem Anspruchsniveau, im
Tschebyschow-Abstand. Die Demo misst beides an der **Entscheidung** eines simulierten Entscheiders.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seine
Gewichte und seinen Nutzen (linear, konkav oder Cobb-Douglas in der Güte, oder linear in den Rohwerten); er ist ein Orakel, kein Mensch.

## Warum dieses Problem

**TOPSIS ist eine gewichtete Summe, deren Abstandsnorm die Gewichte umdeutet.** Mit dem Manhattan-Abstand (p = 1) ist die Nähe C exakt die gewichtete Summe. Mit dem euklidischen
Abstand (p = 2, der Standard) geht jedes Gewicht mit w² ein, und die Nähe ist keine Summe mehr. Die Kernfrage der Demo: Was ändert das an der Entscheidung, und was gewinnt man
dafür, nämlich mehr erreichbare Alternativen?

## Befunde und Korrekturen gegenüber dem Plan

- **Die Identität p = 1 ist exakt,** nicht nur ungefähr: gleiche Rangfolge in allen 1 800 Fällen (3 Kriterienzahlen × 3 Normierungen × 200 Matrizen), bei Min-Max die Nähe selbst
  gleich der Summe (größte Abweichung 4·10⁻¹⁶). Bei Vektor- und z-Normierung ist C eine affine Funktion der Summe.
- **Der Abstand zum Idealpunkt ordnet nicht besser.** Meine Hypothese vor der Messung, die „eingebaute Konkavität“ des Euklid-Abstands helfe bei konkavem Nutzen, ist in jeder
  gemessenen Zelle widerlegt: Die Summe mit denselben Gewichten und derselben Normierung trifft Platz 1 mindestens so oft wie TOPSIS mit p = 2.
- **Wirksame Gewichte erklären nur einen Teil.** Mit den gleichwertigen Gewichten w^(1/p) wird TOPSIS bei linearem Entscheider schlechter, bei Cobb-Douglas besser (8 Kriterien:
  73,0 statt 58,5 %, die Summe 67,5 %). Das heißt nicht, dass TOPSIS dort „besser“ ist: Es gilt für diesen Entscheider.
- **Erreichbarkeit ist nicht Güte.** Der Abstand erreicht mehr nichtdominierte Alternativen als die Summe, der Tschebyschow-Referenzpunkt fast alle; die Trefferquote steigt dadurch nicht.
- **Kokaraki et al. (2019) nur als Richtung.** TOPSIS ist unter Gewichtsstörung 0,2 bis 3,7 Punkte weniger stabil als die Summe (Richtung wie dort berichtet, Betrag klein); mit
  gleichwertigen Gewichten ist es 0,7 bis 4,6 Punkte stabiler als die Summe. Die „Empfindlichkeit“ liegt an der Gewichtssemantik.
- **Ein Anspruchsniveau unter dem Besten macht aus TOPSIS ein Ziel-Treffen.** Der Abstand ist zweiseitig: Alternativen oberhalb des Anspruchs (80 % der Strecke) werden bestraft.
  TOPSIS trifft Platz 1 dann nur noch in 18,5 bis 27,0 % der Fälle (mit dem Idealpunkt der Stichprobe 62,5 bis 85,0 %); der einseitige Tschebyschow-Referenzpunkt in 44,5 bis 59,0 %.
- **Der Idealpunkt der Stichprobe gegen den Nenn-Idealpunkt macht bei n = 20 keinen belastbaren Unterschied** (höchstens 3,0 Punkte, Standardfehler 3 bis 3,5); die
  Stichprobenabhängigkeit wirkt über die Rangumkehr, die in einem eigenen Stück gemessen wird.
- **Die Reichweite ist eine untere Schranke.** Die Gewichtsstichprobe unterschätzt sie (Summe bei 5 Kriterien: 70 % auf der Stichprobe, 79 % per LP); für den Referenzpunkt kommt
  die Konstruktion w_j ∝ 1/(Ideal_j − Wert_j) hinzu. Der Aufschlag ρ = 0,01 kann einzelne Punkte verlieren lassen.
- **Das Preset „Ein Punkt, den keine Summe erreicht“ zeigt Reichweite, keinen Treffer:** Der Tschebyschow-Referenzpunkt wählt dort eine Alternative, die kein Gewicht zum
  Sieger der Summe macht, und verfehlt den Besten.
- **Die Störung der Gewichte ist lognormal** (SplitMix64), nicht Dirichlet wie in der Phase 0; die Stabilitätswerte der Demo sind neu gemessen.
- **Die Experimente starten auf Klick** (rund 20 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n Alternativen (5 bis 40), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit; optional ein Kriterium auf den Anteil δ seiner Streuung gestaucht.
- **Entscheider:** wahre Gewichte gleichverteilt über dem Simplex (Generator); Nutzen linear, konkav (Wurzel) oder Cobb-Douglas (Logarithmus) in der Güte, oder linear in den
  Rohwerten bezogen auf die feste Nenn-Spannweite.
- **TOPSIS:** Normierung (Vektor, Min-Max, z-Wert, Perzentil, Rohwerte) und Gewichtung; Idealpunkt und Anti-Idealpunkt der Stichprobe, der Nenn-Spannweite oder als Anspruchsniveau;
  gewichtete Abstände in der Norm p (1, 1,5, 2, 3, ∞); Nähe C = d⁻ / (d⁺ + d⁻).
- **Referenzpunkt-Verfahren:** Erfüllungsskalarisierung max_j w_j (r_j − n_ij) + ρ Σ_j w_j (r_j − n_ij) mit dem Idealpunkt als Referenz und ρ = 0,01 (kleiner ist besser).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), n = 20, unabhängig vom Regler n; die Reichweite über die ersten 60 Matrizen und 10 000 Gewichte.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, derselbe Entscheider, alle Verfahren:** Vergleiche zwischen Verfahren sind gepaart.
2. **Messung an der Entscheidung:** Platz 1 gegen den Besten des Entscheiders, Kendalls τ-b der Rangfolge, Nutzenverlust.
3. **Vier Entscheider:** Güte/linear, Güte/konkav, Güte/Cobb-Douglas, Rohwerte/linear; die Aussage „TOPSIS verliert“ ist an die Zelle gebunden.
4. **Reichweite per LP und Stichprobe:** Für die Summe exakt (LP), für TOPSIS und den Referenzpunkt über dieselbe Gewichtsstichprobe; Ja oder Nein je nichtdominierter Alternative.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. Trefferquoten über 200 feste Matrizen, n = 20, wahre Gewichte. **Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte;
Unterschiede unter etwa 6 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Ist TOPSIS mit p = 1 die Summe? (3, 5, 8 Kriterien; Min-Max, Vektor, z-Wert) | In allen 1 800 Fällen dieselbe Rangfolge; bei Min-Max C = Summe auf 4·10⁻¹⁶. | `test_manhattan_topsis_is_the_sum_in_every_case` |
| Wie oft wählt TOPSIS mit p = 2 denselben Platz 1 wie die Summe? | 5 Kriterien: Min-Max 84,0 %, Vektor 83,5 %, z-Wert 84,0 %; 3 Kriterien 88,0 bis 88,5 %; 8 Kriterien 76,5 bis 79,5 %. Dieselbe vollständige Rangfolge: höchstens 1,5 %. p = ∞: 5 Kriterien 73,5 bis 77,0 %. | `test_euclid_and_chebyshev_agree_with_the_sum_only_partly` |
| Trefferquote Platz 1, Rohwerte/linear (3 / 5 / 8 Kriterien) | **Summe Min-Max 94,0 / 96,5 / 93,0 %; TOPSIS Min-Max p = 2 87,0 / 85,0 / 80,5 %; TOPSIS Vektor p = 2 85,5 / 79,5 / 74,5 %**; p = ∞ 76,5 / 77,5 / 62,0 %; Tschebyschow 73,5 / 74,0 / 61,5 %. | `test_first_place_for_every_method` |
| Dasselbe, Güte/linear | Summe Min-Max 85,0 / 67,0 / 78,5 %; TOPSIS Min-Max p = 2 80,0 / 62,5 / 69,0 %; Vektor p = 2 77,0 / 65,0 / 66,5 %; Summe Vektor 81,5 / 70,5 / 81,0 %. | `test_first_place_for_every_method` |
| Wie groß ist der Abstand zur Summe bei linearem Entscheider? | 4,5 bis 14,5 Punkte über alle Zellen (Güte/linear und Rohwerte/linear, 3 bis 8 Kriterien, Min-Max und Vektor); bei Min-Max und Rohwerten 7,0 bis 12,5. | `test_the_gap_between_sum_and_euclid…` |
| Konkave Entscheider (Güte/konkav, Güte/Cobb-Douglas; 5 / 8 Kriterien) | Summe Min-Max 72,5 / 78,0 und 69,0 / 67,5 %; TOPSIS Min-Max p = 2 65,5 / 70,0 und 64,0 / 58,5 %: die Summe liegt vorn. | `test_the_sum_is_at_least_as_good…` |
| TOPSIS ohne Normierung | 28,0 bis 41,5 % Platz 1 in allen zwölf Zellen: Die Einheit entscheidet. | `test_topsis_without_normalization…` |
| Wie viel ändern Norm und Normierung an Platz 1? (5 Kriterien, Güte/linear) | TOPSIS Vektor p = 2 gegen Summe Min-Max 80,5 %; Vektor gegen Min-Max (p = 2) 82,5 %; p = 2 gegen p = 1 84,0 %, gegen p = ∞ 88,0 %; **Perzentil gegen Min-Max nur 65,0 %**. | `test_agreement_between_variants` |
| Woher kommt der Referenzpunkt? (5 Kriterien, Min-Max, p = 2) | Ideal der Stichprobe / Nenn-Ideal / Anspruchsniveau 80 %: TOPSIS Rohwerte/linear 85,0 / 84,0 / **27,0 %**, Güte/linear 62,5 / 61,0 / **18,5 %**; Tschebyschow 74,0 / 75,0 / 59,0 % und 56,0 / 57,5 / 44,5 %. | `test_reference_point_source` |
| Reichweite: Anteil der nichtdominierten Alternativen, die bei einer Gewichtung gewinnen (3 / 5 Kriterien) | Summe (LP) 74,8 / 79,2 %; TOPSIS p = 2 (Stichprobe) 85,4 / 83,0 % (Min-Max) und 85,2 / 84,2 % (Vektor); **Tschebyschow-Referenzpunkt 99,1 / 99,6 %** (untere Schranke). | `test_reach` |
| Gleichwertige Gewichte (8 Kriterien; Güte/linear, konkav, Cobb-Douglas, Rohwerte/linear nur 5) | TOPSIS p = 2 mit w / mit √w: Güte/linear 69,0 / 65,0 %, konkav 70,0 / 71,0 %, **Cobb-Douglas 58,5 / 73,0 %** (Summe 67,5 %); 5 Kriterien: konkav 65,5 / 68,5 %, Cobb-Douglas 64,0 / 70,0 %. | `test_equivalent_weights_and_effective_weights` |
| Bleibt Platz 1 unter Gewichtsstörung? (σ = 0,1 / 0,2 / 0,4; 5 Kriterien, Güte/linear) | Summe 95,1 / 90,3 / 80,7 %; TOPSIS Min-Max 94,9 / 88,7 / 77,0 %; Vektor 92,3 / 86,7 / 77,3 %; **gleichwertige Gewichte 95,8 / 92,5 / 85,3 %**; Tschebyschow 93,0 / 84,5 / 70,8 %. | `test_stability_of_first_place` |

## Ehrliche Grenzen

- **Der Entscheider ist ein Orakel.** Er hat einen verborgenen Nutzen in einer von vier Formen; wie Menschen Abstände zum Ideal wahrnehmen, ist nicht gemessen. „TOPSIS verliert“ heißt:
  gegenüber diesem Orakel. Wer Ausgewogenheit als Wert an sich will, hat ein anderes Orakel.
- **Kleine Unterschiede sind Rauschen.** Der Unterschied zwischen Idealpunkt der Stichprobe und Nenn-Idealpunkt (Phase 0), die Stabilität bei σ = 0,1 und die Reihenfolge von Summe
  und TOPSIS in einzelnen Zellen liegen innerhalb von zwei Standardfehlern.
- **Reichweite ist eine untere Schranke.** Stichprobe und Konstruktion unterschätzen; der Aufschlag ρ = 0,01 ist fest.
- **Rangumkehr nicht gemessen.** Eine hinzugefügte Alternative ändert den Idealpunkt der Stichprobe; wie oft das Platz 1 kippt, steht in einem eigenen Stück (in der Phase 0 bei einer
  Alternative mehr 0 bis 2 % von 200 Matrizen, zu klein für eine Aussage).
- **Ein Generator, wahre Gewichte.** n = 20, 3 bis 8 Kriterien; die Gewichte sind bekannt (die Erhebung steht in den Vorgängern). Quellenangaben nur aus Zusammenfassungen.
- **Vier Normierungen, drei Normen.** Fuzzy-TOPSIS und andere Varianten (Summen-, Max-Normierung, andere Abstandsmaße) sind nicht gemessen.

## Verifikation

- **Kern:** Idealpunkte, Abstände in p = 1, 2, ∞, Nähe, Erfüllungsskalarisierung, wirksame und gleichwertige Gewichte gegen Handrechnung auf den Mini-Instanzen (`tests/conftest.py`:
  drei Alternativen mit w = (0,5; 0,3; 0,2), Nähe 0,70 / 0,42 / 0,40 bei p = 1 und 0,6422 / 0,3895 / 0,4519 bei p = 2; der Punkt unter der Hülle A = (1, 0), B = (0, 1), C = (0,4; 0,4)).
- **Orakel** (`tests/test_oracle_topsis.py`): Nähe gegen eine Definitionsschleife mit `scipy.spatial.distance` für p = 1, 1,5, 2, 3, ∞; **p = 1 gleich Summe** über 500
  Zufallsmatrizen und drei Normierungen (Gegenprobe: bei p = 2 nur selten gleiche Rangfolge); Skaleninvarianz bei Vektor, Min-Max und z-Wert (Gegenprobe: Rohwerte nicht);
  Erfüllungsskalarisierung gegen Definition; ein Punkt unter der Hülle wird von Summe und TOPSIS nie, vom Referenzpunkt gefunden; dominierte Alternativen gewinnen nie;
  Quadratsumme mit gleichwertigen Gewichten gleich der gewichteten Summe der Quadrate.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Presets, jede Normierung, Norm und Referenzpunkt, Permalink, PDF und Experimente erst
  nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `tps_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `tps_topsis.py` | Idealpunkte, Abstände, Nähe, Erfüllungsskalarisierung, wirksame Gewichte, Erreichbarkeit |
| `tps_score.py` | Normierungen (auch für beliebige Punkte), Summe, Nutzen des Entscheiders, Dominanz, LP der Reichweite, Kendalls τ-b |
| `tps_evaluation.py` | Fall der App, vier Experimente, Verdikte |
| `tps_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `tps_visualization.py` | Tabellen und Plotly-Abbildungen (Achsen gesperrt) |
| `tps_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` für die LP der Reichweite) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanzen, TOPSIS, Score, Orakel, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Rangumkehr durch eine hinzugefügte Alternative:** Der Idealpunkt der Stichprobe ist die Quelle; das Protokoll (Hinzufügen, Entfernen, Duplizieren) hat ein eigenes Stück.
- **Paarweise Präferenzen** (Mehrheiten, Veto, Präferenzflüsse): andere Verfahren der Linie.
- **Unsichere Gewichte als Bereich:** Die Reichweite ist nur Ja oder Nein je Alternative; die Verteilung über den Gewichtsraum steht beim SMAA.
- **Fuzzy-TOPSIS und weitere Varianten:** nicht gemessen und nicht empfohlen.
- **Suchläufe** (Populations-Metaheuristiken mit Tschebyschow-Zerlegung oder Referenzpunkt-Gitter): Dort entsteht die Menge; hier ist sie gegeben.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
