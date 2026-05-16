# 🧠 KCNQ3-Lens

**Quantitative EEG-Analyse für Familien von Kindern mit seltenen Epilepsien.**

KCNQ3-Lens ist ein Open-Source-Werkzeug, das Familien und Klinikern hilft, pädiatrische EEG-Aufnahmen mit derselben quantitativen Linse zu betrachten, die in der Forschung verwendet wird — ohne Daten in die Cloud zu schicken und ohne ein Forschungslabor zu benötigen.

Es entstand, weil Eltern von Kindern mit KCNQ3-Spektrum-Erkrankungen (und anderen seltenen Channelopathien) häufig EEG-Befunde erhalten wie „multiregionale Spikes, keine klinischen Anfälle" — ohne die quantitativen Details, die die moderne Epilepsie-Forschung als wesentlich betrachtet: *wo genau* die Spikes sind, *wann* sie im Schlaf ihren Höhepunkt erreichen, wie ihre *Morphologie* aussieht, ob die *Schlafspindeln* reduziert sind, und ob es subklinische *anhaltende Bursts* gibt, die nicht als Anfälle markiert sind.

Dieses Werkzeug berechnet diese Dinge und bereitet sie so auf, dass Sie sie mit Ihrer Ärztin / Ihrem Arzt besprechen können.

> ⚠️ **KCNQ3-Lens ist kein Medizinprodukt.** Es diagnostiziert, behandelt oder ersetzt keine klinische EEG-Beurteilung. Siehe [DISCLAIMER.md](DISCLAIMER.md) vor Gebrauch.

---

## Was es tut

Elf quantitative Analysen pro EEG-Aufnahme, in zwei Tiers:

**Tier 1 — Kernanalysen (ab v0.1–v0.7)**

| Analyse | Was gemessen wird |
|---|---|
| **Spike-Topographie** | Per-Kanal-Kurtosis — zeigt, wo epileptiforme Aktivität konzentriert ist |
| **Schlafspindel-Dichte** | Dichte pro Minute auf Cz (oder anderen zentralen Kanälen), altersangemessener Vergleich (YASA-Standard) |
| **Hintergrund-Leistung + PDR** | Posteriore Grundaktivität, Delta/Alpha-Quotient, Bandverteilung |
| **Anhaltende Bursts** | Erkennt rhythmische Bursts ≥3s, die subklinische elektrographische Ereignisse darstellen können |
| **Spike-Morphologie** | Klassifikation: einfacher Spike vs. Sharp Wave vs. komplexe Spike-Wave |
| **Schlafstadien** | NREM/REM/Wach-Klassifikation per 30s-Epoche via YASA + heuristischem Fallback |
| **Spike-Wave-Index (SWI)** | % der NREM-Zeit mit kontinuierlicher SW-Aktivität — formales CSWS/ESES-Kriterium |
| **Wach-/Schlaf-Aufschlüsselung** | Separate Spike-Raten pro Zustand + Aktivierungsfaktor |
| **Bilaterale Synchronie** | Fokal / regional / bilateral synchron / generalisiert pro Spike |

**Tier 2 — Forschungsanalysen (ab v0.13.x)**

| Analyse | Was gemessen wird | Hinweis |
|---|---|---|
| **Slow-Wave-Detektion** | SO-Dichte, Amplitude, Dauer in NREM3 | Deskriptiv — keine pädiatrischen Normwerte |
| **HFO-Ripple-Detektion** | 80–250-Hz-Energiebursts (Staba-Methode) | Forschungsmetrik — erfordert ≥500 Hz Abtastrate |
| **SO-Spindel-Kopplung** | PLV-basierter Kopplungswinkel und -stärke | Reift durch die Adoleszenz — noch keine Altersnormwerte |
| **IED-Detektion** | Ensemble-Heuristik + optionaler SpikeNet-Stub | Regelbasiert, kein ML; SpikeNet erfordert lokale Modellgewichte |

Jede Analyse liefert numerische Befunde, Grafiken und (optional) eine verständliche Interpretation, generiert vom KI-Anbieter Ihrer Wahl mit Ihrem eigenen API-Schlüssel.

**Unterstützte KI-Anbieter** (einen wählen — eigenen Schlüssel mitbringen):
- **Claude** (Anthropic) — [Schlüssel beantragen](https://console.anthropic.com/settings/keys)
- **GPT** (OpenAI) — [Schlüssel beantragen](https://platform.openai.com/api-keys)
- **Gemini** (Google) — [Schlüssel beantragen](https://aistudio.google.com/app/apikey)

---

## Wofür das Tool gemacht ist — und wofür nicht

### ✅ Geeignet für

- Eltern von Kindern mit seltenen Epilepsien, die das EEG ihres Kindes quantitativ verstehen möchten
- Verlauf von EEG-Kennzahlen über mehrere Aufnahmen verfolgen (vor/nach Medikation, vor/nach Therapieänderungen)
- Strukturierte Befunde und gezielte Fragen für die Neuropädiater-Sprechstunde generieren
- KCNQ3-Spektrum-Patienten speziell — Nihon-Kohden-EEG-1200A-Support und Varianten-Kontext
- Open-Source / reproduzierbare quantitative Pipelines, die Familien selbst nutzen können

### ❌ Nicht geeignet für

- Den Ersatz klinischer EEG-Beurteilung durch qualifizierte Neurologen
- Erwachsenen-EEG (Algorithmen und Normwerte sind pädiatrisch eingestellt)
- Diagnose-Stellung
- Akute / Notfall / Intensivmedizin
- Echtzeit- oder Streaming-EEG-Analyse
- Reines EEG-Anschauen (nutzen Sie dafür [EDFbrowser](https://www.teuniz.net/edfbrowser/))
- Forschungs-Quellenrekonstruktion oder fortgeschrittene Statistik (nutzen Sie direkt [MNE-Python](https://mne.tools))

---

## Anwendungsbereich & Mindestanforderungen

KCNQ3-Lens ist für **mehrstündige Übernacht-EEG-Aufnahmen** von Kindern mit seltenen Epilepsien gemacht. Die Analysen setzen bestimmte Aufnahmecharakteristika voraus — Daten außerhalb dieser Bereiche liefern unzuverlässige Ergebnisse.

**Mindestanforderungen an die Aufnahme:**

| Anforderung | Minimum | Empfohlen | Anmerkung |
|---|---|---|---|
| Dauer | 30 Minuten | 6+ Stunden inkl. Schlaf | Mehrere Analysen benötigen eine Schlafphase |
| Abtastrate | 100 Hz | 200–500 Hz | <100 Hz behindert die Filter; >500 Hz für HFO-Support nötig |
| EEG-Kanäle | 6 (frontal + zentral + posterior) | 19 (volles 10-20-System) | Standard-10-20-Kanalnamen erwartet |
| Alter des Kindes | 2–18 Jahre | 3–12 Jahre | Beste Norm-Abdeckung in diesem Bereich |
| Format | EDF, BDF, BrainVision, EEGLAB, Nihon Kohden EEG-1200A | — | Siehe oben |

**Das Tool eignet sich NICHT gut für:**

- Reine Wach-EEGs <30 Minuten (Schlaf-Fenster wird für mehrere Analysen benötigt)
- Aufnahmen mit <6 EEG-Kanälen (Topographie wird sinnlos)
- Aufnahmen mit <100 Hz Abtastrate (Filter- und Morphologie-Einschränkungen)
- Aufnahmen mit überwiegend Bewegungs-/Muskel-Artefakten (Auto-Erkennung geplant für v0.3)
- Erwachsenen-Aufnahmen (Normwerte sind pädiatrisch)

---

## Zwei Analyse-Modi

### 1. Einzelne Aufnahme
Eine EEG-Datei laden, alle fünf Analysen laufen lassen, Ergebnisse pro Analyse-Tab durchsehen.

### 2. Zwei Aufnahmen vergleichen
Zwei EEG-Dateien hochladen (z. B. **vor** und **nach** einer Medikationsumstellung). Das Tool berechnet die Veränderung jeder Kennzahl, markiert Verbesserung / Verschlechterung / unverändert und kann die KI bitten, das klinisch einzuordnen.

Dies ist der wichtigste Workflow für jede Familie, die Therapieantwort tracken möchte.

---

## Unterstützte EEG-Formate

- **Nihon Kohden EEG-1200A** (`.eeg`) — das Langzeit-Übernacht-Format der Nihon Kohden EEG-2100/2200-Systeme. **Dieser Reader ist neuartig** — MNE-Python und EDFbrowser lesen dieses Format nicht korrekt.
- **EDF / EDF+** (`.edf`) — gängiges offenes Format
- **BDF** (`.bdf`)
- **BrainVision** (`.vhdr`)
- **EEGLAB** (`.set`)

---

## Installation

Python 3.10 oder neuer wird benötigt.

```bash
# Repository klonen
git clone https://github.com/therealsamm85/kcnq3-lens.git
cd kcnq3-lens

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv .venv
source .venv/bin/activate           # Linux/Mac
# .venv\Scripts\activate            # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Die App öffnet sich im Browser unter `http://localhost:8501`.

---

## Benutzung

1. **App öffnen.** Läuft lokal — nichts wird hochgeladen.
2. **Sprache wählen** (oben in der Seitenleiste: Englisch / Deutsch).
3. **Modus wählen**: Einzelne Aufnahme oder Vergleich zweier Aufnahmen.
4. **Alter des Kindes eingeben** (für altersangemessene Vergleiche).
5. **Optional: bekannte genetische Variante eingeben** (z. B. „KCNQ3 p.Arg230His") — wird nur in der KI-Interpretation erwähnt.
6. **Wach- und Schlaf-Zeitfenster** in der Seitenleiste setzen.
7. **EEG-Datei(en) laden** — entweder hochladen oder lokalen Dateipfad eingeben.
8. **„Analysen ausführen"** klicken. Dauert 1–5 Minuten je nach Aufnahmelänge.
9. **Befunde durchgehen** über die fünf Analyse-Tabs.
10. (Optional) **KI-Anbieter wählen und API-Schlüssel eingeben** in der Seitenleiste, um eine verständliche Interpretation zu erhalten. Es werden nur numerische Werte gesendet — Roh-EEG verlässt Ihren Computer nie.
11. **JSON-Befunde und/oder Interpretation herunterladen**, um sie mit dem behandelnden Arzt zu teilen.

---

## Datenschutz und Sicherheit

- **Die gesamte EEG-Verarbeitung erfolgt auf Ihrem Computer.** Keine Roh-EEG-Daten verlassen Ihre Maschine.
- **Die KI-Interpretation ist optional.** Wenn Sie sie nutzen, werden nur abgeleitete numerische Werte (z. B. „Spindel-Dichte: 1.3/min") mit Ihrem eigenen API-Schlüssel an den gewählten Anbieter gesendet.
- **Das Tool speichert nichts zwischen Sitzungen.** Schließen des Browser-Tabs löscht die Ergebnisse.
- **Kein Medizinprodukt.** Befunde immer mit der behandelnden Ärztin / dem behandelnden Arzt besprechen, bevor Konsequenzen gezogen werden.

---

## Vergleich mit anderen EEG-Tools

KCNQ3-Lens ersetzt keine bestehenden Tools — es schließt eine spezifische Lücke: **familientaugliche quantitative EEG-Analyse mit optionaler KI-Interpretation, inklusive Unterstützung für das Nihon-Kohden-EEG-1200A-Format, das kein anderes Open-Source-Tool korrekt liest.**

| Tool | Stärken | Wann es stattdessen sinnvoll ist | Wie wir ergänzen |
|---|---|---|---|
| **[MNE-Python](https://mne.tools)** | Wissenschaftliche EEG-Bibliothek für Python — Preprocessing, ICA, Quellrekonstruktion, voller Forschungsstack | Eigene Forschungsanalysen, Source-Modeling, fortgeschrittene Statistik | Wir nutzen MNE intern für Nicht-NK-Formate und Standard-Signalverarbeitung |
| **[YASA](https://github.com/raphaelvallat/yasa)** | Validiertes ML-basiertes Sleep-Staging und Spindel-Detektion | Polysomnographie-Forschung, validierte Schlaf-Analyse | Integriert seit v0.3 — YASA ist der Standard-Spindel-Backend; Heuristik ist der Fallback |
| **[EDFbrowser](https://www.teuniz.net/edfbrowser/)** | Freier, schneller EDF/BDF-Wellenform-Viewer | Visuelles Durchscrollen der Rohwellenform, manuelle Marker | EDFbrowser für visuelle Sichtung, KCNQ3-Lens für quantitative Zusammenfassung |
| **[Persyst](https://www.persyst.com/)** | Klinische EEG-Software (kommerziell) — Spike-Detektion, Sleep-Staging, qEEG-Berichte | Klinik-Workflows mit Persyst-Lizenz | Wir sind die Open-Source-, familientaugliche Alternative — kein Ersatz für klinische Infrastruktur |
| **[Brainstorm](https://neuroimage.usc.edu/brainstorm/)** / **EEGLAB** / **FieldTrip** | MATLAB-Neuroimaging-Suiten mit fortgeschrittenem Source-Modeling | Forschungslabore mit MATLAB-Lizenz | Andere Zielgruppe; wir konkurrieren nicht in Quellenrekonstruktion |
| **[Luna](https://zzz.bwh.harvard.edu/luna/)** | Harvard-Kommandozeilen-Toolkit für Schlaf-EEG | Schlaf-Forschung mit großen Kohorten | Andere Zielgruppe; wir fokussieren auf Familien-Nutzung mit GUI |
| **[NeuroKit2](https://github.com/neuropsychology/NeuroKit)** | Allgemeine Biosignal-Analyse (EEG, EKG, EDA) | Multi-Signal-Forschung | Anderer Scope; wir fokussieren ausschließlich auf pädiatrisches EEG |

**Die einzigartige Kombination, die KCNQ3-Lens bietet:**

1. **Nihon-Kohden-EEG-1200A-Reader** — kein anderes Open-Source-Tool liest dieses Format korrekt
2. **Familientaugliche UI** — fast nichts im Open-Source-Bereich richtet sich an Eltern
3. **Multi-Provider-KI-Interpretation** — Claude / GPT / Gemini, eigener Schlüssel
4. **Pre/Post-Vergleich** — direkt in die UI eingebaut
5. **Mehrsprachigkeit** — Open-Source-EEG ist fast ausschließlich Englisch
6. **KCNQ3-Spektrum-Kontext** — varianten-bewusste Referenzbereiche und Diskussion

---

## Wofür KCNQ3-Lens gut ist (und wo es Grenzen hat)

**Gut darin:**
- Muster zu quantifizieren, die klinische EEG-Befunde oft nur qualitativ zusammenfassen
- Dieselben Zahlen konsistent über mehrere Aufnahmen hinweg zu liefern, sodass Veränderungen über Monate/Jahre nachvollziehbar sind
- Subklinische Muster (lange Bursts, niedrige Spindel-Dichte, langsamer Hintergrund) sichtbar zu machen, die nicht immer markiert werden
- Strukturierte Berichte und konkrete Fragen für das Arztgespräch zu erstellen

**Grenzen:**
- Ersetzt nicht die räumlich-zeitliche Mustererkennung einer erfahrenen Epileptologin
- Schlafstadien werden via YASA klassifiziert (erwachsenentrainiertes Modell — pädiatrische Genauigkeit ist geringer)
- Der Nihon-Kohden-EEG-1200A-Reader wurde anhand einer einzelnen Aufnahme-Familie reverse-engineered — andere Aufnahmen in diesem Format ggf. zu verifizieren
- Altersnormwerte stammen aus der Literatur und repräsentieren nicht jedes Kind perfekt
- Erkennungs-Schwellenwerte sind konservativ; einzelne Ereignisse können übersehen oder überzählig erfasst werden

---

## Aufbauend auf vorhandener Open-Source-Arbeit

KCNQ3-Lens hängt von exzellenter Vorarbeit ab. Wenn dieses Tool für dich nützlich ist, bitte auch die zugrundeliegenden Projekte würdigen:

- **[MNE-Python](https://mne.tools)** — das wissenschaftliche Fundament für EEG-Analyse in Python.
- **[YASA](https://github.com/raphaelvallat/yasa)** — validiertes Sleep-Staging und Spindel-Detektion (integriert seit v0.3).
- **[SciPy](https://scipy.org)** / **[NumPy](https://numpy.org)** — Signalverarbeitung und numerische Berechnung.
- **[Streamlit](https://streamlit.io)** — Frontend-Framework, das ein lokales browser-basiertes GUI praktikabel macht.
- **[Anthropic](https://www.anthropic.com/)**, **[OpenAI](https://openai.com/)**, **[Google](https://ai.google.dev/)** — die LLM-Anbieter für die optionale KI-Interpretation.

---

## Beitragen

Beiträge willkommen, insbesondere:

- **Klinische Validierungsstudie** — Tool-Ausgaben gegen expertenbewertete EEGs vergleichen (≥10 Aufnahmen — der wirkungsvollste nächste Schritt)
- **Pädiatrisches YASA-Tuning** — YASAs Modell ist erwachsenentrainiert; eine pädiatrische Korrektur würde die Schlafstadien-Genauigkeit erheblich verbessern
- **UI-Integration der Tier-2-Befunde** — Slow Waves, HFOs, Kopplung und IED werden berechnet, sind aber nur in JSON/PDF verfügbar
- Validierung gegen andere EEG-Formate und Aufnahmesysteme
- Weitere Nihon-Kohden-Datei-Varianten (andere Abtastraten, Kanal-Layouts)
- Übersetzungen der UI in weitere Sprachen

Bei Beiträgen, die klinische Interpretation betreffen: bitte zuerst ein Issue öffnen, damit Sicherheitsaspekte besprochen werden können.

---

## Lizenz

MIT. Siehe [LICENSE](LICENSE).

---

## Englische Dokumentation

Siehe [README.md](README.md) für die englische Originalversion.
