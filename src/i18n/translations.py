"""UI string translations. English is the source of truth; other languages fall back to it."""

TRANSLATIONS = {
    # ─── English (source) ──────────────────────────────────────────────────
    "en": {
        # App-level
        "app_title": "🧠 KCNQ3-Lens",
        "app_subtitle": (
            "Quantitative EEG analysis for families of children with rare epilepsies. "
            "Runs entirely on your machine. **Not a medical device.**"
        ),
        "disclaimer_header": "⚠️ Important: read this first",
        "disclaimer_body": (
            "This tool is **not** a medical device, **not** a diagnostic tool, and "
            "**not** a substitute for clinical EEG interpretation by a qualified "
            "neurologist. It surfaces quantitative patterns in EEG recordings that "
            "families and clinicians can discuss together. **Never start, stop, or "
            "modify any treatment based on its output.** All therapeutic decisions "
            "belong to your child's doctor.\n\n"
            "The algorithms are research-grade, not clinically validated. The Nihon "
            "Kohden EEG-1200A reader was reverse-engineered from a single recording "
            "family; other recordings in this format may not parse correctly.\n\n"
            "EEG files are processed entirely on your local machine and never "
            "uploaded. If you use the optional AI interpretation feature, only "
            "derived numerical metrics (not raw EEG) are sent to the chosen API "
            "with your own key."
        ),
        "footer": (
            "KCNQ3-Lens is open source under MIT license. Built by and for the "
            "rare-epilepsy community. Not a medical device. See DISCLAIMER.md."
        ),

        # Sidebar — global
        "sidebar_language": "Language",
        "sidebar_mode": "Analysis mode",
        "mode_single": "Single recording",
        "mode_compare": "Compare two recordings",

        # Sidebar — recording settings
        "sidebar_recording_settings": "Recording settings",
        "sidebar_age": "Child's age (years)",
        "sidebar_age_help": "Used to look up age-appropriate normative ranges.",
        "sidebar_variant": "Known genetic variant (optional)",
        "sidebar_variant_placeholder": "e.g. KCNQ3 p.Arg230His",
        "sidebar_variant_help": (
            "Mentioned in the AI interpretation if provided. No effect on "
            "numerical analyses."
        ),
        "sidebar_windows": "Analysis windows",
        "sidebar_windows_caption": (
            "Times below are in seconds from the start of the recording. For "
            "overnight studies, set sleep window to cover the actual sleep period."
        ),
        "wake_start": "Wake window start (s)",
        "wake_end": "Wake window end (s)",
        "sleep_start": "Sleep window start (s)",
        "sleep_end": "Sleep window end (s)",

        # Sidebar — AI
        "sidebar_ai_header": "AI interpretation (optional)",
        "sidebar_ai_caption": (
            "Choose any of the supported LLM providers and supply your own API "
            "key. Only numerical findings are sent; raw EEG never leaves your "
            "machine."
        ),
        "ai_provider": "AI provider",
        "ai_model": "Model",
        "ai_api_key": "{provider} API key",
        "ai_key_link": "🔗 Get a {provider} API key",
        "ai_sdk_missing": "`{package}` is not installed.\n\nRun: `pip install {package}`",

        # File upload
        "step1_header": "1. Load EEG file",
        "step1_header_pre": "1a. Load pre-treatment EEG",
        "step1_header_post": "1b. Load post-treatment EEG",
        "file_picker": "Choose an EEG file",
        "file_picker_help": (
            "Supported: Nihon Kohden .EEG (EEG-1200A), EDF/EDF+, BDF, "
            "BrainVision, EEGLAB"
        ),
        "local_path": "...or enter a local file path (faster for very large recordings)",
        "local_path_help": (
            "If your EEG file is very large (multi-GB), this is faster than "
            "uploading."
        ),
        "reading": "Reading {filename}...",
        "load_error": "Could not read file: {error}",
        "metric_format": "Format",
        "metric_sfreq": "Sampling rate",
        "metric_duration": "Duration",
        "metric_eeg_channels": "EEG channels",
        "channel_layout": "Channel layout",

        # Run analyses
        "step2_header": "2. Run analyses",
        "step2_header_compare": "2. Run analyses on both recordings",
        "load_first_message": "Load a file above to run analyses.",
        "load_both_message": "Load both files above to run the comparison.",
        "run_button": "▶ Run all analyses",
        "run_button_compare": "▶ Compare both recordings",
        "progress_topography": "Computing spike topography... (this is the slowest step)",
        "progress_spindles": "Detecting sleep spindles...",
        "progress_background": "Quantifying background power and posterior dominant rhythm...",
        "progress_bursts": "Searching for sustained rhythmic bursts...",
        "progress_morphology": "Classifying spike morphology...",
        "progress_done": "Analyses complete.",
        "analysis_failed": "{analysis} failed: {error}",
        "running_pre": "Analyzing pre-treatment recording...",
        "running_post": "Analyzing post-treatment recording...",

        # Findings tabs
        "step3_header": "3. Findings",
        "step3_header_compare": "3. Comparison",
        "tab_topography": "Topography",
        "tab_spindles": "Sleep spindles",
        "tab_background": "Background",
        "tab_bursts": "Bursts",
        "tab_morphology": "Morphology",
        "tab_raw": "Raw JSON",
        "tab_summary": "Summary",

        # Topography tab
        "topo_header": "Per-channel spike topography",
        "topo_caption": (
            "Higher median kurtosis = more sharp-transient activity on that "
            "channel. Normal background ≈ 3–5. Values >10 indicate epileptiform "
            "activity."
        ),
        "topo_normal_upper": "normal upper",

        # Spindle tab
        "spindle_header": "Sleep spindle density at {channel}",
        "spindle_density": "Density (spindles/min)",
        "spindle_count": "Total spindles detected",
        "spindle_freq": "Peak frequency (Hz)",
        "spindle_below_norm": (
            "Density is **below** the age-typical range ({low}–{high} "
            "spindles/min). Low spindle density can affect sleep-dependent "
            "memory consolidation."
        ),
        "spindle_above_norm": "Density is above the age-typical range ({low}–{high}).",
        "spindle_in_norm": "Density is within the age-typical range ({low}–{high}).",
        "spindle_caption": (
            "Spindles are 11–16 Hz transient oscillations critical for memory "
            "consolidation during NREM-2 sleep."
        ),

        # Background tab
        "bg_header": "Background EEG power",
        "bg_pdr": "Posterior dominant rhythm",
        "bg_dar": "Delta / Alpha ratio",
        "bg_pdr_normative": "Age-typical range: {low}–{high} Hz",
        "bg_dar_normative": "Wake: typically < 1.0",
        "bg_severely_slow": (
            "Posterior dominant rhythm is **substantially slower** than "
            "age-typical. This is a marker of cortical immaturity or dysfunction."
        ),
        "bg_mildly_slow": "Posterior dominant rhythm is mildly slower than age-typical.",
        "bg_appropriate": "Posterior dominant rhythm is age-appropriate.",

        # Bursts tab
        "bursts_header": "Sustained rhythmic bursts on {channel}",
        "bursts_total": "Total bursts (≥3s)",
        "bursts_5s": "Bursts ≥5s",
        "bursts_10s": "Bursts ≥10s",
        "bursts_max": "Longest (s)",
        "bursts_longest_label": "**Longest bursts found:**",
        "bursts_warn_long": (
            "Found **{count} bursts of 10 seconds or longer**. These may "
            "represent subclinical electrographic events and are worth "
            "discussing with your child's neurologist."
        ),

        # Morphology tab
        "morph_header": "Spike morphology on {channel}",
        "morph_simple": "Simple spikes (<70ms)",
        "morph_sharp": "Sharp waves (70–200ms)",
        "morph_complex": "Complex spike-wave (≥200ms)",
        "morph_events": "**Events detected:** {n} ({rate:.1f}/min)",
        "morph_polyspike": "**Polyspike fraction:** {pct:.0f}% (events <250ms apart)",
        "morph_complex_classification": (
            "Majority of events are **complex spike-wave** complexes. This "
            "morphology is associated with atypical absence / CSWS spectrum."
        ),
        "morph_simple_classification": (
            "Majority of events are **simple spikes**. This morphology is "
            "associated with classic focal patterns (e.g., Rolandic)."
        ),
        "morph_mixed": "Mixed spike morphology.",

        # Raw JSON tab
        "raw_header": "Raw analysis output (JSON)",
        "raw_caption": "Download this and share it with your doctor if helpful.",
        "raw_download": "💾 Download findings as JSON",

        # AI interpretation
        "step4_header": "4. AI interpretation (optional)",
        "ai_need_key": (
            "Enter your **{provider}** API key in the sidebar to enable "
            "plain-language interpretation. Only numerical metrics are sent — "
            "raw EEG never leaves your machine."
        ),
        "ai_sdk_needed": (
            "The `{package}` package is not installed. Run "
            "`pip install {package}` in your terminal, then restart the app."
        ),
        "ai_generate_button": "🤖 Generate interpretation with {provider}",
        "ai_thinking": "Asking {provider} to interpret the findings...",
        "ai_failed": "AI interpretation failed: {error}",
        "ai_generated_by": "_Generated by {provider}_",
        "ai_download_md": "💾 Download interpretation as Markdown",

        # Comparison view
        "compare_summary_header": "What changed between the two recordings",
        "compare_metric_pre": "Pre",
        "compare_metric_post": "Post",
        "compare_metric_change": "Change",
        "compare_label_topography": "Spike topography",
        "compare_label_spindles": "Sleep spindle density",
        "compare_label_background": "Background slowing",
        "compare_label_bursts": "Sustained bursts",
        "compare_label_morphology": "Spike morphology",
        "compare_no_change": "no change",
        "compare_improved": "↓ improved",
        "compare_worsened": "↑ worsened",
        "compare_increased": "↑ increased",
        "compare_decreased": "↓ decreased",
        "compare_unchanged": "≈ unchanged",
        "compare_ai_button": "🤖 Generate change interpretation with {provider}",
        "compare_ai_caption": (
            "The AI will compare pre- and post-treatment numbers and explain "
            "what likely matters clinically."
        ),

        # PDF reports
        "pdf_header": "5. PDF reports",
        "pdf_caption": (
            "Generate a printable summary. Two versions: doctor (technical detail) "
            "and parent (plain language)."
        ),
        "pdf_doctor_button": "📄 Download doctor report (PDF)",
        "pdf_parent_button": "📄 Download parent report (PDF)",

        "tab_time_of_night": "Time of night",
        "topomap_title": "Spike-activity topography (median kurtosis)",
        "topomap_caption": (
            "Hot spots show where epileptiform activity concentrates. "
            "Reds = more activity; blues = less. Cz/Pz redness suggests "
            "midline central-parietal involvement (SMA / speech-motor network)."
        ),
        "ton_header": "Spike burden across the night",
        "ton_caption": (
            "Spike count per minute in each 30-minute window. Peaks during "
            "the first NREM cycle (typically 1–3 hours after sleep onset) "
            "are characteristic of sleep-activated patterns like CSWS/ESES."
        ),
        "ton_peak_label": "Peak: {peak:.1f}/min at {hours:.1f}h",

        # Auto sleep detection
        "auto_detect_button": "🔍 Auto-detect sleep window",
        "auto_detect_success": (
            "Detected sleep window: {start:.1f}h–{end:.1f}h "
            "({duration:.1f}h, confidence: {conf})"
        ),
        "auto_detect_low_conf": (
            "⚠ Low confidence — please verify the detected window manually. "
            "Overnight recordings with heavy artifact or unusual sleep "
            "architecture can confuse the heuristic."
        ),
        "auto_detect_failed": "Auto-detect failed: {error}",

        # Quality control
        "tab_quality": "Quality",
        "qc_header": "Recording quality",
        "qc_grade": "Overall grade",
        "qc_usable_epochs": "Usable epochs",
        "qc_good_channels": "Good channels",
        "qc_flagged_channels_label": "Flagged channels:",
        "qc_no_warnings": "No quality concerns detected.",

        # Insights
        "tab_insights": "🧭 Insights",
        "insights_header": "Proactive insights",
        "insights_caption": (
            "Rule-based interpretation of the findings. Highlights affected "
            "brain networks, possible clinical patterns, and cross-modal "
            "observations. All deterministic — no AI used here."
        ),
        "insights_anatomy_header": "Affected brain regions",
        "insights_anatomy_caption": (
            "Top channels with highest epileptiform activity, mapped to "
            "their underlying brain regions and functional networks."
        ),
        "insights_top_networks_header": "Top affected functional networks",
        "insights_artifact_warning": (
            "⚠ Note: {channels} are frontal-pole electrodes which can be "
            "affected by eye-blink artifact. Their elevated activity should "
            "be interpreted cautiously."
        ),
        "insights_patterns_header": "Possible clinical patterns",
        "insights_patterns_caption": (
            "Pattern matches based on combinations of findings. **These are "
            "NOT diagnoses** — they are starting points for discussion with "
            "your child's neurologist."
        ),
        "insights_pattern_criteria_met": "Criteria met ({n}/{total}):",
        "insights_pattern_questions": "Questions worth asking the doctor:",
        "insights_cross_modal_header": "Cross-modal observations",
        "insights_cross_modal_caption": (
            "Combinations of findings that imply more than each finding alone."
        ),
        "insights_no_patterns": (
            "No clinical patterns matched at the moderate-confidence threshold. "
            "This could mean the findings are within normal range, or the "
            "pattern library doesn't yet cover this presentation."
        ),

        # v0.5: clinical-grade metrics
        "tab_clinical": "Clinical",
        "clinical_header": "Clinical-grade metrics (v0.5)",
        "clinical_caption": (
            "Numbers that clinicians specifically look for: formal SWI per "
            "sleep stage, wake-vs-sleep activation factor, and spike spread "
            "pattern."
        ),
        "swi_header": "Spike-Wave Index (per sleep stage)",
        "swi_caption": (
            "SWI = % of each sleep stage occupied by continuous spike-wave "
            "activity. CSWS / ESES criterion (Tassinari): N3 SWI ≥ 85%."
        ),
        "swi_csws_met": "⚠ CSWS criterion is **MET** (N3 SWI ≥ {threshold}%)",
        "swi_csws_not_met": "CSWS criterion not met.",
        "state_split_header": "Wake vs sleep spike-rate split",
        "state_split_caption": (
            "Separate spike-rate calculations per state. Activation factor "
            "(NREM/wake) ≥ 3 indicates sleep activation; ≥ 10 is dramatic."
        ),
        "synchrony_header": "Bilateral synchrony / spread pattern",
        "synchrony_caption": (
            "How each detected spike spreads across the scalp."
        ),
        "sleep_stages_header": "Sleep architecture",
        "sleep_stages_caption": (
            "Heuristic stage classification. YASA's model is trained on "
            "adult PSG, so pediatric output is approximate."
        ),
    },

    # ─── German ────────────────────────────────────────────────────────────
    "de": {
        "app_title": "🧠 KCNQ3-Lens",
        "app_subtitle": (
            "Quantitative EEG-Analyse für Familien von Kindern mit seltenen "
            "Epilepsien. Läuft komplett lokal. **Kein Medizinprodukt.**"
        ),
        "disclaimer_header": "⚠️ Wichtig: bitte zuerst lesen",
        "disclaimer_body": (
            "Dieses Werkzeug ist **kein** Medizinprodukt, **kein** "
            "Diagnose-Tool und **kein** Ersatz für die klinische "
            "EEG-Beurteilung durch eine qualifizierte Neurologin oder einen "
            "qualifizierten Neurologen. Es macht quantitative Muster in EEG-"
            "Aufnahmen sichtbar, die Familien und Ärztinnen/Ärzte gemeinsam "
            "besprechen können. **Beginnen, stoppen oder verändern Sie niemals "
            "eine Therapie auf Basis dieses Tools.** Alle therapeutischen "
            "Entscheidungen liegen bei der behandelnden Ärztin bzw. dem "
            "behandelnden Arzt.\n\n"
            "Die Algorithmen sind forschungsnah, nicht klinisch validiert. Der "
            "Nihon-Kohden-EEG-1200A-Reader wurde anhand einer einzelnen "
            "Aufnahme-Familie reverse-engineered; andere Aufnahmen in diesem "
            "Format können abweichend sein.\n\n"
            "EEG-Dateien werden vollständig lokal verarbeitet und nie "
            "hochgeladen. Wenn Sie die optionale KI-Interpretation nutzen, "
            "werden nur abgeleitete numerische Werte (kein Roh-EEG) an die "
            "gewählte API mit Ihrem eigenen Schlüssel gesendet."
        ),
        "footer": (
            "KCNQ3-Lens ist Open Source unter MIT-Lizenz. Gebaut von und für "
            "die Seltene-Epilepsien-Community. Kein Medizinprodukt. Siehe "
            "DISCLAIMER.md."
        ),

        "sidebar_language": "Sprache",
        "sidebar_mode": "Analyse-Modus",
        "mode_single": "Einzelne Aufnahme",
        "mode_compare": "Zwei Aufnahmen vergleichen",

        "sidebar_recording_settings": "Aufnahme-Einstellungen",
        "sidebar_age": "Alter des Kindes (Jahre)",
        "sidebar_age_help": (
            "Wird verwendet, um altersangemessene Normalbereiche heranzuziehen."
        ),
        "sidebar_variant": "Bekannte genetische Variante (optional)",
        "sidebar_variant_placeholder": "z.B. KCNQ3 p.Arg230His",
        "sidebar_variant_help": (
            "Wird in der KI-Interpretation erwähnt, falls angegeben. Hat "
            "keinen Einfluss auf die numerischen Analysen."
        ),
        "sidebar_windows": "Analyse-Zeitfenster",
        "sidebar_windows_caption": (
            "Zeiten unten in Sekunden ab Aufnahmebeginn. Bei Übernacht-"
            "Aufnahmen das Schlaf-Fenster auf die tatsächliche Schlafzeit "
            "setzen."
        ),
        "wake_start": "Wach-Fenster Start (s)",
        "wake_end": "Wach-Fenster Ende (s)",
        "sleep_start": "Schlaf-Fenster Start (s)",
        "sleep_end": "Schlaf-Fenster Ende (s)",

        "sidebar_ai_header": "KI-Interpretation (optional)",
        "sidebar_ai_caption": (
            "Wählen Sie einen der unterstützten LLM-Anbieter und geben Sie "
            "Ihren eigenen API-Schlüssel ein. Es werden nur numerische "
            "Befunde gesendet; Roh-EEG verlässt Ihren Computer nie."
        ),
        "ai_provider": "KI-Anbieter",
        "ai_model": "Modell",
        "ai_api_key": "{provider} API-Schlüssel",
        "ai_key_link": "🔗 {provider} API-Schlüssel beantragen",
        "ai_sdk_missing": (
            "`{package}` ist nicht installiert.\n\nAusführen: "
            "`pip install {package}`"
        ),

        "step1_header": "1. EEG-Datei laden",
        "step1_header_pre": "1a. Vor-Behandlungs-EEG laden",
        "step1_header_post": "1b. Nach-Behandlungs-EEG laden",
        "file_picker": "EEG-Datei auswählen",
        "file_picker_help": (
            "Unterstützt: Nihon Kohden .EEG (EEG-1200A), EDF/EDF+, BDF, "
            "BrainVision, EEGLAB"
        ),
        "local_path": (
            "...oder lokalen Dateipfad angeben (schneller bei sehr großen "
            "Aufnahmen)"
        ),
        "local_path_help": (
            "Bei sehr großen EEG-Dateien (mehrere GB) schneller als das "
            "Hochladen."
        ),
        "reading": "Lese {filename} ein...",
        "load_error": "Datei konnte nicht gelesen werden: {error}",
        "metric_format": "Format",
        "metric_sfreq": "Abtastrate",
        "metric_duration": "Dauer",
        "metric_eeg_channels": "EEG-Kanäle",
        "channel_layout": "Kanal-Layout",

        "step2_header": "2. Analysen ausführen",
        "step2_header_compare": "2. Analysen für beide Aufnahmen ausführen",
        "load_first_message": "Datei oben laden, um die Analysen zu starten.",
        "load_both_message": (
            "Beide Dateien oben laden, um den Vergleich zu starten."
        ),
        "run_button": "▶ Alle Analysen ausführen",
        "run_button_compare": "▶ Beide Aufnahmen vergleichen",
        "progress_topography": (
            "Berechne Spike-Topographie... (langsamster Schritt)"
        ),
        "progress_spindles": "Erkenne Schlafspindeln...",
        "progress_background": (
            "Quantifiziere Hintergrund-Leistung und posteriore Grundaktivität..."
        ),
        "progress_bursts": "Suche nach anhaltenden rhythmischen Bursts...",
        "progress_morphology": "Klassifiziere Spike-Morphologie...",
        "progress_done": "Analysen abgeschlossen.",
        "analysis_failed": "{analysis} fehlgeschlagen: {error}",
        "running_pre": "Analysiere Vor-Behandlungs-Aufnahme...",
        "running_post": "Analysiere Nach-Behandlungs-Aufnahme...",

        "step3_header": "3. Befunde",
        "step3_header_compare": "3. Vergleich",
        "tab_topography": "Topographie",
        "tab_spindles": "Schlafspindeln",
        "tab_background": "Hintergrund",
        "tab_bursts": "Bursts",
        "tab_morphology": "Morphologie",
        "tab_raw": "JSON (roh)",
        "tab_summary": "Übersicht",

        "topo_header": "Per-Kanal Spike-Topographie",
        "topo_caption": (
            "Höhere mediane Kurtosis = mehr scharfe Transienten auf diesem "
            "Kanal. Normaler Hintergrund ≈ 3–5. Werte >10 deuten auf "
            "epileptiforme Aktivität hin."
        ),
        "topo_normal_upper": "Normalbereich-Obergrenze",

        "spindle_header": "Schlafspindel-Dichte auf {channel}",
        "spindle_density": "Dichte (Spindeln/min)",
        "spindle_count": "Spindeln gesamt",
        "spindle_freq": "Spitzenfrequenz (Hz)",
        "spindle_below_norm": (
            "Dichte ist **unterhalb** des altersangemessenen Bereichs "
            "({low}–{high} Spindeln/min). Niedrige Spindel-Dichte kann die "
            "schlafabhängige Gedächtniskonsolidierung beeinträchtigen."
        ),
        "spindle_above_norm": (
            "Dichte ist oberhalb des altersangemessenen Bereichs "
            "({low}–{high})."
        ),
        "spindle_in_norm": (
            "Dichte liegt im altersangemessenen Bereich ({low}–{high})."
        ),
        "spindle_caption": (
            "Spindeln sind 11–16 Hz transiente Oszillationen, kritisch für "
            "Gedächtniskonsolidierung im NREM-2-Schlaf."
        ),

        "bg_header": "Hintergrund-EEG-Leistung",
        "bg_pdr": "Posteriore Grundaktivität",
        "bg_dar": "Delta/Alpha-Quotient",
        "bg_pdr_normative": "Altersangemessener Bereich: {low}–{high} Hz",
        "bg_dar_normative": "Wach: typischerweise < 1.0",
        "bg_severely_slow": (
            "Posteriore Grundaktivität ist **deutlich langsamer** als "
            "altersangemessen. Marker für kortikale Unreife oder Dysfunktion."
        ),
        "bg_mildly_slow": (
            "Posteriore Grundaktivität leicht langsamer als altersangemessen."
        ),
        "bg_appropriate": (
            "Posteriore Grundaktivität ist altersangemessen."
        ),

        "bursts_header": "Anhaltende rhythmische Bursts auf {channel}",
        "bursts_total": "Bursts gesamt (≥3s)",
        "bursts_5s": "Bursts ≥5s",
        "bursts_10s": "Bursts ≥10s",
        "bursts_max": "Längster (s)",
        "bursts_longest_label": "**Längste gefundene Bursts:**",
        "bursts_warn_long": (
            "**{count} Bursts ≥10 Sekunden** gefunden. Können subklinische "
            "elektrographische Ereignisse darstellen — mit dem behandelnden "
            "Neuropädiater besprechen."
        ),

        "morph_header": "Spike-Morphologie auf {channel}",
        "morph_simple": "Einfache Spikes (<70ms)",
        "morph_sharp": "Sharp Waves (70–200ms)",
        "morph_complex": "Komplexe Spike-Wave (≥200ms)",
        "morph_events": "**Erkannte Ereignisse:** {n} ({rate:.1f}/min)",
        "morph_polyspike": (
            "**Polyspike-Anteil:** {pct:.0f}% (Ereignisse <250ms entfernt)"
        ),
        "morph_complex_classification": (
            "Mehrheit der Ereignisse sind **komplexe Spike-Wave**-Komplexe. "
            "Morphologie passt zu atypischen Absencen / CSWS-Spektrum."
        ),
        "morph_simple_classification": (
            "Mehrheit der Ereignisse sind **einfache Spikes**. Morphologie "
            "passt zu klassischen fokalen Mustern (z.B. Rolandic)."
        ),
        "morph_mixed": "Gemischte Spike-Morphologie.",

        "raw_header": "Roh-Analyse-Output (JSON)",
        "raw_caption": (
            "Herunterladen und ggf. mit dem Arzt teilen."
        ),
        "raw_download": "💾 Befunde als JSON herunterladen",

        "step4_header": "4. KI-Interpretation (optional)",
        "ai_need_key": (
            "Geben Sie Ihren **{provider}** API-Schlüssel in der Seitenleiste "
            "ein, um eine verständliche Interpretation zu erhalten. Es werden "
            "nur numerische Befunde gesendet — Roh-EEG verlässt Ihren "
            "Computer nie."
        ),
        "ai_sdk_needed": (
            "Das Paket `{package}` ist nicht installiert. Im Terminal "
            "ausführen: `pip install {package}`, dann App neu starten."
        ),
        "ai_generate_button": "🤖 Interpretation mit {provider} erzeugen",
        "ai_thinking": "{provider} interpretiert die Befunde...",
        "ai_failed": "KI-Interpretation fehlgeschlagen: {error}",
        "ai_generated_by": "_Erzeugt von {provider}_",
        "ai_download_md": "💾 Interpretation als Markdown herunterladen",

        "compare_summary_header": "Was sich zwischen den beiden Aufnahmen verändert hat",
        "compare_metric_pre": "Vorher",
        "compare_metric_post": "Nachher",
        "compare_metric_change": "Veränderung",
        "compare_label_topography": "Spike-Topographie",
        "compare_label_spindles": "Schlafspindel-Dichte",
        "compare_label_background": "Hintergrund-Verlangsamung",
        "compare_label_bursts": "Anhaltende Bursts",
        "compare_label_morphology": "Spike-Morphologie",
        "compare_no_change": "keine Veränderung",
        "compare_improved": "↓ verbessert",
        "compare_worsened": "↑ verschlechtert",
        "compare_increased": "↑ erhöht",
        "compare_decreased": "↓ gesunken",
        "compare_unchanged": "≈ unverändert",
        "compare_ai_button": "🤖 Veränderungs-Interpretation mit {provider}",
        "compare_ai_caption": (
            "Die KI vergleicht die Zahlen vor und nach Behandlung und erklärt, "
            "was klinisch wahrscheinlich relevant ist."
        ),

        "pdf_header": "5. PDF-Berichte",
        "pdf_caption": (
            "Druckfähige Zusammenfassung erstellen. Zwei Versionen: für die "
            "Ärztin / den Arzt (technisch) und für Eltern (verständlich)."
        ),
        "pdf_doctor_button": "📄 Arzt-Bericht herunterladen (PDF)",
        "pdf_parent_button": "📄 Eltern-Bericht herunterladen (PDF)",

        "tab_time_of_night": "Tagesverlauf",
        "topomap_title": "Spike-Aktivitäts-Topographie (mediane Kurtosis)",
        "topomap_caption": (
            "Heiße Stellen zeigen, wo epileptiforme Aktivität konzentriert ist. "
            "Rot = mehr Aktivität, Blau = weniger. Cz/Pz-Rot deutet auf "
            "zentro-parietale Mittellinien-Beteiligung (SMA / Sprechmotor-Netzwerk)."
        ),
        "ton_header": "Spike-Belastung über die Nacht",
        "ton_caption": (
            "Spike-Anzahl pro Minute in 30-Minuten-Fenstern. Spitzen im ersten "
            "NREM-Zyklus (typischerweise 1–3 Stunden nach Schlafbeginn) sind "
            "charakteristisch für schlaf-aktivierte Muster wie CSWS/ESES."
        ),
        "ton_peak_label": "Spitze: {peak:.1f}/min bei {hours:.1f}h",

        "auto_detect_button": "🔍 Schlaf-Fenster automatisch erkennen",
        "auto_detect_success": (
            "Erkanntes Schlaf-Fenster: {start:.1f}h–{end:.1f}h "
            "({duration:.1f}h, Konfidenz: {conf})"
        ),
        "auto_detect_low_conf": (
            "⚠ Niedrige Konfidenz — bitte das erkannte Fenster manuell prüfen. "
            "Übernacht-Aufnahmen mit viel Artefakt oder ungewöhnlicher "
            "Schlafarchitektur können die Heuristik verwirren."
        ),
        "auto_detect_failed": "Auto-Erkennung fehlgeschlagen: {error}",

        "tab_quality": "Qualität",
        "qc_header": "Aufnahme-Qualität",
        "qc_grade": "Gesamt-Note",
        "qc_usable_epochs": "Nutzbare Epochen",
        "qc_good_channels": "Saubere Kanäle",
        "qc_flagged_channels_label": "Markierte Kanäle:",
        "qc_no_warnings": "Keine Qualitäts-Probleme erkannt.",

        "tab_insights": "🧭 Insights",
        "insights_header": "Proaktive Insights",
        "insights_caption": (
            "Regelbasierte Interpretation der Befunde. Zeigt betroffene "
            "Hirn-Netzwerke, mögliche klinische Muster und cross-modale "
            "Beobachtungen. Alles deterministisch — keine KI hier."
        ),
        "insights_anatomy_header": "Betroffene Hirnregionen",
        "insights_anatomy_caption": (
            "Top-Kanäle mit höchster epileptiformer Aktivität, übersetzt in "
            "die zugrundeliegenden Hirnregionen und funktionalen Netzwerke."
        ),
        "insights_top_networks_header": "Hauptsächlich betroffene funktionale Netzwerke",
        "insights_artifact_warning": (
            "⚠ Hinweis: {channels} sind frontale Pol-Elektroden, die durch "
            "Augenblinzeln-Artefakt beeinflusst sein können. Erhöhte Aktivität "
            "dort sollte vorsichtig interpretiert werden."
        ),
        "insights_patterns_header": "Mögliche klinische Muster",
        "insights_patterns_caption": (
            "Muster-Matches basierend auf Befund-Kombinationen. **Das sind "
            "KEINE Diagnosen** — sondern Ausgangspunkte für das Gespräch "
            "mit der Neuropädiaterin / dem Neuropädiater."
        ),
        "insights_pattern_criteria_met": "Erfüllte Kriterien ({n}/{total}):",
        "insights_pattern_questions": "Empfohlene Fragen an die Ärztin / den Arzt:",
        "insights_cross_modal_header": "Cross-modale Beobachtungen",
        "insights_cross_modal_caption": (
            "Befund-Kombinationen, die mehr aussagen als jeder Einzel-Befund."
        ),
        "insights_no_patterns": (
            "Keine klinischen Muster mit mittlerer Konfidenz erkannt. Das "
            "kann bedeuten: Befunde im Normalbereich, oder die Muster-"
            "Bibliothek deckt diese Konstellation noch nicht ab."
        ),

        "tab_clinical": "Klinik",
        "clinical_header": "Klinische Kennzahlen (v0.5)",
        "clinical_caption": (
            "Zahlen, die Ärztinnen / Ärzte gezielt sehen wollen: formaler SWI "
            "pro Schlafstadium, Wach-/Schlaf-Aktivierungsfaktor, Spike-"
            "Ausbreitungsmuster."
        ),
        "swi_header": "Spike-Wave Index (pro Schlafstadium)",
        "swi_caption": (
            "SWI = % des Stadiums mit kontinuierlicher Spike-Wave-Aktivität. "
            "CSWS / ESES-Kriterium (Tassinari): N3-SWI ≥ 85%."
        ),
        "swi_csws_met": "⚠ CSWS-Kriterium **ERFÜLLT** (N3-SWI ≥ {threshold}%)",
        "swi_csws_not_met": "CSWS-Kriterium nicht erfüllt.",
        "state_split_header": "Wach vs. Schlaf Spike-Raten",
        "state_split_caption": (
            "Getrennte Spike-Raten pro Zustand. Aktivierungsfaktor "
            "(NREM/Wach) ≥ 3 = Schlaf-Aktivierung; ≥ 10 = dramatisch."
        ),
        "synchrony_header": "Bilaterale Synchronie / Ausbreitungsmuster",
        "synchrony_caption": (
            "Wie jeder erkannte Spike sich über die Kopfhaut ausbreitet."
        ),
        "sleep_stages_header": "Schlafarchitektur",
        "sleep_stages_caption": (
            "Heuristische Stadien-Klassifikation. YASA-Modell ist auf "
            "Erwachsenen-PSG trainiert — pädiatrische Ausgabe näherungsweise."
        ),
    },
}
