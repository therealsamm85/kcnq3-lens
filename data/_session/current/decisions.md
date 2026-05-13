# Locked decisions — KCNQ3-Lens

## 2026-05-13 — YASA as default spindle backend, heuristic as fallback
**Decision:** `compute_spindle_density(method="auto")` uses YASA when installed; falls back to the envelope-percentile heuristic only when YASA is unavailable.
**Why:** Side-by-side validation on Liyana's 6h sleep window showed the heuristic over-counted by ~150× (483 vs 3 spindles). The result was robust across signal scalings (5–100 µV target std) and threshold settings. YASA's three-criteria detector (Lacourse 2019) has been validated against expert-scored polysomnograms. Continuing to use the heuristic as default would produce false reassurance for families and could mask genuinely severe spindle deficits.
**Alternatives considered:** keep heuristic as default and offer YASA opt-in (rejected — too easy to miss clinically meaningful findings); remove heuristic entirely (rejected — keeps the tool usable in offline / minimal-install environments).
**Reversibility:** Reversible at any time via the `method` parameter.

## 2026-05-13 — Multi-AI provider abstraction (no single LLM lock-in)
**Decision:** Abstract LLM access through a `LLMProvider` ABC with three concrete implementations (Anthropic, OpenAI, Google) selected at runtime via the UI.
**Why:** Families come with different API-key situations and different language/cost preferences. Locking the tool to one provider would arbitrarily exclude users. The router pattern adds ~50 LOC and zero ongoing maintenance burden — each provider is one file.
**Alternatives considered:** Anthropic-only (rejected — excludes Google AI Studio's free-tier users); LiteLLM proxy (rejected — adds dependency, more failure modes).
**Reversibility:** Reversible by deleting providers/ subdirectory and inlining one provider in router.py.

## 2026-05-13 — Privacy by architecture: raw EEG never leaves the user's machine
**Decision:** All EEG file processing is local. Only derived numerical metrics are sent to any LLM API. No cloud upload, no shared server, no telemetry.
**Why:** EEG files are medical data with strict regional requirements (GDPR in Germany, HIPAA in the US). Building a multi-jurisdictional compliance posture is beyond what a volunteer open-source project can sustain. Local-only sidesteps the entire category of concerns.
**Alternatives considered:** Cloud upload with encryption (rejected — compliance burden); on-prem optional (rejected — out of scope for v1).
**Reversibility:** Hard to reverse — the privacy guarantee is the tool's most defensible positioning claim; revisiting would damage trust.

## 2026-05-13 — Streamlit frontend, Python core (no web framework)
**Decision:** Use Streamlit as the GUI layer. No React, no Electron, no separate frontend repo.
**Why:** Streamlit gives drag-and-drop file upload, plots, and interactivity in <200 LOC of Python per page. The user base (parents + clinicians) is small enough that "install Python and run one command" is acceptable. A web app would require hosting + compliance overhead.
**Alternatives considered:** Electron desktop app (rejected — packaging complexity); Flask + React (rejected — 10× more code for the same UI surface).
**Reversibility:** Reversible by re-wiring `src/` modules to a different frontend. The analysis layer has no Streamlit dependency.

## 2026-05-13 — English as source of truth for i18n, German as first translation
**Decision:** `src/i18n/translations.py` has English strings as the canonical reference. German is a parallel translation. Missing keys fall back to English at runtime.
**Why:** Most open-source EEG documentation is in English; researchers reading the code expect English. But the first user families (Liyana's case in Hamburg) are German-speaking, and the second-language gap matters for accessibility. Two languages now; structure supports easy addition of more.
**Alternatives considered:** Default German (rejected — international contributor friction); gettext-based i18n (rejected — overkill for <200 strings).
**Reversibility:** Reversible by renaming default language and re-pointing fallback.

## 2026-05-13 — README scope section is non-negotiable
**Decision:** README must explicitly state minimum recording requirements, what the tool is NOT for, and a comparison table with other EEG tools (MNE, YASA, EDFbrowser, Persyst, etc.).
**Why:** Without explicit scope, users will apply the tool to unsuitable recordings (adult EEG, too-short recordings, wrong sample rate) and get misleading results. The "what we are NOT" section is what prevents the tool from being a black box that families over-trust. The comparison table credits prior art and prevents reinvention claims.
**Alternatives considered:** "Light touch" scope section in a separate doc (rejected — users don't read separate docs).
**Reversibility:** Reversible but unwise.
