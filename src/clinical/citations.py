"""Reference citations for every normative value and clinical criterion.

Pediatric neurologists evaluating an EEG report want to know where a
threshold came from. "Spindle density ~1/min at age 5" is uncited;
"Spindle density ~1/min at age 5, McClain et al. 2016 (PMID 27110405)"
is auditable.

Each entry: short label, full citation, PubMed ID (when applicable),
URL, and a one-line note on what the citation supports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    key: str
    short: str
    full: str
    pubmed_id: str | None
    url: str | None
    note: str


CITATIONS: dict[str, Citation] = {
    "tassinari_csws": Citation(
        key="tassinari_csws",
        short="Tassinari 1971",
        full=(
            "Tassinari CA, Bureau M, Dravet C, et al. Epilepsy with continuous "
            "spike-waves during slow sleep. In: Roger J, et al., eds. Epileptic "
            "Syndromes in Infancy, Childhood and Adolescence. 1971."
        ),
        pubmed_id=None,
        url=None,
        note="CSWS / ESES criterion: SWI ≥ 85% during slow-wave sleep.",
    ),
    "mcclain_spindles": Citation(
        key="mcclain_spindles",
        short="McClain et al. 2016",
        full=(
            "McClain IJ, Lustenberger C, Achermann P, Lassonde JM, Kurth S, "
            "LeBourgeois MK. Developmental Changes in Sleep Spindle "
            "Characteristics and Sigma Power across Early Childhood. "
            "Neural Plasticity 2016;2016:3670951."
        ),
        pubmed_id="27110405",
        url="https://pubmed.ncbi.nlm.nih.gov/27110405/",
        note=(
            "Pediatric sleep-spindle density at central derivations "
            "(~1/min at age 5 in NREM2)."
        ),
    ),
    "kwon_spindles": Citation(
        key="kwon_spindles",
        short="Kwon et al. 2023",
        full=(
            "Kwon H, Walsh KG, Berja ED, et al. Sleep spindles in the "
            "healthy brain from birth through 18 years. Sleep "
            "2023;46(4):zsad017."
        ),
        pubmed_id="36719044",
        url="https://pubmed.ncbi.nlm.nih.gov/36719044/",
        note=(
            "Large-cohort developmental spindle norms (n=567, ages 0-18). "
            "Near-linear rate increase from ~3y to ~14y, plateau after."
        ),
    ),
    "lacourse_yasa": Citation(
        key="lacourse_yasa",
        short="Lacourse et al. 2019",
        full=(
            "Lacourse K, Delfrate J, Beaudry J, et al. A sleep spindle "
            "detection algorithm that emulates human expert scorers. "
            "J Neurosci Methods 2019;316:3-11."
        ),
        pubmed_id="30107208",
        url="https://pubmed.ncbi.nlm.nih.gov/30107208/",
        note="YASA spindle-detection algorithm (corr / rel_pow / rms criteria).",
    ),
    "vallat_yasa": Citation(
        key="vallat_yasa",
        short="Vallat & Walker 2021",
        full=(
            "Vallat R, Walker MP. An open-source, high-performance tool for "
            "automated sleep staging. eLife 2021;10:e70092."
        ),
        pubmed_id="34648426",
        url="https://pubmed.ncbi.nlm.nih.gov/34648426/",
        note="YASA SleepStaging model.",
    ),
    "hagne_pdr": Citation(
        key="hagne_pdr",
        short="Hagne 1972",
        full=(
            "Hagne I. Development of the waking EEG in normal infants during "
            "the first year of life. In: Kellaway P, Petersen I, eds. "
            "Clinical Electroencephalography of Children. Stockholm: "
            "Almqvist & Wiksell; 1968:97-118."
        ),
        pubmed_id=None,
        url=None,
        note="Pediatric posterior dominant rhythm developmental norms.",
    ),
    "niedermeyer": Citation(
        key="niedermeyer",
        short="Niedermeyer & Lopes da Silva 2005",
        full=(
            "Niedermeyer E, Lopes da Silva F. Electroencephalography: Basic "
            "Principles, Clinical Applications, and Related Fields. 5th ed. "
            "Lippincott Williams & Wilkins; 2005."
        ),
        pubmed_id=None,
        url=None,
        note="Standard reference for normative EEG, all ages.",
    ),
    "binnie_tci": Citation(
        key="binnie_tci",
        short="Binnie 2003",
        full=(
            "Binnie CD. Cognitive impairment during epileptiform discharges: "
            "is it ever justifiable to treat the EEG? Lancet Neurol "
            "2003;2:725-30."
        ),
        pubmed_id="14636777",
        url="https://pubmed.ncbi.nlm.nih.gov/14636777/",
        note="Transient cognitive impairment from interictal discharges.",
    ),
    "mne_python": Citation(
        key="mne_python",
        short="Gramfort et al. 2013",
        full=(
            "Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis "
            "with MNE-Python. Front Neurosci 2013;7:267."
        ),
        pubmed_id="24431986",
        url="https://pubmed.ncbi.nlm.nih.gov/24431986/",
        note="MNE-Python — file reading and signal processing primitives.",
    ),
    "massimini_sw": Citation(
        key="massimini_sw",
        short="Massimini et al. 2004",
        full=(
            "Massimini M, Huber R, Ferrarelli F, Hill S, Tononi G. "
            "The sleep slow oscillation as a traveling wave. "
            "J Neurosci 2004;24(31):6862-6870."
        ),
        pubmed_id="15282274",
        url="https://pubmed.ncbi.nlm.nih.gov/15282274/",
        note=(
            "Foundational slow-wave detection criteria: negative peak, "
            "positive peak, and peak-to-peak amplitude thresholds."
        ),
    ),
    "carrier_sw_dev": Citation(
        key="carrier_sw_dev",
        short="Carrier et al. 2011",
        full=(
            "Carrier J, Viens I, Poirier G, Robillard R, Lafortune M, "
            "Vandewalle G, Martin N, Barakat M, Paquet J, Filipini D. "
            "Sleep slow wave changes during the middle years of life. "
            "Eur J Neurosci 2011;33(4):758-766."
        ),
        pubmed_id="20813192",
        url="https://pubmed.ncbi.nlm.nih.gov/20813192/",
        note=(
            "Age-related decline in slow-wave amplitude and density across "
            "middle adulthood. No pediatric norms included."
        ),
    ),
    "kurth_pediatric_sw": Citation(
        key="kurth_pediatric_sw",
        short="Kurth et al. 2010",
        full=(
            "Kurth S, Ringli M, Geiger A, LeBourgeois M, Jenni OG, "
            "Huber R. Mapping of cortical activity in the first two decades "
            "of life: a high-density sleep electroencephalogram study. "
            "J Neurosci 2010;30(40):13211-13219."
        ),
        pubmed_id="20534927",
        url="https://pubmed.ncbi.nlm.nih.gov/20534927/",
        note=(
            "Developmental topography of slow-wave activity from childhood "
            "to early adulthood (ages 2–20, high-density EEG). "
            "Shows frontal predominance and developmental trajectory."
        ),
    ),
    "staba_hfo": Citation(
        key="staba_hfo",
        short="Staba et al. 2002",
        full=(
            "Staba RJ, Wilson CL, Bragin A, Fried I, Engel J Jr. "
            "Quantitative analysis of high-frequency oscillations (80-500 Hz) "
            "recorded in human epileptic hippocampus and entorhinal cortex. "
            "J Neurophysiol 2002;88(4):1743-1752."
        ),
        pubmed_id="12239031",
        url="https://pubmed.ncbi.nlm.nih.gov/12239031/",
        note=(
            "Original energy-based HFO detector; ripple (80–250 Hz) and "
            "fast-ripple (250–500 Hz) bands defined; intracranial recording."
        ),
    ),
    "burnos_hfo": Citation(
        key="burnos_hfo",
        short="Burnos et al. 2014",
        full=(
            "Burnos S, Frauscher B, Zelmann R, Haegelen C, Sarnthein J, "
            "Gotman J. Human intracranial high frequency oscillations (HFOs) "
            "detected during NREM sleep mirror the seizure onset zone. "
            "Clin Neurophysiol 2014;125(3):532-540."
        ),
        pubmed_id="24747572",
        url="https://pubmed.ncbi.nlm.nih.gov/24747572/",
        note=(
            "Frequency-specificity criterion: true HFOs have ripple-band power "
            "≥2× high-band power; rejects broad-band transients."
        ),
    ),
    "kuhnke_scalp_hfo": Citation(
        key="kuhnke_scalp_hfo",
        short="Kuhnke et al. 2018",
        full=(
            "Kuhnke N, Schönberger E, Rebenklau R, Kiess W, Merkenschlager A, "
            "Bernhard MK. Spike ripples and fast ripples in childhood epilepsy "
            "with centrotemporal spikes. Clin Neurophysiol 2018;129(12):2450-2456."
        ),
        pubmed_id="30215099",
        url="https://pubmed.ncbi.nlm.nih.gov/30215099/",
        note=(
            "Scalp HFOs in childhood epilepsy with centrotemporal spikes; "
            "demonstrates feasibility of scalp-level ripple detection in pediatric EEG."
        ),
    ),
    "helfrich_coupling": Citation(
        key="helfrich_coupling",
        short="Helfrich et al. 2018",
        full=(
            "Helfrich RF, Mander BA, Jagust WJ, Knight RT, Walker MP. "
            "Old Brains Come Uncoupled in Sleep: Slow Wave-Spindle Synchrony, "
            "Brain Atrophy, and Forgetting. Neuron 2018;97(1):221-230.e4."
        ),
        pubmed_id="29395264",
        url="https://pubmed.ncbi.nlm.nih.gov/29395264/",
        note=(
            "SO-spindle coupling and memory consolidation in aging; "
            "validates PLV as coupling metric in adults."
        ),
    ),
    "hahn_coupling_pediatric": Citation(
        key="hahn_coupling_pediatric",
        short="Hahn et al. 2020",
        full=(
            "Hahn M, Joechner AK, Roell J, Volbert SH, Gruber G, Holz J, "
            "Schabus M, Wilhelm I, Born J, Werkle-Bergner M. "
            "Slow oscillation-spindle coupling predicts enhanced memory formation "
            "from childhood to adolescence. Elife 2020;9:e53730."
        ),
        pubmed_id="32499637",
        url="https://pubmed.ncbi.nlm.nih.gov/32499637/",
        note=(
            "Paediatric SO-spindle coupling, ages 8–19. PLV rises from ~0.15 "
            "(age 8) toward adult levels during adolescence. "
            "No validated cutoffs below age 8."
        ),
    ),
}


def get(key: str) -> Citation | None:
    return CITATIONS.get(key)


def get_short(key: str) -> str:
    c = CITATIONS.get(key)
    return c.short if c else key


def all_citations() -> list[Citation]:
    return list(CITATIONS.values())


def methods_attribution() -> dict[str, str]:
    """Map each analysis name to its primary citation short-form.

    Used when generating the methods section of the PDF report.
    """
    return {
        "spindles": "lacourse_yasa",
        "sleep_stages": "vallat_yasa",
        "swi": "tassinari_csws",
        "background": "hagne_pdr",
        "morphology": "binnie_tci",   # for cognitive-relevance framing
        "slow_waves": "massimini_sw",
        "hfo_ripples": "staba_hfo",
        "coupling": "helfrich_coupling",
    }
