"""Reference citations for every normative value and clinical criterion.

Pediatric neurologists evaluating an EEG report want to know where a
threshold came from. "Spindle density 3-5/min for age 5" is uncited;
"Spindle density 3-5/min, Wamsley et al. 2012 (PMID 22431760)" is
auditable.

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
    "wamsley_spindles": Citation(
        key="wamsley_spindles",
        short="Wamsley et al. 2012",
        full=(
            "Wamsley EJ, Tucker MA, Shinn AK, et al. Reduced sleep spindles "
            "and spindle coherence in schizophrenia: mechanisms of impaired "
            "memory consolidation? Biol Psychiatry 2012;71:154-61."
        ),
        pubmed_id="22431760",
        url="https://pubmed.ncbi.nlm.nih.gov/22431760/",
        note="Sleep spindle density thresholds and N2-stage normative ranges.",
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
        pubmed_id="14636778",
        url="https://pubmed.ncbi.nlm.nih.gov/14636778/",
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
    }
