"""Channel → brain region → function mapping.

This module translates per-channel EEG findings into structured anatomical and
functional insights. The goal: turn "Cz median kurtosis = 7.1" into
"epileptiform activity localized to the supplementary motor area, which
plans speech and complex movements."

All anatomical claims are based on standard 10-20 system source-localization
literature. They are approximations — true source localization requires dense
EEG or MEG and individual MRI co-registration. For a family-facing tool,
these descriptions are accurate enough to support a conversation with a
clinician.

Network groupings reflect well-established large-scale brain networks:

- Speech motor network (SMA, pre-SMA, motor cortex) — speech production
- Language network (left temporal + inferior frontal) — language comprehension/production
- Executive network (DLPFC, anterior cingulate) — attention, working memory
- Sensorimotor network (rolandic strip) — body movement & sensation
- Salience network (insula, ACC — approximated via Fz/F3/F4) — emotional regulation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ─── Per-channel anatomy ─────────────────────────────────────────────────────

CHANNEL_INFO: dict[str, dict[str, Any]] = {
    "Fp1": {
        "region": "left frontal pole",
        "function": "executive function, attention regulation",
        "artifact_prone": True,
        "networks": ["executive"],
    },
    "Fp2": {
        "region": "right frontal pole",
        "function": "executive function, attention regulation",
        "artifact_prone": True,
        "networks": ["executive"],
    },
    "F3": {
        "region": "left dorsolateral prefrontal cortex",
        "function": "working memory, verbal fluency",
        "networks": ["executive", "language"],
    },
    "F4": {
        "region": "right dorsolateral prefrontal cortex",
        "function": "working memory, spatial reasoning",
        "networks": ["executive"],
    },
    "Fz": {
        "region": "anterior cingulate / medial frontal",
        "function": "attention control, motor planning",
        "networks": ["executive", "speech_motor"],
    },
    "F7": {
        "region": "left inferior frontal (Broca's area approach)",
        "function": "speech production, expressive language",
        "networks": ["language", "speech_motor"],
    },
    "F8": {
        "region": "right inferior frontal",
        "function": "social cognition, emotion regulation",
        "networks": ["salience"],
    },
    "C3": {
        "region": "left primary motor cortex (hand/arm area)",
        "function": "right-side body motor control",
        "networks": ["sensorimotor"],
    },
    "C4": {
        "region": "right primary motor cortex (hand/arm area)",
        "function": "left-side body motor control",
        "networks": ["sensorimotor"],
    },
    "Cz": {
        "region": "supplementary motor area (SMA / pre-SMA)",
        "function": "motor planning, speech-motor programming, action sequencing",
        "networks": ["speech_motor", "sensorimotor"],
        "key_region": "sma",
    },
    "T3": {  # also called T7 in 10-10
        "region": "left middle temporal (auditory + language regions)",
        "function": "language comprehension, auditory processing",
        "networks": ["language"],
    },
    "T4": {  # also called T8
        "region": "right middle temporal",
        "function": "prosody, music processing, social perception",
        "networks": ["salience"],
    },
    "T5": {  # also called P7
        "region": "left posterior temporal (Wernicke's area approach)",
        "function": "language comprehension, semantic processing",
        "networks": ["language"],
        "key_region": "wernicke",
    },
    "T6": {  # also called P8
        "region": "right posterior temporal",
        "function": "facial / social-emotional processing",
        "networks": ["salience"],
    },
    "P3": {
        "region": "left parietal cortex",
        "function": "sensory integration, reading, written language",
        "networks": ["sensorimotor", "language"],
    },
    "P4": {
        "region": "right parietal cortex",
        "function": "spatial attention, visuospatial processing",
        "networks": ["sensorimotor"],
    },
    "Pz": {
        "region": "precuneus / posterior medial parietal",
        "function": "self-referential processing, motor planning, episodic memory",
        "networks": ["speech_motor"],
        "key_region": "precuneus",
    },
    "O1": {
        "region": "left occipital cortex",
        "function": "visual processing",
        "networks": ["visual"],
    },
    "O2": {
        "region": "right occipital cortex",
        "function": "visual processing",
        "networks": ["visual"],
    },
}


# ─── Functional networks ─────────────────────────────────────────────────────

NETWORK_INFO: dict[str, dict[str, Any]] = {
    "speech_motor": {
        "name": "Speech-motor planning network",
        "anatomy": "Supplementary motor area (SMA), pre-SMA, precuneus, "
                   "and inferior-frontal Broca region.",
        "function": (
            "Programs the precise sequence of muscle movements needed to "
            "produce speech sounds. Distinct from speech comprehension "
            "and from general motor control."
        ),
        "clinical_implications": [
            "Childhood Apraxia of Speech (CAS) — preserved comprehension, "
            "impaired motor speech",
            "Difficulty with complex motor sequences (jumping, tongue "
            "protrusion, finger sequencing)",
            "PROMPT (Prompts for Restructuring Oral Muscular Phonetic "
            "Targets) is the speech-therapy method designed for this profile",
        ],
    },
    "language": {
        "name": "Language comprehension network",
        "anatomy": "Left temporal cortex (Wernicke), inferior frontal "
                   "(Broca), and dorsolateral prefrontal areas.",
        "function": (
            "Understanding spoken and written language; word retrieval; "
            "semantic processing."
        ),
        "clinical_implications": [
            "Receptive language delay",
            "Verbal comprehension difficulties",
            "Reading and writing difficulties as they develop",
            "If activity is left-lateralized, language regression risk is higher",
        ],
    },
    "executive": {
        "name": "Executive function network",
        "anatomy": "Dorsolateral prefrontal cortex and anterior cingulate.",
        "function": (
            "Attention regulation, working memory, planning, impulse control, "
            "cognitive flexibility."
        ),
        "clinical_implications": [
            "Short attention span",
            "Behavioral regulation difficulties",
            "Learning style affected",
            "Frustration tolerance lower than peers",
        ],
    },
    "sensorimotor": {
        "name": "Sensorimotor network",
        "anatomy": "Primary motor cortex, primary somatosensory cortex, "
                   "and parietal sensory areas (rolandic strip).",
        "function": (
            "Voluntary movement, fine motor control, touch perception, "
            "body schema."
        ),
        "clinical_implications": [
            "Fine motor delays (handwriting, buttoning)",
            "Coordination difficulties",
            "Sensory processing differences",
            "If active during sleep, can fragment sleep architecture",
        ],
    },
    "salience": {
        "name": "Salience / social-emotional network",
        "anatomy": "Right inferior frontal, right temporal, anterior insula.",
        "function": (
            "Detecting emotionally and socially salient stimuli; emotion "
            "regulation."
        ),
        "clinical_implications": [
            "Emotional reactivity",
            "Difficulty reading social cues",
            "Sensory overload more likely",
        ],
    },
    "visual": {
        "name": "Visual processing network",
        "anatomy": "Occipital cortex.",
        "function": "Visual perception, motion, color, form.",
        "clinical_implications": [
            "Photic / visual triggers possible",
            "Visual learning may be a strength if non-occipital activity",
        ],
    },
}


# ─── Public API ──────────────────────────────────────────────────────────────

@dataclass
class AnatomicalInsight:
    top_channels: list[tuple[str, float]]
    region_descriptions: list[dict[str, str]]  # name, region, function
    network_scores: dict[str, float]            # network_id -> aggregated score
    top_networks: list[dict[str, Any]]          # ranked by score, with metadata
    artifact_prone_warning: list[str]           # channel names flagged as artifact-prone


def analyze_topography(topography_findings: dict) -> AnatomicalInsight:
    """Convert topography findings into anatomical/functional insights.

    Parameters
    ----------
    topography_findings : dict
        Output of summarize_topography(), with key "all_channels".
    """
    channels = topography_findings.get("all_channels", [])
    if not channels:
        return AnatomicalInsight([], [], {}, [], [])

    # Rank channels by median kurtosis
    ranked = sorted(channels, key=lambda c: -c["median"])
    top5 = [(c["name"], c["median"]) for c in ranked[:5]]

    # Region descriptions for top channels
    region_descs = []
    for name, val in top5:
        info = CHANNEL_INFO.get(name, {})
        region_descs.append({
            "name": name,
            "value": val,
            "region": info.get("region", "unknown region"),
            "function": info.get("function", ""),
            "artifact_prone": info.get("artifact_prone", False),
        })

    # Network aggregation — sum median kurtosis of each network's channels
    network_scores: dict[str, float] = {}
    for c in channels:
        info = CHANNEL_INFO.get(c["name"], {})
        for net in info.get("networks", []):
            network_scores[net] = network_scores.get(net, 0.0) + c["median"]
    # Normalize to per-channel mean within each network
    network_counts = {net: 0 for net in network_scores}
    for c in channels:
        for net in CHANNEL_INFO.get(c["name"], {}).get("networks", []):
            network_counts[net] += 1
    network_means = {
        net: (network_scores[net] / network_counts[net]) if network_counts[net] else 0
        for net in network_scores
    }

    # Top 3 networks by mean activity
    top_net_ids = sorted(network_means, key=lambda n: -network_means[n])[:3]
    top_networks = []
    for net_id in top_net_ids:
        info = NETWORK_INFO.get(net_id, {})
        top_networks.append({
            "id": net_id,
            "score": round(network_means[net_id], 2),
            "name": info.get("name", net_id),
            "anatomy": info.get("anatomy", ""),
            "function": info.get("function", ""),
            "clinical_implications": info.get("clinical_implications", []),
        })

    # Artifact-prone warning
    artifact_warnings = []
    for name, val in top5[:3]:
        if CHANNEL_INFO.get(name, {}).get("artifact_prone"):
            artifact_warnings.append(name)

    return AnatomicalInsight(
        top_channels=top5,
        region_descriptions=region_descs,
        network_scores=network_means,
        top_networks=top_networks,
        artifact_prone_warning=artifact_warnings,
    )


def summarize_anatomy(insight: AnatomicalInsight) -> dict:
    return {
        "top_channels": [{"name": n, "value": round(v, 2)} for n, v in insight.top_channels],
        "region_descriptions": insight.region_descriptions,
        "network_scores": {k: round(v, 2) for k, v in insight.network_scores.items()},
        "top_networks": insight.top_networks,
        "artifact_prone_warning": insight.artifact_prone_warning,
    }
