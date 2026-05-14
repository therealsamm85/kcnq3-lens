"""Anonymization helper — strip patient identifiers from EEG file headers.

Hospital EEG exports routinely embed patient name, birth date, MRN, and
similar identifiers in the file header. Before a family or clinician
shares a recording with a collaborator, second opinion, or research
registry, those fields should be cleared.

This module currently handles:
- **EDF / EDF+** files (header positions 8–88 contain patient + recording IDs)
- **Nihon Kohden EEG-1200A** files (partial — strips known patient-ID
  locations in the file header; not all NK variants are documented)

OUTPUT: writes a new file with identifiers stripped. Does not modify
the original. Returns the path to the anonymized copy + a list of
fields that were stripped.

LIMITATIONS:
- Only handles fields the maintainers know about. Other proprietary
  formats may embed identifiers elsewhere.
- A determined adversary with access to the raw file might still
  recover identifiers from timing patterns, annotations, etc. This is
  a privacy-helper, not a forensic anonymizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnonymizationResult:
    source_path: Path
    output_path: Path
    fields_stripped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def anonymize_edf(source: Path, output: Path | None = None) -> AnonymizationResult:
    """Strip patient + recording identifiers from an EDF/EDF+ header.

    EDF header layout (first 256 bytes):
      0-7    version
      8-87   local patient identification  (80 bytes)
      88-167 local recording identification (80 bytes)
      168-175 startdate
      176-183 starttime
      ...

    We blank the patient + recording identification fields with 'X' padding
    while keeping the rest of the header intact.
    """
    source = Path(source)
    if output is None:
        output = source.with_name(source.stem + "_anonymized" + source.suffix)
    output = Path(output)

    fields_stripped: list[str] = []
    warnings: list[str] = []

    with open(source, "rb") as fh:
        header = fh.read(256)
        if len(header) < 256:
            warnings.append("File header shorter than 256 bytes — not a valid EDF.")
            return AnonymizationResult(source, output, fields_stripped, warnings)
        body = fh.read()

    # Patient ID: bytes 8-88 — replace with "X X X X" pattern padded to 80 bytes
    new_patient_id = "X X X X".ljust(80, " ").encode("ascii")
    # Recording ID: bytes 88-168 — replace similarly
    new_recording_id = (
        "Startdate X X X X".ljust(80, " ").encode("ascii")
    )

    new_header = (
        header[:8]
        + new_patient_id
        + new_recording_id
        + header[168:]
    )
    fields_stripped.append("patient_identification (bytes 8-88)")
    fields_stripped.append("recording_identification (bytes 88-168)")

    output.write_bytes(new_header + body)
    return AnonymizationResult(source, output, fields_stripped, warnings)


def anonymize_nihon_kohden(
    source: Path, output: Path | None = None
) -> AnonymizationResult:
    """Strip known patient-info locations from a Nihon Kohden EEG-1200A file.

    NK headers embed patient info at file offsets ~0x0030-0x0080 (varies by
    version). We zero-out this conservative range. Waveform data is
    untouched (starts at 0x38E3).

    Caveat: NK headers are partially reverse-engineered. This helper
    handles what the maintainers know; some variants embed identifiers
    elsewhere.
    """
    source = Path(source)
    if output is None:
        output = source.with_name(source.stem + "_anonymized" + source.suffix)
    output = Path(output)

    fields_stripped: list[str] = []
    warnings: list[str] = []

    data = bytearray(source.read_bytes())
    if len(data) < 0x100:
        warnings.append("File smaller than 256 bytes — not a valid NK file.")
        return AnonymizationResult(source, output, fields_stripped, warnings)

    # Conservative zero-fill range for patient info (offsets 0x30 - 0x80)
    # Keeps the file signature at 0x0 - 0x10 intact (otherwise readers fail)
    for i in range(0x30, 0x80):
        data[i] = 0
    fields_stripped.append("patient_info_region (0x30-0x80)")

    warnings.append(
        "NK anonymization is partial — only the documented patient-info "
        "region is stripped. Other identifier locations may exist in "
        "vendor-specific extensions."
    )

    output.write_bytes(bytes(data))
    return AnonymizationResult(source, output, fields_stripped, warnings)


def anonymize_auto(source: Path, output: Path | None = None) -> AnonymizationResult:
    """Auto-detect EDF vs NK and dispatch to the right anonymizer."""
    source = Path(source)
    suffix = source.suffix.lower()

    if suffix in (".edf", ".bdf"):
        return anonymize_edf(source, output)
    if suffix == ".eeg":
        return anonymize_nihon_kohden(source, output)

    return AnonymizationResult(
        source_path=source,
        output_path=source,
        fields_stripped=[],
        warnings=[
            f"Unsupported format for anonymization: {suffix}. "
            "Supported: .edf, .bdf, .eeg (Nihon Kohden)."
        ],
    )
