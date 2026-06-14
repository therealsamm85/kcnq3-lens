"""PDF report generation for KCNQ3-Lens."""

# The PDF builders need reportlab; import them lazily so the pure-stdlib reporting
# modules in this package (score_report, annotations) stay importable in a
# minimal install without reportlab.
try:
    from .pdf import build_doctor_pdf, build_parent_pdf
except ImportError:  # pragma: no cover - exercised only when reportlab is absent
    build_doctor_pdf = build_parent_pdf = None

__all__ = ["build_doctor_pdf", "build_parent_pdf"]
