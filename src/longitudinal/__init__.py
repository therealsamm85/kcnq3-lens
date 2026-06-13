"""Longitudinal tracking — multi-recording trends + development diary."""

from .storage import (
    StoredEntry, save_entry, load_all_entries, delete_entry,
    default_storage_dir,
)
from .diary import (
    DiaryEntry, append_entry, load_all_entries as load_diary,
    to_table as diary_to_table, default_diary_path,
)
from .trends import build_trends_table, get_metric_series, METRICS
from .treatment_response import (
    compute_treatment_response, summarize_treatment_response,
    render_treatment_response_md,
)
from .word_correlation import (
    compute_word_correlation, summarize_word_correlation,
    render_word_correlation_md,
)

__all__ = [
    "StoredEntry", "save_entry", "load_all_entries", "delete_entry",
    "default_storage_dir",
    "DiaryEntry", "append_entry", "load_diary",
    "diary_to_table", "default_diary_path",
    "build_trends_table", "get_metric_series", "METRICS",
    "compute_treatment_response", "summarize_treatment_response",
    "render_treatment_response_md",
    "compute_word_correlation", "summarize_word_correlation",
    "render_word_correlation_md",
]
