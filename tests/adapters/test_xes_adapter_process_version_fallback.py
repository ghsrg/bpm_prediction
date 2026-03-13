from __future__ import annotations

from src.adapters.ingestion.xes_adapter import _resolve_process_version


def test_resolve_process_version_uses_dataset_name_before_filename():
    version = _resolve_process_version(
        event_candidates=[None, None],
        trace_version=None,
        log_version=None,
        dataset_name="bpi2012",
        file_path="/tmp/some_log.xes",
    )
    assert version == "bpi2012"


def test_resolve_process_version_uses_filename_when_dataset_name_missing():
    version = _resolve_process_version(
        event_candidates=[None],
        trace_version=None,
        log_version=None,
        dataset_name=None,
        file_path="/tmp/some_log.xes",
    )
    assert version == "some_log"
