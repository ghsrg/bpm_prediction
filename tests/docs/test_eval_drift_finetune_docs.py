from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_eval_drift_finetune_docs_describe_valid_adaptive_protocol():
    evf_text = (ROOT / "docs" / "EVF_MVP2_5.MD").read_text(encoding="utf-8")
    runbook_text = (ROOT / "docs" / "runbooks" / "mvp2_5-commands.md").read_text(encoding="utf-8")

    assert "one-pass baseline" in evf_text
    assert "newly observed stride" in evf_text
    assert "finetune_optimizer_steps" in runbook_text
    assert "incomplete or ineffective update" in runbook_text
