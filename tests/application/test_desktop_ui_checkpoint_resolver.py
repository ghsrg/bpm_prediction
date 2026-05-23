from __future__ import annotations

from pathlib import Path

from tools.desktop_ui.checkpoint_resolver import (
    DEFAULT_CHECKPOINT_VALUE,
    resolve_checkpoint_candidates,
)


def test_default_checkpoint_value_is_checkpoint_directory():
    assert DEFAULT_CHECKPOINT_VALUE == "checkpoints/"


def test_resolver_finds_checkpoint_by_experiment_name(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    expected = checkpoint_root / "S_Str_UNc_train_best.pth"
    expected.write_text("checkpoint", encoding="utf-8")
    (checkpoint_root / "Other_best.pth").write_text("checkpoint", encoding="utf-8")

    candidates = resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="S_Str_UNc_train",
    )

    assert candidates == [expected]


def test_resolver_returns_sorted_multiple_candidates(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    later = checkpoint_root / "RunA_last.pth"
    best = checkpoint_root / "RunA_best.pth"
    later.write_text("checkpoint", encoding="utf-8")
    best.write_text("checkpoint", encoding="utf-8")

    candidates = resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="RunA",
    )

    assert candidates == [best, later]


def test_resolver_returns_empty_when_no_match(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "Other_best.pth").write_text("checkpoint", encoding="utf-8")

    assert resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="RunA",
    ) == []

