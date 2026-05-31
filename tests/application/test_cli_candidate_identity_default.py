from pathlib import Path


def test_cli_default_candidate_identity_mode_is_topology_native() -> None:
    source = Path("src/cli.py").read_text(encoding="utf-8")

    assert (
        'candidate_identity_mode=training_cfg.get("candidate_identity_mode", "topology_native")'
        in source
    )
    assert (
        'candidate_identity_mode=training_cfg.get("candidate_identity_mode", "fixed_vocab_bridge")'
        not in source
    )
