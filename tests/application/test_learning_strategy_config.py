import pytest

from src.application.services.learning_strategy_config import (
    FusionSupportLevel,
    LearningStrategyConfig,
)


def test_learning_strategy_defaults_to_standard():
    cfg = LearningStrategyConfig.from_training_config({})

    assert cfg.learning_strategy == "standard"
    assert cfg.is_topology_conditioned is False
    assert cfg.allowed_set_loss_enabled is False
    assert cfg.wrong_version_negative_enabled is False
    assert cfg.drop_edges_negative_enabled is False


def test_topology_conditioned_parses_coefficients_and_support_level():
    cfg = LearningStrategyConfig.from_training_config(
        {
            "learning_strategy": "topology_conditioned",
            "topology_conditioning_allowed_set_loss_enabled": "true",
            "topology_conditioning_allowed_set_loss_weight": "0.25",
            "topology_conditioning_wrong_version_negative_enabled": True,
            "topology_conditioning_wrong_version_negative_weight": "0.30",
            "topology_conditioning_drop_edges_negative_enabled": True,
            "topology_conditioning_drop_edges_ratio": "0.40",
        },
        fusion_mode="StructXAttn",
        structural_mode=True,
    )

    assert cfg.learning_strategy == "topology_conditioned"
    assert cfg.is_topology_conditioned is True
    assert cfg.allowed_set_loss_enabled is True
    assert cfg.allowed_set_loss_weight == pytest.approx(0.25)
    assert cfg.wrong_version_negative_weight == pytest.approx(0.30)
    assert cfg.drop_edges_ratio == pytest.approx(0.40)
    assert cfg.fusion_support_level == FusionSupportLevel.SUPPORTED


def test_topology_conditioned_rejects_structural_mode_false():
    with pytest.raises(ValueError, match="requires structural_mode=true"):
        LearningStrategyConfig.from_training_config(
            {"learning_strategy": "topology_conditioned"},
            fusion_mode="ClassAwareStructuralScoring",
            structural_mode=False,
        )


def test_topology_conditioned_rejects_legacy_struct_xattn_contrastive():
    with pytest.raises(ValueError, match="struct_xattn_contrastive_enabled"):
        LearningStrategyConfig.from_training_config(
            {
                "learning_strategy": "topology_conditioned",
                "struct_xattn_contrastive_enabled": True,
            },
            fusion_mode="StructXAttn",
            structural_mode=True,
        )

