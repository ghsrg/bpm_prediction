from __future__ import annotations

import torch
from torch import nn

from src.application.use_cases.trainer import ModelTrainer


class _DiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed_encoder = nn.Linear(2, 2, bias=False)
        self.struct_xattn_proj = nn.Linear(2, 2, bias=False)
        self.struct_encoder = nn.Linear(2, 2, bias=False)


def test_gradient_norm_groups_separate_observed_struct_xattn_and_structural_params():
    model = _DiagnosticModel()
    loss = (
        model.observed_encoder.weight.sum()
        + 2.0 * model.struct_xattn_proj.weight.sum()
        + 3.0 * model.struct_encoder.weight.sum()
    )
    loss.backward()

    groups = ModelTrainer._parameter_grad_norm_groups(model)

    assert groups["observed"] > 0.0
    assert groups["struct_xattn"] > groups["observed"]
    assert groups["structural"] > groups["struct_xattn"]

