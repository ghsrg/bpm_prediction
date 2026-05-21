"""Compatibility helpers for project-owned PyTorch artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_trusted_torch_artifact(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load a project-owned torch artifact that may contain PyG/Data objects.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    The project's graph caches and checkpoints are local trusted artifacts and
    can contain Python objects such as ``torch_geometric.data.Data``.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
