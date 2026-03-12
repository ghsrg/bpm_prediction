"""Model factory and registry for baseline and EOPKG architectures."""

from __future__ import annotations

from typing import Callable, Dict, Type

from src.domain.models.base_gnn import BaseGNN


MODEL_REGISTRY: Dict[str, Type[BaseGNN]] = {}


def register_model(name: str) -> Callable[[Type[BaseGNN]], Type[BaseGNN]]:
    """Register model class by human-readable config key."""

    def _decorator(model_cls: Type[BaseGNN]) -> Type[BaseGNN]:
        MODEL_REGISTRY[name] = model_cls
        return model_cls

    return _decorator


def get_registered_models() -> Dict[str, Type[BaseGNN]]:
    """Return immutable-like copy of registered models."""
    return dict(MODEL_REGISTRY)


def create_model(model_type: str, **kwargs) -> BaseGNN:
    """Instantiate model by registered type name."""
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported model.type '{model_type}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    return model_cls(**kwargs)


# Register baseline models.
from src.domain.models.baseline_gat import BaselineGATv2  # noqa: E402
from src.domain.models.baseline_gcn import BaselineGCN  # noqa: E402

register_model("BaselineGCN")(BaselineGCN)
register_model("BaselineGATv2")(BaselineGATv2)

# Register EOPKG models via decorators declared in eopkg_models.py.
from src.domain.models import eopkg_models as _eopkg_models  # noqa: E402,F401

