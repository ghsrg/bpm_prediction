"""Model factory and registry for baseline and EOPKG architectures."""

from __future__ import annotations

import inspect
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
    signature = inspect.signature(model_cls.__init__)
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_var_kw:
        model_kwargs = kwargs
    else:
        allowed = {name for name in signature.parameters.keys() if name != "self"}
        model_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return model_cls(**model_kwargs)


# Register baseline models.
from src.domain.models.baseline_gat import BaselineGATv2  # noqa: E402
from src.domain.models.baseline_gcn import BaselineGCN  # noqa: E402

register_model("BaselineGCN")(BaselineGCN)
register_model("BaselineGATv2")(BaselineGATv2)

# Register EOPKG models via decorators declared in eopkg_models.py.
from src.domain.models import eopkg_models as _eopkg_models  # noqa: E402,F401
