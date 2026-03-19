from __future__ import annotations

import src.infrastructure.repositories.knowledge_graph_repository_factory as kg_factory
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from src.infrastructure.repositories.in_memory_networkx_repository import (
    InMemoryNetworkXRepository,
)
from src.infrastructure.repositories.knowledge_graph_repository_factory import (
    build_knowledge_graph_repository,
    get_knowledge_graph_settings,
)


def test_factory_builds_in_memory_by_default():
    repo = build_knowledge_graph_repository({})
    assert isinstance(repo, InMemoryNetworkXRepository)


def test_factory_builds_file_repo_when_requested(tmp_path):
    cfg = {
        "mapping": {
            "knowledge_graph": {
                "backend": "file",
                "path": str(tmp_path / "kg"),
            }
        }
    }
    repo = build_knowledge_graph_repository(cfg)
    assert isinstance(repo, FileBasedKnowledgeGraphRepository)


def test_factory_normalizes_settings_defaults():
    cfg = {"mapping": {"knowledge_graph": {"backend": "memory"}}}
    settings = get_knowledge_graph_settings(cfg)
    assert settings["backend"] == "in_memory"
    assert settings["path"] == "data/knowledge_graph"
    assert settings["strict_load"] is False
    assert settings["ingest_split"] == "train"


def test_factory_stage3_fail_fast_when_backend_not_configured():
    cfg = {
        "experiment": {"project": "bpm_prediction_mvp2_5", "name": "MVP2_5_Stage3_Run"},
        "model": {"type": "EOPKGGATv2"},
        "mapping": {"adapter": "camunda"},
    }
    try:
        _ = get_knowledge_graph_settings(cfg)
    except ValueError as exc:
        assert "mapping.knowledge_graph" in str(exc)
    else:
        raise AssertionError("Expected fail-fast when backend is not configured for Stage3 context.")


def test_factory_stage3_fail_fast_when_backend_invalid():
    cfg = {
        "experiment": {"project": "bpm_prediction_mvp2_5", "name": "MVP2_5_Stage3_Run"},
        "model": {"type": "EOPKGGATv2"},
        "mapping": {"knowledge_graph": {"backend": "unknown_backend"}},
    }
    try:
        _ = get_knowledge_graph_settings(cfg)
    except ValueError as exc:
        assert "Unsupported mapping.knowledge_graph.backend" in str(exc)
    else:
        raise AssertionError("Expected fail-fast for unsupported backend in Stage3 context.")


def test_factory_stage3_allows_explicit_in_memory_backend():
    cfg = {
        "experiment": {"project": "bpm_prediction_mvp2_5", "name": "MVP2_5_Stage3_Run"},
        "model": {"type": "EOPKGGATv2"},
        "mapping": {"knowledge_graph": {"backend": "in_memory"}},
    }
    settings = get_knowledge_graph_settings(cfg)
    assert settings["backend"] == "in_memory"


def test_factory_stage3_allows_explicit_neo4j_backend(monkeypatch):
    cfg = {
        "experiment": {"project": "bpm_prediction_mvp2_5", "name": "MVP2_5_Stage4_Run"},
        "model": {"type": "EOPKGGATv2"},
        "mapping": {
            "knowledge_graph": {
                "backend": "neo4j",
                "neo4j": {"connection_profile": "local"},
            }
        },
    }

    class _FakeNeo4jRepo:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        kg_factory,
        "resolve_neo4j_connection_settings",
        lambda **kwargs: {
            "uri": "bolt://localhost:7687",
            "database": "neo4j",
            "user": "neo4j",
            "password": "pass",
        },
    )
    monkeypatch.setattr(kg_factory, "Neo4jKnowledgeGraphRepository", _FakeNeo4jRepo)

    settings = get_knowledge_graph_settings(cfg)
    assert settings["backend"] == "neo4j"
    assert settings["neo4j"]["uri"] == "bolt://localhost:7687"

    repo = build_knowledge_graph_repository(cfg)
    assert isinstance(repo, _FakeNeo4jRepo)
    assert repo.kwargs["database"] == "neo4j"
