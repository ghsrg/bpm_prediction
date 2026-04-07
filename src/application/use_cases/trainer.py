"""Application use-case for MVP1 model training and evaluation."""

# Р’С–РґРїРѕРІС–РґРЅРѕ РґРѕ:
# - ARCHITECTURE_RULES.MD -> СЂРѕР·РґС–Р» 2-4 (Application orchestration С‡РµСЂРµР· РїРѕСЂС‚Рё)
# - EVF_MVP1.MD -> СЂРѕР·РґС–Р» 3 (Strict Temporal Split), СЂРѕР·РґС–Р» 4 (С†С–Р»СЊРѕРІС– РјРµС‚СЂРёРєРё), СЂРѕР·РґС–Р» 5 (tracking)
# - AGENT_GUIDE.MD -> СЂРѕР·РґС–Р» 2 (MVP1 scope, Р±РµР· Critic/Reliability)

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from datetime import datetime, timezone
import logging
import math
import random
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score
from torch import nn
from torch.nn.parameter import UninitializedParameter
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from src.application.ports.graph_builder_port import IGraphBuilder
from src.application.ports.prefix_policy_port import IPrefixPolicy
from src.application.ports.tracker_port import ITracker
from src.application.ports.xes_adapter_port import IXESAdapter
from src.domain.entities.raw_trace import RawTrace
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN
from src.infrastructure.runtime.progress_events import ProgressReporter, emit_progress_event, progress_events_enabled


logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for chronologically split trace subsets."""

    train: List[RawTrace]
    val: List[RawTrace]
    test: List[RawTrace]


class ShardedGraphDataset(Dataset[Data]):
    """Lazy on-disk dataset backed by shard files produced by CLI graph cache."""

    def __init__(self, *, entry_dir: str, shards: Sequence[Dict[str, Any]], max_cached_shards: int = 2) -> None:
        self.entry_dir = Path(str(entry_dir))
        self._shards: List[Dict[str, Any]] = []
        self._offsets: List[int] = []
        total = 0
        for row in shards:
            if not isinstance(row, dict):
                continue
            rel_path = str(row.get("path", "")).strip()
            count = int(row.get("count", 0))
            if not rel_path or count <= 0:
                continue
            self._shards.append({"path": rel_path, "count": int(count)})
            total += int(count)
            self._offsets.append(total)
        self._size = int(total)
        self._max_cached_shards = max(1, int(max_cached_shards))
        self._cached_shards: Dict[int, List[Data]] = {}
        self._cache_lru_order: List[int] = []

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], *, max_cached_shards: int = 2) -> "ShardedGraphDataset":
        return cls(
            entry_dir=str(payload.get("entry_dir", "")),
            shards=payload.get("shards", []),
            max_cached_shards=max_cached_shards,
        )

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Data:
        if index < 0 or index >= self._size:
            raise IndexError(index)
        shard_idx = int(bisect_right(self._offsets, index))
        prev_total = self._offsets[shard_idx - 1] if shard_idx > 0 else 0
        local_idx = index - prev_total
        shard = self._load_shard(shard_idx)
        return shard[local_idx]

    def _load_shard(self, shard_idx: int) -> List[Data]:
        cached = self._cached_shards.get(shard_idx)
        if isinstance(cached, list):
            try:
                self._cache_lru_order.remove(shard_idx)
            except ValueError:
                pass
            self._cache_lru_order.append(shard_idx)
            return cached
        row = self._shards[shard_idx]
        shard_path = self.entry_dir / str(row["path"])
        loaded = torch.load(shard_path, map_location="cpu")
        if not isinstance(loaded, list):
            raise ValueError(f"Invalid shard payload type for {shard_path}.")
        self._cached_shards[shard_idx] = loaded
        self._cache_lru_order.append(shard_idx)
        while len(self._cache_lru_order) > self._max_cached_shards:
            evict_idx = self._cache_lru_order.pop(0)
            self._cached_shards.pop(evict_idx, None)
        return loaded

    def shard_index_ranges(self) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        start = 0
        for end in self._offsets:
            ranges.append((start, int(end)))
            start = int(end)
        return ranges


class _ShardAwareRandomSampler(Sampler[int]):
    """Shuffle by shard then by index within shard to avoid random disk seeks."""

    def __init__(self, dataset: ShardedGraphDataset, *, seed: int = 42) -> None:
        self._dataset = dataset
        self._base_seed = int(seed)
        self._iteration = 0

    def __iter__(self) -> Iterable[int]:
        rng = random.Random(self._base_seed + self._iteration)
        self._iteration += 1
        shard_ranges = self._dataset.shard_index_ranges()
        shard_ids = list(range(len(shard_ranges)))
        rng.shuffle(shard_ids)
        for shard_idx in shard_ids:
            start, end = shard_ranges[shard_idx]
            indices = list(range(int(start), int(end)))
            rng.shuffle(indices)
            for idx in indices:
                yield idx

    def __len__(self) -> int:
        return len(self._dataset)


class ModelTrainer:
    """End-to-end trainer for MVP1 next-activity prediction baseline."""

    def __init__(
        self,
        xes_adapter: IXESAdapter,
        prefix_policy: IPrefixPolicy,
        graph_builder: IGraphBuilder,
        model: BaseGNN,
        log_path: str,
        config: Dict[str, Any],
        tracker: Optional[ITracker] = None,
        class_weights: Optional[torch.Tensor] = None,
        prepared_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.xes_adapter = xes_adapter
        self.prefix_policy = prefix_policy
        self.graph_builder = graph_builder
        self.model = model
        self.log_path = log_path
        self.config = config
        self.tracker = tracker
        self.class_weights = class_weights
        self.prepared_data = dict(prepared_data) if isinstance(prepared_data, dict) else None

        self.batch_size = int(config.get("batch_size", 32))
        self.epochs = int(config.get("epochs", 10))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.device = torch.device(config.get("device", "cpu"))
        self.num_ece_bins = int(config.get("ece_bins", 10))
        self.class_weight_cap = float(config.get("class_weight_cap", 50.0))
        self.show_progress = bool(config.get("show_progress", True))
        self.tqdm_disable = bool(config.get("tqdm_disable", False))
        self.tqdm_leave = bool(config.get("tqdm_leave", False))
        self._structured_progress_enabled = bool(progress_events_enabled())
        self.dataloader_num_workers = max(0, int(config.get("dataloader_num_workers", 0)))
        self.dataloader_pin_memory = bool(config.get("dataloader_pin_memory", False))
        self.dataloader_persistent_workers = bool(
            config.get("dataloader_persistent_workers", self.dataloader_num_workers > 0)
        )
        self.dataloader_prefetch_factor = max(2, int(config.get("dataloader_prefetch_factor", 2)))
        self.sharded_dataset_cached_shards = max(1, int(config.get("sharded_dataset_cached_shards", 2)))
        self.loss_function = str(config.get("loss_function", "cross_entropy")).strip().lower()
        self.patience = int(config.get("patience", 6))
        self.delta = float(config.get("delta", 1e-4))
        self.config_path = str(config.get("config_path", "")).strip()
        self.seed = int(config.get("seed", 42))
        self.retrain = bool(config.get("retrain", False))
        self.mode = str(config.get("mode", self.config.get("experiment_config", {}).get("mode", "train"))).strip().lower()
        self.drift_window_size = int(config.get("drift_window_size", self.config.get("experiment_config", {}).get("drift_window_size", 500)))
        self.drift_window_sliding = int(config.get("drift_window_sliding", self.config.get("experiment_config", {}).get("drift_window_sliding", 0) or 0))
        self.mask_guided_enabled = bool(config.get("mask_guided_enabled", False))
        self.mask_guided_apply_in_eval = bool(config.get("mask_guided_apply_in_eval", True))
        self.mask_guided_hard_threshold = float(config.get("mask_guided_hard_threshold", 1.0))
        self.mask_guided_soft_penalty = float(config.get("mask_guided_soft_penalty", 2.0))
        self.mask_guided_min_samples_for_hard = max(1, int(config.get("mask_guided_min_samples_for_hard", 1)))

        self.checkpoint_dir = str(config.get("checkpoint_dir", "checkpoints")).strip() or "checkpoints"
        checkpoint_override = str(config.get("checkpoint_path", "")).strip()
        experiment_cfg = self.config.get("experiment_config", {})
        experiment_name = str(experiment_cfg.get("name", "default_experiment")).strip() or "default_experiment"
        safe_experiment_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in experiment_name)
        self.experiment_name = safe_experiment_name
        self.checkpoint_path = Path(checkpoint_override) if checkpoint_override else (Path(self.checkpoint_dir) / f"{self.experiment_name}_best.pth")
        resume_checkpoint_override = str(config.get("resume_checkpoint_path", "")).strip()
        self.resume_checkpoint_path = (
            Path(resume_checkpoint_override)
            if resume_checkpoint_override
            else self._derive_last_checkpoint_path(self.checkpoint_path)
        )
        self.last_checkpoint_path = self._derive_last_checkpoint_path(self.checkpoint_path)

        self._topology_nodes: List[int] = []
        self._topology_edges: List[int] = []
        self._prefix_lengths: List[int] = []
        self._version_to_idx: Dict[str, int] = {}
        self._idx_to_version: Dict[int, str] = {}
        self._stats_snapshot_version_to_idx: Dict[str, int] = {}
        self._idx_to_stats_snapshot_version: Dict[int, str] = {}
        self._warned_mixed_snapshot_batches: set[tuple[int, ...]] = set()
        self._logged_mask_debug = False
        self._logged_oos_math = False
        self._logged_peak_vram = False
        self._mask_guided_reliability_rate: float | None = None
        self._mask_guided_reliability_samples = 0

        if self.prepared_data is not None:
            raw_idx_to_version = self.prepared_data.get("idx_to_version", {})
            if isinstance(raw_idx_to_version, dict):
                normalized_idx_to_version: Dict[int, str] = {}
                for raw_idx, raw_label in raw_idx_to_version.items():
                    try:
                        normalized_idx_to_version[int(raw_idx)] = str(raw_label)
                    except (TypeError, ValueError):
                        continue
                self._idx_to_version.update(normalized_idx_to_version)
                self._version_to_idx.update({label: idx for idx, label in self._idx_to_version.items()})
            raw_snapshot_idx_to_version = self.prepared_data.get("idx_to_stats_snapshot_version", {})
            if isinstance(raw_snapshot_idx_to_version, dict):
                normalized_snapshot_idx_to_version: Dict[int, str] = {}
                for raw_idx, raw_label in raw_snapshot_idx_to_version.items():
                    try:
                        normalized_snapshot_idx_to_version[int(raw_idx)] = str(raw_label)
                    except (TypeError, ValueError):
                        continue
                self._idx_to_stats_snapshot_version.update(normalized_snapshot_idx_to_version)
                self._stats_snapshot_version_to_idx.update(
                    {label: idx for idx, label in self._idx_to_stats_snapshot_version.items()}
                )

    def _is_tqdm_disabled(self) -> bool:
        return (not self.show_progress) or self.tqdm_disable or self._structured_progress_enabled

    def run(self) -> Dict[str, Any]:
        """Execute full training flow: data prep, train/val loop, and final test."""
        pipeline_reporter = ProgressReporter(stage="run.pipeline", total=1, min_interval_sec=0.5)
        pipeline_reporter.start(message=f"Pipeline mode={self.mode}", current=0, total=1)
        self.model.to(self.device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
            self.class_weights = torch.clamp(self.class_weights, max=self.class_weight_cap)
        if self.loss_function != "cross_entropy":
            raise ValueError(f"Unsupported training.loss_function '{self.loss_function}' for MVP1.")

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self._log_params()
        self._log_run_context()
        self._log_data_metrics()
        if self.tracker is not None and self.config_path:
            self.tracker.log_artifact(self.config_path)

        is_eval_cross = self.mode == "eval_cross_dataset"
        is_eval_drift = self.mode == "eval_drift"
        is_eval_mode = is_eval_cross or is_eval_drift
        prebuilt_datasets = None
        if self.prepared_data is not None:
            logger.info("Using preloaded prepared data from CLI (single-read mode).")
            prepared_traces_raw = self.prepared_data.get("prepared_traces", [])
            prepared_traces = list(prepared_traces_raw) if prepared_traces_raw is not None else []
            train_traces_raw = self.prepared_data.get("train_traces", [])
            val_traces_raw = self.prepared_data.get("val_traces", [])
            test_traces_raw = self.prepared_data.get("test_traces", [])
            if all(item is not None for item in (train_traces_raw, val_traces_raw, test_traces_raw)):
                split_data = SplitData(
                    train=list(train_traces_raw),
                    val=list(val_traces_raw),
                    test=list(test_traces_raw),
                )
            else:
                split_data = self._prepare_split_data(prepared_traces)

            train_dataset = self.prepared_data.get("train_dataset")
            val_dataset = self.prepared_data.get("val_dataset")
            test_dataset = self.prepared_data.get("test_dataset")
            if all(item is not None for item in (train_dataset, val_dataset, test_dataset)):
                train_payload = train_dataset if isinstance(train_dataset, dict) else list(train_dataset)
                val_payload = val_dataset if isinstance(val_dataset, dict) else list(val_dataset)
                test_payload = test_dataset if isinstance(test_dataset, dict) else list(test_dataset)
                prebuilt_datasets = {
                    "train": train_payload,
                    "val": val_payload,
                    "test": test_payload,
                }
        else:
            raw_traces = self._read_raw_traces_with_progress()
            prepared_traces = self._prepare_data(raw_traces, mode=self.mode)
            split_data = self._prepare_split_data(prepared_traces)

        checkpoint, start_epoch, best_val_loss, best_epoch = self._prepare_checkpoint_state(is_eval_mode=is_eval_mode)

        if is_eval_cross:
            result = self._run_eval_cross_dataset(
                split_data=split_data,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                prebuilt_test_dataset=(prebuilt_datasets or {}).get("test") if prebuilt_datasets else None,
            )
            pipeline_reporter.done(message=f"Pipeline completed mode={self.mode}", current=1, total=1)
            return result

        if is_eval_drift:
            result = self._run_eval_drift(
                split_data=split_data,
                drift_traces=prepared_traces,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
            )
            pipeline_reporter.done(message=f"Pipeline completed mode={self.mode}", current=1, total=1)
            return result

        result = self._run_train_pipeline(
            split_data=split_data,
            checkpoint=checkpoint,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            prebuilt_datasets=prebuilt_datasets,
        )
        pipeline_reporter.done(message=f"Pipeline completed mode={self.mode}", current=1, total=1)
        return result

    def _read_raw_traces_with_progress(self) -> List[RawTrace]:
        """Read raw traces from adapter with heartbeat logging for long operations."""
        logger.info("Reading XES file...")
        started = perf_counter()
        traces: List[RawTrace] = []
        event_count = 0
        read_reporter = ProgressReporter(stage="prepare.read_events", min_interval_sec=0.8)
        read_reporter.start(message=f"Reading events via {self.xes_adapter.__class__.__name__}")

        iterator = self.xes_adapter.read(self.log_path, self.config.get("mapping_config", {}))
        progress = tqdm(
            desc="Read traces",
            unit="trace",
            leave=self.tqdm_leave,
            disable=self._is_tqdm_disabled(),
        )
        try:
            for idx, trace in enumerate(iterator, start=1):
                traces.append(trace)
                event_count += len(trace.events)
                progress.update(1)
                read_reporter.update(
                    message=f"Reading events via {self.xes_adapter.__class__.__name__}",
                    current=int(idx),
                    total=None,
                    payload={"events": int(event_count)},
                    force=(idx == 1),
                )
                if idx % 1000 == 0:
                    progress.set_postfix({"events": event_count})
                    logger.info("Read %d traces (%d events) so far...", idx, event_count)
        finally:
            progress.close()

        duration = perf_counter() - started
        logger.info("Finished reading XES: traces=%d, events=%d, duration=%.2fs", len(traces), event_count, duration)
        read_reporter.done(
            message="Event read finished",
            current=int(len(traces)),
            total=int(len(traces)),
            payload={"events": int(event_count), "duration_sec": float(duration)},
        )
        return traces

    def _prepare_checkpoint_state(self, is_eval_mode: bool) -> Tuple[Optional[Dict[str, Any]], int, float, int]:
        """Load checkpoint according to active scenario and return state tuple."""
        checkpoint: Optional[Dict[str, Any]] = None
        start_epoch = 0
        best_val_loss = float("inf")
        best_epoch = 0

        if is_eval_mode:
            if not self.checkpoint_path.exists():
                raise ValueError(f"Checkpoint required for mode '{self.mode}' was not found: {self.checkpoint_path}")
            checkpoint = self._load_checkpoint(self.checkpoint_path, require_encoder_state=True)
            self._restore_from_checkpoint(checkpoint, require_encoder_state=True)
            best_val_loss = float(checkpoint["val_loss"])
            best_epoch = int(checkpoint["epoch"])
        elif not self.retrain:
            resume_path = self._resolve_resume_checkpoint_path()
            if resume_path is None:
                return checkpoint, start_epoch, best_val_loss, best_epoch
            checkpoint = self._load_checkpoint(resume_path, require_encoder_state=False)
            self._restore_from_checkpoint(checkpoint, require_encoder_state=False)
            start_epoch = int(checkpoint["epoch"])
            best_val_loss = float(checkpoint.get("best_val_loss", checkpoint["val_loss"]))
            best_epoch = int(checkpoint.get("best_epoch", start_epoch))
            logger.info(
                "Loaded resume checkpoint from %s at epoch %d (best_epoch=%d, best_val_loss=%.6f).",
                resume_path,
                start_epoch,
                best_epoch,
                best_val_loss,
            )

        return checkpoint, start_epoch, best_val_loss, best_epoch

    def _run_eval_cross_dataset(
        self,
        split_data: SplitData,
        best_epoch: int,
        best_val_loss: float,
        prebuilt_test_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run eval_cross_dataset scenario with preloaded checkpoint state."""
        test_loader = (
            self._build_loader_from_dataset(prebuilt_test_dataset, shuffle=False)
            if prebuilt_test_dataset is not None
            else self._build_loader(split_data.test, shuffle=False)
        )
        self._log_topology_metrics_and_artifacts()
        logger.info("DataLoaders ready for eval_cross_dataset: test_batches=%d", len(test_loader))
        test_metrics = self._evaluate_test(test_loader, stage_label="inference")
        logger.info("Eval cross-dataset test metrics: %s", test_metrics)
        self._log_test_metrics(test_metrics)
        return {
            "history": [],
            "test_metrics": test_metrics,
            "split_sizes": {"train": len(split_data.train), "val": len(split_data.val), "test": len(split_data.test)},
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "mode": self.mode,
        }

    def _run_eval_drift(self, split_data: SplitData, drift_traces: Sequence[RawTrace], best_epoch: int, best_val_loss: float) -> Dict[str, Any]:
        """Run eval_drift scenario over chronologically sorted windows."""
        drift_metrics = self._evaluate_drift_windows(drift_traces)
        self._log_topology_metrics_and_artifacts()
        logger.info("Eval drift windows completed: windows=%d", len(drift_metrics))
        return {
            "history": [],
            "test_metrics": {},
            "drift_metrics": drift_metrics,
            "split_sizes": {"train": len(split_data.train), "val": len(split_data.val), "test": len(split_data.test)},
            "drift_trace_count": len(drift_traces),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "mode": self.mode,
        }

    def _run_train_pipeline(
        self,
        *,
        split_data: SplitData,
        checkpoint: Optional[Dict[str, Any]],
        start_epoch: int,
        best_val_loss: float,
        best_epoch: int,
        prebuilt_datasets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run train/validation/test scenario with optional resume checkpoint."""
        has_prebuilt = bool(
            prebuilt_datasets
            and all(prebuilt_datasets.get(key) is not None for key in ("train", "val", "test"))
        )
        if has_prebuilt:
            train_loader = self._build_loader_from_dataset(prebuilt_datasets.get("train"), shuffle=True)
            val_loader = self._build_loader_from_dataset(prebuilt_datasets.get("val"), shuffle=False)
            test_loader = self._build_loader_from_dataset(prebuilt_datasets.get("test"), shuffle=False)
        else:
            train_loader = self._build_loader(split_data.train, shuffle=True)
            val_loader = self._build_loader(split_data.val, shuffle=False)
            test_loader = self._build_loader(split_data.test, shuffle=False)
        self._log_topology_metrics_and_artifacts()

        logger.info(
            "DataLoaders ready: train_batches=%d, val_batches=%d, test_batches=%d",
            len(train_loader),
            len(val_loader),
            len(test_loader),
        )
        emit_progress_event(
            stage="trainer.dataloaders",
            status="done",
            message="DataLoaders ready",
            payload={
                "train_batches": int(len(train_loader)),
                "validation_batches": int(len(val_loader)),
                "test_batches": int(len(test_loader)),
            },
        )

        self._perform_dry_run(train_loader, context_label="train")
        self._log_model_and_system_context()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if checkpoint is not None and start_epoch < self.epochs:
            optimizer_state_dict = checkpoint.get("optimizer_state_dict")
            if optimizer_state_dict is None:
                raise ValueError("Checkpoint is missing required key 'optimizer_state_dict' for resume training.")
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            logger.info("Starting training from scratch, start_epoch = 0.")

        history: List[Dict[str, float]] = []
        epochs_without_improvement = 0
        epochs_reporter = ProgressReporter(stage="train.epochs", total=int(self.epochs), min_interval_sec=0.5)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if start_epoch >= self.epochs:
            logger.info("Model already trained for %d epochs. Skipping training loop.", self.epochs)
            epochs_reporter.start(message="Training epochs started", current=int(self.epochs), total=int(self.epochs))
            epochs_reporter.done(
                message="Training skipped: checkpoint already at target epoch",
                current=int(self.epochs),
                total=int(self.epochs),
            )
        else:
            epochs_reporter.start(message="Training epochs started", current=start_epoch, total=int(self.epochs))
            for epoch in range(start_epoch + 1, self.epochs + 1):
                def _update_epoch_progress(batch_idx: int, total_batches: int, phase_name: str) -> None:
                    if total_batches <= 0:
                        return
                    phase_ratio = float(batch_idx) / float(total_batches)
                    phase_offset = 0.0 if phase_name == "train" else 0.5
                    phase_span = 0.5
                    epoch_current = float(epoch - 1) + phase_offset + (phase_span * phase_ratio)
                    epochs_reporter.update(
                        message=f"Epoch {epoch}/{self.epochs} | {phase_name} {batch_idx}/{total_batches}",
                        current=epoch_current,
                        total=int(self.epochs),
                        force=(batch_idx == 1 or batch_idx == total_batches),
                    )

                train_loss, train_macro_f1, _, epoch_duration = self._run_epoch(
                    train_loader,
                    optimizer=optimizer,
                    training=True,
                    epoch_index=epoch,
                    total_epochs=self.epochs,
                    epoch_progress_callback=_update_epoch_progress,
                )
                val_loss, val_macro_f1, val_weighted_f1, _ = self._run_epoch(
                    val_loader,
                    optimizer=None,
                    training=False,
                    epoch_index=epoch,
                    total_epochs=self.epochs,
                    epoch_progress_callback=_update_epoch_progress,
                )

                epoch_metrics = {
                    "train_loss": train_loss,
                    "train_macro_f1": train_macro_f1,
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "val_weighted_f1": val_weighted_f1,
                    "epoch_duration_sec": epoch_duration,
                }
                history.append(epoch_metrics)
                self._log_epoch_metrics(epoch=epoch, metrics=epoch_metrics)
                logger.info(
                    "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | Train F1: %.4f | Val F1: %.4f",
                    epoch,
                    self.epochs,
                    train_loss,
                    val_loss,
                    train_macro_f1,
                    val_macro_f1,
                )
                epochs_reporter.update(
                    message=f"Epoch {epoch}/{self.epochs}",
                    current=int(epoch),
                    total=int(self.epochs),
                    payload={
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "train_f1": float(train_macro_f1),
                        "val_f1": float(val_macro_f1),
                    },
                )

                if self.device.type == "cuda" and torch.cuda.is_available() and not self._logged_peak_vram:
                    peak_vram_mb = float(torch.cuda.max_memory_allocated() / 1048576.0)
                    if self.tracker is not None:
                        self.tracker.log_param("peak_vram_mb", peak_vram_mb)
                    self._logged_peak_vram = True

                improvement = best_val_loss - val_loss
                if improvement > self.delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_checkpoint(
                        self.checkpoint_path,
                        epoch=epoch,
                        val_loss=best_val_loss,
                        optimizer=optimizer,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                    )
                else:
                    epochs_without_improvement += 1
                self._save_checkpoint(
                    self.last_checkpoint_path,
                    epoch=epoch,
                    val_loss=val_loss,
                    optimizer=optimizer,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                )

                if epochs_without_improvement >= self.patience:
                    logger.info("Early stopping triggered at epoch %d (best_epoch=%d, best_val_loss=%.6f).", epoch, best_epoch, best_val_loss)
                    epochs_reporter.done(
                        level="warning",
                        message=f"Early stopping at epoch {epoch}",
                        current=int(epoch),
                        total=int(self.epochs),
                        payload={"best_epoch": int(best_epoch), "best_val_loss": float(best_val_loss)},
                    )
                    if self.tracker is not None:
                        self.tracker.log_metric("early_stopping_epoch", float(epoch), step=epoch)
                        self.tracker.log_metric("best_epoch", float(best_epoch), step=epoch)
                        self.tracker.log_metric("best_val_loss", float(best_val_loss), step=epoch)
                    break
            else:
                epochs_reporter.done(
                    message="Training epochs completed",
                    current=int(self.epochs),
                    total=int(self.epochs),
                )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {self.checkpoint_path}")

        best_checkpoint = self._load_checkpoint(self.checkpoint_path, require_encoder_state=False)
        self.model.load_state_dict(best_checkpoint["model_state_dict"])
        self._restore_encoder_state(best_checkpoint, require_encoder_state=False)
        best_epoch = int(best_checkpoint["epoch"])
        best_val_loss = float(best_checkpoint["val_loss"])

        test_eval_reporter = ProgressReporter(stage="test.eval", total=1, min_interval_sec=0.2)
        test_eval_reporter.start(message="Test evaluation started", current=0, total=1)
        test_metrics = self._evaluate_test(test_loader, stage_label="inference")
        logger.info("Final test metrics: %s", test_metrics)
        test_eval_reporter.done(
            message="Test evaluation completed",
            current=1,
            total=1,
            payload={
                "test_macro_f1": float(test_metrics.get("test_macro_f1", 0.0)),
                "test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
            },
        )
        self._log_test_metrics(test_metrics)

        if self.tracker is not None:
            self.tracker.log_model(self.model, "best_model")
            self.tracker.log_metric("best_epoch", float(best_epoch), step=self.epochs)
            self.tracker.log_metric("best_val_loss", float(best_val_loss), step=self.epochs)

        return {
            "history": history,
            "test_metrics": test_metrics,
            "split_sizes": {"train": len(split_data.train), "val": len(split_data.val), "test": len(split_data.test)},
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "mode": self.mode,
        }

    def _perform_dry_run(self, loader: Iterable[Data], context_label: str) -> None:
        """Initialize lazy modules by one forward pass before optimizer/model-size logging."""
        logger.info("Performing dry run to initialize Lazy modules (%s)...", context_label)
        emit_progress_event(stage="trainer.dry_run", status="start", message=f"Dry run started ({context_label})")
        was_training = self.model.training
        self.model.train()
        try:
            iterator = iter(loader)
            dummy_batch = next(iterator, None)
            if dummy_batch is None:
                logger.warning("Dry run skipped (%s): loader is empty.", context_label)
                emit_progress_event(stage="trainer.dry_run", status="done", level="warning", message=f"Dry run skipped ({context_label})")
                return
            dummy_batch = dummy_batch.to(self.device)
            contract = self._data_to_contract(dummy_batch)
            with torch.no_grad():
                _ = self.model(contract)
            logger.info("Dry run successful (%s). Model initialized.", context_label)
            emit_progress_event(stage="trainer.dry_run", status="done", message=f"Dry run completed ({context_label})", current=1, total=1)
        except Exception as exc:
            logger.error("Dry run failed (%s): %s", context_label, exc)
            emit_progress_event(stage="trainer.dry_run", status="error", level="error", message=f"Dry run failed: {exc}")
            raise
        finally:
            self.model.train(was_training)

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        val_loss: float,
        optimizer: Adam,
        *,
        best_epoch: Optional[int] = None,
        best_val_loss: Optional[float] = None,
    ) -> None:
        """Persist best model checkpoint for future resume/evaluation flows."""
        checkpoint_payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss),
        }
        if best_epoch is not None:
            checkpoint_payload["best_epoch"] = int(best_epoch)
        if best_val_loss is not None and math.isfinite(float(best_val_loss)):
            checkpoint_payload["best_val_loss"] = float(best_val_loss)
        feature_encoder = getattr(self.graph_builder, "feature_encoder", None)
        if feature_encoder is not None and hasattr(feature_encoder, "get_state"):
            checkpoint_payload["encoder_state"] = feature_encoder.get_state()
        if self.tracker is not None and hasattr(self.tracker, "get_run_id"):
            try:
                run_id = getattr(self.tracker, "get_run_id")()
            except Exception:  # noqa: BLE001
                run_id = None
            if isinstance(run_id, str) and run_id.strip():
                checkpoint_payload["mlflow_run_id"] = run_id.strip()
        torch.save(checkpoint_payload, checkpoint_path)

    @staticmethod
    def _derive_last_checkpoint_path(checkpoint_path: Path) -> Path:
        """Derive stable last-training checkpoint path from best/effective checkpoint path."""
        name = checkpoint_path.name
        if name.endswith("_best.pth"):
            return checkpoint_path.with_name(f"{name[:-9]}_last.pth")
        stem = checkpoint_path.stem
        suffix = checkpoint_path.suffix or ".pth"
        if stem.endswith("_best"):
            return checkpoint_path.with_name(f"{stem[:-5]}_last{suffix}")
        return checkpoint_path.with_name(f"{stem}_last{suffix}")

    def _resolve_resume_checkpoint_path(self) -> Optional[Path]:
        """Resolve training-resume checkpoint path, preferring explicit resume/last checkpoint."""
        candidate_paths: List[Path] = []
        if isinstance(self.resume_checkpoint_path, Path):
            candidate_paths.append(self.resume_checkpoint_path)
        if isinstance(self.last_checkpoint_path, Path):
            candidate_paths.append(self.last_checkpoint_path)
        candidate_paths.append(self.checkpoint_path)

        seen: set[str] = set()
        for candidate in candidate_paths:
            normalized = str(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            if candidate.exists():
                return candidate
        return None

    def _load_checkpoint(self, checkpoint_path: Path, require_encoder_state: bool) -> Dict[str, Any]:
        """Load checkpoint with required keys validation."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint payload must be a dictionary.")

        required_keys = {"epoch", "model_state_dict", "val_loss"}
        missing_keys = required_keys.difference(checkpoint.keys())
        if missing_keys:
            raise ValueError(f"Checkpoint is missing required keys: {sorted(missing_keys)}")

        if require_encoder_state and "encoder_state" not in checkpoint:
            raise ValueError("Checkpoint is missing required key 'encoder_state'.")

        return checkpoint

    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any], require_encoder_state: bool) -> None:
        """Restore model and feature encoder states from checkpoint."""
        self._materialize_lazy_model_modules(checkpoint)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            logger.error(
                "Checkpoint architecture mismatch while loading model state. "
                "checkpoint_path=%s model_class=%s error=%s",
                self.checkpoint_path,
                self.model.__class__.__name__,
                exc,
            )
            raise ValueError(
                "Checkpoint architecture mismatch! You are trying to load a checkpoint that belongs to a "
                f"different model architecture.\n"
                f"Current model: {self.model.__class__.__name__}\n"
                f"Checkpoint: {self.checkpoint_path}\n"
                f"Details: {exc}\n"
                "FIX: Either change 'experiment.name' in config, set 'training.retrain: true', "
                "or manually delete the old checkpoint."
            ) from exc
        self._restore_encoder_state(checkpoint, require_encoder_state=require_encoder_state)

    def _materialize_lazy_model_modules(self, checkpoint: Dict[str, Any]) -> None:
        """Materialize lazily-created model modules that may already exist in checkpoint state."""
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            return

        # EOPKGGATv2 creates struct_input_proj lazily on first forward when struct_x dim != struct_hidden_dim.
        # For resume flows we need the module in place before strict state_dict loading.
        has_lazy_proj = hasattr(self.model, "struct_input_proj")
        current_proj = getattr(self.model, "struct_input_proj", None) if has_lazy_proj else None
        if not has_lazy_proj or current_proj is not None:
            return

        weight = model_state.get("struct_input_proj.weight")
        bias = model_state.get("struct_input_proj.bias")
        if not isinstance(weight, torch.Tensor):
            return
        if weight.ndim != 2:
            raise ValueError("Invalid checkpoint tensor shape for struct_input_proj.weight; expected 2D tensor.")

        out_features = int(weight.shape[0])
        in_features = int(weight.shape[1])
        use_bias = isinstance(bias, torch.Tensor)

        layer = nn.Linear(in_features, out_features, bias=use_bias).to(device=self.device, dtype=weight.dtype)
        setattr(self.model, "struct_input_proj", layer)
        logger.info(
            "Materialized lazy module struct_input_proj from checkpoint shape=(out=%d,in=%d).",
            out_features,
            in_features,
        )

    def _restore_encoder_state(self, checkpoint: Dict[str, Any], require_encoder_state: bool) -> None:
        """Restore encoder state into graph_builder.feature_encoder when available."""
        encoder_state = checkpoint.get("encoder_state")
        if encoder_state is None:
            if require_encoder_state:
                raise ValueError("Checkpoint does not contain required encoder_state payload.")
            return

        feature_encoder = getattr(self.graph_builder, "feature_encoder", None)
        if feature_encoder is None or not hasattr(feature_encoder, "load_state"):
            raise ValueError("Graph builder does not expose feature_encoder.load_state required for checkpoint restore.")
        feature_encoder.load_state(encoder_state)

    def _prepare_split_data(self, traces: Sequence[RawTrace]) -> SplitData:
        """Apply micro split into train/val/test on already prepared trace stream."""
        if self.mode in {"eval_drift", "eval_cross_dataset"}:
            logger.info("Eval mode [%s]: bypassing micro split; routing all %d traces to test scope.", self.mode, len(traces))
            return SplitData(train=[], val=[], test=list(traces))
        experiment_cfg = self.config.get("experiment_config", {})
        split_strategy = str(experiment_cfg.get("split_strategy", "temporal")).strip().lower()
        split_ratio = self._parse_split_ratio(experiment_cfg.get("split_ratio", [0.7, 0.2, 0.1]))
        if split_strategy == "time":
            split_strategy = "temporal"
        logger.info("Micro split train/val/test with ratio %s.", list(split_ratio))
        return self._strict_temporal_split(ordered=traces, split_ratio=split_ratio, split_strategy=split_strategy)

    def _prepare_data(self, traces: Sequence[RawTrace], mode: str) -> List[RawTrace]:
        """Apply safe cascade filter: temporal sort -> macro split -> fraction."""
        experiment_cfg = self.config.get("experiment_config", {})
        split_strategy = str(experiment_cfg.get("split_strategy", "temporal")).strip().lower()
        if split_strategy == "time":
            split_strategy = "temporal"
        if split_strategy not in {"temporal", "none"}:
            raise ValueError("experiment.split_strategy must be 'temporal' or 'none'.")

        train_ratio = float(experiment_cfg.get("train_ratio", 0.7))
        if train_ratio < 0.0 or train_ratio > 1.0:
            raise ValueError("experiment.train_ratio must be within [0.0, 1.0].")

        fraction = float(experiment_cfg.get("fraction", 1.0))
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("experiment.fraction must be within (0.0, 1.0].")

        traces_with_events = [trace for trace in traces if trace.events]
        if split_strategy == "temporal":
            logger.info("Found %d traces. Applying chronological sort.", len(traces_with_events))
            ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)
        else:
            logger.info("Found %d traces. split_strategy=none, keeping original order.", len(traces_with_events))
            ordered = list(traces_with_events)

        split_idx = int(len(ordered) * train_ratio)
        mode_label = mode.strip().lower()
        if mode_label == "train":
            macro = ordered[:split_idx]
            logger.info(
                "Macro split mode [train]: using %.2f%% of data (%d traces).",
                train_ratio * 100.0,
                len(macro),
            )
        elif mode_label in {"eval_drift", "eval_cross_dataset"}:
            macro = ordered[split_idx:]
            logger.info(
                "Macro split mode [%s]: using %.2f%% tail data (%d traces).",
                mode_label,
                (1.0 - train_ratio) * 100.0,
                len(macro),
            )
        else:
            macro = ordered
            logger.info("Macro split mode [%s]: no cut applied, kept %d traces.", mode_label, len(macro))

        keep_count = int(len(macro) * fraction)
        filtered = macro if fraction >= 1.0 else macro[:keep_count]
        logger.info("Applied fraction=%.4f. Kept %d traces.", fraction, len(filtered))
        return filtered

    def _parse_split_ratio(self, raw_ratio: Any) -> Tuple[float, float, float]:
        """Validate configured train/val/test split ratio."""
        if not isinstance(raw_ratio, Sequence) or isinstance(raw_ratio, (str, bytes)) or len(raw_ratio) != 3:
            raise ValueError("experiment.split_ratio must be a 3-item list [train, val, test].")
        ratios = [float(item) for item in raw_ratio]
        if any(item < 0.0 for item in ratios):
            raise ValueError("experiment.split_ratio must contain non-negative values.")
        ratio_sum = sum(ratios)
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"experiment.split_ratio must sum to 1.0, got {ratio_sum:.6f}")
        return ratios[0], ratios[1], ratios[2]

    def _strict_temporal_split(self, ordered: Sequence[RawTrace], split_ratio: Tuple[float, float, float], split_strategy: str) -> SplitData:
        """Apply deterministic micro split by configured strategy and ratio."""
        if split_strategy not in {"temporal", "none"}:
            raise ValueError(f"Unsupported experiment.split_strategy '{split_strategy}'.")
        total = len(ordered)
        train_ratio, val_ratio, _ = split_ratio
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        return SplitData(train=list(ordered[:train_end]), val=list(ordered[train_end:val_end]), test=list(ordered[val_end:]))

    def _build_loader(self, traces: Sequence[RawTrace], shuffle: bool) -> DataLoader:
        """Convert traces into graph tensors and wrap them into PyG DataLoader."""
        graphs: List[Data] = []
        for trace in traces:
            slices = self.prefix_policy.generate_slices(trace)
            for prefix_slice in slices:
                contract = self.graph_builder.build_graph(prefix_slice)
                num_nodes = int(contract["num_nodes"])
                num_edges = int(contract["edge_index"].shape[1])
                prefix_length = int(len(prefix_slice.prefix_events))
                version_label = str(prefix_slice.process_version)
                version_idx = self._version_to_idx.setdefault(version_label, len(self._version_to_idx))
                self._idx_to_version[version_idx] = version_label
                self._topology_nodes.append(num_nodes)
                self._topology_edges.append(num_edges)
                self._prefix_lengths.append(prefix_length)
                payload = {
                    "x_cat": contract["x_cat"],
                    "x_num": contract["x_num"],
                    "edge_index": contract["edge_index"],
                    "edge_type": contract["edge_type"],
                    "y": contract["y"],
                    "num_nodes": num_nodes,
                    "prefix_len": torch.tensor([prefix_length], dtype=torch.long),
                    "process_version_idx": torch.tensor([version_idx], dtype=torch.long),
                }
                allowed_mask = contract.get("allowed_target_mask")
                if isinstance(allowed_mask, torch.Tensor):
                    payload["allowed_target_mask"] = allowed_mask.unsqueeze(0) if allowed_mask.dim() == 1 else allowed_mask
                struct_x = contract.get("struct_x")
                if isinstance(struct_x, torch.Tensor):
                    payload["struct_x"] = struct_x
                structural_edge_index = contract.get("structural_edge_index")
                if isinstance(structural_edge_index, torch.Tensor):
                    payload["structural_edge_index"] = structural_edge_index
                structural_edge_weight = contract.get("structural_edge_weight")
                if isinstance(structural_edge_weight, torch.Tensor):
                    payload["structural_edge_weight"] = structural_edge_weight
                snapshot_seq_raw = contract.get("stats_snapshot_version_seq")
                if isinstance(snapshot_seq_raw, (int, float)) and not isinstance(snapshot_seq_raw, bool):
                    snapshot_idx = int(snapshot_seq_raw)
                    snapshot_version = f"k{snapshot_idx:06d}"
                    self._stats_snapshot_version_to_idx.setdefault(snapshot_version, snapshot_idx)
                    self._idx_to_stats_snapshot_version[snapshot_idx] = snapshot_version
                    payload["stats_snapshot_version_idx"] = torch.tensor([snapshot_idx], dtype=torch.long)
                else:
                    # Backward compatibility for contracts that still expose text metadata.
                    stats_snapshot_version = contract.get("stats_snapshot_version")
                    if stats_snapshot_version is not None:
                        snapshot_version = str(stats_snapshot_version).strip()
                        if snapshot_version:
                            snapshot_idx = self._stats_snapshot_version_to_idx.setdefault(
                                snapshot_version,
                                len(self._stats_snapshot_version_to_idx),
                            )
                            self._idx_to_stats_snapshot_version[snapshot_idx] = snapshot_version
                            payload["stats_snapshot_version_idx"] = torch.tensor([snapshot_idx], dtype=torch.long)

                snapshot_as_of_epoch = None
                snapshot_as_of_epoch_raw = contract.get("stats_snapshot_as_of_epoch")
                if isinstance(snapshot_as_of_epoch_raw, (int, float)) and not isinstance(snapshot_as_of_epoch_raw, bool):
                    snapshot_as_of_epoch = float(snapshot_as_of_epoch_raw)
                if snapshot_as_of_epoch is None:
                    snapshot_as_of_epoch = self._safe_iso_to_epoch(contract.get("stats_snapshot_as_of_ts"))
                if snapshot_as_of_epoch is not None:
                    payload["stats_snapshot_as_of_epoch"] = torch.tensor([snapshot_as_of_epoch], dtype=torch.float64)
                graphs.append(Data(**payload))
        return self._create_data_loader(graphs, shuffle=shuffle)

    def _build_loader_from_dataset(self, dataset: Optional[Any], shuffle: bool) -> DataLoader:
        """Wrap prebuilt graph dataset into DataLoader and collect topology diagnostics."""
        if isinstance(dataset, dict) and dataset.get("kind") == "sharded_cache_split":
            lazy_dataset = ShardedGraphDataset.from_payload(
                dataset,
                max_cached_shards=self.sharded_dataset_cached_shards,
            )
            logger.info(
                "Using sharded on-disk dataset: split=%s graphs=%d shards=%d cached_shards=%d",
                str(dataset.get("split", "unknown")),
                int(dataset.get("graphs", len(lazy_dataset))),
                int(len(dataset.get("shards", []))) if isinstance(dataset.get("shards", []), list) else 0,
                int(self.sharded_dataset_cached_shards),
            )
            return self._create_data_loader_from_source(lazy_dataset, shuffle=shuffle)

        graphs_source = dataset or []
        for graph in graphs_source:
            num_nodes_attr = getattr(graph, "num_nodes", None)
            if num_nodes_attr is None:
                num_nodes = int(graph.x_cat.size(0))
            else:
                num_nodes = int(num_nodes_attr)
            num_edges = int(graph.edge_index.shape[1]) if hasattr(graph, "edge_index") else 0
            self._topology_nodes.append(num_nodes)
            self._topology_edges.append(num_edges)

            if hasattr(graph, "prefix_len"):
                prefix_tensor = graph.prefix_len
                if isinstance(prefix_tensor, torch.Tensor) and prefix_tensor.numel() > 0:
                    self._prefix_lengths.append(int(prefix_tensor.view(-1)[0].item()))

            if hasattr(graph, "process_version_idx"):
                version_tensor = graph.process_version_idx
                if isinstance(version_tensor, torch.Tensor) and version_tensor.numel() > 0:
                    idx = int(version_tensor.view(-1)[0].item())
                    if idx not in self._idx_to_version:
                        self._idx_to_version[idx] = f"v{idx}"
                    self._version_to_idx.setdefault(self._idx_to_version[idx], idx)
            if hasattr(graph, "stats_snapshot_version_idx"):
                snapshot_version_tensor = graph.stats_snapshot_version_idx
                if isinstance(snapshot_version_tensor, torch.Tensor) and snapshot_version_tensor.numel() > 0:
                    idx = int(snapshot_version_tensor.view(-1)[0].item())
                    if idx not in self._idx_to_stats_snapshot_version:
                        self._idx_to_stats_snapshot_version[idx] = f"k{idx:06d}"
                    self._stats_snapshot_version_to_idx.setdefault(self._idx_to_stats_snapshot_version[idx], idx)

        return self._create_data_loader_from_source(graphs_source, shuffle=shuffle)

    def _create_data_loader(self, graphs: Sequence[Data], *, shuffle: bool) -> DataLoader:
        """Create DataLoader with optional worker parallelism for batch collation."""
        return self._create_data_loader_from_source(graphs, shuffle=shuffle)

    def _create_data_loader_from_source(self, source: Any, *, shuffle: bool) -> DataLoader:
        """Create DataLoader from in-memory sequence or lazy on-disk dataset source."""
        sampler: Sampler[int] | None = None
        effective_shuffle = bool(shuffle)
        effective_num_workers = int(self.dataloader_num_workers)
        if isinstance(source, ShardedGraphDataset) and shuffle:
            sampler = _ShardAwareRandomSampler(source, seed=self.seed)
            effective_shuffle = False
        if isinstance(source, ShardedGraphDataset) and effective_num_workers > 0:
            logger.info(
                "Sharded dataset detected: forcing dataloader_num_workers=0 "
                "to avoid multiprocess shard cache duplication/IPC overhead."
            )
            effective_num_workers = 0

        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": effective_shuffle,
            "num_workers": effective_num_workers,
            "pin_memory": self.dataloader_pin_memory,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
        if effective_num_workers > 0:
            kwargs["persistent_workers"] = self.dataloader_persistent_workers
            kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
        return DataLoader(source, **kwargs)

    def _run_epoch(
        self,
        loader: Iterable[Data],
        optimizer: Optional[Adam],
        training: bool,
        epoch_index: int = 1,
        total_epochs: int = 1,
        epoch_progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Tuple[float, float, float, float]:
        """Run one epoch (train or eval) and return loss/macro_f1/weighted_f1/duration."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        started = perf_counter()
        total_loss = 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        batches = 0
        forward_stats = self._new_forward_stats_accumulator()
        contract_sanitized_batches = 0
        logits_sanitized_batches = 0
        non_finite_loss_batches = 0
        phase_name = "train" if training else "validation"
        stage_name = "train.batches" if training else "validation.batches"
        batch_reporter = ProgressReporter(stage=stage_name, min_interval_sec=0.8)
        try:
            total_batches = int(len(loader))  # type: ignore[arg-type]
        except Exception:
            total_batches = 0
        batch_reporter.start(
            message=f"{phase_name.title()} epoch {epoch_index}/{total_epochs}",
            current=0,
            total=total_batches if total_batches > 0 else None,
            payload={"epoch": int(epoch_index), "total_epochs": int(total_epochs)},
        )

        iterator = tqdm(loader, desc=("Train epoch" if training else "Validation epoch"), leave=self.tqdm_leave, disable=self._is_tqdm_disabled())
        for batch_idx, data in enumerate(iterator, start=1):
            data = data.to(self.device)
            contract = self._data_to_contract(data)
            if self._sanitize_contract_numeric_tensors(contract):
                contract_sanitized_batches += 1
            self._accumulate_forward_stats(forward_stats, contract)
            if training and optimizer is not None:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = self.model(contract)
                if not self._is_finite_tensor(logits):
                    logits_sanitized_batches += 1
                    logger.warning(
                        "Numeric guard [epoch:%s]: non-finite logits detected. Applying nan_to_num.",
                        "train" if training else "validation",
                    )
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                targets = data.y.view(-1).long()
                allowed_mask = self._normalize_allowed_mask(contract.get("allowed_target_mask"), expected_rows=int(targets.shape[0]))
                batch_target_in_mask_rate, valid_rows = self._batch_target_in_mask_rate(
                    allowed_mask=allowed_mask,
                    targets=targets,
                )
                if batch_target_in_mask_rate is not None:
                    self._update_mask_guided_reliability(rate=batch_target_in_mask_rate, samples=valid_rows)
                mask_policy = self._resolve_mask_guided_policy(
                    training=training,
                    batch_target_in_mask_rate=batch_target_in_mask_rate,
                    batch_samples=valid_rows,
                )
                effective_logits = self._apply_mask_guided_logits(
                    logits=logits,
                    allowed_mask=allowed_mask,
                    policy=mask_policy,
                )
                loss = self.criterion(effective_logits, targets)
                skip_optimizer_step = False
                if not self._is_finite_tensor(loss):
                    non_finite_loss_batches += 1
                    skip_optimizer_step = True
                    logger.warning(
                        "Numeric guard [epoch:%s]: non-finite loss detected. Skipping optimizer step for this batch.",
                        "train" if training else "validation",
                    )
                    if training and optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    loss = torch.zeros_like(loss)
                if training and optimizer is not None:
                    if not skip_optimizer_step:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            batches += 1
            all_pred.extend(torch.argmax(effective_logits.detach(), dim=1).cpu().numpy().tolist())
            all_true.extend(targets.detach().cpu().numpy().tolist())
            batch_reporter.update(
                message=f"{phase_name.title()} epoch {epoch_index}/{total_epochs}",
                current=int(batch_idx),
                total=total_batches if total_batches > 0 else None,
                payload={"epoch": int(epoch_index), "total_epochs": int(total_epochs)},
                force=(batch_idx == 1 or (total_batches > 0 and batch_idx == total_batches)),
            )
            if epoch_progress_callback is not None and total_batches > 0:
                epoch_progress_callback(int(batch_idx), int(total_batches), phase_name)

        duration = perf_counter() - started
        if batches == 0:
            batch_reporter.done(
                message=f"{phase_name.title()} epoch {epoch_index}/{total_epochs} completed (empty)",
                current=0,
                total=total_batches if total_batches > 0 else None,
                payload={"epoch": int(epoch_index), "total_epochs": int(total_epochs)},
            )
            return 0.0, 0.0, 0.0, duration

        macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(all_true, all_pred, average="weighted", zero_division=0))
        avg_loss = total_loss / batches
        if contract_sanitized_batches > 0 or logits_sanitized_batches > 0 or non_finite_loss_batches > 0:
            logger.info(
                "Numeric guard [epoch:%s]: contract_sanitized_batches=%d logits_sanitized_batches=%d non_finite_loss_batches=%d",
                "train" if training else "validation",
                contract_sanitized_batches,
                logits_sanitized_batches,
                non_finite_loss_batches,
            )
        self._log_forward_stats_summary("train" if training else "validation", forward_stats)
        batch_reporter.done(
            message=f"{phase_name.title()} epoch {epoch_index}/{total_epochs} completed",
            current=int(batches),
            total=total_batches if total_batches > 0 else None,
            payload={
                "epoch": int(epoch_index),
                "total_epochs": int(total_epochs),
                "loss": float(avg_loss),
                "macro_f1": float(macro_f1),
            },
        )
        return avg_loss, macro_f1, weighted_f1, duration

    def _evaluate_test(self, loader: Iterable[Data], stage_label: str = "inference") -> Dict[str, Any]:
        """Compute final test metrics after training is complete."""
        self.model.eval()
        self._logged_mask_debug = False
        self._logged_oos_math = False
        all_true: List[int] = []
        all_pred: List[int] = []
        all_probs: List[np.ndarray] = []
        all_oos_flags_aligned: List[float] = []
        all_target_in_mask_flags: List[float] = []
        all_pred_in_mask_flags: List[float] = []
        all_strict_error_but_allowed_flags: List[float] = []
        all_mask_cardinality: List[float] = []
        all_lengths: List[int] = []
        all_versions: List[str] = []
        inference_graphs = 0
        inference_started = perf_counter()
        first_batch_debug_logged = False
        forward_stats = self._new_forward_stats_accumulator()
        contract_sanitized_batches = 0
        logits_sanitized_batches = 0
        probs_sanitized_batches = 0
        test_batches_reporter = ProgressReporter(stage="test.batches", min_interval_sec=0.8)
        try:
            total_batches = int(len(loader))  # type: ignore[arg-type]
        except Exception:
            total_batches = 0
        test_batches_reporter.start(
            message=f"Test evaluation ({stage_label})",
            current=0,
            total=total_batches if total_batches > 0 else None,
        )

        with torch.no_grad():
            iterator = tqdm(loader, desc="Test evaluation", leave=self.tqdm_leave, disable=self._is_tqdm_disabled())
            for batch_idx, data in enumerate(iterator, start=1):
                data = data.to(self.device)
                contract = self._data_to_contract(data)
                if self._sanitize_contract_numeric_tensors(contract):
                    contract_sanitized_batches += 1
                self._accumulate_forward_stats(forward_stats, contract)
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                logits = self.model(contract)
                if not self._is_finite_tensor(logits):
                    logits_sanitized_batches += 1
                    logger.warning(
                        "Numeric guard [%s]: non-finite logits detected. Applying nan_to_num.",
                        stage_label,
                    )
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                target_tensor = data.y.view(-1).long()
                allowed_mask = self._normalize_allowed_mask(
                    contract.get("allowed_target_mask"),
                    expected_rows=int(target_tensor.shape[0]),
                )
                mask_policy = self._resolve_mask_guided_policy(
                    training=False,
                    batch_target_in_mask_rate=None,
                    batch_samples=int(target_tensor.shape[0]),
                )
                logits = self._apply_mask_guided_logits(
                    logits=logits,
                    allowed_mask=allowed_mask,
                    policy=mask_policy,
                )
                probs = torch.softmax(logits, dim=1)
                if not self._is_finite_tensor(probs):
                    probs_sanitized_batches += 1
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                    if probs.dim() == 2 and probs.size(1) > 0:
                        row_sum = probs.sum(dim=1, keepdim=True)
                        safe_uniform = torch.full_like(probs, 1.0 / float(probs.size(1)))
                        probs = torch.where(row_sum > 0.0, probs / row_sum.clamp_min(1e-12), safe_uniform)
                targets = target_tensor.cpu().numpy()
                pred_tensor = torch.argmax(probs, dim=1).long()
                preds = pred_tensor.cpu().numpy()
                batch_size = int(pred_tensor.shape[0])
                batch_oos_aligned = np.full(batch_size, np.nan, dtype=np.float32)
                batch_target_in_mask = np.full(batch_size, np.nan, dtype=np.float32)
                batch_pred_in_mask = np.full(batch_size, np.nan, dtype=np.float32)
                batch_strict_error_but_allowed = np.full(batch_size, np.nan, dtype=np.float32)
                batch_mask_cardinality = np.full(batch_size, np.nan, dtype=np.float32)
                if self.mode == "eval_drift" and not first_batch_debug_logged:
                    logger.info("Drift debug first batch y_true[:5]=%s y_pred[:5]=%s", targets[:5].tolist(), preds[:5].tolist())
                    first_batch_debug_logged = True

                if isinstance(allowed_mask, torch.Tensor):
                    if allowed_mask.dim() == 1:
                        allowed_mask = allowed_mask.unsqueeze(0)
                    if not self._logged_mask_debug:
                        all_true_mask = bool(allowed_mask.all().item())
                        logger.info(
                            "DEBUG MASK: shape=%s, All True? %s",
                            tuple(allowed_mask.shape),
                            all_true_mask,
                        )
                        self._logged_mask_debug = True
                    if not self._logged_oos_math:
                        logger.info("--- OOS MATH DEBUG ---")
                        logger.info(
                            "y_hat shape: %s, min=%s, max=%s",
                            tuple(pred_tensor.shape),
                            int(pred_tensor.min().item()) if pred_tensor.numel() > 0 else None,
                            int(pred_tensor.max().item()) if pred_tensor.numel() > 0 else None,
                        )
                        logger.info(
                            "mask shape: %s, dtype=%s",
                            tuple(allowed_mask.shape),
                            allowed_mask.dtype,
                        )
                        try:
                            batch_size = pred_tensor.size(0)
                            row_ids = torch.arange(batch_size, device=pred_tensor.device)
                            extracted_flags = allowed_mask[row_ids, pred_tensor]
                            logger.info(
                                "extracted_flags (allowed_target_mask[B, y_hat]): shape=%s, trues=%d/%d",
                                tuple(extracted_flags.shape),
                                int(extracted_flags.sum().item()),
                                batch_size,
                            )
                            oos_math_flags = (~extracted_flags).float()
                            logger.info(
                                "oos_flags (~extracted_flags): trues=%.0f/%d",
                                float(oos_math_flags.sum().item()),
                                batch_size,
                            )
                        except Exception as exc:
                            logger.error("Error during OOS indexing: %s", exc)
                        self._logged_oos_math = True
                    row_ids = torch.arange(batch_size, device=pred_tensor.device)
                    pred_in_mask = allowed_mask[row_ids, pred_tensor].bool()
                    target_in_mask = allowed_mask[row_ids, target_tensor.to(pred_tensor.device)].bool()
                    strict_error_but_allowed = (pred_tensor != target_tensor.to(pred_tensor.device)) & pred_in_mask
                    mask_cardinality = allowed_mask.sum(dim=1).float()
                    oos_flags = (~pred_in_mask).float()

                    batch_oos_aligned = oos_flags.detach().cpu().numpy().astype(np.float32, copy=False)
                    batch_target_in_mask = target_in_mask.float().detach().cpu().numpy().astype(np.float32, copy=False)
                    batch_pred_in_mask = pred_in_mask.float().detach().cpu().numpy().astype(np.float32, copy=False)
                    batch_strict_error_but_allowed = (
                        strict_error_but_allowed.float().detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    batch_mask_cardinality = mask_cardinality.detach().cpu().numpy().astype(np.float32, copy=False)

                all_oos_flags_aligned.extend(batch_oos_aligned.tolist())
                all_target_in_mask_flags.extend(batch_target_in_mask.tolist())
                all_pred_in_mask_flags.extend(batch_pred_in_mask.tolist())
                all_strict_error_but_allowed_flags.extend(batch_strict_error_but_allowed.tolist())
                all_mask_cardinality.extend(batch_mask_cardinality.tolist())

                if hasattr(data, "prefix_len"):
                    all_lengths.extend(data.prefix_len.view(-1).long().cpu().tolist())
                else:
                    graph_lengths = torch.bincount(data.batch.view(-1).long(), minlength=pred_tensor.shape[0])
                    all_lengths.extend(graph_lengths.long().cpu().tolist())

                if hasattr(data, "process_version_idx"):
                    version_indices = data.process_version_idx.view(-1).long().cpu().tolist()
                    for idx in version_indices:
                        all_versions.append(self._idx_to_version.get(int(idx), f"v{int(idx)}"))

                all_true.extend(targets.tolist())
                all_pred.extend(preds.tolist())
                all_probs.extend(probs.cpu().numpy())
                inference_graphs += int(targets.shape[0])
                test_batches_reporter.update(
                    message=f"Test evaluation ({stage_label})",
                    current=int(batch_idx),
                    total=total_batches if total_batches > 0 else None,
                    force=(batch_idx == 1 or (total_batches > 0 and batch_idx == total_batches)),
                )

        total_inference_ms = float((perf_counter() - inference_started) * 1000.0)
        inference_ms_per_graph = float(total_inference_ms / inference_graphs) if inference_graphs > 0 else 0.0
        if contract_sanitized_batches > 0 or logits_sanitized_batches > 0 or probs_sanitized_batches > 0:
            logger.info(
                "Numeric guard [%s]: contract_sanitized_batches=%d logits_sanitized_batches=%d probs_sanitized_batches=%d",
                stage_label,
                contract_sanitized_batches,
                logits_sanitized_batches,
                probs_sanitized_batches,
            )
        self._log_forward_stats_summary(stage_label, forward_stats)
        test_batches_reporter.done(
            message=f"Test evaluation ({stage_label}) completed",
            current=total_batches if total_batches > 0 else None,
            total=total_batches if total_batches > 0 else None,
        )

        if not all_true:
            return {
                "test_macro_f1": 0.0,
                "test_weighted_f1": 0.0,
                "test_accuracy": 0.0,
                "test_top3_accuracy": 0.0,
                "test_weighted_f1": 0.0,
                "test_ece": 0.0,
                "test_oos": None,
                "test_target_in_mask_rate": None,
                "test_pred_in_mask_rate": None,
                "test_strict_error_but_allowed_rate": None,
                "test_ambiguous_prefix_rate": None,
                "test_mask_coverage": 0.0,
                "test_precision_macro": 0.0,
                "test_recall_macro": 0.0,
                "test_inference_time_ms_per_graph": inference_ms_per_graph,
            }

        y_true = np.asarray(all_true)
        y_pred = np.asarray(all_pred)
        y_prob = np.asarray(all_probs)
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
        if y_prob.ndim == 2 and y_prob.shape[1] > 0:
            row_sums = np.sum(y_prob, axis=1, keepdims=True)
            zero_rows = row_sums <= 1e-12
            if np.any(zero_rows):
                y_prob[zero_rows[:, 0], :] = 1.0 / float(y_prob.shape[1])
                row_sums = np.sum(y_prob, axis=1, keepdims=True)
            y_prob = y_prob / np.clip(row_sums, 1e-12, None)
        num_classes = y_prob.shape[1]
        top_k = min(3, num_classes)
        if num_classes <= 2:
            top3_accuracy = float(accuracy_score(y_true, y_pred))
        else:
            top3_accuracy = float(top_k_accuracy_score(y_true, y_prob, k=top_k, labels=list(range(num_classes))))

        metrics = {
            "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_top3_accuracy": top3_accuracy,
            "test_ece": float(self._expected_calibration_error(y_true, y_prob)),
            "test_oos": None,
            "test_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_inference_time_ms_per_graph": inference_ms_per_graph,
        }
        oos_flags_aligned = np.asarray(all_oos_flags_aligned, dtype=np.float32) if all_oos_flags_aligned else None
        target_in_mask_flags = np.asarray(all_target_in_mask_flags, dtype=np.float32) if all_target_in_mask_flags else None
        pred_in_mask_flags = np.asarray(all_pred_in_mask_flags, dtype=np.float32) if all_pred_in_mask_flags else None
        strict_error_but_allowed_flags = (
            np.asarray(all_strict_error_but_allowed_flags, dtype=np.float32) if all_strict_error_but_allowed_flags else None
        )
        mask_cardinality = np.asarray(all_mask_cardinality, dtype=np.float32) if all_mask_cardinality else None

        metrics["test_oos"] = self._nanmean_or_none(oos_flags_aligned)
        metrics["test_target_in_mask_rate"] = self._nanmean_or_none(target_in_mask_flags)
        metrics["test_pred_in_mask_rate"] = self._nanmean_or_none(pred_in_mask_flags)
        metrics["test_strict_error_but_allowed_rate"] = self._nanmean_or_none(strict_error_but_allowed_flags)

        ambiguous_prefix_rate: float | None = None
        if isinstance(mask_cardinality, np.ndarray) and int(mask_cardinality.shape[0]) == int(y_true.shape[0]):
            finite_mask_cardinality = np.isfinite(mask_cardinality)
            if bool(np.any(finite_mask_cardinality)):
                ambiguous_prefix_rate = float(np.mean(mask_cardinality[finite_mask_cardinality] > 1.0))
                metrics["test_mask_coverage"] = float(np.mean(finite_mask_cardinality.astype(np.float32)))
            else:
                metrics["test_mask_coverage"] = 0.0
        else:
            metrics["test_mask_coverage"] = 0.0
        metrics["test_ambiguous_prefix_rate"] = ambiguous_prefix_rate

        metrics.update(
            self._compute_sliced_metrics(
                y_true=y_true,
                y_pred=y_pred,
                oos_flags=oos_flags_aligned,
                target_in_mask_flags=target_in_mask_flags,
                pred_in_mask_flags=pred_in_mask_flags,
                strict_error_but_allowed_flags=strict_error_but_allowed_flags,
                mask_cardinality=mask_cardinality,
                prefix_lengths=np.asarray(all_lengths, dtype=np.int64) if all_lengths else None,
                versions=all_versions if all_versions else None,
            )
        )
        return metrics

    @staticmethod
    def _is_finite_tensor(value: Any) -> bool:
        if not isinstance(value, torch.Tensor):
            return True
        if value.numel() == 0:
            return True
        return bool(torch.isfinite(value).all().item())

    def _sanitize_contract_numeric_tensors(self, contract: GraphTensorContract) -> bool:
        sanitized = False
        for key in ("x_num", "struct_x", "structural_edge_weight"):
            tensor = contract.get(key)
            if not isinstance(tensor, torch.Tensor):
                continue
            if self._is_finite_tensor(tensor):
                continue
            contract[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            sanitized = True
        return sanitized

    def _evaluate_drift_windows(self, traces: Sequence[RawTrace]) -> List[Dict[str, float]]:
        """Run fixed-size chronological window evaluation for concept-drift tracking."""
        if self.drift_window_size <= 0:
            raise ValueError("drift_window_size must be positive.")
        windows = self._generate_drift_windows(traces)
        drift_results: List[Dict[str, float]] = []
        logger.info(
            "Drift windows prepared: size=%d, step=%d, keep_short_tail=false, windows=%d",
            self.drift_window_size,
            self._resolve_drift_step(),
            len(windows),
        )

        iterator = tqdm(
            windows,
            desc="Eval drift",
            leave=self.tqdm_leave,
            disable=self._is_tqdm_disabled(),
        )
        total_windows = len(windows)
        emit_progress_event(stage="eval_drift.windows", status="start", message="Evaluating drift windows", current=0, total=total_windows)
        for window_idx, (start, window_traces) in enumerate(iterator):
            window_loader = self._build_loader(window_traces, shuffle=False)
            metrics = self._evaluate_test(window_loader, stage_label="eval_drift")
            macro_f1 = float(metrics.get("test_macro_f1", 0.0))
            ece = float(metrics.get("test_ece", 0.0))
            window_oos = metrics.get("test_oos")
            window_target_in_mask = metrics.get("test_target_in_mask_rate")
            window_pred_in_mask = metrics.get("test_pred_in_mask_rate")
            window_strict_error_but_allowed = metrics.get("test_strict_error_but_allowed_rate")
            window_ambiguous_prefix = metrics.get("test_ambiguous_prefix_rate")

            iterator.set_postfix({"f1": f"{macro_f1:.4f}", "ece": f"{ece:.4f}"})

            if self.tracker is not None:
                self.tracker.log_metric("drift_window_macro_f1", macro_f1, step=window_idx)
                self.tracker.log_metric("drift_window_test_ece", ece, step=window_idx)
                if window_oos is not None:
                    self.tracker.log_metric("drift_window_test_oos", float(window_oos), step=window_idx)
                if window_target_in_mask is not None:
                    self.tracker.log_metric(
                        "drift_window_target_in_mask_rate",
                        float(window_target_in_mask),
                        step=window_idx,
                    )
                if window_pred_in_mask is not None:
                    self.tracker.log_metric(
                        "drift_window_pred_in_mask_rate",
                        float(window_pred_in_mask),
                        step=window_idx,
                    )
                if window_strict_error_but_allowed is not None:
                    self.tracker.log_metric(
                        "drift_window_strict_error_but_allowed_rate",
                        float(window_strict_error_but_allowed),
                        step=window_idx,
                    )
                if window_ambiguous_prefix is not None:
                    self.tracker.log_metric(
                        "drift_window_ambiguous_prefix_rate",
                        float(window_ambiguous_prefix),
                        step=window_idx,
                    )

            start_ts = float(window_traces[0].events[0].timestamp) if window_traces and window_traces[0].events else 0.0
            end_ts = float(window_traces[-1].events[-1].timestamp) if window_traces and window_traces[-1].events else start_ts
            logger.info("Window %d/%d (%.3f - %.3f) | F1: %.4f...", window_idx + 1, total_windows, start_ts, end_ts, macro_f1)

            drift_results.append(
                {
                    "window_index": float(window_idx),
                    "window_start_trace": float(start),
                    "window_end_trace": float(start + len(window_traces) - 1),
                    "window_start_ts": start_ts,
                    "window_end_ts": end_ts,
                    "window_macro_f1": macro_f1,
                    "window_test_ece": ece,
                    "window_test_oos": float(window_oos) if window_oos is not None else float("nan"),
                    "window_target_in_mask_rate": (
                        float(window_target_in_mask) if window_target_in_mask is not None else float("nan")
                    ),
                    "window_pred_in_mask_rate": (
                        float(window_pred_in_mask) if window_pred_in_mask is not None else float("nan")
                    ),
                    "window_strict_error_but_allowed_rate": (
                        float(window_strict_error_but_allowed)
                        if window_strict_error_but_allowed is not None
                        else float("nan")
                    ),
                    "window_ambiguous_prefix_rate": (
                        float(window_ambiguous_prefix) if window_ambiguous_prefix is not None else float("nan")
                    ),
                }
            )

            if self.tracker is not None:
                self.tracker.log_metric("drift_window_start_ts", start_ts, step=window_idx)
                self.tracker.log_metric("drift_window_end_ts", end_ts, step=window_idx)

            emit_progress_event(
                stage="eval_drift.windows",
                status="update",
                message=f"Window {window_idx + 1}/{total_windows}",
                current=int(window_idx + 1),
                total=total_windows,
                payload={
                    "macro_f1": float(macro_f1),
                    "ece": float(ece),
                    "target_in_mask_rate": float(window_target_in_mask) if window_target_in_mask is not None else None,
                    "pred_in_mask_rate": float(window_pred_in_mask) if window_pred_in_mask is not None else None,
                },
            )

        emit_progress_event(stage="eval_drift.windows", status="done", message="Drift evaluation completed", current=total_windows, total=total_windows)

        return drift_results

    def _generate_drift_windows(self, traces: Sequence[RawTrace]) -> List[Tuple[int, List[RawTrace]]]:
        """Generate chronological drift windows and drop short tail windows."""
        ordered_traces = sorted((trace for trace in traces if trace.events), key=lambda trace: trace.events[0].timestamp)
        size = self.drift_window_size
        step = self._resolve_drift_step()

        windows: List[Tuple[int, List[RawTrace]]] = []
        for start in range(0, len(ordered_traces), step):
            window = ordered_traces[start : start + size]
            if len(window) < size:
                continue
            windows.append((start, window))
        return windows

    def _resolve_drift_step(self) -> int:
        """Resolve drift step from sliding-window policy."""
        if self.drift_window_sliding < 0:
            raise ValueError("drift_window_sliding must be non-negative.")
        return self.drift_window_sliding or self.drift_window_size

    @staticmethod
    def _compute_oos_flags(y_hat: torch.Tensor, allowed_target_mask: torch.Tensor) -> torch.Tensor:
        """Return per-sample OOS flags (1.0 when predicted class is not allowed)."""
        if y_hat.dim() != 1:
            y_hat = y_hat.view(-1)
        if allowed_target_mask.dim() != 2:
            raise ValueError("allowed_target_mask must have shape [B, C].")
        batch_size = y_hat.shape[0]
        if allowed_target_mask.shape[0] != batch_size:
            raise ValueError("allowed_target_mask batch dimension must match predictions.")
        row_ids = torch.arange(batch_size, device=y_hat.device)
        return (~allowed_target_mask[row_ids, y_hat]).float()

    @staticmethod
    def _normalize_allowed_mask(raw_mask: Any, *, expected_rows: int) -> Optional[torch.Tensor]:
        if not isinstance(raw_mask, torch.Tensor):
            return None
        mask = raw_mask.bool()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() != 2:
            return None
        if int(mask.shape[0]) != int(expected_rows):
            return None
        return mask

    @staticmethod
    def _batch_target_in_mask_rate(*, allowed_mask: Optional[torch.Tensor], targets: torch.Tensor) -> Tuple[Optional[float], int]:
        if not isinstance(allowed_mask, torch.Tensor):
            return None, 0
        if allowed_mask.dim() != 2:
            return None, 0
        if targets.dim() != 1:
            targets = targets.view(-1)
        if int(allowed_mask.shape[0]) != int(targets.shape[0]):
            return None, 0
        row_ids = torch.arange(int(targets.shape[0]), device=targets.device)
        target_in_mask = allowed_mask[row_ids, targets].bool()
        sample_count = int(target_in_mask.shape[0])
        if sample_count <= 0:
            return None, 0
        return float(target_in_mask.float().mean().item()), sample_count

    def _update_mask_guided_reliability(self, *, rate: float, samples: int) -> None:
        if samples <= 0:
            return
        if not math.isfinite(float(rate)):
            return
        samples_int = int(samples)
        if samples_int <= 0:
            return
        if self._mask_guided_reliability_rate is None or self._mask_guided_reliability_samples <= 0:
            self._mask_guided_reliability_rate = float(rate)
            self._mask_guided_reliability_samples = samples_int
            return
        prev_samples = int(self._mask_guided_reliability_samples)
        prev_rate = float(self._mask_guided_reliability_rate)
        total_samples = prev_samples + samples_int
        weighted = ((prev_rate * prev_samples) + (float(rate) * samples_int)) / float(total_samples)
        self._mask_guided_reliability_rate = float(weighted)
        self._mask_guided_reliability_samples = int(total_samples)

    def _resolve_mask_guided_policy(
        self,
        *,
        training: bool,
        batch_target_in_mask_rate: Optional[float],
        batch_samples: int,
    ) -> str:
        if not self.mask_guided_enabled:
            return "off"
        if training:
            if batch_target_in_mask_rate is None:
                return "soft"
            if int(batch_samples) >= int(self.mask_guided_min_samples_for_hard) and float(batch_target_in_mask_rate) >= float(
                self.mask_guided_hard_threshold
            ):
                return "hard"
            return "soft"
        if not self.mask_guided_apply_in_eval:
            return "off"
        rate = self._mask_guided_reliability_rate
        samples = int(self._mask_guided_reliability_samples)
        if rate is not None and samples >= int(self.mask_guided_min_samples_for_hard) and float(rate) >= float(self.mask_guided_hard_threshold):
            return "hard"
        return "soft"

    def _apply_mask_guided_logits(
        self,
        *,
        logits: torch.Tensor,
        allowed_mask: Optional[torch.Tensor],
        policy: str,
    ) -> torch.Tensor:
        if policy == "off":
            return logits
        if not isinstance(allowed_mask, torch.Tensor):
            return logits
        if allowed_mask.dim() != 2 or logits.dim() != 2:
            return logits
        if int(allowed_mask.shape[0]) != int(logits.shape[0]) or int(allowed_mask.shape[1]) != int(logits.shape[1]):
            return logits
        row_has_allowed = allowed_mask.any(dim=1, keepdim=True)
        effective_disallowed = (~allowed_mask) & row_has_allowed
        if not bool(effective_disallowed.any().item()):
            return logits
        if policy == "hard":
            hard_floor = torch.finfo(logits.dtype).min / 4.0
            return logits.masked_fill(effective_disallowed, float(hard_floor))
        penalty = max(0.0, float(self.mask_guided_soft_penalty))
        if penalty <= 0.0:
            return logits
        return logits - (effective_disallowed.to(logits.dtype) * penalty)

    @staticmethod
    def _nanmean_or_none(values: Optional[np.ndarray]) -> Optional[float]:
        if not isinstance(values, np.ndarray):
            return None
        if values.size == 0:
            return None
        finite_mask = np.isfinite(values)
        if not bool(np.any(finite_mask)):
            return None
        return float(np.mean(values[finite_mask]))

    @staticmethod
    def _length_bin(length: int) -> str:
        """Map prefix length to configured slicing bin."""
        if 1 <= length <= 5:
            return "len_1_5"
        if 6 <= length <= 10:
            return "len_6_10"
        if 11 <= length <= 20:
            return "len_11_20"
        return "len_21_plus"

    @staticmethod
    def _safe_key_suffix(text: str) -> str:
        """Normalize free-form labels into metric-safe suffixes."""
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))

    @classmethod
    def _compute_sliced_metrics(
        cls,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        oos_flags: Optional[np.ndarray],
        target_in_mask_flags: Optional[np.ndarray],
        pred_in_mask_flags: Optional[np.ndarray],
        strict_error_but_allowed_flags: Optional[np.ndarray],
        mask_cardinality: Optional[np.ndarray],
        prefix_lengths: Optional[np.ndarray],
        versions: Optional[List[str]],
    ) -> Dict[str, float]:
        """Compute slicing metrics by prefix length bins and process versions."""
        metrics: Dict[str, float] = {}
        sample_count = int(y_true.shape[0])

        def _append_mask_slice_metrics(metric_suffix: str, idxs: List[int]) -> None:
            if target_in_mask_flags is not None and int(target_in_mask_flags.shape[0]) == sample_count:
                value = cls._nanmean_or_none(target_in_mask_flags[idxs])
                if value is not None:
                    metrics[f"test_target_in_mask_rate_{metric_suffix}"] = value
            if pred_in_mask_flags is not None and int(pred_in_mask_flags.shape[0]) == sample_count:
                value = cls._nanmean_or_none(pred_in_mask_flags[idxs])
                if value is not None:
                    metrics[f"test_pred_in_mask_rate_{metric_suffix}"] = value
            if strict_error_but_allowed_flags is not None and int(strict_error_but_allowed_flags.shape[0]) == sample_count:
                value = cls._nanmean_or_none(strict_error_but_allowed_flags[idxs])
                if value is not None:
                    metrics[f"test_strict_error_but_allowed_rate_{metric_suffix}"] = value

        if prefix_lengths is not None and int(prefix_lengths.shape[0]) == sample_count:
            length_bins = ["len_1_5", "len_6_10", "len_11_20", "len_21_plus"]
            grouped: Dict[str, List[int]] = {name: [] for name in length_bins}
            for idx, raw_len in enumerate(prefix_lengths.tolist()):
                grouped[cls._length_bin(int(raw_len))].append(idx)

            for bin_name, idxs in grouped.items():
                if not idxs:
                    continue
                bin_true = y_true[idxs]
                bin_pred = y_pred[idxs]
                metrics[f"test_accuracy_{bin_name}"] = float(accuracy_score(bin_true, bin_pred))
                metrics[f"test_f1_{bin_name}"] = float(f1_score(bin_true, bin_pred, average="macro", zero_division=0))
                if oos_flags is not None and int(oos_flags.shape[0]) == sample_count:
                    value = cls._nanmean_or_none(oos_flags[idxs])
                    if value is not None:
                        metrics[f"test_oos_{bin_name}"] = value
                _append_mask_slice_metrics(bin_name, idxs)

        if versions is not None and len(versions) == sample_count:
            grouped_idx: Dict[str, List[int]] = {}
            for idx, version in enumerate(versions):
                grouped_idx.setdefault(str(version), []).append(idx)

            for version, idxs in grouped_idx.items():
                ver_true = y_true[idxs]
                ver_pred = y_pred[idxs]
                suffix = cls._safe_key_suffix(version)
                metrics[f"test_accuracy_{suffix}"] = float(accuracy_score(ver_true, ver_pred))
                metrics[f"test_f1_{suffix}"] = float(f1_score(ver_true, ver_pred, average="macro", zero_division=0))
                if oos_flags is not None and int(oos_flags.shape[0]) == sample_count:
                    value = cls._nanmean_or_none(oos_flags[idxs])
                    if value is not None:
                        metrics[f"test_oos_{suffix}"] = value
                _append_mask_slice_metrics(suffix, idxs)

        if mask_cardinality is not None and int(mask_cardinality.shape[0]) == sample_count:
            grouped_mask: Dict[str, List[int]] = {
                "mask_card_1": [],
                "mask_card_2": [],
                "mask_card_3_plus": [],
            }
            for idx, raw in enumerate(mask_cardinality.tolist()):
                if not math.isfinite(float(raw)):
                    continue
                cardinality = int(raw)
                if cardinality == 1:
                    grouped_mask["mask_card_1"].append(idx)
                elif cardinality == 2:
                    grouped_mask["mask_card_2"].append(idx)
                elif cardinality >= 3:
                    grouped_mask["mask_card_3_plus"].append(idx)

            for group_name, idxs in grouped_mask.items():
                if not idxs:
                    continue
                bin_true = y_true[idxs]
                bin_pred = y_pred[idxs]
                metrics[f"test_accuracy_{group_name}"] = float(accuracy_score(bin_true, bin_pred))
                metrics[f"test_f1_{group_name}"] = float(f1_score(bin_true, bin_pred, average="macro", zero_division=0))
                if oos_flags is not None and int(oos_flags.shape[0]) == sample_count:
                    value = cls._nanmean_or_none(oos_flags[idxs])
                    if value is not None:
                        metrics[f"test_oos_{group_name}"] = value
                _append_mask_slice_metrics(group_name, idxs)

        return metrics

    def _expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute ECE over confidence bins for probabilistic calibration quality."""
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        correctness = (predictions == y_true).astype(float)

        bin_edges = np.linspace(0.0, 1.0, self.num_ece_bins + 1)
        ece = 0.0
        total = len(y_true)
        for idx in range(self.num_ece_bins):
            lower = bin_edges[idx]
            upper = bin_edges[idx + 1]
            in_bin = (confidences > lower) & (confidences <= upper)
            count = int(np.sum(in_bin))
            if count == 0:
                continue
            bin_acc = float(np.mean(correctness[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += abs(bin_acc - bin_conf) * (count / total)
        return float(ece)

    @staticmethod
    def _safe_iso_to_epoch(value: Any) -> float | None:
        """Parse ISO timestamp into epoch seconds for tensor payload compatibility."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())

    @staticmethod
    def _epoch_to_iso(epoch: float) -> str | None:
        """Render epoch seconds to UTC ISO string with timezone for logging output."""
        try:
            return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    @staticmethod
    def _new_forward_stats_accumulator() -> Dict[str, Any]:
        return {
            "batches": 0,
            "graphs": 0,
            "batches_with_struct_x": 0,
            "batches_with_struct_edges": 0,
            "batches_with_snapshot_meta": 0,
            "stats_allowed_true": 0,
            "stats_allowed_false": 0,
            "stats_missing_asof_true": 0,
            "stats_missing_asof_false": 0,
            "batches_with_missing_asof": 0,
            "struct_feature_dims": set(),
            "struct_node_rows": set(),
            "struct_edge_counts": [],
            "snapshot_versions": set(),
            "snapshot_as_of_ts": set(),
        }

    def _accumulate_forward_stats(self, bucket: Dict[str, Any], contract: GraphTensorContract) -> None:
        bucket["batches"] = int(bucket.get("batches", 0)) + 1

        graph_count = 0
        batch_tensor = contract.get("batch")
        if isinstance(batch_tensor, torch.Tensor) and batch_tensor.numel() > 0:
            graph_count = int(batch_tensor.max().item()) + 1
        elif isinstance(contract.get("y"), torch.Tensor):
            graph_count = int(contract["y"].view(-1).shape[0])
        bucket["graphs"] = int(bucket.get("graphs", 0)) + max(graph_count, 0)

        struct_x = contract.get("struct_x")
        if isinstance(struct_x, torch.Tensor) and struct_x.numel() > 0:
            bucket["batches_with_struct_x"] = int(bucket.get("batches_with_struct_x", 0)) + 1
            dims = struct_x.dim()
            rows = int(struct_x.size(0)) if dims >= 1 else 0
            features = int(struct_x.size(1)) if dims >= 2 else 1
            bucket.setdefault("struct_feature_dims", set()).add(features)
            bucket.setdefault("struct_node_rows", set()).add(rows)

        structural_edge_index = contract.get("structural_edge_index")
        if isinstance(structural_edge_index, torch.Tensor):
            edge_count = int(structural_edge_index.size(1)) if structural_edge_index.dim() >= 2 else 0
            bucket.setdefault("struct_edge_counts", []).append(edge_count)
            if structural_edge_index.numel() > 0:
                bucket["batches_with_struct_edges"] = int(bucket.get("batches_with_struct_edges", 0)) + 1

        versions = contract.get("stats_snapshot_versions")
        if isinstance(versions, list):
            if versions:
                bucket["batches_with_snapshot_meta"] = int(bucket.get("batches_with_snapshot_meta", 0)) + 1
            for value in versions:
                text = str(value).strip()
                if text:
                    bucket.setdefault("snapshot_versions", set()).add(text)
        else:
            single_version = contract.get("stats_snapshot_version")
            if single_version is not None:
                text = str(single_version).strip()
                if text:
                    bucket.setdefault("snapshot_versions", set()).add(text)

        as_of_values = contract.get("stats_snapshot_as_of_ts_batch")
        if isinstance(as_of_values, list):
            for value in as_of_values:
                text = str(value).strip()
                if text:
                    bucket.setdefault("snapshot_as_of_ts", set()).add(text)
        else:
            single_as_of = contract.get("stats_snapshot_as_of_ts")
            if single_as_of is not None:
                text = str(single_as_of).strip()
                if text:
                    bucket.setdefault("snapshot_as_of_ts", set()).add(text)

        stats_allowed_batch = contract.get("stats_allowed_batch")
        if isinstance(stats_allowed_batch, list):
            for raw in stats_allowed_batch:
                if bool(raw):
                    bucket["stats_allowed_true"] = int(bucket.get("stats_allowed_true", 0)) + 1
                else:
                    bucket["stats_allowed_false"] = int(bucket.get("stats_allowed_false", 0)) + 1
        else:
            single_stats_allowed = contract.get("stats_allowed")
            if single_stats_allowed is not None:
                if bool(single_stats_allowed):
                    bucket["stats_allowed_true"] = int(bucket.get("stats_allowed_true", 0)) + 1
                else:
                    bucket["stats_allowed_false"] = int(bucket.get("stats_allowed_false", 0)) + 1

        stats_missing_batch = contract.get("stats_missing_asof_snapshot_batch")
        if isinstance(stats_missing_batch, list):
            had_missing = False
            for raw in stats_missing_batch:
                if bool(raw):
                    bucket["stats_missing_asof_true"] = int(bucket.get("stats_missing_asof_true", 0)) + 1
                    had_missing = True
                else:
                    bucket["stats_missing_asof_false"] = int(bucket.get("stats_missing_asof_false", 0)) + 1
            if had_missing:
                bucket["batches_with_missing_asof"] = int(bucket.get("batches_with_missing_asof", 0)) + 1
        else:
            single_stats_missing = contract.get("stats_missing_asof_snapshot")
            if single_stats_missing is not None:
                if bool(single_stats_missing):
                    bucket["stats_missing_asof_true"] = int(bucket.get("stats_missing_asof_true", 0)) + 1
                    bucket["batches_with_missing_asof"] = int(bucket.get("batches_with_missing_asof", 0)) + 1
                else:
                    bucket["stats_missing_asof_false"] = int(bucket.get("stats_missing_asof_false", 0)) + 1

    @staticmethod
    def _format_unique_values(values: set[Any], limit: int = 3) -> str:
        if not values:
            return "none"
        ordered = [str(item) for item in sorted(values, key=lambda item: str(item))]
        if len(ordered) <= limit:
            return ",".join(ordered)
        head = ",".join(ordered[:limit])
        return f"{head},...(+{len(ordered) - limit})"

    def _log_forward_stats_summary(self, stage_label: str, bucket: Dict[str, Any]) -> None:
        batches = int(bucket.get("batches", 0))
        if batches <= 0:
            logger.info("Forward stats [%s]: no batches.", stage_label)
            return

        struct_edge_counts = [int(item) for item in bucket.get("struct_edge_counts", [])]
        if struct_edge_counts:
            edge_min = min(struct_edge_counts)
            edge_max = max(struct_edge_counts)
            edge_avg = float(sum(struct_edge_counts) / len(struct_edge_counts))
        else:
            edge_min = 0
            edge_max = 0
            edge_avg = 0.0

        versions = bucket.get("snapshot_versions", set())
        as_of_values = bucket.get("snapshot_as_of_ts", set())
        feature_dims = bucket.get("struct_feature_dims", set())
        struct_rows = bucket.get("struct_node_rows", set())
        stats_allowed_true = int(bucket.get("stats_allowed_true", 0))
        stats_allowed_false = int(bucket.get("stats_allowed_false", 0))
        stats_missing_asof_true = int(bucket.get("stats_missing_asof_true", 0))
        stats_missing_asof_false = int(bucket.get("stats_missing_asof_false", 0))
        batches_with_missing_asof = int(bucket.get("batches_with_missing_asof", 0))
        snapshot_meta_batches = int(bucket.get("batches_with_snapshot_meta", 0))

        logger.info(
            "Forward stats [%s]: batches=%d graphs=%d struct_x_batches=%d struct_edge_batches=%d "
            "snapshot_meta_batches=%d stats_allowed[true/false]=%d/%d "
            "missing_asof_snapshot_batches=%d missing_asof_snapshot[true/false]=%d/%d "
            "struct_feature_dims=%s struct_rows=%s edge_count[min/avg/max]=%d/%.2f/%d "
            "snapshot_versions=%s snapshot_as_of_ts=%s",
            stage_label,
            batches,
            int(bucket.get("graphs", 0)),
            int(bucket.get("batches_with_struct_x", 0)),
            int(bucket.get("batches_with_struct_edges", 0)),
            snapshot_meta_batches,
            stats_allowed_true,
            stats_allowed_false,
            batches_with_missing_asof,
            stats_missing_asof_true,
            stats_missing_asof_false,
            self._format_unique_values(set(feature_dims)),
            self._format_unique_values(set(struct_rows)),
            edge_min,
            edge_avg,
            edge_max,
            self._format_unique_values(set(versions)),
            self._format_unique_values(set(as_of_values)),
        )

    def _warn_if_mixed_snapshot_versions(self, data: Data) -> None:
        snapshot_version_idx = getattr(data, "stats_snapshot_version_idx", None)
        if not isinstance(snapshot_version_idx, torch.Tensor):
            return
        raw = snapshot_version_idx.view(-1).long().cpu().tolist()
        if not raw:
            return
        unique_idx = tuple(sorted({int(item) for item in raw}))
        if len(unique_idx) <= 1:
            return
        if unique_idx in self._warned_mixed_snapshot_batches:
            return
        self._warned_mixed_snapshot_batches.add(unique_idx)
        labels = [self._idx_to_stats_snapshot_version.get(item, f"k{item:06d}") for item in unique_idx]
        warnings.warn(
            "Mixed stats snapshots in one batch detected: "
            f"{','.join(labels)}. Using first graph snapshot for structural tensors.",
            UserWarning,
            stacklevel=2,
        )

    def _select_structural_payload_for_forward(
        self,
        data: Data,
        batch_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Select structural payload for forward; use first graph payload on batched input."""
        self._warn_if_mixed_snapshot_versions(data)

        to_data_list = getattr(data, "to_data_list", None)
        if callable(to_data_list):
            try:
                data_list = list(to_data_list())
            except Exception:
                data_list = []
            if data_list:
                first = data_list[0]
                struct_x = first.struct_x if hasattr(first, "struct_x") and isinstance(first.struct_x, torch.Tensor) else None
                structural_edge_index = (
                    first.structural_edge_index
                    if hasattr(first, "structural_edge_index") and isinstance(first.structural_edge_index, torch.Tensor)
                    else None
                )
                structural_edge_weight = (
                    first.structural_edge_weight
                    if hasattr(first, "structural_edge_weight") and isinstance(first.structural_edge_weight, torch.Tensor)
                    else None
                )
                return struct_x, structural_edge_index, structural_edge_weight

        struct_x = data.struct_x if hasattr(data, "struct_x") and isinstance(data.struct_x, torch.Tensor) else None
        structural_edge_index = (
            data.structural_edge_index
            if hasattr(data, "structural_edge_index") and isinstance(data.structural_edge_index, torch.Tensor)
            else None
        )
        structural_edge_weight = (
            data.structural_edge_weight
            if hasattr(data, "structural_edge_weight") and isinstance(data.structural_edge_weight, torch.Tensor)
            else None
        )

        if isinstance(struct_x, torch.Tensor):
            graph_count = int(batch_tensor.max().item()) + 1 if batch_tensor.numel() > 0 else int(data.y.view(-1).shape[0])
            if graph_count > 1 and struct_x.dim() == 2 and int(struct_x.size(0)) % graph_count == 0:
                cls_rows = int(struct_x.size(0)) // graph_count
                struct_x = struct_x.view(graph_count, cls_rows, int(struct_x.size(1)))[0]

        return struct_x, structural_edge_index, structural_edge_weight

    def _data_to_contract(self, data: Data) -> GraphTensorContract:
        """Convert PyG batch Data to GraphTensorContract for model forward."""
        edge_type = data.edge_type if hasattr(data, "edge_type") else torch.zeros(data.edge_index.shape[1], dtype=torch.long)
        num_nodes = int(data.x_cat.size(0))
        batch_tensor = data.batch if hasattr(data, "batch") else torch.zeros(num_nodes, dtype=torch.long, device=data.x_cat.device)
        if batch_tensor.numel() != num_nodes:
            batch_tensor = torch.zeros(num_nodes, dtype=torch.long, device=data.x_cat.device)
        contract = GraphTensorContract(
            x_cat=data.x_cat,
            x_num=data.x_num,
            edge_index=data.edge_index,
            edge_type=edge_type,
            y=data.y,
            batch=batch_tensor,
            num_nodes=num_nodes,
        )
        if hasattr(data, "allowed_target_mask"):
            mask = data.allowed_target_mask
            if isinstance(mask, torch.Tensor):
                contract["allowed_target_mask"] = mask.bool()
        struct_x, structural_edge_index, structural_edge_weight = self._select_structural_payload_for_forward(
            data,
            batch_tensor,
        )
        if isinstance(struct_x, torch.Tensor):
            contract["struct_x"] = struct_x.float()
        if isinstance(structural_edge_index, torch.Tensor):
            contract["structural_edge_index"] = structural_edge_index
        if isinstance(structural_edge_weight, torch.Tensor):
            contract["structural_edge_weight"] = structural_edge_weight
        if hasattr(data, "stats_snapshot_version_idx"):
            snapshot_version_idx = data.stats_snapshot_version_idx
            if isinstance(snapshot_version_idx, torch.Tensor):
                version_labels: List[str] = []
                for idx in snapshot_version_idx.view(-1).long().cpu().tolist():
                    version_labels.append(self._idx_to_stats_snapshot_version.get(int(idx), f"k{int(idx):06d}"))
                if version_labels:
                    contract["stats_snapshot_versions"] = version_labels
        if hasattr(data, "stats_snapshot_as_of_epoch"):
            snapshot_as_of_epoch = data.stats_snapshot_as_of_epoch
            if isinstance(snapshot_as_of_epoch, torch.Tensor):
                as_of_values: List[str] = []
                for epoch in snapshot_as_of_epoch.view(-1).double().cpu().tolist():
                    iso_value = self._epoch_to_iso(epoch)
                    if iso_value is not None:
                        as_of_values.append(iso_value)
                if as_of_values:
                    contract["stats_snapshot_as_of_ts_batch"] = as_of_values
        if hasattr(data, "stats_allowed"):
            stats_allowed = data.stats_allowed
            if isinstance(stats_allowed, torch.Tensor):
                flags = [bool(item) for item in stats_allowed.view(-1).long().cpu().tolist()]
                if flags:
                    contract["stats_allowed_batch"] = flags
        if hasattr(data, "stats_missing_asof_snapshot"):
            stats_missing = data.stats_missing_asof_snapshot
            if isinstance(stats_missing, torch.Tensor):
                flags = [bool(item) for item in stats_missing.view(-1).long().cpu().tolist()]
                if flags:
                    contract["stats_missing_asof_snapshot_batch"] = flags
        return contract

    def _log_model_and_system_context(self) -> None:
        """Log model size and runtime memory/device metadata via tracker port."""
        if self.tracker is None:
            return

        total_params = 0
        skipped_uninitialized = 0
        for param in self.model.parameters():
            if isinstance(param, UninitializedParameter):
                skipped_uninitialized += 1
                continue
            total_params += int(param.numel())
        self.tracker.log_param("model_total_parameters", total_params)
        if skipped_uninitialized > 0:
            logger.warning(
                "Skipped %d uninitialized model parameters while logging model size.",
                skipped_uninitialized,
            )
            self.tracker.log_param("model_uninitialized_parameters", skipped_uninitialized)
        ram_usage_mb = float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
        self.tracker.log_param("system_ram_mb_before_train", ram_usage_mb)

        model_cfg = self.config.get("model_config", {})
        hidden_dim = int(model_cfg.get("hidden_dim", 64))
        estimated_min_vram_mb = round(((total_params * 16) / 1048576.0) + 500.0 + ((self.batch_size * 100 * hidden_dim * 3 * 8) / 1048576.0), 2)
        self.tracker.log_param("estimated_min_vram_mb", estimated_min_vram_mb)

        if self.device.type == "cuda" and torch.cuda.is_available():
            self.tracker.log_tag("system.gpu_name", torch.cuda.get_device_name(0))

    def _log_topology_metrics_and_artifacts(self) -> None:
        """Log topology summary params and histogram artifacts through tracker port."""
        if self.tracker is None:
            return

        for prefix, values in (("nodes", self._topology_nodes), ("edges", self._topology_edges)):
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            self.tracker.log_param(f"{prefix}_min", int(np.min(arr)))
            self.tracker.log_param(f"{prefix}_max", int(np.max(arr)))
            self.tracker.log_param(f"{prefix}_mean", float(np.mean(arr)))
            self.tracker.log_param(f"{prefix}_median", float(np.median(arr)))

        try:
            import matplotlib.pyplot as plt

            if self._topology_nodes and self._topology_edges:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].hist(self._topology_nodes, bins=30, color="#377eb8", alpha=0.85)
                axes[0].set_title("Nodes per graph")
                axes[1].hist(self._topology_edges, bins=30, color="#4daf4a", alpha=0.85)
                axes[1].set_title("Edges per graph")
                fig.tight_layout()
                with tempfile.NamedTemporaryFile(prefix="topology_hist_", suffix=".png", delete=False) as tmp_file:
                    figure_path = tmp_file.name
                fig.savefig(figure_path, dpi=160)
                plt.close(fig)
                self.tracker.log_artifact(figure_path)

            if self._prefix_lengths:
                fig, axis = plt.subplots(1, 1, figsize=(7, 5))
                axis.hist(self._prefix_lengths, bins=30, color="#984ea3", alpha=0.85)
                axis.set_title("Prefix length distribution")
                fig.tight_layout()
                with tempfile.NamedTemporaryFile(prefix="prefix_hist_", suffix=".png", delete=False) as tmp_file:
                    figure_path = tmp_file.name
                fig.savefig(figure_path, dpi=160)
                plt.close(fig)
                self.tracker.log_artifact(figure_path)
        except Exception as exc:
            logger.warning("Failed to generate topology histogram artifacts: %s", exc)

    def _log_data_metrics(self) -> None:
        """Log parsed data volumes and vocab sizes from prepared artifacts."""
        if self.tracker is None:
            return
        data_metrics = self.config.get("data_metrics", {})
        if isinstance(data_metrics, dict):
            for key in ("data_num_traces", "data_num_events", "vocab_activity_size", "vocab_resource_size"):
                if key in data_metrics:
                    self.tracker.log_param(key, data_metrics[key])

    def _log_params(self) -> None:
        """Log basic run parameters via tracker when available."""
        if self.tracker is None:
            return
        self.tracker.log_param("model_type", self.model.__class__.__name__)
        self.tracker.log_param("learning_rate", self.learning_rate)
        self.tracker.log_param("batch_size", self.batch_size)
        self.tracker.log_param("epochs", self.epochs)
        self.tracker.log_param("patience", self.patience)
        self.tracker.log_param("delta", self.delta)
        self.tracker.log_param("loss_function", self.loss_function)
        self.tracker.log_param("seed", self.seed)
        self.tracker.log_param("dataloader_num_workers", self.dataloader_num_workers)
        self.tracker.log_param("dataloader_pin_memory", self.dataloader_pin_memory)
        self.tracker.log_param("dataloader_persistent_workers", self.dataloader_persistent_workers)
        self.tracker.log_param("dataloader_prefetch_factor", self.dataloader_prefetch_factor)

    def _log_run_context(self) -> None:
        """Log extended run context: tags, flattened params, and feature metadata."""
        experiment_cfg = self.config.get("experiment_config", {})
        tracking_cfg = self.config.get("tracking_config", {})
        data_cfg = self.config.get("data_config", {})
        model_cfg = self.config.get("model_config", {})
        run_profile_raw = self.config.get("run_profile", {})
        run_profile = dict(run_profile_raw) if isinstance(run_profile_raw, dict) else {}

        dataset_label = str(self.config.get("dataset_label", data_cfg.get("dataset_label", experiment_cfg.get("dataset", "unknown_data")))).strip()
        model_label = str(self.config.get("model_label", model_cfg.get("model_label", model_cfg.get("type", "unknown_model")))).strip()
        model_type = str(model_cfg.get("type", model_cfg.get("model_type", "unknown_model"))).strip() or "unknown_model"
        model_family = str(run_profile.get("model_family", "eopkg" if model_type.lower().startswith("eopkg") else "baseline"))
        adapter_kind = str(run_profile.get("adapter_kind", self.config.get("mapping_config", {}).get("adapter", "xes"))).strip() or "xes"
        graph_features_enabled = bool(run_profile.get("graph_features_enabled", False))
        node_feature_count = int(run_profile.get("node_feature_count", 0))
        edge_weight_enabled = bool(run_profile.get("edge_weight_enabled", False))
        global_process_forward_enabled = bool(run_profile.get("global_process_stats_forward_enabled", False))
        stats_quality_gate_enabled = bool(run_profile.get("stats_quality_gate_enabled", False))
        stats_time_policy = str(run_profile.get("stats_time_policy", "latest"))
        on_missing_asof_snapshot = str(run_profile.get("on_missing_asof_snapshot", "disable_stats"))
        xes_use_classifier = run_profile.get("xes_use_classifier")

        logger.info("========== TRAINER PROFILE ==========")
        logger.info(
            "TRAINER_PROFILE mode=%s model=%s model_family=%s adapter=%s dataset=%s preloaded_data=%s",
            self.mode,
            model_type,
            model_family,
            adapter_kind,
            dataset_label or "unknown_data",
            bool(self.prepared_data is not None),
        )
        logger.info(
            "TRAINER_PROFILE forward struct_nodes=%s(node_features=%d) struct_edges=%s global_process=%s stats_quality_gate=%s stats_time_policy=%s",
            "on" if graph_features_enabled and node_feature_count > 0 else "off",
            node_feature_count,
            "on" if graph_features_enabled and edge_weight_enabled else "off",
            "on" if global_process_forward_enabled else "off",
            "on" if stats_quality_gate_enabled else "off",
            stats_time_policy,
        )
        if adapter_kind == "xes":
            logger.info("TRAINER_PROFILE xes use_classifier=%s", xes_use_classifier if xes_use_classifier is not None else "unknown")
        logger.info(
            "TRAINER_CHECKS forward_stats_summary=on mixed_snapshot_guard=on missing_asof_policy=%s",
            on_missing_asof_snapshot,
        )
        logger.info(
            "TRAINER_RUNTIME dataloader_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%d",
            self.dataloader_num_workers,
            self.dataloader_pin_memory,
            self.dataloader_persistent_workers if self.dataloader_num_workers > 0 else False,
            self.dataloader_prefetch_factor if self.dataloader_num_workers > 0 else 0,
        )
        logger.info("=====================================")

        if self.tracker is None:
            return
        if dataset_label:
            self.tracker.log_tag("experiment.dataset_label", dataset_label)
        if model_label:
            self.tracker.log_tag("experiment.model_label", model_label)

        custom_tags = tracking_cfg.get("tags", {})
        if isinstance(custom_tags, dict):
            for tag_key, tag_value in custom_tags.items():
                self.tracker.log_tag(str(tag_key), tag_value)

    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log train/validation metrics for one epoch via tracker port."""
        if self.tracker is None:
            return
        for key, value in metrics.items():
            self.tracker.log_metric(key, float(value), step=epoch)

    def _log_test_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log final test metrics via tracker port."""
        if self.tracker is None:
            return
        for key, value in metrics.items():
            if value is None:
                continue
            self.tracker.log_metric(key, float(value), step=self.epochs)



