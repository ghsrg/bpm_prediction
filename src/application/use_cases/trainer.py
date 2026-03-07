"""Application use-case for MVP1 model training and evaluation."""

# Відповідно до:
# - ARCHITECTURE_RULES.md -> розділ 2-4 (Application orchestration через порти)
# - EVF_MVP1.MD -> розділ 3 (Strict Temporal Split), розділ 4 (цільові метрики), розділ 5 (tracking)
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 scope, без Critic/Reliability)

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.application.ports.graph_builder_port import IGraphBuilder
from src.application.ports.prefix_policy_port import IPrefixPolicy
from src.application.ports.tracker_port import ITracker
from src.application.ports.xes_adapter_port import IXESAdapter
from src.domain.entities.raw_trace import RawTrace
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.models.base_gnn import BaseGNN


logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for chronologically split trace subsets."""

    train: List[RawTrace]
    val: List[RawTrace]
    test: List[RawTrace]


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
    ) -> None:
        self.xes_adapter = xes_adapter
        self.prefix_policy = prefix_policy
        self.graph_builder = graph_builder
        self.model = model
        self.log_path = log_path
        self.config = config
        self.tracker = tracker
        self.class_weights = class_weights

        self.batch_size = int(config.get("batch_size", 32))
        self.epochs = int(config.get("epochs", 10))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.device = torch.device(config.get("device", "cpu"))
        self.num_ece_bins = int(config.get("ece_bins", 10))
        self.class_weight_cap = float(config.get("class_weight_cap", 50.0))
        self.show_progress = bool(config.get("show_progress", True))
        self.tqdm_disable = bool(config.get("tqdm_disable", False))
        self.tqdm_leave = bool(config.get("tqdm_leave", False))
        self.loss_function = str(config.get("loss_function", "cross_entropy")).strip().lower()
        self.patience = int(config.get("patience", 6))
        self.delta = float(config.get("delta", 1e-4))
        self.config_path = str(config.get("config_path", "")).strip()
        self.seed = int(config.get("seed", 42))
        self.retrain = bool(config.get("retrain", False))
        self.mode = str(config.get("mode", self.config.get("experiment_config", {}).get("mode", "train"))).strip().lower()
        self.drift_window_size = int(config.get("drift_window_size", self.config.get("experiment_config", {}).get("drift_window_size", 500)))

        self.checkpoint_dir = str(config.get("checkpoint_dir", "checkpoints")).strip() or "checkpoints"
        checkpoint_override = str(config.get("checkpoint_path", "")).strip()
        experiment_cfg = self.config.get("experiment_config", {})
        experiment_name = str(experiment_cfg.get("name", "default_experiment")).strip() or "default_experiment"
        safe_experiment_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in experiment_name)
        self.experiment_name = safe_experiment_name
        self.checkpoint_path = Path(checkpoint_override) if checkpoint_override else (Path(self.checkpoint_dir) / f"{self.experiment_name}_best.pth")

        self._topology_nodes: List[int] = []
        self._topology_edges: List[int] = []
        self._prefix_lengths: List[int] = []
        self._logged_peak_vram = False

    def run(self) -> Dict[str, Any]:
        """Execute full training flow: data prep, train/val loop, and final test."""
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
        self._log_model_and_system_context()
        if self.tracker is not None and self.config_path:
            self.tracker.log_artifact(self.config_path)

        raw_traces = list(self.xes_adapter.read(self.log_path, self.config.get("mapping_config", {})))

        is_eval_cross = self.mode == "eval_cross_dataset"
        is_eval_drift = self.mode == "eval_drift"
        is_eval_mode = is_eval_cross or is_eval_drift

        split_data = self._prepare_split_data(raw_traces)

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
        elif self.checkpoint_path.exists() and not self.retrain:
            checkpoint = self._load_checkpoint(self.checkpoint_path, require_encoder_state=False)
            self._restore_from_checkpoint(checkpoint, require_encoder_state=False)
            best_val_loss = float(checkpoint["val_loss"])
            start_epoch = int(checkpoint["epoch"])
            best_epoch = start_epoch
            logger.info("Loaded checkpoint from epoch %d with val_loss %.6f.", start_epoch, best_val_loss)

        if is_eval_cross:
            test_loader = self._build_loader(split_data.test, shuffle=False)
            self._log_topology_metrics_and_artifacts()
            logger.info("DataLoaders ready for eval_cross_dataset: test_batches=%d", len(test_loader))
            test_metrics = self._evaluate_test(test_loader)
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

        if is_eval_drift:
            drift_traces = self._prepare_drift_traces(raw_traces)
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
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if start_epoch >= self.epochs:
            logger.info("Model already trained for %d epochs. Skipping training loop.", self.epochs)
        else:
            for epoch in range(start_epoch + 1, self.epochs + 1):
                train_loss, train_macro_f1, _, epoch_duration = self._run_epoch(train_loader, optimizer=optimizer, training=True)
                val_loss, val_macro_f1, val_weighted_f1, _ = self._run_epoch(val_loader, optimizer=None, training=False)

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
                    self._save_checkpoint(self.checkpoint_path, epoch=epoch, val_loss=best_val_loss, optimizer=optimizer)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    logger.info("Early stopping triggered at epoch %d (best_epoch=%d, best_val_loss=%.6f).", epoch, best_epoch, best_val_loss)
                    if self.tracker is not None:
                        self.tracker.log_metric("early_stopping_epoch", float(epoch), step=epoch)
                        self.tracker.log_metric("best_epoch", float(best_epoch), step=epoch)
                        self.tracker.log_metric("best_val_loss", float(best_val_loss), step=epoch)
                    break

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {self.checkpoint_path}")

        best_checkpoint = self._load_checkpoint(self.checkpoint_path, require_encoder_state=False)
        self.model.load_state_dict(best_checkpoint["model_state_dict"])
        self._restore_encoder_state(best_checkpoint, require_encoder_state=False)
        best_epoch = int(best_checkpoint["epoch"])
        best_val_loss = float(best_checkpoint["val_loss"])

        test_metrics = self._evaluate_test(test_loader)
        logger.info("Final test metrics: %s", test_metrics)
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

    def _save_checkpoint(self, checkpoint_path: Path, epoch: int, val_loss: float, optimizer: Adam) -> None:
        """Persist best model checkpoint for future resume/evaluation flows."""
        checkpoint_payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss),
        }
        feature_encoder = getattr(self.graph_builder, "feature_encoder", None)
        if feature_encoder is not None and hasattr(feature_encoder, "get_state"):
            checkpoint_payload["encoder_state"] = feature_encoder.get_state()
        torch.save(checkpoint_payload, checkpoint_path)

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
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._restore_encoder_state(checkpoint, require_encoder_state=require_encoder_state)

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
        """Apply fraction and strict temporal split using config-driven strategy."""
        data_cfg = self.config.get("data_config", {})
        fraction = float(data_cfg.get("fraction", 1.0))
        split_strategy = str(data_cfg.get("split_strategy", "time")).strip().lower()
        split_ratio = self._parse_split_ratio(data_cfg.get("split_ratio", [0.7, 0.2, 0.1]))
        ordered = self._apply_fraction(traces=traces, fraction=fraction)
        return self._strict_temporal_split(ordered=ordered, split_ratio=split_ratio, split_strategy=split_strategy)

    def _prepare_drift_traces(self, traces: Sequence[RawTrace]) -> List[RawTrace]:
        """Prepare full chronological trace stream for drift evaluation windows."""
        data_cfg = self.config.get("data_config", {})
        fraction = float(data_cfg.get("fraction", 1.0))
        return self._apply_fraction(traces=traces, fraction=fraction)

    def _parse_split_ratio(self, raw_ratio: Any) -> Tuple[float, float, float]:
        """Validate configured train/val/test split ratio."""
        if not isinstance(raw_ratio, Sequence) or isinstance(raw_ratio, (str, bytes)) or len(raw_ratio) != 3:
            raise ValueError("data.split_ratio must be a 3-item list [train, val, test].")
        ratios = [float(item) for item in raw_ratio]
        if any(item < 0.0 for item in ratios):
            raise ValueError("data.split_ratio must contain non-negative values.")
        ratio_sum = sum(ratios)
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"data.split_ratio must sum to 1.0, got {ratio_sum:.6f}")
        return ratios[0], ratios[1], ratios[2]

    def _apply_fraction(self, traces: Sequence[RawTrace], fraction: float) -> List[RawTrace]:
        """Select chronological subset of traces by fraction."""
        traces_with_events = [trace for trace in traces if trace.events]
        ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)
        if fraction >= 1.0:
            return ordered
        if fraction <= 0.0:
            raise ValueError("data.fraction must be within (0.0, 1.0].")
        keep_count = max(1, int(len(ordered) * fraction))
        return ordered[:keep_count]

    def _strict_temporal_split(self, ordered: Sequence[RawTrace], split_ratio: Tuple[float, float, float], split_strategy: str) -> SplitData:
        """Apply strict chronological split by configured strategy and ratio."""
        if split_strategy != "time":
            raise ValueError(f"Unsupported data.split_strategy '{split_strategy}'. Only 'time' is allowed for MVP1.")
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
                self._topology_nodes.append(num_nodes)
                self._topology_edges.append(num_edges)
                self._prefix_lengths.append(prefix_length)
                graphs.append(
                    Data(
                        x_cat=contract["x_cat"],
                        x_num=contract["x_num"],
                        edge_index=contract["edge_index"],
                        edge_type=contract["edge_type"],
                        y=contract["y"],
                        num_nodes=num_nodes,
                    )
                )
        return DataLoader(graphs, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(self, loader: Iterable[Data], optimizer: Optional[Adam], training: bool) -> Tuple[float, float, float, float]:
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

        iterator = tqdm(loader, desc=("Train epoch" if training else "Validation epoch"), leave=self.tqdm_leave, disable=(not self.show_progress) or self.tqdm_disable)
        for data in iterator:
            data = data.to(self.device)
            contract = self._data_to_contract(data)
            if training and optimizer is not None:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = self.model(contract)
                targets = data.y.view(-1).long()
                loss = self.criterion(logits, targets)
                if training and optimizer is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            batches += 1
            all_pred.extend(torch.argmax(logits.detach(), dim=1).cpu().numpy().tolist())
            all_true.extend(targets.detach().cpu().numpy().tolist())

        duration = perf_counter() - started
        if batches == 0:
            return 0.0, 0.0, 0.0, duration

        macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(all_true, all_pred, average="weighted", zero_division=0))
        avg_loss = total_loss / batches
        return avg_loss, macro_f1, weighted_f1, duration

    def _evaluate_test(self, loader: Iterable[Data]) -> Dict[str, float]:
        """Compute final test metrics after training is complete."""
        self.model.eval()
        all_true: List[int] = []
        all_pred: List[int] = []
        all_probs: List[np.ndarray] = []
        inference_graphs = 0
        inference_started = perf_counter()

        with torch.no_grad():
            iterator = tqdm(loader, desc="Test evaluation", leave=self.tqdm_leave, disable=(not self.show_progress) or self.tqdm_disable)
            for data in iterator:
                data = data.to(self.device)
                contract = self._data_to_contract(data)
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                logits = self.model(contract)
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                probs = torch.softmax(logits, dim=1)
                targets = data.y.view(-1).long().cpu().numpy()
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                all_true.extend(targets.tolist())
                all_pred.extend(preds.tolist())
                all_probs.extend(probs.cpu().numpy())
                inference_graphs += int(targets.shape[0])

        total_inference_ms = float((perf_counter() - inference_started) * 1000.0)
        inference_ms_per_graph = float(total_inference_ms / inference_graphs) if inference_graphs > 0 else 0.0

        if not all_true:
            return {
                "test_macro_f1": 0.0,
                "test_weighted_f1": 0.0,
                "test_accuracy": 0.0,
                "test_top3_accuracy": 0.0,
                "test_weighted_f1": 0.0,
                "test_ece": 0.0,
                "test_precision_macro": 0.0,
                "test_recall_macro": 0.0,
                "test_inference_time_ms_per_graph": inference_ms_per_graph,
            }

        y_true = np.asarray(all_true)
        y_pred = np.asarray(all_pred)
        y_prob = np.asarray(all_probs)
        num_classes = y_prob.shape[1]
        top_k = min(3, num_classes)

        return {
            "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_top3_accuracy": float(top_k_accuracy_score(y_true, y_prob, k=top_k, labels=list(range(num_classes)))),
            "test_ece": float(self._expected_calibration_error(y_true, y_prob)),
            "test_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_inference_time_ms_per_graph": inference_ms_per_graph,
        }

    def _evaluate_drift_windows(self, traces: Sequence[RawTrace]) -> List[Dict[str, float]]:
        """Run fixed-size chronological window evaluation for concept-drift tracking."""
        if self.drift_window_size <= 0:
            raise ValueError("drift_window_size must be positive.")

        ordered_traces = sorted((trace for trace in traces if trace.events), key=lambda trace: trace.events[0].timestamp)
        drift_results: List[Dict[str, float]] = []
        window_starts = list(range(0, len(ordered_traces), self.drift_window_size))

        iterator = tqdm(
            window_starts,
            desc="Eval drift",
            leave=self.tqdm_leave,
            disable=(not self.show_progress) or self.tqdm_disable,
        )
        for window_idx, start in enumerate(iterator):
            window_traces = ordered_traces[start : start + self.drift_window_size]
            window_loader = self._build_loader(window_traces, shuffle=False)
            metrics = self._evaluate_test(window_loader)
            macro_f1 = float(metrics.get("test_macro_f1", 0.0))
            ece = float(metrics.get("test_ece", 0.0))

            iterator.set_postfix({"f1": f"{macro_f1:.4f}", "ece": f"{ece:.4f}"})

            if self.tracker is not None:
                self.tracker.log_metric("drift_window_macro_f1", macro_f1, step=window_idx)
                self.tracker.log_metric("drift_window_test_ece", ece, step=window_idx)

            drift_results.append(
                {
                    "window_index": float(window_idx),
                    "window_start_trace": float(start),
                    "window_end_trace": float(start + len(window_traces) - 1),
                    "window_macro_f1": macro_f1,
                    "window_test_ece": ece,
                }
            )

        return drift_results

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

    def _data_to_contract(self, data: Data) -> GraphTensorContract:
        """Convert PyG batch Data to GraphTensorContract for model forward."""
        edge_type = data.edge_type if hasattr(data, "edge_type") else torch.zeros(data.edge_index.shape[1], dtype=torch.long)
        num_nodes = int(data.x_cat.size(0))
        batch_tensor = data.batch if hasattr(data, "batch") else torch.zeros(num_nodes, dtype=torch.long, device=data.x_cat.device)
        if batch_tensor.numel() != num_nodes:
            batch_tensor = torch.zeros(num_nodes, dtype=torch.long, device=data.x_cat.device)
        return GraphTensorContract(
            x_cat=data.x_cat,
            x_num=data.x_num,
            edge_index=data.edge_index,
            edge_type=edge_type,
            y=data.y,
            batch=batch_tensor,
            num_nodes=num_nodes,
        )

    def _log_model_and_system_context(self) -> None:
        """Log model size and runtime memory/device metadata via tracker port."""
        if self.tracker is None:
            return

        total_params = int(sum(param.numel() for param in self.model.parameters()))
        self.tracker.log_param("model_total_parameters", total_params)
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

    def _log_run_context(self) -> None:
        """Log extended run context: tags, flattened params, and feature metadata."""
        if self.tracker is None:
            return

        experiment_cfg = self.config.get("experiment_config", {})
        tracking_cfg = self.config.get("tracking_config", {})
        data_cfg = self.config.get("data_config", {})
        model_cfg = self.config.get("model_config", {})

        dataset_label = str(self.config.get("dataset_label", data_cfg.get("dataset_label", experiment_cfg.get("dataset", "unknown_data")))).strip()
        model_label = str(self.config.get("model_label", model_cfg.get("model_label", model_cfg.get("type", "unknown_model")))).strip()
        if dataset_label:
            self.tracker.log_tag("experiment.dataset_label", dataset_label)
        if model_label:
            self.tracker.log_tag("experiment.model_label", model_label)

        custom_tags = tracking_cfg.get("tags", {})
        if isinstance(custom_tags, dict):
            for tag_key, tag_value in custom_tags.items():
                self.tracker.log_tag(str(tag_key), tag_value)

        self.tracker.log_param("data", data_cfg)
        self.tracker.log_param(
            "model",
            {
                "type": model_cfg.get("type", self.model.__class__.__name__),
                "hidden_dim": model_cfg.get("hidden_dim"),
                "dropout": model_cfg.get("dropout"),
                "graph_strategy": model_cfg.get("graph_strategy", "sequential_prefix"),
                "pooling_strategy": model_cfg.get("pooling_strategy", "global_mean"),
            },
        )
        self.tracker.log_param(
            "training",
            {
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "loss_function": self.loss_function,
                "patience": self.patience,
                "delta": self.delta,
                "retrain": self.retrain,
                "checkpoint_dir": self.checkpoint_dir,
                "ece_bins": self.num_ece_bins,
                "class_weight_cap": self.class_weight_cap,
                "mode": self.mode,
                "drift_window_size": self.drift_window_size,
            },
        )

    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log train/validation metrics for one epoch via tracker port."""
        if self.tracker is None:
            return
        for key, value in metrics.items():
            self.tracker.log_metric(key, float(value), step=epoch)

    def _log_test_metrics(self, metrics: Dict[str, float]) -> None:
        """Log final test metrics via tracker port."""
        if self.tracker is None:
            return
        for key, value in metrics.items():
            self.tracker.log_metric(key, float(value), step=self.epochs)
