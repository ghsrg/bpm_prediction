"""Application use-case for MVP1 model training and evaluation."""

# Відповідно до:
# - ARCHITECTURE_RULES.md -> розділ 2-4 (Application orchestration через порти)
# - EVF_MVP1.MD -> розділ 3 (Strict Temporal Split), розділ 4 (цільові метрики), розділ 5 (tracking)
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 scope, без Critic/Reliability)

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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
        self.show_progress = bool(config.get("show_progress", True))
        self.tqdm_disable = bool(config.get("tqdm_disable", False))
        self.tqdm_leave = bool(config.get("tqdm_leave", False))

    def run(self) -> Dict[str, Any]:
        """Execute full training flow: data prep, train/val loop, and final test."""
        self.model.to(self.device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self._log_params()

        raw_traces = list(self.xes_adapter.read(self.log_path, self.config.get("mapping_config", {})))
        split_data = self._strict_temporal_split(raw_traces)

        train_loader = self._build_loader(split_data.train, shuffle=True)
        val_loader = self._build_loader(split_data.val, shuffle=False)
        test_loader = self._build_loader(split_data.test, shuffle=False)

        logger.info(
            "DataLoaders ready: train_batches=%d, val_batches=%d, test_batches=%d",
            len(train_loader),
            len(val_loader),
            len(test_loader),
        )

        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        history: List[Dict[str, float]] = []
        for epoch in range(1, self.epochs + 1):
            train_loss, train_macro_f1, epoch_duration = self._run_epoch(
                train_loader,
                optimizer=optimizer,
                training=True,
            )
            val_loss, val_macro_f1, _ = self._run_epoch(
                val_loader,
                optimizer=None,
                training=False,
            )

            epoch_metrics = {
                "train_loss": train_loss,
                "train_macro_f1": train_macro_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "epoch_duration_sec": epoch_duration,
            }
            history.append(epoch_metrics)
            self._log_epoch_metrics(epoch=epoch, metrics=epoch_metrics)
            logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | val_macro_f1=%.6f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_macro_f1,
            )

        test_metrics = self._evaluate_test(test_loader)
        logger.info("Final test metrics: %s", test_metrics)
        self._log_test_metrics(test_metrics)

        return {
            "history": history,
            "test_metrics": test_metrics,
            "split_sizes": {
                "train": len(split_data.train),
                "val": len(split_data.val),
                "test": len(split_data.test),
            },
        }

    def _strict_temporal_split(self, traces: Sequence[RawTrace]) -> SplitData:
        """Apply strict chronological split (70/10/20) by first event timestamp."""
        traces_with_events = [trace for trace in traces if trace.events]
        ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)

        total = len(ordered)
        train_end = int(total * 0.7)
        val_end = train_end + int(total * 0.1)

        return SplitData(
            train=list(ordered[:train_end]),
            val=list(ordered[train_end:val_end]),
            test=list(ordered[val_end:]),
        )

    def _build_loader(self, traces: Sequence[RawTrace], shuffle: bool) -> DataLoader:
        """Convert traces into graph tensors and wrap them into PyG DataLoader."""
        graphs: List[Data] = []
        for trace in traces:
            slices = self.prefix_policy.generate_slices(trace)
            for prefix_slice in slices:
                contract = self.graph_builder.build_graph(prefix_slice)
                graphs.append(
                    Data(
                        x=contract["x"],
                        edge_index=contract["edge_index"],
                        edge_type=contract["edge_type"],
                        y=contract["y"],
                    )
                )
        return DataLoader(graphs, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: Iterable[Data],
        optimizer: Optional[Adam],
        training: bool,
    ) -> Tuple[float, float, float]:
        """Run one epoch (train or eval) and return loss/macro_f1/duration."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        started = perf_counter()
        total_loss = 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        batches = 0

        phase_name = "Train" if training else "Validation"
        iterator = tqdm(
            loader,
            desc=f"{phase_name} epoch",
            leave=self.tqdm_leave,
            disable=(not self.show_progress) or self.tqdm_disable,
        )

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
                    optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            batches += 1

            preds = torch.argmax(logits.detach(), dim=1).cpu().numpy().tolist()
            truth = targets.detach().cpu().numpy().tolist()
            all_pred.extend(preds)
            all_true.extend(truth)

        duration = perf_counter() - started
        if batches == 0:
            return 0.0, 0.0, duration

        macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
        avg_loss = total_loss / batches
        return avg_loss, macro_f1, duration

    def _evaluate_test(self, loader: Iterable[Data]) -> Dict[str, float]:
        """Compute final test metrics after training is complete."""
        self.model.eval()

        all_true: List[int] = []
        all_pred: List[int] = []
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            iterator = tqdm(
                loader,
                desc="Test evaluation",
                leave=self.tqdm_leave,
                disable=(not self.show_progress) or self.tqdm_disable,
            )
            for data in iterator:
                data = data.to(self.device)
                contract = self._data_to_contract(data)
                logits = self.model(contract)
                probs = torch.softmax(logits, dim=1)

                targets = data.y.view(-1).long().cpu().numpy()
                preds = torch.argmax(probs, dim=1).cpu().numpy()

                all_true.extend(targets.tolist())
                all_pred.extend(preds.tolist())
                all_probs.extend(probs.cpu().numpy())

        if not all_true:
            return {
                "test_macro_f1": 0.0,
                "test_accuracy": 0.0,
                "test_top3_accuracy": 0.0,
                "test_ece": 0.0,
                "test_precision_macro": 0.0,
                "test_recall_macro": 0.0,
            }

        y_true = np.asarray(all_true)
        y_pred = np.asarray(all_pred)
        y_prob = np.asarray(all_probs)

        num_classes = y_prob.shape[1]
        top_k = min(3, num_classes)

        test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        test_accuracy = float(accuracy_score(y_true, y_pred))
        test_top3_accuracy = float(
            top_k_accuracy_score(
                y_true,
                y_prob,
                k=top_k,
                labels=list(range(num_classes)),
            )
        )
        test_ece = float(self._expected_calibration_error(y_true, y_prob))

        return {
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_accuracy,
            "test_top3_accuracy": test_top3_accuracy,
            "test_ece": test_ece,
            "test_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

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
        return GraphTensorContract(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=edge_type,
            y=data.y,
            batch=data.batch,
        )

    def _log_params(self) -> None:
        """Log basic run parameters via tracker when available."""
        if self.tracker is None:
            return
        self.tracker.log_param("model_type", self.model.__class__.__name__)
        self.tracker.log_param("learning_rate", self.learning_rate)
        self.tracker.log_param("batch_size", self.batch_size)
        self.tracker.log_param("epochs", self.epochs)

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
