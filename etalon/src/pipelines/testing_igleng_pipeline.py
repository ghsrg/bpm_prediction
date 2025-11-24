import os
import argparse
from typing import Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# --- Project utils ---
from src.utils.logger import get_logger
from src.utils.file_utils import (
    load_checkpoint, load_prepared_data, load_global_statistics_from_json
)
from src.utils.file_utils_l import join_path, is_file_exist, make_dir
from src.config.config import (
    NN_PR_MODELS_DATA_PATH,
    NN_PR_MODELS_CHECKPOINTS_PATH,
    TEST_PR_DIAGRAMS_PATH,
    PROCESSED_DATA_PATH,
)

# Try to import project visualizer; we'll have a local fallback if absent/old
try:
    from src.utils.visualizer import plot_prefix_metric_lines as plot_prefix_metric_lines_external
except Exception:
    plot_prefix_metric_lines_external = None

# --- Models ---
import src.core.core_MLP_pr as MLP_pr_core
import src.core.core_GATConv_pr as GATConv_pr_core
import src.core.core_TGAT_pr as TGAT_pr_core
import src.core.core_GCN_pr as GCN_pr_core
import src.core.core_GraphSAGE_pr as GraphSAGE_pr_core
import src.core.core_MuseGNN_pr as MuseGNN_pr_core
import src.core.core_APPNP_pr as APPNP_pr_core
import src.core.core_MPGCN_pr as MPGCN_pr_core
import src.core.core_DeepGCN_pr as DeepGCN_pr_core
import src.core.core_TemporalGAT_pr as TemporalGAT_pr_core
import src.core.core_TGCN_pr as TGCN_pr_core
import src.core.core_GRUGAT_pr as GRUGAT_pr_core
import src.core.core_TransformerMLP_pr as TransformerMLP_pr_core
import src.core.core_GraphMixer_pr as GraphMixer_pr_core
import src.core.core_GGNN_pr as GGNN_pr_core
import src.core.core_MixHop_pr as MixHop_pr_core
import src.core.core_GPRGNN_pr as GPRGNN_pr_core
import src.core.core_GATv2_pr as GATv2_pr_core
import src.core.core_Graphormer_pr as Graphormer_pr_core

logger = get_logger(__name__)
torch.set_num_threads(12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAP = {
    "APPNP_pr": (APPNP_pr_core),
    "DeepGCN_pr": (DeepGCN_pr_core),
    "MPGCN_pr": (MPGCN_pr_core),
    "GATConv_pr": (GATConv_pr_core),
    "GATv2_pr": (GATv2_pr_core),
    "GCN_pr": (GCN_pr_core),
    "GGNN_pr": (GGNN_pr_core),
    "GPRGNN_pr": (GPRGNN_pr_core),
    "GraphMixer_pr": (GraphMixer_pr_core),
    "Graphormer_pr": (Graphormer_pr_core),
    "GraphSAGE_pr": (GraphSAGE_pr_core),
    "MixHop_pr": (MixHop_pr_core),
    "MLP_pr": (MLP_pr_core),
    "MuseGNN_pr": (MuseGNN_pr_core),
    "TemporalGAT_pr": (TemporalGAT_pr_core),
    "TGAT_pr": (TGAT_pr_core),
    "TransformerMLP_pr": (TransformerMLP_pr_core),
}


# ---------------------- Helpers ----------------------

def _build_batch_like(d: dict):
    """Builds a light-weight Batch-like object expected by core models."""
    x = d["x"]
    batch_tensor = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    edge_index = d.get("edge_index")
    edge_attr = d.get("edge_attr")
    doc_features = d.get("doc_features")
    timestamps = d.get("timestamps")
    if isinstance(doc_features, torch.Tensor) and doc_features.dim() == 1:
        doc_features = doc_features.unsqueeze(0)
    return type(
        "Batch", (object,), {
            "x": x,
            "edge_index": edge_index,
            "edge_features": edge_attr,
            "batch": batch_tensor,
            "doc_features": doc_features,
            "timestamps": timestamps,
        }
    )


def _find_checkpoint_path(model_type: str, pr_mode: str, seed: int, checkpoint: str | None = None) -> str:
    if checkpoint:
        path = join_path([NN_PR_MODELS_CHECKPOINTS_PATH, f"{checkpoint}.pt"])
        if is_file_exist(path):
            return path
        raise FileNotFoundError(f"Checkpoint не знайдено: {path}")
    path = f"{NN_PR_MODELS_CHECKPOINTS_PATH}/{model_type}_{pr_mode}_seed{seed}_best.pt"
    if is_file_exist(path):
        return path
    raise FileNotFoundError(f"Не знайдено чекпоінт: {path}")


def _load_data(data_file: str):
    data_path = join_path([NN_PR_MODELS_DATA_PATH, f"{data_file}.pt"])
    data, input_dim, doc_dim, global_node_dict = load_prepared_data(data_path)
    if data is None:
        raise FileNotFoundError(f"Підготовлені дані не знайдено: {data_path}")
    # Move tensors to device
    for i in range(len(data)):
        if isinstance(data[i], list):
            for j in range(len(data[i])):
                for key, value in data[i][j].items():
                    if isinstance(value, torch.Tensor):
                        data[i][j][key] = value.to(device)
        else:
            for key, value in data[i].items():
                if isinstance(value, torch.Tensor):
                    data[i][key] = value.to(device)
    return data, input_dim, doc_dim, global_node_dict


def _init_model(model_type: str, input_dim: int, doc_dim: int, output_dim: int, data_example: dict):
    """Initialize model exactly like in training: detect edge_dim if present."""
    stat_path = join_path([PROCESSED_DATA_PATH, "normalized_statistics"])
    global_statistics = load_global_statistics_from_json(stat_path)
    max_node_count = global_statistics["node_count"]["max"]

    core_module = MODEL_MAP[model_type]
    model_class = getattr(core_module, model_type, None)
    if model_class is None:
        raise ValueError(f"Невідома модель: {model_type}")

    edge_dim = None

    edge_features = data_example.get("edge_attr", None)
    if not (isinstance(edge_features, torch.Tensor) and edge_features.ndim == 2 and edge_features.size(1) > 0):
        edge_features = data_example.get("edge_features", None)

    if isinstance(edge_features, torch.Tensor) and edge_features.ndim == 2 and edge_features.size(1) > 0:
        edge_dim = int(edge_features.size(1))

    model = model_class(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        doc_dim=doc_dim,
        edge_dim=edge_dim,
        num_nodes=max_node_count,
    ).to(device)
    return model


# ---------------------- Evaluation ----------------------

@torch.no_grad()
def _evaluate_by_prefix_len(model, data: List[dict], top_ks=(1, 3, 5)) -> pd.DataFrame:
    """
    Обчислює метрики у розрізі довжини префікса. Ключові відмінності від простої версії:
    - Використовує prefix_mask (якщо є) для L.
    - Накладає task_mask (якщо є) на логіти перед argmax/topk.
    - Рахує mean/std/min/max/count для accuracy, f1, top-k, confidence.
    """
    per_L: Dict[int, dict] = {}

    def init_bucket():
        return {
            "acc_list": [],
            "f1_list": [],
            "conf_list": [],
            **{f"top{k}_list": [] for k in top_ks}
        }

    model.eval()

    for d in tqdm(data, desc="Inference by prefix", unit="sample"):
        batch_like = _build_batch_like(d)
        logits, _ = model.forward(batch_like)  # expect shape [1, C] for graph-level output

        # ---- Prefix length (L) ----
        if "prefix_mask" in d:
            active_mask = d["prefix_mask"]
        else:
            active_mask = d["x"][:, -1]  # fallback
        prefix_len = int(active_mask.sum().item())

        # ---- Available classes mask ----
        # task_mask can be binary [C] or indices; we expect binary here
        avail_idx = None
        if "task_mask" in d and isinstance(d["task_mask"], torch.Tensor):
            tm = d["task_mask"].view(-1).bool()
            if tm.numel() == logits.size(1):
                avail_idx = torch.arange(tm.numel(), device=tm.device)[tm]
                logits_avail = logits[:, tm]
            else:
                # size mismatch => ignore mask
                logits_avail = logits

        else:
            logits_avail = logits

        # ---- Predictions / Targets ----
        y = d["y"].view(-1)  # expect [1]
        if logits_avail.dim() == 1:
            logits_avail = logits_avail.unsqueeze(0)

        pred_local = torch.argmax(logits_avail, dim=1)  # local within avail
        if avail_idx is not None:
            pred = avail_idx[pred_local]
        else:
            pred = pred_local

        # Accuracy & F1 for a single example
        acc = accuracy_score(y.cpu().tolist(), pred.cpu().tolist())
        f1 = f1_score(y.cpu().tolist(), pred.cpu().tolist(), average="macro", zero_division=0)

        # Confidence of top-1
        probs = torch.softmax(logits_avail, dim=1)
        conf = torch.max(probs, dim=1).values.item()

        # Top-k hits
        hits_k = {}
        for k in top_ks:
            k_eff = min(k, probs.size(1))
            topk_local = torch.topk(probs, k=k_eff, dim=1).indices[0]
            if avail_idx is not None:
                topk_global = avail_idx[topk_local]
            else:
                topk_global = topk_local
            hits_k[k] = int(int(y.item()) in topk_global.detach().cpu().tolist())

        bucket = per_L.setdefault(prefix_len, init_bucket())
        bucket["acc_list"].append(acc)
        bucket["f1_list"].append(f1)
        bucket["conf_list"].append(conf)
        for k in top_ks:
            bucket[f"top{k}_list"].append(hits_k[k])

    # ---- Aggregate per L ----
    rows = []
    for L in sorted(per_L.keys()):
        b = per_L[L]
        count = len(b["acc_list"])
        if count == 0:
            continue
        row = {
            "prefix_len": L,
            "count": count,
            "accuracy_mean": float(np.mean(b["acc_list"])),
            "accuracy_std": float(np.std(b["acc_list"])),
            "accuracy_min": float(np.min(b["acc_list"])),
            "accuracy_max": float(np.max(b["acc_list"])),
            "f1_macro_mean": float(np.mean(b["f1_list"])),
            "f1_macro_std": float(np.std(b["f1_list"])),
            "f1_macro_min": float(np.min(b["f1_list"])),
            "f1_macro_max": float(np.max(b["f1_list"])),
            "conf_mean": float(np.mean(b["conf_list"])),
            "conf_std": float(np.std(b["conf_list"])),
            "conf_min": float(np.min(b["conf_list"])),
            "conf_max": float(np.max(b["conf_list"])),
        }
        for k in top_ks:
            vals = b[f"top{k}_list"]
            row[f"top{k}_mean"] = float(np.mean(vals))
            row[f"top{k}_std"] = float(np.std(vals))
            row[f"top{k}_min"] = float(np.min(vals))
            row[f"top{k}_max"] = float(np.max(vals))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------- Plotting (fallback) ----------------------

def _plot_prefix_metric_lines_fallback(df: pd.DataFrame, model_type: str, pr_mode: str, base_path: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(base_path, exist_ok=True)

    def _plot_metric(metric: str):
        if f"{metric}_mean" not in df.columns:
            return
        x = df["prefix_len"].values
        mean = df[f"{metric}_mean"].values
        std = df.get(f"{metric}_std", pd.Series([0] * len(df))).values
        lower = np.clip(mean - std, 0, 1)
        upper = np.clip(mean + std, 0, 1)
        min_vals = df.get(f"{metric}_min")
        max_vals = df.get(f"{metric}_max")

        plt.figure(figsize=(10, 4))
        plt.plot(x, mean, label="mean")
        plt.fill_between(x, lower, upper, alpha=0.2, label="± std")
        if min_vals is not None and max_vals is not None:
            plt.plot(x, min_vals, linestyle="dashed", linewidth=1, color="gray", label="min")
            plt.plot(x, max_vals, linestyle="dashed", linewidth=1, color="gray", label="max")
        plt.title(f"{model_type} ({pr_mode}) — {metric} mean ± std")
        plt.xlabel("prefix length")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}/{model_type}_{pr_mode}_{metric}_lineband.png")
        plt.close()

    for metric in ["accuracy", "f1_macro", "conf", "top1", "top3", "top5"]:
        _plot_metric(metric)

    # Bar plot for counts
    if "count" in df.columns:
        plt.figure(figsize=(10, 4))
        ax = sns.barplot(data=df, x="prefix_len", y="count", color="skyblue")
        plt.title(f"{model_type} ({pr_mode}) — Кількість прикладів по довжині префікса")
        plt.xlabel("prefix length")
        plt.ylabel("count")
        plt.grid(True, axis='y', alpha=0.3)
        if len(df["prefix_len"]) > 30:
            step = max(1, len(df["prefix_len"]) // 30)
            for idx, label in enumerate(ax.get_xticklabels()):
                if idx % step != 0:
                    label.set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{base_path}/{model_type}_{pr_mode}_prefix_count_bar.png")
        plt.close()


# ---------------------- Runner ----------------------

def test_model(model_type: str,
               seed: int,
               anomaly_type: str | None = None,
               resume: bool = False,
               checkpoint: str = "",
               data_file: str | None = None,
               pr_mode: str | None = None) -> pd.DataFrame:
    """Main entry: loads data/model, runs per-prefix evaluation, saves CSV & plots."""
    if model_type not in MODEL_MAP:
        raise ValueError(f"Невідомий тип моделі: {model_type}")

    if pr_mode is None:
        if data_file and data_file.endswith("_logs"):
            pr_mode = "logs"
        else:
            pr_mode = "bpmn"

    if not data_file:
        data_file = f"data_{model_type}_{pr_mode}"

    data, input_dim, doc_dim, global_node_dict = _load_data(data_file)
    output_dim = len(global_node_dict)

    # Ініціалізуємо модель з edge_dim як у train (через приклад із даних)
    example = data[0] if isinstance(data, list) else data
    model = _init_model(model_type, input_dim, doc_dim, output_dim, example)

    ckpt_path = _find_checkpoint_path(model_type, pr_mode, seed, checkpoint or None)
    _epoch, _loss, _stats = load_checkpoint(ckpt_path, model)
    model.eval()
    logger.info(f"Чекпоінт завантажено: {ckpt_path}")

    df = _evaluate_by_prefix_len(model, data)
    df.insert(0, "mode", pr_mode)
    df.insert(0, "model", model_type)

    make_dir(TEST_PR_DIAGRAMS_PATH)
    out_csv = join_path([TEST_PR_DIAGRAMS_PATH, f"{model_type}_{pr_mode}_prefix_metrics.csv"])
    df.to_csv(out_csv, index=False)
    logger.info(f"Результати збережено у {out_csv}")

    # Plotting: use external if available, else fallback
    if plot_prefix_metric_lines_external is not None:
        plot_prefix_metric_lines_external(df=df, model_type=model_type, pr_mode=pr_mode,
                                          base_path=TEST_PR_DIAGRAMS_PATH)
    else:
        _plot_prefix_metric_lines_fallback(df=df, model_type=model_type, pr_mode=pr_mode,
                                           base_path=TEST_PR_DIAGRAMS_PATH)

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-prefix metrics for PR models.")
    parser.add_argument("--model_type", type=str, required=True, help="Model key from MODEL_MAP (e.g., GGNN_pr)")
    parser.add_argument("--seed", type=int, required=True, help="Seed used in checkpoint naming")
    parser.add_argument("--data_file", type=str, default=None, help="Prepared data file stem without extension")
    parser.add_argument("--pr_mode", type=str, default=None, choices=["bpmn", "logs"],
                        help="Mode for checkpoint naming")
    parser.add_argument("--checkpoint", type=str, default="", help="Explicit checkpoint name (without .pt)")

    args = parser.parse_args()
    test_model(
        model_type=args.model_type,
        seed=args.seed,
        data_file=args.data_file,
        pr_mode=args.pr_mode,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
