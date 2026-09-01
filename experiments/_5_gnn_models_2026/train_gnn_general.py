#!/usr/bin/env python3
"""CLI version of notebook 3.1: baseline SVC + GNN training for general users.

This script preserves the notebook's training flow while exposing the main
configuration from cells 4 and 5 as command-line arguments.
"""

import argparse
import gc
import glob
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from random import sample
from random import shuffle as _shuffle_list

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.svm import SVC
from torch_geometric.data import Batch, Data

from gnn_models import RetweetGNN, train_model
from tw_dataset.settings import IG_GRAPH_PATH
from utils import create_gnn_train_val_samples, load_dataframe_raw

warnings.filterwarnings("ignore")
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

TRAIN_GROUPS = ["u_train", "au_train"]


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Argparse formatter that shows defaults and preserves newlines."""


class TeeLogger:
    """Tee stdout to both the original stream and a timestamped log file."""

    def __init__(self, log_path, original_stdout):
        self._original = original_stdout
        self._file = open(log_path, "a", buffering=1)
        self._line_buffer = ""

    def write(self, msg):
        self._original.write(msg)
        for char in msg:
            if char == "\n":
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._file.write(f"[{timestamp}] {self._line_buffer}\n")
                self._line_buffer = ""
            else:
                self._line_buffer += char

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        if self._line_buffer:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._file.write(f"[{timestamp}] {self._line_buffer}\n")
            self._line_buffer = ""
        self._file.close()


@contextmanager
def tee_to_log(log_path):
    """Context manager that tees stdout to a timestamped log file."""
    original_stdout = sys.stdout
    tee = TeeLogger(log_path, original_stdout)
    sys.stdout = tee
    try:
        yield log_path
    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"Training log saved to: {log_path}")


class ChunkedGNNTrainLoader:
    """Training loader that transforms GNN samples on-the-fly in user chunks."""

    def __init__(self, user_items, user_data, graph, batch_size, chunk_size=10, target_pos_rate=None):
        self.user_items = list(user_items)
        self.user_data = user_data
        self.graph = graph
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.target_pos_rate = target_pos_rate
        self._failed_users = []
        self._estimated_samples = None

    @property
    def total_samples(self):
        if self._estimated_samples is None:
            self._estimated_samples = sum(self.get_label_counts().values())
        return self._estimated_samples

    def __len__(self):
        return self.total_samples // self.batch_size

    def __iter__(self):
        users = list(self.user_items)
        _shuffle_list(users)

        total_yielded = 0
        self._failed_users = []

        for chunk_start in range(0, len(users), self.chunk_size):
            chunk_users = users[chunk_start : chunk_start + self.chunk_size]
            chunk_data_list = []

            for group, uid in chunk_users:
                X_tr, X_te, y_tr, y_te = self.user_data[group][uid]
                try:
                    train_samples, _ = create_gnn_train_val_samples(uid, self.graph, X_tr, y_tr, X_te, y_te)
                    for sample_dict in train_samples:
                        chunk_data_list.append(_sample_to_pyg_data(sample_dict))
                except Exception as exc:
                    self._failed_users.append((group, uid, str(exc)))
                    continue

            if not chunk_data_list:
                continue

            if self.target_pos_rate is not None:
                pos_idx = [i for i, data in enumerate(chunk_data_list) if data.y.item() == 1]
                n_pos = len(pos_idx)
                n_neg = len(chunk_data_list) - n_pos
                if n_pos > 0 and n_neg > 0:
                    target_n_pos = int(np.ceil(self.target_pos_rate * n_neg / (1.0 - self.target_pos_rate)))
                    n_extra = target_n_pos - n_pos
                    if n_extra > 0:
                        extra_idx = np.random.choice(pos_idx, size=n_extra, replace=True)
                        extra_samples = [chunk_data_list[i] for i in extra_idx]
                        chunk_data_list.extend(extra_samples)

            _shuffle_list(chunk_data_list)

            for start in range(0, len(chunk_data_list) - self.batch_size + 1, self.batch_size):
                batch = Batch.from_data_list(chunk_data_list[start : start + self.batch_size])
                yield batch
                total_yielded += self.batch_size

            del chunk_data_list
            gc.collect()

        if total_yielded > 0:
            self._estimated_samples = total_yielded

    def get_label_counts(self):
        counts = {}
        for group, uid in self.user_items:
            _, _, y_tr, _ = self.user_data[group][uid]
            y_arr = np.asarray(y_tr).ravel()
            for label in y_arr:
                counts[int(label)] = counts.get(int(label), 0) + 1

        if self.target_pos_rate is not None and 1 in counts and 0 in counts:
            n_neg = counts[0]
            target_n_pos = int(np.ceil(self.target_pos_rate * n_neg / (1.0 - self.target_pos_rate)))
            if target_n_pos > counts[1]:
                counts[1] = target_n_pos

        return counts


class CachedValLoader:
    """Validation loader that keeps only one chunk in memory at a time."""

    CHUNK_SIZE = 1000

    def __init__(self, val_user_splits, user_data, graph, batch_size, max_samples=None, cache_path=None):
        self.batch_size = batch_size
        self._cache_path = cache_path
        self._total_samples = 0
        self._label_counts = {}
        self._failed_users = []
        self._chunk_files = []

        if cache_path and os.path.isdir(cache_path):
            meta_path = os.path.join(cache_path, "metadata.json")
            chunk_files = sorted(
                f for f in os.listdir(cache_path) if f.startswith("chunk_") and f.endswith(".pt")
            )
            if chunk_files and os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self._chunk_files = [os.path.join(cache_path, cf) for cf in chunk_files]
                self._total_samples = meta["total_samples"]
                self._label_counts = {int(k): v for k, v in meta["label_counts"].items()}
                print(f"Loaded val cache metadata from: {cache_path}")
                print(f"  {self._total_samples} samples, {len(self._chunk_files)} chunks, {len(self)} batches")
                return

        assert cache_path, "cache_path is required for CachedValLoader (streaming mode)"
        os.makedirs(cache_path, exist_ok=True)

        n_total = len(val_user_splits)
        print(f"Pre-computing validation samples ({n_total} user/split combos)...")

        row_counts = []
        for group, uid, split_name in val_user_splits:
            X_tr, X_te, _, _ = user_data[group][uid]
            row_counts.append(X_tr.shape[0] if split_name == "train" else X_te.shape[0])

        total_available_rows = sum(row_counts)
        allocations = _proportional_allocate(row_counts, max_samples)
        do_subsample = bool(max_samples) and total_available_rows > max_samples
        if do_subsample:
            print(
                f"  Sampling enabled: {max_samples} target from {total_available_rows} available rows"
                f" (allocated {sum(allocations)})"
            )

        pending_samples = []
        chunk_idx = 0

        def _flush_chunk():
            nonlocal pending_samples, chunk_idx
            if not pending_samples:
                return
            chunk_path = os.path.join(cache_path, f"chunk_{chunk_idx:03d}.pt")
            torch.save(pending_samples, chunk_path)
            self._chunk_files.append(chunk_path)
            self._total_samples += len(pending_samples)
            for data in pending_samples:
                label = data.y.item()
                self._label_counts[label] = self._label_counts.get(label, 0) + 1
            chunk_idx += 1
            pending_samples = []
            gc.collect()

        for i, (group, uid, split_name) in enumerate(val_user_splits):
            if (i + 1) % 5 == 0 or (i + 1) == n_total:
                print(
                    f"  [{i+1}/{n_total}] {self._total_samples + len(pending_samples)} samples, "
                    f"{chunk_idx} chunks saved...",
                    end="\r",
                )

            X_tr, X_te, y_tr, y_te = user_data[group][uid]
            n_keep = allocations[i]

            try:
                if do_subsample:
                    if split_name == "test":
                        idx = np.random.choice(X_te.shape[0], size=min(n_keep, X_te.shape[0]), replace=False)
                        _, test_samples = create_gnn_train_val_samples(
                            uid,
                            graph,
                            X_tr.iloc[:1],
                            y_tr.iloc[:1],
                            X_te.iloc[idx],
                            y_te.iloc[idx],
                        )
                        samples = test_samples
                    else:
                        idx = np.random.choice(X_tr.shape[0], size=min(n_keep, X_tr.shape[0]), replace=False)
                        train_samples, _ = create_gnn_train_val_samples(
                            uid,
                            graph,
                            X_tr.iloc[idx],
                            y_tr.iloc[idx],
                            X_te.iloc[:1],
                            y_te.iloc[:1],
                        )
                        samples = train_samples
                else:
                    train_samples, test_samples = create_gnn_train_val_samples(uid, graph, X_tr, y_tr, X_te, y_te)
                    samples = train_samples if split_name == "train" else test_samples

                for sample_dict in samples:
                    pending_samples.append(_sample_to_pyg_data(sample_dict))
                    if len(pending_samples) >= self.CHUNK_SIZE:
                        _flush_chunk()
            except Exception as exc:
                self._failed_users.append((group, uid, str(exc)))
                continue

        _flush_chunk()
        print()

        meta_path = os.path.join(cache_path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"total_samples": self._total_samples, "label_counts": self._label_counts}, f)

        gc.collect()
        print(
            f"  Saved {self._total_samples} val samples in {len(self._chunk_files)} chunks to: {cache_path}"
            f" (from {n_total - len(self._failed_users)} user/split combos, "
            f"rows available: {total_available_rows}, failed: {len(self._failed_users)})"
        )

    @property
    def total_samples(self):
        return self._total_samples

    def __len__(self):
        return -(-self._total_samples // self.batch_size)

    def __iter__(self):
        carry_over = []
        for chunk_path in self._chunk_files:
            chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
            samples = carry_over + chunk_data
            del chunk_data

            start = 0
            while start + self.batch_size <= len(samples):
                batch = Batch.from_data_list(samples[start : start + self.batch_size])
                yield batch
                start += self.batch_size

            carry_over = samples[start:]
            del samples
            gc.collect()

        if carry_over:
            yield Batch.from_data_list(carry_over)

    def get_label_counts(self):
        return dict(self._label_counts)


class SampledTrainEvalLoader:
    """Train-eval loader that samples a fixed subset of rows per user."""

    CHUNK_SIZE = 10

    def __init__(self, train_user_items, user_data, graph, batch_size, max_samples, metadata_path):
        self.user_data = user_data
        self.graph = graph
        self.batch_size = batch_size
        self._failed_users = []
        self._total_samples = 0
        self._label_counts = {}

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            self._allocations = meta["allocations"]
            self._total_samples = meta["total_samples"]
            self._label_counts = {int(k): v for k, v in meta["label_counts"].items()}
            print(f"Loaded train-eval metadata from: {metadata_path}")
            print(f"  {self._total_samples} samples across {len(self._allocations)} users")
        else:
            row_counts = []
            for group, uid in train_user_items:
                X_tr, _, _, _ = user_data[group][uid]
                row_counts.append(X_tr.shape[0])

            total_available = sum(row_counts)
            allocations = _proportional_allocate(row_counts, max_samples)

            self._allocations = []
            for i, (group, uid) in enumerate(train_user_items):
                n_keep = allocations[i]
                X_tr, _, y_tr, _ = user_data[group][uid]
                n_available = X_tr.shape[0]
                idx = np.random.choice(n_available, size=min(n_keep, n_available), replace=False)
                idx_sorted = sorted(idx.tolist())
                self._allocations.append({"group": group, "uid": uid, "row_indices": idx_sorted})

                y_arr = np.asarray(y_tr).ravel()
                for label in y_arr[idx]:
                    self._label_counts[int(label)] = self._label_counts.get(int(label), 0) + 1

            self._total_samples = sum(len(a["row_indices"]) for a in self._allocations)

            meta = {
                "total_samples": self._total_samples,
                "label_counts": self._label_counts,
                "allocations": self._allocations,
            }
            with open(metadata_path, "w") as f:
                json.dump(meta, f)
            print(f"Created train-eval metadata: {metadata_path}")
            print(
                f"  {self._total_samples} samples from {total_available} available "
                f"({len(self._allocations)} users)"
            )

    @property
    def total_samples(self):
        return self._total_samples

    def __len__(self):
        return -(-self._total_samples // self.batch_size)

    def get_label_counts(self):
        return dict(self._label_counts)

    def __iter__(self):
        self._failed_users = []
        buffer = []

        for chunk_start in range(0, len(self._allocations), self.CHUNK_SIZE):
            chunk_allocs = self._allocations[chunk_start : chunk_start + self.CHUNK_SIZE]

            for alloc in chunk_allocs:
                group, uid = alloc["group"], alloc["uid"]
                row_indices = alloc["row_indices"]
                X_tr, X_te, y_tr, y_te = self.user_data[group][uid]

                try:
                    X_sub = X_tr.iloc[row_indices]
                    y_sub = y_tr.iloc[row_indices]
                    train_samples, _ = create_gnn_train_val_samples(
                        uid,
                        self.graph,
                        X_sub,
                        y_sub,
                        X_te.iloc[:1],
                        y_te.iloc[:1],
                    )
                    for sample_dict in train_samples:
                        buffer.append(_sample_to_pyg_data(sample_dict))
                except Exception as exc:
                    self._failed_users.append((group, uid, str(exc)))
                    continue

            while len(buffer) >= self.batch_size:
                batch = Batch.from_data_list(buffer[: self.batch_size])
                buffer = buffer[self.batch_size :]
                yield batch

        if buffer:
            yield Batch.from_data_list(buffer)


def none_or_int(value):
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    return int(value)


def none_or_float(value):
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    return float(value)


def none_or_str(value):
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    return value


def parse_args():
    description = (
        "Train the general-user retweet model from notebook 3.1 as a Python script.\n\n"
        "The script runs:\n"
        "  1. Baseline per-user SVC evaluation\n"
        "  2. On-the-fly GNN train/validation loader creation\n"
        "  3. RetweetGNN training with MLflow logging\n\n"
        "Optional values accept 'none' to disable them where applicable."
    )
    parser = argparse.ArgumentParser(description=description, formatter_class=HelpFormatter)

    parser.add_argument(
        "--reset-gnn-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear checkpoints/logs before training. Use --no-reset-gnn-training to resume from existing artifacts.",
    )
    parser.add_argument(
        "--n-users",
        type=none_or_int,
        default=None,
        help="Total users to sample proportionally across groups. Use 'none' to keep all users.",
    )
    parser.add_argument(
        "--experiment-tag",
        default="v3_full",
        help="Experiment identifier used to derive FINAL_TAG and EXPERIMENT_DIR.",
    )
    parser.add_argument(
        "--hparams-file",
        type=none_or_str,
        default="hparams_from_gpu_tuning.json",
        help="Tuned hyperparameters JSON. Use 'none' to disable JSON overrides.",
    )
    parser.add_argument(
        "--data-path",
        default="/Workspace/Users/pablo.celayes@bolt.eu/learning/data/sna_classifier",
        help="Base data directory containing datasets and cached baseline artifacts.",
    )
    parser.add_argument(
        "--embeddings-path",
        default="/Workspace/Users/pablo.celayes@bolt.eu/learning/data/sna_classifier/node_embeddings.pt",
        help="Path to the node embeddings file consumed by RetweetGNN.",
    )
    parser.add_argument(
        "--train-chunk-size",
        type=int,
        default=10,
        help="Users processed per chunk when creating on-the-fly GNN training samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size used by train, validation, and train-eval loaders before any hparams-file override.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Model dropout rate before any hparams-file override.",
    )
    parser.add_argument(
        "--drop-edge-rate",
        type=float,
        default=0.1,
        help="Edge dropout rate in the GNN before any hparams-file override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=2000,
        help="Log checkpoint metrics every N training steps.",
    )
    parser.add_argument(
        "--train-f1-every-n-epochs",
        type=none_or_int,
        default=1,
        help="Compute train F1 every N epochs. Use 'none' to skip during training.",
    )
    parser.add_argument(
        "--patience-steps",
        type=none_or_int,
        default=10000,
        help="Early-stopping patience expressed in steps before conversion to checkpoint count.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=none_or_int,
        default=8,
        help="Gradient accumulation factor. Use 1 or 'none' to disable accumulation.",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable mixed-precision training.",
    )
    parser.add_argument(
        "--target-pos-rate",
        type=none_or_float,
        default=0.10,
        help="Oversample positives in the train loader to reach this positive rate. Use 'none' to disable.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=none_or_int,
        default=20000,
        help="Cap validation samples. Use 'none' to keep all validation rows.",
    )
    parser.add_argument(
        "--max-train-eval-samples",
        type=none_or_int,
        default=20000,
        help="Cap sampled train-eval rows used for periodic train metrics. Use 'none' for all rows.",
    )
    parser.add_argument(
        "--init-weights-path",
        type=none_or_str,
        default="./experiments/v2_N20/best_retweet_gnn_general.pt",
        help="Optional .pt path for warm-starting model weights. Use 'none' to train from scratch.",
    )
    parser.add_argument(
        "--finetune-lr-factor",
        type=float,
        default=0.05,
        help="Multiplier applied to the base LR when fine-tuning from pre-trained weights.",
    )
    parser.add_argument(
        "--finetune-wd-factor",
        type=float,
        default=0.2,
        help="Multiplier applied to the base weight decay when fine-tuning from pre-trained weights.",
    )
    parser.add_argument(
        "--finetune-warmup-epochs",
        type=int,
        default=2,
        help="Warmup epochs used while fine-tuning.",
    )

    return parser.parse_args()


def _sample_to_pyg_data(sample_dict):
    """Convert a raw sample dict from create_gnn_train_val_samples to PyG Data."""
    central_id = int(sample_dict["central_user_id"])
    neighbor_ids = (
        sample_dict["neighbor_ids"].tolist()
        if hasattr(sample_dict["neighbor_ids"], "tolist")
        else list(sample_dict["neighbor_ids"])
    )
    all_ids = [central_id] + neighbor_ids
    num_nodes = len(all_ids)

    user_ids = torch.tensor(all_ids, dtype=torch.long)

    retweeted_raw = sample_dict["retweeted_ids"]
    retweeted_set = set(
        int(value) for value in (retweeted_raw.tolist() if hasattr(retweeted_raw, "tolist") else retweeted_raw)
    )
    retweet_flag = torch.tensor(
        [1.0 if uid in retweeted_set else 0.0 for uid in all_ids],
        dtype=torch.float,
    ).unsqueeze(1)

    edge_index_raw = sample_dict["edge_index"]
    if hasattr(edge_index_raw, "__len__") and len(edge_index_raw) > 0:
        edge_index_arr = (
            np.array(edge_index_raw, dtype=np.int64)
            if not isinstance(edge_index_raw, np.ndarray)
            else edge_index_raw.astype(np.int64)
        )
        edge_index = torch.from_numpy(edge_index_arr).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    label = torch.tensor(int(sample_dict["label"]), dtype=torch.long)

    return Data(
        user_ids=user_ids,
        retweet_flag=retweet_flag,
        edge_index=edge_index,
        y=label,
        num_nodes=num_nodes,
        central_mask=torch.zeros(num_nodes, dtype=torch.bool).index_fill_(0, torch.tensor([0]), True),
    )


def _proportional_allocate(row_counts, max_samples):
    """Allocate max_samples proportionally across items using largest remainder."""
    total_available = sum(row_counts)
    if not max_samples or total_available <= max_samples:
        return list(row_counts)

    sample_rate = max_samples / total_available
    raw = [n * sample_rate for n in row_counts]
    allocations = [max(1, int(value)) for value in raw]

    deficit = max_samples - sum(allocations)
    if deficit > 0:
        remainders = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in remainders[:deficit]:
            allocations[i] += 1

    while sum(allocations) > max_samples:
        max_idx = max(range(len(allocations)), key=lambda i: allocations[i])
        allocations[max_idx] -= 1

    return allocations


def enrich_config(args):
    args.final_tag = f"{args.experiment_tag}_N{args.n_users}" if args.n_users else args.experiment_tag
    args.experiment_dir = f"./experiments/{args.final_tag}"
    args.patience = -(-args.patience_steps // args.log_every_n_steps) if args.patience_steps else None
    args._tuned_lr = None
    args._tuned_wd = None
    return args


def reset_training_artifacts(args):
    deleted = []

    if args.reset_gnn_training:
        print("⚠️  RESET_GNN_TRAINING=True — clearing checkpoints and training logs...")
        training_artifacts = [
            "best_retweet_gnn_general.pt",
            "training.log",
            "training_history.json",
            "training_history.pkl",
            "checkpoint.pt",
            "training_checkpoint.pt",
            "train_eval_metadata.json",
            "training_config.json",
        ]
        for fname in training_artifacts:
            fpath = os.path.join(args.experiment_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                deleted.append(fpath)

        for meta_file in glob.glob(os.path.join(args.experiment_dir, "train_eval_metadata_*.json")):
            os.remove(meta_file)
            deleted.append(meta_file)

        legacy_cache = os.path.join(args.experiment_dir, "train_eval_samples_cache")
        if os.path.isdir(legacy_cache):
            shutil.rmtree(legacy_cache)
            deleted.append(legacy_cache)

        print("  Cleared training artifacts. GNN training will start fresh.")
    else:
        print("RESET_GNN_TRAINING=False — resuming from existing checkpoint if available.")

    os.makedirs(args.experiment_dir, exist_ok=True)

    if deleted:
        print(f"\nTotal deleted items: {len(deleted)}")
        for path in deleted:
            print(f"  - {path}")


def apply_hparam_overrides(args):
    if args.hparams_file:
        with open(args.hparams_file) as f:
            tuned = json.load(f)
        args.batch_size = tuned.get("batch_size", args.batch_size)
        args.train_chunk_size = tuned.get("train_chunk_size", args.train_chunk_size)
        args.gradient_accumulation_steps = tuned.get(
            "grad_accum_steps", args.gradient_accumulation_steps
        )
        args.dropout = tuned.get("dropout", args.dropout)
        args.drop_edge_rate = tuned.get("drop_edge_rate", args.drop_edge_rate)
        args.finetune_warmup_epochs = tuned.get("warmup_epochs", args.finetune_warmup_epochs)
        args._tuned_lr = tuned.get("lr")
        args._tuned_wd = tuned.get("weight_decay")
        print(f"Loaded tuned hparams from: {args.hparams_file}")

    print(f"Experiment: {args.final_tag}")
    print(f"Output dir: {args.experiment_dir}")
    if args.init_weights_path:
        print(f"Loading pre-trained weights from: {args.init_weights_path}")
    else:
        print("Training from scratch.")


def print_gpu_inventory():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {props.name}, free={free/1e9:.1f}GB / {total/1e9:.1f}GB")
        except Exception:
            print(f"GPU {i}: {props.name}")


def get_free_memory_per_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [int(x) for x in result.stdout.strip().split("\n") if x.strip()]


def get_best_device():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"Device: {device}")
        return device

    free_mem = get_free_memory_per_gpu()
    if free_mem:
        best_gpu = max(range(len(free_mem)), key=lambda i: free_mem[i])
        print(f"Free memory per GPU (MiB): {free_mem}")
        print(f"Selected GPU {best_gpu} with {free_mem[best_gpu]} MiB free")
        device = torch.device(f"cuda:{best_gpu}")
    else:
        device = torch.device("cuda:0")

    print(f"Device: {device}")
    return device


def load_graph_and_users(args):
    graph = nx.read_graphml(IG_GRAPH_PATH)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    with open(f"{args.data_path}/datasets/user_splits.json") as f:
        user_splits = json.load(f)

    print(f"Groups in user_splits: {list(user_splits.keys())}")
    for key, values in user_splits.items():
        print(f"  {key}: {len(values)} users")

    test_groups = [group for group in user_splits.keys() if group not in TRAIN_GROUPS]
    print(f"\nTrain groups: {TRAIN_GROUPS}")
    print(f"Test groups: {test_groups}")

    sample_path = f"{args.experiment_dir}/user_sample.json"

    if os.path.exists(sample_path):
        with open(sample_path) as f:
            saved_sample = json.load(f)
        print(f"\nLoading saved user sample from {sample_path}")

        user_data = {}
        failed_users = []
        for group, uids in saved_sample.items():
            user_data[group] = {}
            for uid in uids:
                try:
                    X_tr, X_te, y_tr, y_te = load_dataframe_raw(uid, sparse=True)
                    if X_tr.shape[0] > 0 and X_te.shape[0] > 0 and y_tr.sum() > 0 and y_te.sum() > 0:
                        user_data[group][uid] = (X_tr, X_te, y_tr, y_te)
                    else:
                        failed_users.append((group, uid, "empty data on reload"))
                except Exception as exc:
                    failed_users.append((group, uid, str(exc)))

        print("  Loaded users per group:")
        for group in saved_sample:
            print(f"    {group}: {len(user_data[group])}/{len(saved_sample[group])}")
        print(f"  Total: {sum(len(user_data[g]) for g in user_data)}")
        if failed_users:
            print(f"  Failed on reload: {len(failed_users)}")
    else:
        print(f"\nNo saved sample for {args.final_tag}, loading all users...")
        user_data = {}
        failed_users = []

        for group in user_splits:
            user_data[group] = {}
            for uid in user_splits[group]:
                try:
                    X_tr, X_te, y_tr, y_te = load_dataframe_raw(uid, sparse=True)
                    if X_tr.shape[0] > 0 and X_te.shape[0] > 0 and y_tr.sum() > 0 and y_te.sum() > 0:
                        user_data[group][uid] = (X_tr, X_te, y_tr, y_te)
                    else:
                        failed_users.append((group, uid, "empty data"))
                except Exception as exc:
                    failed_users.append((group, uid, str(exc)))
                    continue

        print("\nLoaded users per group:")
        total_valid = 0
        for group in user_splits:
            n_valid = len(user_data[group])
            total_valid += n_valid
            print(f"  {group}: {n_valid}/{len(user_splits[group])} valid")
        print(f"  Total valid: {total_valid}")
        print(f"  Failed: {len(failed_users)}")

        if args.n_users is not None:
            group_sizes = {g: len(user_data[g]) for g in user_splits}
            total_available = sum(group_sizes.values())
            train_available = sum(group_sizes.get(g, 0) for g in TRAIN_GROUPS)
            test_available = sum(group_sizes.get(g, 0) for g in test_groups)
            train_slots = int(round(args.n_users * train_available / total_available))
            test_slots = args.n_users - train_slots

            n_u_train = min(group_sizes.get("u_train", 0), train_slots)
            n_au_train = min(group_sizes.get("au_train", 0), train_slots - n_u_train)
            raw_alloc = {"u_train": n_u_train, "au_train": n_au_train}

            test_group_sizes = {g: group_sizes.get(g, 0) for g in test_groups if group_sizes.get(g, 0) > 0}
            total_test_available = sum(test_group_sizes.values())
            for group in test_groups:
                if total_test_available > 0 and group_sizes.get(group, 0) > 0:
                    raw_alloc[group] = int(round(test_slots * group_sizes[group] / total_test_available))
                else:
                    raw_alloc[group] = 0

            test_diff = test_slots - sum(raw_alloc.get(group, 0) for group in test_groups)
            for group in sorted(test_groups, key=lambda g: group_sizes.get(g, 0), reverse=True):
                if test_diff == 0:
                    break
                adjustment = 1 if test_diff > 0 else -1
                raw_alloc[group] = max(1, raw_alloc[group] + adjustment)
                test_diff -= adjustment

            sampled_user_data = {}
            for group in user_splits:
                if group not in raw_alloc or raw_alloc[group] == 0:
                    sampled_user_data[group] = {}
                    continue
                uids = list(user_data[group].keys())
                n_sample = min(raw_alloc[group], len(uids))
                sampled_uids = sample(uids, n_sample)
                sampled_user_data[group] = {uid: user_data[group][uid] for uid in sampled_uids}
            user_data = sampled_user_data

            print(f"\nSampled {args.n_users} users (train priority: u_train first, then au_train):")
            for group in user_splits:
                print(f"  {group}: {len(user_data[group])} (target {raw_alloc.get(group, 0)})")
            print(f"  Total sampled: {sum(len(user_data[group]) for group in user_splits)}")

        sample_to_save = {group: list(user_data[group].keys()) for group in user_data}
        with open(sample_path, "w") as f:
            json.dump(sample_to_save, f, indent=2)
        print(f"  Saved user sample to {sample_path}")

    valid_users = list(user_data.get("u_train", {}).keys()) + list(user_data.get("au_train", {}).keys())
    baseline_users = list(user_data.get("u_train", {}).keys())
    print(
        f"\nTrain-group valid users: {len(valid_users)} "
        f"(baseline SVC: {len(baseline_users)} from u_train only)"
    )

    return graph, user_splits, test_groups, user_data, valid_users, baseline_users


def run_baseline(args, baseline_users, user_data):
    baseline_user_cache_path = f"{args.data_path}/baseline_svc_user_cache.pkl"

    if not os.path.exists(baseline_user_cache_path):
        init_path = "./experiments/v3_full_N150/baseline_svc_results.pkl"
        if os.path.exists(init_path):
            print(f"Initializing user-level baseline cache from {init_path}...")
            with open(init_path, "rb") as f:
                init_data = pickle.load(f)
            cache = {}
            init_f1s = init_data["baseline_f1s"]
            init_params = init_data["baseline_best_params"]
            init_preds = init_data["all_baseline_test_preds"]
            user_ids = list(init_f1s.keys())
            for idx, uid in enumerate(user_ids):
                preds, labels = init_preds[idx]
                cache[uid] = {
                    "f1": init_f1s[uid],
                    "best_params": init_params[uid],
                    "preds": preds,
                    "labels": labels,
                }
            with open(baseline_user_cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"  Initialized cache with {len(cache)} users from v3_full_N150.")
        else:
            print(f"No v3_full_N150 results found at {init_path}, starting empty cache.")
            with open(baseline_user_cache_path, "wb") as f:
                pickle.dump({}, f)

    with open(baseline_user_cache_path, "rb") as f:
        baseline_user_cache = pickle.load(f)
    print(f"Baseline user cache: {len(baseline_user_cache)} users already computed.")

    users_to_compute = [uid for uid in baseline_users if uid not in baseline_user_cache]
    users_cached = [uid for uid in baseline_users if uid in baseline_user_cache]
    print(f"  This experiment: {len(baseline_users)} baseline users")
    print(f"  Already cached:  {len(users_cached)}")
    print(f"  Need computing:  {len(users_to_compute)}")

    if users_to_compute:
        gammas = [0.05, 0.1, 0.2]
        cs_rbf = [0.05, 0.1, 0.2]
        cs_linear = [0.05, 0.07, 0.1]
        degrees = [2, 3]
        coef0s = [1]
        cs_poly = [0.05, 0.1]

        t0 = time.time()
        for i, uid in enumerate(users_to_compute):
            t_user = time.time()
            X_tr, X_te, y_tr, y_te = user_data["u_train"][uid]

            X_tr_sp = X_tr.sparse.to_coo().tocsr() if hasattr(X_tr, "sparse") else X_tr
            X_te_sp = X_te.sparse.to_coo().tocsr() if hasattr(X_te, "sparse") else X_te

            best_f1 = -1
            best_preds = None
            best_params = None

            K_train_lin = linear_kernel(X_tr_sp)
            K_test_lin = linear_kernel(X_te_sp, X_tr_sp)
            for C in cs_linear:
                svc = SVC(C=C, kernel="precomputed", class_weight="balanced", random_state=42)
                svc.fit(K_train_lin, y_tr)
                preds = svc.predict(K_test_lin)
                test_f1 = f1_score(y_te, preds)
                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_preds = preds
                    best_params = ("linear", {"C": C})

            for degree in degrees:
                for coef0 in coef0s:
                    K_train_poly = polynomial_kernel(X_tr_sp, degree=degree, coef0=coef0)
                    K_test_poly = polynomial_kernel(X_te_sp, X_tr_sp, degree=degree, coef0=coef0)
                    for C in cs_poly:
                        svc = SVC(C=C, kernel="precomputed", class_weight="balanced", random_state=42)
                        svc.fit(K_train_poly, y_tr)
                        preds = svc.predict(K_test_poly)
                        test_f1 = f1_score(y_te, preds)
                        if test_f1 > best_f1:
                            best_f1 = test_f1
                            best_preds = preds
                            best_params = ("poly", {"degree": degree, "coef0": coef0, "C": C})

            for gamma in gammas:
                K_train = rbf_kernel(X_tr_sp, gamma=gamma)
                K_test = rbf_kernel(X_te_sp, X_tr_sp, gamma=gamma)
                for C in cs_rbf:
                    svc = SVC(C=C, kernel="precomputed", class_weight="balanced", random_state=42)
                    svc.fit(K_train, y_tr)
                    preds = svc.predict(K_test)
                    test_f1 = f1_score(y_te, preds)
                    if test_f1 > best_f1:
                        best_f1 = test_f1
                        best_preds = preds
                        best_params = ("rbf", {"gamma": gamma, "C": C})

            baseline_user_cache[uid] = {
                "f1": best_f1,
                "best_params": best_params,
                "preds": best_preds,
                "labels": np.array(y_te),
            }

            elapsed = time.time() - t0
            user_time = time.time() - t_user
            avg_per_user = elapsed / (i + 1)
            remaining = avg_per_user * (len(users_to_compute) - i - 1)
            print(
                f"  [{i+1:>3}/{len(users_to_compute)}] uid={uid}  F1={best_f1:.4f}  "
                f"kernel={best_params[0]}  ({user_time:.1f}s | elapsed {int(elapsed)//60}m{int(elapsed)%60:02d}s | "
                f"ETA {int(remaining)//60}m{int(remaining)%60:02d}s)"
            )

        total_time = time.time() - t0
        print(
            f"\nComputed {len(users_to_compute)} new users in {total_time:.1f}s "
            f"({total_time/len(users_to_compute):.1f}s/user avg)."
        )

        with open(baseline_user_cache_path, "wb") as f:
            pickle.dump(baseline_user_cache, f)
        print(f"  Cache updated: {len(baseline_user_cache)} total users.")
    else:
        print("All baseline users already cached — no computation needed.")

    baseline_f1s = {uid: baseline_user_cache[uid]["f1"] for uid in baseline_users}
    baseline_best_params = {uid: baseline_user_cache[uid]["best_params"] for uid in baseline_users}
    all_baseline_test_preds = [
        (baseline_user_cache[uid]["preds"], baseline_user_cache[uid]["labels"])
        for uid in baseline_users
    ]

    kernel_counts = {}
    for params in baseline_best_params.values():
        kernel = params[0]
        kernel_counts[kernel] = kernel_counts.get(kernel, 0) + 1
    print(f"\nBaseline results for {len(baseline_users)} users — kernel distribution: {kernel_counts}")

    f1_values = list(baseline_f1s.values())
    print("=== Baseline SVC (RBF) — Per-user Test F1 Distribution ===")
    print(f"  Mean:   {np.mean(f1_values):.4f}")
    print(f"  Median: {np.median(f1_values):.4f}")
    print(f"  Std:    {np.std(f1_values):.4f}")
    print(f"  Min:    {np.min(f1_values):.4f}")
    print(f"  Max:    {np.max(f1_values):.4f}")

    all_preds = np.concatenate([preds for preds, _ in all_baseline_test_preds])
    all_labels = np.concatenate([labels for _, labels in all_baseline_test_preds])
    combined_f1 = f1_score(all_labels, all_preds)
    print(f"\n  Combined F1 (all users pooled): {combined_f1:.4f}")
    print(f"  Total test samples: {len(all_labels)}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(f1_values, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(np.mean(f1_values), color="red", linestyle="--", label=f"Mean: {np.mean(f1_values):.3f}")
    ax.axvline(np.median(f1_values), color="orange", linestyle="--", label=f"Median: {np.median(f1_values):.3f}")
    ax.set_xlabel("Test F1 Score")
    ax.set_ylabel("Count")
    ax.set_title("Baseline SVC (RBF) — Per-user Test F1 Distribution")
    ax.legend()
    plt.tight_layout()
    baseline_plot_path = os.path.join(args.experiment_dir, "baseline_f1_distribution.png")
    fig.savefig(baseline_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline distribution plot: {baseline_plot_path}")


def build_train_val_assignments(args, user_splits, test_groups, user_data):
    train_user_items = []
    for group in TRAIN_GROUPS:
        for uid in user_data.get(group, {}):
            train_user_items.append((group, uid))

    val_user_splits = []
    for group in TRAIN_GROUPS:
        for uid in user_data.get(group, {}):
            val_user_splits.append((group, uid, "test"))
    for group in test_groups:
        for uid in user_data.get(group, {}):
            val_user_splits.append((group, uid, "train"))
            val_user_splits.append((group, uid, "test"))

    print(f"Train users: {len(train_user_items)} (from {TRAIN_GROUPS})")
    print(
        f"Val user/split combos: {len(val_user_splits)} "
        f"(from {TRAIN_GROUPS} test + {test_groups} both)"
    )
    print(f"\nGNN samples will be created on-the-fly (chunk_size={args.train_chunk_size}).")
    print(f"Val samples capped at {args.max_val_samples} (proportional across users).")

    return train_user_items, val_user_splits


def build_loaders(args, graph, train_user_items, val_user_splits, user_data):
    train_loader = ChunkedGNNTrainLoader(
        train_user_items,
        user_data,
        graph,
        batch_size=args.batch_size,
        chunk_size=args.train_chunk_size,
        target_pos_rate=args.target_pos_rate,
    )

    print(
        f"Train loader: {len(train_user_items)} users, "
        f"chunk_size={args.train_chunk_size}, batch_size={args.batch_size}"
    )

    print("\nComputing class weights from train labels...")
    label_counts = train_loader.get_label_counts()
    total_train = sum(label_counts.values())
    classes = np.array(sorted(label_counts.keys()))
    class_weights = torch.tensor(
        [total_train / (len(classes) * label_counts[c]) for c in classes],
        dtype=torch.float,
    )
    train_pos_rate = label_counts.get(1, 0) / max(total_train, 1)
    print(f"  Label counts: {label_counts}")
    print(f"  Positive rate: {train_pos_rate:.4f} ({label_counts.get(1, 0)}/{total_train})")
    print(f"  Class weights: {class_weights.tolist()}")
    print(f"  Total train samples (from y_tr): {total_train}")

    val_cache_path = (
        os.path.join(args.experiment_dir, f"val_samples_cache_{args.max_val_samples}")
        if args.max_val_samples
        else None
    )
    val_loader = CachedValLoader(
        val_user_splits,
        user_data,
        graph,
        batch_size=args.batch_size,
        max_samples=args.max_val_samples,
        cache_path=val_cache_path,
    )
    val_label_counts = val_loader.get_label_counts()
    val_pos_rate = val_label_counts.get(1, 0) / max(sum(val_label_counts.values()), 1)
    print(f"\nVal loader: {val_loader.total_samples} cached samples, {len(val_loader)} batches")
    print(
        f"  Positive rate: {val_pos_rate:.4f} "
        f"({val_label_counts.get(1, 0)}/{sum(val_label_counts.values())})"
    )

    train_eval_meta_path = os.path.join(
        args.experiment_dir,
        f"train_eval_metadata_{args.max_train_eval_samples}.json",
    )
    train_eval_loader = SampledTrainEvalLoader(
        train_user_items,
        user_data,
        graph,
        batch_size=args.batch_size,
        max_samples=args.max_train_eval_samples,
        metadata_path=train_eval_meta_path,
    )
    train_eval_label_counts = train_eval_loader.get_label_counts()
    train_eval_pos_rate = train_eval_label_counts.get(1, 0) / max(sum(train_eval_label_counts.values()), 1)
    print(
        f"\nTrain-eval loader: {train_eval_loader.total_samples} samples, "
        f"{len(train_eval_loader)} batches"
    )
    print(
        f"  Positive rate: {train_eval_pos_rate:.4f} "
        f"({train_eval_label_counts.get(1, 0)}/{sum(train_eval_label_counts.values())})"
    )

    return train_loader, class_weights, val_loader, train_eval_loader


def build_model(args, device):
    model = RetweetGNN(
        ff_hidden_dim=64,
        gcn_hidden_dim=64,
        transformer_dim=64,
        transformer_heads=4,
        embeddings_path=args.embeddings_path,
        device=device,
        dropout=args.dropout,
        drop_edge_rate=args.drop_edge_rate,
    ).to(device)

    existing_checkpoint = os.path.join(args.experiment_dir, "best_retweet_gnn_general.pt")
    is_fresh_start = args.reset_gnn_training or not os.path.exists(existing_checkpoint)
    apply_init_weights = bool(args.init_weights_path) and is_fresh_start
    args._apply_init_weights = apply_init_weights

    if not is_fresh_start and args.init_weights_path:
        print(f"⚠️  Existing checkpoint found at {existing_checkpoint} — skipping INIT_WEIGHTS_PATH.")
        print("   (Set RESET_GNN_TRAINING=True to force re-initialization from pre-trained weights.)")

    if apply_init_weights:
        print(f"Loading pre-trained weights from: {args.init_weights_path}")
        state_dict = torch.load(args.init_weights_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (will use random init): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        print(f"  Weights loaded successfully. gate_param = {model.gate_param.item():.4f}")
    else:
        with torch.no_grad():
            model.gate_param.fill_(0.0)

    return model


def resolve_training_hparams(args):
    base_lr = 3e-3
    base_wd = 1e-3
    training_config_path = os.path.join(args.experiment_dir, "training_config.json")
    checkpoint_exists = os.path.exists(os.path.join(args.experiment_dir, "training_checkpoint.pt"))
    can_resume = not args.reset_gnn_training and checkpoint_exists

    if can_resume and os.path.exists(training_config_path):
        with open(training_config_path) as f:
            saved_config = json.load(f)
        train_lr = saved_config["lr"]
        train_wd = saved_config["weight_decay"]
        warmup_epochs = saved_config["warmup_epochs"]
        print(f"Resuming with saved training config from {training_config_path}:")
        print(f"  lr={train_lr:.1e}, weight_decay={train_wd:.1e}, warmup_epochs={warmup_epochs}")
        if saved_config.get("is_finetuning"):
            print(f"  (originally fine-tuned from {saved_config.get('init_weights_path', '?')})")
    else:
        if args._tuned_lr is not None:
            train_lr = args._tuned_lr
            train_wd = args._tuned_wd
            warmup_epochs = args.finetune_warmup_epochs
            print(
                f"Using tuned hparams: lr={train_lr:.2e}, wd={train_wd:.2e}, warmup={warmup_epochs}"
            )
        else:
            train_lr = base_lr * args.finetune_lr_factor if args._apply_init_weights else base_lr
            train_wd = base_wd * args.finetune_wd_factor if args._apply_init_weights else base_wd
            warmup_epochs = args.finetune_warmup_epochs if args._apply_init_weights else 5

        config_to_save = {
            "lr": train_lr,
            "weight_decay": train_wd,
            "warmup_epochs": warmup_epochs,
            "is_finetuning": bool(args._apply_init_weights),
            "init_weights_path": args.init_weights_path,
        }
        with open(training_config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)
        print(f"Training config saved to {training_config_path}")
        if args._apply_init_weights:
            print("Fine-tuning mode:")
            print(
                f"  LR reduced from {base_lr:.1e} to {train_lr:.1e} "
                f"(factor={args.finetune_lr_factor})"
            )
            print(
                f"  Weight decay reduced from {base_wd:.1e} to {train_wd:.1e} "
                f"(factor={args.finetune_wd_factor})"
            )
            print(f"  Warmup epochs reduced from 5 to {warmup_epochs}")

    return {
        "base_lr": base_lr,
        "base_wd": base_wd,
        "training_config_path": training_config_path,
        "checkpoint_exists": checkpoint_exists,
        "can_resume": can_resume,
        "train_lr": train_lr,
        "train_wd": train_wd,
        "warmup_epochs": warmup_epochs,
    }


def save_run_params(args, valid_users, train_plan):
    """Save all run parameters to a JSON file in the experiment directory."""
    params = {
        "experiment_tag": args.experiment_tag,
        "final_tag": args.final_tag,
        "n_users": args.n_users if args.n_users else "all",
        "n_train_users": len(valid_users),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_chunk_size": args.train_chunk_size,
        "lr": train_plan["train_lr"],
        "weight_decay": train_plan["train_wd"],
        "warmup_epochs": train_plan["warmup_epochs"],
        "patience": args.patience if args.patience else "disabled",
        "gradient_accumulation_steps": args.gradient_accumulation_steps or 1,
        "mixed_precision": args.mixed_precision,
        "log_every_n_steps": args.log_every_n_steps,
        "train_f1_every_n_epochs": args.train_f1_every_n_epochs or "disabled",
        "max_val_samples": args.max_val_samples or "all",
        "max_train_eval_samples": args.max_train_eval_samples or "all",
        "ff_hidden_dim": 64,
        "gcn_hidden_dim": 64,
        "transformer_dim": 64,
        "transformer_heads": 4,
        "dropout": args.dropout,
        "drop_edge_rate": args.drop_edge_rate,
        "init_weights_path": args.init_weights_path or "none",
        "reset_gnn_training": args.reset_gnn_training,
        "started_at": datetime.now().isoformat(),
    }
    params_path = os.path.join(args.experiment_dir, "run_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Run parameters saved to: {params_path}")


def free_unused_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_and_log(args, device, model, class_weights, train_loader, val_loader, train_eval_loader, train_plan):
    training_log_path = f"{args.experiment_dir}/training.log"

    with tee_to_log(training_log_path):
        if args._apply_init_weights:
            print("\n" + "=" * 60)
            print("INITIAL EVALUATION (pre-trained weights, before training)")
            print("=" * 60)
            model.eval()
            init_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

            val_preds, val_labels, val_loss, val_batches = [], [], 0.0, 0
            t_val_init = time.time()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    val_loss += init_criterion(out, batch.y).item()
                    val_batches += 1
                    val_preds.extend(out.argmax(dim=1).cpu().tolist())
                    val_labels.extend(batch.y.cpu().tolist())
            val_init_elapsed = time.time() - t_val_init
            print(
                f"  Val   — Loss: {val_loss / max(val_batches, 1):.4f} | "
                f"Acc: {accuracy_score(val_labels, val_preds):.4f} | "
                f"F1: {f1_score(val_labels, val_preds):.4f} | "
                f"P: {precision_score(val_labels, val_preds, zero_division=0):.4f} | "
                f"R: {recall_score(val_labels, val_preds, zero_division=0):.4f} | "
                f"samples: {len(val_labels)} | "
                f"took {int(val_init_elapsed)//60}m {int(val_init_elapsed)%60:02d}s"
            )

            tr_preds, tr_labels, tr_loss, tr_batches = [], [], 0.0, 0
            t_tr_init = time.time()
            with torch.no_grad():
                for batch in train_eval_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    tr_loss += init_criterion(out, batch.y).item()
                    tr_batches += 1
                    tr_preds.extend(out.argmax(dim=1).cpu().tolist())
                    tr_labels.extend(batch.y.cpu().tolist())
            tr_init_elapsed = time.time() - t_tr_init
            print(
                f"  Train — Loss: {tr_loss / max(tr_batches, 1):.4f} | "
                f"Acc: {accuracy_score(tr_labels, tr_preds):.4f} | "
                f"F1: {f1_score(tr_labels, tr_preds):.4f} | "
                f"P: {precision_score(tr_labels, tr_preds, zero_division=0):.4f} | "
                f"R: {recall_score(tr_labels, tr_preds, zero_division=0):.4f} | "
                f"samples: {len(tr_labels)} | "
                f"took {int(tr_init_elapsed)//60}m {int(tr_init_elapsed)%60:02d}s"
            )
            print("=" * 60 + "\n")
            del val_preds, val_labels, tr_preds, tr_labels, init_criterion
            free_unused_memory()

        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_dir=args.experiment_dir,
            class_weights=class_weights,
            epochs=args.epochs,
            device=device,
            lr=train_plan["train_lr"],
            log_every_n_steps=args.log_every_n_steps,
            patience=args.patience,
            lr_warmup_epochs=train_plan["warmup_epochs"],
            weight_decay=train_plan["train_wd"],
            resume=train_plan["can_resume"],
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            train_f1_every_n_epochs=args.train_f1_every_n_epochs,
            train_eval_loader=train_eval_loader,
            on_checkpoint=None,
        )

    return history


def save_final_results(args, history):
    """Save training summary metrics and curves to the experiment directory."""
    if history:
        summary = {}
        if history.get("val_f1"):
            summary["best_checkpoint_val_f1"] = max(history["val_f1"])
        if history.get("epoch_val_f1"):
            summary["best_epoch_val_f1"] = max(history["epoch_val_f1"])
        if history.get("epoch_train_f1"):
            summary["best_epoch_train_f1"] = max(history["epoch_train_f1"])
        if history.get("train_loss"):
            summary["final_train_loss"] = history["train_loss"][-1]
        if history.get("val_loss"):
            summary["final_val_loss"] = history["val_loss"][-1]
        summary["completed_at"] = datetime.now().isoformat()
        if summary:
            summary_path = os.path.join(args.experiment_dir, "run_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Run summary saved to: {summary_path}")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

    best_model_path = os.path.join(args.experiment_dir, "best_retweet_gnn_general.pt")
    if os.path.exists(best_model_path):
        print(f"Best model saved at: {best_model_path}")

    if history and (history.get("step") or history.get("epoch_step")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        if history.get("step"):
            ax.plot(history["step"], history["train_loss"], "b-", alpha=0.6, label="Train loss (checkpoint)")
            ax.plot(history["step"], history["val_loss"], "r-", alpha=0.6, label="Val loss (checkpoint)")
        if history.get("epoch_step"):
            ax.plot(history["epoch_step"], history["epoch_val_loss"], "ro-", markersize=5, label="Val loss (end-of-epoch)")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if history.get("step"):
            ax.plot(history["step"], history["val_f1"], "r-", alpha=0.6, label="Val F1 (checkpoint)")
        if history.get("epoch_step"):
            if history.get("epoch_train_f1"):
                ax.plot(history["epoch_step"], history["epoch_train_f1"], "b^-", markersize=5, label="Train F1 (end-of-epoch)")
            ax.plot(history["epoch_step"], history["epoch_val_f1"], "ro-", markersize=5, label="Val F1 (end-of-epoch)")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("F1 Score")
        ax.set_title("Training & Validation F1")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"GNN Training Curves — {args.final_tag}", fontsize=13)
        plt.tight_layout()

        curves_path = os.path.join(args.experiment_dir, "training_curves.png")
        fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curves saved to: {curves_path}")

    print(f"\nTraining run completed. All results in: {args.experiment_dir}")


def main():
    args = enrich_config(parse_args())
    print(f"Running from: {SCRIPT_DIR}")

    reset_training_artifacts(args)
    apply_hparam_overrides(args)

    print_gpu_inventory()
    device = get_best_device()

    graph, user_splits, test_groups, user_data, valid_users, baseline_users = load_graph_and_users(args)
    run_baseline(args, baseline_users, user_data)

    train_user_items, val_user_splits = build_train_val_assignments(args, user_splits, test_groups, user_data)
    train_loader, class_weights, val_loader, train_eval_loader = build_loaders(
        args,
        graph,
        train_user_items,
        val_user_splits,
        user_data,
    )

    model = build_model(args, device)
    free_unused_memory()

    train_plan = resolve_training_hparams(args)
    save_run_params(args, valid_users, train_plan)
    history = train_and_log(
        args,
        device,
        model,
        class_weights,
        train_loader,
        val_loader,
        train_eval_loader,
        train_plan,
    )
    save_final_results(args, history)


if __name__ == "__main__":
    main()
